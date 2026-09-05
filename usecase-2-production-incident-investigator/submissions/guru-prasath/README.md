# Deterministic incident investigator

**Participant:** guruprasath gopal · **Phone:** 8741 6283 · **Email:** guruprasath3200@gmail.com

## Understanding of the problem

The challenge is not finding one document containing the query words. The
answer is distributed across logs, architecture, deployment history, the
known-issues CSV, runbooks, and historical incidents. The investigator must
correlate those sources and calibrate confidence by how independently they
agree.

- **Incident A (pool exhaustion)** has a strong causal chain: payment charges
  succeed all morning, then `ConnectionPoolTimeoutException` + `GATEWAY_TIMEOUT`
  failures start right after deploy v2.4.1 cuts the adapter pool from 50 to 10.
  The issue catalog (KI-101), runbook RB-014, architecture, API contract
  (5000ms acquire timeout, no retry), and prior incident INC-2031 all support
  the same explanation.
- **Incident B (ambiguous delay)** deliberately has thin evidence: emails queue
  then send 40–75 minutes late (queue depth 340) with no `ERROR` entries, no
  correlated deployment, no matching known issue, no precedent, and no
  per-stage timing. The correct result is a low-confidence report requiring
  human review — not a confident guess at consumer vs. provider bottleneck.

## Design

`solution.py` is a thin facade over a small pipeline; each stage lives in its
own module (single responsibility, stdlib only):

| Module | Responsibility |
|---|---|
| `config.py` | All tunables in one place (stopwords, excerpt budget, calibration weights/caps) |
| `text_processing.py` | Tokenization (keeps `payment-gateway-adapter` whole and split, light singular stemming), markdown chunking |
| `models.py` | `Record` (one chunk / one CSV row) and `Evidence` dataclasses passed between stages |
| `retrieval.py` | Ingestion (markdown chunks + per-row CSV candidates) and TF-IDF cosine ranking per file. No hand-tuned phrase bonuses — rare terms win through IDF alone |
| `excerpts.py` | Verbatim-safe quoting: every excerpt is a contiguous slice of its source, so `excerpt in corpus[source]` always holds |
| `correlation.py` | Hypothesis testing across independent source types, recording positive corroboration *and* explicit uncertainty; MTTR is scoped to the relevant runbook section (RB-014's 20 min), never a global first match; incident B returns `null` rather than borrowing RB-002's unconfirmed 15 min |
| `calibration.py` | Corroboration minus uncertainty mapped to 0–100; pool-exhaustion capped at 92 (never absolute certainty), ambiguous themes capped below the review threshold |

Evidence is kept in evidentiary order (logs → deployment → catalog → runbook →
precedent → architecture → contract), not retrieval-rank order — the brief
warns the top-ranked document alone (often the architecture overview) is
misleading. One excerpt per file enforces source-type diversity (the same
anti-redundancy role MMR/host-diversity plays in multi-source RAG). A final
citation-first pass re-cuts any quote missing a key asserted fact without
ever leaving verbatim-substring safety; `needs_human_review` derives directly
from the calibrated score, so the two fields cannot drift apart.

## Research basis

Three ideas from recent RAG literature shaped the final run, adapted to a
no-LLM, exact-lexical setting:

- **BM25 over TF-IDF cosine** — TF saturation plus length normalization
  consistently beats plain TF-IDF on precision for small domain corpora. It
  matters here: logs are long and repetitive, runbooks short and dense. After
  the switch, `logs.md` outranks the architecture overview for incident A on
  its own merits.
- **Set-level sufficiency over top-passage score** (SURE-RAG / EvidentialRAG
  framing: coverage, disagreement, conflict, retrieval uncertainty) — confidence
  comes from corroboration across independent source types minus explicit
  uncertainty phrases, with caps keeping ambiguous themes below review
  threshold and confident ones below absolute certainty.
- **Citation-first verification** (CiteG-style) — every version number, count,
  timeout, and ID asserted in `root_cause` is checked against the quoted
  excerpts; misses trigger a re-cut, and anything unquotable from the corpus
  becomes uncertainty rather than prose.

## Reproducing `answers.json`

```bash
cd usecase-2-production-incident-investigator
python -c "
from data.loader import load_incident
import json, sys
sys.path.insert(0, 'submissions/guru-prasath')
import solution
answers = {}
for name in ['incident_a_pool_exhaustion', 'incident_b_ambiguous_delay']:
    query, corpus = load_incident(name)
    answers[name] = solution.investigate(query, corpus)
with open('submissions/guru-prasath/answers.json', 'w') as f:
    json.dump(answers, f, indent=2)
"
```

## Why these decisions (problem 2 in brief)

- **BM25, not TF-IDF**: logs are long and repetitive, runbooks short and dense — saturation + length norm ranks the true error lines first.
- **Evidentiary order, not rank order**: the top-ranked doc alone (often architecture) is the known trap; the report follows logs → deployment → catalog → runbook → precedent → contract.
- **Capped confidence (92 high / low for B)**: a retrieval-only pipeline never claims certainty; thin evidence must stay below review threshold.
- **Verbatim excerpts + fact re-check**: every quoted line is a contiguous slice (`excerpt in source` always holds); asserted IDs/numbers are verified against the quotes.
- **Scoped MTTR, `null` for B**: MTTR comes from the matching runbook section only; RB-002's unconfirmed 15 min is never borrowed.

## Tradeoffs

Standard library only: no API key, no vector database, runs in the supplied
environment. Domain signatures (pool-timeout triple, queued/sent email pair)
are matched against the retrieved corpus content, never against incident
directory names or question IDs. A production version could swap the lexical
ranker for embeddings and add metrics/log parsers, but the evidence-extraction
and calibration contracts would stay the same.
