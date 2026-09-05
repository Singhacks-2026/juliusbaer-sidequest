# Use Case 2 — Production Incident Investigator

**Author:** Si Hui (Ariel)  
**Phone:** +65 88509353  
**Email:** ong.sihui1@gmail.com

## What judges look for (and how this submission answers)

| Criterion | How this solution addresses it |
|---|---|
| **Genuine retrieval** | Chunked corpus (incl. per-row `known_issues.csv`) ranked with TF-IDF cosine + keyword/error-density boosts — not hardcoded filenames or incident names |
| **Correlation** | Hypotheses are component-scoped; confidence requires independent source types to agree (logs, deploy, known issue, runbook, precedent) — not “restate the top hit” |
| **Honest calibration** | Thin / uncorroborated evidence lands &lt; 50 with `needs_human_review=True`. Manufacturing confidence is treated as worse than saying “not sure” |
| **Generalization** | One `investigate()` path for every incident. No magic strings keyed to `incident_a` / `incident_b` |

## Design

`investigate(query, corpus)` pipeline:

1. **Ingest** — Split logs, deploy rows, markdown sections; split `known_issues.csv` into `known_issues.csv#KI-…` candidates; tag source types.
2. **Hybrid retrieve** — sklearn TF-IDF + symptom/exception boosts; diversify top-N per source type.
3. **Correlate** — Seed hypotheses from ERROR (else WARN) components + query theme; bind signatures to the same log line’s component; require deploy rows to match *that* failure mode; skip decoy/cosmetic known issues; count negative phrases (“unverified”, “no deployment”, “first recorded”, …).
4. **Calibrate** — Score from strong corroboration count; `needs_human_review = confidence_score < 50`.
5. **Evidence selection** — Prefer hard sources (logs / deploy / known issues / runbooks / previous incidents). Drop soft architecture padding when already well corroborated. On thin cases, enrich logs with queued→sent delay and surface negative deploy/precedent statements.
6. **Optional LLM polish** — Opt-in only (`INCIDENT_LLM_POLISH=1`) plus an API key. Rewrites `root_cause` / `remediation` only; scores, MTTR, systems, and evidence stay deterministic. Default path is fully offline so organizers can re-run `solution.py` and match `answers.json`.

```bash
# from repo root
pip install -r requirements.txt
cd usecase-2-production-incident-investigator
python submissions/Si-Hui-Ariel/solution.py   # writes answers.json
```

## Understanding of the problem

The hard part is not search. Architecture mentions “payment” a lot; logs carry decoy lines from unrelated known issues. A trusted assistant has to notice when *independent* documents tell the same story — and when they don’t. One incident is richly corroborated (pool cut → timeouts). The other is deliberately thin (one WARN, incomplete runbook, no deploy, no known issue, no precedent). Fluently guessing there should score worse than low confidence.

## Why this approach (and what I abandoned)

- **TF-IDF + heuristics** over embeddings: matches the brief, no API required for correctness, deps already in-repo.
- **Component-scoped signatures** after a version that leaked exceptions across components and falsely “confirmed” the thin incident via an unrelated payment-gateway deploy.
- **Corroboration counting** for confidence: encodes the rubric directly.
- **Agree/disagree prose** in `root_cause` so the report is defensible to an on-call engineer.
- Abandoned: treating CSV as one blob; treating any deploy row as correlation; always trusting runbook MTTR on thin evidence.

## Contact

Si Hui (Ariel) · +65 88509353 · ong.sihui1@gmail.com
