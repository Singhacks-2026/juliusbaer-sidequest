# Production Incident Investigator — Sumit

**Name:** Sumit  
**Phone:** 97518007  
**Email:** sumit.shinde@u.nus.edu

## Understanding of the problem

This is not a search problem. Ranking documents against the query (TF-IDF alone)
often surfaces the architecture overview because it repeats domain words like
"payment" — while the actual root cause only appears when **independent source
types agree**: runtime errors, a temporally prior deploy change, a known-issue
signature, a matching runbook, and a historical precedent.

The harder half is **calibration**. Incident A has dense multi-source
corroboration and should score high. Incident B has a real symptom (email queue
lag) but almost no corroboration — one WARN, an incomplete/unverified runbook,
no deploy, no known issue, no precedent. Manufacturing a confident answer there
is worse than saying "not sure; investigate next."

## Design

`solution.py` implements a **corroboration-graph investigator**:

1. **Ingest** — split the corpus into semantic units: timestamped log lines,
   per-row known issues (`known_issues.csv#KI-…`), deploy table rows, and
   markdown sections (runbooks / prior incidents).
2. **Retrieve** — hybrid ranking: TF-IDF cosine + BM25, fused with Reciprocal
   Rank Fusion (RRF). Query expansion pulls exception names and service tokens
   from ERROR/WARN logs so shared vocabulary cannot drown the causal docs.
3. **Hypothesize + corroborate (retrieval-driven)** — the RRF top-K chunks are
   the candidate pool for deploy / known-issue / runbook / precedent matching.
   Correlation only promotes a signal when that source type appears in the
   ranked set (with a full-corpus fallback if a type is missing from top-K).
   Same-component decoys (e.g. a refund-webhook known issue on the adapter) are
   rejected unless their signature overlaps the leading ERROR pattern.
4. **Calibrate** — confidence = f(positive independent types) minus penalties
   for missing deploy / known issue / precedent, unverified runbooks, and
   WARN-only (no ERROR) evidence. `needs_human_review = confidence_score < 50`.
5. **Brief** — on-call style report: verdict-first `root_cause`, numbered
   `remediation`, story-ordered `supporting_evidence` (including real negative
   corpus quotes when confidence is low).

```
query + corpus
    -> ingest chunks
    -> hybrid retrieve (TF-IDF + BM25 + RRF)
    -> top-K candidates feed correlation
    -> corroboration graph (+ decoy / negative evidence)
    -> confidence + on-call brief
```

## Approach and tradeoffs

**Chose:** structured multi-source fusion over "top-1 document = answer."
Retrieval ranks the pool; corroboration checks whether independent source
types in that pool actually agree.

**Chose:** deterministic templates for `root_cause` / `remediation` instead of
an LLM narrator. For a seven-document corpus, templates keep the brief
scannable, reproducible, and honest on the thin-evidence case.

**Chose:** hybrid BM25 + TF-IDF + RRF rather than embeddings. No API key, no
GPU, deps already in the repo `requirements.txt`, and RRF is robust when one
ranker overweights shared tokens.

**Abandoned:** treating the whole `known_issues.csv` as one blob (noise from
unrelated rows dominates). Per-row candidates let a pool-exhaustion KI win
without letting a same-component refund-webhook KI hijack the hypothesis.

**Tradeoff under time:** confidence is a transparent additive formula, not a
trained calibrator — easy to audit, and it keeps dense cases high while thin
evidence stays below 50.

## How to regenerate `answers.json`

From `usecase-2-production-incident-investigator/` (repo-root `.venv`):

```bash
python run_answers.py
```
