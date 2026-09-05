# Production Incident Investigator — Praveena Vijayan

## Candidate details

- **Name:** Praveena Vijayan
- **Phone:** ` +65 94876144`
- **Email:** praveenavj2210@gmail.com

## My understanding of the problem

The task is not simply document search. A highly ranked document may describe
the affected system without establishing causality, and realistic logs contain
unrelated warnings. The investigator must identify a leading hypothesis and
then measure whether independent evidence types—runtime logs, deployment
history, known issues, previous incidents, and runbooks—actually corroborate it.

Confidence calibration is therefore part of correctness. Incident A contains a
repeated runtime signature, a temporally aligned configuration change, a known
issue, precedent, and a matching runbook. Incident B contains a genuine delay
and an elevated-queue warning, but no measurement that distinguishes consumer
capacity from downstream provider latency, no correlated deployment, no known
issue match, and no precedent. A reliable system must report that ambiguity
rather than convert a plausible hypothesis into a confident root cause.

## Design

`solution.py` implements four stages:

1. **Format-aware ingestion.** Markdown is split into headings, paragraphs,
   table rows, and individual log lines. `known_issues.csv` is parsed into one
   retrieval unit per row so an irrelevant catalog entry cannot dominate the
   entire file.
2. **Lexical retrieval.** A small local TF-IDF cosine index ranks evidence units
   using query expansion for incident vocabulary such as delays, failures,
   timeouts, queues, recovery, and impacted components. Source-level rankings
   are derived from the strongest unit in each source.
3. **Cross-source correlation.** Log anomalies and stable machine identifiers
   such as exception names are matched to known-issue rows. The leading
   component is checked independently against deployment history, previous
   incidents, and runbook evidence. Negative and qualified statements such as
   "no deployment", "unconfirmed", and "may not apply" are not counted as
   strong corroboration.
4. **Confidence and report construction.** Confidence is calculated from the
   number and strength of independent source types, not the top retrieval
   score. Reports include exact source excerpts, impacted systems, actionable
   remediation, and MTTR only when the causal hypothesis is sufficiently
   corroborated. `needs_human_review` is derived directly from
   `confidence_score < 50`.

The implementation uses only Python's standard library. This keeps the solution
portable even though the repository provides optional data-science packages.

## Why I chose this approach

I prioritized deterministic behavior and auditability. A vector database or LLM
would add setup and variability without solving the central problem: deciding
whether separate sources agree. The corpus is small enough that transparent
TF-IDF ranking and explicit correlation are easier to inspect and defend.

The confidence formula intentionally gives a single runtime signal a score below
50. Strong confidence requires corroboration across several document types. An
explicitly incomplete or unverified runbook contributes only weak support, and
its MTTR is not reported as the incident MTTR. This prevents Incident B's
15-minute runbook estimate—explicitly described as potentially inapplicable—from
becoming a false recovery promise.

I considered treating every document as one text blob, but rejected it because
the known-issues catalog contains many decoy rows and logs contain unrelated
events. I also avoided hard-coding incident directory names or branching on the
two queries. Conclusions are selected from extracted log signatures,
components, known-issue matches, and corroboration signals.

## Results and interpretation

- **Incident A:** five strong independent signals support connection-pool
  exhaustion in `payment-gateway-adapter` following the pool reduction from 50
  to 10. The report assigns high confidence and uses the matching runbook's
  20-minute typical MTTR.
- **Incident B:** the queue-depth warning makes a notification backlog plausible,
  but the bottleneck is unconfirmed. The report leaves MTTR as `null`, assigns
  low confidence, requests human review, and recommends instrumentation before
  choosing between consumer scaling and downstream-provider remediation.

## Reproducing the answers

Run from `usecase-2-production-incident-investigator/`:

```bash
python3 -c "
from data.loader import load_incident
import json, sys
sys.path.insert(0, 'submissions/praveenavj')
import solution
answers = {}
for name in ['incident_a_pool_exhaustion', 'incident_b_ambiguous_delay']:
    query, corpus = load_incident(name)
    answers[name] = solution.investigate(query, corpus)
with open('submissions/praveenavj/answers.json', 'w', encoding='utf-8') as f:
    json.dump(answers, f, indent=2)
"
```
