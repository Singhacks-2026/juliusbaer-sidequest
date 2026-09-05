# Use case 2 — Production Incident Investigator

**Name:** Vinoth
**Phone:** 82727344
**Email:** jvvinoth2@gmail.com
**Repo:** https://github.com/jvvinoth/production-incident-investigator

Files in this submission:

| File | What it is |
|---|---|
| `solution.py` | `investigate(query, corpus)` — the full pipeline, standard library only |
| `answers.json` | Output of `investigate()` for both incidents, produced by `run.py` |
| `run.py` | Loads each incident via `data/loader.py`, prints the per-axis corroboration table, writes `answers.json` |

Reproduce (Python 3.9+, no dependencies, no API keys):

- **From the repo above:** `python3 run.py`
- **From `submissions/vinoth/` inside the hackathon repo**, using its own loader:

  ```bash
  cd usecase-2-production-incident-investigator
  python3 -c "
  from data.loader import load_incident
  import json, sys
  sys.path.insert(0, 'submissions/vinoth')
  import solution
  names = ['incident_a_pool_exhaustion', 'incident_b_ambiguous_delay']
  answers = {n: solution.investigate(*load_incident(n)) for n in names}
  json.dump(answers, open('submissions/vinoth/answers.json', 'w'), indent=2)
  "
  ```

---

## Design

The starter's four-stage shape is kept on purpose — it's the honest shape of the problem.

**1. Ingest — split documents into small units, not whole files.**
Every log line, CSV row, markdown table row, bullet point and paragraph becomes its own retrievable unit, tagged with its source file, its *document type* (`logs`, `deployments`, `known_issues`, `precedent`, `runbooks`, `topology`) and the nearest heading. Log lines are parsed into timestamp / level / component / message; table and CSV rows keep their column names. This granularity is what lets later stages say "this exact line" rather than "somewhere in `logs.md`", and it is what makes `known_issues.csv` usable — one row is one candidate, not one blob.

**2. Retrieve — TF-IDF + cosine, written by hand.**
A light stemmer, a stopword list that removes the boilerplate common to both queries ("identify the probable root cause…"), and hyphenated names kept whole *and* split (`payment-gateway-adapter` also yields `payment`, `gateway`, `adapter`). Retrieval's only job here is to say *which component the query is about*. It does not decide the answer.

**3. Correlate — six independent evidence axes per hypothesis.**
Candidate components are nominated from the top-ranked units (only names the corpus itself treats as components — log emitters, known-issue owners, deploy targets). Failures logged in the same instant are treated as one event, so a downstream `Charge failed` line counts for the upstream component too. Candidates within 60% of the leader's relevance are all evaluated, and **corroboration, not relevance, picks the winner.**

Each candidate is tested against:

| Axis | Passes when | Weight |
|---|---|---|
| logs | repeated ERROR lines carrying an exception class or reason code (strong) · ERROR without one (moderate) · WARN only (weak) | 30 / 22 / 15 |
| known issue | a CSV row owned by the component whose signature matches the log signature | 18 |
| deployment | a deploy to that component timestamped **before** the first symptom, within 3 days | 20 |
| precedent | a prior incident with the same signature, not phrased as a denial | 14 |
| runbook | a runbook whose symptoms match | 10 |
| topology | architecture / API docs describe the component (context only) | 5 |

Every axis is keyed on document type and on generic signals — log level, `\w+Exception`, `reason=CODE`, timestamps, hyphenated component names. There is no incident-specific string anywhere in the file.

**4. Calibrate — confidence from agreement, minus hedges.**
`confidence = Σ(axis weights) − 3 × (number of distinct sources that hedge or deny)`, capped at 92 (a document corpus never *proves* a root cause) and floored at 5. A source "hedges" when it says things like *unconfirmed*, *unverified*, *may not apply*, *not currently instrumented*, *no documented SLA*; it "denies" with *no previous incident*, *no deployment*, *unrelated*, *first recorded*. Those sentences are the single most important calibration signal in the corpus and are exactly what a bag-of-words retriever scores as a *hit*. `needs_human_review` is derived from the score, never set by hand.

MTTR comes from the matched runbook's "Typical MTTR" line, falling back to the precedent's. If the sentence carrying the number hedges it, `mttr_minutes` is `None` and the number is surfaced in `remediation` with the caveat.

---

## What the actual difficulty was

Not retrieval. Both corpora are ~75 units; any similarity measure finds the relevant lines. The difficulty is that **the same machinery has to produce a confident answer for incident A and refuse to for incident B**, and everything about B is built to defeat a fluent first pass:

- B has **zero ERROR lines**. The only signal is one WARN. A system that scores on "did retrieval find something?" answers yes.
- B's documents *talk about* the right component constantly — in sentences that deny a link: "No previous incident … involves `notification-service`", "No deployment touched `notification-service`". Token overlap with the query is high. Semantically it's the opposite of corroboration.
- B's runbook exists and matches — and describes itself as *incomplete*, *unverified*, and gives an MTTR it says *may not apply*. A hit that argues against itself.
- B's known-issue row for the component (KI-114, broken HTML in old webmail clients) matches on component and on the word "email", not on the symptom.

So the real problem is: define "corroboration" tightly enough that each of those fails on its own, without ever mentioning incident B.

Incident A has a mirror-image trap: `payment-service` logs the customer-visible failure (`Charge failed … GATEWAY_TIMEOUT`) and is therefore *more relevant to the query* than `payment-gateway-adapter`, which logs the cause. Pick by relevance and you name the wrong component. Pick by which hypothesis more independent sources agree on, and the adapter wins 6 axes to 3.

---

## Why this approach, what was tried and abandoned, tradeoffs

**Hand-rolled TF-IDF over a vector store.** I considered putting the corpus behind Cloudflare Vectorize with an embedding model. Rejected: the decisive signals are exact tokens (`ConnectionPoolTimeoutException`, a deploy timestamp 17 minutes before the first error, `Typical MTTR: 20 minutes`) that embeddings blur, and dense similarity would rate B's negation sentences as strong hits — the precise failure this task punishes. For 75 chunks, cosine in pure Python is also faster than a network call and the file stays runnable by a reviewer with no account.

**Three things the first run got wrong on incident A**, all fixed by making a rule *more* generic, not less:

1. *Deployment window was 14 days.* That correlated `payment-service`'s v2.4.0 (13 days earlier) with the incident. Tightened to 3 days — "after yesterday's deployment" is not "after a fortnight ago's".
2. *Bullet lists without blank lines ingested as one paragraph.* The architecture doc's component list became a single unit mentioning every service, so `order-service` and `notification-service` leaked into A's impacted systems. Bullets are now their own units.
3. *Relevance gate blocked the right answer.* `payment-gateway-adapter` didn't clear 60% of `payment-service`'s relevance because the adapter's error text shares fewer words with the query. Fixed by sharing relevance across failures logged in the same second — they are one event — so both reach the corroboration stage, which then gets it right.

**Judgment call — `mttr_minutes` for incident B is `None`, not 15.** RB-002 gives 15 minutes and, in the same sentence, says the figure is from an unconfirmed different occurrence and may not apply. Reporting 15 as an integer would be manufacturing precision the source explicitly withdraws. The number is kept in `remediation` so a reader still has it.

**Tradeoffs under the hour.** Document types are classified by filename (`runbook`, `deploy`, `known`…), which is fine for this corpus and would want a content-based fallback in production. The stemmer is four `endswith` rules; "late" in the query and "delay"/"latency" in the docs never meet, and the answer holds anyway because the component signal carries it — a synonym table would be the next thing to add. Confidence weights are hand-set and defended in the table above rather than learned; with two incidents there's nothing to learn from.
