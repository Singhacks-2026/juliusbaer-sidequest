# Production Incident Investigator — Song Yangyu

- **Name:** Song Yangyu
- **Phone:** 87741535
- **Email:** flyfy1@gmail.com

This submission implements `investigate(query: str, corpus: dict) -> dict`
without an LLM, API keys, network access, or third-party dependencies.
`answers.json` contains the reports produced by this implementation for both
provided incidents. The starter, loader, and source datasets are unchanged.

## Intuition

The central idea is to treat a probable root cause as an explanation that
must account for **what failed, where it failed, and when it started**.
Retrieval supplies candidate evidence; correlation checks whether that
evidence actually supports the same explanation. A useful investigation
can also end with a confirmed symptom and an unresolved cause.

### Start with the affected operation

The query tells us which behavior needs explaining. For a payment failure,
a refund-webhook warning in the same adapter is not automatically relevant.
For delayed confirmation emails, an HTML-formatting error does not explain
delivery time. The investigator first narrows the operation and component,
then checks that an event describes a compatible symptom. Infrastructure
events such as queue depth need a documented resource relationship to that
operation. Direct queued/sent measurements help locate the execution path;
a frontend log repeating a customer's complaint is weaker localization
evidence.

### Follow the evidence into a mechanism

A user's description and a technical explanation often use different
vocabulary. “Payments are failing” can lead to a log containing
`ConnectionPoolTimeoutException`; that signature can then retrieve an issue
or runbook which the original query ranked poorly. This motivates two
retrieval passes: start with the symptom, then search using the observed
component and signature. TF-IDF is sufficient for finding these candidates
in this small corpus. Its similarity score does not establish causality.

Each document contributes a different part of the explanation:

| Evidence | Question it helps answer |
|---|---|
| Current logs | What behavior was actually observed, and when? |
| Architecture / API | Which resource and call path could connect it to the reported symptom? |
| Deployment history | Did a relevant change precede the first signal, and could its direction explain the failure? |
| Known issues / previous incidents | Has this signature been associated with the same cause, with applicable supporting history? |
| Runbook | What checks and actions apply, and what qualifications accompany them? |

For incident A, the proposed explanation is that the adapter's pool became
too small. The deployment reduced it from **50 to 10 at 14:30**; matching
timeouts first appear at **14:47:12** and coincide with payment-service
failures. The issue catalog associates that signature with an undersized
pool, the historical incident describes the same mechanism, and the runbook
provides matching diagnostics and remediation. Interleaved successes fit
the reported intermittent degradation. Together, these observations support
a probable configuration-induced capacity problem. Pool-utilization metrics
would still be needed to validate that explanation directly.

The relationships matter as much as the individual facts. A deployment
after the first warning cannot explain that onset. A timeout increase does
not establish a pool-size reduction. A historical connection leak can
produce the same exception while offering a competing cause. These cases
must not receive the same support as agreeing evidence.

### Separate certainty about the symptom from certainty about its cause

For incident B, four order-ID-linked queued/sent pairs establish delivery
intervals of **42.4–75.4 minutes**. They do not locate the delay within those
intervals. Insufficient consumer throughput and a slow downstream provider
could both fit the observations. A single queue-depth warning cannot
distinguish them, and the corpus lacks the needed per-stage measurements.
The useful answer is therefore to preserve the confirmed delay, identify
the missing measurements, and request human investigation.

This is why confidence depends on corroboration and conflicts. Repeating
one error many times strengthens the observation that it recurs; it does
not create a new independent explanation. An architecture document can
establish a dependency without proving which component caused the fault.
The implementation caps contributions by source type and reduces confidence
for conflicts or weak support. Its point values are explicit engineering
judgments, **not probabilities fitted to a representative incident dataset**.

### Carry uncertainty into the recommendation

Confidence in a cause, validity of an action, and confidence in a recovery
estimate are separate questions. An unverified runbook action stays
conditional even if the cause is well supported. A's **20-minute MTTR** is
the matched runbook's typical estimate, not a measured recovery of this
incident. B's runbook explicitly qualifies its **15-minute** estimate, so the
report returns `null`; observed email-delivery intervals cannot replace it.

### Test whether the reasoning changes for the right reasons

The two supplied outputs alone could be reproduced by a lookup table. The
regression tests therefore alter the evidence: remove corroborating sources,
negate a claim, introduce a competing cause, move a deployment into the
future, or add unrelated errors. Confidence or conclusions should respond
to those changes. Renaming files, duplicating evidence, and inserting an
unrelated release should preserve the relevant conclusion. These tests
check the intended reasoning properties within this corpus; broader
generalization still needs independent incidents and evaluation.

## Run

Python 3.9 or newer is sufficient. From the repository root:

```bash
python3 usecase-2-production-incident-investigator/submissions/song-yangyu/solution.py
```

The command discovers incident directories containing `query.txt`, loads
their Markdown/CSV documents, and writes `answers.json` beside `solution.py`.
It also accepts `--data-dir PATH` and `--output PATH`. Importing the module
performs no file I/O; the public `investigate()` function uses only its input
query and corpus and does not mutate either.

Run the additional regression tests with:

```bash
python3 -m unittest discover -s usecase-2-production-incident-investigator/submissions/song-yangyu -p 'test_*.py' -v
```

## Design

1. **Ingest evidence at useful boundaries.** CSV records are parsed with
   `csv.DictReader`, including quoted multiline fields. Each record retains
   its original source excerpt. Timestamped log events are parsed and exact
   duplicates removed. Each runbook and historical incident remains a
   separate section, keeping its symptoms, resolution, MTTR, and caveats
   together. Deployment rows retain their timestamp, component, and change.
   Event groups retain both their earliest onset and strongest representative
   observation, so an early WARN cannot hide later ERROR evidence.
   Source types are inferred from content rather than incident/file names.

2. **Retrieve in two passes.** Sublinear TF-IDF cosine similarity is
   implemented with the standard library. Token normalization splits
   CamelCase exception names, removes transaction identifiers, and handles
   a small explicit vocabulary of variants such as payments/charges and
   late/delayed. The symptom query locates the relevant log component. The
   full query is retained, including symptoms after an initial instruction
   or across paragraphs. Direct queued/sent measurements take precedence
   over observer logs that merely paraphrase the user's complaint.
   Deployment words are excluded during component selection so a deployment
   notification does not displace the actual failing operation. The second
   pass expands retrieval with the observed component and failure mechanism,
   and searches each corroborating source type.

3. **Correlate before concluding.** A linked component enters the impact
   scope only when errors occur within one second and architecture/API prose
   supports a direct relationship. Known issues, historical incidents, and
   runbooks require both component and symptom/signature agreement. Business
   operation and symptom type must also be compatible. An infrastructure
   event with no business noun requires resource-level architecture context;
   merely sharing a component is insufficient. Local negation is checked in
   symptoms and causal mechanisms. All matching issue/history candidates are
   examined for competing causes, and explicitly inapplicable or already
   fixed issues are excluded using the available version/time evidence.
   Historical examples must precede the observation when dated.

   A deployment must precede the earliest matching signal, lie within a
   documented 14-day window, and agree with the proposed causal mechanism
   and change direction. Changes are scanned backward for the latest
   relevant resource change: an unrelated logging release does not hide a
   prior pool reduction, but a later pool restoration supersedes it.
   Increasing a timeout cannot substantiate an undersized pool. Dates come
   from the corpus, not today's date or the word “yesterday” in the query.

4. **Calibrate evidence support.** Each source type contributes at most once:

   | Signal | Points |
   |---|---:|
   | Baseline | 5 |
   | Direct ERROR/FATAL event, or a weaker warning | 20 or 6 |
   | Matching, unqualified known issue | 20 |
   | Matching, unqualified historical precedent | 20 |
   | Correlated deployment | 25 |
   | Applicable runbook, or a qualified/unverified runbook | 10 or 2 |

   Scores are capped at 95. Warning-only evidence is capped at 35. Fewer
   than two supporting categories among known issue/history/deployment,
   competing similarly supported hypotheses, conflicting/inapplicable
   evidence, or no extractable causal explanation caps confidence at 45.
   Architecture/API context and repeated log lines do not add
   causal-confidence points. An empty investigation
   scores 5. These numbers express a transparent evidence-support heuristic,
   not a statistically calibrated probability. Different document types
   also do not guarantee complete independence of their underlying facts.

5. **Produce an auditable report.** The result has exactly the seven required
   fields. Each citation is checked against the supplied original text.
   Qualified runbooks cannot provide an applicable MTTR. For adequately
   supported causes, a matching runbook's typical MTTR takes precedence over
   a matching historical duration; otherwise the value is `null`. The prose
   explicitly labels the estimate as historical/typical, not an observed
   resolution. `needs_human_review` is always derived from
   `confidence_score < 50`.

   Action reliability is assessed separately from cause confidence. An
   explicitly unverified action stays conditional even when other evidence
   strongly supports the cause. Diagnostic steps do not count as a corrective
   action: if a runbook omits remediation, a matching historical resolution
   can supply it. If neither source supplies an action, the report says so.

## Results and validation

| Incident | Finding | Confidence | MTTR | Human review |
|---|---|---:|---:|---|
| A | Adapter pool reduced from 50 to 10 in v2.4.1; connection-pool timeouts propagate to payment-service | 95.0 | 20 minutes, typical runbook estimate | No |
| B | Email delivery delay confirmed; the bottleneck remains unconfirmed | 13.0 | `null` | Yes |

Incident A cites the five corroborating source documents as well as
architecture/API context. It includes successful requests around repeated
failures to support the intermittent nature of the problem. Its remediation
is to restore the pool baseline or size it to traffic, redeploy the adapter,
and verify that failures stop recurring.

Incident B derives the observed 42.4–75.4 minute delay range from four
order-ID-linked queued/sent pairs, without confusing it with recovery time.
The lone queue warning and explicitly unverified runbook cannot establish a
cause. The report identifies missing consumer/provider instrumentation and
keeps scaling consumers conditional on confirming that bottleneck. There
is no matching issue or historical precedent, and the listed deployments
post-date this incident and affect unrelated components.

All 33 regression tests pass. They cover both expected reports, exact output
types and source excerpts, confidence reductions after removing evidence,
duplicate logs/documents, renamed files and reversed input order, future
deployments, a pool increase, changed component names/configuration values/
MTTR, missing or qualified recovery estimates, delays without the queue
warning, multiline CSV, and reproduction of the saved answers. The additional
review regressions cover negation, conflicting causes, instruction-first
queries, unrelated operations/infrastructure/observer logs, action validity,
WARN-to-ERROR escalation, issue applicability, unrelated versus overriding
deployments, and historical action fallback.

Three review-and-fix rounds were completed with **gpt-5.6-sol, High reasoning
effort**. All 14 reported findings were addressed. See
[REVIEW_NOTES.md](REVIEW_NOTES.md) for the findings, changes, and validation
record for each round.

## Approach and tradeoffs

A small deterministic implementation makes retrieval, selection, and
confidence inspectable and reproducible without environment setup. A
standard-library TF-IDF implementation is sufficient for seven documents;
a vector database or agent framework would add setup without addressing
the central evidence-correlation problem. No incident directory names,
issue IDs, component-specific answers, or expected MTTR constants select
the result. IDs, configuration values, component names, and recovery
estimates in the report are extracted from the input. Only the CLI and
tests know the on-disk fixture layout.

During validation, initially ranking all log lines with the full symptom
query selected the deployment notification for incident A. Separating
deployment context from operation selection fixed that retrieval failure.
The counterfactual tests protect against reverting to a lookup based on the
two provided incident names or treating extra documents as extra certainty.

The implementation deliberately targets this corpus's English document
conventions and timestamped log format. The synonym list, business-operation
vocabulary, local negation handling, resource linkage, and cause comparisons
are bounded heuristics, not a general causal model. Cause comparison includes
explicit capacity-shortage, resource-leak, and throttling patterns, with a
conservative lexical fallback. Unfamiliar terminology or complex negation can
require extending these rules. The 14-day deployment window and
one-second error-correlation window would need tuning for another system.
The investigator selects one leading hypothesis and requests review when
evidence is weak or competing; it does not measure live pool utilization,
query external systems, perform remediation, or claim proof of causality.
