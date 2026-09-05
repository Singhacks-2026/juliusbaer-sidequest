# Production incident investigator

Participant: Lim Hur (GitHub: `lhurr`).

This submission implements use case 2 with Python 3.10+ and the standard library.
It runs offline without an API key or model download.
`solution.py` generates `answers.json` from the supplied documents and includes a reproducible behavioral evaluation.

## Findings

| Incident | Finding | Evidence strength | MTTR | Human review |
|---|---|---:|---|---|
| A | The adapter's v2.4.1 pool reduction from 50 to 10 is the probable trigger for intermittent connection-acquisition failures. | 95/100 | 20 minutes, estimated | No |
| B | Queue-to-send waiting times are measurable, but the evidence cannot distinguish consumer capacity from provider latency. | 30/100 | Unknown (`null`) | Yes |

For A, the 14:30 deployment precedes the first matching error at 14:47:12 by 17 minutes 12 seconds.
Five `ConnectionPoolTimeoutException` events match KI-101, RB-014, and INC-2031.
The adapter's 5000ms acquisition timeouts coincide with failed charges in `payment-service`, while successful charges between failures show intermittent degradation.
These two components are directly affected; architecture alone does not establish impairment in other services.
The mitigation is to validate saturation and current configuration, restore suitable pool capacity, redeploy, and verify sustained recovery.
The 20-minute figure is RB-014's typical estimate, supported by a similar incident's 22-minute recovery.
The current incident has no recorded resolution time, so this is not measured MTTR.

For B, four unambiguous queue/send pairs yield 46m35s, 42m26s, 75m24s, and 56m43s, averaging 55.28 minutes.
These measurements describe waiting time; without a latency target they do not independently establish an SLA breach.
A send event also does not establish inbox delivery.
One queue-depth warning cannot identify which processing stage is slow.
The deployment history concerns other components, the catalog describes a cosmetic email issue, and no matching historical incident exists.
The runbook explicitly leaves scaling and its 15-minute estimate unconfirmed.
The next action is human investigation using per-stage timestamps, consumer throughput, backlog age, and provider latency, followed by a supported mitigation.

## My understanding of the problem

The difficulty is deciding whether relevant text actually supports a claim about this incident.
Retrieval can find the right exception in the wrong operation, the right service with a different pool, or a valid procedure for an unrelated cause.
Exact citations can therefore be authentic while the resulting diagnosis is wrong.
The solution must distinguish symptoms, a probable mechanism, a temporally eligible trigger, downstream impact, and an estimated recovery duration.
It must retain enough positive coverage that returning "unknown" for everything cannot pass evaluation.

## Design

1. **Parse source roles and retain provenance.**
   Content identifies logs, deployment records, catalog CSV, architecture descriptions, and procedural sections.
   Filenames are citation labels rather than incident identifiers or answer selectors.
   CSV parsing supports quoted commas and multiline fields.
   Runbook sections remain intact so recommendations keep their scope and uncertainty qualifiers.
   Example logs inside a runbook do not become observations of the current incident.
   All returned excerpts must be nonempty exact substrings of their named sources.
2. **Retrieve candidate evidence.**
   BM25 ranks passages with term-frequency saturation and length normalization (`k1=1.5`, `b=0.75`).
   Query normalization scopes the operation and component, then an affirmative runtime signature expands retrieval to matching catalog mechanisms and references.
   Signatures include exception names, status identifiers, and limited structured symptom phrases.
   Retrieval scores rank candidates; they do not establish causality or determine confidence.
3. **Check applicability and chronology.**
   Runtime observations must affirm the relevant failure, rather than negate it, report zero occurrences, or describe resolved/example events.
   Operation and resource qualifiers prevent a refund failure from explaining a charge incident or a provider pool from explaining a database pool.
   Runbooks, history, and deployment changes must satisfy their own applicability checks.
   Configuration values are associated with their resource, preventing a retry-count change from becoming a pool-capacity recommendation.
   The latest relevant preceding configuration change is considered; an intervening logging-only release does not erase an earlier pool change.
   A later mitigation is context, not evidence that it caused the earlier onset or proved recovery.
   Timestamps normalize to UTC, including fractional seconds and numeric offsets; unzoned timestamps are assumed UTC.
4. **Correlate impact and queue observations.**
   Downstream impact needs an explicit directed dependency, a matching operation, and linked failures.
   Shared identifiers support linkage within a bounded 30-second window; an exact-time, matching-signature fallback handles the supplied legacy logs.
   Queue measurements prefer unique message IDs over broader request, trace, or order IDs.
   Pairs must be one-to-one, chronological, in the same component, affirmative successful sends, and consistent on every shared identifier.
   Ambiguous duplicates or conflicting metadata are excluded rather than silently paired.
5. **Render a supported answer or explicit uncertainty.**
   Competing hypotheses and contradictions are checked before rendering recommendations.
   Insufficient evidence produces an unconfirmed hypothesis and diagnostic steps without prescribing an unsupported rollback.
   A recovery estimate requires an applicable, qualified source; queue waiting time never supplies MTTR.
   The output uses the seven required fields, with `needs_human_review` derived from the final score.

## Confidence and limitations

Confidence is an ordinal evidence-strength rubric, not a statistically calibrated probability.

| Corroboration channel | Maximum contribution |
|---|---:|
| Affirmative matching runtime observations | 25, or 15 for one event |
| Applicable catalog mechanism | 25 |
| Applicable preceding deployment | 20 |
| Applicable earlier incident | 12 |
| Applicable qualified runbook | 10 |
| Architectural consistency | 3 |

Each causal source filename receives at most one vote; repeated events and identical documents cannot accumulate unlimited points.
Fewer than three causal source files, or no supported preceding deployment, caps the score at 49.
Contradictions and close competing hypotheses cap it at 40, and the overall ceiling is 95.
The queue-only path remains below the review threshold, with a queue warning raising its score from 25 to 30.
Source roles are corroboration channels, not statistically independent observations.
Requiring a supported deployment is intentionally conservative and can abstain on a real capacity problem without change records.

This investigator uses heuristic parsing for the supplied English document schemas.
It does not prove causality, understand arbitrary prose, reconstruct missing telemetry, or measure complete recovery from isolated successes.
Operation, resource, negation, and heading recognition can still fail on unfamiliar language or formats.
When multiple distinct capacity changes are present, the numeric baseline is omitted rather than chosen arbitrarily.
The exact-timestamp fallback is weaker than trace context and should be replaced when production telemetry supports explicit causal links.
Document packaging can affect source-count confidence; different files do not necessarily supply independent corroboration.
A production study would need new incidents, human claim review, representative failure modes, and calibration analysis before treating scores as probabilities.

## Why this approach, and what changed

The small corpus favors explicit, inspectable rules over an embedding service, vector database, or generative model.
Those alternatives were considered but not implemented; signatures, operations, configuration changes, and qualifiers carry the decisive evidence here.
BM25 supplies candidate ranking without network or model-version dependencies, and every candidate in a required source role can be checked without a lossy global top-k cutoff.
An exception index reduces repeated log scans.

The initial implementation answered the supplied incidents, but independent counterexamples exposed overconfidence.
I froze that baseline and used separate evaluation and adversarial review to drive repeated changes.

| Round | Evidence from evaluation | Improvement |
|---|---|---|
| Baseline | 21/26 development cases; authentic citations accompanied incorrect claims. | Freeze implementation, inputs, labels, and hashes before tuning. |
| First revision | 23/26 development cases, with temporal and semantic applicability failures. | Normalize timestamps, reject negated observations, distinguish relevant changes from unrelated releases, and correlate message identities. |
| Second revision | 26/26 development cases; first blind holdout improved from baseline 5/10 to 8/10. | Preserve positive coverage while rejecting inapplicable remedies and uncertain recovery estimates. |
| Third revision | After the holdout was released, 36/36 reused cases passed. | Separate resource types and operation subjects, including provider/database pools and report/charge processing. |
| Final targeted review | Reproduced mixed-number and refund-only deployment errors outside the frozen suite. | Bind configuration numbers to their resource and apply operation scope to deployments; retain both as regressions. |

Several early shortcuts were abandoned: treating every later deployment as a contradiction, using the first textual identifier for queue pairing, matching procedures only by component and exception, and selecting the first numeric change in a deployment description.
These shortcuts failed concrete end-to-end examples.
The replacement rules add code and remain imperfect, but make the failure modes explicit and testable.
The single-file submission includes evaluation fixtures to satisfy the three-file requirement, so its length reflects both implementation and tests.
With more time, I would first obtain new labeled formats and strengthen parsing against them, rather than add an unvalidated model or more confidence weights.

## Reproduce and evaluate

Run from the repository root:

```bash
python3 usecase-2-production-incident-investigator/submissions/lim-hur/solution.py
python3 usecase-2-production-incident-investigator/submissions/lim-hur/solution.py --self-test
python3 usecase-2-production-incident-investigator/submissions/lim-hur/solution.py --evaluate --evaluation-output /tmp/incident-evaluation.json
```

The first command regenerates `answers.json` from both official incident directories.
`--data-dir` and `--output` support alternative locations.
The self-test exercises contracts, metamorphic transformations, evidence ablations, and targeted regressions; its printed assertion count is not a count of independent incidents.
The separate `--evaluate` mode runs 36 distinct cases through `investigate(query, corpus)` and saves per-case reports, expectations, dimensions, timings, and hashes.
Use `--split dev` or `--split holdout` to reproduce the original partition, but these cases are now public regression tests.
Evaluation exits nonzero if any case fails.
Investigation never consumes the evaluator's fixtures or labels.

| Frozen 36-case suite | Baseline | Final regression result |
|---|---:|---:|
| Cases passing every dimension | 26/36 | 36/36 |
| Correct answers on answerable cases, confidence at least 50 | 12/16 | 16/16 |
| Cases with high-confidence false claims, score at least 70 | 4 | 0 |
| Exact citation provenance | 36/36 | 36/36 |
| Crashes | 0 | 0 |

The first blind result was **8/10**, not 10/10, compared with baseline 5/10.
All four blind positive cases were answered correctly, compared with baseline two of four.
The final 36/36 includes fixes informed by the released holdout and must not be presented as fresh held-out accuracy.
An additional adversarial suite improved from 5/26 to 26/26, with reference and identifier probes retained separately.
These are selected, related synthetic cases plus the two official incidents, not a representative production benchmark or the organizer's private answer key.
Case labels test cause, impact, MTTR, confidence applicability, provenance, source diversity, schema, and queue arithmetic.
Semantic checks use explicit lexical expectations and need human review when report wording changes.
The unchanged provenance score illustrates why valid citations alone are insufficient evaluation.

The baseline is available at [commit d121f4a](https://github.com/lhurr/juliusbaer-sidequest/blob/d121f4ad59720df1f46f9cf2b0184d3f899c1b7b/usecase-2-production-incident-investigator/submissions/lim-hur/solution.py).
Save that trusted file locally and add `--evaluate-against /path/to/baseline_solution.py` to run the embedded evaluator against it.
The fixture SHA256 is `d6170e81140306c22aef5b0deebccb5a2d602d38cf38cb54abe9b76f736efdfe`.
The baseline implementation SHA256 is `566ba7c7449d1a0564ebce1e369c9c5cee1ccc2b6123f12fa905efac71508a91`.
The pre-disclosure candidate SHA256 was `9f84c47672fdaea481a1cd864e5f29126646db1cc9e94250fb0a187f97878f48`.

Ruff lint and formatting checks pass:

```bash
uvx ruff check usecase-2-production-incident-investigator/submissions/lim-hur/solution.py
uvx ruff format --check usecase-2-production-incident-investigator/submissions/lim-hur/solution.py
```

Development and verification used Python 3.14.4 and Ruff 0.16.6.
Python 3.10-compatible syntax and standard-library APIs are used, but every supported Python version was not tested.
The official data loader was separately checked against the generated JSON.
The end-to-end interface is the corpus, public function, and generated artifact; there is no browser UI in this use case.
An alternating local benchmark of 500 calls per implementation measured median latency of 2.43ms for the baseline and 3.67ms for the final version, with final p95 of 4.90ms.
The extra semantic checks increased latency; I accepted this small absolute cost for the measured correctness improvements.
These two-input measurements are not production throughput estimates.

## Research informing the iteration

- [Stanford IR textbook: BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) informed retrieval ranking and its separation from causal reasoning.
- [Google SRE: Effective Troubleshooting](https://sre.google/sre-book/effective-troubleshooting/) informed testing hypotheses against confirming and disconfirming evidence.
- [OpenTelemetry messaging spans](https://opentelemetry.io/docs/specs/semconv/messaging/messaging-spans/) informed distinctions between message identity, broader conversation identifiers, and linked processing stages.
- [CheckList, ACL 2020](https://aclanthology.org/2020.acl-main.442/) informed capability-based tests of minimum functionality, invariance, and directional changes.
- [Metamorphic testing review](https://www.cs.hku.hk/data/techreps/document/TR-2017-04.pdf) informed transformations where expected relationships between outputs are clearer than a complete answer oracle.
- [Selective classification, Gangrade et al.](https://proceedings.mlr.press/v130/gangrade21a.html) motivated measuring useful answer coverage alongside errors and abstention.
- [Calibration, Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) informed the distinction between an evidence score and a calibrated correctness probability.
- [Empirical evaluation of microservice root-cause analysis](https://arxiv.org/html/2408.13729v2) motivated caution about transferring small synthetic results to unseen incidents.

These sources informed methods and proposed instrumentation only.
All submitted incident findings and evidence excerpts come from the challenge corpus.
AI assistance was used for research, implementation, independent review, and evaluation development.
