# Production Incident Investigator — Victor Gaya

**Name:** Victor Gaya
**Email:** g.victor@hotmail.fr
**Phone:** +65 8314 9070

`solution.py` is standard library only (no numpy / pandas / scikit-learn),
Python 3.9+, and makes no network calls. It produced `answers.json` on its
own; nothing else in this folder is needed to reproduce it.

```bash
cd usecase-2-production-incident-investigator
python3 -c "
from data.loader import load_incident
import json, sys; sys.path.insert(0, 'submissions/victor-gaya')
import solution
answers = {}
for name in ['incident_a_pool_exhaustion', 'incident_b_ambiguous_delay']:
    query, corpus = load_incident(name)
    answers[name] = solution.investigate(query, corpus)
json.dump(answers, open('submissions/victor-gaya/answers.json','w'), indent=2)
"
```

| File | Role |
|---|---|
| `solution.py` | the investigator — deterministic, offline, the graded pipeline |
| `answers.json` | its output for both incidents |
| `README.md` | this write-up |
| `llm_review.py` | **supplementary**: an adversarial QA harness I used to attack my own output. Not part of the pipeline; `solution.py` never calls it |
| `llm_review_findings.json` | the last review it produced, kept as evidence for the claims below |

**Visual walkthrough:** https://claude.ai/code/artifact/0c5ba135-9508-4e11-932b-1700b89d3dd3 — confidence scale, the four-document corroboration matrix, the calibration arithmetic, and the ranked scenario set for incident B.

## Results

| | Incident A | Incident B |
|---|---|---|
| Confidence | 95.0 | 21.2 |
| `needs_human_review` | `false` | `true` |
| Component | `payment-gateway-adapter` | `notification-service` |
| MTTR | 20 | `null` (see below) |
| Corroborating document kinds | 4 (deployment, known issues, runbook, precedent) | 1 (runbook, and it disclaims itself) |
| Evidence cited | 11 excerpts across 7 documents | 8 excerpts across 6 documents |
| Output shape | one corroborated conclusion | **ranked scenario set** — 3 candidates, each with the check that would settle it |

---

## My understanding of the problem

The difficulty is not retrieval. Rank either corpus against its query with
any reasonable similarity measure and the architecture overview comes out
near the top, because that is where the nouns in the query live. It is
also the one document that contains no evidence at all.

The actual difficulty is that **the two incidents are indistinguishable at
the level of a single document**. Both have a query describing a symptom.
Both have logs containing a plausible-looking anomaly for that symptom.
Both have a runbook that names the symptom. A system that answers "what
does the best-matching document say" produces a fluent, confident,
similarly-shaped answer for both — and is wrong on one of them, in the
direction that costs most: telling an on-call engineer to go scale
notification-service consumers at 3am on the strength of one unconfirmed
warning.

What separates them only becomes visible when you ask a different
question: *how many independent kinds of document say the same thing?*

- **Incident A**: the logs show the failure, the deployment record shows a
  pool-size reduction 17 minutes before it starts, the known-issues
  catalogue has the exact signature and names the same component, the
  runbook describes the symptom set and prescribes a fix, and a prior
  incident records the identical root cause. Five documents, four of them
  independent of the logs, all converging.
- **Incident B**: one WARN line. The known-issues catalogue has a
  notification-service row, but it is about HTML rendering, not latency.
  The deployment history explicitly states nothing touched the service that
  month. The incident history explicitly states this is the first
  occurrence. The one runbook that matches disclaims its own applicability
  in the same paragraph as its MTTR figure.

There is a second trap inside the first. Incident B's symptom is *not*
thinly evidenced — the logs quantify it precisely (40–75 minute gaps
between "Email queued" and "Email sent", every send eventually succeeding).
What is thin is the evidence of a **cause**. A report that says "the
evidence is thin" flattens those into one claim and gets both wrong: it
understates what is known and overstates what is guessed. The output
separates them explicitly.

## Design

Four stages, following the shape the starter suggests.

**1. Ingest.** `corpus` becomes a flat list of `Unit`s at a granularity
chosen per document kind: log files per line, CSV catalogues per row, prose
per markdown section *and* per block within a section. Document kind is
inferred from the filename with a content-based fallback, so nothing
depends on a specific name. Every block is a contiguous substring of its
file, which keeps cited excerpts verbatim.

The per-block split matters more than it looks: an architecture section
describing four components in four bullets has to yield four units, or
every component inherits an identical relevance score from the diagram that
names them all. That bug cost me incident B on the first run —
`web-frontend` outranked `notification-service`.

**2. Retrieve.** Hand-rolled TF-IDF with cosine similarity. Two
adjustments earned their place:

- *Query boilerplate is stripped.* Every incident query ends with the same
  asks ("identify the probable root cause", "what is the mean time to
  recover"). That vocabulary is identical across incidents, so it carries
  no signal — but it does damage: "mean **time** to recover" made
  `Checkout page render **time** 3900ms` the top-ranked anomaly for
  incident B. The symptom sentence is weighted 4× the rest.
- *Symptom coverage supplements cosine.* Cosine normalises by document
  length, punishing exactly the long, specific paragraph that explains a
  symptom. Alongside it I compute recall over the query's distinctive
  terms, with prefix-tolerant matching standing in for a stemmer
  (`payments`/`payment` match; `late`/`latency` deliberately does not).

**3. Correlate.** Candidate hypotheses come from *anomalous log lines* —
not from the top-ranked document — grouped by `(component, normalised
signature)`. Each is then put to every non-log document kind: does it
independently corroborate this? A match needs both the component and real
signature overlap; naming the component alone is not corroboration, or
every catalogue row about a service would corroborate every failure of it.
Only the strongest corroboration per kind is kept — five runbook paragraphs
about one component are one independent opinion, not five.

Two signals beyond matching:

- **Temporal support**: a deployment record becomes a *correlation* only if
  the change it describes precedes the first anomaly.
- **Recorded absence**: sentences like "No deployment touched
  `notification-service` in the month before this incident" are parsed as
  evidence in their own right. An absence counts against a hypothesis only
  where it explains a *missing* corroboration. In incident A the deployment
  history also contains an absence sentence ("no *other* deployments
  touched payment-gateway-adapter") — there, deployment evidence is
  present, so the same sentence is bookkeeping, not a penalty. Getting this
  wrong cost incident A six points for no reason.

**4. Calibrate.**

```
25   base — a reproducible anomaly exists in the logs
+15  per distinct corroborating document kind
 +8  a recorded change demonstrably precedes the symptom
+10  × alignment with the question asked
-10  per corroborating source that disclaims itself
 -5  per explicitly recorded absence (capped at 2)
 ×   relevance damping, then clamped to [5, 95]
```

Weights are set so one lone hedged signal cannot reach 50 however well it
matches the query, and three or more independent kinds cannot fall below
it. The 95 ceiling encodes that documents alone never justify certainty.
`needs_human_review` is derived from the score, so the two cannot disagree.

## Scenarios, not a single answer, when information is thin

This is the part of the design I care most about, because it is what makes
the tool usable on the incidents that actually hurt.

A confident single answer is only useful when the evidence supports one. On
a thin incident, handing an on-call engineer one hypothesis dressed as a
conclusion does two kinds of damage: it sends them down one path, and it
leaves them **surprised** when that path dead-ends at 3am — because nothing
told them what the other possibilities were or how to tell them apart. The
failure mode is not just "wrong answer"; it is "no plan for being wrong".

So below the review threshold the report changes shape. Instead of one
conclusion it produces a **ranked scenario set**: the competing
explanations, each scored on the same calibration, each with the specific
evidence that would confirm or eliminate it. That turns an unanswerable
question into an actionable one — the responder stops asking "is this
right?" and starts asking "which of these three is it, and what do I check
first?" Every branch is anticipated, so none of them is a surprise.

Reporting the single best hypothesis as though it were the answer *is* the
overconfidence trap — it just wears a humble face when a low number is
attached to it. A low confidence score alone tells a responder to be
worried; a ranked scenario set tells them what to do about it.

Two kinds of alternative are offered, and they are different in nature:

- **Component-level candidates**, scored on the same calibration. These are
  filtered by how well they address the reported symptom. My first version
  listed the refund-webhook delay and the checkout-latency warning as
  candidates for late emails — but the corpus documents both as unrelated,
  and offering an explanation that does not explain the symptom is the
  overconfidence trap again, one level down. Only hypotheses within 60% of
  the leader's symptom alignment survive.
- **Factor-level candidates**, extracted from design passages that name a
  contributor and then disclaim it. Incident B's architecture says
  "Consumer pool size and the third-party email provider's own latency are
  both outside this service's direct control and are not currently
  instrumented with per-stage timing." Those two factors are the real
  competing explanations, and they are reported **without** a confidence
  figure — the corpus states they are not measured, so attaching a number
  to them would manufacture precisely the false precision this pipeline
  exists to avoid.

Incident B's `root_cause` therefore ends with a ranked shortlist, the
sentence explaining why the alternatives cannot be separated from the
record, and a recommendation for instrumentation rather than remediation.
The responder finishes reading knowing three things they did not know
before: that the failure is real and quantified, that there are exactly
three live explanations, and that the record cannot separate two of them —
so the next action is to instrument, not to guess.

Incident A gets no shortlist. Where one hypothesis is corroborated four
independent ways, manufacturing doubt is as dishonest as manufacturing
confidence, and a shortlist would only dilute a clear instruction. The
scenario set is a response to uncertainty, not a house style.

## Using an LLM: as a critic, never as an author

`solution.py` is deterministic and offline, and it alone produced
`answers.json`. That was a deliberate constraint, for a reason specific to
this submission: the output schema is fixed at exactly seven keys, so there
is no field in which model-written prose could declare itself, and a
reviewer re-running the code without an API key would get different
wording. A pipeline whose output cannot be reproduced from the code that
ships with it is worse than one with plainer prose.

So the model went where it can only help: **attacking my own output**.
`llm_review.py` feeds Gemini the full corpus, the query, and my report, and
asks it to find unsupported claims, missed evidence, and miscalibrated
confidence. Its criticisms are then verified deterministically before I am
allowed to act on them:

- a "missed evidence" excerpt is discarded unless it is a verbatim
  substring of the file the model attributes it to;
- an "unsupported claim" is discarded unless the quoted text really does
  appear in the report under review.

The same discipline as the investigator itself: a language model is not a
source of truth, so its output gets checked against the documents.

**What it caught, that I acted on:**

1. *"resting on a single hedged signal"* (incident B) — verified as
   unsupported. The logs document the symptom extensively; only the *cause*
   is uncorroborated. This is what led to separating symptom evidence from
   causal evidence, the sharpest improvement in the whole submission.
2. *MTTR presented as measured.* I reported 20 minutes for incident A with
   no indication it came from a runbook's typical figure rather than an
   observed recovery. Provenance is now always stated.
3. *Missed mechanism evidence.* The API spec's 5000ms timeout definition
   and the architecture's connection-pool description explain why this
   failure produces this symptom. Both are now cited, as is the
   `payment-service` GATEWAY_TIMEOUT line showing the customer-facing
   effect, and the runbook's actual remediation instruction.
4. *Truncated citations.* An excerpt cut mid-clause ("Compare the current
   pool") reads as a defective citation; excerpts now cut at sentence
   boundaries.

**What it argued for, and I rejected:** the critic insisted, twice, that
incident B should report the runbook's 15-minute MTTR rather than `null`,
because the query asks for MTTR. That figure's own source says it "is from
a different, unconfirmed prior occurrence and may not apply here."
Reporting it as this incident's MTTR would convert a caveated number into
an apparently measured one — exactly the failure mode the brief warns
about, and exactly the direction an LLM pulls by default: toward the
answer-shaped answer. `mttr_minutes` stays `null`; `remediation` states
that the figure exists, gives it, and explains why it was not adopted.

That disagreement is the argument for this architecture. The critic is
useful precisely because it is not in charge.

## Why this approach, and what I abandoned

**Corroboration counting over similarity ranking.** The first version
ranked documents and summarised the top hits. It scored incident B at 71
with a confident-sounding root cause — precisely the failure the brief
warns about. Ranking tells you what a document is *about*; it cannot tell
you whether anything else agrees. Retrieval stayed, but demoted: it ranks,
corroboration decides.

**Standard library over scikit-learn**, though `requirements.txt` offers
it. TF-IDF plus cosine is about thirty lines; taking the dependency would
have hidden the part of the pipeline that is actually being graded.

**A component is a structural fact, not a word shape.** I first detected
components with a hyphenated-lowercase regex. It reported incident A's
impacted systems as `['payment-gateway-adapter', 'ord-88350',
'payment-service', 'ord-88351', ...]` — order IDs lowercase into exactly
the same shape as service names, and prose is full of `third-party` and
`per-stage`. Components now come from two structural signals that
generalise to any log corpus: the emitter field of a log line, and any CSV
column named like "component".

**Threshold-free relevance damping.** Running incident B's query against
incident A's corpus returned 94.8 — well-corroborated, but an answer to a
question nobody asked. The tempting fix was a cutoff on alignment. Every
value that would have worked sat in the narrow gap between 0.179 (the
mismatch) and 0.222 (incident B) — a threshold fitted to two samples, which
is the same overfitting the brief warns about in another guise. I used
smooth multiplicative damping instead. It is weaker: the mismatch still
returns 80.5. But it degrades gradually, and it made an empty query against
a rich corpus drop from 93 to 37 and correctly flag for review.

**Claims are gated on what the documents say.** The narrative for incident
A asserts no other change touched the component in the window. That is true
— the deployment history says so — but the first version asserted it
unconditionally from a template. It now appears only when an absence
statement about that component is actually found, and otherwise degrades to
"the closest recorded change preceding the failure."

## Tradeoffs and known limitations

- **The query/corpus mismatch case still scores 80.5.** Corroboration
  dominates by design and alignment can only damp it. Correct handling
  would mean checking whether the hypothesis explains the *specific
  symptom*, not merely relates to it lexically.
- **The strongest argument for incident B is one I do not make.** The email
  delays begin at 08:55; the queue-depth warning is at 11:10. The single
  piece of causal evidence post-dates the symptom it supposedly explains,
  which ought to reduce confidence further. Detecting that generically
  needs symptom occurrences extracted from `INFO` lines and correlated as
  paired events — I judged it too speculative to build reliably in the time
  and did not want to special-case it.
- **Factor extraction is shallow.** Pulling competing factors out of a
  hedged sentence works on this corpus' phrasing and would need real
  parsing to generalise.
- **Prefix matching is not stemming**; `failing`/`failures` do not match.
- **No `INFO`-level anomaly detection.** Hypotheses come from `WARN` and
  above — which is why incident B correctly bottoms out at "needs human
  review" instead of reaching a wrong answer confidently.

## Verification

Both reports were checked before submission: exact key set and types,
`needs_human_review == (confidence_score < 50)`, every `excerpt` a verbatim
substring of the file it cites, and evidence spanning multiple independent
documents. Robustness was checked against an empty corpus, a corpus with no
log file, logs with no corroborating documents, a malformed CSV, an empty
query, and both query/corpus cross-pairings — none raise, all stay
internally consistent, and all excerpts remain verbatim.
