# Solution Notes — Payment Investigation Assistant

Implementation notes for the reviewer. Evaluation here is manual, so this
covers the design decisions that aren't obvious from reading the diff.

## How to run

```bash
pip install -r requirements.txt          # includes the LLM SDK; no extra install step
cp .env.example .env                     # add OPENAI_API_KEY
python main.py --questions questions/questions.json --output submission.json
```

`main.py` is unmodified. Without an API key the program still completes and
answers all ten questions from the deterministic layer alone — see
"Behaviour without a key" below.

## Architecture

```
Question
   -> deterministic pre-flight        get_payment + get_client_profile, always
   -> LLM tool-calling loop           the model chooses what else it needs
   -> evidence-completeness gate      deterministic backstop
   -> deterministic policy assessment all arithmetic
   -> LLM synthesis                   prose only
   -> Python assembly                 facts / citations / tools_used
```

| Module | Responsibility |
|---|---|
| `tools/data_store.py` | CSV load, index by ID, paths resolved from `__file__` |
| `tools/client_tools.py` | Client profile plus which regional policy governs it |
| `tools/payment_tools.py` | Payment lookup, history, 24-hour aggregation |
| `tools/policy_tools.py` | `search_policy` over a once-built RAG index |
| `tools/risk_rules.py` | Thresholds, jurisdiction risk, structuring — all arithmetic |
| `rag/pipeline.py` | Load, clean, rule-level chunk, TF-IDF index, rerank |
| `agent/agent.py` | Phase orchestration and tool registry |
| `agent/evidence.py` | Ledger backing `facts`, `citations`, `tools_used` |
| `agent/llm.py` | OpenAI adapter, retry classification, fail-soft |
| `agent/prompts.py` | System and synthesis prompts |
| `agent/fallback.py` | Deterministic renderer used when no LLM is reachable |

## The central decision: the LLM writes prose, Python owns everything scored

The model plans, selects tools, interprets and writes. It never computes an
amount, compares a threshold, counts payments, decides what is high-risk, or
chooses a citation. `answer` is the only field it produces.

`facts`, `citations` and `tools_used` are built by the evidence ledger from
tools that actually executed, which makes them true by construction rather than
asserted. `tools_used` cannot drift from reality because it *is* the record of
what ran.

Two deterministic guards sit around the model. The **pre-flight** always fetches
the payment and client, because every official question needs both — so `facts`
is never empty even if the model misbehaves. The **completeness gate** re-checks
gathered evidence against what the question needs: a question about transaction
splitting that never aggregated a 24-hour window gets the aggregation run for
it. Both key off question *wording*, never question ID.

## Data traps this handles

**Structuring requires filtering on client *and* beneficiary.** C2003 paid
Northstar Trading three times on 2026-04-11 (P50003, P50180, P50181) totalling
CHF 110,000 — over the global USD 100,000 structuring threshold. That same day
C2003 also paid Desert Star LLC CHF 25,000, so filtering by client and date
alone returns CHF 135,000 and is wrong. `aggregate_beneficiary_24h` filters on
both and reports which fields it filtered on.

**The decoys defeat naive lexical retrieval.** All four read "This document
contains no payment-monitoring thresholds", so a bag-of-words query for
"payment threshold" ranks them highly — TF-IDF cannot see the negation.
Reranking therefore drops chunks that assert an absence, then requires a chunk
to state something actionable (a currency threshold, a requirement verb, a
high-risk jurisdiction rule, or a numbered procedural step). Neither filter
references a filename, so nothing is keyed to the decoys specifically.

**Thresholds are parsed from the corpus, not hardcoded.** Editing
`data/policies/*.md` changes behaviour. This also forced rule-level rather than
line-level parsing: the global policy's structuring clause wraps across three
source lines, and parsing line by line divorces "potential structuring" from
the "USD 100,000" that governs it. `risk_rules` and the RAG chunker share one
splitter so retrieval and threshold parsing see identical rule boundaries.

**Citations are restricted to policies that actually govern the client.**
Retrieval is decoy-filtered, but a stray query can still surface a regional
procedure that doesn't apply — citing Singapore's thresholds for a Swiss client
is a grounding error even though the document isn't a decoy. Allowed sources are
the client's own policy layers, plus the investigation procedure for process
questions, plus anything a structuring finding rests on.

**`beneficiary_country_code` is authoritative, and the conflict is reported.**
The code drives every risk check. The disagreement with `beneficiary_country` is
also surfaced as an observed fact, since it's something an investigator should
know about the record.

**Currency assumptions are stated, not hidden.** No exchange-rate data is
provided, so where a payment's currency differs from a threshold's the
comparison is made 1:1 and the assumption is recorded in `facts.assumptions` and
in the prose. Likewise `payment_date` has no time component, so a 24-hour window
is approximated by the calendar date.

## Retrieval

Documents are four to eight lines each, so a fixed 500-character window would
swallow whole documents and destroy citation precision. Chunks are single
logical rules with the document heading prepended, giving rule-level granularity
so a citation points at the rule that actually fired.

Query construction matters more than the index. The policy query is built from
resolved facts — the client's jurisdiction, the actual amount and currency, the
destination code — with the question text appended, since that carries the intent
terms the reranker boosts on.

## Behaviour without a key

The organizer re-runs this in a fresh environment, possibly with a different
key, a different model, or none — and crashing on any official question is a
disqualifier. So:

- `run_agent` never raises; failures degrade to a schema-valid record.
- Tool exceptions are captured as evidence rather than propagated.
- Non-retryable API errors (401/403/404/400/422) skip the retry backoff and fall
  through immediately, rather than burning three attempts on a wrong key.
- With no LLM reachable, `agent/fallback.py` renders the same five-part answer
  from the same evidence bundle.

Verified: with no key, all ten questions return grounded answers with
jurisdiction-correct citations and zero decoys.

## Known limitations

- The 24-hour window is calendar-date based, because the data has no time
  component. Real 24-hour rolling windows would need timestamps.
- Currency comparison is 1:1 across currencies. `DATA_NOTES.md` sanctions this
  and the questions are designed not to depend on precise conversion, but a
  production version needs a rate source.
- `find_repeated_beneficiaries` aggregates at most five candidate beneficiaries
  per question, which is ample for this dataset but is a cap.
- The country-name to code map used for conflict detection covers the countries
  present in `payments.csv` rather than being exhaustive.
