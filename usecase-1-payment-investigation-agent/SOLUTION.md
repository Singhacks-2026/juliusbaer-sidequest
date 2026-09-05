# Payment Investigation Assistant — solution

A single OpenAI agent selects tools, retrieves local policy evidence, and explains
its findings. Python owns all calculations and the final `facts` / `tools_used`
fields. The competition's `main.py`, input data, policies, and questions are unchanged.

## Run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r usecase-1-payment-investigation-agent/requirements.txt
cd usecase-1-payment-investigation-agent
python -m unittest -v test_solution
INVESTIGATION_TRACE=artifacts/trace.jsonl python main.py \
  --questions questions/questions.json --output submission.json
python validate_submission.py submission.json --trace artifacts/trace.jsonl
```

The existing repository-root `.env` is loaded automatically. A use-case-local
`.env`, if present, takes precedence; both override inherited shell values.
Neither file is committed or included in model prompts. The run sends selected
synthetic exercise records, policy evidence and questions to the OpenAI API.

Default model: `gpt-6-astra`, with `medium` reasoning. Override with `OPENAI_MODEL`
and `OPENAI_REASONING_EFFORT`; choose a Responses API model supporting reasoning,
function calls and structured outputs. Dependencies are pinned in `requirements.txt`.
All 10 included answers were generated in a live API run and verified against
the source evidence. See [VALIDATION.md](VALIDATION.md) for results and the
question-by-question review; offline tests also simulate the model boundary.

Submit `submission.json` and the source code. The optional trace provides the
actual tool arguments/results, model name, token usage and final answers for
review. It is append-only; the validator uses the latest matching investigation.
It does not contain the API key or private model reasoning.

## Design decisions

- **Data tools:** cached standard-library CSV reads, immutable cache access,
  decimal arithmetic, explicit missing-ID results, validated dates and amounts.
- **RAG:** load all nine documents once; preserve whole policy rules and their
  headings; index with BM25; retrieve query-relevant evidence with source and
  chunk identifiers. Semantic aliases help regional and structuring queries.
  Administrative notes without operative clauses or numbered procedure steps
  are excluded by content, rather than by a filename blacklist.
- **Policy assessment:** extract numeric rules from retrieved policy sources and
  compare them in Python. The agent cannot supply a threshold or calculated
  total. Regional rules add to global policy; they do not override it.
- **Agent:** bounded tool-calling loop using the OpenAI Responses API, strict tool
  schemas and structured final output. The agent chooses searches and history
  checks. Unretrieved sources are rejected; missing applicable policies and
  stale assessments trigger another round before an answer is accepted.
- **Grounding:** Python constructs facts from actual tool returns and records
  invoked tools. The model writes the answer and selects supporting citations;
  citations must belong to sources retrieved during that investigation.
- **Failure behavior:** bounded SDK retries; a failed investigation returns an
  explicit incomplete result, preserving any retrieved facts. It never invents
  a successful answer or aborts the entire batch. The submission validator
  rejects incomplete results.

## Exercise assumptions and limitations

`beneficiary_country_code` is authoritative for destination risk. Client `country`
determines the applicable regional procedure. Country-name/code discrepancies
are surfaced separately.

Thresholds are strictly **above** their stated values. CHF-to-CHF comparisons are
native; nonmatching currencies use the challenge's permitted **1:1 equivalence**,
explicitly labelled as an assumption. Mixed-currency native totals stay separate;
only the separately labelled assumed USD equivalent is combined.

The 24-hour proxy groups payments by **client + beneficiary + calendar date**.
All matching dates are returned. The supplied data cannot establish true rolling
windows across midnight. Multiple payments exceeding the combined threshold are
potential structuring, not proof of intent. Swiss guidance supports a Compliance
escalation recommendation; the policy supplies no deadline or document-collection
sequence. This is recorded as `compliance_escalation_recommended`. Requests for invoices, payment
purpose and similar records are recommendations, not invented policy mandates.

The lexical retrieval filter and rule parser intentionally target the supplied
small Markdown corpus. More diverse or changing policies would need broader
retrieval evaluation and a reviewed machine-readable rule format. Source checks
cannot by themselves prove that every sentence is supported: inspect the final
answers against policy text as well as running the validator.

## Local verification

`python -m unittest -v test_solution` covers exact threshold boundaries, source
policy changes, additive Swiss/global rules, authoritative risk codes, the two
structuring distractors, decimal precision, date separation, mixed currencies,
missing evidence, retrieval ranking, whole-rule chunks, evidence guards, a
simulated multi-round tool loop, and honest API failure behavior. The suite now
contains 30 tests, including replay of all 66 submitted tool calls, missing-policy
uncertainty, target-specific assessment freshness, and artifact-tampering checks.
The validator recomputes derived facts from the original records instead of
trusting submitted aggregates or a matching trace.
