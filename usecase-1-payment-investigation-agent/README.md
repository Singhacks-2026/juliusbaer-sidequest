# Payment Investigation Assistant

**Submitted by: Albert Song**

Implemented payment investigation agent with CSV tools, local BM25 policy retrieval,
deterministic policy checks, and an OpenAI-compatible function-calling loop.
`main.py` is the unchanged competition entry point.

## Run

Python 3.9+:

```bash
cd usecase-1-payment-investigation-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env. Adjust OPENAI_MODEL and optional OPENAI_BASE_URL.
python main.py --questions questions/questions.json --output submission.json
```

The actual supplied question path is `questions/questions.json`, not the
`questions.json` shorthand used in some organizer documents. Data and `.env`
paths resolve relative to this use case, independent of the current directory.
GPT-5.6 and GPT-6 models automatically use Responses API to support reasoning with
function tools; other models default to Chat Completions. Set OPENAI_API_MODE to
responses or chat to override. The model must support function calling and JSON mode. Configuration
uses the OpenAI SDK and also accepts an OpenAI-compatible endpoint. `.env` does not
override existing environment variables. The example uses `gpt-5.6-luna`, the model used for this submission.
Set an accessible compatible model for your account.

No API key is needed for offline tests. Real answer generation requires a model;
there is no hard-coded or rule-only answer fallback. Missing configuration, provider
failures and exhausted loops produce explicit `error` results, so the batch still
writes one record per question. Such results are incomplete investigations, not
valid substantive submission answers. Check the output for `error` before submitting.

## Architecture

- `tools/data_store.py`: CSV access relative to the project directory.
- `tools/payment_tools.py`: payment lookup, client history, beneficiary/date windows
  and Decimal aggregation. Windows retain payment IDs and native-currency totals.
- `tools/client_tools.py`: profiles and country queries.
- `rag/pipeline.py`: clean documents, preserve complete policy rules in chunks,
  build BM25 index and retrieve scored source passages. Empty/unmatched queries
  return no evidence; administrative decoys are excluded.
- `tools/policy_tools.py`: reuse one cached index and safely retrieve known sources.
- `tools/investigation_tools.py`: retrieve applicable global/regional/risk passages,
  extract supplied thresholds, compute strict comparisons and optional full-history
  structuring analysis. Nested tool invocations are reported for auditing.
- `agent/agent.py`: model chooses tools, receives results, makes further calls and
  synthesizes a cited answer. Tool arguments, final shape and citation provenance
  are validated; malformed output can be repaired within a bounded loop.

The returned `facts` and `tools_used` are assembled from actual tool execution,
rather than accepted from model-generated JSON. `tool_trace` adds the invoked
arguments and returned evidence to the required submission fields. Citation checks
validate source provenance; semantic faithfulness of the answer still depends on
the model and must be evaluated with a live run.

## Data assumptions

- Client `country` selects regional policy. Global policy always applies too.
- Jurisdiction risk uses `beneficiary_country_code`, even if the name disagrees.
- Policy comparisons are strictly **above** the threshold; equality does not trigger.
- Native matching currencies are compared directly. Missing FX conversions use the
  exercise's permitted **1:1 equivalence**, explicitly recorded as an assumption.
- Date-only inputs use same-calendar-date windows per `DATA_NOTES.md`. Adjacent dates
  are not combined: exact cross-midnight 24-hour windows cannot be established.
- Aggregation filters both client and beneficiary, keeps currencies separate and
  reports excluded incomplete/invalid records. Mixed currencies have a separately
  labelled USD-equivalent total using the same explicit 1:1 assumption.
- A structuring trigger is a review signal, not evidence of intent or criminality.

Threshold extraction intentionally targets the wording of the supplied policy
corpus. Changed policy formats require updating the parser and regression tests.

## Verify

```bash
python -m unittest discover -s tests -v
```

Tests cover supplied facts, exact boundary comparisons, the CHF 110,000 three-payment
pattern and cross-client/beneficiary decoys, date windows, mixed currencies, missing
values, policy retrieval, path safety, tool execution and error recovery. A simulated
model exercises all 10 official questions through the unchanged `main.py`. These
checks validate plumbing and deterministic evidence, **not live LLM answer quality**.

Organizer requirements and evaluation guidance remain in `PARTICIPANT_INSTRUCTIONS.md`,
`DATA_NOTES.md`, `AI_ARCHITECTURE_REQUIREMENTS.md`, `EVALUATION_CRITERIA.md` and
`SUBMISSION_GUIDE.md`.

The integration follows the official
[OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).
