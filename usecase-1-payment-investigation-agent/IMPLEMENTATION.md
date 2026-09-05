# Running the implemented Side Quest 1 agent

The data tools, policy retrieval, deterministic policy checks, and LLM loop are
implemented. The competition's main.py is unchanged. Source CSVs, policy files,
official questions, and sample answers are unchanged.

## Verification status

All 29 offline tests pass, including actual SDK serialization with simulated
HTTP responses, evidence retention across tool calls, and the unchanged
main.py processing all ten official questions with a simulated LLM. Dependency
checks also pass. These tests verify the implementation and output plumbing;
they do not establish live model answer quality.

A live ten-question output is now available in submission_2.json, generated
using the configured OpenAI gpt-4o model and validated against source facts
and execution traces. The run used LLM_MAX_ROUNDS=24. Three answers were
regenerated through the same agent after prose review and merged by their
unchanged official question IDs. All final answers match their traces in
artifacts/submission_2_traces. This does not certify a private evaluation score.

To validate this particular output in PowerShell:

```powershell
$env:LLM_TRACE_DIR = 'artifacts/submission_2_traces'
.\.venv\Scripts\python.exe validate_submission.py --submission submission_2.json --traces
```

## Start in PowerShell

Run these commands from the usecase-1-payment-investigation-agent directory.
Use the virtual environment's Python explicitly to avoid installing packages
into one Python and then running a different Python.

```powershell
# Only needed if the virtual environment does not exist:
python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create .env from .env.example if it does not exist; otherwise edit the existing
.env. Select a provider and enter its API key and model name. The application
loads this file automatically, relative to the project directory. Existing
environment variables take precedence. Keys are not written to tool traces.
Live runs send the question, retrieved synthetic client/payment records,
policy excerpts, and tool results to the selected LLM provider.

```powershell
# Local configuration/data checks; makes no API requests:
.\.venv\Scripts\python.exe check_setup.py

# Optional: one real question to verify the LLM connection and tool calling:
.\.venv\Scripts\python.exe check_setup.py --live

# The actual competition entry point:
.\.venv\Scripts\python.exe main.py --questions .\questions\questions.json --output submission.json

# Check every answer and compare it with its actual execution trace:
.\.venv\Scripts\python.exe validate_submission.py --submission submission.json --traces

# Offline regression tests; no API calls or credentials needed:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Several original challenge documents abbreviate the questions path as
questions.json. In this checkout, the file is questions/questions.json.

## Provider configuration

| Provider | Required variables | API |
|---|---|---|
| OpenAI | OPENAI_API_KEY, OPENAI_MODEL | responses by default |
| OpenAI-compatible service, including Gemini or a local model | OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL | chat_completions by default |
| Anthropic | ANTHROPIC_API_KEY, ANTHROPIC_MODEL | messages |
| Azure OpenAI | AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION | chat_completions by default |

LLM_PROVIDER=auto selects the sole provider with a configured key. If more than
one provider is configured, explicitly set LLM_PROVIDER to openai, anthropic,
or azure. LLM_API can override the OpenAI API choice. A model must support the
chosen API and function calling; an API-compatible endpoint alone does not
guarantee that every model or feature is supported.

Native OpenAI Responses requests use a strict JSON schema for the final answer
and citations to prevent repeated malformed-JSON failures. The selected model
must support structured outputs as well as function calling. Custom endpoints
keep their existing request format.

LLM_MAX_ROUNDS bounds the loop; LLM_MAX_OUTPUT_TOKENS bounds each response;
LLM_TIMEOUT_SECONDS bounds each request. The SDK retries a transient request
up to twice. For Chat Completions, LLM_CHAT_TOKEN_FIELD can select max_tokens or
max_completion_tokens to match the endpoint. Full assistant tool-call messages
are preserved, including provider metadata such as Gemini thought signatures.
LLM_MIN_REQUEST_INTERVAL_SECONDS optionally spaces model requests (0-60 seconds;
default 0) for providers with low request-rate limits. It does not increase quotas.

Provider implementations follow the official [OpenAI function calling guide](https://developers.openai.com/api/docs/guides/function-calling),
[Anthropic tool-result format](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls), and
[Gemini OpenAI compatibility guide](https://ai.google.dev/gemini-api/docs/openai).

## How the implementation works

1. main.py passes each question and payment ID to run_agent().
2. The LLM chooses among payment lookup, client lookup, payment history, daily
   aggregation, policy search, and deterministic evaluation tools.
3. CSV tools return original records. Amounts are summed with Decimal, and
   daily aggregates match both client and beneficiary. Raw currency totals are
   separate; payments from different dates are never added into one window.
4. RAG loads all nine policies once, preserves complete rules and source names,
   indexes TF-IDF vectors, and ranks passages by cosine similarity. Simple word
   normalization supports terms such as splitting/structuring and workflow/
   procedure. A document disclaimer saying it contains no thresholds is
   excluded as policy evidence; decoy filenames are not hard-coded.
5. evaluate_payment uses all policies discovered so far and retains completed
   structuring checks when a later call focuses on another issue. It parses
   numeric thresholds from the discovered policy files,
   applies strict greater-than comparisons, layers global and regional rules,
   and assesses the authoritative destination code. For pattern questions, it
   checks all same-client/same-beneficiary daily groups in the supplied history.
6. The LLM writes the explanation and selects citations. The application
   attaches facts from the deterministic assessment and tools_used from the
   actual execution log. Unsupported negative structuring claims when history
   has not been checked, invented citations, and premature answers are rejected
   and sent back for correction within the bounded loop.
7. main.py attaches the original IDs and question text and writes submission.json.

No runtime code reads sample_submission.json or branches on question IDs.
The tools also implement the three optional lookup helpers; reranking keeps
the existing deterministic ranking.

## Evidence and assumptions

- Region follows the client country, not beneficiary country or payment currency.
- Destination risk uses beneficiary_country_code even when the name conflicts.
- Same-calendar-date payments are the exercise's 24-hour-window approximation.
- Native-currency thresholds are compared directly. For cross-currency threshold
  checks, the exercise-permitted 1:1 assumption is explicit in the result.
- Mixed currencies remain visible in totals_by_currency. They are combined only
  for an explicitly labelled 1:1 exercise comparison.
- Global policy continues to apply alongside the regional policy.
- Potential structuring does not prove intent. Evidence requests are
  recommendations unless expressly required by a retrieved policy.

The policy parser supports the wording of this supplied corpus. It is not a
general parser for arbitrary bank policies. Claims in
the free-text answer still require review: the program verifies fact fields,
source provenance, and required tool evidence, but it does not prove every
sentence written by an LLM. No live exchange rates or real regulatory lists
are used.

## Submission and diagnostics

Submit the implementation source and the genuinely generated submission.json.
The sample file is a format illustration, not generated output. The validator
checks shape, official IDs, source facts, and optionally exact trace agreement;
it does not reproduce the organiser's private answer key or certify a score.

Tool traces go to artifacts/traces/ by default (git-ignored). They contain
synthetic source evidence, model-requested arguments, effective tool
arguments/results (including accumulated evidence), and the final answer.
They also record final-answer validation errors encountered during correction.
Each filename is a hash of the payment ID and question. Re-running a question
replaces its trace, so validate a submission against the traces from that run.
Set LLM_TRACE_DIR to an empty value to disable traces, and then omit --traces
when validating.

Missing configuration, authentication errors, provider outages, and exhausted
agent loops stop the run with a readable message. They are not silently
replaced with fabricated answers. If a run fails, an older submission.json may
still exist; do not mistake it for a successful new run. HTTP 503 indicates a
provider service failure; retry after the service recovers.
