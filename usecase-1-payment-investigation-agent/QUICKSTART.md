# Quick Start

The implementation is present in this checkout. See
[IMPLEMENTATION.md](IMPLEMENTATION.md) for provider options and verification.

## Prerequisites

- Python 3.10 or later
- An LLM API key (any provider — see `.env.example`)

## Setup

```powershell
# Windows: run from the usecase-1-payment-investigation-agent directory.
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env and add your provider, API key, and model name.
.\.venv\Scripts\python.exe check_setup.py
```

On Linux/macOS, activate .venv/bin/activate and use python in place of
.\.venv\Scripts\python.exe. The required provider SDKs are already listed in
requirements.txt.

## Understand the problem

Follow the reading order in `PARTICIPANT_INSTRUCTIONS.md` → "Before you
start".  The key documents are:

- `PROBLEM_STATEMENT.md` — what you are building
- `DATA_NOTES.md` — data clarifications (read this before coding)
- `AI_ARCHITECTURE_REQUIREMENTS.md` — required components
- `EVALUATION_CRITERIA.md` — how you are scored
- `SUBMISSION_GUIDE.md` — required output format

Inspect the data:
```text
data/clients.csv          — 50 synthetic clients
data/payments.csv         — 184 synthetic payments
data/policies/            — 9 policy documents (5 relevant, 4 decoys)
data/data_dictionary.csv  — field descriptions
questions/questions.json  — 10 evaluation questions
```

## Implementation

```text
tools/client_tools.py     — client data access
tools/payment_tools.py    — payment lookup + deterministic analysis
tools/policy_tools.py     — policy retrieval (connects to RAG)
rag/pipeline.py           — document loading, chunking, indexing, retrieval
agent/agent.py            — LLM/tool-calling agent loop
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py --questions .\questions\questions.json --output submission.json
```

Your program must run without interactive input and produce one result for each
question.

## Verify before submitting

- The output file contains exactly 10 JSON objects (one per question).
- Each object has all required fields: `question_id`, `payment_id`,
  `answer`, `citations`, `facts`, `tools_used`.
- No answer is hard-coded to a `question_id`.
- The program runs in a fresh environment with `pip install -r requirements.txt`
  and the selected provider configured in .env.

```powershell
.\.venv\Scripts\python.exe validate_submission.py --submission submission.json --traces
```
