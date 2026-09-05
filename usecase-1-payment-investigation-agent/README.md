# Julius Baer AI Hackathon — Payment Investigation Assistant

**Local implementation:** The tools, RAG pipeline, and configurable LLM agent
are implemented in this checkout. See [IMPLEMENTATION.md](IMPLEMENTATION.md)
for Windows commands, provider configuration, tests, and submission validation.

Build a small AI-powered assistant for a bank's payment operations and
compliance team.  The assistant answers natural-language
payment-investigation questions by combining structured data, policy
documents (via RAG), deterministic tools, and an LLM agent.

## Documentation

Follow the reading order in `PARTICIPANT_INSTRUCTIONS.md` → "Before you
start".  All documents:

| File                              | Purpose                          |
|-----------------------------------|----------------------------------|
| `PROBLEM_STATEMENT.md`            | What you are building            |
| `PARTICIPANT_INSTRUCTIONS.md`     | Your three tasks and schedule    |
| `DATA_NOTES.md`                   | Important data clarifications    |
| `AI_ARCHITECTURE_REQUIREMENTS.md` | Required components              |
| `EVALUATION_CRITERIA.md`          | How submissions are scored       |
| `SUBMISSION_GUIDE.md`             | Required output format           |
| `ARCHITECTURE_HINTS.md`           | Architecture guidance (optional) |
| `WHY_METHODS_ONLY.md`             | Why interfaces are empty         |

## Directory structure

```text
usecase-1-payment-investigation-agent/
├── main.py                  # Entry point — run this
├── requirements.txt         # Python dependencies
├── .env.example             # LLM configuration template
├── data/
│   ├── clients.csv          # 50 synthetic clients
│   ├── payments.csv         # 184 synthetic payments
│   ├── data_dictionary.csv  # Field descriptions
│   └── policies/            # 9 policy docs (5 relevant, 4 decoys)
├── questions/
│   └── questions.json       # 10 evaluation questions
├── tools/                   # Data access and deterministic policy checks
│   ├── client_tools.py
│   ├── payment_tools.py
│   └── policy_tools.py
├── rag/                     # Local policy retrieval pipeline
│   └── pipeline.py
└── agent/                   # Configurable LLM agent and provider adapters
    └── agent.py
```

## Quick start

```powershell
# From the usecase-1-payment-investigation-agent folder:
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env to select your provider, model, and API key.
.\.venv\Scripts\python.exe check_setup.py
.\.venv\Scripts\python.exe main.py --questions .\questions\questions.json --output submission.json
```

## What you must implement

The original starter provided method-only interfaces. Their implementations
are now present in this checkout:

1. **Tools** — `tools/*.py` — deterministic data access (CSV lookups,
   aggregation, 24h window analysis)
2. **RAG** — `rag/pipeline.py` — policy document retrieval (load, chunk,
   index, retrieve)
3. **Agent** — `agent/agent.py` — LLM/tool-calling loop that decides
   which tools to call and synthesizes a grounded answer
