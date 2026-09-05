# Payment Investigation Agent — Submission

**Submitted by: Albert Song**

This submission implements Use Case 1: an LLM agent that combines deterministic
payment/client tools with retrieval over the supplied policy documents. Answers
are generated from the official questions, without question-ID-specific branches.

## Included

- `submission.json`: ten actual model-generated answers with facts, policy citations,
  executed tool names and tool-result traces.
- `agent/`, `tools/`, `rag/`: complete implementation.
- `main.py`: unchanged organizer entry point.
- `data/`, `questions/`: unchanged organizer synthetic fixtures and official questions.
- `requirements.txt`, `.env.example`, `README.md`: installation and configuration.
- `tests/`: 17 offline regression and SDK transport tests.

## Reproduce

Python 3.9 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env.
python main.py --questions questions/questions.json --output submission.json
python -m unittest discover -s tests -v
```

Model: `gpt-5.6-luna`, using Responses API with the model's default reasoning setting.
GPT-5.6/GPT-6 select Responses automatically. Other models default to Chat Completions;
`OPENAI_API_MODE` can override that selection. No API credentials are included.

## Implementation details

Policy documents are cleaned, chunked at rule boundaries and indexed once with BM25.
Tools retrieve payment facts and client profiles, inspect complete client histories,
and group transactions by both client and beneficiary within date windows. A policy
assessment tool extracts thresholds from retrieved passages and applies exact Decimal
calculations. The LLM selects tools and synthesizes the final cited recommendation.
Facts and tool names in the output are built from execution, not trusted from model JSON.

Global policy always applies; regional policy is selected ONLY by client country.
Beneficiary country codes are used separately for destination risk. Thresholds use
strict `>` comparisons. No FX rates or timestamps are supplied: 1:1 equivalent conversion
and same-calendar-date windows are explicit exercise assumptions. A structuring trigger
is a review concern, not proof of intent or suspicious activity.

## Validation and limits

Tests cover deterministic boundaries, aggregation exclusions, mixed currencies,
retrieval, citation provenance and both API transports. The official questions are
also run through the actual model to produce `submission.json`. Answers can vary on
rerun; schema/citation provenance validation does not guarantee semantic correctness.
Threshold parsing targets the supplied policy wording. No organizer private evaluator
or official score is available.
