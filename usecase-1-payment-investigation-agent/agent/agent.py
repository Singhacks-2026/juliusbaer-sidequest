"""LLM orchestration for grounded payment-investigation answers."""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    evaluate_payment_controls,
    get_client_payments,
    get_payment,
)
from tools.policy_tools import search_policy


TOOLS: dict[str, Callable[..., Any]] = {
    "get_client_profile": get_client_profile,
    "get_payment": get_payment,
    "get_client_payments": get_client_payments,
    "aggregate_beneficiary_24h": aggregate_beneficiary_24h,
    "evaluate_payment_controls": evaluate_payment_controls,
    "search_policy": search_policy,
}

SYSTEM_PROMPT = """
You are a careful bank payment-investigation assistant. Use the supplied
functions to gather evidence before reaching any conclusion.

Non-negotiable rules:
1. Retrieve transaction and client facts before making factual claims.
2. Use evaluate_payment_controls and aggregation results for calculations;
   never perform or invent exact arithmetic yourself.
3. Use beneficiary_country_code, not beneficiary_country, for jurisdiction risk.
4. Global policy always applies. Singapore and Switzerland procedures add to it.
   Regional policy is selected ONLY from the client's country, never from the
   beneficiary country or beneficiary_country_code.
5. For structuring, inspect client history and the same client's payments to the
   same beneficiary on the same date. The data has dates but no timestamps.
6. A trigger is not proof of suspicious intent. Clearly distinguish observed
   facts, policy triggers, assumptions, missing evidence, and next action.
7. Cite only source filenames returned by search_policy. Never cite decoy files.
8. If evidence is insufficient, say exactly what is missing.
9. Do not recommend release while a triggered review remains incomplete.
10. Use exact policy terminology. "Additional review", "RM review", "enhanced
    review", and "escalate to Compliance" are distinct; never rename or merge them.
11. Do not invent a review requirement. If no supplied policy trigger applies,
    recommend standard processing/monitoring without holding for an irrelevant
    regional policy.
12. Never claim a payment-history or structuring check was completed unless the
    corresponding history/aggregation tool evidence is present. For a workflow
    question, describe the required process instead of inventing completed checks.
13. In prose citations use exact filenames such as global_payment_policy.md,
    never chunk IDs such as global_payment_policy.md#0 or tool names.

When evidence collection is complete, return one JSON object only:
{"answer": "substantive grounded narrative", "citations": ["source.md"]}
The narrative should directly answer the question and include material facts,
policy requirements, uncertainty/assumptions, and a practical recommendation.
""".strip()

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_payment",
            "description": "Retrieve authoritative facts for one payment ID.",
            "parameters": {
                "type": "object",
                "properties": {"payment_id": {"type": "string"}},
                "required": ["payment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_profile",
            "description": "Retrieve client country, risk rating, type, and relationship duration.",
            "parameters": {
                "type": "object",
                "properties": {"client_id": {"type": "string"}},
                "required": ["client_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_payments",
            "description": "Retrieve full payment history for pattern analysis.",
            "parameters": {
                "type": "object",
                "properties": {"client_id": {"type": "string"}},
                "required": ["client_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_beneficiary_24h",
            "description": "Deterministically aggregate one client's payments to one beneficiary in same-date 24h windows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                    "beneficiary_name": {"type": "string"},
                },
                "required": ["client_id", "beneficiary_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_payment_controls",
            "description": "Deterministically compare a payment with parsed global/regional thresholds and destination controls.",
            "parameters": {
                "type": "object",
                "properties": {"payment_id": {"type": "string"}},
                "required": ["payment_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Retrieve relevant policy passages and exact source filenames through RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 7},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def _load_dotenv() -> None:
    """Load a local .env without adding another runtime dependency."""
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _client() -> Any:
    _load_dotenv()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add the key.")
    options = {}
    if os.environ.get("OPENAI_BASE_URL"):
        options["base_url"] = os.environ["OPENAI_BASE_URL"]
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], **options)


def _completion(client: Any, **kwargs: Any) -> Any:
    """Create a completion with bounded retry for free-tier rate limits."""
    for attempt in range(8):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            if type(exc).__name__ != "RateLimitError" or attempt == 7:
                raise
            match = re.search(r"try again in\s+([\d.]+)s", str(exc), re.IGNORECASE)
            delay = float(match.group(1)) + 0.75 if match else min(2 ** attempt, 30)
            time.sleep(min(delay, 45))
    raise RuntimeError("Unreachable completion retry state")


def _execute(name: str, arguments: dict, evidence: dict, tools_used: list[str]) -> Any:
    """Safely invoke a registered tool and record the exact result."""
    function = TOOLS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}
    allowed = set(inspect.signature(function).parameters)
    clean_arguments = {key: value for key, value in arguments.items() if key in allowed}
    try:
        result = function(**clean_arguments)
    except Exception as exc:  # return a usable tool error instead of crashing the loop
        result = {"error": f"{type(exc).__name__}: {exc}"}
    tools_used.append(name)
    evidence.setdefault(name, []).append(result)
    return result


def _assistant_message(message: Any) -> dict:
    payload: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return payload


def _ensure_evidence(question: str, payment_id: str, evidence: dict, tools_used: list[str]) -> None:
    """Fill evidence gaps using general domain rules, never question IDs."""
    if "get_payment" not in evidence:
        _execute("get_payment", {"payment_id": payment_id}, evidence, tools_used)
    payment = evidence["get_payment"][0] if evidence["get_payment"] else {}
    if not isinstance(payment, dict) or not payment:
        return

    client_id = str(payment["client_id"])
    if "get_client_profile" not in evidence:
        _execute("get_client_profile", {"client_id": client_id}, evidence, tools_used)
    if "evaluate_payment_controls" not in evidence:
        _execute("evaluate_payment_controls", {"payment_id": payment_id}, evidence, tools_used)

    structuring_terms = ("structur", "splitting", "split", "multiple payment", "payment history")
    if any(term in question.casefold() for term in structuring_terms):
        if "get_client_payments" not in evidence:
            _execute("get_client_payments", {"client_id": client_id}, evidence, tools_used)
        if "aggregate_beneficiary_24h" not in evidence:
            _execute(
                "aggregate_beneficiary_24h",
                {"client_id": client_id, "beneficiary_name": payment["beneficiary_name"]},
                evidence,
                tools_used,
            )

    controls = evidence.get("evaluate_payment_controls", [{}])[0] or {}
    retrieved_sources = {
        hit.get("source")
        for result_set in evidence.get("search_policy", [])
        if isinstance(result_set, list)
        for hit in result_set
        if isinstance(hit, dict)
    }
    expected_sources = set(controls.get("applicable_policy_sources", []))
    if any(term in question.casefold() for term in ("structur", "split")):
        expected_sources.add("global_payment_policy.md")
    if "workflow" in question.casefold():
        expected_sources.add("investigation_procedure.md")
    if not evidence.get("search_policy") or not expected_sources.issubset(retrieved_sources):
        client_profile = evidence.get("get_client_profile", [{}])[0]
        policy_query = " ".join(
            [
                question,
                f"payment amount {payment.get('amount')} {payment.get('currency')}",
                f"client region {client_profile.get('country', '')}",
                f"destination code {payment.get('beneficiary_country_code', '')}",
                "global regional threshold high-risk structuring investigation procedure",
            ]
        )
        _execute("search_policy", {"query": policy_query, "top_k": 5}, evidence, tools_used)


def _parse_json(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return {"answer": content, "citations": []}
        parsed = json.loads(content[start : end + 1])
    return parsed if isinstance(parsed, dict) else {"answer": str(parsed), "citations": []}


def _facts(evidence: dict) -> dict:
    """Build the scored facts object only from deterministic tool results."""
    payment = evidence.get("get_payment", [{}])[0] or {}
    client = evidence.get("get_client_profile", [{}])[0] or {}
    controls = evidence.get("evaluate_payment_controls", [{}])[0] or {}
    facts = {
        key: payment[key]
        for key in (
            "amount", "currency", "beneficiary_country_code", "beneficiary_name",
            "client_id", "channel", "payment_date",
        )
        if key in payment
    }
    for key in ("country", "risk_rating", "client_type", "relationship_years"):
        if key in client:
            facts[f"client_{key}"] = client[key]
    for key in ("high_risk_destination", "threshold_evaluations", "triggered_requirements"):
        if key in controls:
            facts[key] = controls[key]
    aggregate = evidence.get("aggregate_beneficiary_24h", [])
    if aggregate:
        value = aggregate[0]
        facts["24h_aggregation"] = {
            key: value[key]
            for key in (
                "beneficiary_name", "payment_date", "payment_ids", "count",
                "total_amount", "currency", "date_assumption", "structuring_threshold",
                "exceeds_structuring_threshold", "threshold_currency_basis",
            )
            if key in value
        }
        facts["24h_aggregation"]["individual_amounts"] = [
            row["amount"] for row in value.get("payments", [])
        ]
        facts["24h_aggregation"]["channels"] = [
            row["channel"] for row in value.get("payments", [])
        ]
    return facts


def _allowed_policy_sources(question: str, evidence: dict) -> set[str]:
    """Limit synthesis to policies applicable to the payment and question."""
    controls = evidence.get("evaluate_payment_controls", [{}])[0] or {}
    allowed = set(controls.get("applicable_policy_sources", []))
    lowered = question.casefold()
    if any(
        term in lowered
        for term in ("workflow", "additional information", "facts", "assumptions", "release")
    ):
        allowed.add("investigation_procedure.md")
    return allowed


def run_agent(question: str, payment_id: str) -> dict:
    """Run tool planning, enforce evidence completeness, and synthesize JSON."""
    client = _client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Investigate payment_id={payment_id}. Question: {question}",
        },
    ]
    evidence: dict[str, list[Any]] = {}
    tools_used: list[str] = []

    # One model-planned collection turn keeps broad OpenAI-compatible model
    # support. The deterministic guardrail below fills gaps before synthesis.
    response = _completion(
        client,
        model=model,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="required",
    )
    message = response.choices[0].message
    messages.append(_assistant_message(message))
    for call in message.tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        result = _execute(call.function.name, arguments, evidence, tools_used)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    _ensure_evidence(question, payment_id, evidence, tools_used)
    allowed_sources = _allowed_policy_sources(question, evidence)
    synthesis_evidence = dict(evidence)
    synthesis_evidence["search_policy"] = [
        [hit for hit in result_set if hit.get("source") in allowed_sources]
        for result_set in evidence.get("search_policy", [])
        if isinstance(result_set, list)
    ]
    synthesis_payload = json.dumps(synthesis_evidence, ensure_ascii=False)
    final_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\nPayment ID: {payment_id}\n"
                "Authoritative tool evidence follows. Synthesize the final JSON now. "
                "Do not omit relevant regional requirements or uncertainty.\n"
                f"{synthesis_payload}"
            ),
        },
    ]
    try:
        final_response = _completion(
            client,
            model=model,
            messages=final_messages,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        # Some OpenAI-compatible providers implement tools but not JSON mode.
        # Retry only for a clear unsupported-parameter response; authentication,
        # quota, and network failures must remain visible to the participant.
        detail = str(exc).casefold()
        if not any(term in detail for term in ("response_format", "json mode", "unsupported")):
            raise
        final_response = _completion(client, model=model, messages=final_messages)
    parsed = _parse_json(final_response.choices[0].message.content or "")
    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        raise RuntimeError("The LLM returned an empty answer")

    policy_results = evidence.get("search_policy", [])
    available_sources = []
    for result_set in policy_results:
        if isinstance(result_set, list):
            for result in result_set:
                source = result.get("source") if isinstance(result, dict) else None
                if source and source not in available_sources and not source.startswith("decoy_"):
                    available_sources.append(source)
    requested = parsed.get("citations", [])
    citations = [
        source for source in requested
        if source in available_sources and source in allowed_sources
    ]
    controls = evidence.get("evaluate_payment_controls", [{}])[0] or {}
    preferred = list(controls.get("applicable_policy_sources", []))
    if any(term in question.casefold() for term in ("structur", "split")):
        preferred.extend(["global_payment_policy.md", "regional_switzerland.md"])
    if "workflow" in question.casefold():
        preferred.append("investigation_procedure.md")
    for source in preferred:
        if source in available_sources and source not in citations:
            citations.append(source)

    return {
        "answer": answer,
        "citations": citations,
        "facts": _facts(evidence),
        "tools_used": list(dict.fromkeys(tools_used)),
    }
