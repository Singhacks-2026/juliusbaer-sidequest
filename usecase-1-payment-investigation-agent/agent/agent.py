"""
AI AGENT — OpenAI tool-calling investigation loop.

Shows its working: lookup payment → client/region → policy RAG →
deterministic threshold tool → grounded answer with citations.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    evaluate_review_requirements,
    get_client_payments,
    get_payment,
)
from tools.policy_tools import search_policy


TOOLS = {
    "get_client_profile": get_client_profile,
    "get_payment": get_payment,
    "get_client_payments": get_client_payments,
    "aggregate_beneficiary_24h": aggregate_beneficiary_24h,
    "evaluate_review_requirements": evaluate_review_requirements,
    "search_policy": search_policy,
}


SYSTEM_PROMPT = """
You are a bank payment-investigation assistant for compliance officers.
You must show your working with real tool lookups — never guess.

Required investigation pattern:
1) Look up the payment (amount, corridor/code, counterparty).
2) Check the client profile/region (facts the policy turns on).
3) Call evaluate_review_requirements for deterministic threshold arithmetic.
4) Retrieve the applicable policy clause via search_policy (avoid decoys).
5) Answer with citations; every claim must be traceable to tools/policies.

Rules:
1. Always call get_payment, get_client_profile, and evaluate_review_requirements
   for the payment under investigation before concluding.
2. Call search_policy for the policies named in evaluate_review_requirements
   / applicable to the question (thresholds, high-risk AE, structuring,
   investigation workflow). Never cite decoy_operational_*.md.
3. Use aggregate_beneficiary_24h with payment_date from the payment when
   discussing structuring; pass payment_date so you do not mix other days.
4. Jurisdiction risk uses beneficiary_country_code ONLY.
   AE = high-risk → additional review; cite high_risk_jurisdictions.md.
   If beneficiary_country and code disagree, report BOTH; code wins for risk.
5. Regional policy by CLIENT country only:
   Singapore → regional_singapore.md + global_payment_policy.md
   Switzerland → regional_switzerland.md + global_payment_policy.md
   Other → global_payment_policy.md only
6. RM/enhanced thresholds apply to EACH individual payment.
   Combined same-day totals apply ONLY to the global structuring rule
   (> USD 100,000 equivalent, 1:1 if non-USD). Pattern ≠ intent.
7. Switzerland: potential structuring → escalate to Compliance
   (regional_switzerland.md).
8. Separate facts vs assumptions explicitly when asked.
9. For workflow questions, list the steps in investigation_procedure.md.
10. For "which policy documents" questions, list the applicable filenames
    clearly (and cite them).
11. For region-threshold questions, lead with the regional thresholds that
    apply to the client's country.
12. Do not hard-code answers by question id.

When finished, respond with ONLY a JSON object (no markdown fences):
{
  "answer": "grounded investigation answer",
  "citations": ["policy_filename.md", ...],
  "facts": { ... key deterministic values ... },
  "tools_used": ["get_payment", ...]
}
""".strip()


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_payment",
            "description": "Retrieve one payment record by payment_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_id": {"type": "string"},
                },
                "required": ["payment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_profile",
            "description": (
                "Retrieve client profile. Client country determines regional policy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                },
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_payments",
            "description": "Retrieve full payment history for a client.",
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                },
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_beneficiary_24h",
            "description": (
                "Aggregate same-calendar-date payments from one client to one "
                "beneficiary. ALWAYS pass payment_date when evaluating a "
                "specific payment's structuring window."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                    "beneficiary_name": {"type": "string"},
                    "payment_date": {
                        "type": "string",
                        "description": "YYYY-MM-DD; scopes the summary to that day",
                    },
                },
                "required": ["client_id", "beneficiary_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_review_requirements",
            "description": (
                "Deterministic comparison of a payment against regional/global "
                "thresholds, high-risk destination (AE), and same-day structuring. "
                "Use this instead of mental arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payment_id": {"type": "string"},
                },
                "required": ["payment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "RAG over policy corpus. Query for the clause that applies; "
                "ignore decoys."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    fn = TOOLS.get(name)
    if fn is None:
        return {"error": f"unknown_tool:{name}"}
    try:
        return fn(**arguments)
    except TypeError as exc:
        return {"error": "invalid_arguments", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"error": "tool_failed", "message": str(exc)}


def _parse_json_object(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _enrich_answer_from_eval(answer: str, eval_result: dict | None, question: str) -> str:
    """Append missing grounded clauses from deterministic tool output."""
    if not eval_result or eval_result.get("error"):
        return answer

    extras: list[str] = []
    q = question.lower()
    a = answer.lower()

    name = str(eval_result.get("beneficiary_country") or "")
    code = str(eval_result.get("beneficiary_country_code") or "")
    note = eval_result.get("country_code_mismatch_note")
    if note and name and code:
        if name.casefold() not in a or code.casefold() not in a:
            extras.append(
                f"Note: beneficiary_country is '{name}' while "
                f"beneficiary_country_code is '{code}'; the code is authoritative "
                f"for jurisdiction risk."
            )

    client_country = eval_result.get("client_country")
    if client_country and str(client_country).casefold() not in a:
        if "region" in q or "enhanced" in q or "threshold" in q or "review" in q:
            extras.append(
                f"Client country/region is {client_country} "
                f"(risk rating: {eval_result.get('client_risk_rating')})."
            )

    if eval_result.get("possible_structuring"):
        window = eval_result.get("same_day_beneficiary_window") or {}
        ids = window.get("payment_ids") or []
        amounts = window.get("individual_amounts") or []
        channels = window.get("channels") or []
        if ids and not all(str(i).lower() in a for i in ids):
            parts = []
            for i, pid in enumerate(ids):
                amt = amounts[i] if i < len(amounts) else "?"
                ch = channels[i] if i < len(channels) else "?"
                parts.append(f"{pid} ({amt} {eval_result.get('currency')}, {ch})")
            extras.append(
                "Same-day payments: " + "; ".join(parts) + "."
            )
        if "intent" not in a and "suspicious activity" not in a:
            extras.append(
                "This is an observed pattern / policy trigger only; it does not by "
                "itself establish suspicious activity or intent."
            )
        if (
            eval_result.get("client_country") == "Switzerland"
            and "compliance" not in a
        ):
            extras.append(
                "Per regional_switzerland.md, potential structuring should be "
                "escalated to Compliance."
            )

    if not extras:
        return answer
    return answer.rstrip() + " " + " ".join(extras)


def _normalize_result(
    payload: dict | None,
    payment_id: str,
    tools_used: list[str],
    gathered_facts: dict,
    citations: list[str],
    eval_result: dict | None = None,
    question: str = "",
) -> dict:
    payload = payload or {}
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        answer = (
            "Unable to produce a fully grounded answer from available tool "
            "results. Request human review."
        )
    answer = _enrich_answer_from_eval(answer, eval_result, question)

    cites = payload.get("citations")
    if not isinstance(cites, list):
        cites = list(citations)
    cites = [str(c) for c in cites if c and not str(c).startswith("decoy_")]
    if not cites:
        cites = [c for c in citations if not str(c).startswith("decoy_")]

    facts = payload.get("facts")
    if not isinstance(facts, dict):
        facts = dict(gathered_facts)
    else:
        merged = dict(gathered_facts)
        merged.update(facts)
        facts = merged
    if "payment_id" not in facts:
        facts["payment_id"] = payment_id

    code = str(facts.get("beneficiary_country_code") or "").upper()
    if code == "AE" and "high_risk_jurisdictions.md" not in cites:
        cites.append("high_risk_jurisdictions.md")
    if (
        "threshold" in answer.lower()
        or "enhanced" in answer.lower()
        or "structuring" in answer.lower()
    ) and "global_payment_policy.md" not in cites:
        if "global" in answer.lower() or "100,000" in answer or "100000" in answer:
            cites.insert(0, "global_payment_policy.md")

    used = list(dict.fromkeys(tools_used)) if tools_used else []
    if not used:
        model_used = payload.get("tools_used")
        if isinstance(model_used, list):
            used = [str(u) for u in model_used]

    return {
        "answer": answer.strip(),
        "citations": list(dict.fromkeys(cites)),
        "facts": facts,
        "tools_used": used,
    }


def run_agent(
    question: str,
    payment_id: str,
) -> dict:
    """Run the AI assistant with an OpenAI tool-calling loop."""
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    max_rounds = 10

    user_message = (
        f"Payment under investigation: {payment_id}\n"
        f"Question: {question}\n\n"
        "Execute real lookups (do not guess):\n"
        "1) get_payment\n"
        "2) get_client_profile\n"
        "3) evaluate_review_requirements(payment_id) — mandatory for arithmetic\n"
        "4) search_policy for the applicable clauses (region thresholds, AE "
        "high-risk, structuring, and/or investigation workflow as needed)\n"
        "5) For structuring questions also call aggregate_beneficiary_24h with "
        "the payment's payment_date\n"
        "Then return the final JSON only. Ground every claim; cite policies used.\n"
        "Quality checklist for the answer text:\n"
        "- If beneficiary_country and beneficiary_country_code differ, state both.\n"
        "- For structuring: list each payment id/amount/channel; say each is below "
        "individual RM/enhanced thresholds if true; combined total vs structuring "
        "rule; pattern is observed evidence NOT proof of intent; Switzerland → "
        "escalate to Compliance.\n"
        "- For facts-vs-assumptions questions: use explicit Facts / Assumptions / "
        "Recommendation sections.\n"
        "- For workflow questions: enumerate the investigation_procedure.md steps.\n"
        "- For region-threshold questions: state the regional RM and enhanced "
        "thresholds first."
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tools_used: list[str] = []
    gathered_facts: dict[str, Any] = {"payment_id": payment_id}
    citations: list[str] = []
    final_payload: dict | None = None
    eval_result: dict | None = None

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        message = response.choices[0].message
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            final_payload = _parse_json_object(message.content or "")
            break

        for tc in message.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            if name in {"get_payment", "evaluate_review_requirements"}:
                args.setdefault("payment_id", payment_id)

            # Scope 24h aggregation to the payment's date when known
            if (
                name == "aggregate_beneficiary_24h"
                and "payment_date" not in args
                and gathered_facts.get("payment_date")
            ):
                args["payment_date"] = gathered_facts["payment_date"]

            result = _execute_tool(name, args)
            tools_used.append(name)

            if name == "get_payment" and isinstance(result, dict) and "error" not in result:
                for key in (
                    "amount",
                    "currency",
                    "beneficiary_country_code",
                    "beneficiary_country",
                    "beneficiary_name",
                    "client_id",
                    "channel",
                    "payment_date",
                ):
                    if key in result:
                        gathered_facts[key] = result[key]
            elif name == "get_client_profile" and isinstance(result, dict) and "error" not in result:
                gathered_facts["client_country"] = result.get("country")
                gathered_facts["client_risk_rating"] = result.get("risk_rating")
                gathered_facts["client_type"] = result.get("client_type")
            elif name == "search_policy" and isinstance(result, list):
                for hit in result:
                    src = hit.get("source")
                    if src and not str(src).startswith("decoy_") and src not in citations:
                        citations.append(src)
            elif name == "aggregate_beneficiary_24h" and isinstance(result, dict):
                gathered_facts["beneficiary_24h_count"] = result.get("count")
                gathered_facts["beneficiary_24h_total"] = result.get("total_amount")
                gathered_facts["beneficiary_24h_date"] = result.get("payment_date")
                gathered_facts["beneficiary_24h_payment_ids"] = result.get("payment_ids")
                gathered_facts["beneficiary_24h_individual_amounts"] = result.get(
                    "individual_amounts"
                )
            elif name == "evaluate_review_requirements" and isinstance(result, dict):
                eval_result = result
                for key in (
                    "amount",
                    "currency",
                    "beneficiary_country_code",
                    "beneficiary_country",
                    "beneficiary_name",
                    "client_id",
                    "client_country",
                    "client_risk_rating",
                    "payment_date",
                    "channel",
                    "requires_rm_review",
                    "requires_enhanced_review",
                    "requires_additional_review",
                    "high_risk_destination",
                    "possible_structuring",
                    "recommendations",
                    "policy_triggers",
                    "applicable_policies",
                    "currency_assumption",
                ):
                    if key in result and result[key] is not None:
                        gathered_facts[key] = result[key]
                window = result.get("same_day_beneficiary_window") or {}
                if window:
                    gathered_facts["beneficiary_24h_count"] = window.get("count")
                    gathered_facts["beneficiary_24h_total"] = window.get("total_amount")
                    gathered_facts["beneficiary_24h_date"] = window.get("payment_date")
                    gathered_facts["beneficiary_24h_payment_ids"] = window.get(
                        "payment_ids"
                    )
                    gathered_facts["beneficiary_24h_individual_amounts"] = window.get(
                        "individual_amounts"
                    )
                for src in result.get("applicable_policies") or []:
                    if src and src not in citations and not str(src).startswith("decoy_"):
                        citations.append(src)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )

    if final_payload is None:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Using only the tool results above, respond with the final "
                    "JSON object now (answer, citations, facts, tools_used). "
                    "No other text. Align recommendations with "
                    "evaluate_review_requirements output."
                ),
            }
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        final_payload = _parse_json_object(response.choices[0].message.content or "")

    # If the model skipped evaluate_review_requirements, run it once for facts.
    if eval_result is None:
        eval_result = evaluate_review_requirements(payment_id)
        tools_used.append("evaluate_review_requirements")
        if isinstance(eval_result, dict) and "error" not in eval_result:
            for key in (
                "amount",
                "currency",
                "beneficiary_country_code",
                "client_id",
                "client_country",
                "requires_rm_review",
                "requires_enhanced_review",
                "requires_additional_review",
                "high_risk_destination",
                "possible_structuring",
                "recommendations",
            ):
                if key in eval_result:
                    gathered_facts[key] = eval_result[key]

    return _normalize_result(
        final_payload,
        payment_id=payment_id,
        tools_used=tools_used,
        gathered_facts=gathered_facts,
        citations=citations,
        eval_result=eval_result if isinstance(eval_result, dict) else None,
        question=question,
    )
