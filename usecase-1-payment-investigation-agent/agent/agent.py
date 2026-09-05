"""
AI AGENT — orchestration.

    Question
       -> deterministic pre-flight (payment + client always fetched)
       -> LLM tool-calling loop (the model chooses what else it needs)
       -> evidence-completeness gate (deterministic backstop)
       -> deterministic policy assessment (all arithmetic)
       -> LLM synthesis (prose only)
       -> Python assembly of the submission record

The division of labour is the point of this design.  The LLM plans, selects
tools, interprets and writes; it never computes an amount, compares a
threshold, counts payments, decides what is high-risk, or chooses a citation.
``answer`` is the only field it produces.  ``facts``, ``citations`` and
``tools_used`` are built by the evidence ledger from tools that actually ran.

Two deterministic guards sit around the model.  The pre-flight guarantees that
the payment and client records are always present, so ``facts`` is never empty
even if the model misbehaves.  The completeness gate re-checks the evidence
against what the question needs — a question about transaction splitting that
never aggregated a 24-hour window gets the aggregation run for it — which
enforces tool coverage without ever branching on a question ID.
"""

import json
import re

from agent import llm
from agent.evidence import EvidenceLedger
from agent.fallback import render_answer
from agent.prompts import SYNTHESIS_PROMPT, SYSTEM_PROMPT
from tools import risk_rules
from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    find_repeated_beneficiaries,
    get_client_payments,
    get_payment,
)
from tools.policy_tools import search_policy

TOOLS = {
    "get_client_profile": get_client_profile,
    "get_payment": get_payment,
    "get_client_payments": get_client_payments,
    "aggregate_beneficiary_24h": aggregate_beneficiary_24h,
    "find_repeated_beneficiaries": find_repeated_beneficiaries,
    "search_policy": search_policy,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_payment",
            "description": (
                "Retrieve one payment by ID: amount, currency, beneficiary, "
                "beneficiary country and country code, channel and date."
            ),
            "parameters": {
                "type": "object",
                "properties": {"payment_id": {"type": "string"}},
                "required": ["payment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_profile",
            "description": (
                "Retrieve a client's profile: relationship country (which "
                "determines the applicable regional policy), risk rating, "
                "client type and relationship duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {"client_id": {"type": "string"}},
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_client_payments",
            "description": (
                "Retrieve a client's full payment history, oldest first. Use "
                "for transaction-pattern questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {"client_id": {"type": "string"}},
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_repeated_beneficiaries",
            "description": (
                "List beneficiaries a client paid more than once, flagging "
                "dates carrying multiple payments. Use to find which "
                "beneficiary to aggregate when the question names none."
            ),
            "parameters": {
                "type": "object",
                "properties": {"client_id": {"type": "string"}},
                "required": ["client_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_beneficiary_24h",
            "description": (
                "Deterministically aggregate a client's payments to one "
                "beneficiary within a 24-hour window, returning counts and "
                "combined totals. Required for any structuring question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_id": {"type": "string"},
                    "beneficiary_name": {"type": "string"},
                },
                "required": ["client_id", "beneficiary_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Retrieve policy evidence by natural-language query, returning "
                "passages with their source document for citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
]

MAX_TOOL_ITERATIONS = 4
_MAX_AGGREGATION_CANDIDATES = 5

_CLIENT_ID = re.compile(r"\bC\d{4}\b")
_PAYMENT_ID = re.compile(r"\bP\d{5}\b")

_STRUCTURING_TERMS = (
    "structur",
    "splitting",
    "split",
    "aggregat",
    "24 hour",
    "24-hour",
    "same beneficiary",
    "escalat",
    "pattern",
)

_PROCESS_TERMS = (
    "workflow",
    "procedure",
    "process",
    "steps",
    "summarize",
    "summarise",
    "retrieve",
    "before recommending",
    "additional information",
    "next step",
    "should be requested",
)


def run_agent(question: str, payment_id: str) -> dict:
    """
    Answer one investigation question.

    Never raises: ``main.py`` iterates the official questions without its own
    error handling, and a crash on any of them is a disqualifier.
    """
    ledger = EvidenceLedger()

    try:
        context = _gather_evidence(question, payment_id, ledger)
        answer = _synthesize(question, context, ledger)
        citations = _filter_citations(ledger.citations, context)
    except Exception as error:  # noqa: BLE001 - degrade, never abort the run
        answer = (
            f"The assistant could not complete this investigation: "
            f"{type(error).__name__}: {error}. No grounded recommendation is "
            "available and the payment should be reviewed manually."
        )
        citations = ledger.citations

    return {
        "answer": answer,
        "citations": citations,
        "facts": ledger.facts,
        "tools_used": ledger.tools_used,
    }


# -- evidence gathering ----------------------------------------------------


def _invoke(ledger: EvidenceLedger, name: str, **kwargs):
    """
    Single choke point for tool execution.

    Both the LLM loop and the deterministic phases go through here, so every
    invocation lands in the ledger and policy results always become citations.
    """
    function = TOOLS.get(name)
    if function is None:
        return {"error": f"Unknown tool {name!r}."}

    result = ledger.call(name, function, **kwargs)

    if name == "search_policy" and isinstance(result, list):
        ledger.add_policy_evidence(result)

    return result


def _gather_evidence(question: str, payment_id: str, ledger: EvidenceLedger) -> dict:
    """Run the pre-flight, the model's tool loop, then the deterministic gate."""
    payment, client = _preflight(question, payment_id, ledger)
    context: dict = {"payment": payment, "client": client}

    _llm_tool_loop(question, context, ledger)

    assessment = risk_rules.assess_payment(payment, client)
    context["assessment"] = assessment

    context["structuring"] = _ensure_structuring(question, payment, client, ledger)
    _ensure_policy_evidence(question, payment, client, assessment, ledger)

    if _matches(question, _PROCESS_TERMS):
        context["workflow"] = risk_rules.investigation_workflow()

    context["assumptions"] = _collect_assumptions(context)
    _record_facts(context, ledger)

    return context


def _preflight(question: str, payment_id: str, ledger: EvidenceLedger):
    """
    Always fetch the payment and the client.

    Every official question needs both, so this runs unconditionally rather
    than waiting for the model to ask.  The question text is also swept for
    entity IDs: one question asks about a client while supplying a different
    client's payment, so the payment's own ``client_id`` is not always the
    subject.
    """
    payment = _invoke(ledger, "get_payment", payment_id=payment_id)

    client_ids = _CLIENT_ID.findall(question or "")
    if payment.get("found") and payment["client_id"] not in client_ids:
        client_ids.append(payment["client_id"])

    client = {"found": False}
    for client_id in client_ids:
        candidate = _invoke(ledger, "get_client_profile", client_id=client_id)
        if candidate.get("found"):
            client = candidate
            break

    return payment, client


def _llm_tool_loop(question: str, context: dict, ledger: EvidenceLedger) -> None:
    """Let the model request whatever further evidence it judges necessary."""
    if not llm.is_configured():
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Already retrieved:\n"
                f"{json.dumps({'payment': context['payment'], 'client': context['client']}, indent=2, default=str)}\n\n"
                "Call the tools you need to answer this, then stop. Always "
                "retrieve policy evidence before concluding."
            ),
        },
    ]

    for _ in range(MAX_TOOL_ITERATIONS):
        message = llm.chat(messages, tools=TOOL_SCHEMAS)
        if message is None or not message["tool_calls"]:
            return

        messages.append(llm.assistant_turn(message))
        for call in message["tool_calls"]:
            result = _invoke(ledger, call["name"], **call["arguments"])
            messages.append(llm.tool_turn(call["id"], result))


def _ensure_structuring(
    question: str, payment: dict, client: dict, ledger: EvidenceLedger
) -> dict | None:
    """
    Guarantee a 24-hour aggregation for any splitting question.

    Keyed on the question's wording, never on its ID.  When no beneficiary is
    named, every repeated beneficiary is aggregated and the largest window is
    assessed, so the answer covers absence of a pattern as well as presence.
    """
    if not _matches(question, _STRUCTURING_TERMS) or not payment.get("found"):
        return None

    client_id = client["client_id"] if client.get("found") else payment["client_id"]

    aggregation = ledger.result_of("aggregate_beneficiary_24h")
    if not isinstance(aggregation, dict) or not aggregation.get("largest_window"):
        aggregation = _best_aggregation(client_id, payment, ledger)

    if aggregation is None:
        return None

    return risk_rules.assess_structuring(aggregation, client)


def _best_aggregation(client_id: str, payment: dict, ledger: EvidenceLedger):
    """
    Aggregate each repeated beneficiary and keep the highest-value window.

    Walks the same path an investigator would: pull the history, find the
    beneficiaries paid more than once, then aggregate each candidate window.
    """
    _invoke(ledger, "get_client_payments", client_id=client_id)
    repeated = _invoke(ledger, "find_repeated_beneficiaries", client_id=client_id)

    names = [
        item["beneficiary_name"]
        for item in (repeated if isinstance(repeated, list) else [])
        if item.get("dates_with_multiple_payments")
    ][:_MAX_AGGREGATION_CANDIDATES]

    # No same-day repetition: still aggregate this payment's beneficiary so the
    # answer can state the absence of a pattern from evidence rather than
    # silence.
    if not names:
        names = [payment["beneficiary_name"]]

    best = None
    for name in names:
        aggregation = _invoke(
            ledger,
            "aggregate_beneficiary_24h",
            client_id=client_id,
            beneficiary_name=name,
        )
        if not isinstance(aggregation, dict):
            continue

        window = aggregation.get("largest_window")
        if window is None:
            continue

        if best is None or window["total_amount"] > best["largest_window"]["total_amount"]:
            best = aggregation

    return best


def _ensure_policy_evidence(
    question: str,
    payment: dict,
    client: dict,
    assessment: dict,
    ledger: EvidenceLedger,
) -> None:
    """
    Always run a fact-derived policy query.

    Retrieval quality depends far more on the query than on the index, and a
    query built from resolved facts — the client's jurisdiction, the actual
    amount and currency, the destination code — outperforms the raw question.
    The question text is still included because it carries the intent terms the
    reranker boosts on.
    """
    _invoke(
        ledger,
        "search_policy",
        query=_policy_query(question, payment, client, assessment),
        top_k=4,
    )

    if _matches(question, _PROCESS_TERMS):
        _invoke(
            ledger,
            "search_policy",
            query=(
                "investigation procedure workflow steps establish client and "
                "payment facts identify applicable policy record evidence"
            ),
            top_k=3,
        )


def _policy_query(
    question: str, payment: dict, client: dict, assessment: dict
) -> str:
    terms = [question or ""]

    if client.get("found"):
        terms.append(f"{client['country']} regional procedure review threshold")

    if payment.get("found"):
        terms.append(
            f"payment above {payment['currency']} {payment['amount']:.0f} "
            "review threshold"
        )

    if assessment.get("high_risk_destination"):
        terms.append(
            f"high-risk jurisdiction {assessment['beneficiary_country_code']} "
            "destination additional review"
        )

    if _matches(question, _STRUCTURING_TERMS):
        terms.append(
            "multiple payments same beneficiary within 24 hours potential "
            "structuring combined value"
        )

    return " ".join(term for term in terms if term)


# -- synthesis -------------------------------------------------------------


def _synthesize(question: str, context: dict, ledger: EvidenceLedger) -> str:
    """
    Produce the answer prose.

    The model sees only the assembled evidence and writes the narrative.  If it
    is unavailable, or returns nothing usable, the deterministic renderer
    produces the same five-part answer from the same evidence, so the field is
    never empty.
    """
    if llm.is_configured():
        message = llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": SYNTHESIS_PROMPT.format(
                        question=question, evidence=ledger.as_json()
                    ),
                },
            ]
        )
        if message and message["content"].strip():
            return message["content"].strip()

    return render_answer(question, context)


# -- assembly --------------------------------------------------------------


def _collect_assumptions(context: dict) -> list[str]:
    """Gather every assumption the deterministic layer had to make."""
    assumptions = list((context.get("assessment") or {}).get("currency_assumptions", []))

    structuring = context.get("structuring") or {}
    for key in ("window_assumption", "currency_assumption"):
        value = structuring.get(key)
        if value and value not in assumptions:
            assumptions.append(value)

    return assumptions


def _record_facts(context: dict, ledger: EvidenceLedger) -> None:
    """Write the deterministic values into the submission's ``facts`` object."""
    payment = context.get("payment") or {}
    client = context.get("client") or {}
    assessment = context.get("assessment") or {}
    structuring = context.get("structuring") or {}

    if payment.get("found"):
        ledger.add_facts(
            {
                "payment_id": payment["payment_id"],
                "client_id": payment["client_id"],
                "amount": payment["amount"],
                "currency": payment["currency"],
                "beneficiary_name": payment["beneficiary_name"],
                "beneficiary_country": payment["beneficiary_country"],
                "beneficiary_country_code": payment["beneficiary_country_code"],
                "payment_date": payment["payment_date"],
                "channel": payment["channel"],
            }
        )

    if client.get("found"):
        ledger.add_facts(
            {
                "client_country": client["country"],
                "client_risk_rating": client["risk_rating"],
                "client_type": client["client_type"],
                "relationship_years": client["relationship_years"],
                "policy_scope": client["policy_scope"],
            }
        )

    if assessment.get("assessable"):
        # The same requirement can be imposed by two policy layers at once —
        # global and Singapore both set enhanced review at USD 100,000 — so the
        # labels are deduplicated here while the assessment keeps both sources.
        requirements: list[str] = []
        for item in assessment["review_requirements"]:
            if item["requirement"] not in requirements:
                requirements.append(item["requirement"])

        if structuring.get("exceeds_threshold"):
            requirements.append("structuring review")

        ledger.add_facts(
            {
                "applicable_policy_documents": assessment["applicable_policy_documents"],
                "high_risk_destination": assessment["high_risk_destination"],
                "thresholds_evaluated": [
                    {
                        "requirement": evaluation["requirement_label"],
                        "threshold": (
                            f"{evaluation['threshold_currency']} "
                            f"{evaluation['threshold_amount']:,.2f}"
                        ),
                        "exceeded": evaluation["exceeds_threshold"],
                        "source": evaluation["source"],
                    }
                    for evaluation in assessment["threshold_evaluations"]
                ],
                "review_requirements": requirements or ["none"],
                "data_quality_flags": assessment["data_quality_flags"],
            }
        )

    if structuring:
        ledger.add_facts(
            {
                "structuring": {
                    key: structuring[key]
                    for key in (
                        "determination",
                        "beneficiary_name",
                        "payment_count",
                        "payment_ids",
                        "payment_date",
                        "combined_amount",
                        "combined_currency",
                        "threshold_amount",
                        "threshold_currency",
                        "comparison",
                        "exceeds_threshold",
                    )
                    if key in structuring
                }
            }
        )

    for assumption in context.get("assumptions") or []:
        ledger.add_assumption(assumption)


def _filter_citations(citations: list[str], context: dict) -> list[str]:
    """
    Keep only documents that can actually support this answer.

    Retrieval is already decoy-filtered, but a stray query can still surface a
    regional procedure that does not govern this client — citing Singapore's
    thresholds for a Swiss client is a grounding error even though the document
    is not a decoy.  Allowed sources are the client's own policy layers, plus
    the investigation procedure when the question is about process, plus any
    document a structuring finding rests on.
    """
    assessment = context.get("assessment") or {}
    structuring = context.get("structuring") or {}

    allowed = set(
        assessment.get("applicable_policy_documents")
        or risk_rules.applicable_policies(None)
    )
    allowed.update(structuring.get("sources") or [])

    if context.get("workflow"):
        allowed.add(risk_rules.INVESTIGATION_PROCEDURE)

    return [citation for citation in citations if citation in allowed]


def _matches(question: str, terms: tuple[str, ...]) -> bool:
    lowered = (question or "").casefold()
    return any(term in lowered for term in terms)
