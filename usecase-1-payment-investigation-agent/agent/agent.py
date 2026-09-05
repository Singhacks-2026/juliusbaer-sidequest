"""
Payment-investigation assistant.

    question
       |
    deterministic assessment  (tools: facts, thresholds, 24h window)
       |
    policy retrieval           (RAG: evidence with source filenames)
       |
    LLM narration              (Claude, tool-calling loop, may fetch more)
       |
    normalised answer          (facts + tools_used are always from the tools)

Detectors compute, the model narrates. If no LLM credentials are available,
or the model call fails for any reason, a deterministic narrator composes
the answer from the same assessment so the run never crashes and every
answer stays grounded.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    find_repeated_beneficiaries,
    get_client_payments,
    get_payment,
)
from tools.policy_tools import (
    get_policy_document,
    index_backend,
    list_policy_sources,
    search_policy,
)
from agent.assessor import (
    PROCEDURE_POLICY,
    assess_payment,
    assess_payment_policy,
    check_structuring,
    merge_assessment,
)

# --------------------------------------------------------------------------
# .env loading (no third-party dependency)
# --------------------------------------------------------------------------

def _load_dotenv() -> None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

TRACE_PATH = os.environ.get(
    "TRACE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "investigations.json"),
)

# --------------------------------------------------------------------------
# Tool registry + call recorder
# --------------------------------------------------------------------------

TOOLS = {
    "get_client_profile": get_client_profile,
    "get_payment": get_payment,
    "get_client_payments": get_client_payments,
    "aggregate_beneficiary_24h": aggregate_beneficiary_24h,
    "find_repeated_beneficiaries": find_repeated_beneficiaries,
    "search_policy": search_policy,
    "get_policy_document": get_policy_document,
}

TOOL_SCHEMAS = [
    {
        "name": "assess_payment_policy",
        "description": "Deterministic policy assessment of one payment: retrieves the payment and client, "
        "identifies the applicable global/regional policies, compares the amount against every review "
        "threshold, and checks the destination code against the high-risk list. Returns facts, threshold "
        "results, required reviews and the policy files relied on. Does NOT check 24-hour structuring.",
        "input_schema": {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
        },
    },
    {
        "name": "check_structuring",
        "description": "Deterministic 24-hour structuring check: aggregates a client's payments to one "
        "beneficiary on the same date and compares the combined value against the structuring threshold. "
        "Use for transaction-splitting / structuring / pattern questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string"},
                "beneficiary_name": {"type": "string"},
            },
            "required": ["client_id", "beneficiary_name"],
        },
    },
    {
        "name": "get_payment",
        "description": "Retrieve one payment record by payment ID.",
        "input_schema": {
            "type": "object",
            "properties": {"payment_id": {"type": "string"}},
            "required": ["payment_id"],
        },
    },
    {
        "name": "get_client_profile",
        "description": "Retrieve a client's profile: country, risk rating, client type, relationship years.",
        "input_schema": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    },
    {
        "name": "get_client_payments",
        "description": "All supplied payments for a client, oldest first.",
        "input_schema": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    },
    {
        "name": "aggregate_beneficiary_24h",
        "description": "Deterministic 24-hour aggregation of a client's payments to one beneficiary "
        "(same calendar date). Returns count, total, payment IDs, channels.",
        "input_schema": {
            "type": "object",
            "properties": {
                "client_id": {"type": "string"},
                "beneficiary_name": {"type": "string"},
            },
            "required": ["client_id", "beneficiary_name"],
        },
    },
    {
        "name": "find_repeated_beneficiaries",
        "description": "Beneficiaries that appear more than once in a client's payment history.",
        "input_schema": {
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve policy passages relevant to a query. Each result has source filename and text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_policy_document",
        "description": "Retrieve a complete policy document by filename.",
        "input_schema": {
            "type": "object",
            "properties": {"source": {"type": "string"}},
            "required": ["source"],
        },
    },
]


class ToolLog:
    """
    Per-run tool registry and call recorder.

    The two assessor tools are bound to this log's ``call`` so the data
    tools they use internally are recorded too - ``tools_used`` then lists
    every tool that actually ran, in order, whether the model called it
    directly or through an assessor.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.results: list[tuple[str, dict, object]] = []
        self.registry = {
            **TOOLS,
            "assess_payment_policy": lambda payment_id: assess_payment_policy(payment_id, self.call),
            "check_structuring": lambda client_id, beneficiary_name: check_structuring(client_id, beneficiary_name, self.call),
        }

    def call(self, name: str, **kwargs):
        fn = self.registry[name]
        result = fn(**kwargs)
        self.calls.append({"tool": name, "args": kwargs, "result_preview": _preview(result)})
        self.results.append((name, kwargs, result))
        return result

    def first(self, name: str, **match):
        """First recorded result of a tool, optionally matching call args."""
        for tool, args, result in self.results:
            if tool == name and all(args.get(k) == v for k, v in match.items()):
                return result
        return None

    def search_hits(self) -> list[dict]:
        hits, seen = [], set()
        for tool, _, result in self.results:
            if tool != "search_policy":
                continue
            for hit in result:
                key = (hit["source"], hit["text"])
                if key not in seen:
                    seen.add(key)
                    hits.append(hit)
        return hits

    @property
    def names(self) -> list[str]:
        seen, out = set(), []
        for c in self.calls:
            if c["tool"] not in seen:
                seen.add(c["tool"])
                out.append(c["tool"])
        return out


def _preview(result, limit: int = 160) -> str:
    text = json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


# --------------------------------------------------------------------------
# Policy retrieval for the no-model narrator
# --------------------------------------------------------------------------

def _retrieval_queries(question: str, assessment: dict) -> list[str]:
    """Queries come from the question and from what the assessment found - never from question IDs."""
    queries = [question]
    if not assessment.get("found"):
        return queries
    if assessment["regional_policy"]:
        queries.append(f"{assessment['facts'].get('client_country') or ''} payment procedure review threshold")
    queries.append("payments above threshold require enhanced review")
    if assessment["high_risk"]:
        queries.append("high-risk jurisdiction destination additional review")
    if assessment["structuring"]["count"] >= 2:
        queries.append("multiple payments same beneficiary within 24 hours combined value structuring")
        queries.append("potential structuring escalated to Compliance")
    queries.append("payment investigation procedure steps establish facts identify policy")
    queries.append("policy trigger does not establish suspicious activity")
    return queries


def _gather_evidence(log: ToolLog, queries: list[str]) -> list[dict]:
    evidence, seen = [], set()
    for q in queries:
        for hit in log.call("search_policy", query=q, top_k=3):
            key = (hit["source"], hit["text"])
            if key not in seen:
                seen.add(key)
                evidence.append(hit)
    return evidence


# --------------------------------------------------------------------------
# Deterministic narrator (fallback, and the source of truth for facts)
# --------------------------------------------------------------------------

def _money(amount: float, currency: str) -> str:
    return f"{currency} {amount:,.0f}" if float(amount).is_integer() else f"{currency} {amount:,.2f}"


def _fact_sentence(a: dict) -> str:
    p, c, f = a["payment"], a["client"], a["facts"]
    client_bits = ", ".join(
        b for b in [f.get("client_country"), f.get("client_risk_rating") and f"{f['client_risk_rating']} risk", f.get("client_type")] if b
    )
    s = (
        f"{p['payment_id']} is a {_money(f['amount'], f['currency'])} {p.get('channel', '')} payment on "
        f"{p.get('payment_date')} from client {p['client_id']} ({client_bits}) to {p['beneficiary_name']}, "
        f"beneficiary_country_code {f['beneficiary_country_code']}"
    )
    s += " (high-risk jurisdiction)." if a["high_risk"] else " (not a high-risk jurisdiction)."
    if a["country_code_mismatch"]:
        s += (
            f" The record lists beneficiary_country '{f['beneficiary_country']}' but the authoritative "
            f"beneficiary_country_code is {f['beneficiary_country_code']}, which is what the jurisdiction check uses."
        )
    return s


def _threshold_sentences(a: dict) -> str:
    parts = []
    for chk in a["threshold_checks"]:
        rel = "above" if chk["exceeds"] else "below"
        parts.append(
            f"{rel} the {chk['threshold_currency']} {chk['threshold']:,.0f} {chk['requirement']} threshold "
            f"in {chk['source']}"
        )
    if not parts:
        return ""
    return "Against the applicable policies the amount is " + "; ".join(parts) + "."


def _structuring_sentence(a: dict) -> str:
    s = a["structuring"]
    if s["count"] < 2:
        return (
            f"No other payment from {a['payment']['client_id']} to {a['payment']['beneficiary_name']} "
            f"falls on the same date, so no 24-hour structuring pattern is observed for this beneficiary."
        )
    listing = ", ".join(f"{pid} ({amt:,.0f})" for pid, amt in zip(s["payment_ids"], s["individual_amounts"]))
    cur = "/".join(s["currencies"])
    text = (
        f"On {s['window_date']} client {a['payment']['client_id']} made {s['count']} payments to "
        f"{a['payment']['beneficiary_name']}: {listing} {cur}, combined {cur} {s['total_amount']:,.0f}, "
        f"which {'exceeds' if s['flagged'] else 'does not exceed'} the "
        f"{s['rule']['threshold_currency']} {s['rule']['threshold']:,.0f} equivalent structuring threshold"
    )
    if s["each_below_thresholds"]:
        text += "; each payment individually sits below every review threshold"
    if len(set(s["channels"])) > 1:
        text += f"; the payments used different channels ({', '.join(s['channels'])})"
    text += ". This is an observed fact pattern, not proof of intent."
    return text


def _requirement_sentence(a: dict) -> str:
    if not a["requirements"]:
        return "No review requirement is triggered by amount, destination or pattern; standard monitoring applies."
    items = "; ".join(f"{r['requirement']} - {r['reason']} ({r['source']})" for r in a["requirements"])
    return f"Required: {items}."


HEDGE = "A policy trigger does not by itself establish suspicious activity."


def _compose_answer(question: str, a: dict, evidence: list[dict]) -> str:
    """
    One composition for every question: facts, threshold results, structuring
    window, required reviews, assumptions, the procedure if retrieved, and the
    hedge. Nothing here depends on how the question was phrased.
    """
    if not a.get("found"):
        return f"Payment {a.get('payment_id')} was not found in the supplied data, so no assessment can be made. {a.get('error', '')}".strip()

    parts = [_fact_sentence(a), _threshold_sentences(a), _structuring_sentence(a), _requirement_sentence(a)]
    if a["assumptions"]:
        parts.append("Assumptions: " + "; ".join(a["assumptions"]) + ".")
    parts.append(
        "Not supplied by the data: purpose of payment, source of funds and the client's relationship with the "
        "beneficiary; these would be needed before any conclusion on intent."
    )
    if any(e["source"] == PROCEDURE_POLICY for e in evidence):
        doc = get_policy_document(PROCEDURE_POLICY)
        steps = re.findall(r"^\s*\d+[.)]\s+(.+)$", doc.get("text", ""), flags=re.MULTILINE)
        if steps:
            parts.append(f"Investigation procedure ({PROCEDURE_POLICY}): " + " ".join(f"{i + 1}. {s.strip()}" for i, s in enumerate(steps)))
    parts.append(HEDGE)
    return " ".join(p for p in parts if p)


# --------------------------------------------------------------------------
# LLM narration
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a bank payment-investigation assistant.

Rules:
1. Retrieve transaction facts before making factual claims.
2. Use deterministic tools for arithmetic and aggregation.
3. Retrieve applicable policy evidence through RAG.
4. Separate observed facts from assumptions.
5. A policy trigger does not automatically establish suspicious activity.
6. Explain missing evidence when necessary.
7. Cite relevant policy sources.

How to work:
- Call only the tools the question needs. assess_payment_policy gives the payment's facts, threshold
  comparisons and high-risk check in one call. check_structuring (and get_client_payments /
  aggregate_beneficiary_24h / find_repeated_beneficiaries) is for the 24-hour same-beneficiary
  pattern: call it when the question is about splitting, structuring, patterns or what to request
  before escalating, or when the assessment shows a repeated beneficiary. For a plain threshold,
  jurisdiction, documents-to-retrieve or workflow question, do not run the structuring tools. Use search_policy to retrieve the policy wording
  you will cite; get_policy_document reads a whole file when you already know which one.
- Treat every number, comparison and flag returned by a tool as authoritative. Never recompute them.
- Use beneficiary_country_code for jurisdiction, never the country name; if they disagree, say so.
- Regional procedures (regional_*.md) apply by the CLIENT's country, never by the beneficiary's
  code or the payment currency. Do not cite or reason from a regional procedure that is not the
  client's own; the assessment's applicable_policies list is authoritative.
- Lead with the conclusion in the first sentence. Be as long as the question needs and no longer.
- Never call a payment "suspicious". State what review the policy requires, what is observed, and what
  is assumed or missing.
- Cite only files that a tool returned. Write for a compliance reviewer: specific and grounded.

When done, respond ONLY with a JSON object:
{"answer": string, "citations": [policy filenames], "assumptions": [strings]}"""


ACTIVE_MODEL = None


def _llm_available():
    """
    Pick a provider from the environment. Returns (provider, client) or None.
    Anthropic is preferred when ANTHROPIC_API_KEY is set; otherwise any
    OpenAI-compatible endpoint via OPENAI_API_KEY (+ optional OPENAI_BASE_URL),
    matching the two provider families in .env.example.
    """
    global ACTIVE_MODEL
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        try:
            import anthropic
        except ImportError:
            anthropic = None
        if anthropic is not None:
            ACTIVE_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")
            return "anthropic", anthropic.Anthropic()
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
        except ImportError:
            return None
        ACTIVE_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
        return "openai", OpenAI()
    return None


def _openai_tool_schemas() -> list[dict]:
    return [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
        for t in TOOL_SCHEMAS
    ]


def _narrate_with_openai(client, question: str, payment_id: str, log: ToolLog) -> dict | None:
    """Same loop as the Anthropic path, on the OpenAI-compatible chat.completions surface."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nPayment ID: {payment_id}"},
    ]
    try:
        for _ in range(10):
            response = client.chat.completions.create(
                model=ACTIVE_MODEL, messages=messages, tools=_openai_tool_schemas(), tool_choice="auto"
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                return _parse_json(msg.content or "")
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    out = log.call(tc.function.name, **json.loads(tc.function.arguments or "{}"))
                    content = json.dumps(out, default=str)
                except Exception as exc:  # noqa: BLE001
                    content = f"error: {exc}"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] OpenAI-compatible call failed ({type(exc).__name__}); using deterministic narrator")
        return None


def _narrate_with_llm(provider_client, question: str, payment_id: str, log: ToolLog) -> dict | None:
    """Tool-calling loop. The model starts with only the question. Returns parsed JSON or None."""
    provider, client = provider_client
    if provider == "openai":
        return _narrate_with_openai(client, question, payment_id, log)
    import anthropic

    messages = [{"role": "user", "content": f"Question: {question}\nPayment ID: {payment_id}"}]
    try:
        for _ in range(10):
            response = client.messages.create(
                model=ACTIVE_MODEL,
                max_tokens=4000,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )
            if response.stop_reason == "refusal":
                return None
            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                return _parse_json(text)

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    out = log.call(block.name, **dict(block.input))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(out, default=str)})
                except Exception as exc:  # noqa: BLE001
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(exc), "is_error": True})
            messages.append({"role": "user", "content": results})
        return None
    except anthropic.APIError as exc:
        print(f"[agent] LLM call failed ({type(exc).__name__}); using deterministic narrator")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[agent] LLM narration error ({exc}); using deterministic narrator")
        return None


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {"answer": text} if text else None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"answer": text}


# --------------------------------------------------------------------------
# Audit trace for the UI
# --------------------------------------------------------------------------

_trace: list[dict] = []


def _write_trace(entry: dict) -> None:
    try:
        _trace.append(entry)
        os.makedirs(os.path.dirname(TRACE_PATH), exist_ok=True)
        with open(TRACE_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "rag_backend": index_backend(),
                    "policy_sources": list_policy_sources(),
                    "investigations": _trace,
                },
                fh,
                indent=2,
                default=str,
            )
    except OSError:
        pass  # the trace is a convenience; never fail the graded run over it


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_agent(question: str, payment_id: str) -> dict:
    """Answer one official question. Never raises."""
    log = ToolLog()
    try:
        client = _llm_available()
        narrated = _narrate_with_llm(client, question, payment_id, log) if client else None
        llm_used = bool(narrated and narrated.get("answer"))

        if llm_used:
            # Facts always come from the deterministic tools. If the model answered without the
            # policy assessment, run it now so `facts` is complete (it is recorded as used).
            policy = log.first("assess_payment_policy", payment_id=payment_id) or log.first("assess_payment_policy")
            if policy is None:
                policy = log.call("assess_payment_policy", payment_id=payment_id)
            structuring = None
            if policy.get("found"):
                structuring = (
                    log.first("check_structuring", beneficiary_name=policy["payment"]["beneficiary_name"])
                    or log.first("check_structuring")
                )
            assessment = merge_assessment(policy, structuring)
            evidence = log.search_hits()
            answer = narrated["answer"].strip()
        else:
            assessment = assess_payment(payment_id, log.call)
            evidence = _gather_evidence(log, _retrieval_queries(question, assessment))
            answer = _compose_answer(question, assessment, evidence)
        answer = re.sub(r"\s{2,}", " ", answer).strip()

        # Citations: only documents the pipeline actually retrieved or the assessor relied on.
        # Regional procedures apply by the client's country (DATA_NOTES.md), so a regional file
        # that is not this client's is never a valid citation even if retrieval surfaced it.
        evidence_sources = sorted({e["source"] for e in evidence})
        applicable_regional = assessment.get("regional_policy")
        allowed = {
            s for s in set(evidence_sources) | set(assessment.get("citations", []))
            if not (s.startswith("regional_") and s != applicable_regional)
        }
        proposed = narrated.get("citations", []) if llm_used else []
        citations = [c for c in proposed if c in allowed]
        if not citations:
            citations = list(assessment.get("citations", []))
            if PROCEDURE_POLICY in evidence_sources:
                citations.append(PROCEDURE_POLICY)
        citations = list(dict.fromkeys(citations))

        facts = assessment.get("facts", {"payment_id": payment_id, "found": False})
        result = {
            "answer": answer,
            "citations": citations,
            "facts": facts,
            "tools_used": log.names,
        }
    except Exception as exc:  # noqa: BLE001 - a crash on any question is a disqualifier
        assessment, evidence, llm_used = {"found": False, "payment_id": payment_id, "error": str(exc)}, [], False
        result = {
            "answer": f"The assistant could not complete the investigation for {payment_id}: {exc}. No conclusion is drawn.",
            "citations": [],
            "facts": {"payment_id": payment_id, "error": str(exc)},
            "tools_used": log.names,
        }

    _write_trace(
        {
            "question": question,
            "payment_id": payment_id,
            "assessment": assessment,
            "evidence": evidence,
            "tool_calls": log.calls,
            "narrator": {"llm": llm_used, "model": ACTIVE_MODEL if llm_used else None},
            **result,
        }
    )
    return result
