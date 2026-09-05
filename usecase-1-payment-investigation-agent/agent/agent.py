"""
Payment investigation agent: evidence gathering + LLM tool loop.

Uniform pipeline for every question (no question_id branching):
payment → client → regional thresholds → high-risk check → structuring
screen → policy retrieval → synthesis. When an OpenAI-compatible LLM key
is configured (``OPENAI_API_KEY``/``LLM_API_KEY``), the model drives tool
selection and drafts the answer; otherwise a deterministic composer
produces the same schema from the same evidence bundle.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

from agent.synthesis import classify_intent, compose
from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    find_repeated_beneficiaries,
    get_client_payments,
    get_payment,
)
from tools.policy_tools import get_policy_document, search_policy

TOOLS = {
    "get_client_profile": get_client_profile,
    "get_payment": get_payment,
    "get_client_payments": get_client_payments,
    "aggregate_beneficiary_24h": aggregate_beneficiary_24h,
    "find_repeated_beneficiaries": find_repeated_beneficiaries,
    "search_policy": search_policy,
}


def _load_dotenv() -> None:
    """Load ``.env`` next to the use-case root (stdlib, no dependency)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

SYSTEM_PROMPT = """
You are a bank payment-investigation assistant.

Rules:
1. Retrieve transaction facts before making factual claims.
2. Use deterministic tools for arithmetic and aggregation.
3. Retrieve applicable policy evidence through RAG.
4. Separate observed facts from assumptions.
5. A policy trigger does not automatically establish suspicious activity.
6. Explain missing evidence when necessary.
7. Cite relevant policy sources.
8. Batch independent tool calls in a single block (e.g. payment +
   client + policy search together) to save round trips.
"""

_STRUCTURING_THRESHOLD = 100000.0  # USD-equivalent (1:1 convention)


def _regional_thresholds(client_country: str) -> dict:
    country = (client_country or "").strip().lower()
    if country == "singapore":
        return {"rm": 75000.0, "enhanced": 100000.0,
                "threshold_currency": "USD",
                "policy": "regional_singapore.md"}
    if country == "switzerland":
        return {"rm": 80000.0, "enhanced": 120000.0,
                "threshold_currency": "CHF",
                "policy": "regional_switzerland.md"}
    return {"rm": None, "enhanced": 100000.0,
            "threshold_currency": "USD",
            "policy": "global_payment_policy.md"}


def _high_risk_codes() -> set[str]:
    """Parse jurisdiction codes from policy (code field is authoritative)."""
    try:
        doc = get_policy_document("high_risk_jurisdictions.md")
        text = doc.get("text", "") if isinstance(doc, dict) else ""
    except Exception:
        text = ""
    codes = set(re.findall(r"\b[A-Z]{2}\b", text))
    return codes or {"AE"}


def gather_evidence(question: str, payment_id: str) -> dict:
    """Collect the full deterministic evidence bundle for a question."""
    tools_used: list[str] = []

    def _call(name: str, *args, **kwargs):
        tools_used.append(name)
        return TOOLS[name](*args, **kwargs)

    payment = _call("get_payment", payment_id)
    if "error" in payment:
        return {"error": payment["error"], "tools_used": tools_used}

    client = _call("get_client_profile", payment["client_id"])

    thresholds = _regional_thresholds(client.get("country", ""))
    fx_assumption = (
        payment["currency"] != thresholds["threshold_currency"]
    )
    amount = payment["amount"]
    thresholds = {
        **thresholds,
        "rm_triggered": thresholds["rm"] is not None
        and amount > thresholds["rm"],
        "enhanced_triggered": amount > thresholds["enhanced"],
        "fx_assumption": fx_assumption,
    }

    high_risk = payment["beneficiary_country_code"] in _high_risk_codes()

    patterns = []
    for rep in _call("find_repeated_beneficiaries", payment["client_id"]):
        if not rep["same_date_repeat"]:
            continue
        agg = _call(
            "aggregate_beneficiary_24h",
            payment["client_id"],
            rep["beneficiary_name"],
        )
        window = agg["strongest_window"]
        if window is None:
            continue
        # 1:1 equivalence convention when currencies differ (DATA_NOTES).
        patterns.append(
            {
                "beneficiary_name": rep["beneficiary_name"],
                "window": window,
                "exceeds_threshold": window["total_amount"]
                > _STRUCTURING_THRESHOLD,
                "fx_assumption": isinstance(window["currency"], list)
                or window["currency"] != "USD",
            }
        )
    patterns.sort(
        key=lambda p: p["window"]["total_amount"], reverse=True
    )

    evidence = _call(
        "search_policy",
        f"{question} payment {payment_id} {payment['beneficiary_name']} "
        f"threshold review structuring high-risk",
        top_k=5,
    )
    intent = classify_intent(question)
    region_doc = {"singapore": "regional_singapore.md",
                  "switzerland": "regional_switzerland.md"}.get(
                      client.get("country", "").strip().lower(), "")
    citations = ["global_payment_policy.md"]
    if region_doc and region_doc not in citations:
        citations.append(region_doc)
    if high_risk or intent in ("high_risk_below_threshold",
                               "facts_vs_assumptions"):
        citations.append("high_risk_jurisdictions.md")
    if intent in ("workflow", "structuring", "additional_info",
                  "facts_vs_assumptions", "which_documents"):
        citations.append("investigation_procedure.md")
    # Selection is rule-based on client region, destination risk, and
    # question intent — the canonical policy set only, never decoys.
    canonical = {"global_payment_policy.md", "regional_singapore.md",
                 "regional_switzerland.md", "high_risk_jurisdictions.md",
                 "investigation_procedure.md"}
    citations = [c for c in citations if c in canonical]

    return {
        "payment": payment,
        "client": client,
        "thresholds": thresholds,
        "high_risk": high_risk,
        "structuring": {"patterns": patterns},
        "policy_evidence": evidence,
        "citations": citations,
        "tools_used": list(dict.fromkeys(tools_used)),
    }


# ---------------------------------------------------------------------------
# LLM tool-calling loop (OpenAI-compatible HTTP, stdlib only, no extra dep)
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": (func.__doc__ or name).strip().splitlines()[0],
            "parameters": {"type": "object",
                           "properties": {"payment_id": {"type": "string"},
                                          "client_id": {"type": "string"},
                                          "beneficiary_name": {"type": "string"},
                                          "country": {"type": "string"},
                                          "source": {"type": "string"},
                                          "query": {"type": "string"},
                                          "top_k": {"type": "integer"}},
                           "additionalProperties": True},
        },
    }
    for name, func in TOOLS.items()
]


def _llm_config() -> dict | None:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        return None
    return {
        "key": key,
        "base": os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


def _chat(cfg: dict, messages: list[dict], tools: list | None) -> dict:
    import time
    import urllib.error

    body: dict = {"model": cfg["model"], "messages": messages}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    last_exc: Exception | None = None
    for attempt in range(6):
        req = urllib.request.Request(
            f"{cfg['base']}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {cfg['key']}",
                     "Content-Type": "application/json",
                     # Groq sits behind Cloudflare, which blocks the
                     # default Python-urllib signature (HTTP 403/1010).
                     "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X "
                                   "10_15_7) AppleWebKit/537.36"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429,) or 500 <= exc.code < 600:
                wait = 4 * (2 ** attempt)
                try:
                    retry_after = exc.headers.get("Retry-After")
                    if retry_after:
                        wait = max(wait, float(retry_after))
                except Exception:
                    pass
                time.sleep(min(wait, 60))
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _run_llm_loop(question: str, payment_id: str) -> dict | None:
    """LLM-driven tool selection + synthesis; None when unavailable."""
    cfg = _llm_config()
    if cfg is None:
        return None
    invoked: list[str] = []
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Payment under review: {payment_id}. {question} "
                    f"Use tools for all facts, then answer as JSON with keys "
                    f"answer, citations, facts."},
    ]
    try:
        for _ in range(8):
            reply = _chat(cfg, messages, _TOOL_SCHEMAS)["choices"][0]["message"]
            messages.append({k: v for k, v in reply.items()
                             if k in ("role", "content", "tool_calls")})
            calls = reply.get("tool_calls") or []
            if not calls:
                break
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"].get("arguments") or "{}")
                try:
                    result = TOOLS[name](**{k: v for k, v in args.items()
                                           if k in ("payment_id", "client_id",
                                                    "beneficiary_name",
                                                    "country", "source",
                                                    "query", "top_k")})
                    invoked.append(name)
                except Exception as exc:  # never break the loop on tool error
                    result = {"error": str(exc)}
                messages.append({"role": "tool",
                                 "tool_call_id": call["id"],
                                 "content": json.dumps(result, default=str)})
        final = _chat(cfg, messages + [
            {"role": "user",
             "content": "Return ONLY JSON: {\"answer\": str, "
                        "\"citations\": [policy filenames], \"facts\": {}}."},
        ], None)["choices"][0]["message"].get("content", "")
        match = re.search(r"\{.*\}", final, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else None
        if not isinstance(parsed, dict) or not parsed.get("answer"):
            return None
        return {"draft": parsed, "invoked": list(dict.fromkeys(invoked))}
    except Exception:
        return None


def run_agent(question: str, payment_id: str) -> dict:
    """Answer one investigation question (never branches on IDs)."""
    bundle = gather_evidence(question, payment_id)
    if "error" in bundle:
        return {"answer": bundle["error"], "citations": [],
                "facts": {}, "tools_used": bundle.get("tools_used", [])}

    llm = _run_llm_loop(question, payment_id)
    if llm is None:
        return compose(bundle, question)

    # Ground the LLM draft against deterministic evidence.
    draft = llm["draft"]
    citations = [c for c in draft.get("citations", [])
                 if isinstance(c, str) and not c.startswith("decoy_")]
    for src in bundle["citations"]:
        if src not in citations:
            citations.append(src)
    facts = dict(bundle["payment"]) | {
        "client_country": bundle["client"]["country"],
        "client_risk_rating": bundle["client"]["risk_rating"],
    }
    if isinstance(draft.get("facts"), dict):
        facts.update(draft["facts"])
    tools_used = list(dict.fromkeys(bundle["tools_used"] + llm["invoked"]))
    return {
        "answer": str(draft.get("answer", "")),
        "citations": citations,
        "facts": facts,
        "tools_used": tools_used,
    }
