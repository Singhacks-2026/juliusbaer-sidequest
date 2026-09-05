"""
Deterministic policy assessment, exposed as tools.

These functions decide *facts*: which policies apply, which thresholds are
crossed, whether the destination is high-risk, and whether a 24-hour
structuring pattern exists. Everything is plain Python over tool results.
The language model chooses which of them to call and narrates the outcome -
it never originates a number or a verdict.

Two tools, so the model can pick what a question needs:

* ``assess_payment_policy(payment_id)`` - facts, applicable policies,
  threshold comparisons, high-risk check.
* ``check_structuring(client_id, beneficiary_name)`` - 24-hour window
  aggregation compared against the structuring rule.

``assess_payment`` runs both; the deterministic narrator uses it when no
model is available. ``merge_assessment`` combines whichever parts ran into
one shape for the audit trace and the ``facts`` field.

Thresholds and the high-risk list are parsed from the policy documents at
runtime rather than typed in as constants, so each rule can be traced to the
file it came from.
"""

from __future__ import annotations

import re
from typing import Callable

# Client relationship country -> regional procedure (DATA_NOTES.md). The
# global policy always applies; a regional procedure adds to it.
REGIONAL_POLICY = {
    "singapore": "regional_singapore.md",
    "switzerland": "regional_switzerland.md",
}
GLOBAL_POLICY = "global_payment_policy.md"
HIGH_RISK_POLICY = "high_risk_jurisdictions.md"
PROCEDURE_POLICY = "investigation_procedure.md"

_THRESHOLD_RE = re.compile(
    r"above\s+(USD|CHF|SGD|HKD|GBP)\s+([\d,]+)\s+equivalent\s+require\s+(RM review|enhanced review)",
    re.IGNORECASE,
)
_STRUCTURING_RE = re.compile(
    r"combined value exceeds\s+(USD|CHF|SGD|HKD|GBP)\s+([\d,]+)", re.IGNORECASE
)
_CODE_RE = re.compile(r"\b([A-Z]{2})\s*\(")

# Human-readable country labels that agree with a code; anything else is a mismatch.
_LABEL_FOR_CODE = {"AE": "UAE", "SG": "SINGAPORE", "CH": "SWITZERLAND", "HK": "HONG KONG", "GB": "UK"}

ToolCall = Callable[..., object]


# --------------------------------------------------------------------------
# Policy parsing
# --------------------------------------------------------------------------

def _parse_thresholds(text: str, source: str) -> list[dict]:
    rules = []
    for currency, amount, kind in _THRESHOLD_RE.findall(text):
        rules.append(
            {
                "requirement": "RM review" if kind.lower().startswith("rm") else kind.lower(),
                "threshold": float(amount.replace(",", "")),
                "threshold_currency": currency.upper(),
                "source": source,
            }
        )
    return rules


def _parse_structuring(text: str, source: str) -> dict:
    m = _STRUCTURING_RE.search(text)
    if not m:
        return {"threshold": 100000.0, "threshold_currency": "USD", "source": source}
    return {
        "threshold": float(m.group(2).replace(",", "")),
        "threshold_currency": m.group(1).upper(),
        "source": source,
    }


def _parse_high_risk_codes(text: str) -> set[str]:
    return set(_CODE_RE.findall(text)) or {"AE"}


def _policy_context(call: ToolCall, client_country: str) -> dict:
    """Load the global + regional documents for a client and parse their rules."""
    regional_file = REGIONAL_POLICY.get((client_country or "").lower())
    global_doc = call("get_policy_document", source=GLOBAL_POLICY)
    regional_doc = call("get_policy_document", source=regional_file) if regional_file else None
    rules = _parse_thresholds(global_doc.get("text", ""), GLOBAL_POLICY)
    if regional_doc and regional_doc.get("found"):
        rules += _parse_thresholds(regional_doc["text"], regional_file)
    return {
        "regional_file": regional_file,
        "applicable": [GLOBAL_POLICY] + ([regional_file] if regional_file else []),
        "rules": rules,
        "structuring_rule": _parse_structuring(global_doc.get("text", ""), GLOBAL_POLICY),
        "regional_mentions_structuring": bool(regional_doc and "structuring" in regional_doc.get("text", "").lower()),
    }


def _compare(amount: float, currency: str, rule: dict) -> dict:
    """Compare a payment against one threshold rule, recording any assumption."""
    same_currency = currency == rule["threshold_currency"]
    return {
        **rule,
        "payment_amount": amount,
        "payment_currency": currency,
        "compared_natively": same_currency,
        "assumption": None
        if same_currency
        else f"{currency} treated as 1:1 equivalent to {rule['threshold_currency']} (no FX data supplied)",
        "exceeds": amount > rule["threshold"],
    }


# --------------------------------------------------------------------------
# Tool 1: facts, thresholds, jurisdiction
# --------------------------------------------------------------------------

def assess_payment_policy(payment_id: str, call: ToolCall) -> dict:
    payment = call("get_payment", payment_id=payment_id)
    if not payment.get("found"):
        return {"found": False, "payment_id": payment_id, "error": payment.get("error")}

    client = call("get_client_profile", client_id=payment["client_id"])
    amount = float(payment["amount"])
    currency = str(payment["currency"]).upper()
    code = str(payment.get("beneficiary_country_code", "")).upper()
    country_label = str(payment.get("beneficiary_country", "") or "")
    client_country = str(client.get("country", "")) if client.get("found") else ""

    ctx = _policy_context(call, client_country)
    high_risk_codes = _parse_high_risk_codes(call("get_policy_document", source=HIGH_RISK_POLICY).get("text", ""))

    threshold_checks = [_compare(amount, currency, r) for r in ctx["rules"]]
    assumptions = sorted({c["assumption"] for c in threshold_checks if c["assumption"]})
    high_risk = code in high_risk_codes
    code_mismatch = bool(country_label) and _LABEL_FOR_CODE.get(code, code) != country_label.upper()

    requirements = []
    for check in threshold_checks:
        if check["exceeds"]:
            requirements.append(
                {
                    "requirement": check["requirement"],
                    "reason": f"{currency} {amount:,.0f} is above the {check['threshold_currency']} "
                    f"{check['threshold']:,.0f} {check['requirement']} threshold",
                    "source": check["source"],
                }
            )
    if high_risk:
        requirements.append(
            {
                "requirement": "additional review (high-risk destination)",
                "reason": f"beneficiary_country_code {code} is on the high-risk jurisdiction list",
                "source": HIGH_RISK_POLICY,
            }
        )
    requirements = _dedupe(requirements)

    citations = [GLOBAL_POLICY] + ([ctx["regional_file"]] if ctx["regional_file"] else [])
    if high_risk:
        citations.append(HIGH_RISK_POLICY)

    facts = {
        "payment_id": payment["payment_id"],
        "client_id": payment["client_id"],
        "beneficiary_name": payment["beneficiary_name"],
        "amount": amount,
        "currency": currency,
        "beneficiary_country": country_label,
        "beneficiary_country_code": code,
        "channel": payment.get("channel"),
        "payment_date": payment.get("payment_date"),
        "client_country": client_country or None,
        "client_risk_rating": client.get("risk_rating"),
        "client_type": client.get("client_type"),
        "applicable_policies": ctx["applicable"],
        "high_risk_destination": high_risk,
        "thresholds": [
            {
                "requirement": c["requirement"],
                "threshold": c["threshold"],
                "currency": c["threshold_currency"],
                "exceeds": c["exceeds"],
                "source": c["source"],
            }
            for c in threshold_checks
        ],
    }

    return {
        "found": True,
        "payment": payment,
        "client": client,
        "applicable_policies": ctx["applicable"],
        "regional_policy": ctx["regional_file"],
        "threshold_checks": threshold_checks,
        "high_risk": high_risk,
        "high_risk_codes": sorted(high_risk_codes),
        "country_code_mismatch": code_mismatch,
        "requirements": requirements,
        "outcome": _outcome(requirements),
        "assumptions": assumptions,
        "citations": citations,
        "facts": facts,
        "note": "Numbers and flags above are computed deterministically; use them as-is. "
        "Call check_structuring for the 24-hour same-beneficiary pattern.",
    }


# --------------------------------------------------------------------------
# Tool 2: 24-hour structuring window
# --------------------------------------------------------------------------

def check_structuring(client_id: str, beneficiary_name: str, call: ToolCall) -> dict:
    client = call("get_client_profile", client_id=client_id)
    ctx = _policy_context(call, str(client.get("country", "")) if client.get("found") else "")
    rule = ctx["structuring_rule"]

    window = call("aggregate_beneficiary_24h", client_id=client_id, beneficiary_name=beneficiary_name)
    count = int(window.get("count") or 0)
    total = float(window.get("total_amount") or 0.0)
    currencies = list(window.get("currencies", []))
    native = currencies == [rule["threshold_currency"]]
    flagged = count >= 2 and total > rule["threshold"]

    min_threshold = min((r["threshold"] for r in ctx["rules"]), default=float("inf"))
    each_below = count >= 2 and all(a <= min_threshold for a in window.get("individual_amounts", []))

    assumptions = list(window.get("assumptions", []))
    if count >= 2 and not native:
        assumptions.append(
            f"{'/'.join(currencies)} window total treated as 1:1 equivalent to "
            f"{rule['threshold_currency']} for the structuring check"
        )

    requirement = None
    source = GLOBAL_POLICY
    if flagged:
        escalate = ctx["regional_mentions_structuring"]
        source = ctx["regional_file"] if escalate else GLOBAL_POLICY
        requirement = {
            "requirement": "escalate potential structuring to Compliance" if escalate else "review for potential structuring",
            "reason": f"{count} payments to {beneficiary_name} on {window.get('window_date')} total "
            f"{'/'.join(currencies)} {total:,.0f}, above the {rule['threshold_currency']} "
            f"{rule['threshold']:,.0f} equivalent structuring threshold",
            "source": source,
        }

    return {
        "checked": True,
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "flagged": flagged,
        "count": count,
        "total_amount": total,
        "currencies": currencies,
        "window_date": window.get("window_date"),
        "payment_ids": window.get("payment_ids", []),
        "individual_amounts": window.get("individual_amounts", []),
        "channels": window.get("channels", []),
        "each_below_thresholds": each_below,
        "rule": rule,
        "requirement": requirement,
        "assumptions": assumptions,
        "citations": [source] if flagged else [],
        "all_windows": window.get("all_windows", []),
        "note": "A pattern is an observed fact, not proof of intent. "
        "A policy trigger does not by itself establish suspicious activity.",
    }


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------

def _dedupe(requirements: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for r in requirements:
        if r["requirement"] not in seen:
            seen.add(r["requirement"])
            unique.append(r)
    return unique


def _outcome(requirements: list[dict]) -> str:
    return ", ".join(r["requirement"] for r in requirements) or "no policy trigger - standard monitoring applies"


def merge_assessment(policy: dict, structuring: dict | None) -> dict:
    """One assessment object from whichever tools actually ran."""
    if not policy.get("found"):
        return policy
    merged = dict(policy)
    merged.pop("note", None)
    s = structuring or {"checked": False, "flagged": False, "count": 0, "total_amount": 0.0,
                        "currencies": [], "window_date": None, "payment_ids": [], "individual_amounts": [],
                        "channels": [], "each_below_thresholds": False,
                        "rule": {"threshold": 100000.0, "threshold_currency": "USD", "source": GLOBAL_POLICY},
                        "requirement": None, "assumptions": [], "citations": []}
    merged["structuring"] = {k: v for k, v in s.items() if k not in ("note", "all_windows")}

    requirements = list(policy["requirements"])
    if s.get("requirement"):
        requirements.append(s["requirement"])
    merged["requirements"] = _dedupe(requirements)
    merged["outcome"] = _outcome(merged["requirements"])
    merged["assumptions"] = list(dict.fromkeys(list(policy["assumptions"]) + list(s.get("assumptions", []))))
    merged["citations"] = list(dict.fromkeys(list(policy["citations"]) + list(s.get("citations", []))))

    facts = dict(policy["facts"])
    facts["structuring_window"] = {
        "checked": s.get("checked", False),
        "window_date": s.get("window_date"),
        "count": s.get("count", 0),
        "payment_ids": s.get("payment_ids", []),
        "individual_amounts": s.get("individual_amounts", []),
        "channels": s.get("channels", []),
        "combined_amount": s.get("total_amount", 0.0),
        "currency": "/".join(s.get("currencies", [])) or None,
        "threshold": s["rule"]["threshold"],
        "threshold_currency": s["rule"]["threshold_currency"],
        "flagged": s.get("flagged", False),
    }
    merged["facts"] = facts
    return merged


def assess_payment(payment_id: str, call: ToolCall) -> dict:
    """Full deterministic investigation (used by the no-model narrator)."""
    policy = assess_payment_policy(payment_id, call)
    if not policy.get("found"):
        return policy
    structuring = check_structuring(policy["payment"]["client_id"], policy["payment"]["beneficiary_name"], call)
    return merge_assessment(policy, structuring)
