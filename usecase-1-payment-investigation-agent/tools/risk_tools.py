"""Deterministic policy / risk evaluation (policy-as-code)."""

from __future__ import annotations

from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_payment


HIGH_RISK_CODES = {"AE"}

# Policy thresholds (native currencies per DATA_NOTES).
GLOBAL_ENHANCED_USD = 100_000
GLOBAL_STRUCTURING_USD = 100_000
SG_RM_USD = 75_000
SG_ENHANCED_USD = 100_000
CH_RM_CHF = 80_000
CH_ENHANCED_CHF = 120_000

FX_ASSUMPTION = (
    "No FX rates provided; treat 'equivalent' as 1:1 when comparing "
    "non-matching currencies to USD/CHF thresholds."
)


def _amount_vs_threshold(amount: float, currency: str, threshold: float, threshold_ccy: str) -> dict:
    matched = currency == threshold_ccy
    comparable = float(amount)  # 1:1 when currencies differ
    return {
        "amount": amount,
        "currency": currency,
        "threshold": threshold,
        "threshold_currency": threshold_ccy,
        "native_currency_match": matched,
        "comparable_amount": comparable,
        "exceeds": comparable > threshold,
        "fx_assumption_applied": not matched,
    }


def evaluate_payment_risk(
    payment_id: str,
    include_structuring: bool = True,
) -> dict:
    """Evaluate thresholds, high-risk destination, and optional structuring.

    All arithmetic and policy triggers are deterministic. The LLM should
    narrate this pack rather than recompute thresholds.
    """
    if not payment_id:
        return {"error": "payment_id is required"}

    payment = get_payment(payment_id)
    if payment.get("error"):
        return payment

    client_id = payment.get("client_id")
    client = get_client_profile(client_id) if client_id else {"error": "missing client_id"}
    if client.get("error"):
        return {"error": client["error"], "payment": payment}

    amount = float(payment.get("amount") or 0)
    currency = payment.get("currency") or "UNKNOWN"
    code = payment.get("beneficiary_country_code")
    country_name = payment.get("beneficiary_country")
    client_country = client.get("country")
    regional_policy = client.get("regional_policy")

    country_code_mismatch = bool(
        country_name
        and code
        and str(country_name).strip().casefold() != str(code).strip().casefold()
        and not (
            # Allow common name/code pairs that are not contradictions for display;
            # mismatch flag is for clear contradictions like Hong Kong vs AE.
            (str(country_name).casefold() in {"uae", "united arab emirates"} and code == "AE")
            or (str(country_name).casefold() in {"switzerland"} and code == "CH")
            or (str(country_name).casefold() in {"singapore"} and code == "SG")
            or (str(country_name).casefold() in {"hong kong"} and code == "HK")
            or (str(country_name).casefold() in {"uk", "united kingdom"} and code == "GB")
        )
    )
    # Explicit trap from DATA_NOTES: Hong Kong name with AE code.
    if (
        str(country_name or "").casefold() in {"hong kong"}
        and code == "AE"
    ):
        country_code_mismatch = True

    high_risk_destination = code in HIGH_RISK_CODES

    checks: dict = {
        "global_enhanced": _amount_vs_threshold(
            amount, currency, GLOBAL_ENHANCED_USD, "USD"
        ),
    }

    triggers: list[str] = []
    actions: list[str] = []
    required_citations: list[str] = ["global_payment_policy.md"]

    if checks["global_enhanced"]["exceeds"]:
        triggers.append(
            f"Amount exceeds global enhanced-review threshold "
            f"(USD {GLOBAL_ENHANCED_USD:,})."
        )
        actions.append("enhanced_review")

    if client_country == "Singapore":
        checks["singapore_rm"] = _amount_vs_threshold(
            amount, currency, SG_RM_USD, "USD"
        )
        checks["singapore_enhanced"] = _amount_vs_threshold(
            amount, currency, SG_ENHANCED_USD, "USD"
        )
        required_citations.append("regional_singapore.md")
        if checks["singapore_rm"]["exceeds"]:
            triggers.append(
                f"Amount exceeds Singapore RM-review threshold (USD {SG_RM_USD:,})."
            )
            if "rm_review" not in actions:
                actions.append("rm_review")
        if checks["singapore_enhanced"]["exceeds"]:
            triggers.append(
                f"Amount exceeds Singapore enhanced-review threshold "
                f"(USD {SG_ENHANCED_USD:,})."
            )
            if "enhanced_review" not in actions:
                actions.append("enhanced_review")

    elif client_country == "Switzerland":
        checks["switzerland_rm"] = _amount_vs_threshold(
            amount, currency, CH_RM_CHF, "CHF"
        )
        checks["switzerland_enhanced"] = _amount_vs_threshold(
            amount, currency, CH_ENHANCED_CHF, "CHF"
        )
        required_citations.append("regional_switzerland.md")
        if checks["switzerland_rm"]["exceeds"]:
            triggers.append(
                f"Amount exceeds Switzerland RM-review threshold "
                f"(CHF {CH_RM_CHF:,})."
            )
            if "rm_review" not in actions:
                actions.append("rm_review")
        if checks["switzerland_enhanced"]["exceeds"]:
            triggers.append(
                f"Amount exceeds Switzerland enhanced-review threshold "
                f"(CHF {CH_ENHANCED_CHF:,})."
            )
            if "enhanced_review" not in actions:
                actions.append("enhanced_review")

    if high_risk_destination:
        triggers.append(
            f"Destination code {code} is high-risk; additional review required."
        )
        actions.append("additional_review")
        if "high_risk_jurisdictions.md" not in required_citations:
            required_citations.append("high_risk_jurisdictions.md")

    structuring = None
    if include_structuring:
        window = aggregate_beneficiary_24h(
            client_id=client_id,
            beneficiary_name=payment.get("beneficiary_name") or "",
            payment_id=payment_id,
            payment_date=payment.get("payment_date"),
        )
        struct_total = float(window.get("total_amount") or 0)
        struct_ccy = window.get("currency") or currency
        struct_count = int(window.get("count") or 0)
        comparable = struct_total  # 1:1 vs USD structuring threshold
        structuring_flag = struct_count >= 2 and comparable > GLOBAL_STRUCTURING_USD

        structuring = {
            "checked": True,
            "client_id": client_id,
            "beneficiary_name": payment.get("beneficiary_name"),
            "date": window.get("date"),
            "count": struct_count,
            "total_amount": struct_total,
            "currency": struct_ccy,
            "payment_ids": window.get("payment_ids")
            or [p.get("payment_id") for p in window.get("payments") or []],
            "channels": window.get("channels")
            or [p.get("channel") for p in window.get("payments") or []],
            "individual_amounts": [
                p.get("amount") for p in window.get("payments") or []
            ],
            "threshold": GLOBAL_STRUCTURING_USD,
            "threshold_currency": "USD",
            "fx_assumption": FX_ASSUMPTION,
            "possible_structuring": structuring_flag,
            "assumption": window.get("assumption"),
        }

        if structuring_flag:
            triggers.append(
                f"Possible structuring: {struct_count} payments to the same "
                f"beneficiary on {window.get('date')} total {struct_total} "
                f"{struct_ccy}, exceeding USD {GLOBAL_STRUCTURING_USD:,} equivalent."
            )
            if client_country == "Switzerland":
                actions.append("escalate_compliance")
                if "regional_switzerland.md" not in required_citations:
                    required_citations.append("regional_switzerland.md")
            else:
                actions.append("structuring_review")

    if not actions:
        actions.append("standard_monitoring")

    # Deduplicate while preserving order
    seen_actions = set()
    actions = [a for a in actions if not (a in seen_actions or seen_actions.add(a))]
    seen_cites = set()
    required_citations = [
        c for c in required_citations if not (c in seen_cites or seen_cites.add(c))
    ]

    fx_used = any(
        check.get("fx_assumption_applied")
        for check in checks.values()
        if isinstance(check, dict)
    )
    if structuring and structuring.get("currency") not in (None, "USD") and structuring.get(
        "possible_structuring"
    ):
        fx_used = True

    return {
        "payment_id": payment_id,
        "payment": {
            "payment_id": payment.get("payment_id"),
            "client_id": client_id,
            "beneficiary_name": payment.get("beneficiary_name"),
            "beneficiary_country": country_name,
            "beneficiary_country_code": code,
            "amount": amount,
            "currency": currency,
            "channel": payment.get("channel"),
            "payment_date": payment.get("payment_date"),
        },
        "client": {
            "client_id": client_id,
            "country": client_country,
            "risk_rating": client.get("risk_rating"),
            "client_type": client.get("client_type"),
            "relationship_years": client.get("relationship_years"),
            "regional_policy": regional_policy,
            "global_policy": "global_payment_policy.md",
        },
        "jurisdiction": {
            "beneficiary_country_code": code,
            "beneficiary_country": country_name,
            "high_risk": high_risk_destination,
            "country_code_mismatch": country_code_mismatch,
            "authoritative_field": "beneficiary_country_code",
        },
        "threshold_checks": checks,
        "structuring": structuring,
        "policy_triggers": triggers,
        "recommended_actions": actions,
        "required_citations": required_citations,
        "fx_assumption": FX_ASSUMPTION if fx_used else None,
        "notes": [
            "A policy trigger is not proof of suspicious activity or intent.",
            "Use beneficiary_country_code for jurisdiction risk assessment.",
        ],
    }


def get_investigation_workflow() -> dict:
    """Return the official investigation workflow steps."""
    return {
        "source": "investigation_procedure.md",
        "steps": [
            "Establish client and payment facts.",
            "Identify the applicable policy.",
            "Check high-risk destination indicators.",
            "Check for possible transaction splitting.",
            "Separate observed facts from assumptions.",
            "Record the evidence supporting the recommendation.",
        ],
    }
