"""
Deterministic threshold evaluation.

The policy corpus states thresholds in prose; this module encodes the same
thresholds as data and performs the comparisons in Python, so the LLM never
does arithmetic.  Each threshold carries the document it comes from, so the
agent can cite the evidence behind every trigger.

Region mapping (see DATA_NOTES.md): the client's ``country`` selects the
regional procedure that applies *on top of* the global policy.  Countries
without a regional procedure use the global policy only.

Currency handling (see DATA_NOTES.md): no exchange-rate data is supplied.
Amounts are compared in native currency when the threshold is stated in the
payment's currency, otherwise "equivalent" is treated as 1:1 — an assumption
this tool reports explicitly in ``currency_assumption``.
"""

import os
import re

from langchain_core.tools import tool

from tools.client_tools import get_client_profile
from tools.payment_tools import get_payment

_POLICY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "policies",
)

GLOBAL_POLICY = "global_payment_policy.md"
HIGH_RISK_POLICY = "high_risk_jurisdictions.md"

# Thresholds as stated in the policy corpus, each tagged with its source.
GLOBAL_THRESHOLDS = {
    "enhanced_review": {
        "amount": 100_000,
        "currency": "USD",
        "source": GLOBAL_POLICY,
    },
    "structuring_review": {
        "amount": 100_000,
        "currency": "USD",
        "source": GLOBAL_POLICY,
    },
}

REGIONAL_THRESHOLDS = {
    "Singapore": {
        "source": "regional_singapore.md",
        "rm_review": {"amount": 75_000, "currency": "USD"},
        "enhanced_review": {"amount": 100_000, "currency": "USD"},
    },
    "Switzerland": {
        "source": "regional_switzerland.md",
        "rm_review": {"amount": 80_000, "currency": "CHF"},
        "enhanced_review": {"amount": 120_000, "currency": "CHF"},
    },
}

_high_risk_codes = None


def get_high_risk_codes() -> list[str]:
    """Read the high-risk jurisdiction codes from the policy document.

    The list is parsed from ``high_risk_jurisdictions.md`` rather than
    hard-coded, so the assessment stays grounded in the corpus.
    """
    global _high_risk_codes
    if _high_risk_codes is None:
        path = os.path.join(_POLICY_DIR, HIGH_RISK_POLICY)
        try:
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            codes = sorted(set(re.findall(r"\b[A-Z]{2}\b", text)))
        except OSError:
            codes = []
        _high_risk_codes = codes
    return list(_high_risk_codes)


def _compare(amount: float, currency: str, threshold: dict) -> dict:
    """Compare an amount against one threshold, recording the assumption used."""
    same_currency = currency == threshold["currency"]

    if same_currency:
        basis = "native currency"
    elif currency == "mixed":
        basis = (
            "combined total spans more than one currency; summed at 1:1 "
            f"equivalence and compared against the {threshold['currency']} threshold"
        )
    else:
        basis = f"1:1 equivalence assumed between {currency} and {threshold['currency']}"

    return {
        "threshold_amount": threshold["amount"],
        "threshold_currency": threshold["currency"],
        "payment_amount": amount,
        "payment_currency": currency,
        "exceeds": amount > threshold["amount"],
        "comparison_basis": basis,
    }


@tool
def evaluate_payment_thresholds(payment_id: str) -> dict:
    """Evaluate every applicable policy threshold for one payment.

    Resolves the client's region, applies the global policy plus any regional
    procedure, checks the destination against the high-risk jurisdiction list
    (using the authoritative ``beneficiary_country_code``), and returns the
    resulting triggers together with the documents that support them.
    """
    payment = get_payment.invoke({"payment_id": payment_id})
    if "error" in payment:
        return {"error": payment["error"]}

    client = get_client_profile.invoke({"client_id": payment["client_id"]})
    client_country = client.get("country") if "error" not in client else None

    amount = float(payment["amount"])
    currency = str(payment["currency"])
    country_code = str(payment["beneficiary_country_code"])

    regional = REGIONAL_THRESHOLDS.get(client_country)
    sources = [GLOBAL_POLICY]

    checks: dict[str, dict] = {
        "global_enhanced_review": _compare(
            amount, currency, GLOBAL_THRESHOLDS["enhanced_review"]
        )
    }

    if regional:
        sources.append(regional["source"])
        checks["regional_rm_review"] = _compare(
            amount, currency, {**regional["rm_review"], "source": regional["source"]}
        )
        checks["regional_enhanced_review"] = _compare(
            amount,
            currency,
            {**regional["enhanced_review"], "source": regional["source"]},
        )

    high_risk_codes = get_high_risk_codes()
    is_high_risk = country_code in high_risk_codes
    if is_high_risk:
        sources.append(HIGH_RISK_POLICY)

    enhanced_required = checks["global_enhanced_review"]["exceeds"] or checks.get(
        "regional_enhanced_review", {}
    ).get("exceeds", False)
    rm_review_required = checks.get("regional_rm_review", {}).get("exceeds", False)

    triggers = []
    if enhanced_required:
        triggers.append("amount exceeds the applicable enhanced-review threshold")
    if rm_review_required:
        triggers.append("amount exceeds the regional RM-review threshold")
    if is_high_risk:
        triggers.append(
            f"destination {country_code} is on the high-risk jurisdiction list"
        )

    return {
        "payment_id": payment["payment_id"],
        "client_id": payment["client_id"],
        "client_country": client_country,
        "applicable_regional_policy": regional["source"] if regional else None,
        "amount": amount,
        "currency": currency,
        "beneficiary_country_code": country_code,
        "beneficiary_country_name": payment.get("beneficiary_country"),
        "risk_assessment_field": (
            "beneficiary_country_code is authoritative for jurisdiction risk; "
            "the beneficiary_country name field may disagree with it"
        ),
        "high_risk_jurisdiction_list": high_risk_codes,
        "is_high_risk_destination": is_high_risk,
        "threshold_checks": checks,
        "enhanced_review_required": enhanced_required,
        "rm_review_required": rm_review_required,
        "additional_review_required": is_high_risk,
        "policy_triggers": triggers,
        "any_trigger": bool(triggers),
        "currency_assumption": (
            "No exchange-rate data is provided; where the payment currency differs "
            "from the threshold currency, 1:1 equivalence is assumed."
        ),
        "supporting_sources": sources,
    }


@tool
def evaluate_structuring(client_id: str, beneficiary_name: str) -> dict:
    """Check a client's same-beneficiary 24h aggregate against the structuring
    threshold defined in the global policy.

    Returns the deterministic comparison only.  Whether the pattern reflects
    intent to evade is an assumption for the investigator, not a fact this
    tool can establish.
    """
    from tools.payment_tools import aggregate_beneficiary_24h

    aggregate = aggregate_beneficiary_24h.invoke(
        {"client_id": client_id, "beneficiary_name": beneficiary_name}
    )
    if "error" in aggregate:
        return aggregate

    threshold = GLOBAL_THRESHOLDS["structuring_review"]
    client = get_client_profile.invoke({"client_id": client_id})
    regional = REGIONAL_THRESHOLDS.get(
        client.get("country") if "error" not in client else None
    )

    comparison = _compare(
        aggregate["total_amount"],
        aggregate["currency"] or threshold["currency"],
        threshold,
    )

    sources = [GLOBAL_POLICY]
    if regional:
        sources.append(regional["source"])

    return {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "window_date": aggregate["window_date"],
        "window_assumption": aggregate["window_assumption"],
        "payment_count_in_window": aggregate["count"],
        "payment_ids_in_window": [p["payment_id"] for p in aggregate["payments"]],
        "combined_amount": aggregate["total_amount"],
        "combined_currency": aggregate["currency"],
        "totals_by_currency": aggregate["totals_by_currency"],
        "threshold_comparison": comparison,
        "multiple_payments_in_window": aggregate["count"] > 1,
        "structuring_review_triggered": (
            aggregate["count"] > 1 and comparison["exceeds"]
        ),
        "escalation_to_compliance_required": bool(
            regional
            and regional["source"] == "regional_switzerland.md"
            and aggregate["count"] > 1
            and comparison["exceeds"]
        ),
        "currency_assumption": comparison["comparison_basis"],
        "supporting_sources": sources,
    }
