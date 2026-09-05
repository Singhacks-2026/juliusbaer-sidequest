"""Evaluate retrieved policy rules with exact arithmetic, not LLM arithmetic."""
import re
from decimal import Decimal

from tools.client_tools import get_client_profile
from tools.payment_tools import get_payment, get_client_payments, aggregate_beneficiary_24h
from tools.policy_tools import search_policy


def assess_payment(payment_id: str, include_history: bool = False) -> dict:
    """Retrieve applicable evidence and compute layered thresholds and risk.

    Set include_history for splitting, escalation, or full investigation questions.
    The returned supporting_tools records actual nested invocations.
    """
    used = ["get_payment"]
    payment = get_payment(payment_id)
    if "error" in payment:
        return {**payment, "supporting_tools": used}
    used.append("get_client_profile")
    client = get_client_profile(payment["client_id"])
    country = client.get("country")
    queries = ["global payment monitoring enhanced review structuring",
               "high risk jurisdiction destination list"]
    regional = {"Singapore": "regional_singapore.md", "Switzerland": "regional_switzerland.md"}
    if country in regional:
        queries.append(f"{country} payment procedure RM enhanced review")
    sources = {"global_payment_policy.md", "high_risk_jurisdictions.md"}
    if country in regional:
        sources.add(regional[country])
    evidence = {}
    used.append("search_policy")
    for query in queries:
        for chunk in search_policy(query):
            if chunk["source"] in sources:
                evidence[chunk["chunk_id"]] = chunk
    checks = []
    assumptions = []
    amount = Decimal(str(payment["amount"]))
    if not amount.is_finite() or amount < 0:
        return {"error": "Invalid payment amount", "supporting_tools": used}
    for chunk in evidence.values():
        for currency, threshold, requirement in re.findall(
            r"Payments above (USD|CHF) ([\d,]+) equivalent require ([^.\n]+)", chunk["text"]
        ):
            limit = Decimal(threshold.replace(",", ""))
            fx = payment["currency"] != currency
            if fx:
                assumptions.append(f"No FX rates supplied: {payment['currency']} to {currency} "
                                   "equivalent is assumed 1:1 for this exercise.")
            checks.append({"source": chunk["source"], "threshold": float(limit),
                           "threshold_currency": currency, "operator": ">",
                           "amount_compared": float(amount), "triggered": amount > limit,
                           "requirement": requirement, "fx_assumed_1_to_1": fx})
    risk_chunks = [c for c in evidence.values() if c["source"] == "high_risk_jurisdictions.md"]
    risk_codes = set(code for c in risk_chunks for code in re.findall(
        r"\b([A-Z]{2})\s*\([^)]*\) is a high-risk destination", c["text"]))
    result = {"payment": payment, "client": client, "threshold_checks": checks,
              "high_risk_destination": payment["beneficiary_country_code"] in risk_codes if risk_codes else None,
              "authoritative_country_field": "beneficiary_country_code",
              "assumptions": assumptions, "policy_evidence": list(evidence.values()),
              "supporting_tools": used}
    if include_history:
        used.append("get_client_payments")
        history = get_client_payments(payment["client_id"])
        names = sorted({p["beneficiary_name"] for p in history if p["beneficiary_name"]})
        analyses = []
        structuring_limits = [Decimal(v.replace(",", "")) for c in evidence.values()
                             if c["source"] == "global_payment_policy.md"
                             for v in re.findall(r"combined value exceeds\s+USD ([\d,]+)", c["text"])]
        for name in names:
            if "aggregate_beneficiary_24h" not in used:
                used.append("aggregate_beneficiary_24h")
            analysis = aggregate_beneficiary_24h(payment["client_id"], name)
            for window in analysis["windows"]:
                window["potential_structuring"] = (
                    window["count"] > 1 and Decimal(str(window["total_usd_equivalent"])) > structuring_limits[0]
                ) if structuring_limits else None
            analyses.append(analysis)
        result["beneficiary_analysis"] = analyses
        result["history_count"] = len(history)
        result["structuring_threshold_usd"] = float(structuring_limits[0]) if structuring_limits else None
    return result
