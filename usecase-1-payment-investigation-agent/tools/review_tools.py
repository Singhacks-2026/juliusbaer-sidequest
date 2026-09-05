"""Apply the supplied exercise policies with deterministic comparisons.

This parser deliberately supports the rule wording in this challenge corpus.
It is not a parser for arbitrary bank policies or a source of live FX rates.
"""

import re
from collections import defaultdict
from decimal import Decimal

from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_client_payments, get_payment
from tools.policy_tools import get_policy_document


def parse_policy(document: dict, client_country: str) -> dict:
    text = document["text"]
    title = text.splitlines()[0].casefold()
    regional = "payment procedure" in title and "investigation" not in title
    applies = not regional or client_country.casefold() in title
    rules = []
    # Join continuation lines before reading each bullet as a complete rule.
    for paragraph in re.split(r"\n\s*[-*] ", text):
        paragraph = " ".join(paragraph.split())
        match = re.search(r"\b(?:above|exceeds)\s+([A-Z]{3})\s+([\d,]+(?:\.\d+)?)", paragraph)
        if not match or not applies:
            continue
        kind = "structuring" if "multiple payments" in paragraph.casefold() else "payment_amount"
        action = re.search(r"require\s+(.+?)(?:\s+before release|\.|$)", paragraph, re.I)
        rules.append({
            "source": document["source"], "kind": kind,
            "threshold_currency": match.group(1),
            "threshold": float(Decimal(match.group(2).replace(",", ""))),
            "operator": ">", "action": action.group(1) if action else "review for potential structuring",
            "policy_text": paragraph,
        })
    codes = []
    if "jurisdiction" in title:
        codes = re.findall(r"\b([A-Z]{2})\s*(?:\([^)]*\))?\s+is a high-risk destination", text)
    return {"rules": rules, "high_risk_codes": codes, "applicable": applies,
            "global": "global payment" in title,
            "regional": regional and applies,
            "jurisdiction_list": "jurisdiction" in title,
            "escalate_structuring": applies and bool(re.search(
                r"potential structuring should be escalated to compliance", text, re.I)),
            "source": document["source"]}


def compare_amount(amount: float, currency: str, rule: dict) -> dict:
    value = Decimal(str(amount))
    threshold = Decimal(str(rule["threshold"]))
    if not value.is_finite() or not threshold.is_finite():
        raise ValueError("Amounts and thresholds must be finite")
    assumption = None
    if currency != rule["threshold_currency"]:
        assumption = (f"No FX data is supplied. Per DATA_NOTES.md, {currency} is treated as "
                      f"1:1 with {rule['threshold_currency']} for this exercise comparison only.")
    return {**rule, "amount": float(value), "currency": currency,
            "triggered": value > threshold, "currency_assumption": assumption}


def evaluate_payment(payment_id: str, policy_sources: list[str],
                     check_structuring: bool = False) -> dict:
    """Compare trusted CSV values with rules parsed from discovered policies."""
    payment = get_payment(payment_id)
    if not payment:
        return {"error": f"Payment {payment_id} does not exist in the supplied data."}
    client = get_client_profile(payment["client_id"])
    if not client:
        return {"error": "The payment has no corresponding supplied client profile."}
    parsed = []
    for source in dict.fromkeys(policy_sources):
        document = get_policy_document(source)
        if not document:
            return {"error": f"Unknown policy source: {source}"}
        parsed.append(parse_policy(document, client["country"]))
    rules = [rule for item in parsed for rule in item["rules"]]
    checks = [compare_amount(payment["amount"], payment["currency"], rule)
              for rule in rules if rule["kind"] == "payment_amount"]
    has_risk_list = any(item["jurisdiction_list"] for item in parsed)
    codes = sorted({code for item in parsed for code in item["high_risk_codes"]})
    high_risk = payment["beneficiary_country_code"] in codes if has_risk_list else None
    missing = []
    if not any(item["global"] for item in parsed):
        missing.append("Retrieve the global payment policy.")
    if client["country"] in {"Singapore", "Switzerland"} and not any(item["regional"] for item in parsed):
        missing.append(f"Retrieve the {client['country']} payment procedure.")
    if not has_risk_list:
        missing.append("Retrieve the high-risk jurisdiction list before deciding destination risk.")
    pattern_checks, assumptions = [], []
    if check_structuring:
        history = get_client_payments(client["client_id"])
        for beneficiary in sorted({row["beneficiary_name"] for row in history}):
            aggregation = aggregate_beneficiary_24h(client["client_id"], beneficiary)
            assumptions.extend(aggregation["assumptions"])
            days = defaultdict(list)
            for window in aggregation["windows"]:
                days[window["payment_date"]].append(window)
            for day, windows in sorted(days.items()):
                count = sum(window["count"] for window in windows)
                if count < 2:
                    continue
                totals = {window["currency"]: window["total_amount"] for window in windows}
                for rule in rules:
                    if rule["kind"] != "structuring":
                        continue
                    total = sum((Decimal(str(amount)) for amount in totals.values()), Decimal(0))
                    currency = next(iter(totals)) if len(totals) == 1 else rule["threshold_currency"]
                    comparison = compare_amount(float(total), currency, rule)
                    if len(totals) > 1:
                        comparison["currency_assumption"] = (
                            "Mixed currencies are retained in totals_by_currency. For the threshold "
                            "comparison only, each is treated as 1:1 with " + currency +
                            " under DATA_NOTES.md; no actual FX conversion is available."
                        )
                    rows = [row for window in windows for row in window["payments"]]
                    pattern_checks.append({
                        **comparison, "client_id": client["client_id"],
                        "beneficiary_name": beneficiary, "payment_date": day, "count": count,
                        "totals_by_currency": totals,
                        "comparison_amount": float(total), "comparison_currency": rule["threshold_currency"],
                        "payment_ids": [row["payment_id"] for row in rows],
                        "individual_amounts": [row["amount"] for row in rows],
                        "channels": sorted({row["channel"] for row in rows}),
                    })
        if not any(rule["kind"] == "structuring" for rule in rules):
            missing.append("Retrieve the global structuring rule before deciding whether the pattern triggers review.")
    assumptions.extend(check["currency_assumption"] for check in checks + pattern_checks
                       if check["currency_assumption"])
    return {
        "payment": payment, "client": client, "threshold_checks": checks,
        "destination_risk": {"authoritative_field": "beneficiary_country_code",
                             "code": payment["beneficiary_country_code"], "high_risk": high_risk,
                             "high_risk_codes": codes,
                             "sources": [item["source"] for item in parsed if item["jurisdiction_list"]]},
        "structuring_checked": check_structuring, "structuring_checks": pattern_checks,
        "potential_structuring": any(check["triggered"] for check in pattern_checks) if check_structuring else None,
        "escalate_potential_structuring_to_compliance": any(item["escalate_structuring"] for item in parsed),
        "policy_sources": [item["source"] for item in parsed if item["applicable"]],
        "missing_policy_evidence": missing, "assumptions": list(dict.fromkeys(assumptions)),
        "interpretation_limit": "A policy trigger or observed pattern does not establish suspicious intent.",
    }
