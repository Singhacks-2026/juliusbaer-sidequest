"""
Deterministic policy-threshold evaluation.

AI_ARCHITECTURE_REQUIREMENTS.md #4 requires that amounts/thresholds be
compared in Python/tool logic, not left to the LLM. This module parses the
actual threshold numbers out of the retrieved policy documents (rather than
hard-coding them) and does the amount/currency comparison deterministically.

Parsing the real corpus text -- instead of encoding "100000" as a Python
constant -- means this still reflects whatever the policy documents actually
say, and RAG retrieval is what supplies the evidence/citations.
"""

import re

_REVIEW_THRESHOLD_RE = re.compile(
    r"above\s+(USD|CHF)\s*([\d,]+)\s*equivalent\s+require[s]?\s+"
    r"([A-Za-z][A-Za-z \-]*?review)",
    re.IGNORECASE,
)

_STRUCTURING_THRESHOLD_RE = re.compile(
    r"structuring[^.]*?exceeds\s+(USD|CHF)\s*([\d,]+)\s*equivalent"
    r"|exceeds\s+(USD|CHF)\s*([\d,]+)\s*equivalent[^.]*?structuring",
    re.IGNORECASE | re.DOTALL,
)

_HIGH_RISK_CODE_RE = re.compile(r"\b([A-Z]{2})\b\s*\([^)]*\)\s+is a high-risk destination")

# Client country -> regional policy filename, matching DATA_NOTES.md.
_REGIONAL_POLICY_BY_COUNTRY = {
    "Singapore": "regional_singapore.md",
    "Switzerland": "regional_switzerland.md",
}


def extract_review_thresholds(documents: list[dict]) -> list[dict]:
    """
    Parse "payments above <CURRENCY> <AMOUNT> equivalent require <TYPE>
    review" rules out of the policy documents.

    Returns a list of {source, currency, amount, review_type}.
    """
    thresholds = []
    for doc in documents:
        for match in _REVIEW_THRESHOLD_RE.finditer(doc["text"]):
            currency, amount_str, review_type = match.groups()
            thresholds.append(
                {
                    "source": doc["source"],
                    "currency": currency.upper(),
                    "amount": float(amount_str.replace(",", "")),
                    "review_type": review_type.strip().lower(),
                }
            )
    return thresholds


def extract_structuring_threshold(documents: list[dict]) -> dict | None:
    """
    Parse the combined-value structuring threshold, e.g. "... exceeds USD
    100,000 equivalent" in the context of structuring.
    """
    for doc in documents:
        match = _STRUCTURING_THRESHOLD_RE.search(doc["text"])
        if match:
            groups = [g for g in match.groups() if g]
            currency, amount_str = groups[0], groups[1]
            return {
                "source": doc["source"],
                "currency": currency.upper(),
                "amount": float(amount_str.replace(",", "")),
            }
    return None


def extract_high_risk_codes(documents: list[dict]) -> dict[str, str]:
    """
    Parse high-risk jurisdiction country codes, e.g. "AE (UAE) is a
    high-risk destination". Returns {code: source}.
    """
    codes = {}
    for doc in documents:
        for match in _HIGH_RISK_CODE_RE.finditer(doc["text"]):
            codes[match.group(1)] = doc["source"]
    return codes


def evaluate_review_requirements(
    amount: float,
    currency: str,
    beneficiary_country_code: str,
    client_country: str,
    documents: list[dict],
) -> dict:
    """
    Deterministically evaluate which review requirements a payment triggers,
    using thresholds parsed from the actual policy documents.

    No exchange-rate data is available (see DATA_NOTES.md): amounts in a
    currency other than the threshold's stated currency are compared 1:1,
    and that assumption is flagged in the result.
    """
    review_thresholds = extract_review_thresholds(documents)
    high_risk_codes = extract_high_risk_codes(documents)

    global_thresholds = [t for t in review_thresholds if t["source"] == "global_payment_policy.md"]
    regional_policy_file = _REGIONAL_POLICY_BY_COUNTRY.get(client_country)
    regional_thresholds = (
        [t for t in review_thresholds if t["source"] == regional_policy_file]
        if regional_policy_file
        else []
    )

    def _triggered(threshold: dict) -> tuple[bool, bool]:
        """Returns (triggered, used_currency_assumption)."""
        if currency == threshold["currency"]:
            return amount >= threshold["amount"], False
        return amount >= threshold["amount"], True

    triggered_global = []
    for t in global_thresholds:
        is_triggered, assumed = _triggered(t)
        if is_triggered:
            triggered_global.append({**t, "currency_assumption_applied": assumed})

    triggered_regional = []
    for t in regional_thresholds:
        is_triggered, assumed = _triggered(t)
        if is_triggered:
            triggered_regional.append({**t, "currency_assumption_applied": assumed})

    high_risk_source = high_risk_codes.get(beneficiary_country_code)

    return {
        "high_risk_destination": high_risk_source is not None,
        "high_risk_source": high_risk_source,
        "regional_policy": regional_policy_file,
        "triggered_global_reviews": triggered_global,
        "triggered_regional_reviews": triggered_regional,
        "any_currency_assumption_applied": any(
            r["currency_assumption_applied"]
            for r in triggered_global + triggered_regional
        ),
    }
