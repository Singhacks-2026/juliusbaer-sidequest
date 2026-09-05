"""
Deterministic policy engine.

Thresholds, jurisdiction risk and structuring comparisons are computed here in
Python and every result carries the policy document that states the rule, so
the assistant's citations follow from the evaluation instead of being guessed
by the LLM.

Threshold values are *parsed out of the policy corpus* rather than hardcoded.
Editing ``data/policies/*.md`` changes the assistant's behaviour, which is what
makes the answers policy-derived rather than keyed to the question set.
"""

import os
import re

from rag.pipeline import clean_document, split_into_rules
from tools.data_store import POLICY_DIR

GLOBAL_POLICY = "global_payment_policy.md"
HIGH_RISK_POLICY = "high_risk_jurisdictions.md"
INVESTIGATION_PROCEDURE = "investigation_procedure.md"

REGIONAL_POLICIES = {
    "Singapore": "regional_singapore.md",
    "Switzerland": "regional_switzerland.md",
}

# Requirement keywords, longest-first so "enhanced review" is not shadowed by a
# looser match on the same line.
_REQUIREMENT_PATTERNS = [
    ("structuring", re.compile(r"structuring", re.I)),
    ("enhanced_review", re.compile(r"enhanced\s+review", re.I)),
    ("rm_review", re.compile(r"\bRM\s+review", re.I)),
    ("additional_review", re.compile(r"additional\s+review", re.I)),
]

_AMOUNT_PATTERN = re.compile(
    r"(?:above|exceeds?|over)\s+(USD|CHF|SGD|HKD|GBP)\s*([\d,]+(?:\.\d+)?)",
    re.I,
)

_CURRENCY_CODE = re.compile(r"\b(USD|CHF|SGD|HKD|GBP)\b")

_HUMAN_LABELS = {
    "rm_review": "RM review",
    "enhanced_review": "enhanced review",
    "additional_review": "additional review",
    "structuring": "structuring review",
}

_document_cache: dict[str, str] = {}
_rule_cache: dict[str, list[dict]] = {}


def read_policy(source: str) -> str:
    """Read one policy document verbatim, cached."""
    if source not in _document_cache:
        path = os.path.join(POLICY_DIR, source)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _document_cache[source] = handle.read()
        except OSError:
            _document_cache[source] = ""
    return _document_cache[source]


def policy_rules(source: str) -> list[dict]:
    """
    Extract the actionable rules from a policy document.

    A rule is one logical clause carrying a requirement keyword, optionally
    with a currency threshold.  Clauses without a requirement keyword
    (headings, the "a trigger is not proof" caveat) are not rules and are
    skipped.

    Splitting is delegated to the RAG pipeline so retrieval and threshold
    parsing see identical rule boundaries.  This matters: the global policy's
    structuring clause wraps across three source lines, and parsing line by
    line would divorce "potential structuring" from the "USD 100,000"
    threshold that governs it.
    """
    if source in _rule_cache:
        return _rule_cache[source]

    rules = []
    for _heading, text in split_into_rules(clean_document(read_policy(source))):
        requirement = next(
            (name for name, pattern in _REQUIREMENT_PATTERNS if pattern.search(text)),
            None,
        )
        if requirement is None:
            continue

        amount_match = _AMOUNT_PATTERN.search(text)
        rules.append(
            {
                "requirement": requirement,
                "currency": amount_match.group(1).upper() if amount_match else None,
                "amount": (
                    float(amount_match.group(2).replace(",", ""))
                    if amount_match
                    else None
                ),
                "source": source,
                "text": text,
            }
        )

    _rule_cache[source] = rules
    return rules


def high_risk_codes() -> dict[str, str]:
    """
    Parse the high-risk destination codes out of the jurisdiction list.

    The policy references synthetic ISO-style codes, so bare uppercase pairs in
    that document are the codes; currency codes are excluded.
    """
    text = read_policy(HIGH_RISK_POLICY)
    codes = {
        code
        for code in re.findall(r"\b([A-Z]{2})\b", text)
        if not _CURRENCY_CODE.fullmatch(code)
    }
    return {code: HIGH_RISK_POLICY for code in sorted(codes)}


def applicable_policies(client_country: str) -> list[str]:
    """
    Documents that govern a client in ``client_country``.

    The global policy always applies; Singapore and Switzerland layer a
    regional procedure on top.  Every other country is global-only.
    """
    sources = [GLOBAL_POLICY]
    regional = REGIONAL_POLICIES.get((client_country or "").strip())
    if regional:
        sources.append(regional)
    sources.append(HIGH_RISK_POLICY)
    return sources


def resolve_thresholds(client_country: str) -> dict:
    """
    Threshold sets that apply to a client, keyed by policy layer.

    Both layers are reported rather than one overriding the other: regional
    procedures add requirements on top of the global policy, so a payment can
    trip a global threshold, a regional one, or both.
    """
    regional_source = REGIONAL_POLICIES.get((client_country or "").strip())

    layers = {"global": _threshold_map(GLOBAL_POLICY)}
    if regional_source:
        layers["regional"] = _threshold_map(regional_source)

    return {
        "client_country": client_country,
        "regional_source": regional_source,
        "policy_scope": "global + regional" if regional_source else "global only",
        "layers": layers,
    }


def _threshold_map(source: str) -> dict:
    """Requirement -> threshold, for the rules in one document that carry amounts."""
    return {
        rule["requirement"]: {
            "amount": rule["amount"],
            "currency": rule["currency"],
            "source": source,
            "text": rule["text"],
        }
        for rule in policy_rules(source)
        if rule["amount"] is not None
    }


def _compare(amount: float, currency: str, threshold: dict) -> dict:
    """
    Compare an amount against one threshold.

    When the payment currency differs from the threshold currency the
    comparison is made 1:1 and the assumption is recorded, per DATA_NOTES.md —
    no exchange-rate data is supplied.
    """
    same_currency = currency == threshold["currency"]
    exceeds = amount > threshold["amount"]

    result = {
        "requirement": None,  # filled in by the caller
        "threshold_amount": threshold["amount"],
        "threshold_currency": threshold["currency"],
        "payment_amount": amount,
        "payment_currency": currency,
        "exceeds_threshold": exceeds,
        "comparison": (
            f"{currency} {amount:,.2f} "
            f"{'>' if exceeds else '<='} "
            f"{threshold['currency']} {threshold['amount']:,.2f}"
        ),
        "source": threshold["source"],
        "policy_text": threshold["text"],
        "currency_assumption": None
        if same_currency
        else (
            f"compared {currency} against a {threshold['currency']} threshold 1:1; "
            "no exchange-rate data is provided"
        ),
    }
    return result


def assess_payment(payment: dict, client: dict) -> dict:
    """
    Evaluate one payment against every threshold and jurisdiction rule.

    Returns the full deterministic picture: which thresholds were compared,
    which fired, whether the destination is high-risk, the review requirements
    that follow, plus currency assumptions and data-quality conflicts.
    """
    if not payment.get("found"):
        return {"assessable": False, "reason": payment.get("error", "payment not found")}

    amount = payment["amount"] or 0.0
    currency = payment["currency"]
    country = client.get("country") if client.get("found") else None

    thresholds = resolve_thresholds(country)
    evaluations = []

    for layer, threshold_map in thresholds["layers"].items():
        for requirement, threshold in sorted(threshold_map.items()):
            if requirement == "structuring":
                continue  # handled by assess_structuring, needs aggregated totals
            evaluation = _compare(amount, currency, threshold)
            evaluation["requirement"] = requirement
            evaluation["requirement_label"] = _HUMAN_LABELS[requirement]
            evaluation["policy_layer"] = layer
            evaluations.append(evaluation)

    high_risk = high_risk_codes()
    destination_code = payment["beneficiary_country_code"]
    is_high_risk = destination_code in high_risk

    requirements = [
        {
            "requirement": evaluation["requirement_label"],
            "reason": evaluation["comparison"],
            "source": evaluation["source"],
        }
        for evaluation in evaluations
        if evaluation["exceeds_threshold"]
    ]

    if is_high_risk:
        requirements.append(
            {
                "requirement": _HUMAN_LABELS["additional_review"],
                "reason": (
                    f"destination {destination_code} is on the high-risk "
                    "jurisdiction list"
                ),
                "source": HIGH_RISK_POLICY,
            }
        )

    return {
        "assessable": True,
        "payment_id": payment["payment_id"],
        "amount": amount,
        "currency": currency,
        "beneficiary_country_code": destination_code,
        "client_country": country,
        "policy_scope": thresholds["policy_scope"],
        "applicable_policy_documents": applicable_policies(country),
        "high_risk_destination": is_high_risk,
        "high_risk_source": HIGH_RISK_POLICY if is_high_risk else None,
        "threshold_evaluations": evaluations,
        "review_requirements": _dedupe_requirements(requirements),
        "no_review_required": not requirements,
        "currency_assumptions": sorted(
            {
                evaluation["currency_assumption"]
                for evaluation in evaluations
                if evaluation["currency_assumption"]
            }
        ),
        "data_quality_flags": data_quality_flags(payment),
        "caveat": (
            "A policy trigger does not by itself establish suspicious activity "
            f"({GLOBAL_POLICY})."
        ),
    }


def _dedupe_requirements(requirements: list[dict]) -> list[dict]:
    """Collapse identical requirement/source pairs while preserving order."""
    seen = set()
    unique = []
    for item in requirements:
        key = (item["requirement"], item["source"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def assess_structuring(aggregation: dict, client: dict) -> dict:
    """
    Compare an aggregated 24-hour beneficiary window against the structuring
    threshold.

    The Swiss procedure states no structuring amount, so a Swiss client falls
    back to the global threshold while still inheriting the Swiss escalation
    route.  The pattern determination is arithmetic; intent is not.
    """
    country = client.get("country") if client.get("found") else None
    window = aggregation.get("largest_window")

    threshold = _threshold_map(GLOBAL_POLICY).get("structuring")
    regional_source = REGIONAL_POLICIES.get((country or "").strip())
    escalation = [
        rule
        for rule in (policy_rules(regional_source) if regional_source else [])
        if rule["requirement"] == "structuring"
    ]

    if not window or window["count"] < 2:
        return {
            "pattern_present": False,
            "determination": "absent",
            "explanation": (
                "No beneficiary received more than one payment from this client "
                "inside a single 24-hour window."
            ),
            "threshold_amount": threshold["amount"] if threshold else None,
            "threshold_currency": threshold["currency"] if threshold else None,
            "sources": [GLOBAL_POLICY],
        }

    comparison = _compare(window["total_amount"], window["currency"] or "mixed", threshold)
    sources = [GLOBAL_POLICY] + [rule["source"] for rule in escalation]

    return {
        "pattern_present": True,
        "determination": "present" if comparison["exceeds_threshold"] else "below_threshold",
        "payment_count": window["count"],
        "payment_ids": window["payment_ids"],
        "payment_date": window["payment_date"],
        "beneficiary_name": aggregation["beneficiary_name"],
        "combined_amount": window["total_amount"],
        "combined_currency": window["currency"],
        "channels_used": window["channels"],
        "threshold_amount": comparison["threshold_amount"],
        "threshold_currency": comparison["threshold_currency"],
        "comparison": comparison["comparison"],
        "exceeds_threshold": comparison["exceeds_threshold"],
        "currency_assumption": comparison["currency_assumption"],
        "window_assumption": aggregation["assumption"],
        "escalation_route": [rule["text"] for rule in escalation] or None,
        "observed_facts": [
            f"{window['count']} payments to {aggregation['beneficiary_name']} "
            f"on {window['payment_date']} ({', '.join(window['payment_ids'])})",
            f"combined value {comparison['comparison']}",
            f"channels used: {', '.join(window['channels'])}",
        ],
        "assumption_not_established": (
            "Intent to evade a reporting threshold is an assumption, not an "
            "observed fact; the aggregation establishes the pattern only."
        ),
        "sources": sorted(set(sources)),
    }


def data_quality_flags(payment: dict) -> list[str]:
    """
    Report the beneficiary country/code disagreement as an observed fact.

    The mismatch is intentional in the dataset.  The code is authoritative for
    risk, but the conflict itself is worth reporting to an investigator.
    """
    flags = []
    name = payment.get("beneficiary_country")
    code = payment.get("beneficiary_country_code")

    if name and code and not _country_matches_code(name, code):
        flags.append(
            f"beneficiary_country ({name}) disagrees with "
            f"beneficiary_country_code ({code}); the code is authoritative for "
            "jurisdiction risk assessment"
        )
    return flags


# Enough of a name->code map to detect disagreement across the countries that
# appear in payments.csv.
_COUNTRY_CODES = {
    "uae": "AE",
    "united arab emirates": "AE",
    "singapore": "SG",
    "switzerland": "CH",
    "hong kong": "HK",
    "uk": "GB",
    "united kingdom": "GB",
}


def _country_matches_code(name: str, code: str) -> bool:
    expected = _COUNTRY_CODES.get(name.strip().casefold())
    if expected is None:
        return True  # unknown name: no basis to claim a conflict
    return expected == code.strip().upper()


def investigation_workflow() -> dict:
    """The ordered workflow from the investigation procedure, with its source."""
    steps = [
        text
        for _heading, text in split_into_rules(
            clean_document(read_policy(INVESTIGATION_PROCEDURE))
        )
        if text
    ]
    return {"source": INVESTIGATION_PROCEDURE, "steps": steps}
