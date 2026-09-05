"""Policy RAG and source-backed, deterministic threshold evaluation."""

import re
from decimal import Decimal
from functools import lru_cache

from rag.pipeline import build_index, chunk_documents, load_policy_documents, retrieve
from tools.client_tools import get_client_profile
from tools.data import DATA_DIR
from tools.payment_tools import FX_ASSUMPTION, get_payment

REGIONAL_POLICIES = {"Singapore": "regional_singapore.md",
                     "Switzerland": "regional_switzerland.md"}
GLOBAL_POLICY = "global_payment_policy.md"
RISK_POLICY = "high_risk_jurisdictions.md"


@lru_cache(maxsize=1)
def _documents() -> dict:
    return {doc["source"]: doc for doc in load_policy_documents(str(DATA_DIR / "policies"))}


@lru_cache(maxsize=1)
def _get_index() -> dict:
    return build_index(chunk_documents(list(_documents().values())))


def search_policy(query: str, top_k: int = 3) -> list[dict]:
    return retrieve(_get_index(), query, top_k)


def get_policy_document(source: str) -> dict:
    """Exact corpus lookup, never arbitrary filesystem access."""
    doc = _documents().get(source)
    return dict(doc) if doc else {"error": "Policy document not found", "source": source}


def _threshold_rules(document: dict) -> list[dict]:
    # ponytail: parse the supplied bullet-policy grammar; replace with a
    # reviewed rule schema if the corpus gains complex exceptions/tables.
    rules = []
    for paragraph in re.split(r"\n(?=- )", document["text"]):
        text = re.sub(r"\s+", " ", paragraph).strip()
        match = re.search(r"\b(above|exceeds)\s+(USD|CHF)\s+([\d,]+(?:\.\d+)?)", text)
        if not match:
            continue
        kind = ("structuring" if "structuring" in text.lower() else
                "enhanced_review" if "enhanced review" in text.lower() else "rm_review")
        rules.append({"source": document["source"], "rule": text,
                      "kind": kind, "operator": ">", "currency": match[2],
                      "threshold": float(Decimal(match[3].replace(",", "")))})
    return rules


def assess_payment_policy(payment_id: str, sources: list[str],
                          aggregations: list[dict] | None = None) -> dict:
    """Compare facts with rules extracted from previously retrieved sources.

    The agent injects previously executed aggregation results; the model never
    supplies amounts, thresholds, counts, FX rates or calculated booleans.
    """
    payment = get_payment(payment_id)
    if "error" in payment:
        return payment
    client = get_client_profile(payment["client_id"])
    if "error" in client:
        return client
    aggregations = [a for a in aggregations or [] if a["client_id"] == payment["client_id"]]
    required = [GLOBAL_POLICY, RISK_POLICY]
    regional = REGIONAL_POLICIES.get(client["country"])
    if regional:
        required.append(regional)
    applicable = set(required) | {"investigation_procedure.md"}
    selected = [get_policy_document(source) for source in dict.fromkeys(sources)
                if source in applicable]
    missing = sorted(set(required) - {doc["source"] for doc in selected})
    amount = Decimal(str(payment["amount"]))
    threshold_checks = []
    structuring_checks = []
    assumptions = []
    rules = [rule for doc in selected for rule in _threshold_rules(doc)]
    for rule in rules:
        if rule["kind"] == "structuring":
            for analysis in aggregations or []:
                if analysis["client_id"] != payment["client_id"]:
                    continue
                assumptions.append(analysis["date_assumption"])
                for window in analysis["windows"]:
                    if window["fx_assumption"]:
                        assumptions.append(window["fx_assumption"])
                    total = Decimal(str(window["usd_equivalent"]))
                    structuring_checks.append({
                        **rule, **window, "beneficiary_name": analysis["beneficiary_name"],
                        "threshold_currency": rule["currency"],
                        "triggered": window["count"] > 1 and total > Decimal(str(rule["threshold"])),
                        "each_payment_below_structuring_threshold": all(
                            Decimal(str(p["amount"])) < Decimal(str(rule["threshold"]))
                            for p in window["payments"]),
                    })
            continue
        if payment["currency"] != rule["currency"]:
            assumptions.append(FX_ASSUMPTION)
        threshold_checks.append({**rule, "compared_amount": float(amount),
                                 "payment_currency": payment["currency"],
                                 "comparison_basis": "native currency" if payment["currency"] == rule["currency"] else "assumed 1:1 equivalent",
                                 "triggered": amount > Decimal(str(rule["threshold"]))})
    risk_doc = next((doc for doc in selected if doc["source"] == RISK_POLICY), None)
    high_risk_codes = re.findall(r"\b([A-Z]{2})\s*\([^)]+\)\s+is a high-risk destination", risk_doc["text"]) if risk_doc else []
    risk_known = bool(risk_doc and high_risk_codes)
    high_risk = payment["beneficiary_country_code"] in high_risk_codes if risk_known else None
    country_names = {"AE": "UAE", "SG": "Singapore", "CH": "Switzerland",
                     "GB": "UK", "HK": "Hong Kong", "US": "USA"}
    authoritative_name = country_names.get(payment["beneficiary_country_code"])
    structuring_triggered = any(check["triggered"] for check in structuring_checks)
    structuring_known = bool(aggregations) and any(rule["kind"] == "structuring" for rule in rules)
    escalation = structuring_triggered and any(
        "structuring should be escalated to Compliance" in doc["text"] for doc in selected)
    return {
        "payment_id": payment_id, "client_country": client["country"],
        "regional_policy": regional, "sources": [doc["source"] for doc in selected],
        "missing_policy_sources": missing, "threshold_checks": threshold_checks,
        "beneficiary_country_code": payment["beneficiary_country_code"],
        "authoritative_country_name": authoritative_name,
        "country_fields_disagree": (payment["beneficiary_country"] != authoritative_name) if authoritative_name else None,
        "high_risk_destination": high_risk, "high_risk_codes": high_risk_codes,
        "risk_evidence": risk_doc["text"] if risk_doc else None,
        "enhanced_review_required": any(c["triggered"] and c["kind"] == "enhanced_review" for c in threshold_checks) if not missing else None,
        "rm_review_required": any(c["triggered"] and c["kind"] == "rm_review" for c in threshold_checks) if not missing else None,
        "additional_review_required": high_risk,
        "structuring_checked": structuring_known,
        "structuring_checks": structuring_checks,
        "potential_structuring": structuring_triggered if structuring_known else None,
        "compliance_escalation_required": escalation if structuring_known and regional not in missing else None,
        "assumptions": list(dict.fromkeys(assumptions)),
        "limitations": ["A policy trigger does not establish intent or suspicious activity.",
                        "Payment purpose, source of funds, supporting documents, beneficiary relationship and precise timestamps are not supplied."],
    }
