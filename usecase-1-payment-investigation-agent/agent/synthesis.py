"""
Deterministic answer composer (no-LLM fallback and LLM-output validator).

Builds schema-compliant answers from the gathered evidence bundle. Used
directly when no LLM key is configured, and as the grounding backstop for
LLM outputs (facts/citations repaired against tool results).
"""

from __future__ import annotations


def _money(amount: float, currency: str) -> str:
    return f"{currency} {amount:,.2f}".replace(".00", "")


def classify_intent(question: str) -> str:
    """Route on question wording (never on question_id)."""
    q = question.lower()
    if any(k in q for k in ("split", "structur")):
        return "structuring"
    if any(k in q for k in ("workflow", "procedure", "steps", "followed")):
        return "workflow"
    if "which policy documents" in q or "which documents" in q:
        return "which_documents"
    if "additional information" in q:
        return "additional_info"
    if "assumption" in q or "factual evidence" in q:
        return "facts_vs_assumptions"
    if "below" in q and "high-risk" in q.replace("high risk", "high-risk"):
        return "high_risk_below_threshold"
    if "threshold" in q or "region" in q:
        return "threshold"
    if "enhanced review" in q or "review requirement" in q or "risk indicator" in q:
        return "review"
    return "review"


def _review_sentence(bundle: dict) -> str:
    p = bundle["payment"]
    parts = [
        f"{p['payment_id']} is a {_money(p['amount'], p['currency'])} payment "
        f"to {p['beneficiary_name']} (country code {p['beneficiary_country_code']})."
    ]
    t = bundle["thresholds"]
    if t["enhanced_triggered"]:
        parts.append(
            f"The amount exceeds the {_money(t['enhanced'], t['threshold_currency'])} "
            f"enhanced-review threshold ({t['policy']}); enhanced review is required."
        )
    elif t["rm_triggered"]:
        parts.append(
            f"The amount exceeds the {_money(t['rm'], t['threshold_currency'])} "
            f"RM-review threshold ({t['policy']}) but not the enhanced threshold; "
            f"RM review is required."
        )
    elif t["rm"] is None:
        parts.append(
            f"The amount is below the {_money(t['enhanced'], t['threshold_currency'])} "
            f"global enhanced-review threshold; no threshold review is triggered "
            f"(global policy sets no separate RM threshold)."
        )
    else:
        parts.append(
            f"The amount is below the {_money(t['rm'], t['threshold_currency'])} "
            f"RM-review threshold ({t['policy']}); no threshold review is triggered."
        )
    if t.get("fx_assumption"):
        parts.append(
            "Currencies differ, so equivalence is treated 1:1 per the data-notes "
            "convention; the conclusion does not depend on precise conversion."
        )
    if bundle["high_risk"]:
        parts.append(
            f"Destination {p['beneficiary_country_code']} is a high-risk jurisdiction; "
            f"additional review is required."
        )
    else:
        parts.append(
            f"Destination {p['beneficiary_country_code']} is not high-risk."
        )
    c = bundle["client"]
    parts.append(
        f"Client {c['client_id']} is based in {c['country']} "
        f"({c['client_type']}, {c['risk_rating']} risk)."
    )
    return " ".join(parts)


def _structuring_sentence(bundle: dict) -> str:
    s = bundle["structuring"]
    if not s["patterns"]:
        return (
            f"No same-beneficiary, same-date payment clusters were found for "
            f"client {bundle['client']['client_id']}; the data shows no "
            f"transaction-splitting pattern."
        )
    top = s["patterns"][0]
    w = top["window"]
    verdict = (
        "exceeds" if top["exceeds_threshold"] else "does not exceed"
    )
    return (
        f"Client {bundle['client']['client_id']} made {w['count']} payments to "
        f"{top['beneficiary_name']} on {w['window_date']}: "
        f"{', '.join(w['payment_ids'])} totalling {_money(w['total_amount'], w['currency'] if isinstance(w['currency'], str) else '/'.join(w['currency']))}, "
        f"which {verdict} the USD 100,000-equivalent structuring threshold. "
        f"Channels used: {', '.join(w['channels'])}. This is an observed fact "
        f"pattern, not proof of intent to evade."
    )


_MISSING_INFO = (
    "purpose of payment, source of funds, beneficiary relationship history, "
    "and (for structuring) whether the split had a business rationale"
)


def compose(bundle: dict, question: str) -> dict:
    """Compose the final ``{answer, citations, facts, tools_used}`` dict."""
    intent = classify_intent(question)
    p, c = bundle["payment"], bundle["client"]
    citations = bundle["citations"]
    facts: dict = {
        "amount": p["amount"],
        "currency": p["currency"],
        "beneficiary_country_code": p["beneficiary_country_code"],
        "client_id": c["client_id"],
        "client_country": c["country"],
        "client_risk_rating": c["risk_rating"],
    }

    if intent == "structuring":
        s = bundle["structuring"]["patterns"]
        if s:
            w = s[0]["window"]
            facts.update(
                {
                    "beneficiary_name": s[0]["beneficiary_name"],
                    "payment_date": w["window_date"],
                    "payment_ids": w["payment_ids"],
                    "combined_amount": w["total_amount"],
                    "channels": w["channels"],
                }
            )
        answer = _structuring_sentence(bundle) + (
            " Per the applicable procedure, potential structuring should be "
            "escalated to Compliance after requesting: " + _MISSING_INFO + "."
        )
    elif intent == "workflow":
        answer = (
            "The investigation workflow is: (1) establish client and payment "
            "facts; (2) identify the applicable policy; (3) check high-risk "
            "destination indicators; (4) check for possible transaction "
            "splitting; (5) separate observed facts from assumptions; "
            "(6) record the evidence supporting the recommendation."
        )
        facts = {"client_id": c["client_id"], "payment_id": p["payment_id"]}
    elif intent == "which_documents":
        answer = (
            f"Before recommending release of {p['payment_id']}, retrieve: "
            + "; ".join(f"{s} (policy evidence)" for s in citations)
            + ". These cover the applicable thresholds, high-risk "
            "destination rules, and the investigation procedure."
        )
    elif intent == "additional_info":
        answer = (
            "Before escalating, request: " + _MISSING_INFO + ". "
            + _structuring_sentence(bundle)
        )
    elif intent == "facts_vs_assumptions":
        t = bundle["thresholds"]
        trigger = (
            f"amount {'exceeds' if t['enhanced_triggered'] else 'is below'} "
            f"the {_money(t['enhanced'], t['threshold_currency'])} "
            f"enhanced-review threshold ({t['policy']})"
        )
        if t["rm_triggered"]:
            trigger += (
                f" and exceeds the {_money(t['rm'], t['threshold_currency'])} "
                f"RM-review threshold"
            )
        trigger += (
            f"; destination {p['beneficiary_country_code']} "
            f"{'is high-risk, so additional review applies' if bundle['high_risk'] else 'is not high-risk'}."
        )
        answer = (
            f"Observed facts: {p['payment_id']} is {_money(p['amount'], p['currency'])} "
            f"to {p['beneficiary_name']} (code {p['beneficiary_country_code']}); "
            f"client {c['client_id']} is {c['country']}-based ({c['risk_rating']} risk). "
            f"Policy trigger: {trigger} "
            f"Assumptions (not established): any intent to evade controls or "
            f"conceal the destination. Missing evidence: {_MISSING_INFO}. "
            f"Recommendation: hold for the reviews above and request payment-purpose "
            f"documentation before release."
        )
    elif intent == "high_risk_below_threshold":
        answer = (
            f"Even below the global threshold, a high-risk destination requires "
            f"additional review. {_review_sentence(bundle)} Recommendation: "
            f"additional review and payment-purpose documentation before release."
        )
    else:  # review / threshold
        answer = _review_sentence(bundle)
        t = bundle["thresholds"]
        facts.update(
            {
                "threshold_policy": t["policy"],
                "threshold_currency": t["threshold_currency"],
                "rm_threshold": t["rm"],
                "enhanced_threshold": t["enhanced"],
            }
        )
        if intent == "threshold":
            if t["rm"] is not None:
                answer += (
                    f" In this region ({t['policy']}) the applicable ladder is: "
                    f"RM review above {_money(t['rm'], t['threshold_currency'])}, "
                    f"enhanced review above {_money(t['enhanced'], t['threshold_currency'])}."
                )
            else:
                answer += (
                    f" No regional procedure applies, so the global policy governs: "
                    f"enhanced review above {_money(t['enhanced'], t['threshold_currency'])}."
                )
        if intent == "review":
            answer += (
                " Recommendation: complete the required review(s) and request "
                "payment-purpose documentation before release."
            )

    return {
        "answer": answer,
        "citations": citations,
        "facts": facts,
        "tools_used": bundle["tools_used"],
    }
