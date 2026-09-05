"""
Payment and deterministic analysis tools.

Exact calculations happen here, not in the LLM.
"""

from __future__ import annotations

from collections import defaultdict

from tools._data import load_payments, record_to_dict
from tools.client_tools import get_client_profile


def get_payment(payment_id: str) -> dict:
    """Retrieve one payment by payment ID."""
    df = load_payments()
    matches = df[df["payment_id"] == str(payment_id)]
    if matches.empty:
        return {
            "error": "payment_not_found",
            "payment_id": payment_id,
            "message": f"No payment found with payment_id={payment_id}",
        }
    return record_to_dict(matches.iloc[0])


def get_client_payments(client_id: str) -> list[dict]:
    """Retrieve the supplied payment history for a client."""
    df = load_payments()
    matches = df[df["client_id"] == str(client_id)].sort_values(
        by=["payment_date", "payment_id"]
    )
    return [record_to_dict(row) for _, row in matches.iterrows()]


def aggregate_beneficiary_24h(
    client_id: str,
    beneficiary_name: str,
    payment_date: str | None = None,
) -> dict:
    """
    Aggregate payments to a beneficiary within a 24-hour window.

    Because ``payment_date`` has no time component, payments on the same
    calendar date are treated as within the same 24-hour window.
    Filters by both client_id and beneficiary_name.

    If ``payment_date`` is provided, the summary fields refer to that date's
    window (not the largest window elsewhere in history).
    """
    df = load_payments()
    mask = (df["client_id"] == str(client_id)) & (
        df["beneficiary_name"].str.casefold() == str(beneficiary_name).casefold()
    )
    subset = df[mask].copy()
    assumption = (
        "Same calendar payment_date treated as the 24-hour window "
        "(no time component in the data)."
    )
    if subset.empty:
        return {
            "client_id": client_id,
            "beneficiary_name": beneficiary_name,
            "windows": [],
            "assumption": assumption,
            "count": 0,
            "total_amount": 0.0,
            "payments": [],
        }

    windows = []
    for date, group in subset.groupby("payment_date", sort=True):
        payments = [record_to_dict(row) for _, row in group.iterrows()]
        currencies = sorted({p["currency"] for p in payments if p.get("currency")})
        total = float(sum(float(p["amount"]) for p in payments))
        windows.append(
            {
                "payment_date": str(date),
                "count": len(payments),
                "total_amount": total,
                "currencies": currencies,
                "currency": currencies[0] if len(currencies) == 1 else None,
                "payment_ids": [p["payment_id"] for p in payments],
                "individual_amounts": [float(p["amount"]) for p in payments],
                "channels": [p.get("channel") for p in payments],
                "payments": payments,
            }
        )

    if payment_date:
        primary = next(
            (w for w in windows if w["payment_date"] == str(payment_date)),
            None,
        )
        if primary is None:
            return {
                "client_id": client_id,
                "beneficiary_name": beneficiary_name,
                "assumption": assumption,
                "payment_date": str(payment_date),
                "count": 0,
                "total_amount": 0.0,
                "payments": [],
                "windows": windows,
                "message": f"No payments on {payment_date} for this client/beneficiary",
            }
    else:
        # Prefer densest same-day window (structuring signal)
        primary = max(windows, key=lambda w: (w["count"], w["total_amount"]))

    return {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "assumption": assumption,
        "count": primary["count"],
        "total_amount": primary["total_amount"],
        "currency": primary["currency"],
        "currencies": primary["currencies"],
        "payment_date": primary["payment_date"],
        "payment_ids": primary["payment_ids"],
        "individual_amounts": primary["individual_amounts"],
        "channels": primary["channels"],
        "payments": primary["payments"],
        "windows": windows,
    }


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Identify beneficiaries appearing multiple times in client history."""
    payments = get_client_payments(client_id)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for payment in payments:
        by_name[payment["beneficiary_name"]].append(payment)

    results = []
    for name, items in sorted(by_name.items()):
        if len(items) < 2:
            continue
        results.append(
            {
                "beneficiary_name": name,
                "count": len(items),
                "payment_ids": [p["payment_id"] for p in items],
                "total_amount_by_currency": _totals_by_currency(items),
            }
        )
    return results


def evaluate_review_requirements(payment_id: str) -> dict:
    """
    Deterministic threshold / risk / structuring evaluation for one payment.

    Uses policy thresholds from DATA_NOTES + policy docs. Currency comparison:
    native when regional policy matches; otherwise 1:1 equivalent assumption.
    """
    payment = get_payment(payment_id)
    if payment.get("error"):
        return payment

    client = get_client_profile(payment["client_id"])
    if client.get("error"):
        return {**payment, "client_error": client}

    client_country = client.get("country")
    amount = float(payment["amount"])
    currency = str(payment["currency"])
    code = str(payment.get("beneficiary_country_code") or "").upper()
    name_country = payment.get("beneficiary_country")

    if client_country == "Singapore":
        regional_policy = "regional_singapore.md"
        regional_thresholds = {
            "rm_review": {"amount": 75000.0, "currency": "USD"},
            "enhanced_review": {"amount": 100000.0, "currency": "USD"},
        }
    elif client_country == "Switzerland":
        regional_policy = "regional_switzerland.md"
        regional_thresholds = {
            "rm_review": {"amount": 80000.0, "currency": "CHF"},
            "enhanced_review": {"amount": 120000.0, "currency": "CHF"},
        }
    else:
        regional_policy = None
        regional_thresholds = None

    global_enhanced = {"amount": 100000.0, "currency": "USD"}
    structuring_threshold = {"amount": 100000.0, "currency": "USD"}

    currency_assumption = None
    triggers: list[str] = []

    # Regional amount checks (native currency only when currencies match)
    requires_rm = False
    requires_enhanced_regional = False
    if regional_thresholds:
        rm = regional_thresholds["rm_review"]
        enh = regional_thresholds["enhanced_review"]
        if currency == rm["currency"]:
            requires_rm = amount > rm["amount"]
            requires_enhanced_regional = amount > enh["amount"]
        else:
            currency_assumption = (
                f"Regional thresholds are in {rm['currency']} but payment is "
                f"{currency}; treating as 1:1 equivalent for comparison."
            )
            requires_rm = amount > rm["amount"]
            requires_enhanced_regional = amount > enh["amount"]

    # Global enhanced (USD); 1:1 if not USD
    if currency != "USD":
        currency_assumption = (
            (currency_assumption + " ") if currency_assumption else ""
        ) + (
            "Global USD thresholds compared 1:1 to payment currency "
            "(no FX data provided)."
        )
    requires_enhanced_global = amount > global_enhanced["amount"]
    requires_enhanced = requires_enhanced_global or requires_enhanced_regional

    high_risk_destination = code == "AE"
    requires_additional_review = high_risk_destination
    country_code_mismatch = bool(
        name_country
        and code
        and str(name_country).strip().upper() not in {"UAE", "UNITED ARAB EMIRATES"}
        and code == "AE"
    ) or (
        name_country
        and code
        and str(name_country).casefold() != _code_to_name_hint(code)
    )

    # Structuring window for this payment's date
    window = aggregate_beneficiary_24h(
        payment["client_id"],
        payment["beneficiary_name"],
        payment_date=str(payment["payment_date"]),
    )
    combined = float(window.get("total_amount") or 0.0)
    # 1:1 vs USD 100k structuring threshold
    possible_structuring = (
        int(window.get("count") or 0) >= 2 and combined > structuring_threshold["amount"]
    )

    if requires_rm:
        triggers.append("rm_review")
    if requires_enhanced:
        triggers.append("enhanced_review")
    if requires_additional_review:
        triggers.append("additional_review_high_risk_destination")
    if possible_structuring:
        triggers.append("possible_structuring_24h")

    applicable_policies = ["global_payment_policy.md"]
    if regional_policy:
        applicable_policies.append(regional_policy)
    if high_risk_destination:
        applicable_policies.append("high_risk_jurisdictions.md")
    if possible_structuring and client_country == "Switzerland":
        # Switzerland: escalate potential structuring to Compliance
        pass
    applicable_policies.append("investigation_procedure.md")

    recommendations: list[str] = []
    if requires_enhanced:
        recommendations.append("enhanced_review_before_release")
    if requires_rm and not requires_enhanced:
        recommendations.append("rm_review")
    if requires_additional_review:
        recommendations.append("additional_review_high_risk_destination")
    if possible_structuring:
        if client_country == "Switzerland":
            recommendations.append("escalate_potential_structuring_to_compliance")
        else:
            recommendations.append("review_for_potential_structuring")
    if not recommendations:
        recommendations.append("standard_monitoring")

    return {
        "payment_id": payment_id,
        "amount": amount,
        "currency": currency,
        "beneficiary_name": payment.get("beneficiary_name"),
        "beneficiary_country": name_country,
        "beneficiary_country_code": code,
        "country_code_mismatch_note": (
            "beneficiary_country and beneficiary_country_code disagree; "
            "use beneficiary_country_code for jurisdiction risk."
            if str(name_country or "").casefold() != _code_to_name_hint(code)
            else None
        ),
        "client_id": payment["client_id"],
        "client_country": client_country,
        "client_risk_rating": client.get("risk_rating"),
        "payment_date": payment.get("payment_date"),
        "channel": payment.get("channel"),
        "applicable_policies": applicable_policies,
        "regional_policy": regional_policy,
        "regional_thresholds": regional_thresholds,
        "global_enhanced_threshold": global_enhanced,
        "structuring_threshold": structuring_threshold,
        "currency_assumption": currency_assumption,
        "requires_rm_review": requires_rm,
        "requires_enhanced_review": requires_enhanced,
        "requires_additional_review": requires_additional_review,
        "high_risk_destination": high_risk_destination,
        "same_day_beneficiary_window": {
            "count": window.get("count"),
            "total_amount": window.get("total_amount"),
            "currency": window.get("currency"),
            "payment_date": window.get("payment_date"),
            "payment_ids": window.get("payment_ids"),
            "individual_amounts": window.get("individual_amounts"),
            "channels": window.get("channels"),
        },
        "possible_structuring": possible_structuring,
        "policy_triggers": triggers,
        "recommendations": recommendations,
        "note": (
            "Policy triggers are observed indicators only; they do not by "
            "themselves establish suspicious activity or intent."
        ),
    }


def _code_to_name_hint(code: str) -> str:
    mapping = {
        "AE": "uae",
        "SG": "singapore",
        "CH": "switzerland",
        "HK": "hong kong",
        "GB": "uk",
        "UK": "uk",
    }
    return mapping.get(code.upper(), code.casefold())


def _totals_by_currency(payments: list[dict]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for payment in payments:
        totals[str(payment["currency"])] += float(payment["amount"])
    return dict(totals)
