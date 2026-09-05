"""
Payment and deterministic analysis tool interfaces.

These methods intentionally contain NO implementations.

Exact calculations should happen in these tools, not in the LLM.
"""

import csv
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path


_PAYMENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "payments.csv"


@lru_cache(maxsize=1)
def _payments() -> tuple[dict, ...]:
    with _PAYMENTS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            row["amount"] = float(row["amount"])
            rows.append(row)
        return tuple(rows)


def get_payment(payment_id: str) -> dict:
    """
    Retrieve one payment by payment ID.

    The implementation should read from ``data/payments.csv``.

    Returns a structured payment record or a clear empty/error result when
    the payment does not exist.
    """
    key = payment_id.strip().upper()
    return next((dict(row) for row in _payments() if row["payment_id"] == key), {})


def get_client_payments(client_id: str) -> list[dict]:
    """
    Retrieve the supplied payment history for a client.

    Useful for transaction-pattern and structuring questions.
    """
    key = client_id.strip().upper()
    return sorted(
        (dict(row) for row in _payments() if row["client_id"] == key),
        key=lambda row: (row["payment_date"], row["payment_id"]),
    )


def aggregate_beneficiary_24h(
    client_id: str,
    beneficiary_name: str,
) -> dict:
    """
    Aggregate payments to a beneficiary within a 24-hour window.

    This should be deterministic Python/business logic.

    Suggested result:

    {
        "count": 3,
        "total_amount": 140000,
        "payments": [...]
    }

    Consider:
    - date/time parsing;
    - true 24-hour windows;
    - missing values;
    - currency handling;
    - preserving payment IDs.
    """
    key = beneficiary_name.strip().casefold()
    matches = [
        row for row in get_client_payments(client_id)
        if row["beneficiary_name"].casefold() == key
    ]
    by_date_currency: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in matches:
        # The supplied data has calendar dates only. Per DATA_NOTES, only
        # payments on the same calendar date are within the 24-hour window.
        date.fromisoformat(row["payment_date"])
        by_date_currency[(row["payment_date"], row["currency"])].append(row)

    windows = []
    for (payment_date, currency), rows in sorted(by_date_currency.items()):
        windows.append({
            "payment_date": payment_date,
            "currency": currency,
            "count": len(rows),
            "total_amount": round(sum(row["amount"] for row in rows), 2),
            "payment_ids": [row["payment_id"] for row in rows],
            "payments": rows,
        })
    largest = max(windows, key=lambda item: (item["total_amount"], item["count"]), default=None)
    return {
        "client_id": client_id.strip().upper(),
        "beneficiary_name": beneficiary_name,
        "window_assumption": "Same calendar date is treated as within 24 hours; no times are supplied.",
        "windows": windows,
        "largest_window": largest,
        "count": largest["count"] if largest else 0,
        "total_amount": largest["total_amount"] if largest else 0.0,
        "payments": largest["payments"] if largest else [],
        "structuring_threshold_usd_equivalent": 100000.0,
        "uses_1_to_1_equivalent_assumption": bool(
            largest and largest["currency"] != "USD"
        ),
        "potential_structuring_trigger": bool(
            largest
            and largest["count"] > 1
            and largest["total_amount"] > 100000.0
        ),
    }


def assess_payment_review(payment_id: str, client_country: str) -> dict:
    """Deterministically compare a payment with applicable policy thresholds.

    No exchange-rate table is supplied, so non-matching currencies use the
    exercise's documented 1:1-equivalent assumption.  The caller supplies the
    client country obtained through the client tool; beneficiary risk always
    uses the authoritative country-code field in the payment record.
    """
    payment = get_payment(payment_id)
    if not payment:
        return {}

    amount = payment["amount"]
    currency = payment["currency"]
    region = client_country.strip()
    high_risk = payment["beneficiary_country_code"] == "AE"
    thresholds = {
        "global_enhanced_review": {
            "amount": 100000.0,
            "currency": "USD",
            "triggered": amount > 100000.0,
        }
    }
    if region == "Singapore":
        thresholds.update({
            "regional_rm_review": {
                "amount": 75000.0, "currency": "USD", "triggered": amount > 75000.0,
            },
            "regional_enhanced_review": {
                "amount": 100000.0, "currency": "USD", "triggered": amount > 100000.0,
            },
        })
    elif region == "Switzerland":
        thresholds.update({
            "regional_rm_review": {
                "amount": 80000.0, "currency": "CHF", "triggered": amount > 80000.0,
            },
            "regional_enhanced_review": {
                "amount": 120000.0, "currency": "CHF", "triggered": amount > 120000.0,
            },
        })

    triggered_reviews = [
        name for name, detail in thresholds.items() if detail["triggered"]
    ]
    if high_risk:
        triggered_reviews.append("additional_high_risk_destination_review")
    return {
        "payment_id": payment["payment_id"],
        "amount": amount,
        "payment_currency": currency,
        "client_country": region,
        "beneficiary_country_code": payment["beneficiary_country_code"],
        "high_risk_destination": high_risk,
        "thresholds": thresholds,
        "triggered_reviews": triggered_reviews,
        "uses_1_to_1_equivalent_assumption": any(
            detail["currency"] != currency for detail in thresholds.values()
        ),
    }


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """
    OPTIONAL: Identify beneficiaries appearing multiple times in the
    client's payment history.

    Useful for potential structuring analysis.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in get_client_payments(client_id):
        groups[row["beneficiary_name"]].append(row)
    return [
        {"beneficiary_name": name, "count": len(rows), "payments": rows}
        for name, rows in sorted(groups.items()) if len(rows) > 1
    ]
