"""Exact decimal aggregation under DATA_NOTES.md conventions."""

from collections import defaultdict
from decimal import Decimal

from tools.data import read_rows

DATE_ASSUMPTION = (
    "Only calendar dates are supplied: same-date payments are treated as one "
    "24-hour window; true rolling or cross-midnight windows cannot be determined."
)
FX_ASSUMPTION = (
    "No exchange rates are supplied. For non-matching currencies, the exercise's "
    "permitted 1:1 equivalence is assumed; this is not an observed FX rate."
)


def get_payment(payment_id: str) -> dict:
    return next(
        (row for row in read_rows("payments.csv") if row["payment_id"] == payment_id),
        {"error": "Payment not found", "payment_id": payment_id},
    )


def get_client_payments(client_id: str) -> list[dict]:
    return sorted(
        (row for row in read_rows("payments.csv") if row["client_id"] == client_id),
        key=lambda row: (row["payment_date"], row["payment_id"]),
    )


def aggregate_beneficiary_24h(client_id: str, beneficiary_name: str) -> dict:
    """Return every same-date window, filtered by BOTH client and beneficiary.

    Native totals stay separated by currency. The assumed USD equivalent is
    provided separately for the global structuring comparison.
    """
    groups = defaultdict(list)
    for payment in get_client_payments(client_id):
        if payment["beneficiary_name"] == beneficiary_name:
            groups[payment["payment_date"]].append(payment)
    windows = []
    for payment_date, payments in sorted(groups.items()):
        totals = defaultdict(Decimal)
        for payment in payments:
            totals[payment["currency"]] += Decimal(str(payment["amount"]))
        one_currency = len(totals) == 1
        windows.append({
            "payment_date": payment_date,
            "count": len(payments),
            "payment_ids": [p["payment_id"] for p in payments],
            "individual_amounts": [p["amount"] for p in payments],
            "channels": sorted({p["channel"] for p in payments}),
            "totals_by_currency": {key: float(value) for key, value in totals.items()},
            "currency": next(iter(totals)) if one_currency else "MIXED",
            "total_amount": float(next(iter(totals.values()))) if one_currency else None,
            "usd_equivalent": float(sum(totals.values(), Decimal(0))),
            "fx_assumption": FX_ASSUMPTION if set(totals) != {"USD"} else None,
            "payments": payments,
        })
    return {"client_id": client_id, "beneficiary_name": beneficiary_name,
            "date_assumption": DATE_ASSUMPTION, "windows": windows}


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    counts = defaultdict(int)
    for payment in get_client_payments(client_id):
        counts[payment["beneficiary_name"]] += 1
    return [{"beneficiary_name": name, "count": count}
            for name, count in sorted(counts.items()) if count > 1]
