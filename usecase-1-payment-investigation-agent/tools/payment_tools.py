"""Exact money calculations and date-only aggregation per DATA_NOTES.md."""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal

from tools.data_store import read_records

FX_ASSUMPTION = (
    "No FX rates supplied: non-USD amounts are treated as USD equivalent at 1:1 "
    "for this exercise only; native-currency totals remain separate."
)
DATE_ASSUMPTION = (
    "Dates have no timestamps: payments on the same calendar date are treated "
    "as within one 24-hour window; cross-date timing cannot be established."
)


def get_payment(payment_id: str) -> dict:
    """Retrieve a payment by ID; return an explicit error for unknown IDs."""
    return next((r for r in read_records("payments.csv") if r["payment_id"] == payment_id),
                {"error": "Payment not found", "payment_id": payment_id})


def get_client_payments(client_id: str) -> list[dict]:
    """Retrieve the full supplied payment history for a client."""
    return [r for r in read_records("payments.csv") if r["client_id"] == client_id]


def aggregate_beneficiary_24h(client_id: str, beneficiary_name: str) -> dict:
    """Return all same-date windows, never a misleading multi-date grand total."""
    groups = defaultdict(list)
    excluded = []
    for payment in get_client_payments(client_id):
        if payment["beneficiary_name"] != beneficiary_name:
            continue
        try:
            day = date.fromisoformat(payment["payment_date"]).isoformat()
            amount = Decimal(str(payment["amount"]))
            if not amount.is_finite() or amount < 0 or not payment["currency"]:
                raise ValueError("Missing currency or invalid amount")
        except (ValueError, TypeError, ArithmeticError):
            excluded.append(payment["payment_id"])
            continue
        groups[day].append(payment)
    windows = []
    for day, payments in sorted(groups.items()):
        totals = defaultdict(Decimal)
        for payment in payments:
            totals[payment["currency"]] += Decimal(str(payment["amount"]))
        equivalent = sum(totals.values(), Decimal(0))
        windows.append({
            "date": day, "count": len(payments),
            "totals_by_currency": {k: float(v) for k, v in sorted(totals.items())},
            "total_usd_equivalent": float(equivalent),
            "payment_ids": [p["payment_id"] for p in payments],
            "payments": payments,
        })
    return {"client_id": client_id, "beneficiary_name": beneficiary_name,
            "windows": windows, "excluded_payment_ids": excluded,
            "assumptions": [DATE_ASSUMPTION] + ([FX_ASSUMPTION] if any(
                c != "USD" for w in windows for c in w["totals_by_currency"]) else [])}


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Find repeated names; repetition alone is not a structuring finding."""
    counts = Counter(p["beneficiary_name"] for p in get_client_payments(client_id))
    return [{"beneficiary_name": name, "count": count}
            for name, count in sorted(counts.items()) if name and count > 1]
