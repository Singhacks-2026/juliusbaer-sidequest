"""Payment facts and same-client, same-beneficiary daily aggregation."""
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from tools.data_access import read_rows

WINDOW_ASSUMPTION = (
    "Payment dates have no times. Under DATA_NOTES.md, payments on the same "
    "calendar date are treated as within one 24-hour window."
)

def get_payment(payment_id: str) -> dict:
    """Return one payment, or an empty dict for an unknown ID."""
    return next((row for row in read_rows("payments.csv")
                 if row["payment_id"] == payment_id), {})

def get_client_payments(client_id: str) -> list[dict]:
    """Return the client's complete history, ordered by date and ID."""
    return sorted((row for row in read_rows("payments.csv")
                   if row["client_id"] == client_id),
                  key=lambda row: (row["payment_date"], row["payment_id"]))

def aggregate_beneficiary_24h(
    client_id: str, beneficiary_name: str, payment_date: str | None = None,
) -> dict:
    """Return daily windows; keep currencies separate without FX conversion."""
    if payment_date is not None:
        date.fromisoformat(payment_date)
    matching = [row for row in get_client_payments(client_id)
                if row["beneficiary_name"] == beneficiary_name
                and (payment_date is None or row["payment_date"] == payment_date)]
    groups = defaultdict(list)
    for row in matching:
        day = date.fromisoformat(row["payment_date"]).isoformat()
        groups[(day, row["currency"])].append(row)
    windows = []
    for (day, currency), rows in sorted(groups.items()):
        windows.append({
            "payment_date": day, "currency": currency, "count": len(rows),
            "total_amount": float(sum((Decimal(str(row["amount"])) for row in rows), Decimal(0))),
            "payment_ids": [row["payment_id"] for row in rows],
            "individual_amounts": [row["amount"] for row in rows],
            "channels": sorted({row["channel"] for row in rows}),
            "payments": rows,
        })
    return {
        "client_id": client_id, "beneficiary_name": beneficiary_name,
        "matched_payment_count": len(matching), "windows": windows,
        "assumptions": [WINDOW_ASSUMPTION],
        "currency_handling": "Daily totals are separate for each currency; no FX conversion has been performed.",
    }

def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Find repeated names in history; repetition alone is not structuring."""
    counts = Counter(row["beneficiary_name"] for row in get_client_payments(client_id))
    return [{"beneficiary_name": name, "count": count}
            for name, count in sorted(counts.items()) if count > 1]
