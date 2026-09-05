"""
Payment and deterministic analysis tools.

All exact calculations — amounts, counting, aggregation, date-window logic —
happen here rather than in the LLM.
"""

import os
from collections import defaultdict

import pandas as pd
from langchain_core.tools import tool

_PAYMENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "payments.csv",
)

_payments = None


def _load_payments() -> pd.DataFrame:
    """Read payments.csv once and cache it."""
    global _payments
    if _payments is None:
        frame = pd.read_csv(
            _PAYMENTS_CSV,
            dtype={
                "payment_id": str,
                "client_id": str,
                "beneficiary_country_code": str,
            },
        )
        _payments = frame
    return _payments


def _record(row: pd.Series) -> dict:
    """Convert a dataframe row to a plain JSON-serialisable dict."""
    record = row.to_dict()
    record["amount"] = float(record["amount"])
    return record


@tool
def get_payment(payment_id: str) -> dict:
    """Retrieve one payment by payment ID.

    Returns the structured payment record, or ``{"error": ...}`` when the
    payment does not exist.
    """
    if not payment_id:
        return {"error": "No payment_id supplied."}

    payments = _load_payments()
    matches = payments[payments["payment_id"] == str(payment_id).strip()]

    if matches.empty:
        return {"error": f"Payment not found: {payment_id}"}

    return _record(matches.iloc[0])


@tool
def get_client_payments(client_id: str) -> list[dict]:
    """Retrieve a client's full payment history, oldest first.

    Sorted by payment date so same-day clusters — the signal for potential
    transaction splitting — are visible in the result.
    """
    if not client_id:
        return []

    payments = _load_payments()
    matches = payments[payments["client_id"] == str(client_id).strip()]
    matches = matches.sort_values(["payment_date", "payment_id"])

    return [_record(row) for _, row in matches.iterrows()]


@tool
def aggregate_beneficiary_24h(
    client_id: str,
    beneficiary_name: str,
) -> dict:
    """Aggregate payments to one beneficiary within a 24-hour window.

    Filters on **both** ``client_id`` and ``beneficiary_name``: the dataset
    contains same-date payments to the same beneficiary from different
    clients, and same-date payments from the same client to different
    beneficiaries — neither belongs in this aggregation.

    ``payment_date`` carries no time component, so payments sharing a calendar
    date are treated as falling inside the same 24-hour window (stated as an
    assumption in the result).  Every date window is reported; the top-level
    ``count`` / ``total_amount`` describe the largest window, which is the one
    a structuring threshold must be compared against.

    Amounts are also broken out per currency, since the history may mix
    currencies and summing across them would not be meaningful.
    """
    if not client_id or not beneficiary_name:
        return {"error": "client_id and beneficiary_name are both required."}

    payments = _load_payments()
    matches = payments[
        (payments["client_id"] == str(client_id).strip())
        & (
            payments["beneficiary_name"].str.strip().str.lower()
            == str(beneficiary_name).strip().lower()
        )
    ].sort_values(["payment_date", "payment_id"])

    assumption = (
        "payment_date has no time component; payments sharing a calendar date "
        "are treated as being within the same 24-hour window"
    )

    result = {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "window_assumption": assumption,
        "total_payments_to_beneficiary": int(len(matches)),
        "count": 0,
        "total_amount": 0.0,
        "currency": None,
        "totals_by_currency": {},
        "window_date": None,
        "payments": [],
        "windows": [],
    }

    if matches.empty:
        return result

    by_date: dict[str, list[dict]] = defaultdict(list)
    for _, row in matches.iterrows():
        by_date[str(row["payment_date"])].append(_record(row))

    windows = []
    for date, rows in sorted(by_date.items()):
        totals: dict[str, float] = defaultdict(float)
        for row in rows:
            totals[row["currency"]] += row["amount"]

        windows.append(
            {
                "window_date": date,
                "count": len(rows),
                "total_amount": round(sum(row["amount"] for row in rows), 2),
                "totals_by_currency": {
                    currency: round(total, 2) for currency, total in totals.items()
                },
                "payment_ids": [row["payment_id"] for row in rows],
                "payments": rows,
            }
        )

    # A structuring threshold is breached by combined value, not by payment
    # count, so the reported window is the one with the highest total —
    # count only breaks ties between equal-value windows.
    largest = max(windows, key=lambda window: (window["total_amount"], window["count"]))
    currencies = {row["currency"] for row in largest["payments"]}

    result.update(
        {
            "count": largest["count"],
            "total_amount": largest["total_amount"],
            "currency": currencies.pop() if len(currencies) == 1 else "mixed",
            "totals_by_currency": largest["totals_by_currency"],
            "window_date": largest["window_date"],
            "payments": largest["payments"],
            "windows": [
                {key: window[key] for key in window if key != "payments"}
                for window in windows
            ],
        }
    )

    return result


@tool
def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Identify beneficiaries the client paid more than once.

    Flags those with two or more payments on a single calendar date, which is
    the entry point for a structuring check.
    """
    history = get_client_payments.invoke({"client_id": client_id})

    grouped: dict[str, list[dict]] = defaultdict(list)
    for payment in history:
        grouped[payment["beneficiary_name"]].append(payment)

    repeated = []
    for beneficiary, rows in grouped.items():
        if len(rows) < 2:
            continue

        dates: dict[str, int] = defaultdict(int)
        for row in rows:
            dates[str(row["payment_date"])] += 1

        same_day = {date: count for date, count in dates.items() if count > 1}

        repeated.append(
            {
                "beneficiary_name": beneficiary,
                "payment_count": len(rows),
                "payment_ids": [row["payment_id"] for row in rows],
                "same_day_clusters": same_day,
                "has_same_day_cluster": bool(same_day),
            }
        )

    return sorted(repeated, key=lambda item: -item["payment_count"])
