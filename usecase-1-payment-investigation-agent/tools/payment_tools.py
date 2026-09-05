"""
Payment data-access and deterministic analysis tools.

All arithmetic, counting, and window logic lives here — never in the
LLM. Per DATA_NOTES, ``payment_date`` has no time component, so the
24-hour window is approximated by calendar date (stated in answers where
it matters), and aggregation filters by both client and beneficiary.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "payments.csv",
)

_payments: list[dict] | None = None
_by_id: dict[str, dict] | None = None


def _load() -> tuple[list[dict], dict[str, dict]]:
    global _payments, _by_id
    if _payments is None:
        _payments = []
        _by_id = {}
        with open(_DATA_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rec = {
                    "payment_id": row["payment_id"],
                    "client_id": row["client_id"],
                    "beneficiary_name": row["beneficiary_name"],
                    "beneficiary_country": row["beneficiary_country"],
                    "beneficiary_country_code": row[
                        "beneficiary_country_code"
                    ],
                    "amount": float(row["amount"]),
                    "currency": row["currency"],
                    "channel": row["channel"],
                    "payment_date": row["payment_date"],
                }
                _payments.append(rec)
                _by_id[rec["payment_id"]] = rec
    assert _payments is not None and _by_id is not None
    return _payments, _by_id


def get_payment(payment_id: str) -> dict:
    """Return one payment record, or an error dict when unknown."""
    _, by_id = _load()
    if payment_id not in by_id:
        return {"error": f"unknown payment_id: {payment_id}"}
    return dict(by_id[payment_id])


def get_client_payments(client_id: str) -> list[dict]:
    """Return every payment for a client, oldest date first."""
    payments, _ = _load()
    rows = [dict(p) for p in payments if p["client_id"] == client_id]
    rows.sort(key=lambda r: (r["payment_date"], r["payment_id"]))
    return rows


def aggregate_beneficiary_24h(
    client_id: str,
    beneficiary_name: str,
) -> dict:
    """Aggregate one client's payments to one beneficiary per date window.

    Returns the strongest (highest-total) window plus all windows, so the
    agent can compare combined totals against the structuring threshold.
    """
    rows = [
        p
        for p in get_client_payments(client_id)
        if p["beneficiary_name"] == beneficiary_name
    ]
    windows: dict[str, list[dict]] = defaultdict(list)
    for p in rows:
        windows[p["payment_date"]].append(p)

    summaries = []
    for date in sorted(windows):
        group = windows[date]
        currencies = sorted({p["currency"] for p in group})
        summaries.append(
            {
                "window_date": date,
                "count": len(group),
                "total_amount": round(sum(p["amount"] for p in group), 2),
                "currency": currencies[0] if len(currencies) == 1 else currencies,
                "mixed_currency": len(currencies) > 1,
                "channels": sorted({p["channel"] for p in group}),
                "payment_ids": [p["payment_id"] for p in group],
                "payments": group,
            }
        )
    strongest = (
        max(summaries, key=lambda s: s["total_amount"]) if summaries else None
    )
    return {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "window_note": "calendar-date window (≈24h; dates carry no time)",
        "windows": summaries,
        "strongest_window": strongest,
    }


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """List beneficiaries paid more than once, with their date windows."""
    rows = get_client_payments(client_id)
    by_beneficiary: dict[str, list[dict]] = defaultdict(list)
    for p in rows:
        by_beneficiary[p["beneficiary_name"]].append(p)
    out = []
    for name, group in by_beneficiary.items():
        if len(group) > 1:
            dates = sorted({p["payment_date"] for p in group})
            out.append(
                {
                    "beneficiary_name": name,
                    "count": len(group),
                    "dates": dates,
                    "same_date_repeat": len(dates) < len(group),
                    "payment_ids": [p["payment_id"] for p in group],
                }
            )
    out.sort(key=lambda r: (-r["count"], r["beneficiary_name"]))
    return out
