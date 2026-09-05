"""
Payment data-access and aggregation tools.

Everything in this module is deterministic pandas logic. All arithmetic the
assistant reports - amounts, counts, 24-hour totals - is computed here, never
by the language model.

Data notes that shape the implementation (see DATA_NOTES.md):

* ``payment_date`` has no time component. Payments on the same calendar date
  are treated as falling inside the same 24-hour window.
* The 24-hour aggregation filters on BOTH ``client_id`` and
  ``beneficiary_name``. The dataset deliberately contains same-date payments
  to the same beneficiary from *other* clients, and same-date payments from
  the same client to *other* beneficiaries; neither may leak into the total.
"""

from __future__ import annotations

from tools._data import payments_df, row_to_dict


def get_payment(payment_id: str) -> dict:
    """Retrieve one payment by ID, or a clear not-found result."""
    payment_id = (payment_id or "").strip()
    df = payments_df()
    match = df[df["payment_id"] == payment_id]
    if match.empty:
        return {"found": False, "payment_id": payment_id, "error": "payment not found"}
    record = row_to_dict(match.iloc[0])
    record["found"] = True
    return record


def get_client_payments(client_id: str) -> list[dict]:
    """All supplied payments for a client, oldest first."""
    client_id = (client_id or "").strip()
    df = payments_df()
    match = df[df["client_id"] == client_id].sort_values(["payment_date", "payment_id"])
    return [row_to_dict(r) for _, r in match.iterrows()]


def aggregate_beneficiary_24h(client_id: str, beneficiary_name: str) -> dict:
    """
    Aggregate a client's payments to one beneficiary within a 24-hour window.

    Because dates carry no time, each calendar date is one window. The result
    reports the busiest window (most payments, then highest total) plus every
    window found, so the caller can see the full picture.

    Amounts are summed per currency; ``total_amount`` is the sum across
    currencies at an explicit 1:1 assumption, flagged in ``assumptions``.
    """
    client_id = (client_id or "").strip()
    beneficiary_name = (beneficiary_name or "").strip()
    df = payments_df()
    subset = df[
        (df["client_id"] == client_id)
        & (df["beneficiary_name"].str.strip().str.lower() == beneficiary_name.lower())
    ]

    windows = []
    for date, group in subset.groupby("payment_date"):
        group = group.sort_values("payment_id")
        by_currency = {
            cur: round(float(amt), 2)
            for cur, amt in group.groupby("currency")["amount"].sum().items()
        }
        windows.append(
            {
                "window_date": date,
                "count": int(len(group)),
                "total_amount": round(float(group["amount"].sum()), 2),
                "totals_by_currency": by_currency,
                "currencies": sorted(group["currency"].unique().tolist()),
                "channels": group["channel"].tolist(),
                "payment_ids": group["payment_id"].tolist(),
                "individual_amounts": [round(float(a), 2) for a in group["amount"]],
                "payments": [row_to_dict(r) for _, r in group.iterrows()],
            }
        )

    windows.sort(key=lambda w: (w["count"], w["total_amount"]), reverse=True)
    busiest = windows[0] if windows else None

    result = {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "window_definition": "same calendar date (payment_date has no time component)",
        "assumptions": [
            "Payments on the same calendar date are treated as within 24 hours.",
        ],
        "count": busiest["count"] if busiest else 0,
        "total_amount": busiest["total_amount"] if busiest else 0.0,
        "window_date": busiest["window_date"] if busiest else None,
        "currencies": busiest["currencies"] if busiest else [],
        "totals_by_currency": busiest["totals_by_currency"] if busiest else {},
        "payment_ids": busiest["payment_ids"] if busiest else [],
        "individual_amounts": busiest["individual_amounts"] if busiest else [],
        "channels": busiest["channels"] if busiest else [],
        "payments": busiest["payments"] if busiest else [],
        "all_windows": [
            {k: v for k, v in w.items() if k != "payments"} for w in windows
        ],
    }
    if busiest and len(busiest["currencies"]) > 1:
        result["assumptions"].append(
            "Window contains more than one currency; total_amount sums them 1:1."
        )
    return result


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Beneficiaries that appear more than once in a client's history."""
    client_id = (client_id or "").strip()
    df = payments_df()
    subset = df[df["client_id"] == client_id]
    out = []
    for name, group in subset.groupby("beneficiary_name"):
        if len(group) < 2:
            continue
        out.append(
            {
                "beneficiary_name": name,
                "count": int(len(group)),
                "total_amount": round(float(group["amount"].sum()), 2),
                "dates": sorted(group["payment_date"].unique().tolist()),
                "payment_ids": group.sort_values("payment_date")["payment_id"].tolist(),
            }
        )
    out.sort(key=lambda r: (r["count"], r["total_amount"]), reverse=True)
    return out
