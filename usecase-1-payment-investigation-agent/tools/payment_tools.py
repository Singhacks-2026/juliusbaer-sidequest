"""
Payment lookup and deterministic pattern-analysis tools.

Every amount, count, total and date comparison in the assistant's answers
originates here.  The LLM never performs arithmetic.
"""

from tools.data_store import payments, payments_for_client

# payment_date has no time component, so a 24-hour window is approximated by
# the calendar date.  DATA_NOTES.md sanctions this and asks that it be stated.
_WINDOW_ASSUMPTION = (
    "payment_date carries no time component, so payments sharing a calendar "
    "date are treated as falling inside the same 24-hour window"
)

_MIXED_CURRENCY_ASSUMPTION = (
    "the window mixes currencies and no exchange-rate data is provided, so "
    "amounts are summed 1:1"
)


def _public_view(record: dict) -> dict:
    """Project a raw CSV row into the payment shape the agent consumes."""
    return {
        "payment_id": record["payment_id"],
        "client_id": record["client_id"],
        "beneficiary_name": record["beneficiary_name"],
        "beneficiary_country": record["beneficiary_country"],
        "beneficiary_country_code": record["beneficiary_country_code"],
        "amount": record["amount"],
        "currency": record["currency"],
        "channel": record["channel"],
        "payment_date": record["payment_date"],
    }


def get_payment(payment_id: str) -> dict:
    """
    Retrieve one payment by ID.

    ``beneficiary_country`` and ``beneficiary_country_code`` disagree for some
    rows by design.  Both are returned, and the record flags the conflict so
    the agent can report it as an observed fact; the code is authoritative for
    jurisdiction risk (see DATA_NOTES.md).
    """
    payment_id = (payment_id or "").strip()
    record = payments().get(payment_id)

    if record is None:
        return {
            "found": False,
            "payment_id": payment_id,
            "error": f"No payment record for {payment_id!r} in payments.csv.",
        }

    view = _public_view(record)
    view["found"] = True
    view["authoritative_jurisdiction_field"] = "beneficiary_country_code"
    return view


def get_client_payments(client_id: str) -> list[dict]:
    """Retrieve a client's full payment history, oldest first."""
    return [_public_view(record) for record in payments_for_client(client_id)]


def aggregate_beneficiary_24h(client_id: str, beneficiary_name: str) -> dict:
    """
    Aggregate a client's payments to one beneficiary inside a 24-hour window.

    Filters on **both** ``client_id`` and ``beneficiary_name``: the dataset
    contains same-date payments to one beneficiary from different clients, and
    same-date payments from one client to different beneficiaries, so either
    filter alone overstates the total.

    Returns every window containing more than one payment plus ``largest_window``,
    the highest-value window, which is the one a structuring check compares
    against the policy threshold.
    """
    client_id = (client_id or "").strip()
    beneficiary_name = (beneficiary_name or "").strip()

    matching = [
        record
        for record in payments_for_client(client_id)
        if record["beneficiary_name"].casefold() == beneficiary_name.casefold()
    ]

    by_date: dict[str, list[dict]] = {}
    for record in matching:
        by_date.setdefault(record["payment_date"], []).append(record)

    windows = [_summarize_window(date, rows) for date, rows in sorted(by_date.items())]
    multi = [window for window in windows if window["count"] > 1]
    largest = max(windows, key=lambda w: w["total_amount"], default=None)

    return {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "filtered_on": ["client_id", "beneficiary_name"],
        "window_basis": "calendar_date",
        "assumption": _WINDOW_ASSUMPTION,
        "payments_to_beneficiary": len(matching),
        "windows_with_multiple_payments": len(multi),
        "largest_window": largest,
        "windows": windows,
    }


def _summarize_window(date: str, rows: list[dict]) -> dict:
    """Total one calendar-date window, flagging mixed-currency sums."""
    currencies = sorted({row["currency"] for row in rows})
    amounts = [row["amount"] or 0.0 for row in rows]

    window = {
        "payment_date": date,
        "count": len(rows),
        "total_amount": round(sum(amounts), 2),
        "currencies": currencies,
        "mixed_currency": len(currencies) > 1,
        "currency": currencies[0] if len(currencies) == 1 else None,
        "channels": sorted({row["channel"] for row in rows}),
        "payment_ids": [row["payment_id"] for row in rows],
        "payments": [_public_view(row) for row in rows],
    }

    if window["mixed_currency"]:
        window["assumption"] = _MIXED_CURRENCY_ASSUMPTION

    return window


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """
    Identify beneficiaries a client paid more than once.

    Used to decide which beneficiaries are worth running through
    ``aggregate_beneficiary_24h`` when the question is about transaction
    splitting but names no beneficiary.
    """
    history = payments_for_client(client_id)

    grouped: dict[str, list[dict]] = {}
    for record in history:
        grouped.setdefault(record["beneficiary_name"], []).append(record)

    repeated = []
    for name, rows in grouped.items():
        if len(rows) < 2:
            continue

        dates = [row["payment_date"] for row in rows]
        same_date = sorted({date for date in dates if dates.count(date) > 1})

        repeated.append(
            {
                "beneficiary_name": name,
                "payment_count": len(rows),
                "payment_ids": [row["payment_id"] for row in rows],
                "dates": sorted(dates),
                "dates_with_multiple_payments": same_date,
            }
        )

    # Same-day repetition is the structuring-relevant signal, so surface it first.
    repeated.sort(
        key=lambda item: (len(item["dates_with_multiple_payments"]), item["payment_count"]),
        reverse=True,
    )
    return repeated
