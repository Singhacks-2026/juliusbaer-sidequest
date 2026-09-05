"""Payment lookup and deterministic analysis tools."""

from collections import defaultdict

from tools._data import load_payments, row_to_dict


def get_payment(payment_id: str) -> dict:
    """Retrieve one payment by payment ID."""
    if not payment_id:
        return {"error": "payment_id is required"}

    payments = load_payments()
    matches = payments[payments["payment_id"] == payment_id]
    if matches.empty:
        return {"error": f"Unknown payment_id: {payment_id}"}
    return row_to_dict(matches.iloc[0])


def get_client_payments(client_id: str) -> list[dict]:
    """Retrieve the supplied payment history for a client."""
    if not client_id:
        return []

    payments = load_payments()
    matches = payments[payments["client_id"] == client_id]
    records = [row_to_dict(row) for _, row in matches.iterrows()]
    records.sort(key=lambda item: (item.get("payment_date") or "", item.get("payment_id") or ""))
    return records


def aggregate_beneficiary_24h(
    client_id: str,
    beneficiary_name: str,
    payment_id: str | None = None,
    payment_date: str | None = None,
) -> dict:
    """Aggregate same-beneficiary payments on the same calendar date.

    ``payment_date`` has no time component, so same-date payments are treated
    as one 24-hour window. Both ``client_id`` and ``beneficiary_name`` must
    match.

    When ``payment_id`` or ``payment_date`` is supplied, the primary window is
    the one containing that payment / date (not the globally max window).
    """
    if not client_id or not beneficiary_name:
        return {
            "error": "client_id and beneficiary_name are required",
            "count": 0,
            "total_amount": 0,
            "payments": [],
            "windows": [],
        }

    history = get_client_payments(client_id)
    matching = [
        payment
        for payment in history
        if (payment.get("beneficiary_name") or "").casefold()
        == beneficiary_name.casefold()
    ]

    windows_by_date: dict[str, list[dict]] = defaultdict(list)
    for payment in matching:
        windows_by_date[payment.get("payment_date") or "unknown"].append(payment)

    windows = []
    for date, group in sorted(windows_by_date.items()):
        totals_by_currency: dict[str, float] = defaultdict(float)
        for payment in group:
            ccy = payment.get("currency") or "UNKNOWN"
            totals_by_currency[ccy] += float(payment.get("amount") or 0)

        if len(totals_by_currency) == 1:
            currency, total_amount = next(iter(totals_by_currency.items()))
        else:
            currency = ",".join(sorted(totals_by_currency))
            total_amount = sum(totals_by_currency.values())

        windows.append(
            {
                "date": date,
                "count": len(group),
                "total_amount": total_amount,
                "totals_by_currency": dict(totals_by_currency),
                "currency": currency,
                "payment_ids": [payment["payment_id"] for payment in group],
                "channels": [payment.get("channel") for payment in group],
                "payments": group,
            }
        )

    assumption = (
        "Same calendar date is treated as one 24-hour window "
        "because payment_date has no time component."
    )

    if not windows:
        return {
            "client_id": client_id,
            "beneficiary_name": beneficiary_name,
            "assumption": assumption,
            "count": 0,
            "total_amount": 0,
            "payments": [],
            "windows": [],
            "scoped_to_payment_id": payment_id,
            "scoped_to_payment_date": payment_date,
        }

    selected = None
    if payment_id:
        for window in windows:
            if payment_id in window["payment_ids"]:
                selected = window
                break
    if selected is None and payment_date:
        for window in windows:
            if window["date"] == payment_date:
                selected = window
                break
    if selected is None:
        # Fall back to densest window only when no scope was provided.
        selected = max(windows, key=lambda item: (item["count"], item["total_amount"]))

    return {
        "client_id": client_id,
        "beneficiary_name": beneficiary_name,
        "assumption": assumption,
        "count": selected["count"],
        "total_amount": selected["total_amount"],
        "currency": selected["currency"],
        "date": selected["date"],
        "payment_ids": selected["payment_ids"],
        "channels": selected["channels"],
        "payments": selected["payments"],
        "windows": windows,
        "scoped_to_payment_id": payment_id,
        "scoped_to_payment_date": payment_date,
    }


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Identify beneficiaries appearing more than once for a client."""
    history = get_client_payments(client_id)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for payment in history:
        name = payment.get("beneficiary_name")
        if name:
            grouped[name].append(payment)

    results = []
    for name, group in grouped.items():
        if len(group) < 2:
            continue
        results.append(
            {
                "beneficiary_name": name,
                "count": len(group),
                "payment_ids": [payment["payment_id"] for payment in group],
                "dates": sorted({payment.get("payment_date") for payment in group}),
            }
        )
    results.sort(key=lambda item: item["count"], reverse=True)
    return results
