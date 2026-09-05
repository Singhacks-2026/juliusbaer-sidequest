"""
Payment and deterministic analysis tool interfaces.

These methods intentionally contain NO implementations.

Exact calculations should happen in these tools, not in the LLM.
"""

import os

import pandas as pd

_PAYMENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "payments.csv",
)

_payments_df = None


def _get_payments_df() -> pd.DataFrame:
    global _payments_df
    if _payments_df is None:
        _payments_df = pd.read_csv(_PAYMENTS_CSV)
    return _payments_df


def get_payment(payment_id: str) -> dict:
    """
    Retrieve one payment by payment ID.

    The implementation should read from ``data/payments.csv``.

    Returns a structured payment record or a clear empty/error result when
    the payment does not exist.
    """
    df = _get_payments_df()
    matches = df[df["payment_id"] == payment_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


def get_client_payments(client_id: str) -> list[dict]:
    """
    Retrieve the supplied payment history for a client.

    Useful for transaction-pattern and structuring questions.
    """
    df = _get_payments_df()
    matches = df[df["client_id"] == client_id]
    return matches.to_dict(orient="records")


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

    Implementation notes
    ---------------------
    ``payment_date`` carries no time component, so a "24-hour window" is
    treated as "same calendar date" per DATA_NOTES.md. Filtering requires
    matching both ``client_id`` AND ``beneficiary_name`` -- the dataset
    intentionally contains same-beneficiary/different-client and
    same-client/different-beneficiary collisions on the same date.
    """
    df = _get_payments_df()
    matches = df[
        (df["client_id"] == client_id)
        & (df["beneficiary_name"] == beneficiary_name)
    ]

    if matches.empty:
        return {"count": 0, "total_amount": 0, "payments": []}

    results = []
    for date, group in matches.groupby("payment_date"):
        results.append(
            {
                "payment_date": date,
                "count": len(group),
                "total_amount": float(group["amount"].sum()),
                "currency": group["currency"].iloc[0],
                "payments": group.to_dict(orient="records"),
            }
        )

    # Return the window with the largest combined amount -- the one most
    # relevant to a structuring investigation.
    best = max(results, key=lambda r: r["total_amount"])
    return best


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """
    OPTIONAL: Identify beneficiaries appearing multiple times in the
    client's payment history.

    Useful for potential structuring analysis.
    """
    df = _get_payments_df()
    matches = df[df["client_id"] == client_id]
    counts = matches.groupby("beneficiary_name").size()
    repeated = counts[counts > 1]

    return [
        {
            "beneficiary_name": name,
            "count": int(count),
            "payments": matches[matches["beneficiary_name"] == name].to_dict(
                orient="records"
            ),
        }
        for name, count in repeated.items()
    ]
