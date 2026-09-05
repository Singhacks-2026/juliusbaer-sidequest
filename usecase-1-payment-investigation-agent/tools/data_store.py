"""
Shared CSV access for the data tools.

Loads ``data/clients.csv`` and ``data/payments.csv`` once and keeps them in
module-level caches.  Paths are resolved relative to this file rather than the
current working directory, so the tools work regardless of where the organizer
invokes ``main.py`` from.
"""

import csv
import os
from typing import Any

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

POLICY_DIR = os.path.join(_DATA_DIR, "policies")

_NUMERIC_FIELDS = {"amount", "relationship_years"}

_clients_by_id: dict[str, dict] | None = None
_payments_by_id: dict[str, dict] | None = None
_payments_by_client: dict[str, list[dict]] | None = None


def _coerce(field: str, value: str) -> Any:
    """Convert known numeric columns to float, leave everything else as text."""
    if field in _NUMERIC_FIELDS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return (value or "").strip()


def _read_csv(filename: str) -> list[dict]:
    path = os.path.join(_DATA_DIR, filename)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [
            {field: _coerce(field, value) for field, value in row.items() if field}
            for row in csv.DictReader(handle)
        ]


def _load() -> None:
    global _clients_by_id, _payments_by_id, _payments_by_client

    if _clients_by_id is not None:
        return

    _clients_by_id = {row["client_id"]: row for row in _read_csv("clients.csv")}

    _payments_by_id = {}
    _payments_by_client = {}

    for row in _read_csv("payments.csv"):
        _payments_by_id[row["payment_id"]] = row
        _payments_by_client.setdefault(row["client_id"], []).append(row)

    # Deterministic ordering makes aggregation output stable across runs.
    for payments in _payments_by_client.values():
        payments.sort(key=lambda row: (row["payment_date"], row["payment_id"]))


def clients() -> dict[str, dict]:
    _load()
    return _clients_by_id


def payments() -> dict[str, dict]:
    _load()
    return _payments_by_id


def payments_for_client(client_id: str) -> list[dict]:
    _load()
    return _payments_by_client.get((client_id or "").strip(), [])
