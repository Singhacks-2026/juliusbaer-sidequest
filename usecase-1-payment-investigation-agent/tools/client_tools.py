"""Deterministic access to the supplied client data."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

_CLIENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "clients.csv"


def _coerce(value: str) -> Any:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


@lru_cache(maxsize=1)
def _clients() -> tuple[dict, ...]:
    with _CLIENTS_PATH.open(newline="", encoding="utf-8") as handle:
        return tuple(
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        )


def get_client_profile(client_id: str) -> dict:
    """Return a client profile, or an empty dict when the ID is unknown."""
    wanted = str(client_id).strip().upper()
    return next((dict(row) for row in _clients() if row["client_id"] == wanted), {})


def get_clients_by_country(country: str) -> list[dict]:
    """Return clients whose relationship country matches case-insensitively."""
    wanted = str(country).strip().casefold()
    return [
        dict(row)
        for row in _clients()
        if str(row.get("country", "")).casefold() == wanted
    ]
