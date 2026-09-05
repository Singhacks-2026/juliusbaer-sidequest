"""Cached CSV reader. Callers receive copies, never mutable cache entries."""

import csv
from datetime import date
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@lru_cache(maxsize=2)
def _rows(filename: str) -> tuple[dict, ...]:
    with (DATA_DIR / filename).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        field = "amount" if filename == "payments.csv" else "relationship_years"
        value = Decimal(row[field])
        if not value.is_finite() or value < 0:
            raise ValueError(f"Invalid {field} in {filename}")
        row[field] = float(value)
        if filename == "payments.csv":
            date.fromisoformat(row["payment_date"])
    return tuple(rows)


def read_rows(filename: str) -> list[dict]:
    return [dict(row) for row in _rows(filename)]
