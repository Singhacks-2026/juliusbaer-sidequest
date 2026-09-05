"""Read supplied CSVs using paths relative to this module."""
import csv
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@lru_cache(maxsize=2)
def _load_rows(filename: str) -> tuple[dict, ...]:
    with (DATA_DIR / filename).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric = "amount" if filename == "payments.csv" else "relationship_years"
    for row in rows:
        value = Decimal(row[numeric])
        if not value.is_finite() or value < 0:
            raise ValueError(f"Invalid {numeric} in {filename}")
        row[numeric] = float(value)
    return tuple(rows)

def read_rows(filename: str) -> list[dict]:
    # Callers must not mutate the cached source records.
    return deepcopy(list(_load_rows(filename)))
