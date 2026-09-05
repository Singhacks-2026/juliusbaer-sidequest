"""CSV loading independent of the caller's working directory."""
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def read_records(filename: str) -> list[dict]:
    with (DATA_DIR / filename).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            row[key] = value.strip() if value else None
        for key in ("amount", "relationship_years"):
            if key in row and row[key] is not None:
                row[key] = float(row[key])
    return rows
