"""
Shared, cached loaders for the two structured datasets.

Both CSVs are read once per process and reused by every tool call. Nothing
here does any policy reasoning - it is pure data access.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


@lru_cache(maxsize=1)
def payments_df() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"), dtype=str)
    df["amount"] = df["amount"].astype(float)
    return df


@lru_cache(maxsize=1)
def clients_df() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"), dtype=str)
    df["relationship_years"] = df["relationship_years"].astype(float)
    return df


def row_to_dict(row: pd.Series) -> dict:
    """Convert a pandas row to plain JSON-safe Python types."""
    out = {}
    for key, value in row.items():
        if hasattr(value, "item"):  # numpy scalar
            value = value.item()
        out[key] = value
    return out
