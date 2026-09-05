"""Cached CSV loaders for client and payment tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=1)
def load_clients() -> pd.DataFrame:
    path = _DATA_DIR / "clients.csv"
    df = pd.read_csv(path)
    df["client_id"] = df["client_id"].astype(str)
    return df


@lru_cache(maxsize=1)
def load_payments() -> pd.DataFrame:
    path = _DATA_DIR / "payments.csv"
    df = pd.read_csv(path)
    df["payment_id"] = df["payment_id"].astype(str)
    df["client_id"] = df["client_id"].astype(str)
    df["payment_date"] = df["payment_date"].astype(str)
    return df


def record_to_dict(row: pd.Series) -> dict:
    out = {}
    for key, value in row.items():
        if pd.isna(value):
            out[key] = None
        elif hasattr(value, "item"):
            out[key] = value.item()
        else:
            out[key] = value
    return out
