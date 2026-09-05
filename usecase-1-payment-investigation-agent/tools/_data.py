"""Shared CSV loaders for deterministic tools."""

from functools import lru_cache
import os

import pandas as pd


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


def _native(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def row_to_dict(row: pd.Series) -> dict:
    return {key: _native(value) for key, value in row.to_dict().items()}


@lru_cache(maxsize=1)
def load_clients() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))


@lru_cache(maxsize=1)
def load_payments() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "payments.csv"))
    df["payment_date"] = df["payment_date"].astype(str)
    return df
