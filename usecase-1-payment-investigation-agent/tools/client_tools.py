"""
Client data-access tools.

The AI agent interacts with these methods rather than reading ``clients.csv``
directly, keeping AI orchestration separate from deterministic data access.
"""

import os

import pandas as pd
from langchain_core.tools import tool

_CLIENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clients.csv",
)

_clients = None


def _load_clients() -> pd.DataFrame:
    """Read clients.csv once and cache it."""
    global _clients
    if _clients is None:
        _clients = pd.read_csv(_CLIENTS_CSV, dtype={"client_id": str})
    return _clients


@tool
def get_client_profile(client_id: str) -> dict:
    """Retrieve one client's profile.

    Parameters
    ----------
    client_id:
        Example: ``"C2001"``.

    Returns
    -------
    dict
        Client country, risk rating, client type and relationship duration,
        or ``{"error": ...}`` when the client is unknown.
    """
    if not client_id:
        return {"error": "No client_id supplied."}

    clients = _load_clients()
    matches = clients[clients["client_id"] == str(client_id).strip()]

    if matches.empty:
        return {"error": f"Client not found: {client_id}"}

    return matches.iloc[0].to_dict()


@tool
def get_clients_by_country(country: str) -> list[dict]:
    """Retrieve clients associated with a given country (case-insensitive)."""
    if not country:
        return []

    clients = _load_clients()
    matches = clients[
        clients["country"].str.lower() == str(country).strip().lower()
    ]

    return matches.to_dict(orient="records")
