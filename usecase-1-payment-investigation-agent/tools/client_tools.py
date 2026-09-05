"""
Client data-access tools.

Deterministic lookups over ``data/clients.csv``. The agent calls these rather
than reading the CSV directly, which keeps a clean separation between AI
orchestration, data access and answer generation.
"""

from __future__ import annotations

from tools._data import clients_df, row_to_dict


def get_client_profile(client_id: str) -> dict:
    """
    Retrieve one client's profile.

    Returns ``{"found": False, "client_id": ..., "error": ...}`` for unknown
    IDs rather than raising, so the agent can explain the missing evidence.
    """
    client_id = (client_id or "").strip()
    df = clients_df()
    match = df[df["client_id"] == client_id]
    if match.empty:
        return {"found": False, "client_id": client_id, "error": "client not found"}
    record = row_to_dict(match.iloc[0])
    record["found"] = True
    return record


def get_clients_by_country(country: str) -> list[dict]:
    """Retrieve every client whose relationship country matches (case-insensitive)."""
    country = (country or "").strip().lower()
    df = clients_df()
    match = df[df["country"].str.lower() == country]
    return [row_to_dict(r) for _, r in match.iterrows()]
