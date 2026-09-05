"""
Client data-access tools.

Reads from ``data/clients.csv`` via the shared cached loader.
"""

from __future__ import annotations

from tools._data import load_clients, record_to_dict


def get_client_profile(client_id: str) -> dict:
    """
    Retrieve one client's profile.

    Returns a structured result for known clients and a clear error dict
    for unknown IDs.
    """
    df = load_clients()
    matches = df[df["client_id"] == str(client_id)]
    if matches.empty:
        return {
            "error": "client_not_found",
            "client_id": client_id,
            "message": f"No client found with client_id={client_id}",
        }
    return record_to_dict(matches.iloc[0])


def get_clients_by_country(country: str) -> list[dict]:
    """Retrieve clients associated with a given country."""
    df = load_clients()
    matches = df[df["country"].str.casefold() == str(country).casefold()]
    return [record_to_dict(row) for _, row in matches.iterrows()]
