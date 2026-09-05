"""
Client data-access tools (deterministic).

Reads ``data/clients.csv`` with the standard library and caches rows in
memory. The agent must call these instead of opening the CSV directly.
"""

from __future__ import annotations

import csv
import os

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clients.csv",
)

_clients: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _clients
    if _clients is None:
        _clients = {}
        with open(_DATA_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                _clients[row["client_id"]] = {
                    "client_id": row["client_id"],
                    "country": row["country"],
                    "risk_rating": row["risk_rating"],
                    "client_type": row["client_type"],
                    "relationship_years": float(row["relationship_years"] or 0),
                }
    return _clients


def get_client_profile(client_id: str) -> dict:
    """Return one client's profile, or an error dict for unknown IDs."""
    clients = _load()
    if client_id not in clients:
        return {"error": f"unknown client_id: {client_id}"}
    return dict(clients[client_id])


def get_clients_by_country(country: str) -> list[dict]:
    """Return all client records for a country name (case-insensitive)."""
    wanted = country.strip().lower()
    return [
        dict(c) for c in _load().values() if c["country"].lower() == wanted
    ]
