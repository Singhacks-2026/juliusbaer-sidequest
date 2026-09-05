"""Deterministic access to supplied client profiles."""
from tools.data_store import read_records


def get_client_profile(client_id: str) -> dict:
    """Return a client record, or an explicit not-found error."""
    return next((r for r in read_records("clients.csv") if r["client_id"] == client_id),
                {"error": "Client not found", "client_id": client_id})


def get_clients_by_country(country: str) -> list[dict]:
    """Return clients in a country (case-insensitive)."""
    return [r for r in read_records("clients.csv")
            if (r["country"] or "").casefold() == country.strip().casefold()]
