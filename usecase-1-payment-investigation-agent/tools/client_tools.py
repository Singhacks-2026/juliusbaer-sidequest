"""Deterministic lookups of the supplied synthetic client records."""
from tools.data_access import read_rows

def get_client_profile(client_id: str) -> dict:
    """Return a client profile, or an empty dict for an unknown ID."""
    return next((row for row in read_rows("clients.csv")
                 if row["client_id"] == client_id), {})

def get_clients_by_country(country: str) -> list[dict]:
    """Return all clients whose relationship country matches."""
    return [row for row in read_rows("clients.csv")
            if row["country"].casefold() == country.strip().casefold()]
