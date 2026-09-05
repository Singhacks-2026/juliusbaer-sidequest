"""Deterministic client lookups; country determines the regional policy."""

from tools.data import read_rows


def get_client_profile(client_id: str) -> dict:
    return next(
        (row for row in read_rows("clients.csv") if row["client_id"] == client_id),
        {"error": "Client not found", "client_id": client_id},
    )


def get_clients_by_country(country: str) -> list[dict]:
    return [row for row in read_rows("clients.csv")
            if row["country"].casefold() == country.casefold()]
