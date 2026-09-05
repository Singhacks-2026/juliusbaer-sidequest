"""
Client data-access tool interfaces.

These methods intentionally contain NO implementations.

The candidate must implement the data lookup using the supplied
``data/clients.csv`` file.

The AI agent should interact with these methods rather than directly reading
the CSV. This creates a clean separation between:
    - AI orchestration;
    - deterministic data access;
    - final answer generation.
"""

import csv
from functools import lru_cache
from pathlib import Path


_CLIENTS_PATH = Path(__file__).resolve().parents[1] / "data" / "clients.csv"


@lru_cache(maxsize=1)
def _clients() -> tuple[dict, ...]:
    with _CLIENTS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            row["relationship_years"] = float(row["relationship_years"])
            rows.append(row)
        return tuple(rows)


def get_client_profile(client_id: str) -> dict:
    """
    Retrieve one client's profile.

    Parameters
    ----------
    client_id:
        Example: ``"C2001"``.

    Returns
    -------
    dict
        Client information including country, risk rating, client type and
        relationship duration.

    Implementation requirement
    --------------------------
    Return a structured result for known clients and handle unknown IDs
    gracefully.
    """
    key = client_id.strip().upper()
    return next((dict(row) for row in _clients() if row["client_id"] == key), {})


def get_clients_by_country(country: str) -> list[dict]:
    """
    OPTIONAL: Retrieve clients associated with a given country.

    Parameters
    ----------
    country:
        Country name.

    Returns
    -------
    list[dict]
        Matching client records.
    """
    key = country.strip().casefold()
    return [dict(row) for row in _clients() if row["country"].casefold() == key]
