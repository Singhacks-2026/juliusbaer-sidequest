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

import os

import pandas as pd

_CLIENTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "clients.csv",
)

_clients_df = None


def _get_clients_df() -> pd.DataFrame:
    global _clients_df
    if _clients_df is None:
        _clients_df = pd.read_csv(_CLIENTS_CSV)
    return _clients_df


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
    df = _get_clients_df()
    matches = df[df["client_id"] == client_id]
    if matches.empty:
        return {}
    return matches.iloc[0].to_dict()


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
    df = _get_clients_df()
    matches = df[df["country"].str.lower() == country.lower()]
    return matches.to_dict(orient="records")
