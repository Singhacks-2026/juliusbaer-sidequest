"""
Client data-access tools.

The agent reads client context through these functions rather than touching
``data/clients.csv`` directly, which keeps AI orchestration separate from
deterministic data access.
"""

from tools.data_store import clients

# Only these two countries have a regional procedure in the policy corpus.
# Every other client country is governed by the global policy alone.
_REGIONAL_POLICIES = {
    "Singapore": "regional_singapore.md",
    "Switzerland": "regional_switzerland.md",
}


def get_client_profile(client_id: str) -> dict:
    """
    Retrieve one client's profile.

    Returns the client record augmented with the regional policy that applies
    to it, or ``{"found": False, ...}`` for an unknown ID.
    """
    client_id = (client_id or "").strip()
    record = clients().get(client_id)

    if record is None:
        return {
            "found": False,
            "client_id": client_id,
            "error": f"No client record for {client_id!r} in clients.csv.",
        }

    country = record["country"]

    return {
        "found": True,
        "client_id": record["client_id"],
        "country": country,
        "risk_rating": record["risk_rating"],
        "client_type": record["client_type"],
        "relationship_years": record["relationship_years"],
        "regional_policy": _REGIONAL_POLICIES.get(country),
        "policy_scope": (
            "global + regional" if country in _REGIONAL_POLICIES else "global only"
        ),
    }


def get_clients_by_country(country: str) -> list[dict]:
    """Retrieve every client whose relationship country matches ``country``."""
    country = (country or "").strip().casefold()
    return [
        get_client_profile(client_id)
        for client_id, record in clients().items()
        if record["country"].casefold() == country
    ]
