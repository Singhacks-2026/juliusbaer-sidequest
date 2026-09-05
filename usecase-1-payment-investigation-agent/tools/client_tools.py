"""Client data-access tools."""

from tools._data import load_clients, row_to_dict


def get_client_profile(client_id: str) -> dict:
    """Retrieve one client's profile from ``data/clients.csv``."""
    if not client_id:
        return {"error": "client_id is required"}

    clients = load_clients()
    matches = clients[clients["client_id"] == client_id]
    if matches.empty:
        return {"error": f"Unknown client_id: {client_id}"}

    profile = row_to_dict(matches.iloc[0])
    country = profile.get("country")
    if country == "Singapore":
        profile["regional_policy"] = "regional_singapore.md"
    elif country == "Switzerland":
        profile["regional_policy"] = "regional_switzerland.md"
    else:
        profile["regional_policy"] = None
    profile["global_policy"] = "global_payment_policy.md"
    return profile


def get_clients_by_country(country: str) -> list[dict]:
    """Retrieve clients associated with a given country."""
    if not country:
        return []

    clients = load_clients()
    matches = clients[clients["country"].str.casefold() == country.casefold()]
    return [row_to_dict(row) for _, row in matches.iterrows()]
