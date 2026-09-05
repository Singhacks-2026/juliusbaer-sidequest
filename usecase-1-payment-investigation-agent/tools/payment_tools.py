"""Payment lookups and deterministic payment-control calculations."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from tools.client_tools import get_client_profile

_ROOT = Path(__file__).resolve().parents[1]
_PAYMENTS_PATH = _ROOT / "data" / "payments.csv"
_POLICY_PATH = _ROOT / "data" / "policies"


def _coerce(value: str) -> Any:
    if value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


@lru_cache(maxsize=1)
def _payments() -> tuple[dict, ...]:
    with _PAYMENTS_PATH.open(newline="", encoding="utf-8") as handle:
        return tuple(
            {key: _coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        )


def get_payment(payment_id: str) -> dict:
    """Return one payment, or an empty dict for an unknown payment ID."""
    wanted = str(payment_id).strip().upper()
    return next((dict(row) for row in _payments() if row["payment_id"] == wanted), {})


def get_client_payments(client_id: str) -> list[dict]:
    """Return a client's payment history ordered by date and payment ID."""
    wanted = str(client_id).strip().upper()
    matches = [dict(row) for row in _payments() if row["client_id"] == wanted]
    return sorted(matches, key=lambda row: (row["payment_date"], row["payment_id"]))


@lru_cache(maxsize=1)
def _structuring_threshold() -> dict:
    text = (_POLICY_PATH / "global_payment_policy.md").read_text(encoding="utf-8")
    match = re.search(
        r"combined value exceeds\s+(USD|CHF)\s+([\d,]+)\s+equivalent",
        text,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError("Global structuring threshold was not found in policy")
    return {
        "amount": int(match.group(2).replace(",", "")),
        "currency": match.group(1).upper(),
        "source": "global_payment_policy.md",
    }


def aggregate_beneficiary_24h(client_id: str, beneficiary_name: str) -> dict:
    """Aggregate same-client/same-beneficiary payments by calendar date.

    Currency totals are kept separate to avoid adding unlike currencies. The
    primary result is the largest comparable group and all windows are retained.
    """
    wanted_name = str(beneficiary_name).strip().casefold()
    matching = [
        row
        for row in get_client_payments(client_id)
        if str(row.get("beneficiary_name", "")).casefold() == wanted_name
    ]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in matching:
        groups[(str(row["payment_date"]), str(row["currency"]))].append(row)

    windows = []
    for (payment_date, currency), rows in groups.items():
        windows.append(
            {
                "payment_date": payment_date,
                "currency": currency,
                "count": len(rows),
                "total_amount": round(sum(float(row["amount"]) for row in rows), 2),
                "payment_ids": [row["payment_id"] for row in rows],
                "payments": rows,
                "date_assumption": "Same calendar date is treated as within 24 hours; timestamps are unavailable.",
            }
        )
    windows.sort(key=lambda item: (item["count"], item["total_amount"]), reverse=True)
    if not windows:
        return {
            "client_id": str(client_id).strip().upper(),
            "beneficiary_name": beneficiary_name,
            "count": 0,
            "total_amount": 0,
            "payments": [],
            "windows": [],
        }
    primary = dict(windows[0])
    threshold = _structuring_threshold()
    primary.update(
        {
            "client_id": str(client_id).strip().upper(),
            "beneficiary_name": matching[0]["beneficiary_name"],
            "windows": windows,
            "structuring_threshold": threshold,
            "exceeds_structuring_threshold": primary["total_amount"] > threshold["amount"],
            "threshold_currency_basis": (
                "native-currency comparison"
                if primary["currency"] == threshold["currency"]
                else "1:1 equivalent assumption (no exchange rates supplied)"
            ),
        }
    )
    return primary


def find_repeated_beneficiaries(client_id: str) -> list[dict]:
    """Return repeated beneficiary/date/currency groups for a client."""
    names = {row["beneficiary_name"] for row in get_client_payments(client_id)}
    results = [aggregate_beneficiary_24h(client_id, name) for name in names]
    return sorted(
        [result for result in results if result["count"] > 1],
        key=lambda item: (item["count"], item["total_amount"]),
        reverse=True,
    )


def _policy_thresholds(source: str) -> list[dict]:
    """Parse review thresholds directly from a supplied policy document."""
    text = (_POLICY_PATH / source).read_text(encoding="utf-8")
    pattern = re.compile(
        r"above\s+(USD|CHF)\s+([\d,]+)\s+equivalent\s+require\s+([^.\n]+)",
        re.IGNORECASE,
    )
    return [
        {
            "currency": currency.upper(),
            "amount": int(amount.replace(",", "")),
            "requirement": requirement.strip(),
            "source": source,
        }
        for currency, amount, requirement in pattern.findall(text)
    ]


def _high_risk_codes() -> set[str]:
    text = (_POLICY_PATH / "high_risk_jurisdictions.md").read_text(encoding="utf-8")
    return {
        code.upper()
        for code in re.findall(r"\b([A-Z]{2})\s+\([^)]+\)\s+is a high-risk", text)
    }


def evaluate_payment_controls(payment_id: str) -> dict:
    """Deterministically evaluate amount, regional, and destination controls."""
    payment = get_payment(payment_id)
    if not payment:
        return {"error": f"Unknown payment_id: {payment_id}"}
    client = get_client_profile(str(payment["client_id"]))
    client_country = str(client.get("country", ""))
    regional_source = {
        "Singapore": "regional_singapore.md",
        "Switzerland": "regional_switzerland.md",
    }.get(client_country)
    sources = ["global_payment_policy.md"] + ([regional_source] if regional_source else [])

    amount = float(payment["amount"])
    currency = str(payment["currency"])
    evaluations = []
    for source in sources:
        for rule in _policy_thresholds(source):
            exact_currency = currency == rule["currency"]
            evaluations.append(
                {
                    **rule,
                    "payment_amount": amount,
                    "payment_currency": currency,
                    "triggered": amount > rule["amount"],
                    "currency_basis": (
                        "native-currency comparison"
                        if exact_currency
                        else "1:1 equivalent assumption (no exchange rates supplied)"
                    ),
                }
            )

    destination_code = str(payment.get("beneficiary_country_code", "")).upper()
    high_risk = destination_code in _high_risk_codes()
    triggered_requirements = sorted(
        {item["requirement"] for item in evaluations if item["triggered"]}
        | ({"additional review"} if high_risk else set())
    )
    return {
        "payment_id": payment["payment_id"],
        "client_id": payment["client_id"],
        "client_country": client_country,
        "amount": payment["amount"],
        "currency": currency,
        "beneficiary_country_code": destination_code,
        "beneficiary_country_code_is_authoritative": True,
        "high_risk_destination": high_risk,
        "applicable_policy_sources": sources
        + (["high_risk_jurisdictions.md"] if high_risk else []),
        "threshold_evaluations": evaluations,
        "triggered_requirements": triggered_requirements,
    }
