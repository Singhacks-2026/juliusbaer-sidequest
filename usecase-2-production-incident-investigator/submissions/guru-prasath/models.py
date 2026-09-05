"""Typed containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Record:
    """One retrievable unit: a markdown chunk or a single CSV row."""

    source: str  # fine-grained id, e.g. "known_issues.csv#KI-101"
    display_source: str  # corpus filename, e.g. "known_issues.csv"
    source_type: str  # e.g. "known_issues"
    text: str
    tokens: list[str]
    idf: dict[str, float] = field(default_factory=dict)


@dataclass
class Evidence:
    """Correlated findings before confidence calibration."""

    theme: str = "undetermined"
    root_cause: str = ""
    remediation: str = ""
    impacted_systems: list[str] = field(default_factory=list)
    mttr_minutes: int | None = None
    supporting_evidence: list[dict[str, str]] = field(default_factory=list)
    positive_source_types: set[str] = field(default_factory=set)
    uncertainty_signals: int = 0
