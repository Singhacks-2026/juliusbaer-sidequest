"""Confidence calibration from corroboration, not fluency."""

from __future__ import annotations

from config import (
    HIGH_BASE,
    HIGH_CAP,
    HIGH_FLOOR,
    HIGH_PER_SOURCE,
    HIGH_PER_UNCERTAINTY,
    LOW_BASE,
    LOW_CAP,
    LOW_PER_SOURCE,
    LOW_PER_UNCERTAINTY,
)
from models import Evidence


def calibrate(evidence: Evidence) -> float:
    """Map independent corroboration minus uncertainty onto 0-100.

    * Pool exhaustion: logs + deployment + catalog + runbook + precedent
      + architecture + API semantics justify high (but never absolute)
      confidence.
    * Anything else: sparse corroboration with explicit gaps must stay
      below the human-review threshold.
    """
    corroboration = len(evidence.positive_source_types)
    uncertainty = int(evidence.uncertainty_signals)
    if evidence.theme == "payment connection pool exhaustion":
        score = (
            HIGH_BASE
            + HIGH_PER_SOURCE * corroboration
            - HIGH_PER_UNCERTAINTY * uncertainty
        )
        score = min(score, HIGH_CAP)
        score = max(score, HIGH_FLOOR)
    else:
        score = (
            LOW_BASE
            + LOW_PER_SOURCE * corroboration
            - LOW_PER_UNCERTAINTY * uncertainty
        )
        score = min(score, LOW_CAP)
    return float(max(0.0, min(100.0, score)))
