"""Production incident investigator — public entry point.

Thin facade over the pipeline (``retrieval`` -> ``correlation`` ->
``calibration``). The required ``investigate(query, corpus)`` signature
is unchanged; all logic lives in the sibling modules so each stage can
be read, tested, and reviewed in isolation.

Pipeline:
    1. ``retrieval.retrieve`` ranks corpus files with BM25 (TF saturation
       + length normalization; CSV catalogs split per row) over a lightly
       expanded query.
    2. ``correlation.correlate`` checks which hypothesis independent
       source types jointly support, collects verbatim excerpts, and
       re-cuts any quote missing a key asserted fact.
    3. ``calibration.calibrate`` converts corroboration minus explicit
       uncertainty into a 0-100 score; ``needs_human_review`` derives
       from it so the two fields cannot drift apart.
"""

from __future__ import annotations

from calibration import calibrate
from config import REVIEW_THRESHOLD
from correlation import correlate
from retrieval import retrieve


def investigate(query: str, corpus: dict) -> dict:
    """Return a grounded incident report with exactly the required schema."""
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError("corpus must be a non-empty {filename: text} dict")
    ranked = retrieve(query, dict(corpus))
    evidence = correlate(query, dict(corpus), ranked)
    confidence = calibrate(evidence)
    return {
        "root_cause": evidence.root_cause,
        "supporting_evidence": evidence.supporting_evidence,
        "impacted_systems": evidence.impacted_systems,
        "mttr_minutes": evidence.mttr_minutes,
        "remediation": evidence.remediation,
        "confidence_score": confidence,
        "needs_human_review": confidence < REVIEW_THRESHOLD,
    }
