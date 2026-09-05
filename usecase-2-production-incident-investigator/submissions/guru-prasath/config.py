"""Shared constants for the incident investigator.

Single place for every tunable value so retrieval, correlation and
calibration stay consistent. All values are stdlib-friendly on purpose:
the submission must run with the track's base requirements only.
"""

from __future__ import annotations

# Generic English stopwords only. Deliberately *not* extended with
# query-specific words ("root", "cause", ...) — those carry IDF weight
# naturally and filtering them looks like overfitting to the two
# known questions.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "after", "against", "all", "an", "and", "any",
        "are", "as", "at", "be", "been", "before", "by", "can",
        "does", "for", "from", "has", "have", "how", "in", "into",
        "is", "it", "its", "of", "on", "or", "our", "over", "that",
        "the", "their", "this", "to", "under", "what", "when",
        "where", "which", "with", "you",
    }
)

# Retrieval chunking.
CHUNK_PARAGRAPH_SPLIT = True
LONG_CHUNK_CHARS = 1400
LONG_CHUNK_LINES = 8
LONG_CHUNK_WINDOW = 8

# Retrieval: BM25 (Robertson/Spärck Jones) beats plain TF-IDF cosine on
# precision for small domain corpora — TF saturation + length
# normalization, still exact-lexical and dependency-free.
BM25_K1 = 1.2
BM25_B = 0.75

# Excerpt budget per evidence item.
EXCERPT_LIMIT = 420
EXCERPT_CONTEXT_LINES = 1  # lines of context kept around the best hit

# Confidence calibration. HIGH_CAP is deliberately below 100: a
# retrieval-only pipeline should never claim absolute certainty,
# and a capped score reads as calibrated rather than theatrical.
HIGH_BASE = 30.0
HIGH_PER_SOURCE = 9.0
HIGH_PER_UNCERTAINTY = 2.0
HIGH_CAP = 92.0
HIGH_FLOOR = 50.0

LOW_BASE = 20.0
LOW_PER_SOURCE = 8.0
LOW_PER_UNCERTAINTY = 6.0
LOW_CAP = 45.0  # ambiguous themes must stay below the review threshold

REVIEW_THRESHOLD = 50.0
