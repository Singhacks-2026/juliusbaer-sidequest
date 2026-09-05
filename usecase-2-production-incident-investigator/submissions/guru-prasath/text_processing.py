"""Plain-text helpers shared by retrieval and excerpt extraction."""

from __future__ import annotations

import re

from config import STOPWORDS

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Split text into searchable terms.

    Keeps hyphenated component names (``payment-gateway-adapter``) whole
    *and* split, so queries matching either form still hit. Adds a light
    singular form (``payments`` -> ``payment``) to bridge singular/plural
    wording without an NLP dependency.
    """
    raw = _TOKEN_RE.findall(text.lower())
    terms: list[str] = []
    for token in raw:
        terms.append(token)
        if "-" in token or "_" in token:
            terms.extend(re.split(r"[-_]", token))
    normalized = list(terms)
    for term in terms:
        if term.endswith("s") and len(term) > 3:
            normalized.append(term[:-1])
    return [t for t in normalized if t not in STOPWORDS and len(t) > 1]


def source_type(source: str) -> str:
    """Map ``logs.md`` / ``known_issues.csv#KI-101`` -> ``logs`` / ``known_issues``."""
    base = source.split("#", 1)[0].lower()
    return base.rsplit("/", 1)[-1].rsplit(".", 1)[0]


# Lightweight query-side synonym expansion for ops vocabulary. Applied
# to the query only (standard practice), documented here rather than
# buried in scoring code. Kept to generic English/ops equivalences —
# nothing incident-specific — so unseen corpora behave the same way.
_QUERY_SYNONYMS: dict[str, list[str]] = {
    "email": ["notification"],
    "emails": ["notification"],
    "mail": ["notification"],
    "charge": ["payment"],
    "charges": ["payment"],
    "failing": ["failure", "error", "timeout", "exception"],
    "fail": ["failure", "error", "timeout"],
    "late": ["delay", "queued", "latency"],
    "slow": ["delay", "latency"],
}


def expand_query(terms: list[str]) -> list[str]:
    """Append documented synonym tokens to a tokenized query."""
    extra: list[str] = []
    for term in terms:
        extra.extend(_QUERY_SYNONYMS.get(term, ()))
    return terms + [
        t for t in extra if t not in STOPWORDS and len(t) > 1
    ]


def split_markdown(text: str, long_chars: int = 1400,
                   long_lines: int = 8, window: int = 8) -> list[str]:
    """Split a document into retrieval chunks.

    Paragraphs are the default unit. Very long blocks (e.g. log dumps)
    are further cut into small sliding windows so one huge chunk does
    not drown out the rest of the corpus.
    """
    chunks = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    for chunk in chunks:
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if len(chunk) > long_chars and len(lines) > long_lines:
            for start in range(0, len(lines), window):
                out.append("\n".join(lines[start:start + window]))
        else:
            out.append(chunk)
    return out or [text.strip()]
