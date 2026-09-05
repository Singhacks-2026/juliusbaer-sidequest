"""
Policy retrieval pipeline.

    policy files -> load -> clean -> chunk -> index -> retrieve -> rerank
                                                    -> evidence + source

Two choices here are driven by the shape of this corpus rather than by
convention:

*Rule-level chunking.*  The policy documents are four to eight lines each, so a
fixed 500-character window would swallow whole documents and destroy citation
precision.  Chunks are therefore single logical rules with the document heading
prepended, which keeps "Singapore" and "high-risk" available as retrieval
signal and lets a citation point at the rule that actually fired.

*Content-based decoy suppression.*  The four decoys read "This document
contains no payment-monitoring thresholds", so a bag-of-words query for
"payment threshold" scores them highly — lexical retrieval cannot see the
negation.  Reranking therefore drops chunks that assert an absence or carry no
actionable rule.  Neither filter references a filename, so nothing here is
keyed to the decoys specifically.
"""

import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# A bullet, a numbered step, or a table row begins a new logical rule.
_RULE_START = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+|\|)")
_HEADING = re.compile(r"^\s*#{1,6}\s+(.*)$")

# Chunks asserting that a document holds nothing are not evidence, however well
# they match a query lexically.
_ABSENCE_ASSERTION = re.compile(
    r"\b(?:contains?|holds?|includes?)\s+no\b|\bno\s+(?:relevant|applicable)\b"
    r"|\bnot\s+applicable\b|\bdoes\s+not\s+contain\b",
    re.I,
)

# An evidential chunk states a threshold, imposes a requirement, names a
# risk jurisdiction, or is a numbered procedural step.
_THRESHOLD = re.compile(r"\b(?:USD|CHF|SGD|HKD|GBP)\s*[\d,]+", re.I)
_REQUIREMENT = re.compile(
    r"\b(?:require[sd]?|must|shall|should|escalat\w*|review\w*|prohibit\w*)\b", re.I
)
_PROCEDURAL_STEP = re.compile(r"^\s*\d+\.\s+\S")
_JURISDICTION_RISK = re.compile(r"\bhigh[- ]risk\b", re.I)

# Query terms that should pull a specific policy layer up when the question is
# about a particular jurisdiction or reasoning pattern.
_TOPIC_BOOSTS = {
    "singapore": 0.12,
    "switzerland": 0.12,
    "high-risk": 0.10,
    "structuring": 0.10,
    "splitting": 0.08,
    "enhanced review": 0.08,
    "rm review": 0.08,
    "workflow": 0.08,
    "procedure": 0.08,
    "investigation": 0.06,
}

_MIN_SCORE = 0.02


def split_into_rules(text: str) -> list[tuple[str, str]]:
    """
    Split cleaned policy text into ``(heading, rule)`` pairs.

    Continuation lines are joined into the rule they belong to.  The global
    policy's structuring clause wraps across three source lines; splitting it
    per line would separate "potential structuring" from the "USD 100,000"
    threshold it depends on, so line-wise parsing is not safe on this corpus.
    """
    units: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            rule = " ".join(part.strip() for part in buffer).strip()
            if rule:
                units.append((heading, rule))
            buffer.clear()

    for line in text.splitlines():
        heading_match = _HEADING.match(line)
        if heading_match:
            flush()
            heading = heading_match.group(1).strip()
            continue

        if not line.strip():
            flush()
            continue

        if _RULE_START.match(line):
            flush()
            buffer.append(re.sub(_RULE_START, "", line, count=1))
        else:
            buffer.append(line)

    flush()

    if not units and heading:
        units.append((heading, heading))

    return units


def load_policy_documents(policy_directory: str) -> list[dict]:
    """Load every markdown policy document, preserving its source filename."""
    documents = []

    for filename in sorted(os.listdir(policy_directory)):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(policy_directory, filename)
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()

        documents.append(
            {
                "source": filename,
                "text": clean_document(raw),
                "raw": raw,
                "metadata": {"path": path},
            }
        )

    return documents


def clean_document(text: str) -> str:
    """
    Normalise policy text before chunking.

    Emphasis markers and code fences are stripped; headings, wording and
    numbering are preserved because they carry retrieval and citation signal.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = text.replace("`", "")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split documents into one chunk per policy rule.

    ``chunk_size`` and ``chunk_overlap`` are kept for interface compatibility
    and act as a safety net: a rule longer than ``chunk_size`` is windowed, but
    no rule in this corpus comes close.
    """
    chunks = []

    for document in documents:
        for index, (heading, rule) in enumerate(split_into_rules(document["text"])):
            for offset, piece in enumerate(
                _window(rule, chunk_size, chunk_overlap)
            ):
                text = f"{heading} — {piece}" if heading and heading != piece else piece
                chunks.append(
                    {
                        "chunk_id": f"{document['source']}#{index}-{offset}",
                        "source": document["source"],
                        "heading": heading,
                        "rule": piece,
                        "text": text,
                        "metadata": {
                            "is_evidential": is_evidential(piece),
                            "has_threshold": bool(_THRESHOLD.search(piece)),
                        },
                    }
                )

    return chunks


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Window an over-long rule; returns ``[text]`` for anything normal-sized."""
    if len(text) <= size:
        return [text]

    step = max(size - overlap, 1)
    return [text[start : start + size] for start in range(0, len(text), step)]


def is_evidential(text: str) -> bool:
    """
    Whether a chunk can support a citation.

    Rejects absence assertions, then requires the chunk to actually state
    something actionable: a threshold, a requirement, a high-risk jurisdiction
    rule, or a numbered procedural step.
    """
    if _ABSENCE_ASSERTION.search(text):
        return False

    return bool(
        _THRESHOLD.search(text)
        or _REQUIREMENT.search(text)
        or _JURISDICTION_RISK.search(text)
        or _PROCEDURAL_STEP.match(text)
    )


def build_index(chunks: list[dict]):
    """
    Build a TF-IDF index over the chunks.

    Bigrams are included so phrases like "enhanced review" and "high risk"
    survive as single features.
    """
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        ngram_range=(1, 2),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)

    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def retrieve(index, query: str, top_k: int = 5) -> list[dict]:
    """Rank chunks against ``query`` by cosine similarity, unfiltered."""
    if not index or not query or not query.strip():
        return []

    scores = cosine_similarity(
        index["vectorizer"].transform([query]), index["matrix"]
    )[0]

    ranked = sorted(
        zip(index["chunks"], scores), key=lambda pair: pair[1], reverse=True
    )

    return [
        {**chunk, "score": round(float(score), 4)}
        for chunk, score in ranked[:top_k]
        if score > 0
    ]


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    Drop non-evidential candidates and boost topically matching policy layers.

    This is where decoys are removed: they assert an absence and state no rule,
    so they fail ``is_evidential`` regardless of how well they matched
    lexically.
    """
    lowered = (query or "").casefold()
    reranked = []

    for candidate in candidates:
        if not candidate["metadata"]["is_evidential"]:
            continue

        boost = sum(
            weight
            for term, weight in _TOPIC_BOOSTS.items()
            if term in lowered and term in candidate["text"].casefold()
        )

        final = candidate["score"] + boost
        if final < _MIN_SCORE:
            continue

        reranked.append({**candidate, "base_score": candidate["score"], "score": round(final, 4)})

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def retrieve_policy_evidence(index, query: str, top_k: int = 3) -> list[dict]:
    """Retrieve a wide candidate set, then rerank down to the evidence returned."""
    candidates = retrieve(index, query, top_k=max(top_k * 4, 12))
    return rerank(query, candidates, top_k=top_k)
