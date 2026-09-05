"""
RAG pipeline for policy document retrieval.

TF-IDF / cosine similarity with section-aware chunking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DECOY_PREFIX = "decoy_operational_"

# Soft boosts for real policy filenames when query hints match.
FILENAME_BOOSTS = {
    "high_risk": ["high_risk_jurisdictions.md"],
    "jurisdiction": ["high_risk_jurisdictions.md"],
    "ae": ["high_risk_jurisdictions.md"],
    "uae": ["high_risk_jurisdictions.md"],
    "threshold": [
        "global_payment_policy.md",
        "regional_singapore.md",
        "regional_switzerland.md",
    ],
    "enhanced": [
        "global_payment_policy.md",
        "regional_singapore.md",
        "regional_switzerland.md",
    ],
    "structuring": ["global_payment_policy.md", "regional_switzerland.md"],
    "splitting": ["global_payment_policy.md", "regional_switzerland.md"],
    "24": ["global_payment_policy.md"],
    "singapore": ["regional_singapore.md"],
    "switzerland": ["regional_switzerland.md"],
    "chf": ["regional_switzerland.md"],
    "workflow": ["investigation_procedure.md"],
    "investigation": ["investigation_procedure.md"],
    "procedure": ["investigation_procedure.md"],
    "assumption": ["investigation_procedure.md"],
    "release": [
        "global_payment_policy.md",
        "high_risk_jurisdictions.md",
        "investigation_procedure.md",
        "regional_singapore.md",
    ],
}


def load_policy_documents(policy_directory: str) -> list[dict]:
    """Load policy documents from the supplied directory."""
    directory = Path(policy_directory)
    documents = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(
            {
                "source": path.name,
                "text": text,
                "metadata": {
                    "is_decoy": path.name.startswith(DECOY_PREFIX),
                    "path": str(path),
                },
            }
        )
    return documents


def clean_document(text: str) -> str:
    """Normalize policy text before chunking; preserve headings/rules."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split policy documents into retrieval chunks.

    Prefer heading / bullet-level splits so individual rules stay intact.
    """
    chunks: list[dict] = []
    for doc in documents:
        source = doc["source"]
        cleaned = clean_document(doc.get("text", ""))
        pieces = _split_policy_text(cleaned, chunk_size=chunk_size, overlap=chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{source}#{i}",
                    "source": source,
                    "text": piece,
                    "metadata": {
                        **(doc.get("metadata") or {}),
                        "chunk_index": i,
                    },
                }
            )
    return chunks


def _split_policy_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []

    # Split on markdown headings first
    sections = re.split(r"(?=^#+\s)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    if len(sections) <= 1:
        # Bullet / paragraph units for short policy files
        bullets = re.split(r"(?=^-\s)", text, flags=re.MULTILINE)
        bullets = [b.strip() for b in bullets if b.strip()]
        sections = bullets if len(bullets) > 1 else [text]

    pieces: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            pieces.append(section)
            continue
        # Further split long sections by paragraphs / bullets
        units = re.split(r"\n\s*\n|(?=^-\s)", section, flags=re.MULTILINE)
        units = [u.strip() for u in units if u.strip()]
        buf = ""
        for unit in units:
            if not buf:
                buf = unit
            elif len(buf) + 1 + len(unit) <= chunk_size:
                buf = f"{buf}\n{unit}"
            else:
                pieces.append(buf)
                # overlap: keep tail of previous buffer
                if overlap > 0 and len(buf) > overlap:
                    buf = buf[-overlap:] + "\n" + unit
                else:
                    buf = unit
        if buf:
            pieces.append(buf)
    return pieces or [text]


def build_index(chunks: list[dict]) -> dict[str, Any]:
    """Build a reusable TF-IDF retrieval index."""
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts) if texts else None
    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }


def retrieve(
    index,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the most relevant policy chunks; preserve source names."""
    if not index or not index.get("chunks") or index.get("matrix") is None:
        return []

    chunks: list[dict] = index["chunks"]
    vectorizer: TfidfVectorizer = index["vectorizer"]
    matrix = index["matrix"]

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()

    scored: list[tuple[float, dict]] = []
    q_lower = query.lower()
    for i, chunk in enumerate(chunks):
        score = float(sims[i])
        source = chunk["source"]
        # Prefer real policies over decoys
        if source.startswith(DECOY_PREFIX):
            score *= 0.35
        else:
            score += 0.05
        # Filename keyword boosts
        for keyword, sources in FILENAME_BOOSTS.items():
            if keyword in q_lower and source in sources:
                score += 0.12
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    seen_sources: set[str] = set()
    for score, chunk in scored:
        if score <= 0 and len(results) >= top_k:
            break
        results.append(
            {
                "source": chunk["source"],
                "text": chunk["text"],
                "score": round(score, 4),
                "chunk_id": chunk["chunk_id"],
            }
        )
        seen_sources.add(chunk["source"])
        if len(results) >= top_k:
            break
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Light rerank: demote decoys, prefer diverse high-signal sources."""
    q_lower = query.lower()

    def key(item: dict) -> tuple:
        source = item.get("source", "")
        is_decoy = 1 if source.startswith(DECOY_PREFIX) else 0
        boost = 0
        for keyword, sources in FILENAME_BOOSTS.items():
            if keyword in q_lower and source in sources:
                boost += 1
        return (is_decoy, -boost, -float(item.get("score", 0)))

    ordered = sorted(candidates, key=key)
    # Diversify by source while keeping order
    diversified: list[dict] = []
    seen: set[str] = set()
    for item in ordered:
        src = item.get("source", "")
        if src in seen and len(diversified) >= top_k:
            continue
        if src not in seen or len(diversified) < top_k:
            diversified.append(item)
            seen.add(src)
        if len(diversified) >= top_k:
            break
    return diversified[:top_k]


def retrieve_policy_evidence(
    index,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Convenience method for the policy tool."""
    candidates = retrieve(index, query, top_k=max(10, top_k * 3))
    return rerank(query, candidates, top_k=top_k)
