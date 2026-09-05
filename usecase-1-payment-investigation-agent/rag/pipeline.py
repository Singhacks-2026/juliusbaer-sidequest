"""Small, deterministic, source-preserving TF-IDF policy retriever."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")
_SYNONYMS = {
    "release": {"review", "procedure", "payment"},
    "workflow": {"procedure", "evidence", "facts", "policy"},
    "process": {"procedure", "workflow"},
    "split": {"splitting", "structuring", "multiple", "beneficiary"},
    "splitting": {"structuring", "multiple", "beneficiary"},
    "structuring": {"splitting", "multiple", "beneficiary", "compliance"},
    "destination": {"jurisdiction", "country", "risk"},
    "risk": {"high", "jurisdiction", "review"},
    "threshold": {"amount", "above", "review"},
    "region": {"singapore", "switzerland", "global"},
    "regional": {"singapore", "switzerland", "global"},
    "assumption": {"facts", "evidence", "trigger"},
}


def _tokens(text: str, expand: bool = False) -> list[str]:
    tokens = _TOKEN.findall(text.casefold())
    normalized = [token[:-1] if token.endswith("s") and len(token) > 4 else token for token in tokens]
    if expand:
        for token in tuple(normalized):
            normalized.extend(_SYNONYMS.get(token, ()))
    return normalized


def load_policy_documents(policy_directory: str) -> list[dict]:
    directory = Path(policy_directory)
    return [
        {"source": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(directory.glob("*.md"))
        if path.is_file()
    ]


def clean_document(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def chunk_documents(documents: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    """Create rule-aware chunks while retaining the document heading."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")
    chunks = []
    for document in documents:
        text = clean_document(str(document.get("text", "")))
        if not text:
            continue
        lines = text.splitlines()
        heading = next((line for line in lines if line.startswith("#")), "")
        units, current = [], ""
        for line in lines:
            candidate = f"{current}\n{line}".strip()
            if current and len(candidate) > chunk_size:
                units.append(current)
                overlap = current[-chunk_overlap:] if chunk_overlap else ""
                current = f"{overlap}\n{line}".strip()
            else:
                current = candidate
        if current:
            units.append(current)
        for number, unit in enumerate(units):
            if heading and heading not in unit:
                unit = f"{heading}\n{unit}"
            chunks.append(
                {
                    "chunk_id": f"{document['source']}#{number}",
                    "source": document["source"],
                    "text": unit,
                    "metadata": {"chunk_number": number},
                }
            )
    return chunks


def build_index(chunks: list[dict]) -> dict:
    """Build a reusable TF-IDF index without an external ML dependency."""
    token_counts = [Counter(_tokens(chunk["text"])) for chunk in chunks]
    document_frequency = Counter()
    for counts in token_counts:
        document_frequency.update(counts.keys())
    count = max(len(chunks), 1)
    idf = {
        token: math.log((1 + count) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }
    vectors, norms = [], []
    for counts in token_counts:
        vector = {token: (1 + math.log(freq)) * idf[token] for token, freq in counts.items()}
        vectors.append(vector)
        norms.append(math.sqrt(sum(value * value for value in vector.values())))
    return {"chunks": chunks, "idf": idf, "vectors": vectors, "norms": norms}


def retrieve(index: dict, query: str, top_k: int = 5) -> list[dict]:
    if top_k <= 0:
        return []
    query_counts = Counter(_tokens(query, expand=True))
    query_vector = {
        token: (1 + math.log(freq)) * index["idf"].get(token, 1.0)
        for token, freq in query_counts.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0
    scored = []
    for chunk, vector, norm in zip(index["chunks"], index["vectors"], index["norms"]):
        if re.search(r"\bcontains no\b", chunk["text"], re.IGNORECASE):
            continue
        dot = sum(query_vector[token] * vector.get(token, 0.0) for token in query_vector)
        score = dot / (query_norm * norm) if norm else 0.0
        if score > 0:
            scored.append({**chunk, "score": round(score, 6)})
    scored.sort(key=lambda item: (-item["score"], item["source"], item["chunk_id"]))
    return scored[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    query_tokens = set(_tokens(query))
    reranked = []
    for candidate in candidates:
        exact_bonus = 0.025 * len(query_tokens & set(_tokens(candidate["text"])))
        reranked.append({**candidate, "score": round(candidate["score"] + exact_bonus, 6)})
    reranked.sort(key=lambda item: (-item["score"], item["source"]))
    return reranked[:top_k]


def retrieve_policy_evidence(index: dict, query: str, top_k: int = 3) -> list[dict]:
    return rerank(query, retrieve(index, query, top_k=max(top_k * 3, 10)), top_k=top_k)
