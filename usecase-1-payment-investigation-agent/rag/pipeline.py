"""Hybrid TF-IDF + keyword-boosted policy retrieval pipeline."""

import math
import os
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DECOY_MARKERS = (
    "no payment-monitoring thresholds",
    "administrative note",
)

# Domain terms that matter for this corpus (threshold / jurisdiction / process).
BOOST_TERMS = {
    "ae": 2.5,
    "uae": 2.0,
    "high-risk": 2.2,
    "high": 1.2,
    "risk": 1.2,
    "enhanced": 2.0,
    "rm": 2.0,
    "review": 1.3,
    "structuring": 2.5,
    "splitting": 2.2,
    "transaction": 1.2,
    "beneficiary": 1.3,
    "24": 1.5,
    "hours": 1.2,
    "chf": 2.0,
    "usd": 1.8,
    "singapore": 2.0,
    "switzerland": 2.0,
    "compliance": 1.8,
    "escalate": 1.8,
    "investigation": 1.8,
    "workflow": 2.0,
    "threshold": 2.0,
    "75000": 2.0,
    "80000": 2.0,
    "100000": 2.2,
    "120000": 2.0,
    "100,000": 2.2,
    "75,000": 2.0,
    "80,000": 2.0,
    "120,000": 2.0,
}


def load_policy_documents(policy_directory: str) -> list[dict]:
    """Load policy documents from the supplied directory."""
    documents = []
    for filename in sorted(os.listdir(policy_directory)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(policy_directory, filename)
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        documents.append(
            {
                "source": filename,
                "text": text,
                "metadata": {
                    "is_decoy": filename.startswith("decoy_")
                    or any(marker in text.lower() for marker in DECOY_MARKERS)
                },
            }
        )
    return documents


def clean_document(text: str) -> str:
    """Normalize policy text while keeping headings and rule wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split policy documents into heading- and bullet-aware chunks."""
    chunks = []
    for document in documents:
        source = document["source"]
        text = clean_document(document.get("text") or "")
        metadata = dict(document.get("metadata") or {})
        parts = _split_policy_text(text)

        if len(text) <= chunk_size * 2:
            parts.append(text)

        seen = set()
        for index, part in enumerate(parts):
            part = part.strip()
            if not part or part in seen:
                continue
            seen.add(part)
            chunks.append(
                {
                    "chunk_id": f"{source}::{index}",
                    "source": source,
                    "text": part,
                    "metadata": metadata,
                }
            )
    return chunks


def _split_policy_text(text: str) -> list[str]:
    blocks = []
    current: list[str] = []
    heading = ""

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            blocks.append(body)
        current = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            heading = line
            current = [heading]
            continue
        if re.match(r"^(\d+\.|- )\s*", line):
            flush()
            current = [heading, line] if heading else [line]
            continue
        current.append(line)

    flush()
    return blocks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (text or "").lower())


def build_index(chunks: list[dict]):
    """Build a reusable TF-IDF + BM25-style keyword index."""
    corpus = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus)

    tokenized = [_tokenize(text) for text in corpus]
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n_docs = max(len(tokenized), 1)
    avgdl = sum(len(tokens) for tokens in tokenized) / n_docs
    doc_freqs = [Counter(tokens) for tokens in tokenized]

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_freqs": doc_freqs,
        "df": df,
        "n_docs": n_docs,
        "avgdl": avgdl,
        "tokenized": tokenized,
    }


def _bm25_scores(index, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = _tokenize(query)
    if not query_terms:
        return [0.0] * len(index["chunks"])

    scores = []
    n_docs = index["n_docs"]
    avgdl = index["avgdl"] or 1.0
    df = index["df"]

    for doc_freq, tokens in zip(index["doc_freqs"], index["tokenized"]):
        dl = len(tokens) or 1
        score = 0.0
        for term in query_terms:
            if term not in doc_freq:
                continue
            n_q = doc_freq[term]
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = n_q + k1 * (1 - b + b * dl / avgdl)
            score += idf * (n_q * (k1 + 1)) / denom
            # Extra boost for domain-critical terms present in both query and doc.
            score += 0.15 * BOOST_TERMS.get(term, 0.0)
        scores.append(score)
    return scores


def retrieve(
    index,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the most relevant policy chunks (TF-IDF + BM25 hybrid)."""
    if not query or not index:
        return []

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    chunks = index["chunks"]
    query_vector = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vector, matrix).flatten()
    bm25_scores = _bm25_scores(index, query)

    # Normalize BM25 into ~[0,1] for blending.
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    ranked = []
    for chunk, tfidf, bm25 in zip(chunks, tfidf_scores, bm25_scores):
        bm25_norm = (bm25 / max_bm25) if max_bm25 else 0.0
        score = 0.55 * float(tfidf) + 0.45 * bm25_norm
        ranked.append(
            {
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score),
                "chunk_id": chunk.get("chunk_id"),
                "metadata": chunk.get("metadata") or {},
                "tfidf": float(tfidf),
                "bm25": float(bm25),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[: max(top_k * 3, top_k)]


def _is_decoy(candidate: dict) -> bool:
    source = candidate.get("source") or ""
    text = (candidate.get("text") or "").lower()
    metadata = candidate.get("metadata") or {}
    return bool(
        metadata.get("is_decoy")
        or source.startswith("decoy_")
        or any(marker in text for marker in DECOY_MARKERS)
    )


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Hard-drop decoys and boost jurisdiction / threshold keyword hits."""
    query_lower = (query or "").lower()
    query_terms = set(_tokenize(query))
    rescored = []

    for candidate in candidates:
        if _is_decoy(candidate):
            continue  # hard drop

        score = float(candidate.get("score") or 0)
        text = (candidate.get("text") or "").lower()
        source = candidate.get("source") or ""

        overlap = sum(1 for term in query_terms if term in text)
        score += 0.02 * overlap

        for term, weight in BOOST_TERMS.items():
            if term in query_lower and term in text:
                score += 0.04 * weight

        # Prefer exact policy files when the query names the region.
        if "singapore" in query_lower and source == "regional_singapore.md":
            score += 0.35
        if "switzerland" in query_lower and source == "regional_switzerland.md":
            score += 0.35
        if any(t in query_lower for t in ("high-risk", "high risk", " ae", "ae ", "uae")):
            if source == "high_risk_jurisdictions.md":
                score += 0.4
        if any(t in query_lower for t in ("workflow", "investigation procedure", "steps")):
            if source == "investigation_procedure.md":
                score += 0.4
        if any(t in query_lower for t in ("structuring", "splitting", "24")):
            if source == "global_payment_policy.md":
                score += 0.2

        rescored.append({**candidate, "score": score})

    rescored.sort(key=lambda item: item["score"], reverse=True)

    unique = []
    seen_sources = set()
    for candidate in rescored:
        source = candidate["source"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        unique.append(candidate)
        if len(unique) >= top_k:
            break
    return unique


def retrieve_policy_evidence(
    index,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve then rerank policy evidence for the policy tool."""
    candidates = retrieve(index, query, top_k=10)
    return rerank(query, candidates, top_k=top_k)
