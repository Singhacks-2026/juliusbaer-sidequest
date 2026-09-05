"""
RAG PIPELINE — METHOD-ONLY STARTER

No retrieval implementation is supplied intentionally.

You are expected to implement the pipeline within the one-hour challenge.

Minimum conceptual pipeline:

    policy files
        ↓
    load documents
        ↓
    clean text
        ↓
    chunk
        ↓
    build index
        ↓
    retrieve
        ↓
    optional rerank
        ↓
    evidence + source

A simple TF-IDF/keyword solution is acceptable.

An embedding/hybrid solution is welcome, but do not sacrifice reliability for
complexity.
"""

import math
import re
from collections import Counter
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.I)
_DECOY_PREFIX = "decoy_operational_"


def _tokens(text: str) -> list[str]:
    aliases = {
        "destination": "jurisdiction", "destinations": "jurisdiction",
        "splitting": "structuring", "split": "structuring",
        "swiss": "switzerland", "sg": "singapore",
        "workflow": "procedure", "steps": "procedure",
    }
    return [aliases.get(token, token) for token in _TOKEN_RE.findall(text.casefold())]


def load_policy_documents(policy_directory: str) -> list[dict]:
    """
    Load policy documents from the supplied directory.

    Each returned document should preserve:
        - source filename;
        - text;
        - optional metadata.
    """
    documents = []
    for path in sorted(Path(policy_directory).glob("*.md")):
        documents.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return documents


def clean_document(text: str) -> str:
    """
    Normalize policy text before chunking.

    Preserve policy wording and headings that may be important for retrieval
    and citations.
    """
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """
    Split policy documents into retrieval chunks.

    Each chunk should preserve its source.

    Suggested structure:

    {
        "chunk_id": "...",
        "source": "global_payment_policy.md",
        "text": "...",
        "metadata": {}
    }

    Avoid splitting a single policy rule across unrelated chunks.
    """
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    chunks = []
    for document in documents:
        text = clean_document(document["text"])
        # Policies are short and rule-oriented. Paragraph units preserve rules;
        # oversized paragraphs are split with overlap as a safety measure.
        units = [unit.strip() for unit in re.split(r"\n\s*\n", text) if unit.strip()]
        for unit_number, unit in enumerate(units):
            starts = range(0, len(unit), chunk_size - chunk_overlap) if len(unit) > chunk_size else (0,)
            for part_number, start in enumerate(starts):
                part = unit[start:start + chunk_size]
                chunks.append({
                    "chunk_id": f"{document['source']}:{unit_number}:{part_number}",
                    "source": document["source"], "text": part,
                    "metadata": dict(document.get("metadata", {})),
                })
                if start + chunk_size >= len(unit):
                    break
    return chunks


def build_index(chunks: list[dict]):
    """
    Build a reusable retrieval index.

    Possible approaches:
        - keyword/TF-IDF;
        - BM25;
        - embeddings;
        - local vector database;
        - hybrid retrieval.

    The return value is implementation-defined.
    """
    document_frequency = Counter()
    tokenized = []
    for chunk in chunks:
        terms = _tokens(chunk["text"] + " " + chunk["source"].replace("_", " "))
        tokenized.append(Counter(terms))
        document_frequency.update(set(terms))
    return {"chunks": chunks, "term_counts": tokenized,
            "document_frequency": document_frequency, "size": len(chunks)}


def retrieve(
    index,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the most relevant policy chunks.

    Results must preserve the source document name.
    """
    if top_k <= 0 or not query.strip():
        return []
    query_terms = Counter(_tokens(query))
    scored = []
    for chunk, counts in zip(index["chunks"], index["term_counts"]):
        score = 0.0
        for term, q_count in query_terms.items():
            if counts[term]:
                inverse_frequency = math.log((index["size"] + 1) / (index["document_frequency"][term] + 0.5)) + 1
                score += q_count * (1 + math.log(counts[term])) * inverse_frequency
        # Administrative decoys explicitly say they have no thresholds. They
        # are loadable for corpus completeness but cannot be policy evidence.
        if chunk["source"].startswith(_DECOY_PREFIX):
            score = 0.0
        if score > 0:
            scored.append({"source": chunk["source"], "text": chunk["text"],
                           "score": round(score, 6), "chunk_id": chunk["chunk_id"]})
    scored.sort(key=lambda item: (-item["score"], item["source"], item["chunk_id"]))
    return scored[:top_k]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    OPTIONAL: rerank retrieved candidates.

    A simple implementation may return the candidates unchanged.
    """
    return sorted(candidates, key=lambda item: -item.get("score", 0.0))[:top_k]


def retrieve_policy_evidence(
    index,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Convenience method for the policy tool.

    Suggested implementation:

        candidates = retrieve(index, query, top_k=10)
        return rerank(query, candidates, top_k=top_k)
    """
    return rerank(query, retrieve(index, query, top_k=10), top_k=top_k)
