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

import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_policy_documents(policy_directory: str) -> list[dict]:
    """
    Load policy documents from the supplied directory.

    Each returned document should preserve:
        - source filename;
        - text;
        - optional metadata.
    """
    documents = []
    for filename in sorted(os.listdir(policy_directory)):
        if not filename.endswith(".md"):
            continue
        path = os.path.join(policy_directory, filename)
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        documents.append({"source": filename, "text": text, "metadata": {}})
    return documents


def clean_document(text: str) -> str:
    """
    Normalize policy text before chunking.

    Preserve policy wording and headings that may be important for retrieval
    and citations.
    """
    # Collapse repeated blank lines / trailing whitespace without touching
    # headings or wording -- both matter for citation fidelity.
    text = text.replace("\r\n", "\n")
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

    Each chunk should preserve its source.

    Suggested structure:

    {
        "chunk_id": "...",
        "source": "global_payment_policy.md",
        "text": "...",
        "metadata": {}
    }

    Avoid splitting a single policy rule across unrelated chunks.

    Implementation note
    --------------------
    These policy documents are tiny (79-439 bytes each). One chunk per
    document avoids splitting a single rule mid-sentence across chunks,
    while ``chunk_size``/``chunk_overlap`` are honored for any document
    that does exceed the size limit.
    """
    chunks = []
    for document in documents:
        text = clean_document(document["text"])
        source = document["source"]

        if len(text) <= chunk_size:
            chunks.append(
                {
                    "chunk_id": f"{source}::0",
                    "source": source,
                    "text": text,
                    "metadata": document.get("metadata", {}),
                }
            )
            continue

        start = 0
        index = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": f"{source}::{index}",
                    "source": source,
                    "text": chunk_text,
                    "metadata": document.get("metadata", {}),
                }
            )
            index += 1
            start = end - chunk_overlap

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
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts) if texts else None
    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}


def retrieve(
    index,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the most relevant policy chunks.

    Results must preserve the source document name.
    """
    chunks = index["chunks"]
    if not chunks or index["matrix"] is None:
        return []

    vectorizer = index["vectorizer"]
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, index["matrix"])[0]

    ranked = sorted(
        zip(chunks, scores), key=lambda pair: pair[1], reverse=True
    )

    return [
        {"source": chunk["source"], "text": chunk["text"], "score": float(score)}
        for chunk, score in ranked[:top_k]
        if score > 0
    ]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    OPTIONAL: rerank retrieved candidates.

    A simple implementation may return the candidates unchanged.
    """
    return candidates[:top_k]


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
    candidates = retrieve(index, query, top_k=10)
    return rerank(query, candidates, top_k=top_k)
