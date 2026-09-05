"""
RAG PIPELINE — policy document retrieval.

    policy files -> load -> clean -> chunk -> index -> retrieve -> rerank
                 -> evidence + source

Implementation notes
--------------------
Built on LangChain's in-memory vector store (``InMemoryVectorStore``) so the
whole pipeline stays local and deterministic: no embedding API call, no
network, identical results on every run.

The embedding function is a TF-IDF vector fitted on the policy corpus itself
(``TfidfEmbeddings``).  The corpus is nine short markdown files, so a lexical
representation is both sufficient and more predictable than a general-purpose
embedding model.

Two corpus-specific behaviours are deliberate:

* ``clean_document`` drops non-substantive documents.  The corpus contains
  administrative notes that state they hold no thresholds; a bag-of-words
  scorer cannot read that negation and would happily rank them highly.  They
  are filtered by content (a policy document states at least one number —
  a threshold, a limit, or a numbered procedure step), not by filename.
* Each chunk is indexed with its source filename and heading prepended, so
  queries such as "Singapore procedure" or "investigation workflow" match the
  document that carries the term in its name.
"""

from __future__ import annotations

import os
import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer


# A substantive policy document states at least one figure: a threshold, a
# currency amount, a year, or a numbered procedure step.
_HAS_NUMBER = re.compile(r"\d")


class TfidfEmbeddings(Embeddings):
    """TF-IDF embedding function fitted on the policy corpus.

    Implements the LangChain ``Embeddings`` interface so it can back an
    ``InMemoryVectorStore`` without calling out to an embedding provider.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddings must index documents first.")
        return self._vectorizer.transform([text]).toarray()[0].tolist()

    def has_vocabulary(self, text: str) -> bool:
        """True when the query shares at least one term with the corpus.

        A query with no overlap embeds to an all-zero vector, for which cosine
        similarity is undefined and the resulting ranking is arbitrary.
        """
        return bool(self._vectorizer.transform([text]).nnz)


def load_policy_documents(policy_directory: str) -> list[dict]:
    """Load policy documents from the supplied directory.

    Returns one dict per document preserving the source filename, the cleaned
    text and the document title.  Non-substantive documents are skipped.
    """
    documents: list[dict] = []

    for filename in sorted(os.listdir(policy_directory)):
        if not filename.endswith(".md"):
            continue

        path = os.path.join(policy_directory, filename)
        with open(path, "r", encoding="utf-8") as file:
            raw = file.read()

        text = clean_document(raw)
        if not text:
            continue

        heading = next(
            (
                line.lstrip("# ").strip()
                for line in text.splitlines()
                if line.startswith("#")
            ),
            filename,
        )

        documents.append(
            {
                "source": filename,
                "text": text,
                "metadata": {"title": heading},
            }
        )

    return documents


def clean_document(text: str) -> str:
    """Normalize policy text before chunking.

    Collapses blank lines and trailing whitespace while preserving headings,
    bullets and policy wording verbatim so retrieved passages remain quotable.

    Returns an empty string for non-substantive documents (no figures, no
    numbered steps), which keeps administrative decoys out of the index.
    """
    if not text:
        return ""

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    cleaned = "\n".join(line for line in lines if line).strip()

    if not _HAS_NUMBER.search(cleaned):
        return ""

    return cleaned


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split policy documents into retrieval chunks.

    Splits on paragraph and line boundaries first so a single policy rule
    (one bullet, one procedure step) is never cut in half.  Each chunk keeps
    its source filename for citation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[dict] = []

    for document in documents:
        for position, piece in enumerate(splitter.split_text(document["text"])):
            chunks.append(
                {
                    "chunk_id": f"{document['source']}#{position}",
                    "source": document["source"],
                    "text": piece,
                    "metadata": dict(document["metadata"]),
                }
            )

    return chunks


def build_index(chunks: list[dict]) -> InMemoryVectorStore:
    """Build a reusable in-memory vector index over the policy chunks.

    The indexed text is prefixed with the source filename and document title
    so that document-identifying queries ("Singapore procedure", "high-risk
    jurisdiction list") match on those terms too.  The original passage is
    kept in metadata and is what retrieval returns.
    """
    documents = [
        Document(
            id=chunk["chunk_id"],
            page_content=(
                f"Source: {chunk['source'].replace('_', ' ').replace('.md', '')}\n"
                f"{chunk['metadata'].get('title', '')}\n"
                f"{chunk['text']}"
            ),
            metadata={
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                **chunk["metadata"],
            },
        )
        for chunk in chunks
    ]

    embeddings = TfidfEmbeddings()
    return InMemoryVectorStore.from_documents(documents, embeddings)


def retrieve(
    index: InMemoryVectorStore,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve the most relevant policy chunks, highest score first."""
    if not query or not query.strip():
        return []

    embeddings = index.embedding
    if isinstance(embeddings, TfidfEmbeddings) and not embeddings.has_vocabulary(query):
        # No lexical overlap with the corpus: any ranking would be arbitrary.
        return []

    results = index.similarity_search_with_score(query, k=top_k)

    return [
        {
            "source": document.metadata["source"],
            "text": document.metadata["text"],
            "score": round(float(score), 4),
            "chunk_id": document.metadata["chunk_id"],
        }
        for document, score in results
        if score > 0
    ]


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Rerank retrieved candidates.

    Keeps the retriever's ordering but de-duplicates by source document, so
    top-k spends its slots on k distinct policies rather than on several
    chunks of the same one — which is what the agent needs for citations.
    """
    seen: set[str] = set()
    reranked: list[dict] = []

    for candidate in candidates:
        if candidate["source"] in seen:
            continue
        seen.add(candidate["source"])
        reranked.append(candidate)

        if len(reranked) == top_k:
            break

    return reranked


def retrieve_policy_evidence(
    index: InMemoryVectorStore,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Convenience entry point used by ``tools/policy_tools.search_policy``."""
    candidates = retrieve(index, query, top_k=10)
    return rerank(query, candidates, top_k=top_k)
