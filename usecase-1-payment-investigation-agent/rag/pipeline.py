"""
Policy RAG pipeline: load → clean → chunk → index → retrieve → evidence.

Lexical BM25 retrieval (TF saturation + length normalization), stdlib
only. Paragraphs are the chunk unit so single policy rules are never
split; every chunk preserves its source filename for citations.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "all", "an", "and", "any", "are",
        "as", "at", "be", "before", "by", "can", "does", "for", "from",
        "has", "have", "how", "in", "into", "is", "it", "its", "of",
        "on", "or", "our", "over", "should", "that", "the", "their",
        "this", "to", "under", "what", "when", "where", "which", "with",
        "you", "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")
_K1 = 1.2
_B = 0.75


def _tokens(text: str) -> list[str]:
    raw = _TOKEN_RE.findall(text.lower())
    terms: list[str] = []
    for tok in raw:
        terms.append(tok)
        if "-" in tok or "_" in tok:
            terms.extend(re.split(r"[-_]", tok))
    out = list(terms)
    for tok in terms:
        if tok.endswith("s") and len(tok) > 3:
            out.append(tok[:-1])
    return [t for t in out if t not in _STOPWORDS and len(t) > 1]


def load_policy_documents(policy_directory: str) -> list[dict]:
    """Load every ``*.md`` file, preserving source filename and text."""
    docs = []
    for name in sorted(os.listdir(policy_directory)):
        if not name.endswith(".md"):
            continue
        with open(
            os.path.join(policy_directory, name), encoding="utf-8"
        ) as fh:
            docs.append(
                {"source": name, "text": fh.read(), "metadata": {}}
            )
    return docs


def clean_document(text: str) -> str:
    """Normalize whitespace while preserving wording and headings."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split on paragraph boundaries; hard-split only oversized units.

    ``chunk_size``/``chunk_overlap`` are character budgets applied only
    when a single paragraph exceeds the budget (rare for policy text).
    """
    chunks = []
    for doc in documents:
        text = clean_document(doc["text"])
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for para in paras:
            if len(para) <= chunk_size:
                units = [para]
            else:
                units = [
                    para[i : i + chunk_size]
                    for i in range(0, len(para), chunk_size - chunk_overlap)
                ]
            for unit in units:
                chunks.append(
                    {
                        "chunk_id": f"{doc['source']}#{len(chunks)}",
                        "source": doc["source"],
                        "text": unit,
                        "metadata": dict(doc.get("metadata", {})),
                    }
                )
    return chunks


def build_index(chunks: list[dict]) -> dict:
    """Build a BM25 index over chunk tokens (implementation-defined)."""
    tokenized = [_tokens(c["text"]) for c in chunks]
    total = max(len(chunks), 1)
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))
    idf = {
        t: math.log((total - f + 0.5) / (f + 0.5) + 1)
        for t, f in df.items()
    }
    avg_len = sum(len(t) for t in tokenized) / total
    return {
        "chunks": chunks,
        "tokens": tokenized,
        "idf": idf,
        "avg_len": avg_len,
        "unseen_idf": math.log((total + 0.5) / 0.5 + 1),
    }


def _bm25(query_terms: list[str], doc_terms: list[str], index: dict) -> float:
    counts = Counter(doc_terms)
    if not query_terms or not counts or index["avg_len"] <= 0:
        return 0.0
    norm = _K1 * (1 - _B + _B * len(doc_terms) / index["avg_len"])
    score = 0.0
    for term in set(query_terms):
        freq = counts.get(term, 0)
        if not freq:
            continue
        idf = index["idf"].get(term, index["unseen_idf"])
        score += idf * freq * (_K1 + 1) / (freq + norm)
    return score


def retrieve(
    index: dict,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """Return top-k ``{source, text, score}`` chunks, best first."""
    query_terms = _tokens(query)
    scored = [
        (_bm25(query_terms, toks, index), i)
        for i, toks in enumerate(index["tokens"])
    ]
    scored.sort(key=lambda p: (-p[0], p[1]))
    out = []
    for score, i in scored[: max(top_k, 0)]:
        chunk = index["chunks"][i]
        out.append(
            {"source": chunk["source"], "text": chunk["text"],
             "score": round(score, 4)}
        )
    return out


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Keep BM25 order (decoys already sink); trivially deterministic."""
    return list(candidates[: max(top_k, 0)])


def retrieve_policy_evidence(
    index: dict,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve broadly, then keep the top-k (over-retrieve + trim)."""
    return rerank(query, retrieve(index, query, top_k=10), top_k=top_k)
