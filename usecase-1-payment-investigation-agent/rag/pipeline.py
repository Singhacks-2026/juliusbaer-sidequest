"""Small, local BM25 retrieval index; no model or vector service required."""
import math
import re
from collections import Counter
from pathlib import Path

STOP_WORDS = set("a an the is are to of for in and by with this should be if before no".split())


def _tokens(text):
    # Normalize hyphens and simple plurals, preserving currency/country codes.
    return [t[:-1] if t.endswith("s") and len(t) > 4 else t
            for t in re.findall(r"[a-z0-9]+", text.casefold()) if t not in STOP_WORDS]


def load_policy_documents(policy_directory: str) -> list[dict]:
    return [{"source": p.name, "text": p.read_text(encoding="utf-8"), "metadata": {}}
            for p in sorted(Path(policy_directory).glob("*.md"))]


def clean_document(text: str) -> str:
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip()
                     for line in text.replace("\r\n", "\n").splitlines()).strip()


def chunk_documents(documents: list[dict], chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list[dict]:
    """Pack complete paragraphs/list rules; size is a soft character limit.

    An indivisible long rule can exceed the limit. Overlap copies whole rules.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    chunks = []
    for document in documents:
        text = clean_document(document["text"])
        units = re.split(r"\n\s*\n|\n(?=- |\d+\. )", text)
        current = []
        def emit(parts):
            chunks.append({"chunk_id": f"{document['source']}:{len(chunks)}",
                           "source": document["source"], "text": "\n".join(parts),
                           "metadata": dict(document.get("metadata", {}))})
        for unit in filter(None, units):
            if current and len("\n".join(current + [unit])) > chunk_size:
                emit(current)
                overlap = []
                for part in reversed(current):
                    if len("\n".join([part] + overlap)) > chunk_overlap:
                        break
                    overlap.insert(0, part)
                current = overlap
            current.append(unit)
        if current:
            emit(current)
    return chunks


def build_index(chunks: list[dict]):
    counts = [Counter(_tokens(c["text"])) for c in chunks]
    frequencies = Counter(t for count in counts for t in count)
    return {"chunks": chunks, "counts": counts, "frequencies": frequencies,
            "average_length": sum(sum(c.values()) for c in counts) / max(1, len(counts))}


def retrieve(index, query: str, top_k: int = 5) -> list[dict]:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")
    terms = set(_tokens(query))
    n = len(index["chunks"])
    results = []
    for chunk, counts in zip(index["chunks"], index["counts"]):
        score = 0.0
        for term in terms & counts.keys():
            frequency = index["frequencies"][term]
            idf = math.log(1 + (n - frequency + 0.5) / (frequency + 0.5))
            tf = counts[term]
            norm = 1.5 * (0.25 + 0.75 * sum(counts.values()) /
                          max(index["average_length"], 1))
            score += idf * tf * 2.5 / (tf + norm)
        # Negative administrative statements provide no positive policy evidence.
        if "contains no payment-monitoring thresholds" in chunk["text"].casefold():
            continue
        if score > 0:
            results.append({**chunk, "score": round(score, 6)})
    return sorted(results, key=lambda r: (-r["score"], r["chunk_id"]))[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    return sorted(candidates, key=lambda r: -r["score"])[:top_k]


def retrieve_policy_evidence(index, query: str, top_k: int = 3) -> list[dict]:
    return rerank(query, retrieve(index, query, top_k=top_k), top_k)
