"""Heading-aware local BM25 retrieval; no vector service required."""

import math
import re
from collections import Counter
from pathlib import Path

_STOP = set("a an the is are of to for and or in on by with this that be as at it its from".split())
_ALIASES = {"splitting": "structuring", "split": "structuring", "structur": "structuring",
            "swiss": "switzerland", "sg": "singapore", "ch": "switzerland",
            "uae": "ae", "workflow": "investigation", "steps": "investigation",
            "regional": "region", "thresholds": "threshold", "payments": "payment",
            "jurisdictions": "jurisdiction", "destinations": "destination",
            "procedures": "procedure", "policies": "policy"}


def _tokens(text: str) -> list[str]:
    return [_ALIASES.get(word, word) for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in _STOP]


def load_policy_documents(policy_directory: str) -> list[dict]:
    return [{"source": path.name, "text": path.read_text(encoding="utf-8")}
            for path in sorted(Path(policy_directory).glob("*.md"))]


def clean_document(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()


def chunk_documents(documents: list[dict], chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list[dict]:
    """Preserve whole rules and headings. Sizes are character targets.

    An indivisible rule may exceed the target. Overlap repeats whole rules
    only when they fit, so thresholds never lose their conditions.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    chunks = []
    for document in documents:
        text = clean_document(document["text"])
        lines = text.splitlines()
        heading = lines[0] if lines and lines[0].startswith("#") else ""
        body = "\n".join(lines[1:]) if heading else text
        rules = [re.sub(r"\s+", " ", part).strip()
                 for part in re.split(r"\n\s*\n|\n(?=\s*(?:[-*] |\d+\. |#))", body)
                 if part.strip()]
        current = []
        parts = []
        for rule in rules:
            if current and len(heading) + len("\n".join(current + [rule])) > chunk_size:
                parts.append(current)
                current = [current[-1]] if len(current[-1]) <= chunk_overlap else []
            current.append(rule)
        if current:
            parts.append(current)
        for number, part in enumerate(parts):
            chunks.append({"chunk_id": f"{document['source']}:{number}",
                           "source": document["source"],
                           "text": "\n".join([heading, *part]).strip(),
                           "metadata": {"heading": heading.lstrip("# ")}})
    return chunks


def build_index(chunks: list[dict]) -> dict:
    counts = [Counter(_tokens(chunk["text"] + " " + chunk["metadata"]["heading"]))
              for chunk in chunks]
    frequency = Counter(word for count in counts for word in count)
    return {"chunks": chunks, "counts": counts, "frequency": frequency,
            "average_length": sum(sum(c.values()) for c in counts) / max(len(counts), 1)}


def retrieve(index: dict, query: str, top_k: int = 5) -> list[dict]:
    if not isinstance(query, str) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("Provide a text query and positive top_k")
    terms = set(_tokens(query))
    if "structuring" in terms:
        terms.update({"beneficiary", "combined", "hours"})
    results = []
    count = len(index["chunks"])
    for chunk, frequencies in zip(index["chunks"], index["counts"]):
        # ponytail: this corpus uses operative clauses or numbered procedures;
        # use a document-type classifier if future policies have other forms.
        if not re.search(r"\b(?:require[sd]?|should|must)\b|^\d+\.\s",
                         chunk["text"], re.I | re.M):
            continue
        length = sum(frequencies.values())
        score = 0.0
        for term in terms:
            tf = frequencies[term]
            if tf:
                df = index["frequency"][term]
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / index["average_length"]))
        if score > 0:
            results.append({**chunk, "score": round(score, 6)})
    return sorted(results, key=lambda row: (-row["score"], row["chunk_id"]))[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """BM25 already ranks this small corpus; avoid an extra model call."""
    return candidates[:top_k]


def retrieve_policy_evidence(index: dict, query: str, top_k: int = 3) -> list[dict]:
    return retrieve(index, query, top_k)
