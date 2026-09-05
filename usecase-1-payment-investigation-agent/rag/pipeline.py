"""Small local TF-IDF retrieval pipeline with traceable policy passages."""
import math
import re
from collections import Counter
from pathlib import Path

_STOP = set("a an and are as at be before by for from if in is it of on or per that the their this to using which with would".split())
_ALIASES = {
    "payments": "payment", "thresholds": "threshold", "reviews": "review",
    "requires": "require", "required": "require", "requirements": "require",
    "requirement": "require", "policies": "policy", "procedures": "procedure",
    "splitting": "structuring", "split": "structuring",
    "destinations": "destination", "jurisdictions": "jurisdiction",
    "clients": "client", "beneficiaries": "beneficiary", "facts": "fact",
    "assumptions": "assumption", "workflow": "procedure", "steps": "procedure",
    "regional": "region", "swiss": "switzerland",
}

def _tokens(text: str) -> list[str]:
    return [_ALIASES.get(word, word) for word in re.findall(r"[a-z0-9]+", text.casefold())
            if word not in _STOP]

def load_policy_documents(policy_directory: str) -> list[dict]:
    return [{"source": path.name, "text": path.read_text(encoding="utf-8-sig")}
            for path in sorted(Path(policy_directory).glob("*.md"))]

def clean_document(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip()
                     for line in text.splitlines()).strip()

def chunk_documents(documents: list[dict], chunk_size: int = 500,
                    chunk_overlap: int = 50) -> list[dict]:
    """Size/overlap are character targets; do not split a policy rule mid-rule."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    chunks = []
    for doc in documents:
        text = clean_document(doc["text"])
        if not text:
            continue
        blocks = re.split(r"\n\s*\n|\n(?=(?:[-*] |\d+\. |#{1,6} ))", text)
        heading = next((block for block in blocks if block.startswith("#")), "")
        groups, current = [], []
        for block in blocks:
            if current and len("\n".join(current + [block])) > chunk_size:
                groups.append(current)
                overlap = []
                for previous in reversed(current):
                    if len("\n".join([previous] + overlap)) > chunk_overlap:
                        break
                    overlap.insert(0, previous)
                current = overlap
            current.append(block)
        if current:
            groups.append(current)
        for number, group in enumerate(groups):
            chunk_text = "\n".join(group)
            if heading and heading not in group:
                chunk_text = heading + "\n" + chunk_text
            chunks.append({"chunk_id": f"{doc['source']}:{number}",
                           "source": doc["source"], "text": chunk_text,
                           "metadata": dict(doc.get("metadata", {}))})
    return chunks

def build_index(chunks: list[dict]) -> dict:
    counts = [Counter(_tokens(chunk["text"])) for chunk in chunks]
    frequencies = Counter(word for count in counts for word in count)
    idf = {word: math.log((1 + len(chunks)) / (1 + frequency)) + 1
           for word, frequency in frequencies.items()}
    vectors = [{word: (1 + math.log(count)) * idf[word] for word, count in words.items()}
               for words in counts]
    norms = [math.sqrt(sum(value * value for value in vector.values())) for vector in vectors]
    return {"chunks": chunks, "idf": idf, "vectors": vectors, "norms": norms}

def retrieve(index, query: str, top_k: int = 5) -> list[dict]:
    if top_k <= 0:
        return []
    query_counts = Counter(_tokens(query))
    query_vector = {word: (1 + math.log(count)) * index["idf"][word]
                    for word, count in query_counts.items() if word in index["idf"]}
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    if not query_norm:
        return []
    results = []
    for chunk, vector, norm in zip(index["chunks"], index["vectors"], index["norms"]):
        # A disclaimer that no rules are present is not evidence for a rule.
        if re.search(r"\bcontains? no (?:[\w-]+\s+){0,3}(?:thresholds?|rules?|policy)\b",
                     chunk["text"].casefold()):
            continue
        score = sum(query_vector[word] * vector.get(word, 0) for word in query_vector)
        score = score / (query_norm * norm) if norm else 0
        if score > 0:
            results.append({**chunk, "score": round(score, 8)})
    return sorted(results, key=lambda item: (-item["score"], item["source"], item["chunk_id"]))[:top_k]

def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """Optional stage: retain the deterministic retrieval ranking."""
    return candidates[:max(0, top_k)]

def retrieve_policy_evidence(index, query: str, top_k: int = 3) -> list[dict]:
    return rerank(query, retrieve(index, query, top_k=max(10, top_k)), top_k=top_k)
