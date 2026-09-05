"""Ingestion + BM25 retrieval over heterogeneous corpus files.

Every corpus value is treated as plain text: markdown documents are
chunked by paragraph, while ``known_issues.csv`` is split into one
candidate per row so a single relevant issue (e.g. KI-101) can be
retrieved without dragging the whole catalog along.

Ranking is BM25 (Robertson/Spärck Jones, k1=1.2, b=0.75) rather than
plain TF-IDF cosine: term-frequency saturation stops a log dump that
repeats one word from swamping a focused paragraph, and length
normalization stops long files from winning on bulk alone. Both
properties matter here — logs are long and repetitive, runbooks short
and dense. No hand-tuned phrase bonuses: rare domain terms already
outrank generic words through IDF alone.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter

from config import (
    BM25_B,
    BM25_K1,
    LONG_CHUNK_CHARS,
    LONG_CHUNK_LINES,
    LONG_CHUNK_WINDOW,
)
from models import Record
from text_processing import (
    expand_query,
    source_type,
    split_markdown,
    tokenize,
)


def ingest_corpus(corpus: dict[str, str]) -> list[Record]:
    """Normalize the raw ``filename -> text`` corpus into records."""
    records: list[Record] = []
    for source, text in corpus.items():
        if source.lower().endswith(".csv"):
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                row_text = " | ".join(
                    f"{k}: {v}" for k, v in row.items() if v
                )
                issue_id = row.get("issue_id", "row")
                records.append(
                    Record(
                        source=f"{source}#{issue_id}",
                        display_source=source,
                        source_type=source_type(source),
                        text=row_text,
                        tokens=tokenize(row_text),
                    )
                )
        else:
            for chunk in split_markdown(
                text, LONG_CHUNK_CHARS, LONG_CHUNK_LINES, LONG_CHUNK_WINDOW
            ):
                records.append(
                    Record(
                        source=source,
                        display_source=source,
                        source_type=source_type(source),
                        text=chunk,
                        tokens=tokenize(chunk),
                    )
                )

    # BM25 IDF (Lucene variant, always positive) shared across records.
    total = max(len(records), 1)
    doc_freq = Counter()
    for record in records:
        doc_freq.update(set(record.tokens))
    idf = {
        term: math.log((total - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in doc_freq.items()
    }
    for record in records:
        record.idf = idf
    return records


def _bm25(query_terms: list[str], record: Record, avg_len: float,
          unseen_idf: float) -> float:
    counts = Counter(record.tokens)
    if not query_terms or not counts or avg_len <= 0:
        return 0.0
    length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(record.tokens) / avg_len)
    score = 0.0
    # Unique query terms: repeating a word in the question must not
    # inflate its weight (TF saturation applies to documents, and the
    # query here is a short natural-language question).
    for term in set(query_terms):
        freq = counts.get(term, 0)
        if not freq:
            continue
        idf = record.idf.get(term, unseen_idf)
        score += idf * freq * (BM25_K1 + 1) / (freq + length_norm)
    return score


def retrieve(query: str, corpus: dict[str, str]) -> list[tuple[str, float]]:
    """Rank corpus filenames against the query.

    Returns ``[(filename, best_chunk_score)]`` sorted most-relevant
    first. Scores are per-chunk BM25; the best chunk wins for its file.
    """
    records = ingest_corpus(corpus)
    query_terms = expand_query(tokenize(query))
    total = max(len(records), 1)
    avg_len = sum(len(r.tokens) for r in records) / total
    unseen_idf = math.log((total + 0.5) / 0.5 + 1)
    best: dict[str, float] = {}
    for record in records:
        score = _bm25(query_terms, record, avg_len, unseen_idf)
        prev = best.get(record.display_source, 0.0)
        best[record.display_source] = max(prev, score)
    # Deterministic tie-break on filename so repeated runs agree.
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
