# Production Incident Investigator

Komal Chandiramani
komal.chandiramani7@gmail.com
+6584381135

Run: needs `pip install nltk` (Porter stemmer).

## Ingestion

Chunking is based on what the text looks like, not which file it came from, so the same rules run on both incidents.

- Log blocks: one chunk per line, with timestamp, level and component parsed out.
- CSV and markdown tables: one chunk per row, with column names glued onto each value so a row reads on its own.
- Bullet lists: one chunk per bullet.
- Everything else: split on `##` headings, or blank lines if there are none.

56 chunks for A, 40 for B. Every chunk keeps its raw text, so the submitted excerpts are word-for-word from the source. I drop the paragraph at the end of each `logs.md` since it explains the answer and a real log file wouldn't have it.

Normalization is Porter stemming. The query says "Payments" and "failing", the docs say "payment" and "failed". The stem is emitted next to the original word, not instead of it, so `payment-gateway-adapter` and `ConnectionPoolTimeoutException` stay intact.

## Retrieval

**Cosine alone was bad.** `architecture.md` came out rank 1 for A. KI-101 was rank 16, the v2.4.1 deployment row rank 24, and the five `ConnectionPoolTimeoutException` ERROR lines ranked 42–46 out of 47.

**BM25 barely helped.** Spearman 0.978 against cosine, 9 of the top 10 identical. I thought length normalization would lift the short log lines. It didn't, because the issue is that `payment` appears in 38 of 56 chunks so it carries almost no weight, and that ERROR line matches nothing else in the query.

**The real problem is vocabulary.** The query is about symptoms ("payments are failing after yesterday's deployment"). The documents that solve it are about causes ("ConnectionPoolTimeoutException", "pool size", "v2.4.1"). Those words are never in the query, so no keyword scorer connects them in one pass.

**So: two-hop search.** Take the top hits, pull their most distinctive terms, search again. The second query picks up `pool`, `connect`, `timeout`, `deploy`, `size` — which hits KI-101 directly, since its signature says "ConnectionPoolTimeoutException in payment-gateway-adapter". The two rankings are fused with RRF.

| Evidence | cosine | BM25 | two-hop |
|---|---|---|---|
| ConnectionPoolTimeout ERROR line | 42 | 43 | 18 |
| KI-101 | 16 | 16 | 7 |
| v2.4.1 deployment row | 24 | 24 | 16 |

My first version let the second query *replace* the first. B's top hit is a decoy, so the expansion drifted into payment vocabulary and a payment deployment row landed at rank 9 of an email incident. The second hop now adds to the query instead of replacing it: A's ranks drop a few places, B no longer pulls in payment records.

Chunks scoring below 75% of the top score are dropped.

## Correlation

Pick the component that has ERROR/WARN log lines and is most present in the retrieved evidence. A gives `payment-gateway-adapter`, B gives `notification-service`. Then count how many independent source types back it: logs, known issues, deployment history, runbooks, previous incidents.

`architecture.md` and `api_specs.md` are excluded. They describe the system so they match any question, and they're rank 1–2 in both incidents. Letting them corroborate would make B look confident.

Three more rules:

- A chunk that mentions the component but denies the link is a **refutation**, not support. B's files say "No previous incident", "No deployment touched notification-service", "unverified".
- A deployment only counts if it precedes the errors. B's two deployments post-date its incident.
- Only ERROR/WARN log lines count. "Email sent" is not evidence of a fault.

A: 5 of 5 sources agree. B: 1, plus 4 refutations.

## Confidence

Fixed bands on how many sources agree, minus 5 per refutation. 5→92, 4→82, 3→70, 2→52, 1→30, 0→12. Capped at 95, never 100.

A lands at 87. B lands at 10 and is flagged. `needs_human_review` is computed from `confidence_score < 50`, so they can't disagree.

## Design decisions

- Chunk by content shape, not filename, so no rule is tied to a specific incident.
- Store raw text separately from the normalized text, so excerpts stay verbatim.
- Emit stems alongside original tokens rather than replacing them, to keep component and exception names matchable.
- Retrieval returns chunk ids, not filenames, so correlation can work at record level.
- Second hop extends the query instead of replacing it, to avoid expansion drift.
- Confidence bands are fixed constants, not fitted, since there are only two incidents to tune against.
- Corroboration counts distinct source types, not chunks, so five log lines from one file count once.
- Negation is a keyword list restricted to non-log chunks.
- MTTR is taken from the runbook before the previous incident, as the runbook is current guidance.
