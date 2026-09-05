# RAG Pipeline

The original method-only starter is now implemented with local TF-IDF retrieval.
Source filenames and complete policy rules are preserved. See
../IMPLEMENTATION.md for the implementation and verification commands.

## Minimum viable implementation

You can implement:

1. read Markdown files;
2. split into chunks;
3. tokenize;
4. score query/chunk overlap;
5. return top-k chunks with source names.

That is enough for a working baseline.

## Better implementation

If time permits:
- TF-IDF/BM25;
- embeddings;
- metadata filtering;
- reranking;
- source-aware retrieval.

## Important

Preserve source names.

The final assistant should be able to say which policy document supports its
answer.
