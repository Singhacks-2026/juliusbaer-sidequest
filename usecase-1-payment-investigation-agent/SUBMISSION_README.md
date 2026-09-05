# Payment Investigation Agent — submission notes

**Julius Baer AI Hackathon · jb-sidequest · Use case 1**

```bash
pip install -r requirements.txt anthropic
cp .env.example .env            # add ANTHROPIC_API_KEY (optional — see "Without a key")
python main.py --questions questions/questions.json --output submission.json
```

`main.py` is unmodified. The run is non-interactive and never crashes: every
question is wrapped so a failure in any stage degrades to a grounded
deterministic answer rather than an exception.

## Design — detectors compute, the model narrates

```
question + payment_id  (nothing else)
   │
   ▼
agent/agent.py      Claude tool-calling loop — decides which tools the question
   │                  needs, calls them, then writes the answer as JSON
   │
   ├─ assess_payment_policy   facts · applicable policies · every threshold
   │                          comparison · high-risk code check        (assessor)
   ├─ check_structuring       24h same-beneficiary window vs threshold  (assessor)
   ├─ get_payment · get_client_profile · get_client_payments ·
   │  aggregate_beneficiary_24h · find_repeated_beneficiaries          (data tools)
   └─ search_policy · get_policy_document                              (RAG)
   │
   ▼
normalise           facts from the assessor · tools_used from the call log ·
   │                  citations ∩ files a tool actually returned
   ▼
submission.json     answer · citations · facts · tools_used
ui/investigations.json   audit trace (every tool call, every passage retrieved)
```

The decision that shapes everything else: **the language model chooses the
tools and writes the prose; it never originates a number or a verdict.**
`agent/assessor.py` exposes the policy logic as two tools — thresholds and
jurisdiction in one, the 24-hour structuring window in the other — so the
model calls what a question needs (a structuring question pulls
`check_structuring` and `aggregate_beneficiary_24h`; a threshold question
doesn't). Tool calls made *inside* an assessor are recorded too, so
`tools_used` lists every tool that really ran, in order. If the model never
requested the policy assessment, it is run afterwards so `facts` is always
complete — and that run is recorded as well.

`rag/pipeline.py`: one chunk per policy rule, TF-IDF cosine (numpy), and a
reranker that penalises self-negating "this document contains no…" chunks so
the four decoys never reach a citation. Citations proposed by the model are
filtered to files a tool actually returned.

If no model is available, or a call fails, a deterministic narrator runs the
full assessment and composes the answer from it — same tools, same numbers.

### Things the data is designed to trip

| Trap (DATA_NOTES.md) | Handling |
|---|---|
| `beneficiary_country` disagrees with `beneficiary_country_code` | The code is authoritative; the mismatch is stated in the answer as a data-quality indicator |
| 24h window must filter on client **and** beneficiary | `aggregate_beneficiary_24h` filters on both; same-date payments to other beneficiaries (e.g. P50183) are excluded |
| No time component on dates | Same calendar date = one window; stated as an assumption |
| No FX data | CHF compared natively against the Swiss procedure; otherwise 1:1, stated explicitly in `assumptions` |
| Decoy policy documents | Rejected by the reranker on content, not filename |
| "Policy trigger ≠ suspicious" | Every answer says what review is required and never labels a payment suspicious |

Thresholds and the high-risk list are **parsed from the policy documents at
runtime** (`above USD 100,000 equivalent require enhanced review`), so every
rule the assessor applies is traceable to a file. Nothing keys on
`question_id` or on question wording: the model reads the question and
decides; the keyless fallback composes one fixed shape from the assessment.

## Without a key

Run with no `.env` and the output is still valid and grounded — the
deterministic narrator writes the prose. With `ANTHROPIC_API_KEY` set, Claude
(`ANTHROPIC_MODEL`, default `claude-opus-5`) narrates instead and may call
additional tools; the `narrator` field in `ui/investigations.json` records
which path produced each answer.

## Retrieval backend

`RAG_BACKEND=local` (default) — from-scratch TF-IDF, offline, deterministic.
`RAG_BACKEND=cloudflare` — embeds the same chunks with Cloudflare Workers AI
(`@cf/baai/bge-base-en-v1.5`; needs `CF_ACCOUNT_ID`, `CF_API_TOKEN`) and
falls back to local on any error. The corpus is nine files, so the local
index is the right default; the Cloudflare path exists to show the pipeline
is backend-agnostic.

## Review UI

```bash
python3 -m http.server 8765
open http://127.0.0.1:8765/ui/
```

A static page over `ui/investigations.json`: computed checks above the
written answer, citations that expand to the retrieved passage, the documents
retrieval excluded, and the ordered tool trace. Julius Baer palette
(`#141E55`), Jost display, IBM Plex Mono for every figure. No build step,
no backend — the graded run does not depend on it.
