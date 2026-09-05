# Use Case 2: Production Incident Investigator — Submission Report

**Author**: Hitesh Shah  
**Phone**: +65 9424 7574  
**Email**: hns78@yahoo.com  
**Track**: Retrieval & Multi-Source Evidence Correlation (RAG-Shaped, Zero-LLM)  

---

## 1. System Design & Architecture

The incident investigator is engineered as a deterministic, multi-stage retrieval and cross-source corroboration pipeline:

```
                               ┌────────────────────────┐
                               │     Query & Corpus     │
                               └───────────┬────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 1. Ingestion & Structured Chunking      │
                      │    - CSV row parsing (known issues)     │
                      │    - Log parsing & level tagging        │
                      │    - Section extraction (RB, INC, Deploy)│
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 2. Hybrid Retrieval & Ranking           │
                      │    - TF-IDF sublinear n-gram similarity │
                      │    - Entity & symptom contextual boost  │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 3. Log Disambiguation & Noise Filtering │
                      │    - Match & filter background noise     │
                      │      (KI-055, KI-093, KI-114, KI-142)   │
                      │    - Isolate true anomaly signatures    │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 4. Cross-Source Corroboration Engine    │
                      │    - 5 independent evidentiary pillars: │
                      │      [Logs, Deploy, KI, Incident, RB]   │
                      │    - Detect explicit negative signals   │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ 5. Calibrated Confidence & Synthesis    │
                      │    - Corroboration consensus score      │
                      │    - Strict rule: human_review = (C<50) │
                      │    - Structured report generation       │
                      └─────────────────────────────────────────┘
```

### Key Modules:
1. **`_ingest_corpus`**:
   - Converts heterogeneous raw documents into typed `DocumentChunk` records.
   - Treats `known_issues.csv` as discrete rows (`known_issues.csv#KI-xxx`) rather than an unstructured blob.
   - Breaks `logs.md` into timestamped entries with extracted severity (`INFO`, `WARN`, `ERROR`).
   - Slices `runbooks.md`, `previous_incidents.md`, and `deployment_history.md` into semantic sections.

2. **`_retrieve_relevant_documents`**:
   - Computes cosine similarity between TF-IDF representations of the query and all corpus chunks (n-gram range (1, 2) with sublinear term-frequency scaling).
   - Contextually boosts chunks containing error signatures and direct symptom terms.

3. **`_analyze_logs_and_noise`**:
   - Real production logs contain background noise from concurrent benign issues (e.g. search indexing lag, cosmetic email rendering, DB failovers).
   - Disambiguates background noise from the primary incident anomaly.

4. **`_correlate_evidence`**:
   - Evaluates multi-source agreement across five independent evidentiary pillars:
     - **Logs**: Error spikes vs isolated warnings vs normal latency.
     - **Deployments**: Correlated configuration changes vs zero recent changes.
     - **Known Issues Catalog**: Exact signature matches vs missing entries.
     - **Previous Incidents**: Historical precedents vs first-time observations.
     - **Runbooks**: Verified procedures with typical MTTR vs unverified/incomplete runbooks.
     - **Architecture Specs**: Verifies component dependency paths.

5. **`_calibrate_confidence`**:
   - Computes an honest, calibrated confidence score (0–100) reflecting multi-source corroboration consensus.
   - Automatically assigns `needs_human_review = (confidence_score < 50)`.

---

## 2. Understanding of the Problem & Key Challenges

Through deep analysis of the problem statement and dataset, four core challenges were identified:

1. **Retrieval Alone is Insufficient (The TF-IDF Trap)**:
   - High text-similarity search hits can be misleading. For example, `architecture.md` contains the highest frequency of system keywords, but does not provide root cause or timestamped causality.
   - A valid conclusion requires verifying that *independent documents agree with each other*.

2. **Log Noise & Decoy Disambiguation**:
   - In both incidents, logs contain entries from unrelated systems (`search-service` reindex lag, `web-frontend` render time, `auth-service` session expiry, `notification-service` HTML fallback).
   - The investigator must not falsely correlate these benign background anomalies with the incident query.

3. **Confidence Calibration vs. "Confidence Theater"**:
   - Standard LLMs and naive classifiers tend to hallucinate plausible-sounding explanations and report high confidence even when evidence is completely absent.
   - For **Incident B**, the correct engineering behavior is to acknowledge that evidence is thin (only a single unverified queue warning, no correlated deployment, no matching known issue, no historical precedent) and assign a low confidence score (<50) requiring human review.

4. **Heterogeneous Document Structures**:
   - Markdown documents, tabular markdown tables, raw unstructured log streams, and CSV catalogs cannot be ingested uniformly without losing structure.

---

## 3. Rationale, Iterations & Tradeoffs

### Approaches Evaluated & Tradeoffs:
- **Naive TF-IDF on Whole Files vs. Granular Chunking**:
  - *Attempted & Abandoned*: Ranking whole files produced noisy scores because large files (like `architecture.md` or full `logs.md`) dominated query similarity.
  - *Adopted*: Granular semantic chunking with per-row CSV parsing and section extraction dramatically improved precision.

- **Corroboration Pillar Consensus vs. Pure Similarity Aggregation**:
  - Rather than summing retrieval similarities (which rewards repetitive documents), the system evaluates independent source types. 5 matching pillars yield high confidence (92%), whereas 1 weak warning with multiple explicit negative signals yields calibrated low confidence (23%).

- **MTTR & System Extraction**:
  - Extracted directly from corroborated runbooks and previous incidents (`20` minutes for Incident A; `15` minutes for Incident B with note of unconfirmed status).

---

## 4. Evaluation Summary

| Incident | Root Cause | Multi-Source Corroboration | Confidence | Needs Human Review | Impacted Systems | MTTR |
|---|---|---|---|---|---|---|
| **Incident A** (`incident_a_pool_exhaustion`) | Pool exhaustion in `payment-gateway-adapter` from deploy v2.4.1 reducing pool size from 50 to 10 | 5/5 Pillars (Logs, Deploy, KI-101, INC-2031, RB-014) | **92.0%** | **False** | `payment-gateway-adapter`, `payment-service` | 20 min |
| **Incident B** (`incident_b_ambiguous_delay`) | Suspected queue delay / downstream email provider latency (unconfirmed due to lack of instrumentation) | 1/5 Pillars (1 WARN log; zero deploys, zero KIs, zero precedents, incomplete RB-002) | **23.0%** | **True** | `notification-service` | 15 min |

---

## 5. Candidate Contact Information

- **Name**: Hitesh Shah
- **Phone**: +65 9424 7574
- **Email**: hns78@yahoo.com
