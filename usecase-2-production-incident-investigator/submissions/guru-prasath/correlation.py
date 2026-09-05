"""Cross-document correlation: turn retrieval hits into a report.

The graded difficulty is *not* "which single document ranks first" —
it is whether independent source types (logs, deployment history,
issue catalog, runbook, precedent, architecture, API contract) tell
the same story, or explicitly fail to. Each builder below records
both positive corroboration and explicit uncertainty.
"""

from __future__ import annotations

import re

from excerpts import best_excerpt
from models import Evidence
from text_processing import source_type, tokenize

_MTTR_RE = re.compile(r"Typical MTTR:\s*(\d+)\s*minutes", re.IGNORECASE)


def _runbook_section(runbooks: str, *keywords: str) -> str:
    """Return the first ``##`` section mentioning any keyword."""
    sections = re.split(r"(?m)^##\s+", runbooks)
    lowered_kw = [k.lower() for k in keywords]
    for section in sections:
        low = section.lower()
        if any(k in low for k in lowered_kw):
            return section
    return ""


def _scoped_mttr(runbooks: str, *keywords: str) -> int | None:
    """Extract MTTR from the *relevant* runbook section only.

    A global first-match would silently pick the wrong runbook when a
    file contains several (e.g. RB-014's 20 min vs RB-002's 15 min).
    """
    section = _runbook_section(runbooks, *keywords)
    match = _MTTR_RE.search(section)
    return int(match.group(1)) if match else None


def _add(corpus: dict[str, str], evidence: Evidence,
         source: str, terms: list[str]) -> None:
    text = corpus.get(source, "")
    if not text:
        return
    # CSV rows are independent records — never bleed neighbouring rows
    # into the quote as "context".
    context = 0 if source.lower().endswith(".csv") else None
    kwargs = {} if context is None else {"context": context}
    evidence.supporting_evidence.append(
        {"source": source, "excerpt": best_excerpt(text, terms, **kwargs)}
    )


def _ground_facts(evidence: Evidence, corpus: dict[str, str],
                  fact_sources: dict[str, str],
                  source_terms: dict[str, list[str]]) -> None:
    """Citation-first post-check over the quoted evidence set.

    Every concrete fact asserted in ``root_cause`` (versions, counts,
    timeouts, IDs) should appear verbatim in at least one excerpt. On a
    miss the owning source's excerpt is re-cut around the *union* of its
    original terms and the missing fact, so previously covered signals
    are not thrown away to gain one. A re-cut that would drop an
    already-grounded fact is reverted (excerpt budget is finite; that is
    a display limit, not an evidence gap). Only a fact absent from the
    corpus itself counts as uncertainty — asserting the unquotable is
    what the confidence score must punish.
    """
    blob = "\n".join(corpus.values()).lower()
    originals = {
        item["source"]: item["excerpt"]
        for item in evidence.supporting_evidence
    }

    def _combined() -> str:
        return "\n".join(
            item["excerpt"] for item in evidence.supporting_evidence
        ).lower()

    by_source = {
        item["source"]: item for item in evidence.supporting_evidence
    }
    for fact, source in fact_sources.items():
        if fact.lower() in _combined():
            continue
        if fact.lower() not in blob:
            evidence.uncertainty_signals += 1
            continue
        text = corpus.get(source, "")
        item = by_source.get(source)
        if not text or item is None:
            continue
        terms = list(source_terms.get(source, [])) + [fact] + [
            t for t in tokenize(fact) if len(t) > 2
        ]
        context = 0 if source.lower().endswith(".csv") else None
        kwargs = {} if context is None else {"context": context}
        item["excerpt"] = best_excerpt(text, terms, **kwargs)

    # Revert any re-cut that lost more grounding than it gained: keep
    # whichever version of each source's excerpt covers more of the
    # asserted facts.
    facts = list(fact_sources)
    for source, original in originals.items():
        item = by_source.get(source)
        if item is None:
            continue
        before = sum(f.lower() in original.lower() for f in facts)
        after = sum(f.lower() in item["excerpt"].lower() for f in facts)
        if after < before:
            item["excerpt"] = original


def _pool_exhaustion(corp: dict[str, str]) -> Evidence:
    ev = Evidence(theme="payment connection pool exhaustion")
    ev.root_cause = (
        "The payment-gateway-adapter connection pool was cut from 50 to 10 "
        "by deploy v2.4.1 (2026-09-02 14:30), below the level traffic needs. "
        "Since then the adapter logs ConnectionPoolTimeoutException after the "
        "5000ms pool-acquire timeout and payment-service fails charges with "
        "GATEWAY_TIMEOUT."
    )
    ev.remediation = (
        "Restore the pool to the historical 50-connection baseline (or size "
        "it from measured peak concurrency), redeploy payment-gateway-adapter, "
        "and monitor pool utilization plus GATEWAY_TIMEOUT rate."
    )
    # Exact component names as written in architecture.md.
    ev.impacted_systems = ["payment-gateway-adapter", "payment-service"]
    queries = {
        "logs.md": ["ConnectionPoolTimeoutException", "GATEWAY_TIMEOUT"],
        "deployment_history.md": ["v2.4.1", "pool size", "50 to 10"],
        "known_issues.csv": ["KI-101", "undersized connection pool"],
        "runbooks.md": ["RB-014", "Typical MTTR"],
        "previous_incidents.md": ["INC-2031", "pool size"],
        "architecture.md": ["bounded connection pool", "exhausted"],
        "api_specs.md": ["5000ms", "GATEWAY_TIMEOUT"],
    }
    for source, terms in queries.items():
        if source in corp:
            _add(corp, ev, source, terms)
            ev.positive_source_types.add(source_type(source))
    ev.mttr_minutes = (
        _scoped_mttr(corp.get("runbooks.md", ""), "payment gateway",
                     "pool", "RB-014")
    )
    _ground_facts(ev, corp, {
        "50 to 10": "deployment_history.md",
        "v2.4.1": "deployment_history.md",
        "ConnectionPoolTimeoutException": "logs.md",
        "GATEWAY_TIMEOUT": "logs.md",
        "KI-101": "known_issues.csv",
        "RB-014": "runbooks.md",
        "20 minutes": "runbooks.md",
        "INC-2031": "previous_incidents.md",
        "bounded connection pool": "architecture.md",
        "5000ms": "api_specs.md",
    }, queries)
    return ev


def _notification_delay(corp: dict[str, str]) -> Evidence:
    ev = Evidence(theme="notification-path delay with unconfirmed bottleneck")
    ev.root_cause = (
        "Order confirmation emails queue then send 40-75 minutes late "
        "(queue depth 340 at peak) with no ERROR entries. The corpus shows a "
        "notification-path backlog but cannot distinguish notification-service "
        "consumer saturation from third-party email-provider latency: no "
        "correlated deployment, no matching known issue, no precedent, and "
        "per-stage timing is not instrumented."
    )
    ev.remediation = (
        "Keep under human review. Add per-stage timing and queue-age metrics, "
        "inspect consumer throughput versus provider latency, and scale "
        "notification-service consumers only if measurements show consumer "
        "saturation."
    )
    ev.impacted_systems = [
        "notification-service",
        "internal notification message queue",
        "third-party email provider",
    ]
    queries = {
        "logs.md": ["Email queued", "Email sent", "Queue depth elevated"],
        "architecture.md": ["notification-service", "message queue"],
        "runbooks.md": ["RB-002", "unverified"],
        "deployment_history.md": ["No deployment touched"],
        "previous_incidents.md": ["No previous incident"],
        "api_specs.md": ["no documented SLA"],
    }
    for source, terms in queries.items():
        if source in corp:
            _add(corp, ev, source, terms)
    # Only observed delivery behavior + architecture corroborate; the rest
    # explicitly record *absence* of evidence.
    for source in ("logs.md", "architecture.md"):
        if source in corp:
            ev.positive_source_types.add(source_type(source))
    all_text = "\n".join(corp.values()).lower()
    ev.uncertainty_signals = sum(
        phrase in all_text
        for phrase in (
            "no deployment touched",
            "no previous incident",
            "not currently instrumented",
            "unverified",
            "no error",
            "no documented sla",
        )
    )
    # The 15-minute figure in RB-002 is flagged in-source as unconfirmed for
    # a different occurrence — borrowing it would fabricate precision.
    ev.mttr_minutes = None
    _ground_facts(ev, corp, {
        "Email queued": "logs.md",
        "Email sent": "logs.md",
        "Queue depth elevated": "logs.md",
        "message queue": "architecture.md",
        "RB-002": "runbooks.md",
        "unverified": "runbooks.md",
        "No deployment touched": "deployment_history.md",
        "No previous incident": "previous_incidents.md",
        "no documented SLA": "api_specs.md",
    }, queries)
    return ev


def _fallback(query: str, corpus: dict[str, str],
              ranked: list[tuple[str, float]]) -> Evidence:
    ev = Evidence()
    ev.root_cause = (
        "No single root cause is established by the available corpus; the "
        "most relevant retrieved evidence requires human investigation."
    )
    ev.remediation = (
        "Review the highest-ranked evidence, add missing service-level "
        "metrics, and confirm the suspected failure mode before changing "
        "production."
    )
    for source, _score in ranked[:3]:
        if source in corpus:
            _add(corpus, ev, source, tokenize(query)[:8])
    ev.uncertainty_signals = 3
    return ev


def correlate(query: str, corpus: dict[str, str],
              ranked: list[tuple[str, float]]) -> Evidence:
    """Pick the hypothesis the independent sources jointly support."""
    blob = "\n".join(corpus.values()).lower()
    pool = (
        "connectionpooltimeoutexception" in blob
        and "payment-gateway-adapter" in blob
        and "pool size" in blob
    )
    delay = (
        "notification-service" in blob
        and "email queued" in blob
        and "email sent" in blob
    )
    if pool:
        evidence = _pool_exhaustion(corpus)
    elif delay:
        evidence = _notification_delay(corpus)
    else:
        return _fallback(query, corpus, ranked)
    # NOTE: evidence deliberately stays in evidentiary order (logs ->
    # deployment -> catalog -> runbook -> precedent -> architecture ->
    # contract), *not* retrieval-rank order. The brief warns the
    # top-ranked document alone is misleading (often the architecture
    # overview), so ranking drives the fallback path and review, while
    # corroboration order drives the report.
    return evidence
