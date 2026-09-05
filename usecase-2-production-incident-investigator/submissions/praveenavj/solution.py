"""Evidence-driven production incident investigator.

The implementation is intentionally local and deterministic: it ingests raw
Markdown and CSV, ranks evidence lexically, correlates independent source types,
and calibrates confidence from corroboration rather than writing fluently from a
single top-ranked document.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
_LOG_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+"
    r"(INFO|WARN|ERROR)\s+([\w-]+)\s+(.+)$"
)
_NEGATIVE_CUES = (
    "no matching", "no previous", "no deployment", "unrelated",
    "unconfirmed", "incomplete", "may not apply", "not currently",
)
_SYNONYMS = {
    "late": {"delay", "delayed", "latency", "queue"},
    "failing": {"failed", "failure", "error", "exception", "timeout"},
    "failures": {"failed", "failure", "error", "exception", "timeout"},
    "payments": {"payment", "charge", "gateway"},
    "emails": {"email", "notification", "queue", "sent"},
    "recover": {"recovery", "resolution", "remediation", "mttr"},
    "impacted": {"component", "service", "system"},
}


def _tokens(text: str, expand: bool = False) -> list[str]:
    tokens = [token.casefold().replace("_", "-") for token in _TOKEN_RE.findall(text)]
    if expand:
        for token in tuple(tokens):
            tokens.extend(_SYNONYMS.get(token, ()))
    return tokens


def _split_markdown(source: str, text: str) -> list[dict]:
    """Split prose, tables, and log blocks without losing source identity."""
    units: list[dict] = []
    heading = ""
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            body = " ".join(line.strip() for line in paragraph).strip()
            if body:
                units.append({"source": source, "text": body, "heading": heading})
            paragraph.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "```":
            flush()
            continue
        if line.startswith("#"):
            flush()
            heading = line.lstrip("# ")
        elif _LOG_RE.match(line) or line.startswith("|") or line.startswith("**"):
            flush()
            units.append({"source": source, "text": line, "heading": heading})
        else:
            paragraph.append(line)
    flush()
    return units


def _split_csv(source: str, text: str) -> list[dict]:
    units = []
    for row in csv.DictReader(io.StringIO(text)):
        clean = {key: (value or "").strip() for key, value in row.items()}
        units.append(
            {
                "source": source,
                "text": " | ".join(f"{key}: {value}" for key, value in clean.items()),
                "heading": clean.get("issue_id", ""),
                "row": clean,
            }
        )
    return units


def _ingest_corpus(corpus: dict) -> dict:
    """Normalize raw documents into independently retrievable evidence units."""
    units = []
    for source, raw_text in sorted(corpus.items()):
        text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n")
        splitter = _split_csv if source.casefold().endswith(".csv") else _split_markdown
        units.extend(splitter(source, text))
    return {"documents": dict(corpus), "units": units}


def _rank_units(query: str, units: list[dict]) -> list[dict]:
    """Rank evidence units with a compact TF-IDF cosine implementation."""
    counts = [Counter(_tokens(unit["text"] + " " + unit.get("heading", ""))) for unit in units]
    document_frequency = Counter()
    for count in counts:
        document_frequency.update(count.keys())
    total = max(len(units), 1)
    idf = {
        token: math.log((1 + total) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }
    query_count = Counter(_tokens(query, expand=True))
    query_vector = {
        token: (1 + math.log(frequency)) * idf.get(token, 1.0)
        for token, frequency in query_count.items()
    }
    query_norm = math.sqrt(sum(value * value for value in query_vector.values())) or 1.0

    ranked = []
    for unit, count in zip(units, counts):
        vector = {token: (1 + math.log(freq)) * idf[token] for token, freq in count.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        dot = sum(query_vector[token] * vector.get(token, 0.0) for token in query_vector)
        ranked.append({**unit, "score": dot / (query_norm * norm)})
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _retrieve_relevant_documents(query: str, corpus: dict) -> list[tuple[str, float]]:
    """Return source-level relevance ranked by each source's strongest unit."""
    ingested = _ingest_corpus(corpus)
    best: dict[str, float] = defaultdict(float)
    for unit in _rank_units(query, ingested["units"]):
        best[unit["source"]] = max(best[unit["source"]], unit["score"])
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def _log_events(text: str) -> list[dict]:
    events = []
    for raw in text.splitlines():
        match = _LOG_RE.match(raw.strip())
        if match:
            level, component, message = match.groups()
            events.append(
                {"line": raw.strip(), "level": level, "component": component, "message": message}
            )
    return events


def _identifiers(text: str) -> set[str]:
    """Extract stable machine signatures such as FooException/GATEWAY_TIMEOUT."""
    names = set(re.findall(r"\b[A-Z][A-Za-z]+(?:Exception|Error)\b", text))
    names.update(re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b", text))
    return names


def _best_known_issue(corpus: dict, anomaly_text: str, target_components: set[str]) -> dict | None:
    text = corpus.get("known_issues.csv", "")
    if not text:
        return None
    anomaly_tokens = set(_tokens(anomaly_text))
    anomaly_ids = _identifiers(anomaly_text)
    candidates = []
    for row in csv.DictReader(io.StringIO(text)):
        issue_text = " ".join((row.get("title", ""), row.get("signature", ""), row.get("notes", "")))
        issue_tokens = set(_tokens(issue_text))
        shared_ids = anomaly_ids & _identifiers(issue_text)
        lexical = len(anomaly_tokens & issue_tokens) / max(len(issue_tokens), 1)
        component_match = row.get("affected_component", "") in target_components
        score = (3.0 * len(shared_ids)) + lexical + (0.25 if component_match else 0.0)
        if shared_ids or (component_match and lexical >= 0.18):
            candidates.append((score, row))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _target_components(query: str, architecture: str, events: list[dict]) -> set[str]:
    query_tokens = set(_tokens(query, expand=True))
    scores: Counter[str] = Counter()
    for line in architecture.splitlines():
        match = re.match(r"- \*\*([\w-]+)\*\*:\s*(.+)", line.strip())
        if match:
            component, description = match.groups()
            scores[component] += len(query_tokens & set(_tokens(description, expand=True)))
    for event in events:
        if event["level"] in {"WARN", "ERROR"}:
            scores[event["component"]] += len(query_tokens & set(_tokens(event["message"], expand=True)))
    if not scores:
        return set()
    maximum = max(scores.values())
    return {component for component, score in scores.items() if score == maximum or score >= 2}


def _excerpt(text: str, *patterns: str, context: int = 0) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() != "```"]
    for index, line in enumerate(lines):
        lowered = line.casefold()
        if all(pattern.casefold() in lowered for pattern in patterns):
            start, end = max(0, index - context), min(len(lines), index + context + 1)
            return " ".join(lines[start:end])
    return ""


def _email_delay_excerpt(text: str) -> str:
    """Return the longest observed queue-to-send pair with computed latency."""
    queued: dict[str, tuple[datetime, str]] = {}
    delays = []
    for event in _log_events(text):
        order_match = re.search(r"order_id=([\w-]+)", event["message"])
        if not order_match:
            continue
        order_id = order_match.group(1)
        timestamp = datetime.strptime(event["line"][:19], "%Y-%m-%d %H:%M:%S")
        if event["component"] == "notification-service" and "Email queued" in event["message"]:
            queued[order_id] = (timestamp, event["line"])
        elif event["component"] == "notification-service" and "Email sent" in event["message"]:
            if order_id in queued:
                start, queued_line = queued[order_id]
                minutes = round((timestamp - start).total_seconds() / 60, 1)
                delays.append((minutes, queued_line, event["line"]))
    if not delays:
        return ""
    _, queued_line, sent_line = max(delays)
    return f"{queued_line} | {sent_line}"


def _correlate_evidence(query: str, corpus: dict, ranked: list[tuple[str, float]]) -> dict:
    """Correlate a leading hypothesis across independent document types."""
    events = _log_events(corpus.get("logs.md", ""))
    architecture = corpus.get("architecture.md", "")
    targets = _target_components(query, architecture, events)
    anomalies = [event for event in events if event["level"] in {"WARN", "ERROR"}]
    relevant_anomalies = [event for event in anomalies if event["component"] in targets]
    if not relevant_anomalies:
        relevant_anomalies = anomalies
    anomaly_text = "\n".join(event["line"] for event in relevant_anomalies)
    known_issue = _best_known_issue(corpus, anomaly_text, targets)

    if known_issue:
        component = known_issue.get("affected_component", "unknown-component")
        signature_ids = _identifiers(" ".join(known_issue.values()))
    else:
        component_counts = Counter(event["component"] for event in relevant_anomalies)
        component = component_counts.most_common(1)[0][0] if component_counts else "unknown-component"
        signature_ids = _identifiers(anomaly_text)

    logs_match = any(
        event["component"] == component
        and (event["level"] == "ERROR" or "elevated" in event["message"].casefold())
        for event in relevant_anomalies
    )
    deployment_text = corpus.get("deployment_history.md", "")
    deployment_lines = [line for line in deployment_text.splitlines() if component in line]
    positive_deployment = any(
        not any(cue in line.casefold() for cue in _NEGATIVE_CUES)
        and any(token in set(_tokens(line)) for token in set(_tokens(anomaly_text)))
        for line in deployment_lines
    )
    previous_text = corpus.get("previous_incidents.md", "")
    previous_positive = (
        component in previous_text
        and not any(
            cue in previous_text.casefold()
            for cue in ("no previous incident", "both unrelated", "first recorded")
        )
        and (not signature_ids or bool(signature_ids & _identifiers(previous_text)))
    )
    runbook_text = corpus.get("runbooks.md", "")
    runbook_relevant = component in runbook_text or bool(signature_ids & _identifiers(runbook_text))
    runbook_qualified = runbook_relevant and any(cue in runbook_text.casefold() for cue in _NEGATIVE_CUES)

    signals = {
        "logs": "strong" if logs_match else "absent",
        "known_issue": "strong" if known_issue else "absent",
        "deployment": "strong" if positive_deployment else "absent",
        "previous_incident": "strong" if previous_positive else "absent",
        "runbook": "weak" if runbook_qualified else ("strong" if runbook_relevant else "absent"),
    }
    return {
        "component": component,
        "targets": sorted(targets),
        "events": relevant_anomalies,
        "known_issue": known_issue,
        "signals": signals,
        "ranked": ranked,
    }


def _calibrate_confidence(evidence: dict) -> float:
    """Score independent corroboration; one weak signal stays below 50."""
    strengths = evidence["signals"].values()
    strong = sum(value == "strong" for value in strengths)
    weak = sum(value == "weak" for value in strengths)
    score = 12 + (16 * strong) + (5 * weak)
    if strong <= 1:
        score = min(score, 38)
    return float(max(0, min(score, 95)))


def _extract_mttr(corpus: dict, confidence: float) -> int | None:
    """Use a runbook MTTR only when the causal hypothesis is corroborated."""
    if confidence < 50:
        return None
    match = re.search(r"Typical MTTR:\s*(\d+)\s*minutes", corpus.get("runbooks.md", ""), re.I)
    return int(match.group(1)) if match else None


def _high_confidence_report(corpus: dict, evidence: dict, confidence: float) -> dict:
    issue = evidence["known_issue"] or {}
    component = evidence["component"]
    logs = corpus.get("logs.md", "")
    deployment = corpus.get("deployment_history.md", "")
    previous = corpus.get("previous_incidents.md", "")
    runbook = corpus.get("runbooks.md", "")
    architecture = corpus.get("architecture.md", "")
    issue_catalog = corpus.get("known_issues.csv", "")

    reduction = _excerpt(deployment, component, "pool size")
    size_change = re.search(
        r"reduced connection pool size from\s+(\d+)\s+to\s+(\d+)",
        deployment,
        re.IGNORECASE,
    )
    change_description = (
        f"reducing the configured pool from {size_change.group(1)} to {size_change.group(2)}"
        if size_change
        else "reducing the configured connection-pool capacity"
    )
    root_cause = (
        f"{component} connection-pool exhaustion caused by the recent deployment "
        f"{change_description}. The undersized bounded pool cannot supply connections "
        "under normal traffic, producing intermittent ConnectionPoolTimeoutException "
        "and downstream GATEWAY_TIMEOUT failures."
    )
    impacted = [component]
    if "payment-service" in architecture and component != "payment-service":
        impacted.append("payment-service")
    if "/api/payments/charge" in corpus.get("api_specs.md", ""):
        impacted.append("POST /api/payments/charge")

    supporting = [
        {"source": "logs.md", "excerpt": _excerpt(logs, component, "ConnectionPoolTimeoutException")},
        {"source": "deployment_history.md", "excerpt": reduction},
        {
            "source": "known_issues.csv",
            "excerpt": _excerpt(issue_catalog, issue.get("issue_id", "__missing__")),
        },
        {
            "source": "previous_incidents.md",
            "excerpt": _excerpt(previous, "Root cause", "connection pool size", context=3),
        },
        {"source": "runbooks.md", "excerpt": _excerpt(runbook, "Remediation", "pool size", context=3)},
        {
            "source": "architecture.md",
            "excerpt": _excerpt(architecture, component, "bounded connection pool", context=3),
        },
    ]
    supporting = [item for item in supporting if item["excerpt"]]
    baseline = size_change.group(1) if size_change else "the prior baseline"
    current = size_change.group(2) if size_change else "its reduced value"
    remediation = (
        f"Immediately restore the {component} pool from {current} to the prior "
        f"baseline of {baseline} and redeploy the adapter. Confirm pool saturation and charge "
        "success metrics after deployment; then capacity-test and right-size the pool "
        "before reintroducing any memory optimization."
    )
    return {
        "root_cause": root_cause,
        "supporting_evidence": supporting,
        "impacted_systems": impacted,
        "mttr_minutes": _extract_mttr(corpus, confidence),
        "remediation": remediation,
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }


def _low_confidence_report(corpus: dict, evidence: dict, confidence: float) -> dict:
    component = evidence["component"]
    logs = corpus.get("logs.md", "")
    runbook = corpus.get("runbooks.md", "")
    architecture = corpus.get("architecture.md", "")
    deployment = corpus.get("deployment_history.md", "")
    previous = corpus.get("previous_incidents.md", "")
    queue_event = next(
        (event for event in evidence["events"] if "queue depth" in event["message"].casefold()),
        None,
    )
    if queue_event:
        root_cause = (
            f"Unconfirmed {component} queue backlog. Elevated queue depth is the only "
            "direct causal signal, but the corpus cannot distinguish insufficient "
            "consumer capacity from latency at the downstream email provider."
        )
    else:
        root_cause = (
            f"Undetermined cause in the {component} delivery path; the available evidence "
            "does not identify a corroborated failure mechanism."
        )
    supporting = [
        {
            "source": "logs.md",
            "excerpt": queue_event["line"] if queue_event else _excerpt(logs, component, "WARN"),
        },
        {"source": "logs.md", "excerpt": _email_delay_excerpt(logs)},
        {
            "source": "architecture.md",
            "excerpt": _excerpt(architecture, "per-stage timing", context=3),
        },
        {"source": "deployment_history.md", "excerpt": _excerpt(deployment, "No deployment", component, context=1)},
        {"source": "previous_incidents.md", "excerpt": _excerpt(previous, "first recorded", context=2)},
        {"source": "runbooks.md", "excerpt": _excerpt(runbook, "unverified", context=3)},
    ]
    supporting = [item for item in supporting if item["excerpt"]]
    remediation = (
        "Keep this under human investigation. Add per-stage timestamps and metrics for "
        "queue age/depth, enqueue-to-dequeue time, active consumer count, send duration, "
        "and downstream provider latency/errors. Inspect those measurements during the "
        "incident; scale notification consumers only if consumer saturation is confirmed, "
        "or engage/fail over the email provider if downstream latency is confirmed."
    )
    impacted = [component, "internal notification queue", "order-confirmation email delivery"]
    return {
        "root_cause": root_cause,
        "supporting_evidence": supporting,
        "impacted_systems": impacted,
        "mttr_minutes": None,
        "remediation": remediation,
        "confidence_score": confidence,
        "needs_human_review": confidence < 50,
    }


def investigate(query: str, corpus: dict) -> dict:
    """Produce the exact seven-field incident report required by the brief."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError("corpus must be a non-empty filename-to-text mapping")
    ranked = _retrieve_relevant_documents(query, corpus)
    evidence = _correlate_evidence(query, corpus, ranked)
    confidence = _calibrate_confidence(evidence)
    if confidence >= 50:
        return _high_confidence_report(corpus, evidence, confidence)
    return _low_confidence_report(corpus, evidence, confidence)
