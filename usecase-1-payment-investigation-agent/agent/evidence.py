"""
Evidence ledger.

Every tool the agent invokes goes through this object, which is what makes the
submission's ``tools_used``, ``facts`` and ``citations`` fields true by
construction rather than asserted by the LLM:

* ``tools_used``  — the tools that actually ran, in order, deduplicated.
* ``facts``       — deterministic values returned by those tools.
* ``citations``   — policy documents that either supplied retrieved evidence or
                    state a rule that fired.

The LLM never writes into any of these.  It only sees them and writes prose.
"""

import json
from typing import Any, Callable


class EvidenceLedger:
    """Records tool invocations and accumulates the deterministic evidence."""

    def __init__(self) -> None:
        self._tools_used: list[str] = []
        self._calls: list[dict] = []
        self._facts: dict[str, Any] = {}
        self._citations: list[str] = []
        self._policy_evidence: list[dict] = []
        self._assumptions: list[str] = []

    # -- recording ---------------------------------------------------------

    def call(self, name: str, function: Callable, **kwargs) -> Any:
        """
        Invoke a tool, recording the call and its result.

        Tool failures are captured as an error result rather than raised: a
        single bad lookup must not abort the run, since crashing on any
        official question is a disqualifier.
        """
        try:
            result = function(**kwargs)
        except Exception as error:  # noqa: BLE001 - tool errors become evidence
            result = {"error": f"{type(error).__name__}: {error}"}

        if name not in self._tools_used:
            self._tools_used.append(name)

        self._calls.append({"tool": name, "arguments": kwargs, "result": result})
        return result

    def add_facts(self, facts: dict) -> None:
        """Merge deterministic values into the facts object, skipping empties."""
        for key, value in facts.items():
            if value is None or value == [] or value == {}:
                continue
            self._facts[key] = value

    def add_policy_evidence(self, evidence: list[dict]) -> None:
        """Record retrieved policy passages and cite their source documents."""
        for item in evidence or []:
            source = item.get("source")
            if not source:
                continue

            if not any(
                existing["source"] == source and existing["text"] == item.get("text")
                for existing in self._policy_evidence
            ):
                self._policy_evidence.append(
                    {
                        "source": source,
                        "text": item.get("text", ""),
                        "score": item.get("score"),
                    }
                )

            self.cite(source)

    def cite(self, source: str) -> None:
        """Add a policy document to the citation list, preserving first-seen order."""
        if source and source not in self._citations:
            self._citations.append(source)

    def add_assumption(self, assumption: str) -> None:
        if assumption and assumption not in self._assumptions:
            self._assumptions.append(assumption)

    # -- accessors ---------------------------------------------------------

    @property
    def tools_used(self) -> list[str]:
        return list(self._tools_used)

    @property
    def facts(self) -> dict:
        facts = dict(self._facts)
        if self._assumptions:
            facts["assumptions"] = list(self._assumptions)
        return facts

    @property
    def citations(self) -> list[str]:
        return list(self._citations)

    @property
    def policy_evidence(self) -> list[dict]:
        return list(self._policy_evidence)

    def called(self, name: str) -> bool:
        return name in self._tools_used

    def result_of(self, name: str) -> Any:
        """Most recent result for a tool, or ``None`` if it never ran."""
        for record in reversed(self._calls):
            if record["tool"] == name:
                return record["result"]
        return None

    def evidence_bundle(self) -> dict:
        """The compact, JSON-serialisable view handed to the LLM for synthesis."""
        return {
            "facts": self.facts,
            "policy_evidence": self._policy_evidence,
            "tools_used": self._tools_used,
        }

    def as_json(self, max_chars: int = 6000) -> str:
        """Serialise the bundle for a prompt, truncating pathological sizes."""
        text = json.dumps(self.evidence_bundle(), indent=2, default=str)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text
