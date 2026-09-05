"""Agent-loop test using an in-memory Chat Completions stand-in."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent import run_agent


class _Completions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            function = SimpleNamespace(
                name="get_payment",
                arguments=json.dumps({"payment_id": "P50001"}),
            )
            message = SimpleNamespace(
                content="", tool_calls=[SimpleNamespace(id="call-1", function=function)]
            )
        else:
            message = SimpleNamespace(
                content=json.dumps(
                    {
                        "answer": "Enhanced, RM, and additional review are required before release.",
                        "citations": [
                            "global_payment_policy.md",
                            "regional_singapore.md",
                            "high_risk_jurisdictions.md",
                            "decoy_operational_1.md",
                        ],
                    }
                ),
                tool_calls=[],
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AgentTests(unittest.TestCase):
    def test_agent_completes_evidence_and_normalizes_output(self):
        completions = _Completions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        with patch("agent.agent._client", return_value=fake_client):
            result = run_agent("What review requirement applies and why?", "P50001")
        self.assertIn("get_payment", result["tools_used"])
        self.assertIn("get_client_profile", result["tools_used"])
        self.assertIn("evaluate_payment_controls", result["tools_used"])
        self.assertIn("search_policy", result["tools_used"])
        self.assertTrue(result["facts"]["high_risk_destination"])
        self.assertNotIn("decoy_operational_1.md", result["citations"])
        self.assertIn("global_payment_policy.md", result["citations"])


if __name__ == "__main__":
    unittest.main()
