"""Test official runner I/O with simulated LLM decisions, never a fake submission."""

import json
import os
import re
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.providers import Settings
from validate_submission import validate

BASE = Path(__file__).resolve().parents[1]


class FixtureConversation:
    """Network-free provider fixture. All tools and the runner remain real."""
    def __init__(self, client, settings, system, prompt):
        self.input = json.loads(prompt)
        self.step = 0
        self.payment, self.client, self.sources = None, None, []
        self.pattern = bool(re.search(r"structur|splitt|pattern", self.input["question"], re.I))

    def request(self, schemas):
        self.step += 1
        def call(name, arguments):
            return {"id": name, "name": name, "arguments": arguments}
        if self.step == 1:
            return "", [call("get_payment", {"payment_id": self.input["payment_id"]})]
        if self.step == 2:
            return "", [call("get_client_profile", {"client_id": self.payment["client_id"]})]
        if self.step == 3:
            calls = [call("search_policy", {"query": "global payment " + self.client["country"] +
                     " high-risk jurisdiction investigation procedure", "top_k": 9})]
            if self.pattern:
                calls.extend([
                    call("get_client_payments", {"client_id": self.client["client_id"]}),
                    call("aggregate_beneficiary_24h", {"client_id": self.client["client_id"],
                         "beneficiary_name": self.payment["beneficiary_name"], "payment_date": None}),
                ])
            return "", calls
        if self.step == 4:
            return "", [call("evaluate_payment", {"payment_id": self.input["payment_id"],
                             "policy_sources": self.sources, "check_structuring": self.pattern})]
        return json.dumps({"answer": "Simulated provider response for testing runner I/O only.",
                           "citations": ["global_payment_policy.md"]}), []

    def add_results(self, results):
        for result in results:
            if result["id"] == "get_payment":
                self.payment = result["result"]
            elif result["id"] == "get_client_profile":
                self.client = result["result"]
            elif result["id"] == "search_policy":
                self.sources = list(dict.fromkeys(p["source"] for p in result["result"]))

    def correct(self, message):
        raise AssertionError(message)


class EntryPointTests(unittest.TestCase):
    def test_all_official_questions_and_trace_validation(self):
        questions = json.loads((BASE / "questions/questions.json").read_text())
        with tempfile.TemporaryDirectory(prefix="sidequest-test-") as folder:
            destination = Path(folder) / "test-output.json"
            trace_dir = Path(folder) / "traces"
            with patch("agent.agent.read_settings", return_value=Settings("openai", "test", "chat_completions")), patch(
                    "agent.agent.make_client", return_value=object()), patch(
                    "agent.agent.Conversation", FixtureConversation), patch.dict(
                    os.environ, {"LLM_TRACE_DIR": str(trace_dir)}), patch.object(
                    sys, "argv", ["main.py", "--questions", str(BASE / "questions/questions.json"), "--output", str(destination)]):
                runpy.run_path(str(BASE / "main.py"), run_name="__main__")
            answers = json.loads(destination.read_text())
            self.assertEqual(validate(answers, questions, trace_dir), [])
            by_id = {result["question_id"]: result for result in answers}
            for qid in ("Q04", "Q07"):
                summary = by_id[qid]["facts"]["structuring_summary"]
                self.assertEqual(summary["count"], 3)
                self.assertEqual(summary["comparison_amount"], 110000)
                self.assertEqual(set(summary["payment_ids"]), {"P50003", "P50180", "P50181"})
            self.assertFalse(by_id["Q01"]["facts"]["destination_risk"]["high_risk"])
            self.assertTrue(by_id["Q03"]["facts"]["destination_risk"]["high_risk"])
            self.assertTrue(by_id["Q09"]["facts"]["destination_risk"]["high_risk"])
            self.assertEqual(by_id["Q08"]["facts"]["client_country"], "Switzerland")
            self.assertFalse(any(c["triggered"] for c in by_id["Q08"]["facts"]["threshold_checks"]))
            answers[0]["facts"]["amount"] = 999999
            self.assertTrue(any("amount" in error for error in validate(answers, questions, trace_dir)))
            answers.append(answers[0])
            self.assertTrue(any("duplicate" in error for error in validate(answers, questions)))


if __name__ == "__main__":
    unittest.main()
