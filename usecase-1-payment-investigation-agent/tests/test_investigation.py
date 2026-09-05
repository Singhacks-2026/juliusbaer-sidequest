"""Offline regression and protocol tests; these do not evaluate LLM answer quality."""
import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.agent import run_agent
from rag.pipeline import build_index, chunk_documents, retrieve
from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_payment
from tools.policy_tools import get_policy_document, search_policy, _get_index
from tools.investigation_tools import assess_payment


class Message:
    def __init__(self, content=None, calls=None):
        self.content = content
        self.tool_calls = [SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(
            name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
            for i, (name, args) in enumerate(calls or [])]

    def model_dump(self, **kwargs):
        output = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            output["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in self.tool_calls]
        return output


def final(citations=None, answer="Grounded test response."):
    return Message(json.dumps({"answer": answer, "citations": citations or ["global_payment_policy.md"]}))


class FakeClient:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self.messages))])


class DataTests(unittest.TestCase):
    def test_unknown_and_country_authority(self):
        self.assertIn("error", get_payment("missing"))
        self.assertIn("error", get_client_profile("missing"))
        r = assess_payment("P50002")
        self.assertEqual(r["payment"]["beneficiary_country"], "Hong Kong")
        self.assertTrue(r["high_risk_destination"])

    def test_supplied_thresholds(self):
        expected = {"P50000": [False], "P50001": [True, True, True],
                    "P50002": [False, True, False], "P50004": [False, False, False]}
        for payment_id, flags in expected.items():
            with self.subTest(payment=payment_id):
                self.assertEqual([r["triggered"] for r in assess_payment(payment_id)["threshold_checks"]], flags)
        swiss = assess_payment("P50004")
        self.assertEqual([r["threshold"] for r in swiss["threshold_checks"]], [100000, 80000, 120000])
        self.assertEqual([r["fx_assumed_1_to_1"] for r in swiss["threshold_checks"]], [True, False, False])

    def test_strict_threshold_boundary(self):
        payment = get_payment("P50001")
        for amount, flags in [(75000, [False, False, False]), (100000, [False, True, False]),
                              (100000.01, [True, True, True])]:
            with patch("tools.investigation_tools.get_payment", return_value={**payment, "amount": amount}):
                self.assertEqual([r["triggered"] for r in assess_payment("P50001")["threshold_checks"]], flags)

    def test_structuring_filters_both_ids(self):
        result = assess_payment("P50003", True)
        windows = [w for a in result["beneficiary_analysis"] for w in a["windows"] if w["potential_structuring"]]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["payment_ids"], ["P50003", "P50180", "P50181"])
        self.assertEqual(windows[0]["total_usd_equivalent"], 110000)
        self.assertEqual(windows[0]["totals_by_currency"], {"CHF": 110000})
        self.assertEqual(windows[0]["count"], 3)

    def test_date_currency_decimal_and_missing_values(self):
        def row(pid, day, amount, currency="USD", client="C", name="B"):
            return dict(payment_id=pid, payment_date=day, amount=amount, currency=currency,
                        client_id=client, beneficiary_name=name)
        rows = [row("1", "2026-01-01", 0.1), row("2", "2026-01-01", 0.2),
                row("3", "2026-01-01", 10, "CHF"), row("4", "2026-01-02", 90000),
                row("5", None, 5), row("6", "2026-01-01", 999999, client="OTHER"),
                row("7", "2026-01-01", 999999, name="OTHER"), row("8", "2026-01-01", None)]
        with patch("tools.payment_tools.read_records", return_value=rows):
            result = aggregate_beneficiary_24h("C", "B")
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(result["windows"][0]["totals_by_currency"], {"USD": 0.3, "CHF": 10})
        self.assertEqual(result["windows"][0]["total_usd_equivalent"], 10.3)
        self.assertEqual(result["windows"][0]["count"], 3)
        self.assertEqual(result["excluded_payment_ids"], ["5", "8"])

    def test_empty_aggregation(self):
        self.assertEqual(aggregate_beneficiary_24h("missing", "missing")["windows"], [])


class RetrievalTests(unittest.TestCase):
    def test_relevance_and_no_decoys(self):
        for query, source in [("Singapore RM review", "regional_singapore.md"),
                              ("Switzerland threshold", "regional_switzerland.md"),
                              ("high-risk AE", "high_risk_jurisdictions.md"),
                              ("investigation workflow", "investigation_procedure.md")]:
            results = search_policy(query, 3)
            self.assertEqual(results[0]["source"], source)
            self.assertFalse(any("decoy" in r["source"] for r in results))
        self.assertEqual(search_policy("zxqvunknown"), [])
        self.assertEqual(search_policy(""), [])
        self.assertIs(_get_index(), _get_index())

    def test_chunk_preserves_long_rule_and_source(self):
        rule = "- Multiple payments " + "same beneficiary " * 40
        chunks = chunk_documents([{"source": "policy.md", "text": "# Heading\n\n" + rule}], 100, 10)
        self.assertTrue(any(rule.strip() in c["text"] for c in chunks))
        self.assertTrue(all(c["source"] == "policy.md" for c in chunks))
        self.assertTrue(retrieve(build_index(chunks), "beneficiary"))
        with self.assertRaises(ValueError):
            chunk_documents([], 20, 20)
        self.assertEqual(retrieve(build_index([]), "policy"), [])

    def test_path_traversal(self):
        self.assertIn("error", get_policy_document("../clients.csv"))
        self.assertIn("error", get_policy_document("/etc/passwd"))
        self.assertIn("error", get_policy_document("absent.md"))
        self.assertIn("text", get_policy_document("global_payment_policy.md"))


class AgentTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4o", "OPENAI_API_MODE": "chat"})
        env.start()
        self.addCleanup(env.stop)

    def test_tool_loop_and_server_owned_facts(self):
        client = FakeClient([Message(calls=[("assess_payment", {"payment_id": "P50003", "include_history": True})]),
                             final()])
        result = run_agent("Investigate splitting", "P50003", client=client)
        self.assertNotIn("error", result)
        self.assertEqual(result["facts"]["amount"], 45000)
        self.assertIn("aggregate_beneficiary_24h", result["tools_used"])
        self.assertIn("search_policy", result["tools_used"])
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertEqual(client.requests[-1]["messages"][3]["role"], "tool")

    def test_repair_hallucinated_citation_and_malformed_json(self):
        client = FakeClient([Message("not json"),
                             Message(calls=[("assess_payment", {"payment_id": "P50001"})]),
                             final(["invented.md"]), final()])
        result = run_agent("Review", "P50001", client=client)
        self.assertNotIn("error", result)
        self.assertNotIn("invented.md", result["citations"])
        self.assertEqual(len(client.requests), 4)

    def test_invalid_tools_and_arguments_recover(self):
        client = FakeClient([Message(calls=[("bad_tool", {}), ("get_payment", "bad json"),
                                           ("get_payment", {"payment_id": 1}),
                                           ("get_policy_document", {"source": "regional_singapore.md"})]),
                             Message(calls=[("assess_payment", {"payment_id": "P50001"})]), final()])
        result = run_agent("Review", "P50001", client=client)
        self.assertNotIn("error", result)
        self.assertEqual(len([t for t in result["tool_trace"] if "error" in t["result"]]), 4)
        self.assertNotIn("bad_tool", result["tools_used"])

    def test_requires_correct_payment_evidence(self):
        client = FakeClient([Message(calls=[("assess_payment", {"payment_id": "P50002"})]), final()])
        result = run_agent("Review", "P50001", client=client, max_rounds=2)
        self.assertEqual(result["error"], "round_limit")
        self.assertEqual(result["facts"], {})

    def test_provider_failure_is_explicit_and_redacted(self):
        with patch("agent.agent._create_client", side_effect=RuntimeError("secret-key")):
            result = run_agent("Review", "P50001")
        self.assertEqual(result["error"], "RuntimeError")
        self.assertEqual(result["tools_used"], [])
        self.assertNotIn("secret-key", json.dumps(result))

    def test_real_sdk_with_mock_http_transport(self):
        try:
            import httpx
            from openai import OpenAI
        except ImportError:
            self.skipTest("Install requirements for SDK transport verification")
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                message = Message(calls=[("assess_payment", {"payment_id": "P50001"})]).model_dump()
            else:
                self.assertEqual(payload["messages"][-1]["role"], "tool")
                self.assertTrue(json.loads(payload["messages"][-1]["content"])["high_risk_destination"])
                message = final().model_dump()
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                                            "model": "test", "choices": [{"index": 0, "message": message,
                                            "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}]})

        with OpenAI(api_key="offline-test", base_url="https://example.invalid/v1",
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            result = run_agent("Review this payment", "P50001", client=client)
        self.assertNotIn("error", result)
        self.assertEqual(len(requests), 2)
        self.assertEqual(result["facts"]["amount"], 125000)

    def test_responses_preserves_reasoning_calls_and_repair(self):
        import httpx
        from openai import OpenAI
        requests = []
        reasoning = {"id": "rs_test", "type": "reasoning", "summary": [], "encrypted_content": "opaque-test"}

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertFalse(payload["store"])
            self.assertEqual(payload["text"]["format"]["type"], "json_object")
            self.assertTrue(all(t["type"] == "function" and "function" not in t for t in payload["tools"]))
            if len(requests) == 1:
                output = [reasoning, {"type": "function_call", "id": "fc_test", "call_id": "call_test",
                                     "name": "assess_payment", "arguments": json.dumps({"payment_id": "P50001"})}]
            else:
                self.assertIn(reasoning, payload["input"])
                tool_results = [i for i in payload["input"] if i.get("type") == "function_call_output"]
                self.assertEqual(len(tool_results), 1)
                self.assertEqual(tool_results[0]["call_id"], "call_test")
                self.assertTrue(json.loads(tool_results[0]["output"])["high_risk_destination"])
                content = "invalid JSON" if len(requests) == 2 else final().content
                output = [{"id": f"msg_{len(requests)}", "type": "message", "role": "assistant",
                           "status": "completed", "content": [{"type": "output_text", "text": content, "annotations": []}]}]
                if len(requests) == 3:
                    self.assertEqual(payload["input"][-1]["role"], "user")
            return httpx.Response(200, json={"id": f"resp_{len(requests)}", "object": "response",
                                            "created_at": 0, "status": "completed", "model": "gpt-5.6-luna", "output": output})

        with patch.dict(os.environ, {"OPENAI_MODEL": "gpt-5.6-luna", "OPENAI_API_MODE": "responses"}), OpenAI(
            api_key="offline-test", base_url="https://example.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(respond))
        ) as client:
            result = run_agent("Review", "P50001", client=client)
        self.assertNotIn("error", result)
        self.assertEqual(len(requests), 3)

    def test_main_all_ten_questions_with_mock_model(self):
        class GenericClient:
            def __init__(self):
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                messages = kwargs["messages"]
                if not any(m["role"] == "tool" for m in messages):
                    payment_id = json.loads(messages[1]["content"])["payment_id"]
                    msg = Message(calls=[("assess_payment", {"payment_id": payment_id, "include_history": True})])
                else:
                    msg = final()
                return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.json"
            with patch("agent.agent._create_client", return_value=GenericClient()), patch.object(sys, "argv", [
                "main.py", "--questions", str(ROOT / "questions/questions.json"), "--output", str(output)
            ]):
                runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
            results = json.loads(output.read_text())
        questions = json.loads((ROOT / "questions/questions.json").read_text())
        self.assertEqual(len(results), 10)
        for expected, result in zip(questions, results):
            self.assertNotIn("error", result)
            for key in ("question_id", "question", "payment_id"):
                self.assertEqual(result[key], expected[key])
            for key in ("answer", "citations", "facts", "tools_used"):
                self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
