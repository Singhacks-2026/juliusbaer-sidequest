"""Offline regression tests. No LLM credentials or network calls are used."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.agent import Investigation, run_agent
from agent.providers import ConfigurationError, Settings, public_api_error, read_settings
from rag.pipeline import build_index, chunk_documents, retrieve
from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_client_payments, get_payment
from tools.policy_tools import _get_index, search_policy
from tools.review_tools import compare_amount, evaluate_payment, parse_policy

POLICIES = ["global_payment_policy.md", "regional_singapore.md", "regional_switzerland.md",
            "high_risk_jurisdictions.md", "investigation_procedure.md"]


class DataAndPolicyTests(unittest.TestCase):
    def test_structuring_excludes_other_clients_and_beneficiaries(self):
        result = aggregate_beneficiary_24h("C2003", "Northstar Trading")
        self.assertEqual(result["matched_payment_count"], 3)
        self.assertEqual(len(result["windows"]), 1)
        window = result["windows"][0]
        self.assertEqual(window["total_amount"], 110000)
        self.assertEqual(set(window["payment_ids"]), {"P50003", "P50180", "P50181"})
        self.assertEqual(window["currency"], "CHF")

    def test_groups_keep_dates_and_currencies_separate(self):
        rows = [dict(payment_id=f"NEW{i}", client_id="NEW", beneficiary_name="Example",
                     payment_date=day, currency=currency, amount=amount, channel="Online")
                for i, (day, currency, amount) in enumerate([
                    ("2026-08-01", "USD", 0.1), ("2026-08-01", "USD", 0.2),
                    ("2026-08-01", "CHF", 50), ("2026-08-02", "USD", 99999)])]
        with patch("tools.payment_tools.read_rows", return_value=rows):
            result = aggregate_beneficiary_24h("NEW", "Example")
            self.assertEqual(len(result["windows"]), 3)
            usd_day1 = next(w for w in result["windows"] if w["currency"] == "USD" and w["payment_date"] == "2026-08-01")
            self.assertEqual(usd_day1["total_amount"], 0.3)
            self.assertEqual(aggregate_beneficiary_24h("NEW", "Example", "2026-08-02")["matched_payment_count"], 1)

    def test_unknown_ids_and_cached_records_are_safe(self):
        self.assertEqual(get_payment("missing"), {})
        self.assertEqual(get_client_profile("missing"), {})
        self.assertEqual(get_client_payments("missing"), [])
        payment = get_payment("P50001")
        payment["amount"] = 0
        self.assertEqual(get_payment("P50001")["amount"], 125000)

    def test_invalid_date_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_beneficiary_24h("C2003", "Northstar Trading", "2026-02-31")

    def test_comparison_boundaries_are_strict(self):
        rule = {"threshold": 100000, "threshold_currency": "USD"}
        self.assertFalse(compare_amount(100000, "USD", rule)["triggered"])
        self.assertTrue(compare_amount(100000.01, "USD", rule)["triggered"])
        self.assertIsNone(compare_amount(1, "USD", rule)["currency_assumption"])
        self.assertIn("1:1", compare_amount(1, "CHF", rule)["currency_assumption"])

    def test_thresholds_are_parsed_from_content(self):
        policy = {"source": "renamed.md", "text": "# Global Payment Monitoring Policy\n\n- Payments above USD 123,456 equivalent require enhanced review before release."}
        rule = parse_policy(policy, "UK")["rules"][0]
        self.assertEqual(rule["threshold"], 123456)
        self.assertFalse(compare_amount(123456, "USD", rule)["triggered"])

    def test_country_code_and_client_region_are_authoritative(self):
        result = evaluate_payment("P50002", POLICIES)
        self.assertEqual(result["payment"]["beneficiary_country"], "Hong Kong")
        self.assertTrue(result["destination_risk"]["high_risk"])
        triggered = [check for check in result["threshold_checks"] if check["triggered"]]
        self.assertEqual([check["action"] for check in triggered], ["RM review"])
        self.assertFalse(any(check["source"] == "regional_switzerland.md" for check in result["threshold_checks"]))
        self.assertFalse(evaluate_payment("P50000", POLICIES)["destination_risk"]["high_risk"])

    def test_global_policy_remains_alongside_swiss_policy(self):
        payment = {**get_payment("P50004"), "amount": 110000, "payment_id": "NEW"}
        with patch("tools.review_tools.get_payment", return_value=payment):
            checks = evaluate_payment("NEW", POLICIES)["threshold_checks"]
        self.assertEqual({(c["source"], c["threshold"]): c["triggered"] for c in checks}, {
            ("global_payment_policy.md", 100000): True,
            ("regional_switzerland.md", 80000): True,
            ("regional_switzerland.md", 120000): False,
        })

    def test_missing_evidence_is_not_a_negative_risk_result(self):
        result = evaluate_payment("P50002", ["global_payment_policy.md"])
        self.assertIsNone(result["destination_risk"]["high_risk"])
        self.assertTrue(result["missing_policy_evidence"])
        self.assertIsNone(result["potential_structuring"])

    def test_structuring_is_supported_without_asserting_intent(self):
        result = evaluate_payment("P50003", POLICIES, check_structuring=True)
        self.assertTrue(result["potential_structuring"])
        self.assertTrue(result["escalate_potential_structuring_to_compliance"])
        self.assertEqual(len(result["structuring_checks"]), 1)
        self.assertEqual(result["structuring_checks"][0]["comparison_amount"], 110000)
        self.assertTrue(any("1:1" in a for a in result["assumptions"]))
        self.assertTrue(any("calendar date" in a for a in result["assumptions"]))

    def test_mixed_currency_pattern_discloses_equivalence(self):
        history = [{"beneficiary_name": "Example"}]
        def window(currency, amount, identifier):
            row = {"payment_id": identifier, "amount": amount, "channel": "Online"}
            return {"currency": currency, "total_amount": amount, "count": 1,
                    "payment_date": "2026-08-01", "payments": [row]}
        aggregate = {"assumptions": [], "windows": [window("USD", 60000, "NEW1"), window("CHF", 60000, "NEW2")]}
        with patch("tools.review_tools.get_client_payments", return_value=history), patch(
                "tools.review_tools.aggregate_beneficiary_24h", return_value=aggregate):
            result = evaluate_payment("P50003", POLICIES, True)
        check = result["structuring_checks"][0]
        self.assertEqual(check["totals_by_currency"], {"USD": 60000, "CHF": 60000})
        self.assertTrue(check["triggered"])
        self.assertIn("1:1", check["currency_assumption"])


class RetrievalTests(unittest.TestCase):
    def test_targets_rank_and_decoys_are_excluded(self):
        for query, expected in [
            ("Singapore RM review threshold", "regional_singapore.md"),
            ("Switzerland RM enhanced review", "regional_switzerland.md"),
            ("high-risk jurisdiction AE", "high_risk_jurisdictions.md"),
            ("investigation workflow procedure steps", "investigation_procedure.md"),
            ("multiple payments beneficiary transaction splitting", "global_payment_policy.md"),
        ]:
            with self.subTest(query=query):
                hits = search_policy(query, top_k=2)
                self.assertIn(expected, [hit["source"] for hit in hits])
                self.assertFalse(any("decoy" in hit["source"] for hit in hits))
        self.assertFalse(any("decoy" in hit["source"] for hit in search_policy("payment monitoring thresholds", 9)))
        self.assertEqual(search_policy("xylophone quasar"), [])

    def test_index_reuse(self):
        _get_index.cache_clear()
        search_policy("global payment")
        search_policy("Singapore procedure")
        self.assertEqual(_get_index.cache_info().misses, 1)
        self.assertEqual(_get_index.cache_info().hits, 1)

    def test_chunks_keep_multiline_rules_and_sources(self):
        text = "# Example\n\n- Multiple payments from one client\n  to one beneficiary should be reviewed.\n- Another complete rule."
        chunks = chunk_documents([{"source": "example.md", "text": text}], chunk_size=60, chunk_overlap=5)
        self.assertTrue(any("from one client\nto one beneficiary" in c["text"] for c in chunks))
        self.assertTrue(all(c["source"] == "example.md" for c in chunks))
        self.assertEqual(retrieve(build_index([]), "payment"), [])


class AgentTests(unittest.TestCase):
    @staticmethod
    def prepared():
        investigation = Investigation("Review this payment.", "P50002")
        for name, args in [
            ("get_payment", {"payment_id": "P50002"}),
            ("get_client_profile", {"client_id": "C2002"}),
            ("search_policy", {"query": "global Singapore payment high-risk jurisdiction", "top_k": 9}),
        ]:
            investigation.execute({"name": name, "arguments": args})
        investigation.execute({"name": "evaluate_payment", "arguments": {
            "payment_id": "P50002", "policy_sources": list(investigation.retrieved), "check_structuring": False}})
        return investigation

    def test_invalid_calls_and_unretrieved_sources_are_rejected(self):
        investigation = Investigation("Review", "P50001")
        for call in [
            {"name": "unknown", "arguments": {}},
            {"name": "get_payment", "arguments": {"payment_id": 42}},
            {"name": "get_payment", "arguments": "not JSON"},
            {"name": "evaluate_payment", "arguments": {"payment_id": "P50001", "policy_sources": POLICIES, "check_structuring": False}},
        ]:
            self.assertIn("error", investigation.execute(call))
        self.assertFalse(any(call["invoked"] for call in investigation.trace))

    def test_unknown_citation_is_rejected(self):
        investigation = self.prepared()
        with self.assertRaisesRegex(ValueError, "citation"):
            investigation.finalize(json.dumps({"answer": "Review needed", "citations": ["invented.md"]}))

    def test_facts_and_tool_usage_come_from_execution(self):
        investigation = self.prepared()
        result = investigation.finalize(json.dumps({
            "answer": "RM and destination review are required.",
            "citations": ["regional_singapore.md", "high_risk_jurisdictions.md"],
            "facts": {"amount": 1}, "tools_used": ["invented_tool"],
        }))
        self.assertEqual(result["facts"]["amount"], 85000)
        self.assertIn("evaluate_payment", result["tools_used"])
        self.assertNotIn("invented_tool", result["tools_used"])

    def test_pattern_answer_requires_history_and_aggregation(self):
        investigation = self.prepared()
        investigation.question = "Is there a structuring pattern?"
        with self.assertRaisesRegex(ValueError, "history"):
            investigation.finalize(json.dumps({"answer": "Unknown", "citations": ["global_payment_policy.md"]}))

    def test_missing_configuration_is_explanatory(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv"):
            with self.assertRaisesRegex(ConfigurationError, "OPENAI_API_KEY"):
                read_settings()

    def test_provider_error_does_not_echo_secrets(self):
        self.assertNotIn("secret-key", public_api_error(RuntimeError("secret-key")))

    def test_agent_can_recover_from_a_premature_answer(self):
        class ScriptedConversation:
            def __init__(self, *args):
                self.step = 0
                self.corrections = []
            def request(self, schemas):
                self.step += 1
                if self.step == 1:
                    return '{"answer":"Premature","citations":["global_payment_policy.md"]}', []
                if self.step == 2:
                    return "", [
                        {"id":"1", "name":"get_payment", "arguments":{"payment_id":"P50002"}},
                        {"id":"2", "name":"get_client_profile", "arguments":{"client_id":"C2002"}},
                        {"id":"3", "name":"search_policy", "arguments":{"query":"global Singapore high-risk jurisdiction payment", "top_k":9}},
                    ]
                if self.step == 3:
                    return "", [{"id":"4", "name":"evaluate_payment", "arguments":{
                        "payment_id":"P50002", "policy_sources":["global_payment_policy.md", "regional_singapore.md", "high_risk_jurisdictions.md"], "check_structuring":False}}]
                return '{"answer":"RM and additional destination review required.","citations":["regional_singapore.md","high_risk_jurisdictions.md"]}', []
            def add_results(self, results):
                pass
            def correct(self, message):
                self.corrections.append(message)
        with patch("agent.agent.read_settings", return_value=Settings("openai", "test", "chat_completions")), patch(
                "agent.agent.make_client", return_value=object()), patch("agent.agent.Conversation", ScriptedConversation), patch.dict(
                os.environ, {"LLM_TRACE_DIR": ""}):
            result = run_agent("Review this payment", "P50002")
        self.assertEqual(result["facts"]["amount"], 85000)
        self.assertEqual(set(result["tools_used"]), {"get_payment", "get_client_profile", "search_policy", "evaluate_payment"})


if __name__ == "__main__":
    unittest.main()
