import os
import unittest
from unittest.mock import patch

from agent.agent import run_agent
from rag.pipeline import build_index, chunk_documents, load_policy_documents, retrieve
from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, assess_payment_review, get_payment


class DataToolTests(unittest.TestCase):
    def test_lookup_types_and_unknowns(self):
        self.assertEqual(get_payment("p50001")["amount"], 125000.0)
        self.assertEqual(get_client_profile("c2003")["relationship_years"], 12.1)
        self.assertEqual(get_payment("missing"), {})
        self.assertEqual(get_client_profile("missing"), {})

    def test_aggregate_filters_client_beneficiary_date_and_currency(self):
        result = aggregate_beneficiary_24h("C2003", "Northstar Trading")
        window = result["largest_window"]
        self.assertEqual(window["payment_ids"], ["P50003", "P50180", "P50181"])
        self.assertEqual(window["count"], 3)
        self.assertEqual(window["total_amount"], 110000.0)
        self.assertNotIn("P50182", window["payment_ids"])  # different client
        self.assertNotIn("P50183", window["payment_ids"])  # different beneficiary
        self.assertTrue(result["potential_structuring_trigger"])

    def test_review_assessment_is_deterministic(self):
        result = assess_payment_review("P50002", "Singapore")
        self.assertEqual(result["triggered_reviews"], [
            "regional_rm_review", "additional_high_risk_destination_review"
        ])
        self.assertFalse(result["thresholds"]["global_enhanced_review"]["triggered"])


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = load_policy_documents(os.path.join(os.path.dirname(__file__), "..", "data", "policies"))
        cls.index = build_index(chunk_documents(docs))

    def test_discriminating_sources(self):
        self.assertEqual(retrieve(self.index, "AE high-risk destination additional review", 1)[0]["source"], "high_risk_jurisdictions.md")
        self.assertEqual(retrieve(self.index, "Switzerland CHF RM review threshold", 1)[0]["source"], "regional_switzerland.md")
        self.assertEqual(retrieve(self.index, "investigation workflow steps", 1)[0]["source"], "investigation_procedure.md")

    def test_decoys_are_never_returned(self):
        for result in retrieve(self.index, "administrative payment monitoring thresholds", 20):
            self.assertFalse(result["source"].startswith("decoy_"))


class FallbackTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_structuring_answer_is_evidence_driven(self):
        result = run_agent("Does this client show transaction splitting?", "P50003")
        self.assertEqual(
            result["facts"]["beneficiary_24h_analysis"]["total_amount"],
            110000.0,
        )
        self.assertIn("aggregate_beneficiary_24h", result["tools_used"])
        self.assertIn("regional_switzerland.md", result["citations"])
        self.assertFalse(any(source.startswith("decoy_") for source in result["citations"]))

    @patch.dict(os.environ, {}, clear=True)
    def test_all_official_questions_have_the_required_shape(self):
        import json
        from pathlib import Path

        questions = json.loads((Path(__file__).parents[1] / "questions" / "questions.json").read_text())
        required = {"answer", "citations", "facts", "tools_used"}
        for question in questions:
            result = run_agent(question["question"], question["payment_id"])
            self.assertEqual(set(result), required)
            self.assertTrue(result["answer"])
            self.assertTrue(result["citations"])
            self.assertTrue(result["facts"])
            self.assertIn("assess_payment_review", result["tools_used"])


if __name__ == "__main__":
    unittest.main()
