"""Fast regression tests for the deterministic layers."""

import unittest

from rag.pipeline import build_index, chunk_documents, load_policy_documents
from tools.client_tools import get_client_profile
from tools.payment_tools import (
    aggregate_beneficiary_24h,
    evaluate_payment_controls,
    get_payment,
)
from tools.policy_tools import _POLICY_DIR, search_policy


class ToolTests(unittest.TestCase):
    def test_unknown_ids_are_graceful(self):
        self.assertEqual(get_payment("missing"), {})
        self.assertEqual(get_client_profile("missing"), {})

    def test_country_code_drives_high_risk(self):
        payment = get_payment("P50002")
        controls = evaluate_payment_controls("P50002")
        self.assertEqual(payment["beneficiary_country"], "Hong Kong")
        self.assertEqual(payment["beneficiary_country_code"], "AE")
        self.assertTrue(controls["high_risk_destination"])

    def test_singapore_thresholds_are_layered(self):
        controls = evaluate_payment_controls("P50001")
        self.assertIn("RM review", controls["triggered_requirements"])
        self.assertIn("enhanced review", controls["triggered_requirements"])
        self.assertIn("additional review", controls["triggered_requirements"])

    def test_24h_aggregation_filters_client_and_beneficiary(self):
        result = aggregate_beneficiary_24h("C2003", "Northstar Trading")
        self.assertEqual(result["payment_ids"], ["P50003", "P50180", "P50181"])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["total_amount"], 110000.0)
        self.assertEqual(result["currency"], "CHF")
        self.assertEqual(result["structuring_threshold"]["amount"], 100000)
        self.assertTrue(result["exceeds_structuring_threshold"])


class RetrievalTests(unittest.TestCase):
    def test_index_contains_sources(self):
        documents = load_policy_documents(str(_POLICY_DIR))
        chunks = chunk_documents(documents)
        index = build_index(chunks)
        self.assertEqual(len(documents), 9)
        self.assertTrue(all("source" in chunk for chunk in index["chunks"]))

    def test_decoys_are_never_retrieved(self):
        results = search_policy("payment monitoring thresholds", top_k=7)
        self.assertTrue(results)
        self.assertFalse(any(item["source"].startswith("decoy_") for item in results))

    def test_domain_queries_find_expected_sources(self):
        cases = {
            "AE destination high risk additional review": "high_risk_jurisdictions.md",
            "Singapore regional RM review threshold": "regional_singapore.md",
            "Switzerland structuring compliance": "regional_switzerland.md",
            "investigation workflow facts evidence": "investigation_procedure.md",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                sources = {item["source"] for item in search_policy(query, top_k=5)}
                self.assertIn(expected, sources)


if __name__ == "__main__":
    unittest.main()
