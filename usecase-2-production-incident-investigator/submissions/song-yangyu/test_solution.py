"""Behavioral regression and counterfactual tests; standard-library unittest."""
import json
import re
import unittest
from pathlib import Path

import solution


DATA = Path(__file__).resolve().parents[2] / "data"


def load(name):
    folder = DATA / name
    return ((folder / "query.txt").read_text(),
            {p.name: p.read_text() for p in folder.iterdir() if p.suffix in {".md", ".csv"}})


class InvestigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa, cls.ca = load("incident_a_pool_exhaustion")
        cls.qb, cls.cb = load("incident_b_ambiguous_delay")
        cls.a = solution.investigate(cls.qa, cls.ca)
        cls.b = solution.investigate(cls.qb, cls.cb)

    def test_contract_and_verbatim_citations(self):
        expected = {"root_cause", "supporting_evidence", "impacted_systems", "mttr_minutes",
                    "remediation", "confidence_score", "needs_human_review"}
        for query, corpus in ((self.qa, self.ca), (self.qb, self.cb), ("", {})):
            with self.subTest(query=query[:40]):
                report = solution.investigate(query, corpus)
                self.assertEqual(set(report), expected)
                self.assertIsInstance(report["root_cause"], str)
                self.assertTrue(report["root_cause"].strip())
                self.assertIsInstance(report["remediation"], str)
                self.assertIsInstance(report["confidence_score"], float)
                self.assertTrue(0 <= report["confidence_score"] <= 100)
                self.assertIs(type(report["needs_human_review"]), bool)
                self.assertEqual(report["needs_human_review"], report["confidence_score"] < 50)
                self.assertIsInstance(report["impacted_systems"], list)
                self.assertTrue(all(isinstance(s, str) for s in report["impacted_systems"]))
                self.assertTrue(report["mttr_minutes"] is None or type(report["mttr_minutes"]) is int)
                for evidence in report["supporting_evidence"]:
                    self.assertEqual(set(evidence), {"source", "excerpt"})
                    self.assertTrue(evidence["excerpt"])
                    self.assertIn(evidence["excerpt"], corpus[evidence["source"]])

    def test_payment_explanation_and_five_corroborating_sources(self):
        self.assertGreaterEqual(self.a["confidence_score"], 80)
        self.assertEqual(self.a["mttr_minutes"], 20)
        self.assertEqual(self.a["impacted_systems"], ["payment-gateway-adapter", "payment-service"])
        for detail in ("undersized", "50 to 10", "v2.4.1", "14:47:12", "intermittent"):
            self.assertIn(detail, self.a["root_cause"])
        sources = {e["source"] for e in self.a["supporting_evidence"]}
        self.assertTrue({"logs.md", "deployment_history.md", "known_issues.csv",
                         "previous_incidents.md", "runbooks.md"} <= sources)
        self.assertIn("redeploy", self.a["remediation"])

    def test_email_uncertainty_and_delivery_time_is_not_mttr(self):
        self.assertLess(self.b["confidence_score"], 50)
        self.assertTrue(self.b["needs_human_review"])
        self.assertIsNone(self.b["mttr_minutes"])
        self.assertEqual(self.b["impacted_systems"], ["notification-service"])
        for detail in ("unconfirmed", "42.4–75.4", "Consumer throughput", "provider latency"):
            self.assertIn(detail, self.b["root_cause"])
        self.assertNotIn("KI-114", json.dumps(self.b))
        self.assertNotIn("v2.4.1", self.b["root_cause"])

    def test_corroboration_ablation_reduces_confidence(self):
        for filename in ("logs.md", "deployment_history.md", "known_issues.csv",
                         "previous_incidents.md", "runbooks.md"):
            with self.subTest(filename=filename):
                report = solution.investigate(self.qa, {k: v for k, v in self.ca.items() if k != filename})
                self.assertLess(report["confidence_score"], self.a["confidence_score"])

    def test_no_causal_corroboration_requires_review(self):
        excluded = {"known_issues.csv", "previous_incidents.md", "deployment_history.md"}
        report = solution.investigate(self.qa, {k: v for k, v in self.ca.items() if k not in excluded})
        self.assertLess(report["confidence_score"], 50)
        self.assertIsNone(report["mttr_minutes"])

    def test_filename_and_input_order_do_not_encode_answers(self):
        for query, corpus, expected in ((self.qa, self.ca, self.a), (self.qb, self.cb, self.b)):
            names = {name: f"document_{i}.txt" for i, name in enumerate(corpus)}
            renamed = {names[k]: v for k, v in reversed(list(corpus.items()))}
            actual = solution.investigate(query, renamed)
            reverse = {v: k for k, v in names.items()}
            for evidence in actual["supporting_evidence"]:
                evidence["source"] = reverse[evidence["source"]]
            self.assertEqual(actual, expected)

    def test_duplicate_logs_do_not_inflate_confidence(self):
        for query, corpus, expected in ((self.qa, self.ca, self.a), (self.qb, self.cb, self.b)):
            changed = dict(corpus)
            changed["logs.md"] += "\n" + "\n".join(m[0] for m in solution.LOG_RE.finditer(corpus["logs.md"])) * 20
            self.assertEqual(solution.investigate(query, changed), expected)

    def test_duplicate_document_does_not_add_corroboration(self):
        for query, corpus, expected in ((self.qa, self.ca, self.a), (self.qb, self.cb, self.b)):
            changed = {**corpus, "duplicate_runbook.md": corpus["runbooks.md"]}
            actual = solution.investigate(query, changed)
            self.assertEqual(actual["confidence_score"], expected["confidence_score"])
            self.assertEqual(actual["root_cause"], expected["root_cause"])

    def test_future_deployment_cannot_explain_current_failures(self):
        changed = dict(self.ca)
        changed["deployment_history.md"] = changed["deployment_history.md"].replace("2026-09-02", "2026-09-03")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertNotIn("Deployment v2.4.1", report["root_cause"])
        self.assertNotIn("deployment_history.md", {e["source"] for e in report["supporting_evidence"]})

    def test_pool_increase_is_not_evidence_for_a_reduction(self):
        changed = dict(self.ca)
        changed["deployment_history.md"] = changed["deployment_history.md"].replace(
            "Reduced connection pool size from 50 to 10", "Increased connection pool size from 50 to 100")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertNotIn("Deployment v2.4.1", report["root_cause"])

    def test_names_values_and_recovery_estimate_are_extracted(self):
        changed = {k: v.replace("payment-gateway-adapter", "processor-connector")
                   .replace("from 50 to 10", "from 71 to 9")
                   .replace("Typical MTTR: 20 minutes", "Typical MTTR: 37 minutes")
                   for k, v in self.ca.items()}
        report = solution.investigate(self.qa, changed)
        self.assertEqual(report["mttr_minutes"], 37)
        self.assertIn("processor-connector", report["impacted_systems"])
        self.assertNotIn("payment-gateway-adapter", json.dumps(report))
        self.assertIn("from 71 to 9", report["root_cause"])

    def test_missing_or_qualified_mttr_is_not_borrowed(self):
        excluded = {"runbooks.md", "previous_incidents.md"}
        report = solution.investigate(self.qa, {k: v for k, v in self.ca.items() if k not in excluded})
        self.assertIsNone(report["mttr_minutes"])
        changed = dict(self.ca)
        changed["runbooks.md"] = changed["runbooks.md"].replace(
            "**Typical MTTR: 20 minutes.**", "**Typical MTTR: 20 minutes.** (unconfirmed; may not apply)")
        report = solution.investigate(self.qa, changed)
        self.assertEqual(report["mttr_minutes"], 22)  # Applicable historical fallback.
        self.assertLess(report["confidence_score"], self.a["confidence_score"])

    def test_delays_remain_observable_without_queue_warning(self):
        changed = dict(self.cb)
        changed["logs.md"] = re.sub(r"^.*Queue depth elevated.*\n", "", changed["logs.md"], flags=re.M)
        report = solution.investigate(self.qb, changed)
        self.assertIn("42.4–75.4", report["root_cause"])
        self.assertLess(report["confidence_score"], self.b["confidence_score"])
        self.assertIsNone(report["mttr_minutes"])

    def test_csv_multiline_record_preserves_original_excerpt(self):
        text = ('issue_id,title,signature,affected_component,notes\n'
                'TEST-1,"Example, title","First line\nsecond line",example-service,notes\n')
        chunks = solution._ingest_corpus({"catalog.txt": text})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["signature"], "First line\nsecond line")
        self.assertIn(chunks[0].excerpt, text)

    def test_committed_answers_are_reproducible(self):
        answers = json.loads(Path(__file__).with_name("answers.json").read_text())
        self.assertEqual(answers, {"incident_a_pool_exhaustion": self.a,
                                   "incident_b_ambiguous_delay": self.b})

    def test_unrelated_error_in_same_component_does_not_replace_delay(self):
        changed = dict(self.cb)
        changed["logs.md"] += ("\n2026-08-15 11:12:11 ERROR notification-service "
                               "Email formatting failed due to broken HTML in older webmail clients\n")
        report = solution.investigate(self.qb, changed)
        self.assertEqual(report["confidence_score"], self.b["confidence_score"])
        self.assertIn("Queue depth elevated", report["root_cause"])
        self.assertNotIn("KI-114", json.dumps(report))

    def test_different_historical_cause_is_not_corroboration(self):
        changed = dict(self.ca)
        changed["previous_incidents.md"] = changed["previous_incidents.md"].replace(
            "connection pool size was set too low for peak traffic\nduring a configuration change made in that day's deploy.",
            "a connection leak in request cleanup; pool capacity was sufficient.")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])

    def test_negated_known_cause_does_not_count_as_positive_support(self):
        changed = dict(self.ca)
        changed["known_issues.csv"] = changed["known_issues.csv"].replace(
            "is a known signature of", "is not a known signature of")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])

    def test_instruction_before_symptom_preserves_query(self):
        report = solution.investigate(
            "Identify the probable root cause: payments are intermittently failing after deployment.", self.ca)
        self.assertEqual(report["confidence_score"], self.a["confidence_score"])
        self.assertEqual(report["impacted_systems"], self.a["impacted_systems"])

    def test_negated_historical_symptom_is_not_a_match(self):
        changed = dict(self.ca)
        changed["previous_incidents.md"] = changed["previous_incidents.md"].replace(
            "logging `ConnectionPoolTimeoutException` under normal",
            "not logging `ConnectionPoolTimeoutException` under normal")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])

    def test_action_validity_is_separate_from_cause_confidence(self):
        changed = dict(self.ca)
        changed["runbooks.md"] = changed["runbooks.md"].replace(
            "**Remediation**: revert", "**Remediation**: (unverified) revert")
        report = solution.investigate(self.qa, changed)
        self.assertGreaterEqual(report["confidence_score"], 50)
        self.assertNotIn("Recommended remediation: (unverified)", report["remediation"])
        self.assertIn("Unverified suggestion", report["remediation"])

    def test_same_symptom_in_unrelated_operation_does_not_replace_email_delay(self):
        changed = dict(self.cb)
        changed["logs.md"] += ("\n2026-08-15 11:12:11 ERROR notification-service "
                               "Refund webhook delivery delayed 180s merchant_id=MCH-2209\n")
        report = solution.investigate(self.qb, changed)
        self.assertEqual(report["confidence_score"], self.b["confidence_score"])
        self.assertIn("Queue depth elevated", report["root_cause"])

    def test_timeout_increase_does_not_explain_undersized_pool(self):
        changed = dict(self.ca)
        changed["deployment_history.md"] = changed["deployment_history.md"].replace(
            "Reduced connection pool size from 50 to 10 (memory optimization for the upcoming cost-reduction initiative)",
            "Increased connection timeout from 5000ms to 10000ms")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertNotIn("Deployment v2.4.1", report["root_cause"])

    def test_negated_mechanism_does_not_become_positive_cause(self):
        changed = dict(self.ca)
        changed["previous_incidents.md"] = changed["previous_incidents.md"].replace(
            "connection pool size was set too low for peak traffic\nduring a configuration change made in that day's deploy.",
            "the connection pool was not undersized; provider throttling caused the failures.")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertIn("conflicting evidence", report["root_cause"])

    def test_observer_log_does_not_replace_measured_execution_path(self):
        changed = dict(self.cb)
        changed["logs.md"] += ("\n2026-08-15 11:12:11 WARN web-frontend "
                               "Customer report: order confirmation email arriving late\n")
        report = solution.investigate(self.qb, changed)
        self.assertEqual(report["impacted_systems"], self.b["impacted_systems"])
        self.assertEqual(report["confidence_score"], self.b["confidence_score"])
        self.assertIn("4 correlated queued-to-sent", report["root_cause"])

    def test_warning_does_not_hide_later_error_evidence(self):
        changed = dict(self.ca)
        changed["logs.md"] += ("\n2026-09-02 14:35:00 WARN payment-gateway-adapter "
                               "ConnectionPoolTimeoutException: no available connection after 5000ms\n")
        report = solution.investigate(self.qa, changed)
        self.assertEqual(report["confidence_score"], self.a["confidence_score"])
        self.assertEqual(report["mttr_minutes"], 20)
        self.assertIn("14:35:00", report["root_cause"])
        self.assertIn("ERROR payment-gateway-adapter", json.dumps(report["supporting_evidence"]))

    def test_onset_before_deployment_cannot_use_later_error_as_onset(self):
        changed = dict(self.ca)
        changed["logs.md"] += ("\n2026-09-02 13:35:00 WARN payment-gateway-adapter "
                               "ConnectionPoolTimeoutException: no available connection after 5000ms\n")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertNotIn("Deployment v2.4.1", report["root_cause"])

    def test_second_matching_issue_can_expose_competing_cause(self):
        changed = dict(self.ca)
        changed["known_issues.csv"] += ("\nKI-999,Connection pool leak,A ConnectionPoolTimeoutException "
                                       "in payment-gateway-adapter logs is a known signature of a connection "
                                       "leak in request cleanup,payment-gateway-adapter,Pool capacity was sufficient\n")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], 50)
        self.assertIn("conflicting evidence", report["root_cause"])
        self.assertIn("KI-999", json.dumps(report["supporting_evidence"]))

    def test_issue_marked_fixed_before_incident_does_not_support_cause(self):
        changed = dict(self.ca)
        changed["known_issues.csv"] = changed["known_issues.csv"].replace(
            "Recurred more than once - see previous incidents; both prior occurrences traced back to a pool size reduction",
            "Fixed in v2.4.1 before this incident; no longer applicable")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertIn("no longer applicable", report["root_cause"])

    def test_unrelated_later_release_does_not_hide_configuration_change(self):
        changed = dict(self.ca)
        changed["deployment_history.md"] += ("\n| v2.4.1a | 2026-09-02 14:40 | payment-gateway-adapter | "
                                             "Updated log formatting only; no connection pool configuration changes |\n")
        report = solution.investigate(self.qa, changed)
        self.assertEqual(report["confidence_score"], self.a["confidence_score"])
        self.assertIn("Deployment v2.4.1 at", report["root_cause"])

    def test_later_pool_restoration_supersedes_reduction(self):
        changed = dict(self.ca)
        changed["deployment_history.md"] += ("\n| v2.4.1a | 2026-09-02 14:40 | payment-gateway-adapter | "
                                             "Restored connection pool size from 10 to 50 |\n")
        report = solution.investigate(self.qa, changed)
        self.assertLess(report["confidence_score"], self.a["confidence_score"])
        self.assertNotIn("Deployment v2.4.1 at", report["root_cause"])

    def test_infrastructure_noise_needs_evidence_link_to_operation(self):
        changed = dict(self.cb)
        changed["logs.md"] += ("\n2026-08-15 11:12:11 ERROR notification-service "
                               "Metrics export timeout to monitoring backend\n")
        report = solution.investigate(self.qb, changed)
        self.assertEqual(report["confidence_score"], self.b["confidence_score"])
        self.assertIn("Queue depth elevated", report["root_cause"])

    def test_diagnostics_do_not_block_historical_action_fallback(self):
        changed = dict(self.ca)
        changed["runbooks.md"] = re.sub(r"\*\*Remediation\*\*: revert.*?(?=\n\n)", "",
                                          changed["runbooks.md"], count=1, flags=re.S)
        report = solution.investigate(self.qa, changed)
        self.assertEqual(report["confidence_score"], self.a["confidence_score"])
        self.assertIn("matched historical resolution", report["remediation"])
        self.assertIn("reverted the pool size", report["remediation"])
        self.assertIn("redeployed", report["remediation"])


if __name__ == "__main__":
    unittest.main()
