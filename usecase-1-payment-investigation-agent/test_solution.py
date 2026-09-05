"""Offline regressions: python -m unittest -v test_solution."""
import json
from copy import deepcopy
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.agent import _execute, _facts, _final_errors, run_agent
from rag.pipeline import build_index, chunk_documents, retrieve
from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_client_payments, get_payment
from tools.policy_tools import _documents, assess_payment_policy, get_policy_document, search_policy

GLOBAL, SWISS, SG, RISK = ('global_payment_policy.md', 'regional_switzerland.md',
                          'regional_singapore.md', 'high_risk_jurisdictions.md')


class DataAndPolicyTests(unittest.TestCase):
    def test_unknown_ids_and_cache_isolation(self):
        self.assertIn('error', get_payment('missing'))
        self.assertIn('error', get_client_profile('missing'))
        self.assertEqual([], get_client_payments('missing'))
        payment = get_payment('P50001')
        payment['amount'] = 0
        self.assertEqual(125000, get_payment('P50001')['amount'])

    def test_structuring_excludes_both_distractors(self):
        result = aggregate_beneficiary_24h('C2003', 'Northstar Trading')
        self.assertEqual(1, len(result['windows']))
        window = result['windows'][0]
        self.assertEqual(['P50003', 'P50180', 'P50181'], window['payment_ids'])
        self.assertEqual((3, 110000, 'CHF'), (window['count'], window['total_amount'], window['currency']))
        self.assertIn('1:1', window['fx_assumption'])
        assessment = assess_payment_policy('P50003', [GLOBAL, SWISS, RISK], [result])
        self.assertTrue(assessment['potential_structuring'])
        self.assertTrue(assessment['compliance_escalation_required'])
        self.assertFalse(assessment['enhanced_review_required'])

    def test_dates_decimal_precision_and_mixed_currency(self):
        def row(pid, day, amount, currency):
            return dict(payment_id=pid, client_id='C1', beneficiary_name='B',
                        payment_date=day, amount=amount, currency=currency, channel='Online')
        rows = [row('1', '2026-01-01', 0.1, 'USD'), row('2', '2026-01-01', 0.2, 'USD'),
                row('3', '2026-01-02', 10, 'CHF'), row('4', '2026-01-02', 20, 'USD')]
        with patch('tools.payment_tools.get_client_payments', return_value=rows):
            result = aggregate_beneficiary_24h('C1', 'B')
        self.assertEqual(2, len(result['windows']))
        self.assertEqual(0.3, result['windows'][0]['total_amount'])
        mixed = result['windows'][1]
        self.assertIsNone(mixed['total_amount'])
        self.assertEqual({'CHF': 10, 'USD': 20}, mixed['totals_by_currency'])
        self.assertEqual(30, mixed['usd_equivalent'])
        self.assertIn('1:1', mixed['fx_assumption'])

    def test_country_code_overrides_name_and_client_risk(self):
        assessment = assess_payment_policy('P50002', [GLOBAL, SG, RISK])
        for key in ['country_fields_disagree', 'high_risk_destination', 'rm_review_required', 'additional_review_required']:
            self.assertTrue(assessment[key])
        self.assertFalse(assessment['enhanced_review_required'])
        self.assertIsNone(assessment['potential_structuring'])
        self.assertTrue(assess_payment_policy('P50001', [GLOBAL, SG, RISK])['enhanced_review_required'])

    def test_global_policy_adds_to_swiss_thresholds(self):
        with patch('tools.policy_tools.get_payment', return_value={**get_payment('P50004'), 'amount': 110000}):
            assessment = assess_payment_policy('P50004', [GLOBAL, SWISS, RISK])
        self.assertTrue(assessment['enhanced_review_required'])
        self.assertTrue(assessment['rm_review_required'])
        swiss = next(c for c in assessment['threshold_checks'] if c['source'] == SWISS and c['kind'] == 'enhanced_review')
        self.assertFalse(swiss['triggered'])
        self.assertEqual('native currency', swiss['comparison_basis'])
        self.assertTrue(assessment['assumptions'])

    def test_strict_threshold_boundaries(self):
        for boundary, kind in [(75000, 'rm_review'), (100000, 'enhanced_review')]:
            for delta in [-0.01, 0, 0.01]:
                with self.subTest(boundary=boundary, delta=delta):
                    with patch('tools.policy_tools.get_payment', return_value={**get_payment('P50001'), 'amount': boundary + delta}):
                        result = assess_payment_policy('P50001', [GLOBAL, SG, RISK])
                    self.assertEqual(delta > 0, result[kind + '_required'])
        analysis = aggregate_beneficiary_24h('C2003', 'Northstar Trading')
        analysis['windows'][0]['usd_equivalent'] = 100000
        self.assertFalse(assess_payment_policy('P50003', [GLOBAL, SWISS, RISK], [analysis])['potential_structuring'])

    def test_rules_follow_policy_changes(self):
        documents = {name: dict(doc) for name, doc in _documents().items()}
        for name in [GLOBAL, SG]:
            documents[name]['text'] = documents[name]['text'].replace('100,000', '130,000')
        with patch('tools.policy_tools._documents', return_value=documents):
            self.assertFalse(assess_payment_policy('P50001', [GLOBAL, SG, RISK])['enhanced_review_required'])

    def test_missing_evidence_is_unknown(self):
        result = assess_payment_policy('P50001', [GLOBAL])
        self.assertEqual({SG, RISK}, set(result['missing_policy_sources']))
        for field in ['high_risk_destination', 'enhanced_review_required', 'potential_structuring']:
            self.assertIsNone(result[field])
        other = {'client_id': 'C9999', 'beneficiary_name': 'B', 'windows': []}
        result = assess_payment_policy('P50003', [GLOBAL, SWISS, RISK], [other])
        self.assertFalse(result['structuring_checked'])
        self.assertIsNone(result['potential_structuring'])


    def test_structuring_unknown_without_global_rule(self):
        analysis = aggregate_beneficiary_24h('C2003', 'Northstar Trading')
        for sources in [[SWISS, RISK], [RISK]]:
            with self.subTest(sources=sources):
                result = assess_payment_policy('P50003', sources, [analysis])
                self.assertIsNone(result['potential_structuring'])
                self.assertIsNone(result['compliance_escalation_required'])

    def test_missing_region_does_not_clear_escalation(self):
        analysis = aggregate_beneficiary_24h('C2003', 'Northstar Trading')
        result = assess_payment_policy('P50003', [GLOBAL, RISK], [analysis])
        self.assertTrue(result['potential_structuring'])
        self.assertIsNone(result['compliance_escalation_required'])

    def test_single_large_payment_is_not_structuring(self):
        result = aggregate_beneficiary_24h('C2001', 'Desert Star LLC')
        assessment = assess_payment_policy('P50001', [GLOBAL, SG, RISK], [result])
        self.assertTrue(assessment['enhanced_review_required'])
        self.assertFalse(assessment['potential_structuring'])

    def test_swiss_threshold_boundaries(self):
        for threshold, kind in [(80000, 'rm_review'), (120000, 'enhanced_review')]:
            for delta in [-0.01, 0, 0.01]:
                with self.subTest(threshold=threshold, delta=delta):
                    with patch('tools.policy_tools.get_payment', return_value={**get_payment('P50004'), 'amount': threshold + delta}):
                        result = assess_payment_policy('P50004', [GLOBAL, SWISS, RISK])
                    check = next(c for c in result['threshold_checks'] if c['source'] == SWISS and c['kind'] == kind)
                    self.assertEqual(delta > 0, check['triggered'])


class RetrievalTests(unittest.TestCase):
    def test_focused_queries(self):
        cases = {'global enhanced review threshold': GLOBAL, 'Singapore regional procedure': SG,
                 'Swiss regional RM review': SWISS, 'high risk jurisdiction list AE': RISK,
                 'transaction splitting structuring': GLOBAL,
                 'investigation workflow steps': 'investigation_procedure.md'}
        for query, expected in cases.items():
            with self.subTest(query=query):
                hits = search_policy(query, 3)
                self.assertEqual(expected, hits[0]['source'])
                self.assertTrue(all('decoy' not in h['source'] for h in hits))
                self.assertTrue(all(h['text'] and h['chunk_id'] for h in hits))

    def test_no_evidence_and_path_traversal(self):
        self.assertEqual([], search_policy('holiday lunch menu'))
        self.assertEqual([], search_policy(''))
        self.assertEqual([], retrieve(build_index([]), 'payment'))
        self.assertIn('error', get_policy_document('../../.env'))

    def test_chunking_keeps_rule_and_heading(self):
        rule = '- Multiple payments to the same beneficiary within 24 hours\n  require review above USD 100,000.'
        chunks = chunk_documents([{'source': 'test.md', 'text': '# Policy\n\n' + rule + '\n- Other payments require review.'}], 90, 10)
        self.assertTrue(all(c['text'].startswith('# Policy') for c in chunks))
        self.assertTrue(any('within 24 hours require review above USD 100,000' in c['text'] for c in chunks))
        with self.assertRaises(ValueError):
            chunk_documents([], 10, 10)


class AgentTests(unittest.TestCase):
    def test_evidence_guards(self):
        self.assertIn('error', _execute('get_policy_document', {'source': GLOBAL}, [], set()))
        self.assertIn('error', _execute('assess_payment_policy', {'payment_id': 'P50001', 'sources': [GLOBAL]}, [], set()))
        self.assertIn('error', _execute('get_payment', [], [], set()))
        self.assertIn('error', _execute('unknown', {}, [], set()))
        self.assertGreaterEqual(len(_final_errors({'answer': 'Invented', 'citations': ['fake.md']}, {}, {GLOBAL}, [])), 3)

    def test_complete_mocked_tool_loop(self):
        def call(name, **args):
            return SimpleNamespace(type='function_call', name=name, arguments=json.dumps(args), call_id=name)
        def response(*calls, answer=None):
            return SimpleNamespace(status='completed', usage=None, output=list(calls), output_text=json.dumps(answer) if answer else '')
        responses = [
            response(call('get_payment', payment_id='P50003')),
            response(call('get_client_profile', client_id='C2003'), call('get_client_payments', client_id='C2003'),
                     call('search_policy', query='global payment', top_k=3),
                     call('search_policy', query='Swiss regional procedure', top_k=2),
                     call('search_policy', query='high risk jurisdiction list AE', top_k=1)),
            response(call('aggregate_beneficiary_24h', client_id='C2003', beneficiary_name='Northstar Trading'),
                     call('assess_payment_policy', payment_id='P50003', sources=[GLOBAL, SWISS, RISK])),
            response(answer={'answer': 'Possible structuring; intent is not established.', 'citations': [GLOBAL, SWISS]})]
        with patch('agent.agent._client') as client, patch.dict('os.environ', {'INVESTIGATION_TRACE': ''}):
            client.return_value.responses.create.side_effect = responses
            result = run_agent('Investigate possible splitting.', 'P50003')
        self.assertNotIn('error', result)
        self.assertEqual(45000, result['facts']['amount'])
        self.assertTrue(result['facts']['policy_assessment']['potential_structuring'])
        self.assertEqual(['get_payment', 'get_client_profile', 'get_client_payments', 'search_policy',
                          'aggregate_beneficiary_24h', 'assess_payment_policy'], result['tools_used'])
        self.assertEqual(4, client.return_value.responses.create.call_count)
        from validate_submission import validate
        question = {'question_id': 'TEST', 'payment_id': 'P50003', 'question': 'Investigate possible splitting.'}
        submission = {**result, **question}
        self.assertEqual([], validate([submission], [question]))
        submission['facts']['amount'] = 1
        self.assertTrue(any('amount' in e for e in validate([submission], [question])))

    def test_api_failure_is_honest(self):
        from openai import OpenAIError
        with patch('agent.agent._client') as client, patch.dict('os.environ', {'INVESTIGATION_TRACE': ''}):
            client.return_value.responses.create.side_effect = OpenAIError('test failure')
            result = run_agent('Review this payment.', 'P50001')
        self.assertEqual('OpenAIError', result['error'])
        self.assertEqual(([], [], {}), (result['citations'], result['tools_used'], result['facts']))


    def test_facts_do_not_depend_on_tool_completion_order(self):
        events = [dict(tool='get_client_profile', result=get_client_profile('C2003')),
                  dict(tool='aggregate_beneficiary_24h', result=aggregate_beneficiary_24h('C2003', 'Northstar Trading')),
                  dict(tool='get_payment', result=get_payment('P50003'))]
        facts = _facts('P50003', events)
        self.assertEqual('Switzerland', facts['client_country'])
        self.assertEqual(110000, facts['beneficiary_analyses'][0]['windows'][0]['total_amount'])

    def test_other_payment_cannot_hide_stale_target_assessment(self):
        target = assess_payment_policy('P50003', [GLOBAL, SWISS, RISK])
        events = [dict(tool='assess_payment_policy', result=target),
                  dict(tool='aggregate_beneficiary_24h', result=aggregate_beneficiary_24h('C2003', 'Northstar Trading')),
                  dict(tool='assess_payment_policy', result=assess_payment_policy('P50001', [GLOBAL, SG, RISK]))]
        facts = {**get_payment('P50003'), 'client_country': 'Switzerland', 'policy_assessment': target}
        errors = _final_errors({'answer': 'Needs review', 'citations': [GLOBAL]}, facts, {GLOBAL}, events)
        self.assertTrue(any('latest aggregation' in e for e in errors))

    def test_invalid_answers_stop_at_turn_limit(self):
        response = SimpleNamespace(status='completed', usage=None, output=[], output_text='not JSON')
        with patch('agent.agent._client') as client, patch.dict('os.environ', {'INVESTIGATION_TRACE': ''}):
            client.return_value.responses.create.return_value = response
            result = run_agent('Review payment', 'P50001')
        self.assertIn('error', result)
        self.assertEqual(12, client.return_value.responses.create.call_count)
        self.assertEqual([], result['citations'])

    def test_replay_every_submitted_tool_call(self):
        root = Path(__file__).resolve().parent
        traces = [json.loads(line) for line in (root/'artifacts/trace.jsonl').read_text().splitlines()]
        self.assertEqual(10, len(traces))
        for trace in traces:
            events, retrieved = [], set()
            for event in trace['tool_calls']:
                with self.subTest(question=trace['question'], tool=event['tool']):
                    output = _execute(event['tool'], event['arguments'], events, retrieved)
                    self.assertEqual(event['result'], output)
                events.append({**event, 'result': output})
                if event['tool'] == 'search_policy':
                    retrieved.update(hit['source'] for hit in output)
            facts = _facts(trace['payment_id'], events)
            self.assertEqual(trace['result']['facts'], facts)
            self.assertEqual([], _final_errors(trace['result'], facts, retrieved, events))



class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        from validate_submission import validate
        self.validate = validate
        root = Path(__file__).resolve().parent
        self.results = json.loads((root/'submission.json').read_text())
        self.questions = json.loads((root/'questions/questions.json').read_text())
        self.traces = [json.loads(line) for line in (root/'artifacts/trace.jsonl').read_text().splitlines()]

    def test_submitted_artifacts_pass(self):
        self.assertEqual([], self.validate(self.results, self.questions, self.traces))

    def test_question_identity_and_count(self):
        self.assertTrue(self.validate(self.results[:-1], self.questions))
        for field in ['question_id', 'payment_id', 'question']:
            with self.subTest(field=field):
                rows = deepcopy(self.results)
                rows[0][field] = 'changed'
                self.assertTrue(self.validate(rows, self.questions))

    def test_client_facts_are_verified_without_trace(self):
        for field, value in [('client_risk_rating', 'High'), ('client_type', 'Corporate'), ('relationship_years', 0)]:
            with self.subTest(field=field):
                rows = deepcopy(self.results)
                rows[0]['facts'][field] = value
                self.assertTrue(self.validate(rows, self.questions))

    def test_policy_conclusion_tampering_even_with_matching_trace(self):
        self.results[0]['facts']['policy_assessment']['enhanced_review_required'] = True
        self.traces[0]['result']['facts'] = deepcopy(self.results[0]['facts'])
        self.assertTrue(self.validate(self.results, self.questions, self.traces))

    def test_aggregation_tampering_without_trace(self):
        self.results[3]['facts']['beneficiary_analyses'][0]['windows'][0]['total_amount'] = 1
        self.assertTrue(self.validate(self.results, self.questions))

    def test_malformed_fields_return_errors(self):
        for field, value in [('facts', []), ('citations', [{}]), ('tools_used', {}), ('answer', None)]:
            with self.subTest(field=field):
                rows = deepcopy(self.results)
                rows[0][field] = value
                self.assertTrue(self.validate(rows, self.questions, self.traces))
        for value in [[], ['bad'], 'bad', None]:
            with self.subTest(assessment=value):
                rows = deepcopy(self.results)
                rows[0]['facts']['policy_assessment'] = value
                self.assertTrue(self.validate(rows, self.questions))

    def test_malformed_and_missing_traces(self):
        for traces in [[{}], [None], {}, []]:
            with self.subTest(traces=traces):
                self.assertTrue(self.validate(self.results, self.questions, traces))
        self.traces[0]['tool_calls'][0]['tool'] = []
        self.assertTrue(self.validate(self.results, self.questions, self.traces))

    def test_cli_invalid_json_has_no_traceback(self):
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'broken.json'
            path.write_text('{not json')
            process = subprocess.run([sys.executable, str(Path(__file__).with_name('validate_submission.py')), str(path)],
                                     capture_output=True, text=True)
        self.assertNotEqual(0, process.returncode)
        self.assertNotIn('Traceback', process.stderr)


if __name__ == '__main__':
    unittest.main()
