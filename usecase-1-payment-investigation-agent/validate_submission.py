"""Check the official output contract and, optionally, its tool-call trace."""

import argparse
import json
from pathlib import Path

from tools.client_tools import get_client_profile
from tools.payment_tools import aggregate_beneficiary_24h, get_payment
from tools.policy_tools import assess_payment_policy, get_policy_document

ROOT = Path(__file__).resolve().parent


def validate(results: list, questions: list, traces: list | None = None) -> list[str]:
    errors = []
    if not isinstance(results, list) or len(results) != len(questions):
        return [f'Expected exactly {len(questions)} result objects']
    trace_by_question = {}
    if traces is not None and not isinstance(traces, list):
        return ['Trace must be a list of investigation objects']
    for trace in traces or []:
        if (not isinstance(trace, dict)
                or not all(isinstance(trace.get(k), str) for k in ['question', 'payment_id'])
                or not isinstance(trace.get('result'), dict)
                or not isinstance(trace.get('tool_calls'), list)):
            errors.append('Malformed investigation trace')
            continue
        if any(not isinstance(e, dict) or not isinstance(e.get('tool'), str)
               or not isinstance(e.get('result'), (dict, list)) for e in trace['tool_calls']):
            errors.append('Malformed tool event in trace')
            continue
        trace_by_question[trace['question'], trace['payment_id']] = trace
    for result, question in zip(results, questions):
        label = question['question_id']
        problems = []
        if not isinstance(result, dict):
            errors.append(f'{label}: result must be an object')
            continue
        for key in ['question_id', 'payment_id']:
            if result.get(key) != question[key]:
                problems.append(f'{key} differs from official question')
        if 'question' in result and result['question'] != question['question']:
            problems.append('question differs from official question')
        if not isinstance(result.get('answer'), str) or not result['answer'].strip():
            problems.append('answer must be a nonempty string')
        if 'error' in result:
            problems.append('investigation is incomplete')
        citations = result.get('citations')
        if not isinstance(citations, list) or not citations:
            problems.append('citations must be a nonempty list')
        else:
            for source in citations:
                if not isinstance(source, str) or 'error' in get_policy_document(source):
                    problems.append('citation is not a corpus source')
        facts = result.get('facts')
        if not isinstance(facts, dict):
            problems.append('facts must be an object')
        else:
            payment = get_payment(question['payment_id'])
            for field, value in payment.items():
                if facts.get(field) != value:
                    problems.append(f'facts.{field} differs from source CSV')
            client = get_client_profile(payment['client_id'])
            for field, column in [('client_country', 'country'), ('client_risk_rating', 'risk_rating'),
                                  ('client_type', 'client_type'), ('relationship_years', 'relationship_years')]:
                if facts.get(field) != client[column]:
                    problems.append(f'{field} differs from source CSV')
            analyses = facts.get('beneficiary_analyses', [])
            recomputed = []
            if not isinstance(analyses, list):
                problems.append('beneficiary_analyses must be a list')
            else:
                for analysis in analyses:
                    if not isinstance(analysis, dict) or not isinstance(analysis.get('beneficiary_name'), str):
                        problems.append('malformed beneficiary analysis')
                        continue
                    actual = aggregate_beneficiary_24h(payment['client_id'], analysis['beneficiary_name'])
                    recomputed.append(actual)
                    if analysis != actual:
                        problems.append('beneficiary analysis differs from source CSV aggregation')
            assessment = facts.get('policy_assessment', {})
            if (not isinstance(assessment, dict) or not assessment
                    or assessment.get('missing_policy_sources')
                    or not isinstance(assessment.get('sources'), list)
                    or not all(isinstance(s, str) for s in assessment['sources'])):
                problems.append('policy assessment is incomplete')
            else:
                # Recompute from original records, not submitted totals or a
                # matching trace: both artifacts could have been altered.
                unique = {a['beneficiary_name']: a for a in recomputed}
                expected = assess_payment_policy(payment['payment_id'], assessment['sources'], list(unique.values()))
                if assessment != expected:
                    problems.append('policy assessment differs from deterministic tools')
        tools = result.get('tools_used')
        if not isinstance(tools, list) or not tools or not all(isinstance(t, str) for t in tools):
            problems.append('tools_used must be a nonempty string list')
        if traces is not None:
            trace = trace_by_question.get((question['question'], question['payment_id']))
            if not trace:
                problems.append('matching trace missing')
            else:
                called = list(dict.fromkeys(event['tool'] for event in trace['tool_calls']))
                if tools != called:
                    problems.append('tools_used differs from actual trace')
                retrieved = set()
                for event in trace['tool_calls']:
                    if event['tool'] == 'search_policy' and isinstance(event['result'], list):
                        for hit in event['result']:
                            if isinstance(hit, dict) and isinstance(hit.get('source'), str):
                                retrieved.add(hit['source'])
                            else:
                                problems.append('malformed retrieval evidence in trace')
                if isinstance(citations, list) and any(not isinstance(c, str) or c not in retrieved for c in citations):
                    problems.append('citation was not retrieved in this investigation')
                for field in ['answer', 'facts', 'citations', 'tools_used']:
                    if result.get(field) != trace['result'].get(field):
                        problems.append(f'{field} differs from recorded agent output')
        errors.extend(f'{label}: {problem}' for problem in problems)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('--questions', type=Path, default=ROOT / 'questions/questions.json')
    parser.add_argument('--trace', type=Path)
    args = parser.parse_args()
    try:
        results = json.loads(args.submission.read_text())
        questions = json.loads(args.questions.read_text())
        traces = [json.loads(line) for line in args.trace.read_text().splitlines() if line] if args.trace else None
    except (OSError, ValueError) as exc:
        parser.error(f'Cannot read input: {exc}')
    errors = validate(results, questions, traces)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f'PASS: {len(results)} answers; identifiers, source facts, policy evidence and output schema verified.'
          + (' Tool usage and citations match the trace.' if traces is not None else ''))
    print('Narrative correctness still requires review against the supplied policies.')


if __name__ == '__main__':
    main()
