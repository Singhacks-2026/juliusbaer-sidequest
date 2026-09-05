"""Check the official output contract and, optionally, its tool-call trace."""

import argparse
import json
from pathlib import Path

from tools.client_tools import get_client_profile
from tools.payment_tools import get_payment
from tools.policy_tools import get_policy_document

ROOT = Path(__file__).resolve().parent


def validate(results: list, questions: list, traces: list | None = None) -> list[str]:
    errors = []
    if not isinstance(results, list) or len(results) != len(questions):
        return [f'Expected exactly {len(questions)} result objects']
    trace_by_question = {(t['question'], t['payment_id']): t for t in traces or []}
    for result, question in zip(results, questions):
        label = question['question_id']
        problems = []
        if not isinstance(result, dict):
            errors.append(f'{label}: result must be an object')
            continue
        for key in ['question_id', 'payment_id']:
            if result.get(key) != question[key]:
                problems.append(f'{key} differs from official question')
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
            if facts.get('client_country') != client['country']:
                problems.append('client_country differs from source CSV')
            assessment = facts.get('policy_assessment', {})
            if not assessment or assessment.get('missing_policy_sources'):
                problems.append('policy assessment is incomplete')
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
                retrieved = {hit['source'] for event in trace['tool_calls']
                             if event['tool'] == 'search_policy' and isinstance(event['result'], list)
                             for hit in event['result']}
                if isinstance(citations, list) and any(c not in retrieved for c in citations):
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
    results = json.loads(args.submission.read_text())
    questions = json.loads(args.questions.read_text())
    traces = [json.loads(line) for line in args.trace.read_text().splitlines() if line] if args.trace else None
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
