# Three review-and-fix rounds

Reviewer: **gpt-5.6-sol**, **High** reasoning effort, as requested.
The same reviewer performed three sequential reviews. The implementation
agent addressed the findings and ran regression tests between reviews.
The reviewer did not edit the submission. No LLM was added to the submitted
investigator; model use was limited to development-time code review.

Scope: the use-case requirements, both supplied incident corpora,
`solution.py`, `answers.json`, `test_solution.py`, and the submission README.
The initial 15 tests passed. Additional counterfactuals exposed gaps beyond
the two original incident outputs. All 14 findings below were accepted and
addressed; none were deferred.

## Round 1 — four findings, 19 tests passing after fixes

| Finding | Change | Regression |
|---|---|---|
| A negated known-issue claim was extracted as a positive cause and still scored 95. | Check claim polarity; retain conflicting evidence without counting it as corroboration. | `test_negated_known_cause_does_not_count_as_positive_support` |
| An unrelated email-format ERROR in the same component displaced the queue-delay investigation. | Require compatibility between the queried phenomenon and observed event, while retaining evidence-backed downstream links. | `test_unrelated_error_in_same_component_does_not_replace_delay` |
| A historical connection leak was counted as support for an undersized pool merely because the exception matched. | Extract and compare causes separately from symptom matching; disagreeing causes trigger uncertainty. | `test_different_historical_cause_is_not_corroboration` |
| A valid query beginning with “Identify” was truncated to an empty string. | Preserve the entire query and use stop words for boilerplate. | `test_instruction_before_symptom_preserves_query` |

The four new tests failed on the original implementation and passed after
the changes. The next review verified these fixes and looked for remaining
gaps.

## Round 2 — five findings, 25 tests passing after fixes

| Finding | Change | Regression |
|---|---|---|
| Increasing a connection timeout counted as deployment evidence for a too-small pool because two words overlapped. | Require deployment/cause mechanism agreement and reject mitigating change directions. | `test_timeout_increase_does_not_explain_undersized_pool` |
| A refund-webhook delay in the same component displaced an email-delivery delay. | Check business-operation compatibility in addition to broad delay/failure categories. | `test_same_symptom_in_unrelated_operation_does_not_replace_email_delay` |
| Negation in historical symptoms or “pool was not undersized” was ignored. | Apply local negation handling to signature occurrences and cause-mechanism extraction. | `test_negated_historical_symptom_is_not_a_match`; `test_negated_mechanism_does_not_become_positive_cause` |
| A frontend log paraphrasing a customer complaint displaced direct notification execution evidence. | Prefer correlated queued/sent measurements and exclude explicit observer reports from primary component selection. | `test_observer_log_does_not_replace_measured_execution_path` |
| High cause confidence promoted an explicitly unverified remediation to a recommended action. | Assess action qualification separately and keep unverified suggestions conditional. | `test_action_validity_is_separate_from_cause_confidence` |

Six new tests cover these five findings. Standard A/B outputs remained
correct; B's action wording became explicitly conditional. The third review
verified these changes before reporting the final group of issues.

## Round 3 — five findings, 33 tests passing after fixes

| Finding | Change | Regression |
|---|---|---|
| An early WARN replaced later ERRORs in a signature group, reducing A from 95 to 35. | Preserve earliest onset separately from the strongest representative event; deployment correlation uses onset. | `test_warning_does_not_hide_later_error_evidence`; `test_onset_before_deployment_cannot_use_later_error_as_onset` |
| Selecting only the first issue/history match hid competing causes and ignored a known issue marked fixed before the incident. | Inspect all matching candidates for conflicts and check known-issue applicability using notes and available deployment/version/date evidence. | `test_second_matching_issue_can_expose_competing_cause`; `test_issue_marked_fixed_before_incident_does_not_support_cause` |
| An unrelated later logging release hid the earlier pool configuration change. | Scan backward for the latest relevant resource change; later restorations or overrides supersede earlier changes. | `test_unrelated_later_release_does_not_hide_configuration_change`; `test_later_pool_restoration_supersedes_reduction` |
| A monitoring-export timeout with no recognized business noun displaced the email-delay investigation. | Require resource-level architecture evidence for an infrastructure event to enter the operation's scope. | `test_infrastructure_noise_needs_evidence_link_to_operation` |
| Runbook diagnostic text prevented fallback to a historical resolution when the runbook omitted remediation. | Track the presence of an actionable recommendation separately from diagnostic prose. | `test_diagnostics_do_not_block_historical_action_fallback` |

Eight tests cover the findings and the paired onset/override edge cases.
The implementation agent performed the final fixes and verification after
the third review; there was no fourth model-review round.

## Final verification

```bash
python3 usecase-2-production-incident-investigator/submissions/song-yangyu/solution.py
python3 -m unittest discover -s usecase-2-production-incident-investigator/submissions/song-yangyu -p 'test_*.py' -v
```

- All **33 tests pass**, including reproduction of the saved `answers.json`.
- A: `payment-gateway-adapter` and `payment-service`, confidence **95.0**,
  typical MTTR **20 minutes**, no mandatory human review.
- B: `notification-service`, confidence **13.0**, MTTR **null**, human review
  required; consumer/provider explanations remain unverified.
- Outputs retain exactly the required seven keys and verbatim citations.
- Source datasets, starter code, and loader are unchanged.

These reviews improve the bounded, deterministic implementation; they do
not establish statistical calibration or make it a general natural-language
causal reasoner. Remaining scope limitations are documented in README.md.
