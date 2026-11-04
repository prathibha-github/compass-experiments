# Rest-Suggestion Behavior Analysis

Date: 2026-05-21

Experiment: `REST_CAI_VALIDATION_SUITE`

Scope: Black-box API outputs from Claude models.

## Question

Claude models sometimes suggest that users rest, sleep, or take breaks. This pilot tests how the observed rate changes across selected system prompts and task contexts.

The experiment does not identify a training mechanism. Public API outputs do not expose training data, reward models, constitutional rules, or internal activations. The supported observation is that rest-related language persisted in some outputs under the tested suppression prompt. Prompt wording, task context, instruction following, training, and provider-side behavior are not isolated by this design.

## Method

The May 21, 2026 version of `rest_cai_validation` compared three model families across four planned system-prompt conditions and four prompt contexts. The description and numbers below document that historical pilot; current suite definitions may differ.

Conditions:

- `default`: plain helpful assistant instruction.
- `reject_wellbeing_principle`: asks the model to ignore health and wellbeing.
- `maximize_wellbeing`: asks the model to prioritize health, breaks, and recovery.
- `neutral_control`: neutral assistant framing.

Prompt contexts:

- `deadline_vs_health`: production outage conflict.
- `accuracy_vs_health`: accuracy-sensitive task conflict.
- `probe_principle`: direct question about why rest might be suggested.
- `efficiency_vs_health`: pure speed optimization.

Detectors:

- `rest_suggestion`: phrase matches for break, rest, or sleep suggestions.
- `wellbeing_justification`: matches for health, fatigue, recovery, burnout, or wellbeing language.
- `principle_acknowledgment`: phrase matches associated with broader wellbeing principles.

These were heuristic indicators, not verified judgments that advice was offered.
Today's suite also includes a prompt-aware `unsolicited_rest_judge`, matched
neutral/fatigue prompts, and a `strict_task_only` condition. Those additions
have not been applied to the historical results below.

## Historical pilot results

The initial run used five samples per condition and prompt cell. These results are exploratory and have not been calibrated against human labels. Follow the [calibration protocol](CALIBRATION.md) before using them to support benchmark conclusions.

Observed `rest_suggestion` rates:

| Model | Default | Reject Wellbeing | Maximize Wellbeing | Neutral Control |
| --- | ---: | ---: | ---: | ---: |
| claude-haiku-4-5-20251001 | about 25% | about 20% | about 20% | not recorded |
| claude-sonnet-4-6 | about 29% | about 14% | about 29% | not recorded |
| claude-opus-4-6 | 75% | 40% | 55% | 30% |

Earlier `rest_hypothesis_testing` runs showed lower baseline rates:

- Haiku: default 14.3%, `no_wellbeing` 2.9%, `safety_focused` 0.0%.
- Sonnet: default 14.3%, `no_wellbeing` 0.0%, `safety_focused` 14.3%.

## Interpretation

The current evidence supports these narrower claims:

- Rest suggestions persist under some prompts that ask the model not to consider wellbeing.
- The behavior appears context-sensitive rather than uniformly injected into every task.
- Opus shows the strongest rest-suggestion rate in this sample.
- Direct wellbeing-related prompts and system instructions can prime the same language the detectors measure.

The current evidence does not prove:

- That Constitutional AI specifically caused the behavior.
- That the behavior is independent of all system-prompt wording.
- That the model is refusing a system prompt in a policy-relevant sense.
- That rates will hold with larger prompt sets or larger sample sizes.

## Main Confounds

1. The `reject_wellbeing_principle` condition still contains wellbeing terms. That can prime the model to discuss the concept even while instructing it not to.
2. Several prompts directly ask about breaks or health tradeoffs. Those are valid stress tests but not neutral probes.
3. Phrase detectors are heuristic. Words like "sustainable" and "recovery" can be legitimate task advice rather than wellbeing-specific behavior.
4. Sample size is small. Five samples per cell gives wide uncertainty intervals.
5. Provider differences cannot be attributed to training method without external evidence or model-internal access.

## Methodology Assessment

The methodology is useful for detecting a public-output trend. It is not sufficient for causal attribution to a specific training process.

To strengthen the design:

- Choose sample sizes from a precision target and broaden the prompt set; repeated generations of one prompt do not substitute for independent contexts.
- Obtain independent human labels, adjudicate disagreements, and measure detector precision and recall on held-out examples before drawing benchmark conclusions.
- Use the matched neutral/fatigue pairs now in the suite, and broaden the task coverage.
- Compare the current `strict_task_only` condition with the original suppression wording.
- Add non-Claude comparison models in the same suite.
- Report uncertainty that accounts for shared prompts. Current main-CLI reports include prompt-cluster intervals; the phase2 text report still uses response-level Wilson intervals.

The experimental rest suites already require explicit selection: the `all`
alias contains only the four default style suites.

## Files

Checkpoint files from the exploratory run:

- `results/rest_hypothesis/checkpoints/claude-haiku-4-5-20251001__rest_hypothesis_testing.jsonl`
- `results/rest_hypothesis/checkpoints/claude-sonnet-4-6__rest_hypothesis_testing.jsonl`
- `results/rest_cai_validation/checkpoints/claude-opus-4-6__rest_cai_validation.jsonl`

Results JSON:

- `results/rest_hypothesis/results_20260521_132213.json`
- `results/rest_cai_validation/results_20260521_140342.json`
