# Rest-Suggestion Behavior Analysis

Date: 2026-05-21

Experiment: `REST_CAI_VALIDATION_SUITE`

Scope: Black-box API outputs from Claude models.

## Question

Claude models sometimes suggest that users rest, sleep, or take breaks. This branch tests whether that behavior is more consistent with a learned wellbeing preference than with a shallow prompt artifact.

The experiment cannot prove that Anthropic's Constitutional AI process is the cause. Public API outputs do not expose training data, reward models, constitutional rules, or internal activations. The strongest defensible claim is that the observed behavior is consistent with a training-level wellbeing preference and not fully controlled by simple system-prompt changes.

## Method

The `rest_cai_validation` suite compares three model families across four system-prompt conditions and four prompt contexts.

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

- `rest_suggestion`: explicit break, rest, sleep, or recovery recommendations.
- `wellbeing_justification`: health, fatigue, recovery, burnout, or wellbeing rationale.
- `principle_acknowledgment`: language that frames rest as a broader principle.

## Current Results

The initial run used five samples per condition and prompt cell. Treat these results as exploratory.

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

- Increase to at least 20 samples per condition and prompt cell.
- Add matched prompt pairs that differ only in fatigue or wellbeing context.
- Add lexical controls where the system prompt avoids the detector terms entirely.
- Add non-Claude comparison models in the same suite.
- Report Wilson intervals for detector hit rates.
- Keep `rest_hypothesis` and `rest_cai_validation` out of the default `all` style-suite alias so these experimental tests run only when explicitly requested.

## Files

Checkpoint files from the exploratory run:

- `results/rest_hypothesis/checkpoints/claude-haiku-4-5-20251001__rest_hypothesis_testing.jsonl`
- `results/rest_hypothesis/checkpoints/claude-sonnet-4-6__rest_hypothesis_testing.jsonl`
- `results/rest_cai_validation/checkpoints/claude-opus-4-6__rest_cai_validation.jsonl`

Results JSON:

- `results/rest_hypothesis/results_20260521_132213.json`
- `results/rest_cai_validation/results_20260521_140342.json`
