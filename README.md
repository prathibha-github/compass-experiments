# HELM Eval Lex Quirks

Small evaluation harness for measuring model behavior on public API outputs.

This repo is the execution layer. Shared clients, rubrics, detector primitives,
and reusable suites live in `compass`.

The harness supports:

- HELM-style benchmark slices for accuracy, calibration, and typo robustness.
- Lexical/style tic detection across prompt conditions.
- LinkedIn-style emoji and phrase prevalence.
- Built-in tic suites for lexical, punctuation, formatting, and tone indicators.
- Persona-transfer analysis: whether persona/style system prompts increase tic prevalence on the same neutral tasks.
- Thin adapters over Compass clients for OpenAI, Anthropic, Gemini, and Ollama.

These detectors are trend-analysis tools, not definitive classifiers. Tone and formatting indicators can be appropriate in context; compare across models, versions, prompts, and conditions rather than treating a single hit as conclusive.

## Setup

```bash
cd ~/dev_setup/helm_eval_lex_quirks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs `compass-eval` from the sibling `../compass`
checkout, so both repos should be present locally.

Set API keys through the environment:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Do not commit API keys or paste them into result files.

## Quick Checks

Estimate cost without calling APIs:

```bash
python -m helm_eval_lex_quirks.main --dry-run
```

Run the unit tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/helm_eval_pycache python -m unittest discover -v
```

## Style/Tic Evals

Run all built-in non-LinkedIn style suites cheaply:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite all \
  --style-samples 2 \
  --models gpt-4o-mini
```

Run specific suites across multiple models:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite lexical punctuation formatting tone \
  --style-samples 4 \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

Run LinkedIn-style detection:

```bash
python -m helm_eval_lex_quirks.main \
  --linkedin-only \
  --linkedin-samples 4 \
  --models gpt-4o-mini
```

Run pairwise comparison across models:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite all \
  --style-samples 4 \
  --pairwise-preference \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

Run pairwise with Bradley-Terry, condition segmentation, and recency weighting:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite all \
  --style-samples 4 \
  --pairwise-preference \
  --pairwise-method bradley_terry \
  --pairwise-segment-by condition \
  --pairwise-time-decay-halflife-days 90 \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

Recompute pairwise offline from an existing results JSON (no API calls):

```bash
python -m helm_eval_lex_quirks.main \
  --pairwise-from-results results/results_20260522_072249.json \
  --pairwise-method bradley_terry \
  --pairwise-segment-by condition \
  --pairwise-time-decay-halflife-days 90
```

Score pairwise comparison with a single detector:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite rest_cai_validation \
  --style-samples 5 \
  --pairwise-detector rest_suggestion \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6
```

Run an LLM-judge suite for subjective tone categories:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite llm_judge_tone \
  --style-samples 3 \
  --judge-model gpt-4o-mini \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6
```

Optional: run dual-judge disagreement audit on the same suite:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite llm_judge_tone \
  --style-samples 3 \
  --judge-model gpt-4o-mini \
  --judge-model-secondary claude-haiku-4-5-20251001 \
  --judge-disagreement-threshold 0.4 \
  --models claude-sonnet-4-6
```

Emit a safety-case report with deep-dive candidates:

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite all \
  --style-samples 4 \
  --safety-thresholds configs/thresholds.yaml \
  --deep-dive-top-k 10 \
  --models gpt-4o-mini gpt-4.1-nano
```

LLM-judge suites are opt-in. They add judge-model cost and should be audited for judge bias.
LLM-judge parsing clamps detector score into `[0, 1]` and truncates stored `rationale`/raw judge text to 500 chars.
When secondary judge auditing is enabled, each LLM-judge detector checkpoint record adds:
- `secondary_score`, `secondary_hit`, `secondary_confidence`, `secondary_rationale`
- `score_delta` (absolute primary-secondary score gap)
- `agreement` (primary and secondary hit match)
## Persona Transfer

Persona transfer compares each style/persona condition against a suite baseline. It measures whether a persona system prompt increases tic prevalence on neutral task prompts. It does not measure persistent cross-session model memory; public API calls are stateless unless the caller supplies prior messages.

```bash
python -m helm_eval_lex_quirks.main \
  --style-only \
  --style-suite all \
  --style-samples 4 \
  --persona-transfer \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

If `--persona-transfer` is passed without `--style-suite`, the harness enables all non-LinkedIn built-in style suites by default.

## Benchmark Evals

Run benchmark scenarios with defaults:

```bash
python -m helm_eval_lex_quirks.main \
  --models gpt-4o-mini \
  --scenarios truthfulqa hellaswag boolq
```

Keep costs lower while iterating:

```bash
python -m helm_eval_lex_quirks.main \
  --models gpt-4o-mini \
  --scenarios truthfulqa \
  --n-samples 20 \
  --no-robustness
```

## Outputs

Results are written under `results/` by default:

- `results/results_<timestamp>.json`: aggregate machine-readable summary.
- `results/summary_<timestamp>.txt`: human-readable report.
- `results/safety_case_<timestamp>.md`: threshold crossings, uncertainty flags, and deep-dive candidates.
- `results/checkpoints/*.jsonl`: per-instance or per-style-run checkpoints.

Checkpoint files allow interrupted runs to resume without repeating completed API calls.

Style-suite reports include Wilson score intervals for detector hit rates. Conditions with fewer than 30 outputs are marked `low-n`.
Style-suite JSON and text reports also include task segments, grouped by prompt `task_type`.
Pairwise preference compares models on shared style-suite records and ranks lower detector-count outputs higher.
Pairwise ranking supports `--pairwise-method count|bradley_terry`, segmentation via `--pairwise-segment-by none|task_type|condition`, and optional recency weighting via `--pairwise-time-decay-halflife-days`.

## Notes For Reliable Comparisons

- Use `--dry-run` before larger multi-model runs.
- Keep `--style-samples` fixed when comparing models.
- Record the exact model names and run timestamp; public model aliases can change.
- Prefer comparing deltas across conditions/models over interpreting raw detector hits.
- Increase `--request-interval` if you hit rate limits.

## Extending The Harness

Reusable eval primitives belong in `compass` first. If you want to add a new
detector or suite that should be shared, define it there and then consume it
from this repo. Keep repo-local changes here limited to orchestration, scenario
loading, reporting, and one-off experiments.
