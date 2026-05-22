# HELM Eval Lex Quirks

Small evaluation harness for measuring model behavior on public API outputs.

The harness supports:

- HELM-style benchmark slices for accuracy, calibration, and typo robustness.
- Lexical/style tic detection across prompt conditions.
- LinkedIn-style emoji and phrase prevalence.
- Built-in tic suites for lexical, punctuation, formatting, and tone indicators.
- Persona-transfer analysis: whether persona/style system prompts increase tic prevalence on the same neutral tasks.
- OpenAI and Claude model clients.

These detectors are trend-analysis tools, not definitive classifiers. Tone and formatting indicators can be appropriate in context; compare across models, versions, prompts, and conditions rather than treating a single hit as conclusive.

## Setup

```bash
cd ~/dev_setup/helm_eval_lex_quirks
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

LLM-judge suites are opt-in. They add judge-model cost and should be audited for judge bias.
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
- `results/checkpoints/*.jsonl`: per-instance or per-style-run checkpoints.

Checkpoint files allow interrupted runs to resume without repeating completed API calls.

Style-suite reports include Wilson score intervals for detector hit rates. Conditions with fewer than 30 outputs are marked `low-n`.
Style-suite JSON and text reports also include task segments, grouped by prompt `task_type`.
Pairwise preference compares models on shared style-suite records and ranks lower detector-count outputs higher.

## Notes For Reliable Comparisons

- Use `--dry-run` before larger multi-model runs.
- Keep `--style-samples` fixed when comparing models.
- Record the exact model names and run timestamp; public model aliases can change.
- Prefer comparing deltas across conditions/models over interpreting raw detector hits.
- Increase `--request-interval` if you hit rate limits.
