# Compass Experiments

Experiment harness for evaluating LLM behavior, style, and constitutional
compliance through public model APIs, built on [Compass](https://github.com/prathibha-github/compass).

The project grew out of work with HELM and evolved into an independent
behavioral evaluation framework. Compass contains the reusable framework:
clients, rubrics, judges, detector primitives, and versioned evaluation suites.
This repository provides the experiment harness: orchestration, scenario
loading, configuration, and reporting. It has no runtime dependency on HELM.

The harness supports:

- Standardized benchmark slices for accuracy, calibration, and typo robustness.
- Lexical/style tic detection across prompt conditions.
- LinkedIn-style emoji and phrase prevalence.
- Built-in tic suites for lexical, punctuation, formatting, and tone indicators.
- Persona-transfer analysis: whether persona/style system prompts increase tic prevalence on the same neutral tasks.
- Thin adapters over Compass clients for OpenAI, Anthropic, Gemini, and Ollama.

These detectors are trend-analysis tools, not definitive classifiers. Tone and formatting indicators can be appropriate in context; compare across models, versions, prompts, and conditions rather than treating a single hit as conclusive.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the repo layout and execution flow.

## Setup

### Fresh machine

There is nothing to compile; compass is pure Python and installs via pip. It is
consumed as an editable sibling checkout, so clone both repos into the same
parent directory and keep compass on `main` (the two-phase suite runner used by
`phase2_benchmark.py` lives there).

```bash
mkdir -p ~/dev_setup && cd ~/dev_setup
git clone git@github.com:prathibha-github/compass.git
git clone git@github.com:prathibha-github/helm-eval-lex-quirks.git compass-experiments
cd ~/dev_setup/compass && git checkout main && git pull
cd ~/dev_setup/compass-experiments && ./setup.sh
```

The clone command uses the current GitHub URL and explicitly names the local
directory `compass-experiments`. The GitHub repository rename is a separate
step; setup does not depend on it.

`setup.sh` creates `.venv`, installs `requirements-base.txt`, and installs compass
editable (`pip install -e ../compass --no-deps`). Confirm the wiring:

```bash
.venv/bin/python -c "from compass import generate_suite_completions; print('compass two-phase OK')"
```

### Existing checkout

Run these commands from the root of your existing checkout. Keep its current
directory name and virtual environment; create `.venv` only if it is missing.

```bash
test -d .venv || python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` expects a sibling `../compass` checkout. `setup.sh` enforces
the same layout.

The project is named Compass Experiments; its Python package is
`compass_experiments`. Update existing imports and saved commands to use the
new package name. Neither a package rename nor a GitHub repository rename
changes your local checkout's directory name.

If you only want the public dependencies without the private `compass` install
step, use:

```bash
pip install -r requirements-base.txt
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
python -m compass_experiments.main --dry-run
```

Run the unit tests:

```bash
PYTHONPYCACHEPREFIX=/tmp/compass_experiments_pycache python -m unittest discover -v
```

## Style/Tic Evals

Run all built-in non-LinkedIn style suites cheaply:

```bash
python -m compass_experiments.main \
  --style-only \
  --style-suite all \
  --style-samples 2 \
  --models gpt-4o-mini
```

Run specific suites across multiple models:

```bash
python -m compass_experiments.main \
  --style-only \
  --style-suite lexical punctuation formatting tone \
  --style-samples 4 \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

Run LinkedIn-style detection:

```bash
python -m compass_experiments.main \
  --linkedin-only \
  --linkedin-samples 4 \
  --models gpt-4o-mini
```

Run pairwise comparison across models:

```bash
python -m compass_experiments.main \
  --style-only \
  --style-suite all \
  --style-samples 4 \
  --pairwise-preference \
  --models gpt-4o-mini gpt-4.1-nano gpt-5-nano
```

Run pairwise with Bradley-Terry, condition segmentation, and recency weighting:

```bash
python -m compass_experiments.main \
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
python -m compass_experiments.main \
  --pairwise-from-results results/results_20260522_072249.json \
  --pairwise-method bradley_terry \
  --pairwise-segment-by condition \
  --pairwise-time-decay-halflife-days 90
```

Score pairwise comparison with a single detector:

```bash
python -m compass_experiments.main \
  --style-only \
  --style-suite rest_cai_validation \
  --style-samples 5 \
  --pairwise-detector rest_suggestion \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6 claude-opus-4-6
```

Run an LLM-judge suite for subjective tone categories:

```bash
python -m compass_experiments.main \
  --style-only \
  --style-suite llm_judge_tone \
  --style-samples 3 \
  --judge-model gpt-4o-mini \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6
```

Optional: run dual-judge disagreement audit on the same suite:

```bash
python -m compass_experiments.main \
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
python -m compass_experiments.main \
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
python -m compass_experiments.main \
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
python -m compass_experiments.main \
  --models gpt-4o-mini \
  --scenarios truthfulqa hellaswag boolq
```

Keep costs lower while iterating:

```bash
python -m compass_experiments.main \
  --models gpt-4o-mini \
  --scenarios truthfulqa \
  --n-samples 20 \
  --no-robustness
```

### Constitutional-compliance suites (phase2)

The constitutional-compliance suites run through `phase2_benchmark.py`, which uses compass's two-phase suite runner: completions are generated once per `(model, prompt, condition, sample)` and persisted under `results/<suite>/`, then scored separately. Adding or re-running a detector (including an LLM judge) re-scores the saved completions without regenerating.

For sampling-based suites like `rest_cai_validation`, set `--temperature` above 0.0 so the per-condition samples are independent draws. `rest_cai_validation` includes a prompt-aware LLM-judge detector (`unsolicited_rest_judge`), so a `--judge-model` is required; use one fixed judge across all models so the comparison is not biased by self-judging.

Preview the plan and API-call exposure first with `--dry-run` (no calls made):

```bash
python phase2_benchmark.py \
  --suites rest_cai_validation \
  --models claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5-20251001 \
  --judge-model claude-haiku-4-5-20251001 \
  --temperature 1.0 \
  --samples 20 \
  --output-dir results/benchmark_pilot \
  --auto-resume \
  --dry-run
```

Drop `--dry-run` to run it. `--auto-resume` continues from existing completions in `--output-dir`, so re-running the same command never re-pays for generations already done. The summary breaks results down by model, condition, and detector with Wilson intervals. See `PHASE2_BENCHMARK.md` for the full set of suites and options.

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
