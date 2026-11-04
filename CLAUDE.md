# compass_experiments

Experiment runner for LLM style and constitutional compliance evaluations. This repo is the execution layer; it does not own evaluation primitives — those live in compass.

## Two-repo architecture

```
compass/  (~/.../compass)            ← the library — owns all stable primitives
compass_experiments/                ← this repo — consumes compass
```

Dependency direction is one-way: this repo imports from compass. Compass never imports from here.

## What lives where

**compass owns:**
- Rubric definitions (`RubricLibrary`)
- LLM judges, evaluation cache, checkpoint manager
- Client abstractions (AnthropicClient, OpenAIClient, GoogleAIClient, OllamaClient, OpenAIResponsesClient)
- TicSuite framework and builtin suite definitions
- Pairwise ranking, judge reliability auditing

**This repo owns:**
- Runner and orchestration (`runner.py`, `phase2_benchmark.py`)
- Scenario loaders (MMLU, BoolQ, HellaSwag, TruthfulQA, BBQ)
- Reporting (`report.py`)
- CLI and config (`main.py`, `config.py`)
- Experiment-specific one-off code

## Compass migration (complete)

Both migration sessions are done. Do not re-run them.

**Session 1 (compass):** upgraded clients with retry/throttle/pricing; added `compass/detectors/` with TicSuite framework and builtin suites.

**Session 2 (this repo):**
- `compass_experiments/models/` clients are now thin adapters over compass clients
  - `AnthropicClient` wraps `compass.clients.AnthropicClient`
  - `OpenAIClient` wraps `compass.clients.OpenAIClient` (adds logprob pass-through for ECE)
  - `GeminiClient` wraps `compass.clients.GoogleAIClient`; safety-filter disabling is opt-in
  - `OllamaClient` wraps `compass.clients.OllamaClient`
  - `OpenAIResponsesClient` wraps `compass.clients.OpenAIResponsesClient`
- `metrics/tic_suite.py` is a pure re-export shim from `compass.detectors`
- Rubric text in `builtin_tic_suites.py` uses `compass.RubricLibrary` references
- `requirements.txt` expects a sibling `../compass` checkout; `requirements-base.txt` contains only the non-compass dependencies

## Adding a new benchmark

Define it in compass first:
1. Write prompts, conditions, and rubrics/detectors as a `TicSuite` in `compass/detectors/suites.py`
2. Register it in `SUITE_REGISTRY`
3. Expose the suite in this repo's `STYLE_SUITE_REGISTRY` in `compass_experiments/metrics/builtin_tic_suites.py`.
4. Run it from this repo: `python phase2_benchmark.py --suites my_new_suite --models ...`

Anything worth repeating or sharing belongs in compass as a versioned artifact. Throw-away exploration can live here.

## Key model notes

- The harness configures `temperature=1.0` for GPT-5 and o4 prefixes; Compass adapters apply the explicit policy passed by their caller.
- The harness routes GPT-5 through `OpenAIResponsesClient` and configures a 10x output-token budget multiplier. Preserve the effective settings in comparisons.
- Gemini safety-filter disabling is opt-in. The harness does not pass `ModelConfig.max_requests` to the shared client; provider quotas still apply.
- Harness estimates use `config.py` `MODEL_CONFIGS`; returned call costs use Compass client pricing. Check both tables when adding a model.

## Commit style

No AI co-authorship lines. Concise imperative messages.
