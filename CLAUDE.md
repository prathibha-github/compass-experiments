# helm_eval_lex_quirks

Experiment runner for LLM style and constitutional compliance evaluations. This repo is the execution layer; it does not own evaluation primitives — those live in compass.

## Two-repo architecture

```
compass/  (~/.../compass)            ← the library — owns all stable primitives
helm_eval_lex_quirks/                ← this repo — consumes compass
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
- `helm_eval_lex_quirks/models/` clients are now thin adapters over compass clients
  - `AnthropicClient` wraps `compass.clients.AnthropicClient`
  - `OpenAIClient` wraps `compass.clients.OpenAIClient` (adds logprob pass-through for ECE)
  - `GeminiClient` wraps `compass.clients.GoogleAIClient` (adds safety_settings=BLOCK_NONE and max_requests guard)
  - `OllamaClient` wraps `compass.clients.OllamaClient`
  - `OpenAIResponsesClient` wraps `compass.clients.OpenAIResponsesClient`
- `metrics/tic_suite.py` is a pure re-export shim from `compass.detectors`
- Rubric text in `builtin_tic_suites.py` uses `compass.RubricLibrary` references
- `compass-eval` is in `requirements.txt`

## Adding a new benchmark

Define it in compass first:
1. Write prompts, conditions, and rubrics/detectors as a `TicSuite` in `compass/detectors/suites.py`
2. Register it in `SUITE_REGISTRY`
3. Run it from this repo: `python phase2_benchmark.py --suites my_new_suite --models ...`

Anything worth repeating or sharing belongs in compass as a versioned artifact. Throw-away exploration can live here.

## Key model notes

- GPT-5 and o4 models require `temperature=1.0` — enforced in the client
- GPT-5 uses the Responses API (`OpenAIResponsesClient`), not Chat Completions
- Gemini free tier has rate limits (`max_requests` in `ModelConfig`); paid tier does not
- Pricing is in `config.py` `MODEL_CONFIGS` — update when adding new models

## Commit style

No AI co-authorship lines. Concise imperative messages.
