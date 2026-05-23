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
- TicSuite framework and builtin suite definitions (planned, see below)
- Pairwise ranking, judge reliability auditing

**This repo owns:**
- Runner and orchestration (`runner.py`, `phase2_benchmark.py`)
- Scenario loaders (MMLU, BoolQ, HellaSwag, TruthfulQA, BBQ)
- Reporting (`report.py`)
- CLI and config (`main.py`, `config.py`)
- Experiment-specific one-off code

## Planned refactor

The following migration is planned but not yet done. Do not start it without reading the compass CLAUDE.md first.

**Phase (Session 1 — happens in compass first):**
- compass gets upgraded clients with retry/throttle/pricing
- compass gets `compass/detectors/` module with TicSuite framework + builtin suites

**Phase (Session 2 — happens here):**
- `helm_eval_lex_quirks/models/` clients replaced with compass clients
  - `AnthropicClient` → `compass.clients.AnthropicClient`
  - `OpenAIClient` → `compass.clients.OpenAIClient`
  - `GeminiClient` → `compass.clients.GoogleAIClient`
  - `OllamaClient` → `compass.clients.OllamaClient`
  - `OpenAIResponsesClient` → `compass.clients.OpenAIResponsesClient`
- Duplicated rubric text in `builtin_tic_suites.py` replaced with `compass.RubricLibrary` references
- `wilson_interval` in `metrics/tic_suite.py` replaced with `compass.judges.reliability.wilson_interval`
- `LLMJudgeDetector` JSON parsing upgraded to use `compass.judges.parsing.parse_judge_response`
- `helm_eval_lex_quirks` adds `compass-eval` to its dependencies

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
