# Constitutional-compliance suite runner

`phase2_benchmark.py` runs Compass `TicSuite` definitions with separate generation
and scoring phases. It supports heuristic detectors and LLM judges, including
judges that need the original user request as context. The results measure
behavior under the selected prompts and conditions; they do not identify the
training procedure that caused it.

## Setup and first run

Follow the [README setup instructions](README.md#setup). The harness expects a
sibling Compass checkout and uses its shared clients and suite runner.
Set the keys needed by both the subject models and the judge:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Preview a run without making API calls:

```bash
python phase2_benchmark.py \
  --models claude-haiku-4-5-20251001 claude-sonnet-4-6 \
  --suites task_focus truthfulness \
  --judge-model gpt-4o-mini \
  --samples 5 \
  --temperature 1.0 \
  --max-tokens 1000 \
  --disallow-mixed-token-budgets \
  --output-dir results/compliance_pilot \
  --dry-run
```

Remove `--dry-run` to execute. To resume, repeat the same command and output
directory; automatic resume is enabled by default. Keep the judge and other
experiment settings unchanged.

For local generation, install and start Ollama and pull the requested model.
The runner recognizes the `llama3.1`, `llama3`, `llama3.2`, `mistral`, and `phi`
model prefixes, with optional tags. Use a local judge as well if the entire run
must avoid cloud calls:

```bash
ollama pull llama3.1
python phase2_benchmark.py \
  --models llama3.1 \
  --judge-model llama3.1 \
  --suites task_focus truthfulness \
  --samples 1 \
  --output-dir results/local_pilot
```

The default judge is `gpt-4o-mini`, so selecting local subject models alone does
not make a run local-only. Local API costs are recorded as zero; hardware and
energy costs are not included. A model judging itself also needs calibration.

## CLI options

Run `python phase2_benchmark.py --help` for the complete interface.

| Option | Default | Behavior |
| --- | --- | --- |
| `--models` | `llama3.1 mistral phi` | Subject models |
| `--judge-model` | `gpt-4o-mini` | Judge for LLM detectors; ignored by heuristic-only suites |
| `--suites` | `task_focus truthfulness` | Keys in the harness suite registry |
| `--samples` | `20` | Samples per model, prompt, and condition |
| `--temperature` | `0.0` | Requested generation temperature; clients may enforce model-specific behavior |
| `--output-dir` | `results/benchmark` | Root containing a directory per suite |
| `--max-tokens` | unset | Uniform requested generation token budget |
| `--max-tokens-by-model` | unset | Explicit `MODEL=TOKENS` entries for every selected model |
| `--disallow-mixed-token-budgets` | off | Reject unequal effective generation budgets |
| `--auto-resume` | on | Reuse saved generations and completed detector scores |
| `--resume` | off | Explicitly enable resume, even if `--no-auto-resume` is supplied |
| `--no-auto-resume` | off | Reset generation and scoring for selected suites unless `--resume` is also set |
| `--dry-run` | off | Print the plan and estimated call exposure without model calls |

`--max-tokens` and `--max-tokens-by-model` cannot be combined. Without an
explicit budget, the runner uses 1,000 tokens by default and 2,000 for Gemini
model prefixes. Estimates use local pricing tables and assumed token counts;
actual usage, retries, and provider availability need to be checked separately.

## Output files

For a suite such as `task_focus`, the output directory contains:

```text
results/compliance_pilot/
  task_focus/
    suite_generations.jsonl
    suite_evaluations_gpt-4o-mini.jsonl
    .cache/
  truthfulness/
    suite_generations.jsonl
    suite_evaluations_gpt-4o-mini.jsonl
    .cache/
  benchmark_summary_<timestamp>.txt
```

Heuristic-only suites use `suite_evaluations_heuristic.jsonl`. This runner does
not write the old combined `benchmark.jsonl` file.

Generation records include the full completion, prompt text, model, suite,
condition, sample index, token use, and generation-quality diagnostics such as
`hit_token_cap` and `is_fragment`. Evaluation records contain one row per
detector and sample, with `score`, `hit`, `count`, judge name, and copied
quality fields. Completion excerpts and rationales in evaluation rows are
truncated; retain the generation file for full outputs.

The text summary groups by model, suite, condition, and detector. It reports
hit rates, Wilson intervals, and evaluated row counts. It does not automatically
exclude quality-flagged generations or account for missing evaluations. Inspect
quality fields and runtime failure logs before comparing rates.

## Resume and rescoring

Generation identities are `(model, suite, prompt_id, condition, sample_idx)`.
Evaluation identities add the detector name. Completed records are skipped;
missing generations or detector evaluations can be retried independently.

Use a fresh output directory when changing prompts, system conditions,
sampling settings, or generation budgets. Resume fingerprints the saved prompt
and condition definitions, model, temperature, and token budget; scoring also
checks detector definitions and saved generations. Custom-client behavior and
provider changes behind a model alias remain outside those checks. Increasing
`--samples` on an otherwise unchanged run requests additional sample indices.

Adding a detector with a new name or selecting a new judge can score the saved
completions. Changing an existing detector's definition under the same name
requires explicit rescoring. For example, run this from the harness root after
the corresponding generation file exists:

```python
from pathlib import Path
from compass import evaluate_suite_completions
from compass_experiments.metrics.builtin_tic_suites import STYLE_SUITE_REGISTRY
from phase2_benchmark import create_client

suite_dir = Path("results/compliance_pilot/task_focus")
evaluate_suite_completions(
    STYLE_SUITE_REGISTRY["task_focus"],
    suite_dir / "suite_generations.jsonl",
    suite_dir,
    judge_model="gpt-4o-mini",
    resume=False,
    client_factory=lambda model: create_client(model)[0],
)
```

This rebuilds detector scores while retaining generations. Judge calls can
still hit Compass's cache when their evaluation inputs are unchanged.
`--no-auto-resume` resets generations too, so use it only for an intentional
fresh run. Legacy generation files remain readable for explicit rescoring, but
resume rejects rows without input fingerprints. Missing sample indices are
interpreted as zero; lost sample distinctions cannot be reconstructed. Keep
`.manifest.json` files beside generation and evaluation JSONL files. Failures
raise while preserving successful work; resume the failed phase before reporting.
Imported legacy generations without manifests cannot verify intended coverage.

## Interpreting scores

For `task_focus`, a high judge score means unsolicited tangential advice. For
`truthfulness`, it means fabrication or unjustified certainty. Lower is better
for these rubrics. Heuristic `score` and `count` fields can have a different
scale, so inspect the detector definition before comparing them.

A fixed judge makes comparisons more consistent but does not eliminate judge
bias. Audit examples, controls, invalid judge responses, and agreement with
human labels. Use varied prompts and enough samples; repeated low-temperature
outputs are not independent evidence. This CLI's text summary uses response-level
Wilson intervals, which do not model shared prompt effects. Compass's
`summarize_suite_evaluations` helper provides distinct prompt counts and
prompt-cluster intervals; these still describe individual cells, not differences
between models or conditions. See [Compass's scoring limitations](https://github.com/prathibha-github/compass#current-scoring-limitations)
for invalid-judgment handling. Follow [CALIBRATION.md](CALIBRATION.md) to export
saved examples for independent human labeling and report detector error rates.

## Extending the suite registry

Reusable definitions belong in Compass's `compass/detectors/suites.py` and
`SUITE_REGISTRY`. The harness exposes the desired keys through
`compass_experiments/metrics/builtin_tic_suites.py` and its
`STYLE_SUITE_REGISTRY`. Experiment-specific orchestration stays here.

Inspect the currently exposed suites:

```bash
python examples/custom_constitutional_eval.py
```

That script prints suite details and example commands without running models.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the repository boundary.

## Troubleshooting and checks

- A missing provider key may belong to the judge rather than the subject model.
- For Ollama connection errors, confirm the service is running and the requested
  model is installed.
- For checkpoint corruption, back up the file before repairing it. Generation
  scoring reads saved generations strictly and can reject malformed rows;
  evaluation resume also rejects malformed or incompatible rows.
- Use `--dry-run` before a paid run. Its plan describes the requested workload,
  not an exact bill for only the remaining checkpoint rows.

Run the resume and CLI tests from the harness root:

```bash
python -m pytest tests/test_phase2_benchmark.py tests/test_phase2_resume_semantics.py tests/test_cli_entrypoints.py -q
```
