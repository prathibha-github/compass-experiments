# Compass Experiments Architecture

`compass_experiments` is the execution layer that runs scenario evals, style
suite evals, pairwise analysis, and reporting on top of shared Compass
primitives.

## Repo Boundary

```mermaid
flowchart LR
    Compass[compass<br/>shared clients rubrics detectors comparison]
    Experiments[compass_experiments<br/>runner scenarios reports cli]
    Results[results and checkpoints]

    Compass --> Experiments
    Experiments --> Results
```

## Main Evaluation Flow

```mermaid
flowchart TD
    CLI[main.py CLI flags] --> Config[EvalConfig]
    Config --> Runner[EvalRunner]
    Runner --> ClientFactory[models.create_client]
    ClientFactory --> CompassClients[Compass-backed client adapters]

    Runner --> ScenarioLoop[benchmark scenarios]
    Runner --> StyleLoop[style suites]

    ScenarioLoop --> ScenarioLoad[scenario loaders]
    ScenarioLoop --> Metrics[accuracy calibration robustness]

    StyleLoop --> Suites[builtin or LinkedIn suites]
    Suites --> CompassDetectors[Compass detector primitives]
    StyleLoop --> JudgeClients[judge model clients]
    StyleLoop --> Pairwise[pairwise and persona transfer]

    Metrics --> Report[report.py]
    Pairwise --> Report
    Report --> Results[JSON TXT MD outputs]
    ScenarioLoop --> Checkpoints[JSONL checkpoints]
    StyleLoop --> Checkpoints
```

## Key Files

- `compass_experiments/main.py`: CLI entrypoint and config wiring
- `compass_experiments/runner.py`: main orchestration path
- `compass_experiments/models/`: thin adapters over Compass clients
- `compass_experiments/metrics/`: suite wiring, reporting helpers, pairwise logic
- `compass_experiments/scenarios/`: dataset-specific loaders
- `phase2_benchmark.py`: focused constitutional benchmark runner
- `compass_experiments/report.py`: text, JSON, and safety-case report output

## Boundaries

- Shared primitives should be added to `compass` first.
- Repo-local changes here should stay focused on orchestration, scenario
  loading, reporting, and one-off evaluation flows.
