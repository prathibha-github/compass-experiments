from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str
    input_cost_per_million: float   # USD per 1M input tokens
    output_cost_per_million: float  # USD per 1M output tokens


MODEL_CONFIGS = {
    "gpt-4o-mini":   ModelConfig("gpt-4o-mini",   0.15,  0.60),
    "gpt-4o":        ModelConfig("gpt-4o",        2.50, 10.00),
    "gpt-4-turbo":   ModelConfig("gpt-4-turbo",  10.00, 30.00),
    "gpt-4.1-nano":  ModelConfig("gpt-4.1-nano",  0.10,  0.40),
    "gpt-4.1-mini":  ModelConfig("gpt-4.1-mini",  0.40,  1.60),
    "o4-mini":       ModelConfig("o4-mini",       1.10,  4.40),
}


@dataclass
class ScenarioConfig:
    name: str
    n_samples: int = 100
    n_few_shot: int = 5


DEFAULT_SCENARIO_CONFIGS = {
    "mmlu":       ScenarioConfig("mmlu",       n_samples=100, n_few_shot=5),
    "truthfulqa": ScenarioConfig("truthfulqa", n_samples=100, n_few_shot=5),
    "hellaswag":  ScenarioConfig("hellaswag",  n_samples=100, n_few_shot=5),
    "boolq":      ScenarioConfig("boolq",      n_samples=100, n_few_shot=3),
    "bbq":        ScenarioConfig("bbq",        n_samples=100, n_few_shot=3),
}


@dataclass
class EvalConfig:
    api_key: str
    models: List[str]            = field(default_factory=lambda: ["gpt-4o-mini"])
    scenarios: List[str]         = field(default_factory=lambda: list(DEFAULT_SCENARIO_CONFIGS))
    n_samples: Optional[int]     = None   # overrides per-scenario default when set
    output_dir: str              = "results"
    run_robustness: bool         = True
    run_quirk_detection: bool    = True
    run_linkedin_style: bool     = False
    linkedin_only: bool          = False
    linkedin_samples: int        = 4     # repetitions per prompt for style detection
    seed: int                    = 42
    dry_run: bool                = False
    request_interval: float      = 2.0   # seconds between API calls; increase for free-tier accounts
