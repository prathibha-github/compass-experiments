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
    "claude-haiku-4-5-20251001": ModelConfig("claude-haiku-4-5-20251001", 1.00, 5.00),
    "claude-haiku-4-5":          ModelConfig("claude-haiku-4-5",          1.00, 5.00),
    "claude-sonnet-4-6":         ModelConfig("claude-sonnet-4-6",         3.00, 15.00),
    "claude-opus-4-7":           ModelConfig("claude-opus-4-7",           5.00, 25.00),
}


def model_provider(model_name: str) -> str:
    """Return the API provider for a model name."""
    if model_name.startswith("claude-"):
        return "anthropic"
    return "openai"


def model_config(model_name: str) -> ModelConfig:
    """Return pricing config; unknown models fall back to a conservative same-provider default."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    if model_provider(model_name) == "anthropic":
        return MODEL_CONFIGS["claude-haiku-4-5-20251001"]
    return MODEL_CONFIGS["gpt-4o-mini"]


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
    api_key: Optional[str]
    anthropic_api_key: Optional[str] = None
    models: List[str]            = field(default_factory=lambda: ["gpt-4o-mini"])
    scenarios: List[str]         = field(default_factory=lambda: list(DEFAULT_SCENARIO_CONFIGS))
    n_samples: Optional[int]     = None   # overrides per-scenario default when set
    output_dir: str              = "results"
    run_robustness: bool         = True
    run_quirk_detection: bool    = False
    run_linkedin_style: bool     = False
    linkedin_only: bool          = False
    linkedin_samples: int        = 4     # repetitions per prompt for style detection
    style_suites: List[str]      = field(default_factory=list)
    style_only: bool             = False
    style_samples: int           = 4     # repetitions per prompt for generic tic suites
    run_persona_transfer: bool   = False
    run_pairwise_preference: bool = False
    pairwise_detector: Optional[str] = None
    pairwise_method: str         = "count"
    pairwise_segment_by: str     = "task_type"
    pairwise_time_decay_halflife_days: Optional[int] = None
    run_tag: Optional[str]       = None
    judge_model: Optional[str]   = None
    judge_model_secondary: Optional[str] = None
    judge_disagreement_threshold: float = 0.4
    seed: int                    = 42
    dry_run: bool                = False
    request_interval: float      = 2.0   # seconds between API calls; increase for free-tier accounts
