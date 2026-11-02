"""Shared helpers for unit test configuration setup."""

from compass_experiments.config import EvalConfig


def make_eval_config(**overrides) -> EvalConfig:
    base = dict(
        api_key="dummy",
        anthropic_api_key="anthropic-key",
        google_api_key="fake-google-key",
        output_dir="results",
        scenarios=[],
        models=["gpt-4o-mini"],
        style_only=True,
        style_suites=[],
    )
    base.update(overrides)
    return EvalConfig(**base)
