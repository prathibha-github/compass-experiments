"""Shim: re-exports TicSuite framework from compass.detectors.

All evaluation primitives live in compass. Existing importers continue to
work unchanged; new code should import from compass.detectors directly.
"""
from typing import Dict

from compass.detectors import (
    CharacterCountDetector,
    DetectorResult,
    EMOJI_PATTERN,
    EmojiDetector,
    JudgeFn,
    LLM_JUDGE_SYSTEM_PROMPT,
    LLMJudgeDetector,
    LOW_SAMPLE_THRESHOLD,
    PhraseSetDetector,
    RecordExtras,
    RegexDetector,
    StyleCondition,
    StylePrompt,
    TicDetector,
    TicSuite,
    analyze_text,
    compute_uplift,
    suite_uses_llm_judge,
    summarize_detector_records,
    summarize_outputs,
)
from compass.judges.reliability import wilson_interval as _wilson_base


def wilson_interval(hit_count: int, n_outputs: int) -> Dict[str, float]:
    """Wilson 95% CI as a {low, high} percentage dict (backward-compat wrapper)."""
    if n_outputs <= 0:
        return {"low": 0.0, "high": 0.0}
    low, high = _wilson_base(hit_count, n_outputs)
    return {"low": round(100 * low, 1), "high": round(100 * high, 1)}


__all__ = [
    "CharacterCountDetector",
    "DetectorResult",
    "EmojiDetector",
    "JudgeFn",
    "LLM_JUDGE_SYSTEM_PROMPT",
    "LLMJudgeDetector",
    "LOW_SAMPLE_THRESHOLD",
    "PhraseSetDetector",
    "RecordExtras",
    "RegexDetector",
    "StyleCondition",
    "StylePrompt",
    "TicDetector",
    "TicSuite",
    "analyze_text",
    "compute_uplift",
    "suite_uses_llm_judge",
    "summarize_detector_records",
    "summarize_outputs",
    "wilson_interval",
]
