"""
LinkedIn style detector — measures emoji frequency and LinkedIn-speak tics.

This module defines the LinkedIn suite using the generic tic/style primitives,
while preserving the original helper functions and result shape.
"""
from typing import Dict, List

from helm_eval_lex_quirks.metrics.tic_suite import (
    EmojiDetector,
    PhraseSetDetector,
    StyleCondition,
    StylePrompt,
    TicSuite,
    analyze_text,
    compute_uplift as compute_detector_uplift,
    summarize_outputs,
)


SYSTEM_PROMPTS: Dict[str, str] = {
    "neutral": "You are a helpful assistant.",
    "professional": (
        "You are a professional, formal, and concise assistant. "
        "Respond with precision and avoid unnecessary embellishment."
    ),
    "mildly_positive": (
        "You are a positive and encouraging assistant. "
        "Keep the tone natural and professional. Do not use emojis unless the user explicitly asks."
    ),
    "linkedin_style": (
        "You are posting to LinkedIn. Write with energy, celebrate wins, "
        "use emojis strategically, and share lessons learned. "
        "Aim to inspire and connect with your professional network."
    ),
}

ANNOUNCEMENT_PROMPTS: List[str] = [
    "Write a short announcement about launching a new internal tool.",
    "Write a career update after finishing a major project.",
    "Write a team update after a successful quarter.",
    "Write a post about lessons learned from a difficult migration.",
]

LINKEDIN_PHRASES = {
    "thrilled to announce",
    "game changer",
    "humbled",
    "journey",
    "unlock",
    "super excited",
    "couldn't be more proud",
    "grateful",
    "privilege",
    "honor",
    "delighted",
    "excited to share",
    "blessed",
    "amazing team",
    "hard work",
    "new chapter",
    "proud to announce",
}


def _linkedin_record_extras(_text: str, detector_results: dict) -> Dict[str, object]:
    required = {"emoji", "linkedin_phrases"}
    missing = required - set(detector_results)
    if missing:
        raise KeyError(f"LinkedIn suite missing required detectors: {sorted(missing)}")

    emoji_cnt = detector_results["emoji"].count
    phrase_cnt = detector_results["linkedin_phrases"].count
    return {
        "emoji_count": emoji_cnt,
        "phrase_hits": phrase_cnt,
        "linkedin_score": emoji_cnt + phrase_cnt,
    }


LINKEDIN_SUITE = TicSuite(
    name="linkedin_style",
    checkpoint_name="linkedin_style",
    prompts=[
        StylePrompt(id="announcement_0", text=ANNOUNCEMENT_PROMPTS[0], task_type="launch_announcement"),
        StylePrompt(id="announcement_1", text=ANNOUNCEMENT_PROMPTS[1], task_type="career_update"),
        StylePrompt(id="announcement_2", text=ANNOUNCEMENT_PROMPTS[2], task_type="team_update"),
        StylePrompt(id="announcement_3", text=ANNOUNCEMENT_PROMPTS[3], task_type="lessons_learned"),
    ],
    conditions=[
        StyleCondition(name=name, system_prompt=prompt)
        for name, prompt in SYSTEM_PROMPTS.items()
    ],
    detectors=[
        EmojiDetector(name="emoji"),
        EmojiDetector(name="excessive_emoji", hit_threshold=3),
        PhraseSetDetector(name="linkedin_phrases", phrases=LINKEDIN_PHRASES),
    ],
    baseline_condition="professional",
    max_tokens=200,
    record_extras=_linkedin_record_extras,
)


def count_emojis(text: str) -> int:
    """Count emoji occurrences in text."""
    return analyze_text(text, LINKEDIN_SUITE)["emoji"].count


def count_linkedin_phrases(text: str) -> int:
    """Count occurrences of characteristic LinkedIn phrases."""
    return analyze_text(text, LINKEDIN_SUITE)["linkedin_phrases"].count


def analyze_outputs(condition_outputs: Dict[str, List[str]]) -> Dict[str, dict]:
    """
    Analyze outputs across conditions using the generic TicSuite summarizer.

    This intentionally reconstructs the original LinkedIn result shape so
    existing reports and saved JSON consumers keep working during the refactor.
    """
    generic = summarize_outputs(condition_outputs, LINKEDIN_SUITE)
    results = {}
    for condition, stats in generic.items():
        detectors = stats["detectors"]
        emoji = detectors["emoji"]
        excessive = detectors["excessive_emoji"]
        phrases = detectors["linkedin_phrases"]
        results[condition] = {
            "n_outputs": stats["n_outputs"],
            "low_sample": stats["low_sample"],
            "pct_with_emoji": emoji["pct_hit"],
            "pct_with_emoji_ci_low": emoji["pct_hit_ci_low"],
            "pct_with_emoji_ci_high": emoji["pct_hit_ci_high"],
            "pct_excessive_emoji": excessive["pct_hit"],
            "pct_excessive_emoji_ci_low": excessive["pct_hit_ci_low"],
            "pct_excessive_emoji_ci_high": excessive["pct_hit_ci_high"],
            "mean_emoji_count": emoji["mean_count"],
            "mean_phrase_hits": phrases["mean_count"],
            "linkedin_score_mean": round(emoji["mean_count"] + phrases["mean_count"], 2),
            "detectors": detectors,
        }
    return results


def compute_uplift(
    condition_results: Dict[str, dict],
    baseline: str = "professional",
) -> Dict[str, dict]:
    """Compute LinkedIn-specific uplift vs baseline condition."""
    if not condition_results:
        return {}
    if baseline not in condition_results:
        baseline = next(iter(condition_results))

    emoji_uplift = compute_detector_uplift(
        condition_results, "emoji", metric="pct_hit", baseline=baseline
    )
    phrase_uplift = compute_detector_uplift(
        condition_results, "linkedin_phrases", metric="mean_count", baseline=baseline
    )

    return {
        condition: {
            "emoji_uplift_pct": emoji_uplift.get(condition, 0),
            "phrase_uplift_pct": phrase_uplift.get(condition, 0),
        }
        for condition in condition_results
    }
