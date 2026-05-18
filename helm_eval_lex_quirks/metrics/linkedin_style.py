"""
LinkedIn style detector — measures emoji frequency and LinkedIn-speak tics.

Analogous to goblin quirk detection but targets a specific style drift:
emoji prevalence and characteristic phrases from professional social media.
"""
import re
from typing import Dict, List

SYSTEM_PROMPTS: Dict[str, str] = {
    "neutral": (
        "You are a helpful assistant."
    ),
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


def count_emojis(text: str) -> int:
    """Count emoji occurrences in text (emoji Unicode ranges)."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "♀-♂"
        "☀-⭕"
        "‍"
        "⏏"
        "⏩"
        "⌚"
        "️"
        "〰"
        "]",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(text))


def count_linkedin_phrases(text: str) -> int:
    """Count occurrences of characteristic LinkedIn phrases."""
    text_lower = text.lower()
    count = 0
    for phrase in LINKEDIN_PHRASES:
        count += text_lower.count(phrase)
    return count


def analyze_outputs(
    condition_outputs: Dict[str, List[str]],
) -> Dict[str, dict]:
    """
    Analyze outputs across conditions.
    Returns per-condition statistics: emoji %, excessive emoji %, mean emojis, phrase prevalence.
    """
    results = {}

    for condition, outputs in condition_outputs.items():
        if not outputs:
            results[condition] = {
                "n_outputs": 0,
                "pct_with_emoji": 0.0,
                "pct_excessive_emoji": 0.0,
                "mean_emoji_count": 0.0,
                "mean_phrase_hits": 0.0,
                "linkedin_score_mean": 0.0,
            }
            continue

        emoji_counts = [count_emojis(o) for o in outputs]
        phrase_counts = [count_linkedin_phrases(o) for o in outputs]

        has_emoji = sum(1 for c in emoji_counts if c > 0)
        excessive = sum(1 for c in emoji_counts if c >= 3)

        results[condition] = {
            "n_outputs": len(outputs),
            "pct_with_emoji": round(100 * has_emoji / len(outputs), 1),
            "pct_excessive_emoji": round(100 * excessive / len(outputs), 1),
            "mean_emoji_count": round(sum(emoji_counts) / len(outputs), 2),
            "mean_phrase_hits": round(sum(phrase_counts) / len(outputs), 2),
            "linkedin_score_mean": round(
                (sum(emoji_counts) + sum(phrase_counts)) / len(outputs), 2
            ),
        }

    return results


def compute_uplift(
    condition_results: Dict[str, dict],
    baseline: str = "professional",
) -> Dict[str, dict]:
    """
    Compute uplift vs baseline condition (emoji % and phrase hits).
    """
    if baseline not in condition_results:
        baseline = list(condition_results.keys())[0]

    baseline_emoji = condition_results[baseline]["pct_with_emoji"]
    baseline_phrases = condition_results[baseline]["mean_phrase_hits"]

    uplift = {}
    for cond, stats in condition_results.items():
        emoji_uplift = (
            100 * (stats["pct_with_emoji"] - baseline_emoji) / max(baseline_emoji, 1)
            if baseline_emoji > 0
            else (100 if stats["pct_with_emoji"] > 0 else 0)
        )
        phrase_uplift = (
            100 * (stats["mean_phrase_hits"] - baseline_phrases) / max(baseline_phrases, 0.1)
            if baseline_phrases > 0
            else (100 if stats["mean_phrase_hits"] > 0 else 0)
        )

        uplift[cond] = {
            "emoji_uplift_pct": round(emoji_uplift, 1),
            "phrase_uplift_pct": round(phrase_uplift, 1),
        }

    return uplift
