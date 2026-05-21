"""Built-in tic/style suites for public API output evaluations."""
from typing import Dict

from helm_eval_lex_quirks.metrics.linkedin_style import LINKEDIN_SUITE
from helm_eval_lex_quirks.metrics.tic_suite import (
    CharacterCountDetector,
    EmojiDetector,
    PhraseSetDetector,
    RegexDetector,
    StyleCondition,
    StylePrompt,
    TicSuite,
)


STANDARD_CONDITIONS = [
    StyleCondition(name="neutral", system_prompt="You are a helpful assistant."),
    StyleCondition(
        name="professional",
        system_prompt=(
            "You are a professional, formal, and concise assistant. "
            "Respond with precision and avoid unnecessary embellishment."
        ),
    ),
    StyleCondition(
        name="playful",
        system_prompt=(
            "You are a playful and imaginative assistant. Use a vivid, whimsical voice, "
            "but still answer the user's request."
        ),
    ),
    StyleCondition(
        name="enthusiastic",
        system_prompt=(
            "You are an enthusiastic assistant. Be warm, energetic, and encouraging."
        ),
    ),
]


GENERAL_WRITING_PROMPTS = [
    StylePrompt(
        id="project_update",
        text="Write a short project update for a cross-functional engineering team.",
    ),
    StylePrompt(
        id="product_summary",
        text="Summarize a new productivity feature for internal users.",
    ),
    StylePrompt(
        id="technical_explanation",
        text="Explain why a database migration might take longer than expected.",
    ),
    StylePrompt(
        id="launch_note",
        text="Draft a short launch note for a tool that helps analysts review data faster.",
    ),
]


LEXICAL_TIC_SUITE = TicSuite(
    name="lexical_tics",
    prompts=GENERAL_WRITING_PROMPTS,
    conditions=STANDARD_CONDITIONS,
    detectors=[
        RegexDetector(name="goblin", pattern=r"\bgoblins?\b"),
        RegexDetector(name="delve", pattern=r"\bdelv(?:e|es|ed|ing)\b"),
        RegexDetector(name="tapestry", pattern=r"\btapestr(?:y|ies)\b"),
        RegexDetector(name="little_x", pattern=r"\blittle\s+[a-z][a-z-]*\b"),
        RegexDetector(
            name="not_just_but",
            pattern=r"\bnot\s+just\b[\s\S]{0,120}?\bbut\b",
        ),
    ],
    baseline_condition="neutral",
)


PUNCTUATION_TIC_SUITE = TicSuite(
    name="punctuation_tics",
    prompts=GENERAL_WRITING_PROMPTS,
    conditions=STANDARD_CONDITIONS,
    detectors=[
        CharacterCountDetector(name="em_dash", character="—", hit_threshold=2),
        CharacterCountDetector(name="semicolon", character=";", hit_threshold=2),
        EmojiDetector(name="emoji"),
        EmojiDetector(name="excessive_emoji", hit_threshold=3),
    ],
    baseline_condition="neutral",
)


FORMATTING_TIC_SUITE = TicSuite(
    name="formatting_tics",
    prompts=GENERAL_WRITING_PROMPTS,
    conditions=STANDARD_CONDITIONS,
    detectors=[
        RegexDetector(name="bullet_lines", pattern=r"(?m)^\s*[-*•]\s+"),
        RegexDetector(name="numbered_lines", pattern=r"(?m)^\s*\d+[\.)]\s+"),
        RegexDetector(name="markdown_headings", pattern=r"(?m)^\s{0,3}#{1,6}\s+"),
        RegexDetector(name="markdown_tables", pattern=r"(?m)^\s*\|.+\|\s*$"),
        RegexDetector(name="trailing_question", pattern=r"\?\s*$"),
    ],
    baseline_condition="neutral",
)


TONE_TIC_SUITE = TicSuite(
    name="tone_tics",
    prompts=GENERAL_WRITING_PROMPTS,
    conditions=STANDARD_CONDITIONS,
    detectors=[
        PhraseSetDetector(
            name="over_apology",
            phrases={
                "sorry",
                "apologize",
                "apologies",
                "i understand your frustration",
            },
        ),
        PhraseSetDetector(
            name="cheerleading",
            phrases={
                "great job",
                "you've got this",
                "excited for you",
                "amazing work",
                "love this",
            },
        ),
        PhraseSetDetector(
            name="hedging",
            phrases={
                "it depends",
                "may",
                "might",
                "could be",
                "generally speaking",
                "in many cases",
            },
        ),
        PhraseSetDetector(
            name="therapy_speak",
            phrases={
                "hold space",
                "valid",
                "your feelings",
                "boundaries",
                "process this",
            },
        ),
        PhraseSetDetector(
            name="corporate_jargon",
            phrases={
                "align",
                "leverage",
                "stakeholders",
                "move the needle",
                "north star",
                "cross-functional",
                "unlock",
            },
        ),
    ],
    baseline_condition="neutral",
)


STYLE_SUITE_REGISTRY: Dict[str, TicSuite] = {
    "linkedin": LINKEDIN_SUITE,
    "lexical": LEXICAL_TIC_SUITE,
    "punctuation": PUNCTUATION_TIC_SUITE,
    "formatting": FORMATTING_TIC_SUITE,
    "tone": TONE_TIC_SUITE,
}


def expand_suite_names(names):
    """Expand user-facing aliases into concrete registry keys."""
    expanded = []
    for name in names:
        if name == "all":
            expanded.extend(k for k in STYLE_SUITE_REGISTRY if k != "linkedin")
        else:
            expanded.append(name)
    return expanded
