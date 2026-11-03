import re
from typing import List


ANSWER_PARSER_VERSION = 2


def extract_mc_answer(completion: str, valid_choices: List[str]):
    """Return an explicit leading choice label, or None for an invalid answer."""
    text = completion.strip().upper()
    for choice in valid_choices:
        # A label may be followed by punctuation or whitespace, not a word suffix.
        if re.match(r"^" + re.escape(choice.upper()) + r"(?:$|[.):]|\s)", text):
            return choice
    return None


def multiple_choice_accuracy(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)
