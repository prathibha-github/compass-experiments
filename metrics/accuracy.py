import re
from typing import List


def extract_mc_answer(completion: str, valid_choices: List[str]) -> str:
    c = completion.strip().upper()
    for choice in valid_choices:
        if c == choice or c.startswith(choice + ".") or c.startswith(choice + " "):
            return choice
    m = re.match(r"^([A-D])", c)
    if m and m.group(1) in valid_choices:
        return m.group(1)
    return valid_choices[0]


def multiple_choice_accuracy(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == r for p, r in zip(predictions, references)) / len(predictions)
