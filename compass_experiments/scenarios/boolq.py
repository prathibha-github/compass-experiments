import random
from typing import List, Tuple

from datasets import load_dataset

from compass_experiments.scenarios.base import Instance, Scenario

MAX_PASSAGE_CHARS = 800  # keep prompts short for cost


class BoolQScenario(Scenario):
    name = "boolq"
    description = "BoolQ — reading comprehension yes/no questions"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng = random.Random(seed)
        for hf_id in ("google/boolq", "boolq"):
            try:
                ds = load_dataset(hf_id, trust_remote_code=True)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load boolq dataset")

        data = list(ds["validation"])
        rng.shuffle(data)
        instances = [self._make(ex, i) for i, ex in enumerate(data)]
        return instances[:n_samples], instances[n_samples: n_samples + n_few_shot]

    def _make(self, ex, idx) -> Instance:
        passage = ex["passage"][:MAX_PASSAGE_CHARS]
        return Instance(
            id=f"boolq_{idx}",
            input=f"Passage: {passage}\nQuestion: {ex['question']}",
            choices=["A", "B"],
            choice_texts=["Yes", "No"],
            correct="A" if ex["answer"] else "B",
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["Read the passage and answer Yes (A) or No (B).\n"]
        for ex in few_shot:
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [inst.input, "A. Yes", "B. No"]
        lines.append(f"Answer: {inst.correct}\n" if answer else "Answer:")
        return "\n".join(lines)
