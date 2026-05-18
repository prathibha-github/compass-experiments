import math
import random
from typing import List, Tuple

from datasets import load_dataset

from helm_eval.scenarios.base import Instance, Scenario

LABELS = ["A", "B", "C", "D"]
SUBJECTS = [
    "anatomy",
    "high_school_history",
    "college_mathematics",
    "professional_law",
]


class MMLUScenario(Scenario):
    name = "mmlu"
    description = "Massive Multitask Language Understanding — diverse multiple-choice knowledge"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng      = random.Random(seed)
        # Ceiling division so total >= n_samples regardless of n_samples % n_subjects
        per_subj = max(1, math.ceil(n_samples / len(SUBJECTS)))
        test_instances: List[Instance] = []
        few_shot_pool:  List[Instance] = []

        for subj in SUBJECTS:
            ds = load_dataset("cais/mmlu", subj, trust_remote_code=True)
            test_split = list(ds["test"])
            dev_split  = list(ds.get("dev") or ds.get("validation") or [])

            rng.shuffle(test_split)
            test_instances += [self._make(ex, subj, i) for i, ex in enumerate(test_split[:per_subj])]
            few_shot_pool  += [self._make(ex, subj, f"fs{i}") for i, ex in enumerate(dev_split[:n_few_shot])]

        # Trim to exactly n_samples (ceiling can overshoot by up to n_subjects-1)
        return test_instances[:n_samples], few_shot_pool[:n_few_shot]

    def _make(self, ex, subject, idx) -> Instance:
        choices = ex["choices"]
        return Instance(
            id=f"mmlu_{subject}_{idx}",
            input=ex["question"],
            choices=LABELS[:len(choices)],
            choice_texts=choices,
            correct=LABELS[ex["answer"]],
            metadata={"subject": subject},
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["The following are multiple choice questions (with answers).\n"]
        for ex in few_shot:
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [f"Question: {inst.input}"]
        for lbl, txt in zip(inst.choices, inst.choice_texts):
            lines.append(f"{lbl}. {txt}")
        lines.append(f"Answer: {inst.correct}\n" if answer else "Answer:")
        return "\n".join(lines)
