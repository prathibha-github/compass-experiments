import random
from typing import List, Tuple

from datasets import load_dataset

from compass_experiments.scenarios.base import Instance, Scenario

LABELS = ["A", "B", "C", "D"]
SUBJECTS = [
    "anatomy",
    "high_school_us_history",
    "college_mathematics",
    "professional_law",
]


class MMLUScenario(Scenario):
    name = "mmlu"
    description = "Massive Multitask Language Understanding — diverse multiple-choice knowledge"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng      = random.Random(seed)
        # Allocate the remainder with the seed, independently of subject order.
        subject_order = sorted(SUBJECTS)
        rng.shuffle(subject_order)
        base, remainder = divmod(n_samples, len(subject_order))
        allocation = {subject: base + (i < remainder)
                      for i, subject in enumerate(subject_order)}
        test_instances: List[Instance] = []
        few_shot_pool:  List[Instance] = []

        for subj in subject_order:
            ds = load_dataset("cais/mmlu", subj, trust_remote_code=True)
            test_split = list(ds["test"])
            dev_split  = list(ds.get("dev") or ds.get("validation") or [])

            rng.shuffle(test_split)
            test_instances += [self._make(ex, subj, i) for i, ex in enumerate(test_split[:allocation[subj]])]
            few_shot_pool  += [self._make(ex, subj, f"fs{i}") for i, ex in enumerate(dev_split[:n_few_shot])]

        rng.shuffle(test_instances)
        return test_instances, few_shot_pool

    def _make(self, ex, subject, idx) -> Instance:
        choices = ex["choices"]
        return Instance(
            id=f"mmlu_{subject}_{idx}",
            input=ex["question"],
            choices=LABELS[:len(choices)],
            choice_texts=choices,
            correct=LABELS[ex["answer"]],
            metadata={"subject": subject, "sampling_policy": "balanced_subjects_v2",
                      "few_shot_policy": "same_subject"},
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["The following are multiple choice questions (with answers).\n"]
        for ex in few_shot:
            if ex.metadata["subject"] != instance.metadata["subject"]:
                continue
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [f"Question: {inst.input}"]
        for lbl, txt in zip(inst.choices, inst.choice_texts):
            lines.append(f"{lbl}. {txt}")
        lines.append(f"Answer: {inst.correct}\n" if answer else "Answer:")
        return "\n".join(lines)
