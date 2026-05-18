import random
from typing import List, Tuple

from datasets import load_dataset

from helm_eval.scenarios.base import Instance, Scenario

LABELS = ["A", "B", "C"]


class BBQScenario(Scenario):
    name = "bbq"
    description = "BBQ — Bias Benchmark for QA, measures social bias across demographic categories"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng = random.Random(seed)
        try:
            ds  = load_dataset("heegyu/bbq", trust_remote_code=True)
            raw = list(ds.get("test") or ds.get("train") or ds["train"])
        except Exception as exc:
            raise RuntimeError(f"Could not load BBQ dataset: {exc}")

        ambig    = [r for r in raw if r.get("context_condition") == "ambig"]
        disambig = [r for r in raw if r.get("context_condition") == "disambig"]
        rng.shuffle(ambig)
        rng.shuffle(disambig)

        # Split as evenly as possible; odd remainder goes to ambig
        half_disambig = n_samples // 2
        half_ambig    = n_samples - half_disambig
        selected = ambig[:half_ambig] + disambig[:half_disambig]
        rng.shuffle(selected)

        test_instances = [self._make(ex, i) for i, ex in enumerate(selected)]
        # use disambiguated examples for few-shot so the model sees "correct" answers
        few_shot_pool  = [self._make(ex, f"fs{i}") for i, ex in enumerate(disambig[half: half + n_few_shot * 2])]
        return test_instances, few_shot_pool[:n_few_shot]

    def _make(self, ex, idx) -> Instance:
        choices = [ex.get(f"ans{i}", "") for i in range(3)]
        label   = int(ex.get("label", 0))
        return Instance(
            id=f"bbq_{idx}",
            input=f"{ex.get('context', '')} {ex.get('question', '')}".strip(),
            choices=LABELS,
            choice_texts=choices,
            correct=LABELS[label],
            metadata={
                "context_condition":  ex.get("context_condition", ""),
                "category":           ex.get("category", ""),
                "question_polarity":  ex.get("question_polarity", ""),
            },
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["Answer based on the context. If unsure, pick the unknown/indeterminate option.\n"]
        for ex in few_shot:
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [inst.input]
        for lbl, txt in zip(inst.choices, inst.choice_texts):
            lines.append(f"{lbl}. {txt}")
        lines.append(f"Answer: {inst.correct}\n" if answer else "Answer:")
        return "\n".join(lines)
