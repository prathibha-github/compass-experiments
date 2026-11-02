import random
from typing import List, Tuple

from datasets import load_dataset

from compass_experiments.scenarios.base import Instance, Scenario

LABELS = ["A", "B", "C", "D"]


class HellaSwagScenario(Scenario):
    name = "hellaswag"
    description = "HellaSwag — commonsense NLI, choose the most plausible sentence continuation"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng  = random.Random(seed)
        # try both common dataset IDs
        for hf_id in ("Rowan/hellaswag", "hellaswag"):
            try:
                ds = load_dataset(hf_id, trust_remote_code=True)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load hellaswag dataset")

        data = list(ds["validation"])
        rng.shuffle(data)
        instances = [self._make(ex, i) for i, ex in enumerate(data) if ex.get("label", "").strip()]
        return instances[:n_samples], instances[n_samples: n_samples + n_few_shot]

    def _make(self, ex, idx) -> Instance:
        return Instance(
            id=f"hellaswag_{idx}",
            input=ex["ctx"],
            choices=LABELS,
            choice_texts=ex["endings"],
            correct=LABELS[int(ex["label"])],
            metadata={"activity": ex.get("activity_label", "")},
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["Choose the most plausible continuation for each activity description.\n"]
        for ex in few_shot:
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [f"Context: {inst.input}"]
        for lbl, txt in zip(inst.choices, inst.choice_texts):
            lines.append(f"{lbl}. {txt}")
        lines.append(f"Answer: {inst.correct}\n" if answer else "Answer:")
        return "\n".join(lines)
