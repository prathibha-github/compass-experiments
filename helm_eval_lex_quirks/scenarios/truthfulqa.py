import random
import warnings
from typing import List, Tuple

from datasets import load_dataset

from helm_eval_lex_quirks.scenarios.base import Instance, Scenario

LABELS = ["A", "B", "C", "D"]
N_DISTRACTORS = 3   # correct answer + this many wrong answers = 4 choices max


class TruthfulQAScenario(Scenario):
    name = "truthfulqa"
    description = "TruthfulQA — tests whether models generate truthful answers"

    def load(self, n_samples: int, n_few_shot: int, seed: int = 42) -> Tuple[List[Instance], List[Instance]]:
        rng = random.Random(seed)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*loading script.*")
            ds = load_dataset("truthful_qa", "multiple_choice", trust_remote_code=True)
        data = list(ds["validation"])
        rng.shuffle(data)

        # _make now takes rng to sample distractors reproducibly
        instances = [self._make(ex, i, rng) for i, ex in enumerate(data) if self._valid(ex)]
        return instances[:n_samples], instances[n_samples: n_samples + n_few_shot]

    @staticmethod
    def _valid(ex) -> bool:
        return 1 in ex["mc1_targets"]["labels"]

    def _make(self, ex, idx, rng: random.Random) -> Instance:
        all_choices = ex["mc1_targets"]["choices"]
        all_labels  = ex["mc1_targets"]["labels"]

        # Locate the correct answer before any slicing
        correct_idx  = all_labels.index(1)
        correct_text = all_choices[correct_idx]

        # Sample distractors from the wrong answers
        wrong_indices = [i for i, l in enumerate(all_labels) if l != 1]
        chosen_wrong  = rng.sample(wrong_indices, min(N_DISTRACTORS, len(wrong_indices)))
        options       = [all_choices[i] for i in chosen_wrong] + [correct_text]
        rng.shuffle(options)

        correct_label = LABELS[options.index(correct_text)]
        return Instance(
            id=f"truthfulqa_{idx}",
            input=ex["question"],
            choices=LABELS[:len(options)],
            choice_texts=options,
            correct=correct_label,
        )

    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        parts = ["Answer each question with the truthful choice.\n"]
        for ex in few_shot:
            parts.append(self._fmt(ex, answer=True))
        parts.append(self._fmt(instance, answer=False))
        return "\n".join(parts)

    def _fmt(self, inst: Instance, answer: bool) -> str:
        lines = [f"Q: {inst.input}"]
        for lbl, txt in zip(inst.choices, inst.choice_texts):
            lines.append(f"{lbl}. {txt}")
        lines.append(f"A: {inst.correct}\n" if answer else "A:")
        return "\n".join(lines)
