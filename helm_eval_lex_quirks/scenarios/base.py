from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Instance:
    id: str
    input: str
    choices: List[str]       # labels: ["A","B","C","D"] or ["A","B"]
    choice_texts: List[str]  # human-readable answer options
    correct: str             # correct label
    metadata: Dict           = field(default_factory=dict)


class Scenario(ABC):
    name: str
    description: str

    @abstractmethod
    def load(
        self, n_samples: int, n_few_shot: int, seed: int = 42
    ) -> Tuple[List[Instance], List[Instance]]:
        """Return (test_instances, few_shot_pool)."""
        ...

    @abstractmethod
    def format_prompt(self, instance: Instance, few_shot: List[Instance]) -> str:
        ...

    def extract_answer(self, completion: str, instance: Instance) -> str:
        c = completion.strip().upper()
        for choice in instance.choices:
            if c == choice or c.startswith(choice + ".") or c.startswith(choice + " "):
                return choice
        return c[:1] if c else instance.choices[0]
