import hashlib
import random


def _stable_seed(text: str) -> int:
    """Process-stable hash so perturbations are identical across runs."""
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:4], "big")


def inject_typos(text: str, p: float = 0.05, seed: int = 42) -> str:
    """Swap adjacent alphabetic characters with probability p."""
    rng   = random.Random(seed ^ _stable_seed(text))
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def compute_robustness_score(base_acc: float, perturbed_acc: float) -> float:
    if base_acc == 0:
        return 1.0 if perturbed_acc == 0 else 0.0
    return min(perturbed_acc / base_acc, 1.0)
