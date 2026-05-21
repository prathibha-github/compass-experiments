"""Generic tic/style evaluation primitives."""
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class StyleCondition:
    """A system-prompt condition to evaluate."""

    name: str
    system_prompt: str
    description: str = ""


@dataclass(frozen=True)
class StylePrompt:
    """A user prompt in a tic/style suite."""

    id: str
    text: str


@dataclass(frozen=True)
class DetectorResult:
    """Normalized output for one detector on one completion."""

    name: str
    count: int
    score: float
    hit: bool
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "score": self.score,
            "hit": self.hit,
            "metadata": self.metadata,
        }


class TicDetector:
    """Base detector interface for concrete tic detectors."""

    name: str

    def detect(self, text: str) -> DetectorResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RegexDetector(TicDetector):
    """Count regex matches, useful for lexical, punctuation, and format tics."""

    name: str
    pattern: str
    flags: int = re.IGNORECASE
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        matches = list(re.finditer(self.pattern, text, flags=self.flags))
        count = len(matches)
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={"matches": [m.group(0) for m in matches[:20]]},
        )


@dataclass(frozen=True)
class PhraseSetDetector(TicDetector):
    """Count case-insensitive phrase occurrences from a fixed phrase set."""

    name: str
    phrases: Iterable[str]
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        text_lower = text.lower()
        phrase_hits: Dict[str, int] = {}
        total = 0
        for phrase in self.phrases:
            count = text_lower.count(phrase.lower())
            if count:
                phrase_hits[phrase] = count
                total += count
        return DetectorResult(
            name=self.name,
            count=total,
            score=float(total),
            hit=total >= self.hit_threshold,
            metadata={"phrases": phrase_hits},
        )


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "♀-♂"
    "☀-⭕"
    "‍"
    "⏏"
    "⏩"
    "⌚"
    "️"
    "〰"
    "]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class EmojiDetector(TicDetector):
    """Count emoji occurrences."""

    name: str = "emoji"
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        count = len(EMOJI_PATTERN.findall(text))
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={},
        )


@dataclass(frozen=True)
class CharacterCountDetector(TicDetector):
    """Count a literal character, such as em dashes or semicolons."""

    name: str
    character: str
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        count = text.count(self.character)
        return DetectorResult(
            name=self.name,
            count=count,
            score=float(count),
            hit=count >= self.hit_threshold,
            metadata={"character": self.character},
        )


RecordExtras = Callable[[str, Dict[str, DetectorResult]], Dict[str, object]]


@dataclass(frozen=True)
class TicSuite:
    """A reusable tic/style eval suite."""

    name: str
    prompts: List[StylePrompt]
    conditions: List[StyleCondition]
    detectors: List[TicDetector]
    baseline_condition: str
    checkpoint_name: Optional[str] = None
    max_tokens: int = 200
    record_extras: Optional[RecordExtras] = None

    @property
    def checkpoint_stem(self) -> str:
        return self.checkpoint_name or self.name

    def extra_record_fields(
        self, text: str, detector_results: Dict[str, DetectorResult]
    ) -> Dict[str, object]:
        if not self.record_extras:
            return {}
        return self.record_extras(text, detector_results)


def analyze_text(text: str, suite: TicSuite) -> Dict[str, DetectorResult]:
    """Run all suite detectors on a completion."""
    return {detector.name: detector.detect(text) for detector in suite.detectors}


def summarize_outputs(condition_outputs: Dict[str, List[str]], suite: TicSuite) -> Dict[str, dict]:
    """Aggregate detector prevalence and intensity by condition."""
    summary: Dict[str, dict] = {}

    for condition, outputs in condition_outputs.items():
        detector_values = {detector.name: [] for detector in suite.detectors}
        detector_hits = {detector.name: 0 for detector in suite.detectors}

        for output in outputs:
            results = analyze_text(output, suite)
            for name, result in results.items():
                detector_values[name].append(result.count)
                if result.hit:
                    detector_hits[name] += 1

        detectors = {}
        for name, values in detector_values.items():
            n_outputs = len(outputs)
            detectors[name] = {
                "mean_count": round(sum(values) / n_outputs, 2) if n_outputs else 0.0,
                "pct_hit": round(100 * detector_hits[name] / n_outputs, 1) if n_outputs else 0.0,
            }

        summary[condition] = {
            "n_outputs": len(outputs),
            "detectors": detectors,
        }

    return summary


def compute_uplift(
    condition_results: Dict[str, dict],
    detector_name: str,
    metric: str = "mean_count",
    baseline: Optional[str] = None,
) -> Dict[str, float]:
    """Compute detector uplift by condition against a baseline condition."""
    if not condition_results:
        return {}

    baseline = baseline or next(iter(condition_results))
    if baseline not in condition_results:
        baseline = next(iter(condition_results))

    base_stats = condition_results[baseline]["detectors"].get(detector_name, {})
    base_value = base_stats.get(metric, 0.0)

    uplift = {}
    for condition, stats in condition_results.items():
        value = stats["detectors"].get(detector_name, {}).get(metric, 0.0)
        if base_value > 0:
            delta = 100 * (value - base_value) / max(base_value, 0.1)
        else:
            delta = 100 if value > 0 else 0
        uplift[condition] = round(delta, 1)
    return uplift
