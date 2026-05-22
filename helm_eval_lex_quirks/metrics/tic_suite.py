"""Generic tic/style evaluation primitives."""
import json
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
    task_type: str = "general"


@dataclass(frozen=True)
class DetectorResult:
    """
    Normalized output for one detector on one completion.

    Consumers should rely on name/count/score/hit for aggregation. Metadata is
    detector-specific, JSON-serializable diagnostic detail such as matched regex
    spans, phrase hit counts, or the counted punctuation character.
    """

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


JudgeFn = Callable[[str, int], str]
LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. Return only valid JSON with keys "
    "score, hit, confidence, and rationale."
)


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
    """Count case-insensitive whole-phrase occurrences from a fixed phrase set."""

    name: str
    phrases: Iterable[str]
    hit_threshold: int = 1

    def detect(self, text: str) -> DetectorResult:
        phrase_hits: Dict[str, int] = {}
        total = 0
        for phrase in self.phrases:
            pattern = rf"\b{re.escape(phrase)}\b"
            count = len(re.findall(pattern, text, flags=re.IGNORECASE))
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


@dataclass(frozen=True)
class LLMJudgeDetector(TicDetector):
    """Score subjective style categories with a separate judge model."""

    name: str
    rubric: str
    hit_threshold: float = 0.5
    max_tokens: int = 180

    def build_prompt(self, text: str) -> str:
        return (
            "Evaluate the assistant response below using this rubric.\n\n"
            f"Rubric:\n{self.rubric}\n\n"
            "Return JSON exactly in this shape:\n"
            '{"score": 0.0, "hit": false, "confidence": 0.0, "rationale": "short reason"}\n\n'
            "Assistant response:\n"
            f"{text}"
        )

    def detect(self, text: str) -> DetectorResult:
        raise RuntimeError("LLMJudgeDetector requires a judge function")

    def detect_with_judge(self, text: str, judge_fn: JudgeFn) -> DetectorResult:
        raw = judge_fn(self.build_prompt(text), self.max_tokens)
        return self.parse_response(raw)

    def parse_response(self, raw: str) -> DetectorResult:
        # Keep score bounded to [0, 1] so downstream aggregation remains
        # comparable across judge prompts/models and robust to malformed output.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = self._extract_json_object(raw)

        if payload is None:
            return DetectorResult(
                name=self.name,
                count=0,
                score=0.0,
                hit=False,
                metadata={"parse_error": True, "raw": raw[:500]},
            )

        try:
            score = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        hit = bool(payload.get("hit", score >= self.hit_threshold))
        return DetectorResult(
            name=self.name,
            count=1 if hit else 0,
            score=score,
            hit=hit,
            metadata={
                "confidence": payload.get("confidence"),
                # Cap rationale/raw payload snapshots for checkpoint size hygiene.
                "rationale": str(payload.get("rationale", ""))[:500],
                "raw": raw[:500],
            },
        )

    @staticmethod
    def _extract_json_object(raw: str) -> Optional[dict]:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None


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
LOW_SAMPLE_THRESHOLD = 30
WILSON_Z_95 = 1.96


@dataclass(frozen=True)
class TicSuite:
    """
    A reusable tic/style eval suite.

    Example:
        suite = TicSuite(
            name="punctuation_tics",
            prompts=[StylePrompt(id="update", text="Write a short update.")],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[CharacterCountDetector(name="em_dash", character="—")],
            baseline_condition="neutral",
        )

    record_extras can be supplied when a suite needs backward-compatible or
    suite-specific checkpoint fields derived from detector results. It receives
    the completion text and detector results keyed by detector name.
    """

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


def suite_uses_llm_judge(suite: TicSuite) -> bool:
    return any(isinstance(detector, LLMJudgeDetector) for detector in suite.detectors)


def analyze_text(
    text: str,
    suite: TicSuite,
    judge_fn: Optional[JudgeFn] = None,
) -> Dict[str, DetectorResult]:
    """Run all suite detectors on a completion."""
    results: Dict[str, DetectorResult] = {}
    for detector in suite.detectors:
        if isinstance(detector, LLMJudgeDetector):
            if judge_fn is None:
                raise ValueError(f"Detector {detector.name!r} requires a judge function")
            results[detector.name] = detector.detect_with_judge(text, judge_fn)
        else:
            results[detector.name] = detector.detect(text)
    return results


def wilson_interval(hit_count: int, n_outputs: int, z: float = WILSON_Z_95) -> Dict[str, float]:
    """Return a Wilson score interval as percentages."""
    if n_outputs <= 0:
        return {"low": 0.0, "high": 0.0}

    p_hat = hit_count / n_outputs
    z2 = z * z
    denom = 1 + z2 / n_outputs
    center = (p_hat + z2 / (2 * n_outputs)) / denom
    margin = z * ((p_hat * (1 - p_hat) + z2 / (4 * n_outputs)) / n_outputs) ** 0.5 / denom
    return {
        "low": round(100 * max(0.0, center - margin), 1),
        "high": round(100 * min(1.0, center + margin), 1),
    }


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
            ci = wilson_interval(detector_hits[name], n_outputs)
            detectors[name] = {
                "hit_count": detector_hits[name],
                "mean_count": round(sum(values) / n_outputs, 2) if n_outputs else 0.0,
                "pct_hit": round(100 * detector_hits[name] / n_outputs, 1) if n_outputs else 0.0,
                "pct_hit_ci_low": ci["low"],
                "pct_hit_ci_high": ci["high"],
                "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            }

        summary[condition] = {
            "n_outputs": len(outputs),
            "low_sample": 0 < len(outputs) < LOW_SAMPLE_THRESHOLD,
            "detectors": detectors,
        }

    return summary


def summarize_detector_records(
    condition_records: Dict[str, List[Dict[str, dict]]],
    suite: TicSuite,
) -> Dict[str, dict]:
    """Aggregate detector results already stored in checkpoint records."""
    summary: Dict[str, dict] = {}

    for condition, records in condition_records.items():
        detector_values = {detector.name: [] for detector in suite.detectors}
        detector_hits = {detector.name: 0 for detector in suite.detectors}

        for detector_record in records:
            for detector in suite.detectors:
                payload = detector_record.get(detector.name, {})
                count = float(payload.get("count", 0.0))
                hit = bool(payload.get("hit", count > 0))
                detector_values[detector.name].append(count)
                if hit:
                    detector_hits[detector.name] += 1

        n_outputs = len(records)
        detector_summary = {}
        for detector in suite.detectors:
            values = detector_values[detector.name]
            hit_count = detector_hits[detector.name]
            ci = wilson_interval(hit_count, n_outputs)
            detector_summary[detector.name] = {
                "hit_count": hit_count,
                "pct_hit": round(100 * hit_count / n_outputs, 1) if n_outputs else 0.0,
                "pct_hit_ci_low": ci["low"],
                "pct_hit_ci_high": ci["high"],
                "mean_count": round(sum(values) / n_outputs, 3) if n_outputs else 0.0,
                "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            }

        summary[condition] = {
            "n_outputs": n_outputs,
            "low_sample": 0 < n_outputs < LOW_SAMPLE_THRESHOLD,
            "detectors": detector_summary,
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
