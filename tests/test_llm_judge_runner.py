import tempfile
import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.config import EvalConfig
from helm_eval_lex_quirks.metrics.tic_suite import (
    DetectorResult,
    LLMJudgeDetector,
    StyleCondition,
    StylePrompt,
    TicSuite,
)
from helm_eval_lex_quirks.models.openai_client import ModelResponse
from helm_eval_lex_quirks.runner import EvalRunner


class LLMJudgeRunnerTests(unittest.TestCase):
    def _suite(self):
        return TicSuite(
            name="judge_suite",
            prompts=[StylePrompt(id="p0", text="Respond to the user.")],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy.")],
            baseline_condition="neutral",
        )

    def test_run_tic_suite_uses_judge_client(self):
        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                return ModelResponse(completion="You are absolutely right.", logprobs=None)

        class JudgeClient:
            def __init__(self):
                self._tokens = {"input": 10, "output": 4}

            @property
            def total_tokens(self):
                return self._tokens

            @property
            def total_cost_usd(self):
                return 0.001

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                return ModelResponse(
                    completion=(
                        '{"score": 1.0, "hit": true, '
                        '"confidence": 0.9, "rationale": "uncritical agreement"}'
                    ),
                    logprobs=None,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvalConfig(
                api_key="dummy",
                output_dir=tmpdir,
                scenarios=[],
                style_only=True,
                style_suites=[],
                judge_model="judge-model",
            )
            with patch("helm_eval_lex_quirks.runner.create_client", return_value=JudgeClient()):
                result = EvalRunner(cfg)._run_tic_suite(
                    client=SubjectClient(),
                    model_name="subject-model",
                    suite=self._suite(),
                    samples=1,
                )

        detector_stats = result["analysis"]["neutral"]["detectors"]["sycophancy"]
        self.assertEqual(detector_stats["hit_count"], 1)
        self.assertEqual(result["judge"]["model"], "judge-model")
        self.assertEqual(result["judge"]["token_usage"], {"input": 10, "output": 4})

    def test_run_tic_suite_requires_judge_model(self):
        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                return ModelResponse(completion="Response.", logprobs=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvalConfig(
                api_key="dummy",
                output_dir=tmpdir,
                scenarios=[],
                style_only=True,
                style_suites=[],
            )
            with self.assertRaisesRegex(ValueError, "requires --judge-model"):
                EvalRunner(cfg)._run_tic_suite(
                    client=SubjectClient(),
                    model_name="subject-model",
                    suite=self._suite(),
                    samples=1,
                )


class LLMJudgeDetectorParsingTests(unittest.TestCase):
    def _detector(self):
        return LLMJudgeDetector(name="test_judge", rubric="Score the response.")

    def test_parse_well_formed_json(self):
        detector = self._detector()
        raw = '{"score": 0.75, "hit": true, "confidence": 0.95, "rationale": "clear and helpful"}'
        result = detector.parse_response(raw)

        self.assertEqual(result.name, "test_judge")
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.hit, True)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.metadata["confidence"], 0.95)
        self.assertEqual(result.metadata["rationale"], "clear and helpful")

    def test_parse_malformed_json_with_fallback(self):
        detector = self._detector()
        raw = 'The judge says: {"score": 0.5, "hit": false, "confidence": 0.8, "rationale": "mixed"} in this response'
        result = detector.parse_response(raw)

        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.hit, False)
        self.assertEqual(result.count, 0)

    def test_parse_completely_invalid_json(self):
        detector = self._detector()
        raw = "No JSON here at all, just text"
        result = detector.parse_response(raw)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.hit, False)
        self.assertEqual(result.count, 0)
        self.assertTrue(result.metadata.get("parse_error"))
        self.assertIn("No JSON", result.metadata["raw"])

    def test_score_clamped_to_upper_bound(self):
        detector = self._detector()
        raw = '{"score": 1.5, "hit": true, "confidence": 0.9, "rationale": "high"}'
        result = detector.parse_response(raw)

        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.hit, True)

    def test_score_clamped_to_lower_bound(self):
        detector = self._detector()
        raw = '{"score": -0.5, "hit": false, "confidence": 0.9, "rationale": "low"}'
        result = detector.parse_response(raw)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.hit, False)

    def test_hit_defaults_to_score_threshold_comparison(self):
        detector = self._detector()
        raw = '{"score": 0.6, "confidence": 0.8, "rationale": "above threshold"}'
        result = detector.parse_response(raw)

        # Default hit_threshold is 0.5, score is 0.6, so hit should be True
        self.assertEqual(result.hit, True)

    def test_hit_overrides_score_threshold(self):
        detector = self._detector()
        raw = '{"score": 0.3, "hit": true, "confidence": 0.9, "rationale": "override"}'
        result = detector.parse_response(raw)

        # Even though score (0.3) < threshold (0.5), hit is explicitly true
        self.assertEqual(result.score, 0.3)
        self.assertEqual(result.hit, True)

    def test_missing_confidence_extracted_safely(self):
        detector = self._detector()
        raw = '{"score": 0.7, "hit": true, "rationale": "no confidence field"}'
        result = detector.parse_response(raw)

        self.assertIsNone(result.metadata.get("confidence"))
        self.assertEqual(result.metadata["rationale"], "no confidence field")

    def test_missing_rationale_defaults_to_empty_string(self):
        detector = self._detector()
        raw = '{"score": 0.7, "hit": true, "confidence": 0.8}'
        result = detector.parse_response(raw)

        self.assertEqual(result.metadata["rationale"], "")

    def test_rationale_truncated_to_500_chars(self):
        detector = self._detector()
        long_rationale = "x" * 600
        raw = f'{{"score": 0.7, "hit": true, "confidence": 0.8, "rationale": "{long_rationale}"}}'
        result = detector.parse_response(raw)

        self.assertEqual(len(result.metadata["rationale"]), 500)

    def test_non_numeric_score_defaults_to_zero(self):
        detector = self._detector()
        raw = '{"score": "not a number", "hit": false, "confidence": 0.8, "rationale": "bad score"}'
        result = detector.parse_response(raw)

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.hit, False)

    def test_extract_json_object_finds_innermost_braces(self):
        detector = self._detector()
        raw = '{"score": 0.8, "hit": true, "confidence": 0.9, "rationale": "good"}'
        result = detector._extract_json_object(raw)

        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 0.8)

    def test_extract_json_object_returns_none_without_braces(self):
        detector = self._detector()
        raw = 'no braces here'
        result = detector._extract_json_object(raw)

        self.assertIsNone(result)

    def test_extract_json_object_returns_none_with_mismatched_braces(self):
        detector = self._detector()
        raw = '{missing closing brace'
        result = detector._extract_json_object(raw)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
