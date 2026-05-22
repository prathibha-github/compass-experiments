import tempfile
import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.config import EvalConfig
from helm_eval_lex_quirks.metrics.tic_suite import (
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


if __name__ == "__main__":
    unittest.main()
