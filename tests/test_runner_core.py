import tempfile
import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.config import EvalConfig, ScenarioConfig
from helm_eval_lex_quirks.metrics.tic_suite import (
    EmojiDetector,
    StyleCondition,
    StylePrompt,
    TicSuite,
)
from helm_eval_lex_quirks.runner import (
    EvalResults,
    EvalRunner,
    estimate_model_cost,
)
from helm_eval_lex_quirks.scenarios.base import Instance


class RunnerCoreTests(unittest.TestCase):
    def _config(self, **overrides):
        base = dict(
            api_key="dummy",
            output_dir=tempfile.gettempdir(),
            scenarios=[],
            models=["gpt-4o-mini"],
            style_only=True,
            style_suites=[],
        )
        base.update(overrides)
        return EvalConfig(**base)

    def _suite(self):
        return TicSuite(
            name="unit_suite",
            prompts=[
                StylePrompt(id="p0", text="Write a short note.", task_type="general"),
                StylePrompt(id="p1", text="Write an update.", task_type="update"),
            ],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[EmojiDetector(name="emoji")],
            baseline_condition="neutral",
        )

    def test_estimate_model_cost_applies_gpt5_multiplier(self):
        prompts = ["abcd"]
        gpt5 = estimate_model_cost("gpt-5-mini", prompts, max_tokens=2)
        regular = estimate_model_cost("gpt-4o-mini", prompts, max_tokens=2)
        self.assertGreater(gpt5, regular)

    def test_estimate_cost_increases_with_judge_models(self):
        cfg_base = self._config(style_suites=["llm_judge_tone"], style_samples=1)
        cfg_judged = self._config(
            style_suites=["llm_judge_tone"],
            style_samples=1,
            judge_model="gpt-4o-mini",
            judge_model_secondary="gpt-4.1-mini",
        )
        base_cost = EvalRunner(cfg_base).estimate_cost()["gpt-4o-mini"]
        judged_cost = EvalRunner(cfg_judged).estimate_cost()["gpt-4o-mini"]
        self.assertGreater(judged_cost, base_cost)

    def test_run_attaches_pairwise_results(self):
        cfg = self._config(
            models=["m1", "m2"],
            run_pairwise_preference=True,
            pairwise_detector="emoji",
        )
        fake_pairwise = {"suite": {"overall": {"ranking": []}}}
        with patch("helm_eval_lex_quirks.runner.create_client", return_value=object()), patch.object(
            EvalRunner, "_run_model", side_effect=[EvalResults(model="m1"), EvalResults(model="m2")]
        ), patch(
            "helm_eval_lex_quirks.runner.analyze_pairwise_preference",
            return_value=fake_pairwise,
        ):
            out = EvalRunner(cfg).run()
        self.assertEqual(out["m1"].pairwise_preference_results, fake_pairwise)
        self.assertEqual(out["m2"].pairwise_preference_results, fake_pairwise)

    def test_run_model_skips_unknown_scenario(self):
        class SubjectClient:
            total_tokens = {"input": 11, "output": 4}
            total_cost_usd = 0.01

        cfg = self._config(scenarios=["unknown_scenario"], style_only=False)
        runner = EvalRunner(cfg)
        with self.assertLogs("helm_eval_lex_quirks.runner", level="WARNING") as logs:
            result = runner._run_model(SubjectClient(), "gpt-4o-mini")
        self.assertIn("Unknown scenario", "\n".join(logs.output))
        self.assertEqual(result.token_usage, {"input": 11, "output": 4})
        self.assertEqual(result.total_cost_usd, 0.01)

    def test_run_model_handles_load_error(self):
        class BrokenScenario:
            name = "broken"

            def load(self, n_samples, n_few_shot, seed):
                raise RuntimeError("load failed")

            def format_prompt(self, instance, few_shot):
                return "prompt"

        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

        cfg = self._config(scenarios=["broken"], style_only=False)
        runner = EvalRunner(cfg)
        with patch.dict("helm_eval_lex_quirks.runner.SCENARIO_REGISTRY", {"broken": BrokenScenario}), patch.dict(
            "helm_eval_lex_quirks.runner.DEFAULT_SCENARIO_CONFIGS",
            {"broken": ScenarioConfig("broken", n_samples=1, n_few_shot=0)},
        ):
            with self.assertLogs("helm_eval_lex_quirks.runner", level="ERROR") as logs:
                result = runner._run_model(SubjectClient(), "gpt-4o-mini")
        self.assertEqual(result.scenario_results, {})
        self.assertIn("Failed to load broken", "\n".join(logs.output))

    def test_aggregate_warnings_for_partial_and_missing_coverage(self):
        records_partial = [
            {
                "prediction": "A",
                "correct": "A",
                "confidence": 0.9,
                "is_correct": True,
                "coverage": 1.0,
                "perturbed_is_correct": None,
                "metadata": {},
            },
            {
                "prediction": "B",
                "correct": "A",
                "confidence": 0.6,
                "is_correct": False,
                "coverage": 0.2,
                "perturbed_is_correct": None,
                "metadata": {},
            },
        ]
        with self.assertLogs("helm_eval_lex_quirks.runner", level="WARNING") as logs_partial:
            sr_partial = EvalRunner._aggregate("s", records_partial, 2, "x.jsonl")
        self.assertIsNotNone(sr_partial.ece)
        self.assertLess(sr_partial.ece_coverage, 1.0)
        self.assertIn("incomplete logprob coverage", "\n".join(logs_partial.output))

        records_none = [
            {
                "prediction": "A",
                "correct": "A",
                "confidence": 0.9,
                "is_correct": True,
                "coverage": 0.1,
                "perturbed_is_correct": None,
                "metadata": {},
            }
        ]
        with self.assertLogs("helm_eval_lex_quirks.runner", level="WARNING") as logs_none:
            sr_none = EvalRunner._aggregate("s", records_none, 1, "x.jsonl")
        self.assertIsNone(sr_none.ece)
        self.assertIn("ECE cannot be computed", "\n".join(logs_none.output))

    def test_aggregate_bbq_metrics_are_populated(self):
        records = [
            {
                "prediction": "A",
                "correct": "A",
                "confidence": 0.9,
                "is_correct": True,
                "coverage": 1.0,
                "perturbed_is_correct": True,
                "metadata": {"context_condition": "ambig"},
            },
            {
                "prediction": "B",
                "correct": "B",
                "confidence": 0.8,
                "is_correct": True,
                "coverage": 1.0,
                "perturbed_is_correct": True,
                "metadata": {"context_condition": "disambig"},
            },
        ]
        sr = EvalRunner._aggregate("bbq", records, 2, "x.jsonl")
        self.assertEqual(sr.ambig_unknown_rate, 1.0)
        self.assertEqual(sr.disambig_accuracy, 1.0)
        self.assertIsNotNone(sr.robustness_score)

    def test_record_task_type_fallbacks(self):
        suite = self._suite()
        self.assertEqual(
            EvalRunner._record_task_type({"task_type": "explicit"}, suite),
            "explicit",
        )
        self.assertEqual(
            EvalRunner._record_task_type({"prompt_id": "p1"}, suite),
            "update",
        )
        self.assertEqual(
            EvalRunner._record_task_type({"prompt_idx": 0}, suite),
            "general",
        )
        self.assertEqual(
            EvalRunner._record_task_type({}, suite),
            "general",
        )

    def test_extract_and_summarize_disagreements_with_edge_inputs(self):
        malformed_record = {
            "detectors": {
                "a": "not-a-dict",
                "b": {"metadata": "not-a-dict"},
                "c": {"score": 0.5, "metadata": {"secondary_score": 0.2, "score_delta": 0.3}},
            },
            "condition": "neutral",
            "task_type": "task",
        }
        rows = EvalRunner._extract_judge_disagreements(malformed_record)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["detector"], "c")

        summary_empty = EvalRunner._summarize_judge_disagreements([], threshold=0.4)
        self.assertEqual(summary_empty["overall"]["n"], 0)
        self.assertEqual(summary_empty["overall"]["agreement_pct"], 0.0)

    def test_eval_scenario_continues_on_instance_error(self):
        class ScenarioStub:
            name = "stub"

            def format_prompt(self, instance, few_shot):
                if instance.id == "bad":
                    raise RuntimeError("bad prompt")
                return f"Prompt: {instance.input}"

        class ClientStub:
            def complete(self, prompt, max_tokens=1, logprobs=True, top_logprobs=20, system=None):
                class Resp:
                    completion = "A"
                    logprobs = []
                return Resp()

        instances = [
            Instance(id="good", input="x", choices=["A", "B"], choice_texts=["x", "y"], correct="A"),
            Instance(id="bad", input="y", choices=["A", "B"], choice_texts=["x", "y"], correct="A"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(output_dir=tmpdir)
            runner = EvalRunner(cfg)
            ckpt_path = runner._checkpoint_path("m", "stub")
            with patch("helm_eval_lex_quirks.runner.extract_choice_probs", return_value=({"A": 0.8}, 1.0)), patch(
                "helm_eval_lex_quirks.runner.extract_mc_answer", return_value="A"
            ):
                sr = runner._eval_scenario(
                    client=ClientStub(),
                    scenario=ScenarioStub(),
                    instances=instances,
                    few_shot=[],
                    run_robustness=False,
                    checkpoint_path=ckpt_path,
                )
        self.assertEqual(sr.n_completed, 1)


if __name__ == "__main__":
    unittest.main()
