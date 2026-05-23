"""
Unit tests for runner core logic.

Covers unit-level branching for cost estimation, scenario loading/error paths,
aggregation behavior, and judge disagreement helpers.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from helm_eval_lex_quirks.config import ScenarioConfig
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
from tests._test_base import make_eval_config


class RunnerCoreTests(unittest.TestCase):
    def _config(self, **overrides):
        base = {"output_dir": tempfile.gettempdir()}
        base.update(overrides)
        return make_eval_config(**base)

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

    def test_estimate_cost_increases_with_robustness_and_judges_combined(self):
        cfg_base = self._config(
            scenarios=["boolq"],
            style_only=False,
            style_suites=["llm_judge_tone"],
            style_samples=1,
            run_robustness=False,
            judge_model=None,
            judge_model_secondary=None,
        )
        cfg_combo = self._config(
            scenarios=["boolq"],
            style_only=False,
            style_suites=["llm_judge_tone"],
            style_samples=1,
            run_robustness=True,
            judge_model="gpt-4o-mini",
            judge_model_secondary="gpt-4.1-mini",
        )
        base_cost = EvalRunner(cfg_base).estimate_cost()["gpt-4o-mini"]
        combo_cost = EvalRunner(cfg_combo).estimate_cost()["gpt-4o-mini"]
        self.assertGreater(combo_cost, base_cost)

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

    def test_run_model_mixed_scenario_success_and_failure(self):
        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

        class GoodScenario:
            name = "good"

            def load(self, n_samples, n_few_shot, seed):
                return (["inst"], [])

            def format_prompt(self, instance, few_shot):
                return "prompt"

        class BadScenario:
            name = "bad"

            def load(self, n_samples, n_few_shot, seed):
                raise RuntimeError("boom")

            def format_prompt(self, instance, few_shot):
                return "prompt"

        cfg = self._config(scenarios=["good", "bad"], style_only=False)
        runner = EvalRunner(cfg)
        with patch.dict(
            "helm_eval_lex_quirks.runner.SCENARIO_REGISTRY",
            {"good": GoodScenario, "bad": BadScenario},
        ), patch.dict(
            "helm_eval_lex_quirks.runner.DEFAULT_SCENARIO_CONFIGS",
            {
                "good": ScenarioConfig("good", n_samples=1, n_few_shot=0),
                "bad": ScenarioConfig("bad", n_samples=1, n_few_shot=0),
            },
        ), patch.object(
            runner,
            "_eval_scenario",
            return_value=object(),
        ) as eval_mock:
            with self.assertLogs("helm_eval_lex_quirks.runner", level="ERROR") as logs:
                result = runner._run_model(SubjectClient(), "gpt-4o-mini")
        self.assertIn("good", result.scenario_results)
        self.assertNotIn("bad", result.scenario_results)
        self.assertEqual(eval_mock.call_count, 1)
        self.assertIn("Failed to load bad", "\n".join(logs.output))

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

    def test_run_tic_suite_style_samples_increment_appends_new_records(self):
        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

            def __init__(self):
                self.calls = 0

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                self.calls += 1
                class Resp:
                    completion = "plain completion"
                    logprobs = None
                return Resp()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(
                output_dir=tmpdir,
                style_samples_increment=True,
                style_only=True,
                style_suites=[],
            )
            runner = EvalRunner(cfg)
            client = SubjectClient()
            suite = self._suite()

            first = runner._run_tic_suite(
                client=client,
                model_name="subject-model",
                suite=suite,
                samples=1,
                progress_label="unit-suite",
            )
            second = runner._run_tic_suite(
                client=client,
                model_name="subject-model",
                suite=suite,
                samples=1,
                progress_label="unit-suite",
            )

            self.assertEqual(client.calls, 4)
            self.assertEqual(first["analysis"]["neutral"]["n_outputs"], 2)
            self.assertEqual(second["analysis"]["neutral"]["n_outputs"], 4)

            with open(second["checkpoint_path"]) as f:
                records = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(len(records), 4)
        by_prompt = {}
        for rec in records:
            by_prompt.setdefault(rec["prompt_idx"], set()).add(rec["sample_idx"])
        self.assertEqual(by_prompt[0], {0, 1})
        self.assertEqual(by_prompt[1], {0, 1})

    def test_run_tic_suite_style_samples_increment_gap_does_not_backfill(self):
        class SubjectClient:
            total_tokens = {"input": 0, "output": 0}
            total_cost_usd = 0.0

            def __init__(self):
                self.calls = 0

            def complete(self, prompt, max_tokens=1, logprobs=False, system=None):
                self.calls += 1

                class Resp:
                    completion = "new completion"
                    logprobs = None

                return Resp()

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(
                output_dir=tmpdir,
                style_samples_increment=True,
                style_only=True,
                style_suites=[],
            )
            runner = EvalRunner(cfg)
            suite = self._suite()
            client = SubjectClient()

            ckpt_dir = os.path.join(tmpdir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "subject-model__unit_suite.jsonl")
            seeded_records = [
                {
                    "schema_version": 1,
                    "model": "subject-model",
                    "suite": "unit_suite",
                    "condition": "neutral",
                    "prompt_id": "p0",
                    "task_type": "general",
                    "prompt_idx": 0,
                    "sample_idx": 0,
                    "prompt": "Write a short note.",
                    "completion": "old p0 s0",
                    "detectors": {},
                },
                {
                    "schema_version": 1,
                    "model": "subject-model",
                    "suite": "unit_suite",
                    "condition": "neutral",
                    "prompt_id": "p0",
                    "task_type": "general",
                    "prompt_idx": 0,
                    "sample_idx": 2,
                    "prompt": "Write a short note.",
                    "completion": "old p0 s2",
                    "detectors": {},
                },
                {
                    "schema_version": 1,
                    "model": "subject-model",
                    "suite": "unit_suite",
                    "condition": "neutral",
                    "prompt_id": "p1",
                    "task_type": "update",
                    "prompt_idx": 1,
                    "sample_idx": 0,
                    "prompt": "Write an update.",
                    "completion": "old p1 s0",
                    "detectors": {},
                },
                {
                    "schema_version": 1,
                    "model": "subject-model",
                    "suite": "unit_suite",
                    "condition": "neutral",
                    "prompt_id": "p1",
                    "task_type": "update",
                    "prompt_idx": 1,
                    "sample_idx": 2,
                    "prompt": "Write an update.",
                    "completion": "old p1 s2",
                    "detectors": {},
                },
            ]
            with open(ckpt_path, "w") as f:
                for rec in seeded_records:
                    f.write(json.dumps(rec) + "\n")

            result = runner._run_tic_suite(
                client=client,
                model_name="subject-model",
                suite=suite,
                samples=1,
                progress_label="unit-suite",
            )

            self.assertEqual(client.calls, 2)
            with open(result["checkpoint_path"]) as f:
                records = [json.loads(line) for line in f if line.strip()]

        by_prompt = {}
        for rec in records:
            by_prompt.setdefault(rec["prompt_idx"], set()).add(rec["sample_idx"])
        self.assertEqual(by_prompt[0], {0, 2, 3})
        self.assertEqual(by_prompt[1], {0, 2, 3})


    def test_gemini_self_harm_safety_tokens_aggregated(self):
        """Safety-off client tokens and cost are merged into the final result.

        _run_tic_suite is mocked so this test focuses purely on the token/cost
        aggregation in _run_model, not on suite execution (which would require a
        judge model and real LLM calls).
        """
        import tempfile

        class _Client:
            def __init__(self, tokens, cost):
                self._tokens = tokens
                self._cost = cost

            @property
            def total_tokens(self):
                return dict(self._tokens)

            @property
            def total_cost_usd(self):
                return self._cost

        default_client = _Client({"input": 10, "output": 5}, 0.01)
        safety_client  = _Client({"input": 20, "output": 8}, 0.02)

        created = iter([default_client, safety_client])

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._config(
                output_dir=tmpdir,
                style_only=True,
                style_suites=["self_harm_safety"],
                models=["gemini-2.0-flash"],
                google_api_key="fake-key",
            )
            runner = EvalRunner(cfg)

            # create_client is called twice: once for the main client in run(), once
            # for the safety-off client inside _run_model for self_harm_safety.
            # _run_tic_suite is stubbed out so we don't need a real judge model.
            with patch("helm_eval_lex_quirks.runner.create_client", side_effect=created), \
                 patch.object(EvalRunner, "_run_tic_suite", return_value={}):
                results = runner.run()

        result = results["gemini-2.0-flash"]
        self.assertEqual(result.token_usage, {"input": 30, "output": 13})
        self.assertAlmostEqual(result.total_cost_usd, 0.03)


if __name__ == "__main__":
    unittest.main()
