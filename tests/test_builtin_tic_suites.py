import unittest

from compass.detectors.suites import SUITE_REGISTRY as COMPASS_SUITE_REGISTRY

from helm_eval_lex_quirks.config import EvalConfig
from helm_eval_lex_quirks.metrics.builtin_tic_suites import (
    SELF_HARM_SAFETY_SUITE,
    STYLE_SUITE_REGISTRY,
    expand_suite_names,
)
from helm_eval_lex_quirks.metrics.tic_suite import LLMJudgeDetector, TicSuite
from helm_eval_lex_quirks.runner import EvalRunner


class BuiltinTicSuiteTests(unittest.TestCase):
    def test_expand_suite_names_all_excludes_linkedin(self):
        result = expand_suite_names(["all"])

        self.assertIn("lexical", result)
        self.assertIn("punctuation", result)
        self.assertIn("formatting", result)
        self.assertIn("tone", result)
        self.assertNotIn("linkedin", result)
        self.assertNotIn("llm_judge_tone", result)
        self.assertNotIn("self_harm_safety", result)
        self.assertNotIn("rest_hypothesis", result)
        self.assertNotIn("rest_cai_validation", result)

    def test_expand_suite_names_keeps_rest_suites_explicit(self):
        result = expand_suite_names(["rest_hypothesis", "rest_cai_validation"])

        self.assertEqual(result, ["rest_hypothesis", "rest_cai_validation"])

    def test_expand_suite_names_keeps_mild_insult_suite_explicit(self):
        result = expand_suite_names(["mild_insult_reactivity"])

        self.assertEqual(result, ["mild_insult_reactivity"])

    def test_expand_suite_names_keeps_agentic_side_effects_explicit(self):
        result = expand_suite_names(["agentic_side_effects"])

        self.assertEqual(result, ["agentic_side_effects"])

    def test_compass_owned_suites_come_from_compass_registry(self):
        compass_owned_keys = {
            "self_harm_safety",
            "rest_hypothesis",
            "rest_cai_validation",
            "rest_offtask",
            "rest_intensity",
            "mild_insult_reactivity",
            "agentic_side_effects",
            "task_focus",
            "truthfulness",
        }

        for key in compass_owned_keys:
            with self.subTest(key=key):
                self.assertIs(STYLE_SUITE_REGISTRY[key], COMPASS_SUITE_REGISTRY[key])

    def test_expand_suite_names_keeps_self_harm_suite_explicit(self):
        result = expand_suite_names(["self_harm_safety"])

        self.assertEqual(result, ["self_harm_safety"])

    def test_self_harm_suite_structure(self):
        suite = SELF_HARM_SAFETY_SUITE

        self.assertEqual(suite.name, "self_harm_safety")
        self.assertEqual(suite.baseline_condition, "neutral")
        self.assertEqual(len(suite.conditions), 1)
        self.assertEqual(suite.conditions[0].name, "neutral")
        self.assertEqual(len(suite.prompts), 6)
        self.assertIn("benign_control", {prompt.id for prompt in suite.prompts})
        self.assertIn("benign_control", {prompt.task_type for prompt in suite.prompts})
        self.assertEqual(len(suite.detectors), 2)
        self.assertEqual(
            {detector.name for detector in suite.detectors},
            {"unsafe_assistance", "missing_supportive_redirect"},
        )
        for detector in suite.detectors:
            self.assertIsInstance(detector, LLMJudgeDetector)
            self.assertEqual(detector.hit_threshold, 0.5)

    def test_expand_suite_names_rejects_unknown_suite(self):
        with self.assertRaisesRegex(ValueError, "Unknown style suite"):
            expand_suite_names(["missing"])

    def test_enabled_tic_suites_deduplicates_linkedin(self):
        cfg = EvalConfig(
            api_key="dummy",
            run_linkedin_style=True,
            style_suites=["linkedin"],
            linkedin_samples=2,
            style_samples=3,
        )

        suites = EvalRunner(cfg)._enabled_tic_suites()

        self.assertEqual(sum(1 for key, _suite, _samples in suites if key == "linkedin"), 1)
        self.assertEqual(suites[0][2], 2)

    def test_enabled_tic_suites_rejects_unknown_suite(self):
        cfg = EvalConfig(api_key="dummy", style_suites=["missing"])

        with self.assertRaisesRegex(ValueError, "Unknown style suite"):
            EvalRunner(cfg)._enabled_tic_suites()

    def test_registry_integrity(self):
        for key, suite in STYLE_SUITE_REGISTRY.items():
            with self.subTest(key=key):
                self.assertIsInstance(suite, TicSuite)
                self.assertTrue(suite.prompts)
                self.assertTrue(suite.conditions)
                self.assertTrue(suite.detectors)
                self.assertTrue(suite.baseline_condition)
                self.assertTrue(all(prompt.task_type for prompt in suite.prompts))
                self.assertEqual(len({detector.name for detector in suite.detectors}), len(suite.detectors))


if __name__ == "__main__":
    unittest.main()
