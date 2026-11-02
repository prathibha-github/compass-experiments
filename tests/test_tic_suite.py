import unittest

from compass_experiments.metrics.tic_suite import (
    CharacterCountDetector,
    EmojiDetector,
    LLMJudgeDetector,
    PhraseSetDetector,
    RegexDetector,
    StyleCondition,
    StylePrompt,
    TicSuite,
    analyze_text,
    compute_uplift,
    summarize_detector_records,
    summarize_outputs,
    wilson_interval,
)


class TicSuiteDetectorTests(unittest.TestCase):
    def test_emoji_detector(self):
        result = EmojiDetector().detect("hello 😀 world")

        self.assertEqual(result.count, 1)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.hit)
        self.assertEqual(result.metadata, {})

    def test_phrase_set_detector_counts_case_insensitive_phrases(self):
        detector = PhraseSetDetector(name="jargon", phrases={"unlock", "north star"})

        result = detector.detect("Unlock the roadmap around our north star.")

        self.assertEqual(result.count, 2)
        self.assertTrue(result.hit)
        self.assertEqual(result.metadata["phrases"], {"unlock": 1, "north star": 1})

    def test_phrase_set_detector_uses_word_boundaries(self):
        detector = PhraseSetDetector(name="tone", phrases={"may", "valid", "align"})

        result = detector.detect("Maybe invalidate this realignment, but may align later.")

        self.assertEqual(result.count, 2)
        self.assertEqual(result.metadata["phrases"], {"may": 1, "align": 1})

    def test_regex_detector_records_matches(self):
        detector = RegexDetector(name="not_just_but", pattern=r"\bnot\s+just\b.*\bbut\b")

        result = detector.detect("This is not just useful, but durable.")

        self.assertEqual(result.count, 1)
        self.assertTrue(result.hit)
        self.assertEqual(result.metadata["matches"], ["not just useful, but"])

    def test_character_count_detector(self):
        detector = CharacterCountDetector(name="em_dash", character="—", hit_threshold=2)

        result = detector.detect("One — two — three")

        self.assertEqual(result.count, 2)
        self.assertTrue(result.hit)
        self.assertEqual(result.metadata, {"character": "—"})

    def test_llm_judge_detector_parses_json_response(self):
        detector = LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy.")

        result = detector.parse_response(
            '{"score": 0.8, "hit": true, "confidence": 0.7, "rationale": "too agreeable"}'
        )

        self.assertEqual(result.count, 1)
        self.assertEqual(result.score, 0.8)
        self.assertTrue(result.hit)
        self.assertEqual(result.metadata["confidence"], 0.7)

    def test_llm_judge_detector_records_parse_failure(self):
        detector = LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy.")

        result = detector.parse_response("not json")

        self.assertEqual(result.count, 0)
        self.assertFalse(result.hit)
        self.assertTrue(result.metadata["parse_error"])

    def test_analyze_text_uses_judge_function_for_llm_judge_detector(self):
        suite = TicSuite(
            name="judge_suite",
            prompts=[StylePrompt(id="p0", text="Respond.")],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy.")],
            baseline_condition="neutral",
        )

        results = analyze_text(
            "You are absolutely right about skipping tests.",
            suite,
            judge_fn=lambda _prompt, _max_tokens: (
                '{"score": 1.0, "hit": true, "confidence": 0.9, "rationale": "no pushback"}'
            ),
        )

        self.assertTrue(results["sycophancy"].hit)

    def test_analyze_text_requires_judge_function_for_llm_judge_detector(self):
        suite = TicSuite(
            name="judge_suite",
            prompts=[StylePrompt(id="p0", text="Respond.")],
            conditions=[StyleCondition(name="neutral", system_prompt="You are helpful.")],
            detectors=[LLMJudgeDetector(name="sycophancy", rubric="Score sycophancy.")],
            baseline_condition="neutral",
        )

        with self.assertRaisesRegex(ValueError, "requires a judge function"):
            analyze_text("Response", suite)


class TicSuiteAggregationTests(unittest.TestCase):
    def setUp(self):
        self.suite = TicSuite(
            name="test_suite",
            prompts=[StylePrompt(id="p0", text="Write an update.")],
            conditions=[
                StyleCondition(name="neutral", system_prompt="You are helpful."),
                StyleCondition(name="persona", system_prompt="Use emojis."),
            ],
            detectors=[
                EmojiDetector(name="emoji"),
                PhraseSetDetector(name="jargon", phrases={"unlock"}),
            ],
            baseline_condition="neutral",
        )

    def test_summarize_outputs_handles_empty_condition(self):
        summary = summarize_outputs({"neutral": []}, self.suite)

        self.assertEqual(summary["neutral"]["n_outputs"], 0)
        self.assertFalse(summary["neutral"]["low_sample"])
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["mean_count"], 0.0)
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["pct_hit"], 0.0)
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["hit_count"], 0)
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["pct_hit_ci_low"], 0.0)
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["pct_hit_ci_high"], 0.0)

    def test_summarize_outputs_handles_single_condition(self):
        summary = summarize_outputs(
            {"persona": ["We can unlock this 😀", "No tic here."]},
            self.suite,
        )

        self.assertEqual(summary["persona"]["n_outputs"], 2)
        self.assertTrue(summary["persona"]["low_sample"])
        self.assertEqual(summary["persona"]["detectors"]["emoji"]["mean_count"], 0.5)
        self.assertEqual(summary["persona"]["detectors"]["emoji"]["pct_hit"], 50.0)
        self.assertEqual(summary["persona"]["detectors"]["emoji"]["hit_count"], 1)
        self.assertTrue(summary["persona"]["detectors"]["emoji"]["low_sample"])
        self.assertGreater(summary["persona"]["detectors"]["emoji"]["pct_hit_ci_high"], 50.0)
        self.assertEqual(summary["persona"]["detectors"]["jargon"]["mean_count"], 0.5)

    def test_summarize_detector_records_uses_stored_results(self):
        summary = summarize_detector_records(
            {
                "neutral": [
                    {
                        "emoji": {"count": 1, "hit": True},
                        "jargon": {"count": 0, "hit": False},
                    },
                    {
                        "emoji": {"count": 0, "hit": False},
                        "jargon": {"count": 2, "hit": True},
                    },
                ]
            },
            self.suite,
        )

        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["hit_count"], 1)
        self.assertEqual(summary["neutral"]["detectors"]["emoji"]["mean_count"], 0.5)
        self.assertEqual(summary["neutral"]["detectors"]["jargon"]["hit_count"], 1)
        self.assertEqual(summary["neutral"]["detectors"]["jargon"]["mean_count"], 1.0)

    def test_wilson_interval(self):
        interval = wilson_interval(hit_count=5, n_outputs=10)

        self.assertGreater(interval["low"], 0.0)
        self.assertLess(interval["low"], 50.0)
        self.assertGreater(interval["high"], 50.0)
        self.assertLess(interval["high"], 100.0)

    def test_wilson_interval_edge_cases(self):
        zero_hits = wilson_interval(hit_count=0, n_outputs=10)
        all_hits = wilson_interval(hit_count=10, n_outputs=10)
        one_of_one = wilson_interval(hit_count=1, n_outputs=1)

        self.assertEqual(zero_hits["low"], 0.0)
        self.assertLess(zero_hits["high"], 30.0)
        self.assertGreater(all_hits["low"], 70.0)
        self.assertEqual(all_hits["high"], 100.0)
        self.assertGreater(one_of_one["low"], 0.0)
        self.assertEqual(one_of_one["high"], 100.0)

    def test_compute_uplift_edge_cases(self):
        condition_results = {
            "neutral": {
                "detectors": {
                    "emoji": {"mean_count": 0.0, "pct_hit": 0.0},
                    "jargon": {"mean_count": 2.0, "pct_hit": 100.0},
                }
            },
            "persona": {
                "detectors": {
                    "emoji": {"mean_count": 1.0, "pct_hit": 50.0},
                    "jargon": {"mean_count": 1.0, "pct_hit": 50.0},
                }
            },
        }

        self.assertEqual(compute_uplift({}, "emoji"), {})
        self.assertEqual(
            compute_uplift(condition_results, "emoji", baseline="missing"),
            {"neutral": 0, "persona": 100},
        )
        self.assertEqual(
            compute_uplift(condition_results, "jargon", baseline="neutral"),
            {"neutral": 0.0, "persona": -50.0},
        )
        self.assertEqual(
            compute_uplift(condition_results, "missing_detector", baseline="neutral"),
            {"neutral": 0, "persona": 0},
        )


if __name__ == "__main__":
    unittest.main()
