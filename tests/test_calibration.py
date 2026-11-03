import unittest

from compass_experiments.metrics.calibration import compute_ece


class CalibrationTests(unittest.TestCase):
    def test_shared_boundary_counts_once(self):
        self.assertAlmostEqual(compute_ece([0.5], [True]), 0.5)
        self.assertAlmostEqual(compute_ece([0.1], [True]), 0.9)

    def test_endpoints_and_empty_input(self):
        self.assertEqual(compute_ece([0.0, 1.0], [False, True]), 0.0)
        self.assertEqual(compute_ece([0.0, 1.0], [True, False]), 1.0)
        self.assertEqual(compute_ece([], []), 0.0)

    def test_weighted_bins(self):
        # First bin: error .25, weight 2/3. Last bin: error .1, weight 1/3.
        self.assertAlmostEqual(
            compute_ece([0.2, 0.3, 0.9], [False, True, True], n_bins=2), 0.2
        )

    def test_rejects_invalid_inputs(self):
        for values, labels, bins in [
            ([0.5], [], 10), ([float("nan")], [True], 10),
            ([float("inf")], [True], 10), ([-0.1], [False], 10),
            ([1.1], [True], 10), ([], [], 0), ([], [], 1.5),
        ]:
            with self.subTest(values=values, bins=bins), self.assertRaises(ValueError):
                compute_ece(values, labels, bins)


class ChoiceProbabilityTests(unittest.TestCase):
    @staticmethod
    def candidates(entries):
        import math
        from types import SimpleNamespace
        return [SimpleNamespace(token=token, logprob=math.log(probability))
                for token, probability in entries]

    def test_sum_distinct_spellings_independent_of_candidate_order(self):
        from compass_experiments.metrics.calibration import extract_choice_probs
        entries = self.candidates([('B', .3), ('A', .2), (' A', .2), ('C', .2), ('D', .1)])
        for candidates in [entries, list(reversed(entries))]:
            probabilities, coverage = extract_choice_probs(candidates, list('ABCD'))
            self.assertEqual(coverage, 1)
            for choice, expected in zip('ABCD', [.4, .3, .2, .1]):
                self.assertAlmostEqual(probabilities[choice], expected)

    def test_confidence_conditions_on_observed_answer_mass(self):
        from compass_experiments.metrics.calibration import extract_choice_probs
        probabilities, coverage = extract_choice_probs(self.candidates([('A', .1), (' a', .1), ('B', .3), ('Other', .5)]), ['A', 'B'])
        self.assertEqual(coverage, 1)
        self.assertAlmostEqual(probabilities['A'], .4)

    def test_missing_candidates_and_labels_do_not_have_full_coverage(self):
        from compass_experiments.metrics.calibration import extract_choice_probs
        self.assertEqual(extract_choice_probs([], ['A', 'B']), ({'A': .5, 'B': .5}, 0))
        probabilities, coverage = extract_choice_probs(self.candidates([('A', .2), (' a', .3)]), ['A', 'B'])
        self.assertEqual(coverage, .5)
        self.assertAlmostEqual(sum(probabilities.values()), 1)

    def test_duplicate_candidate_is_not_extra_mass(self):
        from compass_experiments.metrics.calibration import extract_choice_probs
        entries = self.candidates([('A', .2), ('A', .2), ('B', .3)])
        probabilities, _ = extract_choice_probs(entries, ['A', 'B'])
        self.assertAlmostEqual(probabilities['A'], .4)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            extract_choice_probs(self.candidates([('A', .2), ('A', .3)]), ['A', 'B'])

    def test_very_small_probabilities_and_saved_candidates_roundtrip(self):
        import json
        from types import SimpleNamespace
        from compass_experiments.metrics.calibration import extract_choice_probs, serialize_top_logprobs
        candidates = [SimpleNamespace(token=token, logprob=-10000) for token in ['A', ' A', 'B']]
        saved = json.loads(json.dumps(serialize_top_logprobs(candidates)))
        probabilities, coverage = extract_choice_probs(saved, ['A', 'B'])
        self.assertEqual(coverage, 1)
        self.assertAlmostEqual(probabilities['A'], 2/3)
        self.assertEqual(extract_choice_probs(candidates, ['A', 'B']), (probabilities, coverage))

    def test_invalid_probabilities_and_choice_lists_fail(self):
        from compass_experiments.metrics.calibration import extract_choice_probs
        for value in [float('nan'), float('inf'), -float('inf'), .1, True, None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                extract_choice_probs([{'token': 'A', 'logprob': value}], ['A', 'B'])
        for choices in [[], ['A', 'A']]:
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                extract_choice_probs([], choices)
