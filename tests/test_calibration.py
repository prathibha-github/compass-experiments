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
