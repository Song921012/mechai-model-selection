import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechai_model_selection import (
    ObservableGeometry,
    aggregate_fold_deviance,
    aic,
    aicc,
    bic,
    coverage_summary,
    criterion_weights,
    geometric_confidence_region,
    gic_effective,
    gic_volume,
    model_average,
    predictive_deviance,
    quotient_rank,
    rolling_origin_splits,
    stacking_weights,
    waic,
)


class CriteriaTests(unittest.TestCase):
    def test_classical_criteria(self):
        self.assertEqual(aic(-10.0, 3), 26.0)
        self.assertAlmostEqual(bic(-10.0, 3, 100), 20.0 + 3.0 * math.log(100))
        self.assertEqual(aicc(-10.0, 3, 4), math.inf)

    def test_waic_matches_manual_formula(self):
        ll = torch.tensor([[-1.0, -2.0], [-1.2, -1.8], [-0.9, -2.1]])
        result = waic(ll)
        lppd = torch.logsumexp(ll, dim=0) - math.log(3)
        p = torch.var(ll, dim=0, unbiased=True)
        expected = -2.0 * torch.sum(lppd - p)
        self.assertAlmostEqual(result["waic"], float(expected), places=6)

    def test_weights_are_normalized(self):
        weights = criterion_weights(torch.tensor([10.0, 12.0, 14.0]))
        self.assertTrue(torch.allclose(weights.sum(), torch.tensor(1.0)))
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_geometric_criteria(self):
        geometry = ObservableGeometry(torch.tensor([9.0, 1.0, 0.0], dtype=torch.float64))
        dimension = 9.0 / 10.0 + 1.0 / 2.0
        complexity = math.log(10.0) + math.log(2.0)
        self.assertAlmostEqual(
            gic_effective(12.0, geometry, penalty_factor=3.0),
            12.0 + 3.0 * dimension,
        )
        self.assertAlmostEqual(
            gic_volume(12.0, geometry, penalty_factor=3.0, volume_weight=0.5),
            12.0 + 3.0 * dimension + 0.5 * complexity,
        )

    def test_model_average_variance_decomposition(self):
        result = model_average(
            torch.tensor([0.0, 2.0]), torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5])
        )
        self.assertAlmostEqual(float(result["mean"]), 1.0)
        self.assertAlmostEqual(float(result["within"]), 1.0)
        self.assertAlmostEqual(float(result["between"]), 1.0)
        self.assertAlmostEqual(float(result["variance"]), 2.0)

    def test_model_average_prediction_interval(self):
        result = model_average(
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            torch.ones((2, 2)),
            torch.tensor([0.5, 0.5]),
            confidence=0.9,
        )
        self.assertEqual(result["lower"].shape, torch.Size([2]))
        self.assertTrue(torch.all(result["lower"] < result["mean"]))
        self.assertTrue(torch.all(result["upper"] > result["mean"]))

    def test_rank_aware_confidence_and_coverage(self):
        information = torch.diag(torch.tensor([9.0, 0.0], dtype=torch.float64))
        reference = torch.eye(2, dtype=torch.float64)
        self.assertEqual(quotient_rank(information, reference), 1)
        region = geometric_confidence_region(
            torch.zeros(2, dtype=torch.float64),
            information,
            reference,
            confidence=0.95,
        )
        self.assertEqual(region["rank"], 1)
        self.assertAlmostEqual(float(region["radius_squared"]), 3.841458820694124, places=10)
        self.assertAlmostEqual(float(region["covariance"][0, 0]), 1.0 / 9.0)
        self.assertAlmostEqual(float(region["covariance"][1, 1]), 0.0)
        summary = coverage_summary(torch.tensor([True, True, False, True]))
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["coverage"], 0.75)

    def test_predictive_fold_scores(self):
        self.assertEqual(predictive_deviance(torch.tensor([-1.0, -2.0])), 6.0)
        self.assertEqual(
            aggregate_fold_deviance([torch.tensor([-1.0]), torch.tensor([-2.0])]),
            6.0,
        )

    def test_rolling_origin_and_stacking(self):
        splits = rolling_origin_splits(20)
        endpoints = [(train.stop, valid.start, valid.stop) for train, valid in splits]
        self.assertEqual(endpoints, [(10, 10, 12), (12, 12, 14), (14, 14, 16)])
        log_scores = torch.tensor([[-0.1, -0.2, -0.1], [-2.0, -1.5, -2.0]])
        weights = stacking_weights(log_scores)
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertGreater(float(weights[0]), float(weights[1]))


if __name__ == "__main__":
    unittest.main()
