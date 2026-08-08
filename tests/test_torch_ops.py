import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechai_model_selection import (
    fisher_pullback,
    pullback_geometry,
    residual_jacobian,
    sensitivity_gramian,
    wasserstein_pullback_1d,
)

torch.set_default_dtype(torch.float64)


class TorchOperationTests(unittest.TestCase):
    def test_linear_solution_map(self):
        design = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [0.5, 0.2]])
        theta = torch.tensor([0.2, -0.4])
        jacobian = residual_jacobian(lambda value: design @ value, theta)
        gramian, returned = sensitivity_gramian(lambda value: design @ value, theta)
        self.assertTrue(torch.allclose(jacobian, design))
        self.assertTrue(torch.allclose(returned, design))
        self.assertTrue(torch.allclose(gramian, design.mT @ design))

    def test_fisher_and_common_interface_match_sensitivity_gramian(self):
        design = torch.tensor([[1.0, 2.0], [-1.0, 3.0], [0.5, 0.2]])
        theta = torch.tensor([0.2, -0.4])
        precision = torch.diag(torch.tensor([2.0, 1.0, 0.5]))
        expected, _ = sensitivity_gramian(
            lambda value: design @ value,
            theta,
            observation_precision=precision,
        )
        direct, _ = fisher_pullback(
            lambda value: design @ value,
            theta,
            observation_precision=precision,
        )
        common, _ = pullback_geometry(
            lambda value: design @ value,
            theta,
            metric="fisher",
            observation_precision=precision,
        )
        self.assertTrue(torch.allclose(expected, direct))
        self.assertTrue(torch.allclose(expected, common))

    def test_wasserstein_quantile_pullback(self):
        design = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        theta = torch.tensor([0.2, -0.4])
        weights = torch.tensor([1.0, 2.0, 1.0])
        gramian, jacobian = wasserstein_pullback_1d(
            lambda value: design @ value,
            theta,
            weights=weights,
        )
        normalized = weights / weights.sum()
        expected = design.mT @ (normalized.unsqueeze(1) * design)
        self.assertTrue(torch.allclose(jacobian, design))
        self.assertTrue(torch.allclose(gramian, expected))


if __name__ == "__main__":
    unittest.main()
