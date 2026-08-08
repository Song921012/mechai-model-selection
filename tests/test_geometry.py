import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechai_model_selection.geometry import (
    ObservableGeometry,
    block_reference_metric,
    generalized_spectrum,
    geometry_sensitivity_grid,
    observable_complexity,
    observable_dimension,
)
from mechai_model_selection.ogic import ogic_evidence

torch.set_default_dtype(torch.float64)


class GeometryTests(unittest.TestCase):
    def test_block_reference_and_sensitivity_grid(self):
        reference = block_reference_metric((2, 1), (1.0, 4.0))
        self.assertTrue(
            torch.allclose(torch.diag(reference), torch.tensor([1.0, 1.0, 4.0]))
        )
        information = torch.diag(torch.tensor([9.0, 1.0, 4.0]))
        grid = geometry_sensitivity_grid(
            information, [reference, torch.eye(3)], torch.tensor([1.0, 10.0])
        )
        self.assertEqual(grid["dimension"].shape, torch.Size([2, 2]))
        self.assertGreater(float(grid["dimension"][0, 0]), float(grid["dimension"][0, 1]))

    def test_generalized_spectrum_is_congruence_invariant(self):
        g = torch.tensor([[8.0, 1.0], [1.0, 2.0]])
        r = torch.tensor([[2.0, 0.2], [0.2, 1.0]])
        change = torch.tensor([[2.0, -0.4], [0.3, 1.5]])
        expected = generalized_spectrum(g, r)
        transformed = generalized_spectrum(change.mT @ g @ change, change.mT @ r @ change)
        self.assertTrue(torch.allclose(expected, transformed, rtol=1e-8, atol=1e-8))

    def test_rank_deficient_limit_and_complexity_derivative(self):
        values = torch.tensor([12.0, 3.0, 0.0])
        self.assertTrue(torch.allclose(observable_dimension(values, 1e-9), torch.tensor(2.0)))
        tau = 0.7
        eps = 1e-5
        upper = observable_complexity(values, tau * torch.exp(torch.tensor(eps)))
        lower = observable_complexity(values, tau * torch.exp(torch.tensor(-eps)))
        derivative = (upper - lower) / (2 * eps)
        self.assertTrue(torch.allclose(-derivative, observable_dimension(values, tau), rtol=2e-5, atol=2e-5))

    def test_observable_geometry_properties(self):
        geometry = ObservableGeometry(torch.tensor([9.0, 1.0, 0.0]), resolution=1.0)
        self.assertAlmostEqual(geometry.effective_dimension, 1.4, places=12)
        self.assertGreater(geometry.complexity, 0)

    def test_evidence_score_is_invariant_with_transformed_reference(self):
        g = torch.tensor([[7.0, 0.6], [0.6, 1.5]])
        r = torch.tensor([[1.5, 0.1], [0.1, 0.9]])
        change = torch.tensor([[1.8, 0.3], [-0.2, 0.7]])
        original = ObservableGeometry.from_matrices(g, r)
        transformed = ObservableGeometry.from_matrices(change.mT @ g @ change, change.mT @ r @ change)
        self.assertAlmostEqual(
            ogic_evidence(20.0, original, prior_energy=1.2),
            ogic_evidence(20.0, transformed, prior_energy=1.2),
            places=8,
        )


if __name__ == "__main__":
    unittest.main()
