"""Minimal observable-geometry model comparison."""

import torch

from mechai_model_selection import (
    ObservableGeometry,
    criterion_weights,
    ogic_evidence,
    ogic_predictive,
)


information = torch.tensor([[12.0, 1.5], [1.5, 0.4]], dtype=torch.float64)
reference = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
geometry = ObservableGeometry.from_matrices(information, reference)

deviance = 31.2
prior_energy = 0.8
print("observable dimension:", geometry.effective_dimension)
print("observable complexity:", geometry.complexity)
print("OGIC-P:", ogic_predictive(deviance, geometry))
print("OGIC-E:", ogic_evidence(deviance, geometry, prior_energy=prior_energy))
print("weights:", criterion_weights(torch.tensor([35.0, 37.0, 42.0])))
