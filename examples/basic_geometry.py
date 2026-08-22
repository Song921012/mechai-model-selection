"""Minimal first-principles geometric model comparison."""

import torch

from mechai_model_selection import (
    PullbackGeometry,
    criterion_weights,
    gic_bic_approximation,
    gic_evidence,
    gic_predictive,
)

information = torch.tensor([[12.0, 1.5], [1.5, 0.4]], dtype=torch.float64)
reference = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
geometry = PullbackGeometry.from_matrices(information, reference, resolution=1.0)

deviance = 31.2
prior_energy = 0.8
print("effective dimension:", geometry.effective_dimension)
print("relative log-volume:", geometry.relative_log_volume)
print("GIC-pred:", gic_predictive(deviance, geometry))
print("GIC-evid:", gic_evidence(deviance, geometry, prior_energy=prior_energy))
print("geometric BIC:", gic_bic_approximation(deviance, geometry, n_observations=80))
print("relative support:", criterion_weights(torch.tensor([35.0, 37.0, 42.0])))
