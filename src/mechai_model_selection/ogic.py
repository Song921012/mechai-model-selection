"""Information criteria derived from predictive risk and local evidence."""

from __future__ import annotations

import warnings
import math

import torch

from .geometry import PullbackGeometry


def generalized_optimism(
    risk_curvature: torch.Tensor,
    score_covariance: torch.Tensor,
    reference_metric: torch.Tensor,
    regularization: float = 1.0,
) -> float:
    """Return ``tr((A + regularization R)^-1 B)`` for a penalized M-estimator."""
    if regularization < 0:
        raise ValueError("regularization must be nonnegative")
    a = torch.as_tensor(risk_curvature)
    b = torch.as_tensor(score_covariance, dtype=a.dtype, device=a.device)
    r = torch.as_tensor(reference_metric, dtype=a.dtype, device=a.device)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape or r.shape != a.shape:
        raise ValueError(
            "risk_curvature, score_covariance, and reference_metric must be equal square matrices"
        )
    system = 0.5 * (a + a.mT) + regularization * 0.5 * (r + r.mT)
    solution = torch.linalg.solve(system, 0.5 * (b + b.mT))
    return float(torch.trace(solution))

def gic_effective(
    deviance: float,
    geometry: PullbackGeometry,
    *,
    penalty_factor: float,
) -> float:
    """Fit plus an observable-dimension penalty."""
    if penalty_factor < 0:
        raise ValueError("penalty_factor must be nonnegative")
    return float(deviance) + float(penalty_factor) * geometry.effective_dimension


def gic_volume(
    deviance: float,
    geometry: PullbackGeometry,
    *,
    penalty_factor: float,
    volume_weight: float = 1.0,
) -> float:
    """Effective-dimension GIC plus invariant relative log-volume."""
    if volume_weight < 0:
        raise ValueError("volume_weight must be nonnegative")
    return gic_effective(
        deviance, geometry, penalty_factor=penalty_factor,
    ) + float(volume_weight) * geometry.complexity


def gic_predictive(deviance: float, geometry: PullbackGeometry) -> float:
    """Predictive-risk criterion ``deviance + 2 effective_dimension``."""
    return gic_effective(deviance, geometry, penalty_factor=2.0)


def ogic_predictive(deviance: float, geometry: PullbackGeometry) -> float:
    """Deprecated alias for :func:`gic_predictive`."""
    warnings.warn("ogic_predictive is deprecated; use gic_predictive", DeprecationWarning, stacklevel=2)
    return gic_predictive(deviance, geometry)


def gic_laplace(
    deviance: float,
    geometry: PullbackGeometry,
    *,
    prior_energy: float = 0.0,
) -> float:
    """Local Gaussian evidence score using the relative log-volume."""
    return float(deviance) + float(prior_energy) + geometry.complexity


def gic_evidence(
    deviance: float,
    geometry: PullbackGeometry,
    *,
    prior_energy: float = 0.0,
) -> float:
    """Local-evidence criterion derived from a normalized Laplace expansion."""
    return gic_laplace(deviance, geometry, prior_energy=prior_energy)


def gic_bic_approximation(
    deviance: float,
    geometry: PullbackGeometry,
    n_observations: int,
) -> float:
    """Leading-order BIC-type approximation using finite-resolution rank."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive")
    return gic_effective(deviance, geometry, penalty_factor=math.log(n_observations))


def ogic_evidence(
    deviance: float,
    geometry: PullbackGeometry,
    *,
    prior_energy: float = 0.0,
) -> float:
    """Deprecated alias for :func:`gic_evidence`."""
    warnings.warn("ogic_evidence is deprecated; use gic_evidence", DeprecationWarning, stacklevel=2)
    return gic_evidence(deviance, geometry, prior_energy=prior_energy)
