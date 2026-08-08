"""Observable-Geometric Information Criteria (OGIC)."""

from __future__ import annotations

import warnings

from .geometry import ObservableGeometry


def gic_effective(
    deviance: float,
    geometry: ObservableGeometry,
    *,
    penalty_factor: float,
) -> float:
    """Fit plus an observable-dimension penalty."""
    if penalty_factor < 0:
        raise ValueError("penalty_factor must be nonnegative")
    return float(deviance) + float(penalty_factor) * geometry.effective_dimension


def gic_volume(
    deviance: float,
    geometry: ObservableGeometry,
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


def ogic_predictive(deviance: float, geometry: ObservableGeometry) -> float:
    """Predictive criterion with an effective-degrees-of-freedom penalty."""
    return gic_effective(deviance, geometry, penalty_factor=2.0)


def gic_laplace(
    deviance: float,
    geometry: ObservableGeometry,
    *,
    prior_energy: float = 0.0,
) -> float:
    """Local Gaussian evidence score using the relative log-volume."""
    return float(deviance) + float(prior_energy) + geometry.complexity


def ogic_evidence(
    deviance: float,
    geometry: ObservableGeometry,
    *,
    prior_energy: float = 0.0,
) -> float:
    """Deprecated alias for :func:`gic_laplace`."""
    warnings.warn("ogic_evidence is deprecated; use gic_laplace", DeprecationWarning, stacklevel=2)
    return gic_laplace(deviance, geometry, prior_energy=prior_energy)
