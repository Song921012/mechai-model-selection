"""Rank-aware local confidence summaries for observable geometry."""

from __future__ import annotations

import math

import torch
from scipy.stats import chi2

from .geometry import generalized_spectrum


def quotient_rank(
    information: torch.Tensor,
    reference_metric: torch.Tensor,
    *,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> int:
    """Numerical rank of the observable tangent space in reference units."""
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("rank tolerances must be nonnegative")
    values = generalized_spectrum(information, reference_metric)
    if values.numel() == 0:
        return 0
    threshold = max(absolute_tolerance, relative_tolerance * float(torch.max(values)))
    return int(torch.sum(values > threshold))


def _chi2_quantile(probability: float, degrees_of_freedom: int, dtype, device):
    if not 0 < probability < 1:
        raise ValueError("confidence level must lie strictly between zero and one")
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    return torch.tensor(
        float(chi2.ppf(probability, degrees_of_freedom)),
        dtype=dtype,
        device=device,
    )


def geometric_confidence_region(
    estimate: torch.Tensor,
    information: torch.Tensor,
    reference_metric: torch.Tensor,
    *,
    confidence: float = 0.95,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> dict[str, torch.Tensor | int | float]:
    """Return a local ellipsoid on the identifiable tangent quotient."""
    theta = torch.as_tensor(estimate)
    g = torch.as_tensor(information, dtype=theta.dtype, device=theta.device)
    r = torch.as_tensor(reference_metric, dtype=theta.dtype, device=theta.device)
    if g.shape != (theta.numel(), theta.numel()) or r.shape != g.shape:
        raise ValueError("estimate, information, and reference metric are incompatible")
    chol = torch.linalg.cholesky(0.5 * (r + r.mT))
    left = torch.linalg.solve_triangular(chol, 0.5 * (g + g.mT), upper=False)
    whitened = torch.linalg.solve_triangular(chol, left.mT, upper=False).mT
    values, vectors = torch.linalg.eigh(0.5 * (whitened + whitened.mT))
    maximum = float(torch.clamp(torch.max(values), min=0.0))
    threshold = max(absolute_tolerance, relative_tolerance * maximum)
    keep = values > threshold
    rank = int(torch.sum(keep))
    covariance = torch.zeros_like(g)
    projector = torch.zeros_like(g)
    radius_squared = torch.tensor(math.nan, dtype=theta.dtype, device=theta.device)
    if rank:
        identifiable = vectors[:, keep]
        whitened_covariance = (identifiable * (1.0 / values[keep])) @ identifiable.mT
        identity = torch.eye(theta.numel(), dtype=theta.dtype, device=theta.device)
        inv_chol = torch.linalg.solve_triangular(chol, identity, upper=False)
        covariance = inv_chol.mT @ whitened_covariance @ inv_chol
        projector = inv_chol.mT @ (identifiable @ identifiable.mT) @ chol.mT
        radius_squared = _chi2_quantile(confidence, rank, theta.dtype, theta.device)
    return {
        "center": theta,
        "covariance": 0.5 * (covariance + covariance.mT),
        "projector": projector,
        "rank": rank,
        "confidence": float(confidence),
        "radius_squared": radius_squared,
        "rank_threshold": float(threshold),
    }


def coverage_summary(
    covered: torch.Tensor,
    widths: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Summarize repeated-sampling coverage with a Wilson 95% interval."""
    values = torch.as_tensor(covered, dtype=torch.bool).flatten()
    if values.numel() == 0:
        raise ValueError("at least one coverage indicator is required")
    n = values.numel()
    successes = int(torch.sum(values))
    proportion = successes / n
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n)
    ) / denominator
    result: dict[str, float | int] = {
        "n": n,
        "successes": successes,
        "coverage": proportion,
        "wilson_lower": center - half,
        "wilson_upper": center + half,
    }
    if widths is not None:
        width_values = torch.as_tensor(widths, dtype=torch.float64).flatten()
        if width_values.numel() != n:
            raise ValueError("widths must match the coverage indicators")
        result["mean_width"] = float(torch.mean(width_values))
        result["median_width"] = float(torch.median(width_values))
    return result
