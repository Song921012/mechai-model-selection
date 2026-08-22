"""Coordinate-covariant pullback geometry for statistical model manifolds."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


def generalized_spectrum(
    information: torch.Tensor,
    reference_metric: torch.Tensor,
    *,
    jitter: float = 1e-10,
) -> torch.Tensor:
    """Generalized eigenvalues of ``information v = mu reference_metric v``.

    The computation whitens by a Cholesky factor of the reference metric. Both
    matrices are interpreted as covariant tensors and must therefore transform
    by congruence under a change of coordinates.
    """
    g = torch.as_tensor(information)
    r = torch.as_tensor(reference_metric, dtype=g.dtype, device=g.device)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or r.shape != g.shape:
        raise ValueError("information and reference_metric must be square and equal-sized")
    eye = torch.eye(g.shape[0], dtype=g.dtype, device=g.device)
    chol = torch.linalg.cholesky(_symmetric(r) + jitter * eye)
    left = torch.linalg.solve_triangular(chol, _symmetric(g), upper=False)
    whitened = torch.linalg.solve_triangular(chol, left.mT, upper=False).mT
    values = torch.linalg.eigvalsh(_symmetric(whitened))
    return torch.clamp(values, min=0.0)


def effective_dimension(eigenvalues: torch.Tensor, resolution: float = 1.0) -> torch.Tensor:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    values = torch.clamp(torch.as_tensor(eigenvalues), min=0.0)
    return torch.sum(values / (values + resolution))


def relative_log_volume(eigenvalues: torch.Tensor, resolution: float = 1.0) -> torch.Tensor:
    """Log determinant ratio relative to the declared reference geometry."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    values = torch.clamp(torch.as_tensor(eigenvalues), min=0.0)
    return torch.sum(torch.log1p(values / resolution))


def relative_volume(eigenvalues: torch.Tensor, resolution: float = 1.0) -> torch.Tensor:
    """Return the reference-normalized geometric volume exp(C_vol / 2)."""
    return torch.exp(0.5 * relative_log_volume(eigenvalues, resolution))


def observable_dimension(eigenvalues: torch.Tensor, resolution: float = 1.0) -> torch.Tensor:
    """Deprecated alias for :func:`effective_dimension`."""
    warnings.warn(
        "observable_dimension is deprecated; use effective_dimension",
        DeprecationWarning,
        stacklevel=2,
    )
    return effective_dimension(eigenvalues, resolution)


def observable_complexity(eigenvalues: torch.Tensor, resolution: float = 1.0) -> torch.Tensor:
    """Deprecated alias for relative_log_volume."""
    warnings.warn(
        "observable_complexity is deprecated; use relative_log_volume",
        DeprecationWarning,
        stacklevel=2,
    )
    return relative_log_volume(eigenvalues, resolution)


def resolution_profile(
    eigenvalues: torch.Tensor, resolutions: torch.Tensor
) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(eigenvalues)
    scales = torch.as_tensor(resolutions, dtype=values.dtype, device=values.device)
    if scales.ndim != 1 or torch.any(scales <= 0):
        raise ValueError("resolutions must be a positive one-dimensional tensor")
    ratios = values.unsqueeze(0) / scales.unsqueeze(1)
    return {
        "resolution": scales,
        "dimension": torch.sum(ratios / (1.0 + ratios), dim=1),
        "complexity": torch.sum(torch.log1p(ratios), dim=1),
    }


def block_reference_metric(
    block_sizes: Sequence[int],
    block_precisions: Sequence[float],
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Construct a block-isotropic reference metric."""
    if len(block_sizes) != len(block_precisions) or not block_sizes:
        raise ValueError("block_sizes and block_precisions must have equal nonzero length")
    if any(size <= 0 for size in block_sizes):
        raise ValueError("block sizes must be positive")
    if any(value <= 0 for value in block_precisions):
        raise ValueError("block precisions must be positive")
    diagonal = torch.cat(
        [
            torch.full((size,), value, dtype=dtype, device=device)
            for size, value in zip(block_sizes, block_precisions)
        ]
    )
    return torch.diag(diagonal)


def geometry_sensitivity_grid(
    information: torch.Tensor,
    reference_metrics: Sequence[torch.Tensor],
    resolutions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate dimension and relative volume over reference and resolution."""
    if not reference_metrics:
        raise ValueError("at least one reference metric is required")
    spectra = torch.stack(
        [generalized_spectrum(information, reference) for reference in reference_metrics]
    )
    scales = torch.as_tensor(resolutions, dtype=spectra.dtype, device=spectra.device)
    if scales.ndim != 1 or torch.any(scales <= 0):
        raise ValueError("resolutions must be a positive one-dimensional tensor")
    ratios = spectra.unsqueeze(1) / scales.reshape(1, -1, 1)
    return {
        "eigenvalues": spectra,
        "resolution": scales,
        "dimension": torch.sum(ratios / (1.0 + ratios), dim=-1),
        "complexity": torch.sum(torch.log1p(ratios), dim=-1),
    }


@dataclass(frozen=True)
class PullbackGeometry:
    """Generalized spectrum and finite-resolution model-manifold geometry."""

    eigenvalues: torch.Tensor
    resolution: float = 1.0

    @classmethod
    def from_matrices(
        cls,
        information: torch.Tensor,
        reference_metric: torch.Tensor,
        resolution: float = 1.0,
    ) -> PullbackGeometry:
        return cls(generalized_spectrum(information, reference_metric), resolution)

    @property
    def effective_dimension(self) -> float:
        return float(effective_dimension(self.eigenvalues, self.resolution))

    @property
    def complexity(self) -> float:
        return float(relative_log_volume(self.eigenvalues, self.resolution))

    @property
    def relative_log_volume(self) -> float:
        return self.complexity

    @property
    def relative_volume(self) -> float:
        return float(torch.exp(0.5 * relative_log_volume(self.eigenvalues, self.resolution)))


class ObservableGeometry(PullbackGeometry):
    """Deprecated compatibility name for :class:`PullbackGeometry`."""

    def __init__(self, eigenvalues: torch.Tensor, resolution: float = 1.0) -> None:
        warnings.warn(
            "ObservableGeometry is deprecated; use PullbackGeometry",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(eigenvalues, resolution)
