"""Predictive validation score helpers."""

from __future__ import annotations

import torch


def rolling_origin_splits(
    n_observations: int,
    *,
    train_fractions: tuple[float, ...] = (0.5, 0.6, 0.7),
    validation_fraction: float = 0.1,
) -> list[tuple[slice, slice]]:
    """Return ordered train/validation slices for time-indexed observations."""
    if n_observations < 3:
        raise ValueError("at least three observations are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    splits: list[tuple[slice, slice]] = []
    for fraction in train_fractions:
        if not 0 < fraction < 1:
            raise ValueError("train fractions must lie strictly between zero and one")
        train_end = max(2, round(fraction * n_observations))
        valid_end = min(
            n_observations,
            train_end + max(1, round(validation_fraction * n_observations)),
        )
        if valid_end <= train_end:
            raise ValueError("a rolling split has an empty validation block")
        splits.append((slice(0, train_end), slice(train_end, valid_end)))
    return splits


def predictive_deviance(pointwise_log_predictive_density: torch.Tensor) -> float:
    values = torch.as_tensor(pointwise_log_predictive_density)
    if values.ndim != 1:
        raise ValueError("pointwise_log_predictive_density must be one-dimensional")
    return float(-2.0 * torch.sum(values))


def aggregate_fold_deviance(fold_log_predictive_densities: list[torch.Tensor]) -> float:
    if not fold_log_predictive_densities:
        raise ValueError("at least one fold is required")
    return sum(predictive_deviance(fold) for fold in fold_log_predictive_densities)
