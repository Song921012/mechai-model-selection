"""PyTorch differentiation helpers for solution-to-observation maps."""

from __future__ import annotations

from collections.abc import Callable

import torch


def residual_jacobian(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
) -> torch.Tensor:
    """Return the flattened residual Jacobian with rows indexed by observations."""
    theta = parameters.detach().clone().requires_grad_(True)
    jac = torch.autograd.functional.jacobian(residual_fn, theta, vectorize=True)
    return jac.reshape(-1, theta.numel())


def sensitivity_gramian(
    residual_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    observation_precision: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    jac = residual_jacobian(residual_fn, parameters)
    if observation_precision is None:
        gram = jac.mT @ jac
    else:
        precision = torch.as_tensor(observation_precision, dtype=jac.dtype, device=jac.device)
        gram = jac.mT @ precision @ jac
    return 0.5 * (gram + gram.mT), jac


def fisher_pullback(
    solution_map: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    observation_precision: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull back a Gaussian/Fisher observation metric through a solution map."""
    return sensitivity_gramian(
        solution_map,
        parameters,
        observation_precision=observation_precision,
    )


def wasserstein_pullback_1d(
    quantile_map: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull back the one-dimensional W2 metric in quantile coordinates."""
    jac = residual_jacobian(quantile_map, parameters)
    if weights is None:
        weights = torch.ones(jac.shape[0], dtype=jac.dtype, device=jac.device)
    weights = torch.as_tensor(weights, dtype=jac.dtype, device=jac.device)
    if weights.ndim != 1 or weights.numel() != jac.shape[0]:
        raise ValueError("weights must match the number of quantile coordinates")
    if torch.any(weights < 0) or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite and nonnegative")
    total = torch.sum(weights)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    normalized = weights / total
    gram = jac.mT @ (normalized.unsqueeze(1) * jac)
    return 0.5 * (gram + gram.mT), jac


def pullback_geometry(
    map_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    *,
    metric: str = "fisher",
    observation_precision: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a supported pullback metric with a common interface."""
    key = metric.lower().replace("-", "_")
    if key in {"fisher", "fisher_rao", "gaussian"}:
        if weights is not None:
            raise ValueError("weights are only used by the Wasserstein metric")
        return fisher_pullback(
            map_fn,
            parameters,
            observation_precision=observation_precision,
        )
    if key in {"wasserstein", "wasserstein_1d", "w2"}:
        if observation_precision is not None:
            raise ValueError("observation_precision is only used by the Fisher metric")
        return wasserstein_pullback_1d(map_fn, parameters, weights=weights)
    raise ValueError(f"unsupported pullback metric: {metric}")
