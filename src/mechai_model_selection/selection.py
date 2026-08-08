"""Criterion weights and stability summaries."""

from __future__ import annotations

import math

import torch


def criterion_weights(scores: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(scores)
    shifted = values - torch.min(values)
    return torch.softmax(-0.5 * shifted, dim=0)


def selection_entropy(weights: torch.Tensor, *, normalize: bool = False) -> float:
    w = torch.as_tensor(weights)
    w = w / torch.sum(w)
    entropy = -torch.sum(torch.where(w > 0, w * torch.log(w), torch.zeros_like(w)))
    if normalize and w.numel() > 1:
        entropy = entropy / math.log(w.numel())
    return float(entropy)


def stacking_weights(
    pointwise_log_predictive_density: torch.Tensor,
    *,
    max_steps: int = 1000,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Compute simplex-constrained predictive stacking weights."""
    values = torch.as_tensor(pointwise_log_predictive_density)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("pointwise log predictive density must have shape (models, observations)")
    if not torch.all(torch.isfinite(values)):
        raise ValueError("pointwise log predictive density must be finite")
    logits = torch.zeros(
        values.shape[0], dtype=values.dtype, device=values.device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [logits], lr=0.5, max_iter=max_steps, tolerance_grad=tolerance,
        tolerance_change=tolerance, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        log_weights = torch.log_softmax(logits, dim=0).unsqueeze(1)
        loss = -torch.sum(torch.logsumexp(log_weights + values, dim=0))
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.softmax(logits.detach(), dim=0)


def model_average(
    means: torch.Tensor,
    variances: torch.Tensor,
    weights: torch.Tensor,
    *,
    confidence: float | None = None,
) -> dict[str, torch.Tensor]:
    """Decompose model-averaged uncertainty into within- and between-model parts."""
    mu = torch.as_tensor(means)
    var = torch.as_tensor(variances, dtype=mu.dtype, device=mu.device)
    w = torch.as_tensor(weights, dtype=mu.dtype, device=mu.device)
    if mu.shape != var.shape or mu.shape[0] != w.numel():
        raise ValueError("means and variances must match, with models on the first axis")
    w = w / torch.sum(w)
    expand = (w.shape[0],) + (1,) * (mu.ndim - 1)
    shaped = w.reshape(expand)
    averaged = torch.sum(shaped * mu, dim=0)
    within = torch.sum(shaped * var, dim=0)
    between = torch.sum(shaped * (mu - averaged) ** 2, dim=0)
    total = within + between
    result = {"mean": averaged, "within": within, "between": between, "variance": total}
    if confidence is not None:
        if not 0 < confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one")
        normal = torch.distributions.Normal(
            torch.tensor(0.0, dtype=mu.dtype, device=mu.device),
            torch.tensor(1.0, dtype=mu.dtype, device=mu.device),
        )
        quantile = normal.icdf(
            torch.tensor(0.5 + confidence / 2.0, dtype=mu.dtype, device=mu.device)
        )
        standard_deviation = torch.sqrt(torch.clamp(total, min=0.0))
        result.update(
            {
                "standard_deviation": standard_deviation,
                "lower": averaged - quantile * standard_deviation,
                "upper": averaged + quantile * standard_deviation,
            }
        )
    return result
