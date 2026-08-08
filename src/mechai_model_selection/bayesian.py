"""Posterior and local-evidence criteria."""

from __future__ import annotations

import math

import torch


def waic(pointwise_log_likelihood: torch.Tensor) -> dict[str, float]:
    """Compute WAIC from draws-by-observations pointwise log likelihoods."""
    ll = torch.as_tensor(pointwise_log_likelihood)
    if ll.ndim != 2 or ll.shape[0] < 2:
        raise ValueError("pointwise_log_likelihood must have shape (draws, observations)")
    lppd_i = torch.logsumexp(ll, dim=0) - math.log(ll.shape[0])
    p_waic_i = torch.var(ll, dim=0, unbiased=True)
    elpd = torch.sum(lppd_i - p_waic_i)
    return {
        "waic": float(-2.0 * elpd),
        "elpd_waic": float(elpd),
        "p_waic": float(torch.sum(p_waic_i)),
    }


def dic(log_likelihood_draws: torch.Tensor, log_likelihood_at_posterior_mean: float) -> dict[str, float]:
    """Deviance information criterion using ``p_D = E[D] - D(E[theta])``."""
    ll = torch.as_tensor(log_likelihood_draws)
    if ll.ndim != 1:
        raise ValueError("log_likelihood_draws must be one-dimensional")
    mean_deviance = float(torch.mean(-2.0 * ll))
    deviance_mean = -2.0 * float(log_likelihood_at_posterior_mean)
    p_d = mean_deviance - deviance_mean
    return {"dic": mean_deviance + p_d, "p_dic": p_d}


def wbic(log_likelihood_draws_at_temperature: torch.Tensor) -> float:
    """WBIC on the deviance scale from draws at inverse temperature 1/log(n)."""
    ll = torch.as_tensor(log_likelihood_draws_at_temperature)
    if ll.ndim != 1:
        raise ValueError("log_likelihood_draws_at_temperature must be one-dimensional")
    return float(-2.0 * torch.mean(ll))


def laplace_deviance(
    log_likelihood_at_map: float,
    log_prior_at_map: float,
    posterior_hessian: torch.Tensor,
    parameter_dimension: int,
) -> float:
    """Negative twice the Laplace log evidence, including normalization terms."""
    h = torch.as_tensor(posterior_hessian)
    sign, logdet = torch.linalg.slogdet(h)
    if float(sign) <= 0:
        return math.inf
    return float(
        -2.0 * (log_likelihood_at_map + log_prior_at_map)
        - parameter_dimension * math.log(2.0 * math.pi)
        + logdet
    )
