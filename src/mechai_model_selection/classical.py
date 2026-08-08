"""Classical likelihood-based information criteria."""

from __future__ import annotations

import math

import torch


def aic(log_likelihood: float, n_parameters: int) -> float:
    return -2.0 * float(log_likelihood) + 2.0 * int(n_parameters)


def aicc(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    k = int(n_parameters)
    n = int(n_observations)
    if n <= k + 1:
        return math.inf
    return aic(log_likelihood, k) + 2.0 * k * (k + 1) / (n - k - 1)


def bic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    return -2.0 * float(log_likelihood) + int(n_parameters) * math.log(int(n_observations))


def hqic(log_likelihood: float, n_parameters: int, n_observations: int) -> float:
    n = int(n_observations)
    if n <= 1:
        raise ValueError("n_observations must exceed one")
    return -2.0 * float(log_likelihood) + 2.0 * int(n_parameters) * math.log(math.log(n))


def tic(log_likelihood: float, score_covariance: torch.Tensor, expected_hessian: torch.Tensor) -> float:
    """Takeuchi's criterion on the conventional deviance scale.

    The penalty is ``2 tr(H^{-1} K)``. A linear solve is used instead of an
    explicit inverse.
    """
    h = torch.as_tensor(expected_hessian)
    k = torch.as_tensor(score_covariance, dtype=h.dtype, device=h.device)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or k.shape != h.shape:
        raise ValueError("score_covariance and expected_hessian must be square and equal-sized")
    penalty = 2.0 * torch.trace(torch.linalg.solve(h, k))
    return -2.0 * float(log_likelihood) + float(penalty)
