# mechai-model-selection

[![CI](https://github.com/Song921012/mechai-model-selection/actions/workflows/ci.yml/badge.svg)](https://github.com/Song921012/mechai-model-selection/actions/workflows/ci.yml)
[![Documentation](https://github.com/Song921012/mechai-model-selection/actions/workflows/docs.yml/badge.svg)](https://github.com/Song921012/mechai-model-selection/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyTorch tools for geometric model selection in mechanism-AI dynamical systems.
The package compares a pullback metric on the statistical model manifold with a
declared reference geometry. It is independent of a particular ODE solver:
users provide a differentiable solution map, a residual Jacobian, or an
information matrix.

## Installation

    python -m pip install -e .

## Quick start

    import torch
    from mechai_model_selection import (
        PullbackGeometry,
        gic_bic_approximation,
        gic_evidence,
        gic_predictive,
    )

    G = torch.tensor([[8.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    R = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    geometry = PullbackGeometry.from_matrices(G, R, resolution=1.0)

    print(geometry.effective_dimension)
    print(geometry.relative_log_volume)
    print(gic_predictive(120.0, geometry))
    print(gic_evidence(120.0, geometry, prior_energy=0.7))
    print(gic_bic_approximation(120.0, geometry, n_observations=80))

## What the criteria mean

- generalized_optimism(A, B) computes the sandwich complexity
  trace(A^{-1} B) for a loss-matched penalized estimator.
- gic_predictive uses deviance plus 2 d_eff when the Fisher information
  identity is appropriate.
- gic_evidence uses the normalized local-Gaussian evidence expression:
  deviance, prior energy, and relative log-volume.
- gic_bic_approximation retains only the regular large-sample
  log(n) d_eff penalty.
- gic_volume is retained for compatibility as a sensitivity score with an
  additional user-specified volume weight; it is not a uniquely derived
  information criterion.

The package also provides AIC, AICc, BIC, HQIC, TIC, DIC, WAIC, WBIC,
Laplace diagnostics, Fisher and one-dimensional Wasserstein pullbacks,
geometric confidence regions on the identifiable quotient, blocked validation, stacking, and
trajectory-valued model averaging.

ObservableGeometry, observable_dimension, observable_complexity,
ogic_predictive, and ogic_evidence remain as deprecated aliases so older
result archives are readable. New code should use the canonical names shown
above.

See docs/ for derivations, metric conventions, examples, and limitations.
The paper reproduction repository is
[MechAIModelSelection](https://github.com/Song921012/MechAIModelSelection).
