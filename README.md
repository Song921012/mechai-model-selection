# mechai-model-selection

[![CI](https://github.com/Song921012/mechai-model-selection/actions/workflows/ci.yml/badge.svg)](https://github.com/Song921012/mechai-model-selection/actions/workflows/ci.yml)
[![Documentation](https://github.com/Song921012/mechai-model-selection/actions/workflows/docs.yml/badge.svg)](https://github.com/Song921012/mechai-model-selection/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

PyTorch-first tools for selecting and assessing mechanism-AI coupled dynamical
systems. The package measures complexity through a pullback metric on the full
parameter-to-solution-to-observation map. It is independent of a particular ODE
solver: users provide a differentiable observation map, residual Jacobian, or
information matrix.

## Installation

```bash
python -m pip install -e .
```

## Quick start

```python
import math
import torch
from mechai_model_selection import (
    ObservableGeometry,
    effective_dimension,
    gic_effective,
    gic_volume,
    relative_volume,
)

G = torch.tensor([[8.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
R = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
geometry = ObservableGeometry.from_matrices(G, R, resolution=1.0)

print(effective_dimension(geometry.eigenvalues))
print(relative_volume(geometry.eigenvalues))
print(gic_effective(120.0, geometry, penalty_factor=math.log(80)))
print(gic_volume(120.0, geometry, penalty_factor=math.log(80), volume_weight=0.5))
```

The public API includes classical criteria (AIC, AICc, BIC, HQIC, TIC), local
Bayesian diagnostics (DIC, WAIC, WBIC, Laplace), GIC-eff, GIC-vol, Fisher and
one-dimensional Wasserstein pullbacks, quotient confidence regions, rolling
validation, stacking, and trajectory-valued model averaging.

`effective_dimension`, `relative_volume`, and `gic_laplace` are the canonical
0.3 API. The former names `observable_dimension`, `observable_complexity`,
`ogic_predictive`, and `ogic_evidence` remain available with deprecation
warnings for reproducibility of earlier scripts.

See `docs/` for the mathematical conventions, examples, API, and limitations.
The experiment repository is
[MechAIModelSelection](https://github.com/Song921012/MechAIModelSelection).
