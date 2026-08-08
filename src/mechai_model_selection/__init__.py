"""Model selection for mechanism--AI coupled dynamical systems."""

from .bayesian import dic, laplace_deviance, waic, wbic
from .classical import aic, aicc, bic, hqic, tic
from .confidence import coverage_summary, geometric_confidence_region, quotient_rank
from .geometry import (
    ObservableGeometry,
    block_reference_metric,
    effective_dimension,
    generalized_spectrum,
    geometry_sensitivity_grid,
    observable_complexity,
    observable_dimension,
    relative_volume,
    resolution_profile,
)
from .ogic import gic_effective, gic_laplace, gic_volume, ogic_evidence, ogic_predictive
from .selection import criterion_weights, model_average, selection_entropy, stacking_weights
from .torch_ops import (
    fisher_pullback,
    pullback_geometry,
    residual_jacobian,
    sensitivity_gramian,
    wasserstein_pullback_1d,
)
from .validation import aggregate_fold_deviance, predictive_deviance, rolling_origin_splits

__all__ = [
    "ObservableGeometry",
    "aggregate_fold_deviance",
    "aic",
    "aicc",
    "bic",
    "block_reference_metric",
    "coverage_summary",
    "criterion_weights",
    "dic",
    "effective_dimension",
    "fisher_pullback",
    "generalized_spectrum",
    "geometric_confidence_region",
    "geometry_sensitivity_grid",
    "gic_effective",
    "gic_laplace",
    "gic_volume",
    "hqic",
    "laplace_deviance",
    "model_average",
    "observable_complexity",
    "observable_dimension",
    "ogic_evidence",
    "ogic_predictive",
    "predictive_deviance",
    "pullback_geometry",
    "quotient_rank",
    "relative_volume",
    "residual_jacobian",
    "resolution_profile",
    "rolling_origin_splits",
    "selection_entropy",
    "sensitivity_gramian",
    "stacking_weights",
    "tic",
    "waic",
    "wasserstein_pullback_1d",
    "wbic",
]

