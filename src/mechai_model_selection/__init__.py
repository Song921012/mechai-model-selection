"""Model selection for mechanism--AI coupled dynamical systems."""

from .bayesian import dic, laplace_deviance, waic, wbic
from .classical import aic, aicc, bic, hqic, tic
from .confidence import coverage_summary, geometric_confidence_region, quotient_rank
from .geometry import (
    ObservableGeometry,
    PullbackGeometry,
    block_reference_metric,
    effective_dimension,
    generalized_spectrum,
    geometry_sensitivity_grid,
    observable_complexity,
    observable_dimension,
    relative_log_volume,
    relative_volume,
    resolution_profile,
)
from .ogic import (
    generalized_optimism,
    gic_bic_approximation,
    gic_effective,
    gic_evidence,
    gic_laplace,
    gic_predictive,
    gic_volume,
    ogic_evidence,
    ogic_predictive,
)
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
    "PullbackGeometry",
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
    "generalized_optimism",
    "generalized_spectrum",
    "geometric_confidence_region",
    "geometry_sensitivity_grid",
    "gic_bic_approximation",
    "gic_effective",
    "gic_evidence",
    "gic_laplace",
    "gic_predictive",
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
    "relative_log_volume",
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

