import math

import pytest
import torch

from mechai_model_selection import (
    ObservableGeometry,
    PullbackGeometry,
    generalized_optimism,
    gic_bic_approximation,
    gic_evidence,
    gic_predictive,
    relative_log_volume,
)


def test_generalized_optimism_matches_ridge_effective_df():
    information = torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64))
    reference = torch.eye(2, dtype=torch.float64)
    value = generalized_optimism(information, information, reference, 1.0)
    assert value == pytest.approx(4.0 / 5.0 + 1.0 / 2.0)


def test_predictive_and_evidence_scores_have_derived_penalties():
    geometry = PullbackGeometry(torch.tensor([4.0, 1.0], dtype=torch.float64))
    assert gic_predictive(10.0, geometry) == pytest.approx(12.6)
    expected_volume = math.log(5.0) + math.log(2.0)
    assert gic_evidence(10.0, geometry, prior_energy=3.0) == pytest.approx(
        13.0 + expected_volume
    )
    assert gic_bic_approximation(10.0, geometry, 100) == pytest.approx(
        10.0 + math.log(100.0) * 1.3
    )


def test_relative_volume_scale_derivative_equals_effective_dimension():
    eigenvalues = torch.tensor([7.0, 0.5, 0.0], dtype=torch.float64)
    scale = 1.7
    step = 1e-5
    upper = relative_log_volume(eigenvalues, scale * math.exp(step))
    lower = relative_log_volume(eigenvalues, scale * math.exp(-step))
    derivative = -(upper - lower) / (2.0 * step)
    geometry = PullbackGeometry(eigenvalues, scale)
    assert float(derivative) == pytest.approx(geometry.effective_dimension, rel=2e-8)


def test_observable_geometry_remains_a_compatible_alias():
    with pytest.deprecated_call():
        legacy = ObservableGeometry(torch.tensor([2.0], dtype=torch.float64))
    assert isinstance(legacy, PullbackGeometry)
    assert legacy.effective_dimension == pytest.approx(2.0 / 3.0)
