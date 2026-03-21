"""Unit tests for confidence interval and hypothesis testing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.inference.confidence_interval import (
    ForecastInference,
    mean_confidence_interval,
)
from src.inference.hypothesis_test import TimeSeriesTests, two_sample_mean_test


@pytest.fixture
def inference_model_dict() -> dict:
    """Synthetic model-like payload for interval construction tests."""
    rng = np.random.default_rng(123)
    return {
        "predictions": np.linspace(0.1, 1.2, 120),
        "residuals": rng.normal(0, 0.15, 500),
    }


def test_mean_confidence_interval_ordering() -> None:
    """Mean CI should return ordered bounds for non-empty data."""
    data = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
    lower, upper = mean_confidence_interval(data)
    assert lower < upper, "Lower CI bound should be smaller than upper bound."


def test_forecast_inference_interval_methods(inference_model_dict: dict) -> None:
    """ForecastInference should support asymptotic, bootstrap, and prediction intervals."""
    model = ForecastInference(model=inference_model_dict, conf_level=0.95)

    asym = model.asymptotic_ci(horizon=15)
    pct = model.bootstrap_ci(n_bootstraps=250, method="percentile", horizon=15)
    bca = model.bootstrap_ci(n_bootstraps=250, method="bca", horizon=15)
    pred = model.prediction_interval(horizon=15)

    for df in [asym, pct, bca, pred]:
        assert len(df) == 15, "Interval output should match requested horizon."
        assert {"forecast", "lower", "upper"}.issubset(
            df.columns
        ), "Interval DataFrame missing required columns."


def test_forecast_inference_coverage_and_compare(inference_model_dict: dict) -> None:
    """Coverage simulation and method comparison should return expected summary fields."""
    model = ForecastInference(model=inference_model_dict, conf_level=0.95)
    truth = np.linspace(0.15, 1.0, 20)

    coverage = model.coverage_simulation(true_values=truth, n_simulations=80)
    assert (
        "coverage_probability" in coverage
    ), "Coverage output missing coverage_probability."
    assert (
        0.0 <= coverage["coverage_probability"] <= 1.0
    ), "Coverage probability must lie in [0, 1]."

    cmp_df = model.compare_methods(horizon=10, n_bootstraps=250)
    assert set(cmp_df["method"]) == {
        "asymptotic",
        "bootstrap_percentile",
        "bootstrap_bca",
    }, "Method comparison should include all three CI constructions."


def test_time_series_tests_core_hypotheses() -> None:
    """TimeSeriesTests should run stationarity, cointegration, granger, and model comparison tests."""
    pytest.importorskip("statsmodels")
    tester = TimeSeriesTests()
    rng = np.random.default_rng(21)

    n = 220
    e1 = rng.normal(0, 1, n)
    e2 = rng.normal(0, 1, n)
    x = np.cumsum(e1)
    y = x + 0.1 * e2
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.5 * z[t - 1] + 0.3 * x[t - 1] + rng.normal(0, 1)

    df = pd.DataFrame({"Close": x + rng.normal(0, 0.1, n)})

    adf_res = tester.stationarity_test(df, column="Close", test="adf")
    kpss_res = tester.stationarity_test(df, column="Close", test="kpss")
    coint_res = tester.cointegration_test(x, y)
    granger_res = tester.granger_causality_test(x, z, max_lag=3)
    cusum_res = tester.structural_break_test(df, test="cusum", column="Close")
    dm_res = tester.compare_models(rng.normal(0, 1, n), rng.normal(0, 1.2, n))

    for res in [adf_res, kpss_res, coint_res, granger_res, cusum_res, dm_res]:
        expected = {
            "test_statistic",
            "p_value",
            "critical_value",
            "reject_null",
            "conclusion",
        }
        assert expected.issubset(
            res.keys()
        ), "Test result dictionary missing required keys."


def test_multiple_testing_correction_and_visualization() -> None:
    """Multiple-testing correction and plotting API should return structured outputs."""
    tester = TimeSeriesTests()
    correction_bonf = tester.apply_multiple_testing_correction(
        [0.01, 0.02, 0.3], method="bonferroni"
    )
    correction_fdr = tester.apply_multiple_testing_correction(
        [0.01, 0.02, 0.3], method="fdr"
    )

    assert (
        len(correction_bonf["adjusted_p_values"]) == 3
    ), "Bonferroni correction should preserve p-value count."
    assert (
        len(correction_fdr["adjusted_p_values"]) == 3
    ), "FDR correction should preserve p-value count."

    fig = tester.plot_test_results(
        {
            "ADF": {
                "test_statistic": -3.1,
                "p_value": 0.02,
                "critical_value": -2.9,
                "reject_null": True,
                "conclusion": "Stationary",
            },
            "KPSS": {
                "test_statistic": 0.4,
                "p_value": 0.08,
                "critical_value": 0.46,
                "reject_null": False,
                "conclusion": "Fail to reject",
            },
        }
    )
    assert len(fig.data) >= 1, "Visualization should contain plotted traces."


def test_two_sample_mean_test_and_edge_cases() -> None:
    """Compatibility two-sample test should return z-stat and p-value and reject invalid input."""
    a = np.array([0.1, 0.2, 0.05, 0.12, 0.18])
    b = np.array([0.01, 0.02, 0.03, 0.0, 0.04])
    out = two_sample_mean_test(a, b)
    assert (
        "test_statistic" in out and "p_value" in out
    ), "Two-sample test output missing required keys."

    with pytest.raises(ValueError):
        two_sample_mean_test(np.array([1.0]), np.array([2.0]))
