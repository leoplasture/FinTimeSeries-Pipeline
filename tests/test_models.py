"""Unit tests for ARIMA, GARCH, and LSTM forecasting modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_close_df() -> pd.DataFrame:
    """Create synthetic non-stationary close-price series."""
    n = 180
    rng = np.random.default_rng(7)
    trend = np.linspace(0, 12, n)
    close = 100 + trend + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({"Close": close})


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create synthetic return series for volatility tests."""
    n = 320
    rng = np.random.default_rng(11)
    # Mild clustering proxy: scale random noise by slow-moving volatility factor.
    vol = 0.8 + 0.4 * np.sin(np.linspace(0, 10, n)) ** 2
    returns = rng.normal(0, vol, n)
    return pd.Series(returns)


def test_arima_fit_forecast_ci_and_persistence(
    tmp_path, sample_close_df: pd.DataFrame
) -> None:
    """ARIMA should fit, forecast with CI, and support save/load of artifacts."""
    pytest.importorskip("statsmodels")
    from src.models.arima import ARIMAModel

    model = ARIMAModel(conf_level=0.95)
    model.fit(sample_close_df, column="Close", auto_order=True)

    fc = model.forecast(steps=12, confidence_interval=True)
    assert fc.shape[0] == 12, "Forecast horizon length should match requested steps."
    assert {"forecast", "lower", "upper"}.issubset(
        fc.columns
    ), "Forecast output missing confidence interval columns."

    ci = model.get_confidence_interval(alpha=0.05, steps=8)
    assert (
        ci.shape[0] == 8
    ), "Confidence interval should return one row per forecast step."

    coeff = model.hypothesis_test_coefficients()
    assert "p_value" in coeff.columns, "Coefficient test output missing p-values."

    artifact_path = tmp_path / "artifacts" / "arima.pkl"
    saved = model.save_artifact(str(artifact_path))
    loaded = ARIMAModel.load_artifact(str(saved))
    assert loaded is not None, "Loaded ARIMA artifact should not be None."


def test_arima_edge_cases_short_series_and_missing_values() -> None:
    """ARIMA should reject too-short series and handle missing values when sufficient data remains."""
    pytest.importorskip("statsmodels")
    from src.models.arima import ARIMAModel

    short_df = pd.DataFrame({"Close": np.arange(20, dtype=float)})
    with pytest.raises(ValueError, match="At least 30 observations"):
        ARIMAModel().fit(short_df)

    long_df = pd.DataFrame({"Close": np.linspace(100, 120, 80)})
    long_df.loc[5:10, "Close"] = np.nan
    model = ARIMAModel()
    model.fit(long_df.dropna(), column="Close", auto_order=False)
    assert (
        model.fitted_model is not None
    ), "ARIMA should fit after dropping missing values."


def test_garch_forecast_risk_backtest(sample_returns: pd.Series) -> None:
    """GARCH should forecast volatility, compute risk metrics, and run backtest."""
    pytest.importorskip("arch")
    from src.models.garch import GARCHModel

    model = GARCHModel(order=(1, 1), variant="GARCH", conf_level=0.95)
    model.fit(sample_returns)

    vol_fc = model.forecast_volatility(steps=10, confidence_interval=True)
    assert len(vol_fc) == 10, "Volatility forecast must match requested horizon."
    assert {"variance", "volatility", "lower", "upper"}.issubset(
        vol_fc.columns
    ), "Volatility forecast missing expected columns."

    risk = model.compute_risk_metrics(confidence_level=0.95)
    assert (
        "VaR" in risk and "ExpectedShortfall" in risk
    ), "Risk metrics output missing VaR/ES."

    bt = model.backtest(strategy="long_short", lookback=120)
    assert {"sharpe", "cumulative_return", "hit_rate", "max_drawdown"}.issubset(
        bt.keys()
    ), "Backtest output missing required metrics."

    bt_exp = model.backtest(
        strategy="long_short", lookback=120, window_mode="expanding"
    )
    assert (
        bt_exp["window_mode"] == "expanding"
    ), "Backtest should preserve selected window mode in output."


def test_garch_edge_case_short_series() -> None:
    """GARCH should reject short return series."""
    pytest.importorskip("arch")
    from src.models.garch import GARCHModel

    short_returns = pd.Series(np.random.default_rng(1).normal(0, 1, 40))
    with pytest.raises(ValueError, match="At least 100 observations"):
        GARCHModel().fit(short_returns)


def test_lstm_sequence_training_and_prediction(sample_close_df: pd.DataFrame) -> None:
    """LSTM baseline should create windows, train quickly, and predict horizon outputs."""
    pytest.importorskip("sklearn")
    from src.models.lstm_forecast import LSTMForecastModel, build_lstm_input

    series = sample_close_df["Close"].to_numpy(dtype=float)
    x, y = build_lstm_input(series, window_size=12)
    assert (
        x.shape[1] == 12
    ), "Sequence window width should match configured window_size."
    assert len(x) == len(y), "Input window and target arrays must align in length."

    model = LSTMForecastModel(
        window_size=12, hidden_units=8, epochs=1, backend="sklearn"
    )
    model.fit(series)
    preds = model.forecast(series, steps=6)
    assert preds.shape == (6,), "Forecast should return one value per requested step."


def test_lstm_edge_case_short_series() -> None:
    """LSTM baseline should reject sequences shorter than window size."""
    pytest.importorskip("sklearn")
    from src.models.lstm_forecast import LSTMForecastModel

    with pytest.raises(ValueError, match="too short"):
        LSTMForecastModel(window_size=20, backend="sklearn").fit(
            np.arange(10, dtype=float)
        )


def test_integration_like_data_to_arima_metrics(sample_close_df: pd.DataFrame) -> None:
    """Integration-style check from model fit to forecast metric computation."""
    pytest.importorskip("statsmodels")
    from src.models.arima import ARIMAModel

    train = sample_close_df.iloc[:-20].copy()
    test = sample_close_df.iloc[-20:].copy()

    model = ARIMAModel(order=(1, 1, 1), conf_level=0.95)
    model.fit(train, auto_order=False)
    fc = model.forecast(steps=20)
    scores = model.evaluate(test["Close"].to_numpy(), fc["forecast"].to_numpy())

    assert "rmse" in scores and np.isfinite(
        scores["rmse"]
    ), "Integration flow should produce finite RMSE."
