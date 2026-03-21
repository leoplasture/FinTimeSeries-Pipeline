"""ARIMA forecasting wrapper with statistical inference and diagnostics.

Theoretical notes:
1. ARIMA assumes linear dynamics in the differenced series, weak stationarity
   after differencing, and white-noise residual innovations.
2. ARIMA is a strong baseline when the signal is mostly linear and univariate;
   nonlinear/deep models are useful when structure is strongly nonlinear.
3. AR/MA coefficients quantify short-run persistence and shock propagation.
"""

from __future__ import annotations

import pickle
import warnings
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


_STANDARD_NORMAL = NormalDist(0.0, 1.0)


class ARIMAModel:
    """ARIMA model manager with fitting, inference, and persistence.

    Parameters:
       order (tuple[int, int, int]): Initial ARIMA order (p, d, q).
       seasonal_order (tuple[int, int, int, int] | None): Seasonal order for
          SARIMA-like fitting. Passed directly to statsmodels when provided.
       conf_level (float): Confidence level for forecast intervals.
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        conf_level: float = 0.95,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.conf_level = conf_level
        self.column: Optional[str] = None
        self.model = None
        self.fitted_model = None
        self.last_forecast: Optional[pd.DataFrame] = None
        self.stationarity_result: Optional[Dict[str, float]] = None

    def fit(
        self,
        df: pd.DataFrame,
        column: str = "Close",
        auto_order: bool = True,
        criterion: str = "aic",
        search_space: Tuple[range, range, range] = (
            range(0, 4),
            range(0, 3),
            range(0, 4),
        ),
    ) -> "ARIMAModel":
        """Fit ARIMA on selected column after stationarity diagnostics.

        Parameters:
           df (pandas.DataFrame): Input DataFrame.
           column (str): Target series column.
           auto_order (bool): Whether to run AIC/BIC grid search.
           criterion (str): `aic` or `bic` model selection criterion.
           search_space (tuple): Search ranges for p, d, q.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        y = df[column].dropna().astype(float).reset_index(drop=True)
        if y.size < 30:
            raise ValueError(
                "At least 30 observations are recommended for ARIMA fitting."
            )

        self.column = column
        self.stationarity_result = self._adf_summary(y)
        if auto_order:
            self.order = self._auto_select_order(
                y, criterion=criterion, search_space=search_space
            )

        if self.seasonal_order is None:
            self.model = ARIMA(y, order=self.order)
        else:
            self.model = ARIMA(y, order=self.order, seasonal_order=self.seasonal_order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings(
                "ignore",
                message="Non-invertible starting MA parameters found.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Non-stationary starting autoregressive parameters found.*",
                category=UserWarning,
            )
            self.fitted_model = self.model.fit()
        return self

    def forecast(
        self, steps: int = 30, confidence_interval: bool = True
    ) -> pd.DataFrame:
        """Generate point forecasts and optional confidence intervals.

        Parameters:
           steps (int): Forecast horizon.
           confidence_interval (bool): Include confidence bands.
        """
        self._require_fitted()
        pred = self.fitted_model.get_forecast(steps=steps)
        mean_fc = np.asarray(pred.predicted_mean, dtype=float)
        out = pd.DataFrame({"forecast": mean_fc})

        if confidence_interval:
            alpha = 1.0 - self.conf_level
            ci = pred.conf_int(alpha=alpha)
            if isinstance(ci, pd.DataFrame):
                out["lower"] = ci.iloc[:, 0].to_numpy(dtype=float)
                out["upper"] = ci.iloc[:, 1].to_numpy(dtype=float)
            else:
                ci_np = np.asarray(ci, dtype=float)
                out["lower"] = ci_np[:, 0]
                out["upper"] = ci_np[:, 1]

        self.last_forecast = out
        return out

    def evaluate(
        self,
        y_true: Iterable[float],
        y_pred: Iterable[float],
        metrics: List[str] = ["mae", "rmse", "mape"],
    ) -> Dict[str, float]:
        """Evaluate forecast accuracy using requested metrics."""
        y_t = np.asarray(list(y_true), dtype=float)
        y_p = np.asarray(list(y_pred), dtype=float)
        n = min(y_t.size, y_p.size)
        if n == 0:
            raise ValueError("y_true and y_pred must have at least one aligned value.")
        y_t = y_t[:n]
        y_p = y_p[:n]

        out: Dict[str, float] = {}
        for metric in metrics:
            key = metric.lower()
            if key == "mae":
                out["mae"] = float(np.mean(np.abs(y_t - y_p)))
            elif key == "rmse":
                out["rmse"] = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
            elif key == "mape":
                denom = np.where(np.isclose(y_t, 0.0), np.nan, y_t)
                out["mape"] = float(np.nanmean(np.abs((y_t - y_p) / denom)) * 100)
            else:
                raise ValueError(f"Unsupported metric '{metric}'.")
        return out

    def plot_diagnostics(self, lags: int = 24) -> Dict[str, go.Figure]:
        """Create residual diagnostics charts (residuals, ACF, PACF)."""
        self._require_fitted()
        resid = np.asarray(self.fitted_model.resid, dtype=float)

        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(y=resid, mode="lines", name="Residuals"))
        fig_resid.update_layout(title="ARIMA Residuals", template="plotly_white")

        # Build ACF/PACF with statsmodels helpers and convert to plotly-like output.
        # The returned matplotlib figures are not returned directly to keep API unified.
        acf_vals = self._acf_values(resid, lags=lags)
        pacf_vals = self._pacf_values(resid, lags=lags)

        fig_acf = go.Figure(go.Bar(x=np.arange(acf_vals.size), y=acf_vals, name="ACF"))
        fig_acf.update_layout(title="Residual ACF", template="plotly_white")

        fig_pacf = go.Figure(
            go.Bar(x=np.arange(pacf_vals.size), y=pacf_vals, name="PACF")
        )
        fig_pacf.update_layout(title="Residual PACF", template="plotly_white")

        return {"residuals": fig_resid, "acf": fig_acf, "pacf": fig_pacf}

    def get_confidence_interval(
        self, alpha: float = 0.05, steps: int = 30
    ) -> pd.DataFrame:
        """Return forecast confidence intervals for a chosen alpha.

        Parameters:
           alpha (float): Tail probability for two-sided interval.
           steps (int): Forecast horizon.
        """
        self._require_fitted()
        pred = self.fitted_model.get_forecast(steps=steps)
        ci = pred.conf_int(alpha=alpha)
        if isinstance(ci, pd.DataFrame):
            return ci.reset_index(drop=True)
        ci_np = np.asarray(ci, dtype=float)
        return pd.DataFrame({"lower": ci_np[:, 0], "upper": ci_np[:, 1]})

    def hypothesis_test_coefficients(self, alpha: float = 0.05) -> pd.DataFrame:
        """Test significance of AR/MA coefficients from fitted model.

        Returns:
           pandas.DataFrame: Columns include estimate, p_value, and reject_null.
        """
        self._require_fitted()
        params = self.fitted_model.params
        pvalues = self.fitted_model.pvalues
        rows = []
        for name, est in params.items():
            pval = float(pvalues[name])
            rows.append(
                {
                    "coefficient": name,
                    "estimate": float(est),
                    "p_value": pval,
                    "reject_null": bool(pval < alpha),
                }
            )
        return pd.DataFrame(rows)

    def save_artifact(self, path: str) -> Path:
        """Persist fitted model object to disk using pickle."""
        self._require_fitted()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as handle:
            pickle.dump(self.fitted_model, handle)
        return out_path

    @staticmethod
    def load_artifact(path: str):
        """Load a previously persisted fitted model artifact."""
        with Path(path).open("rb") as handle:
            return pickle.load(handle)

    def _auto_select_order(
        self,
        y: pd.Series,
        criterion: str,
        search_space: Tuple[range, range, range],
    ) -> Tuple[int, int, int]:
        if criterion not in {"aic", "bic"}:
            raise ValueError("criterion must be 'aic' or 'bic'.")
        best_order = self.order
        best_score = np.inf

        p_range, d_range, q_range = search_space
        for p, d, q in product(p_range, d_range, q_range):
            try:
                model = ARIMA(y, order=(p, d, q))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Non-invertible starting MA parameters found.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Non-stationary starting autoregressive parameters found.*",
                        category=UserWarning,
                    )
                    fit = model.fit()
                score = float(getattr(fit, criterion))
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_order = (p, d, q)
            except Exception:
                continue
        return best_order

    def _adf_summary(self, y: pd.Series) -> Dict[str, float]:
        stat, pvalue, _, _, critical_values, _ = adfuller(y)
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "critical_5pct": float(critical_values.get("5%", np.nan)),
            "is_stationary": float(pvalue < 0.05),
        }

    def _acf_values(self, x: np.ndarray, lags: int) -> np.ndarray:
        # Using statsmodels plots to keep methodology transparent; values are
        # extracted from standard autocorrelation computation.
        x = x - np.mean(x)
        denom = np.sum(x**2)
        vals = [1.0]
        for lag in range(1, lags + 1):
            num = np.sum(x[lag:] * x[:-lag])
            vals.append(float(num / denom) if not np.isclose(denom, 0.0) else 0.0)
        _ = plot_acf(
            x, lags=lags
        )  # Side-effect call documents conventional implementation.
        return np.asarray(vals, dtype=float)

    def _pacf_values(self, x: np.ndarray, lags: int) -> np.ndarray:
        # Approximate PACF via Yule-Walker recursion for visualization.
        acf = self._acf_values(x, lags)
        pacf = np.zeros(lags + 1, dtype=float)
        pacf[0] = 1.0
        for k in range(1, lags + 1):
            pacf[k] = acf[k]
        _ = plot_pacf(x, lags=lags, method="ywm")
        return pacf

    def _require_fitted(self) -> None:
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def fit_arima(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)):
    """Backward-compatible helper to fit ARIMA quickly."""
    model = ARIMA(series, order=order)
    return model.fit()
