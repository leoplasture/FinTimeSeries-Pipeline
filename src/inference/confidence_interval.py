"""Confidence interval construction for time-series forecasts.

Theoretical notes:
1. Asymptotic intervals rely on large-sample normality of estimators and valid
   standard-error estimation under correct model specification.
2. Bootstrap intervals replace analytical assumptions with resampling and are
   often more robust in finite samples and mildly misspecified settings.
3. Coverage probability is the long-run frequency with which an interval
   contains the true value, not the posterior probability for one realized
   interval.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


class ForecastInference:
    """Build confidence and prediction intervals for forecasting models.

    Parameters:
       model (Any): Fitted model object with forecast-like behavior. The class
          supports statsmodels-like interfaces (`get_forecast`, `forecast`,
          `resid`, `fittedvalues`) and a dictionary form with
          `predictions`/`residuals`.
       conf_level (float): Confidence level in (0, 1).
    """

    def __init__(self, model: Any, conf_level: float = 0.95) -> None:
        if not (0 < conf_level < 1):
            raise ValueError("conf_level must be between 0 and 1.")
        self.model = model
        self.conf_level = conf_level

    def asymptotic_ci(self, horizon: int = 30) -> pd.DataFrame:
        """Construct asymptotic normal confidence intervals.

        Parameters:
           horizon (int): Forecast horizon.

        Returns:
           pandas.DataFrame: Columns are `forecast`, `lower`, and `upper`.
        """
        forecast = self._get_forecast_array(horizon)
        se = self._get_standard_errors(horizon, fallback_forecast=forecast)
        alpha = 1.0 - self.conf_level
        z = _STANDARD_NORMAL.inv_cdf(1 - alpha / 2)
        lower = forecast - z * se
        upper = forecast + z * se
        return pd.DataFrame({"forecast": forecast, "lower": lower, "upper": upper})

    def bootstrap_ci(
        self,
        n_bootstraps: int = 1000,
        method: str = "percentile",
        horizon: int = 30,
    ) -> pd.DataFrame:
        """Construct bootstrap confidence intervals.

        Parameters:
           n_bootstraps (int): Number of bootstrap replications.
           method (str): `percentile` or `bca` (bias-corrected accelerated).
           horizon (int): Forecast horizon.

        Returns:
           pandas.DataFrame: Columns are `forecast`, `lower`, and `upper`.
        """
        if n_bootstraps < 200:
            raise ValueError(
                "n_bootstraps should be at least 200 for stable inference."
            )
        if method not in {"percentile", "bca"}:
            raise ValueError("method must be 'percentile' or 'bca'.")

        baseline = self._get_forecast_array(horizon)
        residuals = self._get_residuals_array()
        if residuals.size < 20:
            raise ValueError("Residual series is too short for bootstrap inference.")

        boot_draws = np.empty((n_bootstraps, horizon), dtype=float)
        for i in range(n_bootstraps):
            sampled = np.random.choice(residuals, size=horizon, replace=True)
            boot_draws[i, :] = baseline + sampled

        alpha = 1.0 - self.conf_level
        if method == "percentile":
            lower = np.percentile(boot_draws, 100 * (alpha / 2), axis=0)
            upper = np.percentile(boot_draws, 100 * (1 - alpha / 2), axis=0)
        else:
            lower, upper = self._bca_bounds(boot_draws, baseline, alpha)

        return pd.DataFrame({"forecast": baseline, "lower": lower, "upper": upper})

    def prediction_interval(
        self, horizon: int = 30, conf_level: Optional[float] = None
    ) -> pd.DataFrame:
        """Compute prediction intervals that include innovation uncertainty.

        Parameters:
           horizon (int): Forecast horizon.
           conf_level (float | None): Overrides class confidence level.

        Returns:
           pandas.DataFrame: Columns are `forecast`, `lower`, and `upper`.
        """
        level = conf_level if conf_level is not None else self.conf_level
        if not (0 < level < 1):
            raise ValueError("conf_level must be between 0 and 1.")

        forecast = self._get_forecast_array(horizon)
        residuals = self._get_residuals_array()
        sigma = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 1.0

        # Horizon-dependent uncertainty inflation approximates multi-step errors.
        step_scale = np.sqrt(np.arange(1, horizon + 1, dtype=float))
        pred_se = sigma * step_scale
        alpha = 1 - level
        z = _STANDARD_NORMAL.inv_cdf(1 - alpha / 2)
        lower = forecast - z * pred_se
        upper = forecast + z * pred_se
        return pd.DataFrame({"forecast": forecast, "lower": lower, "upper": upper})

    def coverage_simulation(
        self, true_values: np.ndarray, n_simulations: int = 500
    ) -> Dict[str, float]:
        """Estimate empirical coverage probability and average CI length.

        Parameters:
           true_values (numpy.ndarray): Realized values to check interval coverage.
           n_simulations (int): Number of simulated interval draws.

        Returns:
           dict: Coverage and average length metrics.
        """
        truth = np.asarray(true_values, dtype=float)
        horizon = int(truth.size)
        if horizon == 0:
            raise ValueError("true_values must be non-empty.")
        if n_simulations < 50:
            raise ValueError("n_simulations should be at least 50.")

        intervals = self.bootstrap_ci(
            n_bootstraps=max(300, n_simulations),
            method="percentile",
            horizon=horizon,
        )
        covered = (truth >= intervals["lower"].to_numpy()) & (
            truth <= intervals["upper"].to_numpy()
        )
        lengths = intervals["upper"].to_numpy() - intervals["lower"].to_numpy()
        return {
            "coverage_probability": float(np.mean(covered)),
            "average_ci_length": float(np.mean(lengths)),
            "target_confidence": float(self.conf_level),
        }

    def compare_methods(
        self, horizon: int = 30, n_bootstraps: int = 1000
    ) -> pd.DataFrame:
        """Compare interval methods by average width.

        Parameters:
           horizon (int): Forecast horizon.
           n_bootstraps (int): Number of bootstrap replications.

        Returns:
           pandas.DataFrame: Method comparison table.
        """
        asym = self.asymptotic_ci(horizon=horizon)
        pct = self.bootstrap_ci(
            n_bootstraps=n_bootstraps, method="percentile", horizon=horizon
        )
        bca = self.bootstrap_ci(
            n_bootstraps=n_bootstraps, method="bca", horizon=horizon
        )

        def _avg_width(df: pd.DataFrame) -> float:
            return float(np.mean(df["upper"] - df["lower"]))

        return pd.DataFrame(
            {
                "method": ["asymptotic", "bootstrap_percentile", "bootstrap_bca"],
                "avg_interval_length": [
                    _avg_width(asym),
                    _avg_width(pct),
                    _avg_width(bca),
                ],
            }
        )

    def plot_forecast_with_bands(
        self,
        horizon: int = 30,
        method: str = "asymptotic",
        n_bootstraps: int = 1000,
    ) -> go.Figure:
        """Plot forecasts with confidence bands.

        Parameters:
           horizon (int): Forecast horizon.
           method (str): `asymptotic`, `bootstrap_percentile`, or `bootstrap_bca`.
           n_bootstraps (int): Used for bootstrap methods.

        Returns:
           plotly.graph_objects.Figure: Interactive confidence-band chart.
        """
        if method == "asymptotic":
            ci = self.asymptotic_ci(horizon=horizon)
        elif method == "bootstrap_percentile":
            ci = self.bootstrap_ci(
                n_bootstraps=n_bootstraps, method="percentile", horizon=horizon
            )
        elif method == "bootstrap_bca":
            ci = self.bootstrap_ci(
                n_bootstraps=n_bootstraps, method="bca", horizon=horizon
            )
        else:
            raise ValueError("Unsupported method.")

        x = np.arange(1, horizon + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ci["forecast"], mode="lines", name="Forecast"))
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate(
                    [ci["upper"].to_numpy(), ci["lower"].to_numpy()[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)",
                line={"color": "rgba(255,255,255,0)"},
                name=f"{int(self.conf_level*100)}% CI",
            )
        )
        fig.update_layout(
            title=f"Forecast with {method} confidence bands",
            xaxis_title="Forecast Horizon",
            yaxis_title="Forecast Value",
            template="plotly_white",
        )
        return fig

    def _get_forecast_array(self, horizon: int) -> np.ndarray:
        if hasattr(self.model, "get_forecast"):
            out = self.model.get_forecast(steps=horizon)
            if hasattr(out, "predicted_mean"):
                return np.asarray(out.predicted_mean, dtype=float)
        if hasattr(self.model, "forecast"):
            return np.asarray(self.model.forecast(steps=horizon), dtype=float)
        if isinstance(self.model, dict) and "predictions" in self.model:
            base = np.asarray(self.model["predictions"], dtype=float)
            if base.size < horizon:
                raise ValueError(
                    "Model prediction array shorter than requested horizon."
                )
            return base[:horizon]
        raise TypeError("Model does not expose a supported forecast interface.")

    def _get_standard_errors(
        self, horizon: int, fallback_forecast: np.ndarray
    ) -> np.ndarray:
        if hasattr(self.model, "get_forecast"):
            out = self.model.get_forecast(steps=horizon)
            if hasattr(out, "se_mean"):
                return np.asarray(out.se_mean, dtype=float)
            if hasattr(out, "var_pred_mean"):
                return np.sqrt(np.asarray(out.var_pred_mean, dtype=float))

        residuals = self._get_residuals_array()
        sigma = (
            float(np.std(residuals, ddof=1))
            if residuals.size > 1
            else float(np.std(fallback_forecast, ddof=1))
        )
        if np.isnan(sigma) or np.isclose(sigma, 0.0):
            sigma = 1.0
        return np.full(horizon, sigma, dtype=float)

    def _get_residuals_array(self) -> np.ndarray:
        if hasattr(self.model, "resid"):
            return np.asarray(self.model.resid, dtype=float)
        if isinstance(self.model, dict) and "residuals" in self.model:
            return np.asarray(self.model["residuals"], dtype=float)
        if (
            hasattr(self.model, "fittedvalues")
            and hasattr(self.model, "model")
            and hasattr(self.model.model, "endog")
        ):
            observed = np.asarray(self.model.model.endog, dtype=float)
            fitted = np.asarray(self.model.fittedvalues, dtype=float)
            n = min(observed.size, fitted.size)
            return observed[:n] - fitted[:n]
        raise TypeError(
            "Model does not expose residuals required for bootstrap inference."
        )

    def _bca_bounds(
        self, boot_draws: np.ndarray, baseline: np.ndarray, alpha: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_boot, horizon = boot_draws.shape
        lower = np.empty(horizon, dtype=float)
        upper = np.empty(horizon, dtype=float)

        for j in range(horizon):
            sample = boot_draws[:, j]
            theta_hat = baseline[j]
            prop_less = np.clip(np.mean(sample < theta_hat), 1e-6, 1 - 1e-6)
            z0 = _STANDARD_NORMAL.inv_cdf(float(prop_less))

            jack = np.delete(
                sample, np.arange(0, n_boot, max(1, n_boot // min(n_boot, 100)))
            )
            if jack.size == 0:
                jack = sample
            jack_mean = np.mean(jack)
            num = np.sum((jack_mean - jack) ** 3)
            den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5 + 1e-12)
            a = num / den

            z_alpha_low = _STANDARD_NORMAL.inv_cdf(alpha / 2)
            z_alpha_high = _STANDARD_NORMAL.inv_cdf(1 - alpha / 2)
            adj_low = _STANDARD_NORMAL.cdf(
                z0 + (z0 + z_alpha_low) / (1 - a * (z0 + z_alpha_low) + 1e-12)
            )
            adj_high = _STANDARD_NORMAL.cdf(
                z0 + (z0 + z_alpha_high) / (1 - a * (z0 + z_alpha_high) + 1e-12)
            )

            lower[j] = np.quantile(sample, np.clip(adj_low, 0.0, 1.0))
            upper[j] = np.quantile(sample, np.clip(adj_high, 0.0, 1.0))

        return lower, upper


def mean_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Compatibility helper for mean confidence interval from sample data.

    Parameters:
       data (numpy.ndarray): Input observations.
       confidence (float): Confidence level.

    Returns:
       tuple[float, float]: Lower and upper bounds for mean interval.
    """
    sample = np.asarray(data, dtype=float)
    if sample.size == 0:
        raise ValueError("Input data must be non-empty.")
    z = _STANDARD_NORMAL.inv_cdf(0.5 + confidence / 2.0)
    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    margin = z * std / np.sqrt(sample.size)
    return mean - margin, mean + margin
