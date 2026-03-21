"""GARCH-family volatility modeling for forecasting and risk analytics.

Theoretical notes:
1. ARCH (Engle, 1982) models time-varying conditional variance.
2. GARCH (Bollerslev, 1986) adds lagged variance terms for persistence.
3. Volatility forecasts support risk management via VaR/Expected Shortfall.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from arch import arch_model
except Exception:  # pragma: no cover - handled at runtime by informative error
    arch_model = None


_STANDARD_NORMAL = NormalDist(0.0, 1.0)


class GARCHModel:
    """Wrapper for GARCH/EGARCH/GJR-GARCH fitting and volatility forecasting.

    Parameters:
       order (tuple[int, int]): Volatility order (p, q).
       distribution (str): Error distribution (e.g., normal, t, skewt).
       conf_level (float): Default confidence level used in interval/risk outputs.
       variant (str): One of `GARCH`, `EGARCH`, `GJR-GARCH`.
    """

    def __init__(
        self,
        order: Tuple[int, int] = (1, 1),
        distribution: str = "normal",
        conf_level: float = 0.95,
        variant: str = "GARCH",
    ) -> None:
        self.order = order
        self.distribution = distribution
        self.conf_level = conf_level
        self.variant = variant.upper()
        self.returns: Optional[pd.Series] = None
        self.model = None
        self.fitted_model = None
        self.last_vol_forecast: Optional[pd.DataFrame] = None

    def fit(self, returns: Iterable[float]) -> "GARCHModel":
        """Fit selected GARCH-family model on return series.

        Parameters:
           returns (Iterable[float]): Return series, preferably in percentage units.
        """
        if arch_model is None:
            raise ImportError(
                "arch package is required for GARCHModel. Install with `pip install arch`."
            )

        ret = pd.Series(list(returns), dtype=float).dropna()
        if ret.size < 100:
            raise ValueError(
                "At least 100 observations are recommended for GARCH fitting."
            )
        self.returns = ret

        p, q = self.order
        if self.variant == "GARCH":
            self.model = arch_model(
                ret, mean="Constant", vol="GARCH", p=p, q=q, dist=self.distribution
            )
        elif self.variant == "EGARCH":
            self.model = arch_model(
                ret, mean="Constant", vol="EGARCH", p=p, q=q, dist=self.distribution
            )
        elif self.variant in {"GJR-GARCH", "GJRGARCH", "GJR"}:
            self.model = arch_model(
                ret, mean="Constant", vol="GARCH", p=p, o=1, q=q, dist=self.distribution
            )
        else:
            raise ValueError("variant must be one of: GARCH, EGARCH, GJR-GARCH")

        self.fitted_model = self.model.fit(disp="off")
        return self

    def forecast_volatility(
        self, steps: int = 30, confidence_interval: bool = True
    ) -> pd.DataFrame:
        """Forecast future conditional volatility.

        Returns:
           pandas.DataFrame: Columns include `variance`, `volatility`, and optional bands.
        """
        self._require_fitted()
        fc = self.fitted_model.forecast(horizon=steps, reindex=False)
        variance = fc.variance.iloc[-1].to_numpy(dtype=float)
        volatility = np.sqrt(np.maximum(variance, 0.0))
        out = pd.DataFrame({"variance": variance, "volatility": volatility})

        if confidence_interval:
            alpha = 1.0 - self.conf_level
            z = _STANDARD_NORMAL.inv_cdf(1 - alpha / 2)
            se = np.maximum(volatility / np.sqrt(max(self.returns.size, 1)), 1e-10)
            out["lower"] = np.maximum(volatility - z * se, 0.0)
            out["upper"] = volatility + z * se

        self.last_vol_forecast = out
        return out

    def plot_volatility(self, steps: int = 30) -> go.Figure:
        """Visualize historical and forecast volatility to show clustering/persistence."""
        self._require_fitted()
        if self.last_vol_forecast is None or len(self.last_vol_forecast) != steps:
            self.forecast_volatility(steps=steps, confidence_interval=True)

        hist_vol = self.fitted_model.conditional_volatility
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=np.asarray(hist_vol, dtype=float), mode="lines", name="Historical Vol"
            )
        )
        x_future = np.arange(len(hist_vol), len(hist_vol) + steps)
        fig.add_trace(
            go.Scatter(
                x=x_future,
                y=self.last_vol_forecast["volatility"],
                mode="lines",
                name="Forecast Vol",
            )
        )
        if "lower" in self.last_vol_forecast and "upper" in self.last_vol_forecast:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_future, x_future[::-1]]),
                    y=np.concatenate(
                        [
                            self.last_vol_forecast["upper"].to_numpy(),
                            self.last_vol_forecast["lower"].to_numpy()[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor="rgba(255, 127, 14, 0.2)",
                    line={"color": "rgba(255,255,255,0)"},
                    name=f"{int(self.conf_level*100)}% CI",
                )
            )
        fig.update_layout(
            title=f"{self.variant} Volatility Forecast", template="plotly_white"
        )
        return fig

    def compute_risk_metrics(self, confidence_level: float = 0.95) -> Dict[str, float]:
        """Compute one-step VaR and Expected Shortfall from latest volatility estimate."""
        self._require_fitted()
        if not (0 < confidence_level < 1):
            raise ValueError("confidence_level must be between 0 and 1.")

        mu = float(self.fitted_model.params.get("mu", 0.0))
        sigma = float(self.fitted_model.conditional_volatility.iloc[-1])
        alpha = 1.0 - confidence_level
        z_alpha = _STANDARD_NORMAL.inv_cdf(alpha)
        var = mu + sigma * z_alpha

        # ES for Gaussian losses under return convention.
        phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z_alpha * z_alpha)
        es = mu - sigma * (phi / max(alpha, 1e-8))
        return {
            "confidence_level": float(confidence_level),
            "VaR": float(var),
            "ExpectedShortfall": float(es),
        }

    def backtest(
        self,
        strategy: str = "long_short",
        lookback: int = 252,
        window_mode: str = "rolling",
    ) -> Dict[str, float]:
        """Run a simple volatility-aware strategy backtest.

        Strategy:
           `long_short` uses sign(mean forecast return) with inverse-vol scaling.

        Parameters:
           strategy (str): Currently supports `long_short`.
           lookback (int): Initial calibration window length.
           window_mode (str): `rolling` for fixed-size window or `expanding`.
        """
        self._require_fitted()
        returns = self.returns.dropna().to_numpy(dtype=float)
        if returns.size <= lookback + 2:
            raise ValueError("Not enough observations for backtest lookback window.")
        if window_mode not in {"rolling", "expanding"}:
            raise ValueError("window_mode must be 'rolling' or 'expanding'.")

        pnl = []
        for t in range(lookback, returns.size - 1):
            if window_mode == "rolling":
                window = returns[t - lookback : t]
            else:
                window = returns[:t]
            signal = np.sign(np.mean(window))
            if strategy != "long_short":
                raise ValueError("Only long_short strategy is implemented.")
            vol = np.std(window, ddof=1)
            weight = 1.0 / max(vol, 1e-6)
            pnl.append(signal * weight * returns[t + 1])

        pnl_arr = np.asarray(pnl, dtype=float)
        cumulative_path = np.cumsum(pnl_arr)
        running_peak = np.maximum.accumulate(cumulative_path)
        drawdowns = cumulative_path - running_peak
        max_drawdown = float(np.min(drawdowns)) if drawdowns.size else 0.0

        sharpe = float(
            np.mean(pnl_arr) / (np.std(pnl_arr, ddof=1) + 1e-12) * np.sqrt(252)
        )
        cumulative = float(np.sum(pnl_arr))
        hit_rate = float(np.mean(pnl_arr > 0))
        return {
            "sharpe": sharpe,
            "cumulative_return": cumulative,
            "hit_rate": hit_rate,
            "max_drawdown": max_drawdown,
            "trades": float(len(pnl_arr)),
            "window_mode": window_mode,
        }

    def get_confidence_interval(
        self, alpha: float = 0.05, steps: int = 30
    ) -> pd.DataFrame:
        """Return confidence interval around volatility forecasts.

        Parameters:
           alpha (float): Two-sided tail probability.
           steps (int): Forecast horizon.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")

        previous_conf = self.conf_level
        self.conf_level = 1.0 - alpha
        out = self.forecast_volatility(steps=steps, confidence_interval=True)
        self.conf_level = previous_conf
        return out[["lower", "upper"]].copy()

    def _require_fitted(self) -> None:
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def fit_garch(returns: pd.Series):
    """Backward-compatible helper for quick GARCH fit."""
    model = GARCHModel()
    return model.fit(returns)
