"""Hypothesis tests for time-series econometrics.

Theoretical notes:
1. ADF/KPSS tests evaluate stationarity from complementary null hypotheses.
2. Cointegration captures stable long-run equilibrium among non-stationary series.
3. Granger causality tests predictive content, not structural causation.
4. Structural break tests (CUSUM) assess parameter stability over time.
5. Diebold-Mariano compares predictive accuracy via loss-differential means.
"""

from __future__ import annotations

import contextlib
from io import StringIO
from statistics import NormalDist
from typing import Any, Dict, Iterable, Optional
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests, kpss


_STANDARD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


class TimeSeriesTests:
    """Run core inferential tests for financial time series."""

    def stationarity_test(
        self,
        df: pd.DataFrame,
        column: str = "Close",
        test: str = "adf",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Run stationarity test (ADF or KPSS).

        Parameters:
           df (pandas.DataFrame): Input DataFrame.
           column (str): Column name.
           test (str): `adf` or `kpss`.
           significance (float): Type-I error threshold.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")
        series = df[column].dropna().astype(float)
        if series.size < 20:
            raise ValueError("At least 20 observations are recommended.")

        if test == "adf":
            stat, p_value, _, _, critical, _ = adfuller(series)
            crit = float(critical.get("5%", np.nan))
            reject = bool(p_value < significance)
            conclusion = "Reject unit root null; evidence of stationarity."
            if not reject:
                conclusion = (
                    "Fail to reject unit root null; evidence of non-stationarity."
                )
            return self._result(float(stat), float(p_value), crit, reject, conclusion)

        if test == "kpss":
            with warnings.catch_warnings():
                # KPSS p-values are table-bounded and may trigger interpolation
                # warnings for very extreme statistics.
                warnings.simplefilter("ignore", InterpolationWarning)
                stat, p_value, _, critical = kpss(series, regression="c", nlags="auto")
            crit = float(critical.get("5%", np.nan))
            reject = bool(p_value < significance)
            conclusion = "Reject stationarity null; evidence of non-stationarity."
            if not reject:
                conclusion = (
                    "Fail to reject stationarity null; series appears stationary."
                )
            return self._result(float(stat), float(p_value), crit, reject, conclusion)

        raise ValueError("test must be 'adf' or 'kpss'.")

    def cointegration_test(
        self,
        series1: Iterable[float],
        series2: Iterable[float],
        test: str = "engle_granger",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Run Engle-Granger two-step cointegration test."""
        if test != "engle_granger":
            raise NotImplementedError("Only engle_granger is currently implemented.")

        x = np.asarray(list(series1), dtype=float)
        y = np.asarray(list(series2), dtype=float)
        n = min(x.size, y.size)
        if n < 30:
            raise ValueError("At least 30 aligned observations are recommended.")
        stat, p_value, critical = coint(x[:n], y[:n])
        crit = float(critical[1])  # 5% critical value
        reject = bool(p_value < significance)
        conclusion = "Reject no-cointegration null; evidence of long-run equilibrium."
        if not reject:
            conclusion = "Fail to reject no-cointegration null."
        return self._result(float(stat), float(p_value), crit, reject, conclusion)

    def granger_causality_test(
        self,
        series1: Iterable[float],
        series2: Iterable[float],
        max_lag: int = 4,
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Test whether series1 Granger-causes series2 using F-statistics."""
        x = np.asarray(list(series1), dtype=float)
        y = np.asarray(list(series2), dtype=float)
        n = min(x.size, y.size)
        if n < max(40, 5 * max_lag):
            raise ValueError("Insufficient observations for requested max_lag.")

        data = pd.DataFrame({"y": y[:n], "x": x[:n]})
        with contextlib.redirect_stdout(StringIO()):
            gc = grangercausalitytests(data[["y", "x"]], maxlag=max_lag)

        pvals = []
        stats = []
        for lag in range(1, max_lag + 1):
            f_stat, p_val, _, _ = gc[lag][0]["ssr_ftest"]
            pvals.append(float(p_val))
            stats.append(float(f_stat))

        best_idx = int(np.argmin(pvals))
        min_p = pvals[best_idx]
        best_stat = stats[best_idx]
        reject = bool(min_p < significance)
        conclusion = f"Reject no-Granger-causality null at lag {best_idx + 1}."
        if not reject:
            conclusion = "Fail to reject no-Granger-causality null."
        return self._result(best_stat, min_p, np.nan, reject, conclusion)

    def structural_break_test(
        self,
        df: pd.DataFrame,
        test: str = "cusum",
        column: str = "Close",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Test parameter stability with CUSUM on trend-regression residuals."""
        if test != "cusum":
            raise NotImplementedError("Only CUSUM test is implemented.")
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")

        y = df[column].dropna().astype(float).to_numpy()
        n = y.size
        if n < 30:
            raise ValueError("At least 30 observations are recommended for CUSUM.")

        x = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        stat, p_value, critical = breaks_cusumolsresid(resid, ddof=x.shape[1])

        # statsmodels may return critical values as scalar, vector, or nested tuples.
        if isinstance(critical, (tuple, list, np.ndarray)):
            crit = np.nan
            for item in critical:
                if isinstance(item, (tuple, list, np.ndarray)):
                    if len(item) >= 2:
                        try:
                            crit = float(item[1])
                            break
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        crit = float(item)
                        break
                    except (TypeError, ValueError):
                        continue
            if np.isnan(crit):
                crit = float("nan")
        else:
            crit = float(critical)

        reject = bool(p_value < significance)
        conclusion = "Reject stability null; structural break likely present."
        if not reject:
            conclusion = "Fail to reject stability null; no clear structural break."
        return self._result(float(stat), float(p_value), crit, reject, conclusion)

    def compare_models(
        self,
        model1: Any,
        model2: Any,
        test: str = "diebold_mariano",
        significance: float = 0.05,
        power: int = 2,
    ) -> Dict[str, Any]:
        """Compare forecast accuracy with Diebold-Mariano test.

        Parameters:
           model1 (Any): First forecast model or errors array.
           model2 (Any): Second forecast model or errors array.
           test (str): Test name. Supports `diebold_mariano`.
           significance (float): Type-I error threshold.
           power (int): Loss power, 1 for absolute and 2 for squared error.
        """
        if test != "diebold_mariano":
            raise NotImplementedError("Only diebold_mariano is implemented.")

        e1 = self._extract_errors(model1)
        e2 = self._extract_errors(model2)
        n = min(e1.size, e2.size)
        if n < 20:
            raise ValueError("At least 20 error observations are required for DM test.")

        d = np.abs(e1[:n]) ** power - np.abs(e2[:n]) ** power
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        if np.isclose(d_var, 0.0):
            raise ValueError(
                "Loss differential variance is near zero; DM test is undefined."
            )

        dm_stat = float(d_mean / np.sqrt(d_var / n))
        p_value = float(2 * (1 - _STANDARD_NORMAL.cdf(np.abs(dm_stat))))
        reject = bool(p_value < significance)
        conclusion = "Reject equal-accuracy null; predictive performance differs."
        if not reject:
            conclusion = "Fail to reject equal-accuracy null."
        return self._result(dm_stat, p_value, 1.96, reject, conclusion)

    def apply_multiple_testing_correction(
        self,
        p_values: Iterable[float],
        method: str = "bonferroni",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Apply Bonferroni or FDR multiple-testing correction."""
        values = np.asarray(list(p_values), dtype=float)
        if values.size == 0:
            raise ValueError("p_values must be non-empty.")

        selected = "bonferroni" if method == "bonferroni" else "fdr_bh"
        reject, adjusted, _, _ = multipletests(values, alpha=alpha, method=selected)
        return {
            "method": method,
            "alpha": float(alpha),
            "reject": reject.tolist(),
            "adjusted_p_values": adjusted.tolist(),
        }

    def plot_test_results(self, result_map: Dict[str, Dict[str, Any]]) -> go.Figure:
        """Visualize p-values and critical threshold across tests."""
        tests = list(result_map.keys())
        pvals = [float(result_map[t]["p_value"]) for t in tests]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=tests, y=pvals, name="p-value"))
        fig.add_hline(
            y=0.05, line_dash="dash", line_color="red", annotation_text="alpha=0.05"
        )
        fig.update_layout(
            title="Hypothesis Test p-values",
            xaxis_title="Test",
            yaxis_title="p-value",
            template="plotly_white",
        )
        return fig

    def _extract_errors(self, model_or_errors: Any) -> np.ndarray:
        if isinstance(model_or_errors, (list, tuple, np.ndarray, pd.Series)):
            return np.asarray(model_or_errors, dtype=float)
        if isinstance(model_or_errors, dict) and "errors" in model_or_errors:
            return np.asarray(model_or_errors["errors"], dtype=float)
        if hasattr(model_or_errors, "resid"):
            return np.asarray(model_or_errors.resid, dtype=float)
        raise TypeError(
            "Provide error arrays, a dict with 'errors', or model with residuals."
        )

    def _result(
        self,
        test_statistic: float,
        p_value: float,
        critical_value: float,
        reject_null: bool,
        conclusion: str,
    ) -> Dict[str, Any]:
        return {
            "test_statistic": float(test_statistic),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "reject_null": bool(reject_null),
            "conclusion": conclusion,
        }


def two_sample_mean_test(
    sample_a: np.ndarray, sample_b: np.ndarray
) -> Dict[str, float]:
    """Compatibility helper that performs a two-sided z-style mean comparison."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if a.size < 2 or b.size < 2:
        raise ValueError("Both samples must contain at least two observations.")

    mean_diff = float(a.mean() - b.mean())
    se = np.sqrt(np.var(a, ddof=1) / a.size + np.var(b, ddof=1) / b.size)
    if np.isclose(se, 0.0):
        raise ValueError("Standard error is near zero; test statistic undefined.")
    z = mean_diff / se
    p_value = float(2 * (1 - _STANDARD_NORMAL.cdf(abs(z))))
    return {"test_statistic": float(z), "p_value": p_value}
