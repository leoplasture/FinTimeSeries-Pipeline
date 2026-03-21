"""Evaluation metrics for forecasting, risk, and interval diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    y_t, y_p = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    y_t, y_p = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_t - y_p)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute percentage error in percent units."""
    y_t, y_p = _align(y_true, y_pred)
    denom = np.where(np.isclose(y_t, 0.0), np.nan, y_t)
    return float(np.nanmean(np.abs((y_t - y_p) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error in percent units."""
    y_t, y_p = _align(y_true, y_pred)
    denom = (np.abs(y_t) + np.abs(y_p)) / 2.0
    denom = np.where(np.isclose(denom, 0.0), np.nan, denom)
    return float(np.nanmean(np.abs(y_t - y_p) / denom) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R-squared."""
    y_t, y_p = _align(y_true, y_pred)
    ss_res: float = float(np.sum((y_t - y_p) ** 2))
    ss_tot: float = float(np.sum((y_t - np.mean(y_t)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sign prediction accuracy for returns or first differences."""
    y_t, y_p = _align(y_true, y_pred)
    if y_t.size < 2:
        return 0.0
    true_direction = np.sign(np.diff(y_t))
    pred_direction = np.sign(np.diff(y_p))
    return float(np.mean(true_direction == pred_direction))


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Compute empirical coverage rate for prediction intervals."""
    y_t, lo = _align(y_true, lower)
    y_t, up = _align(y_t, upper)
    covered = (y_t >= lo) & (y_t <= up)
    return float(np.mean(covered))


def interval_average_length(lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute average interval width."""
    lo, up = _align(lower, upper)
    return float(np.mean(up - lo))


def summarize_forecast_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    lower: Optional[Iterable[float]] = None,
    upper: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Return a standard metrics summary for model comparison tables."""
    y_t = np.asarray(list(y_true), dtype=float)
    y_p = np.asarray(list(y_pred), dtype=float)

    summary = {
        "mae": mae(y_t, y_p),
        "rmse": rmse(y_t, y_p),
        "mape": mape(y_t, y_p),
        "smape": smape(y_t, y_p),
        "r2": r2_score(y_t, y_p),
        "directional_accuracy": directional_accuracy(y_t, y_p),
    }

    if lower is not None and upper is not None:
        lo = np.asarray(list(lower), dtype=float)
        up = np.asarray(list(upper), dtype=float)
        summary["interval_coverage"] = interval_coverage(y_t, lo, up)
        summary["interval_avg_length"] = interval_average_length(lo, up)

    return summary


def _align(a: np.ndarray, b: np.ndarray):
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    n = min(a_np.size, b_np.size)
    if n == 0:
        raise ValueError("Input arrays must contain at least one value.")
    return a_np[:n], b_np[:n]
