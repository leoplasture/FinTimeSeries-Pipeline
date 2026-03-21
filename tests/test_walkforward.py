"""Unit tests for walk-forward evaluation helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from run import _walkforward_on_series


def test_walkforward_helper_returns_detail_and_summary() -> None:
    """Walk-forward helper should produce per-window metrics and aggregate summary."""
    rng = np.random.default_rng(8)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 220)))

    detail, summary = _walkforward_on_series(
        series=series,
        order=(1, 1, 1),
        initial_train_size=120,
        horizon=10,
        step_size=10,
        max_windows=5,
        auto_order=False,
        conf_level=0.95,
    )

    assert not detail.empty, "Detail output should contain evaluated windows."
    assert "mean_rmse" in summary, "Summary should include mean RMSE."
    assert summary["n_windows"] == 5.0, "Summary window count should match max_windows."


def test_walkforward_helper_rejects_short_series() -> None:
    """Walk-forward helper should reject insufficient series length."""
    series = pd.Series(np.linspace(1, 10, 50))
    with pytest.raises(ValueError, match="too short"):
        _walkforward_on_series(
            series=series,
            order=(1, 1, 1),
            initial_train_size=45,
            horizon=10,
            step_size=5,
            max_windows=2,
            auto_order=False,
            conf_level=0.95,
        )
