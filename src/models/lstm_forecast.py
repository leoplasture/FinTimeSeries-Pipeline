"""LSTM-style deep learning baseline for univariate time-series forecasting.

The implementation prefers TensorFlow/Keras LSTM when available and falls back
to an `sklearn` MLP baseline if TensorFlow is unavailable, so the pipeline
remains runnable in lightweight environments.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

try:
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    _HAS_TF = True
except Exception:  # pragma: no cover - depends on runtime environment
    _HAS_TF = False


class LSTMForecastModel:
    """Train a deep learning baseline for sequence forecasting.

    Parameters:
       window_size (int): Number of lag observations per training sample.
       hidden_units (int): Hidden units for LSTM/MLP model.
       epochs (int): Number of epochs when TensorFlow backend is available.
       batch_size (int): Batch size for TensorFlow training.
       backend (str): `auto`, `tensorflow`, or `sklearn`.
    """

    def __init__(
        self,
        window_size: int = 20,
        hidden_units: int = 32,
        epochs: int = 25,
        batch_size: int = 32,
        backend: str = "auto",
    ) -> None:
        self.window_size = window_size
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.backend = backend
        self.model = None
        self.last_train_shape: Optional[Tuple[int, ...]] = None

    def fit(self, series: Iterable[float]) -> "LSTMForecastModel":
        """Fit model on univariate series."""
        arr = np.asarray(list(series), dtype=float)
        x, y = build_lstm_input(arr, window_size=self.window_size)
        if x.size == 0:
            raise ValueError("Series too short for the configured window_size.")

        selected = self._select_backend()
        if selected == "tensorflow":
            x_tf = x.reshape((x.shape[0], x.shape[1], 1))
            model = Sequential(
                [
                    LSTM(self.hidden_units, input_shape=(self.window_size, 1)),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                x_tf, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0
            )
            self.model = model
            self.last_train_shape = x_tf.shape
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden_units,), max_iter=600, random_state=42
            )
            mlp.fit(x, y)
            self.model = mlp
            self.last_train_shape = x.shape
        return self

    def forecast(self, history: Iterable[float], steps: int = 30) -> np.ndarray:
        """Recursive multi-step forecasting from latest history window."""
        self._require_fitted()
        hist = np.asarray(list(history), dtype=float)
        if hist.size < self.window_size:
            raise ValueError("History length must be at least window_size.")

        preds = []
        state = hist.copy()
        for _ in range(steps):
            window = state[-self.window_size :]
            if self._select_backend() == "tensorflow":
                x_in = window.reshape((1, self.window_size, 1))
                next_pred = float(self.model.predict(x_in, verbose=0).ravel()[0])
            else:
                x_in = window.reshape((1, self.window_size))
                next_pred = float(self.model.predict(x_in).ravel()[0])
            preds.append(next_pred)
            state = np.append(state, next_pred)
        return np.asarray(preds, dtype=float)

    def evaluate(
        self, y_true: Iterable[float], y_pred: Iterable[float]
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and MAPE for forecast evaluation."""
        y_t = np.asarray(list(y_true), dtype=float)
        y_p = np.asarray(list(y_pred), dtype=float)
        n = min(y_t.size, y_p.size)
        if n == 0:
            raise ValueError("y_true and y_pred must have aligned observations.")
        y_t = y_t[:n]
        y_p = y_p[:n]

        denom = np.where(np.isclose(y_t, 0.0), np.nan, y_t)
        return {
            "mae": float(mean_absolute_error(y_t, y_p)),
            "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "mape": float(np.nanmean(np.abs((y_t - y_p) / denom)) * 100.0),
        }

    def plot_forecast(
        self, history: Iterable[float], forecast: Iterable[float]
    ) -> go.Figure:
        """Visualize historical observations and forecast trajectory."""
        hist = np.asarray(list(history), dtype=float)
        fc = np.asarray(list(forecast), dtype=float)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=np.arange(hist.size), y=hist, mode="lines", name="History")
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(hist.size, hist.size + fc.size),
                y=fc,
                mode="lines",
                name="Forecast",
            )
        )
        fig.update_layout(title="LSTM Baseline Forecast", template="plotly_white")
        return fig

    def _select_backend(self) -> str:
        if self.backend == "auto":
            return "tensorflow" if _HAS_TF else "sklearn"
        if self.backend == "tensorflow" and not _HAS_TF:
            raise ImportError(
                "TensorFlow backend requested but tensorflow is not installed."
            )
        if self.backend not in {"tensorflow", "sklearn"}:
            raise ValueError("backend must be one of: auto, tensorflow, sklearn")
        return self.backend

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def build_lstm_input(series: np.ndarray, window_size: int = 20):
    """Transform a 1D series into supervised sequence windows."""
    if len(series) <= window_size:
        return np.array([]), np.array([])
    x, y = [], []
    for i in range(window_size, len(series)):
        x.append(series[i - window_size : i])
        y.append(series[i])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
