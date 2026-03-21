"""Streamlit dashboard for interactive financial time-series analysis."""

from __future__ import annotations

import json
from datetime import date
from io import StringIO
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.clean import DataCleaner
from src.data.fetch import DataFetcher
from src.inference.hypothesis_test import TimeSeriesTests
from src.models.arima import ARIMAModel
from src.models.garch import GARCHModel
from src.models.lstm_forecast import LSTMForecastModel
from src.utils.config import load_config
from src.utils.metrics import summarize_forecast_metrics


@st.cache_data(show_spinner=False)
def _load_runtime_config(path: str = "config/params.yaml") -> Dict:
    try:
        return load_config(path)
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _fetch_and_clean(
    ticker: str,
    start_date: str,
    end_date: str,
    missing_method: str,
    outlier_threshold: float,
) -> Tuple[pd.DataFrame, Dict]:
    fetcher = DataFetcher(cache_dir="data/cache", use_cache=True)
    raw = fetcher.fetch_stock_data(
        ticker=ticker, start_date=start_date, end_date=end_date
    )
    cleaner = DataCleaner(
        missing_method=missing_method, outlier_threshold=outlier_threshold
    )
    cleaned, report = cleaner.clean_pipeline(raw, ticker=ticker)
    return cleaned, report


@st.cache_data(show_spinner=False)
def _run_arima(df: pd.DataFrame, horizon: int, conf_level: float):
    model = ARIMAModel(conf_level=conf_level)
    model.fit(df, column="Close", auto_order=True)
    fc = model.forecast(steps=horizon, confidence_interval=True)
    coeff_test = model.hypothesis_test_coefficients(alpha=1.0 - conf_level)
    return fc, coeff_test, model.order


@st.cache_data(show_spinner=False)
def _run_garch(returns: pd.Series, horizon: int, conf_level: float, variant: str):
    garch = GARCHModel(order=(1, 1), conf_level=conf_level, variant=variant)
    garch.fit(returns)
    vol_fc = garch.forecast_volatility(steps=horizon, confidence_interval=True)
    risk = garch.compute_risk_metrics(confidence_level=conf_level)
    return vol_fc, risk


@st.cache_data(show_spinner=False)
def _run_lstm(close: pd.Series, horizon: int, window_size: int):
    model = LSTMForecastModel(window_size=window_size, backend="auto")
    model.fit(close.values)
    preds = model.forecast(close.values, steps=horizon)
    return preds


@st.cache_data(show_spinner=False)
def _run_tests(df: pd.DataFrame):
    tester = TimeSeriesTests()
    results = {
        "ADF Stationarity": tester.stationarity_test(df, column="Close", test="adf"),
        "KPSS Stationarity": tester.stationarity_test(df, column="Close", test="kpss"),
        "CUSUM Break": tester.structural_break_test(df, test="cusum", column="Close"),
    }
    correction = tester.apply_multiple_testing_correction(
        p_values=[r["p_value"] for r in results.values()],
        method="fdr",
        alpha=0.05,
    )
    return results, correction


def _candlestick_volume_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} OHLC",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            yaxis="y2",
            name="Volume",
            marker_color="rgba(55, 83, 109, 0.35)",
        )
    )
    fig.update_layout(
        title=f"{ticker} Price and Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2={
            "title": "Volume",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
        },
        xaxis_rangeslider_visible=False,
        legend={"orientation": "h"},
        template="plotly_white",
        height=560,
    )
    return fig


def _forecast_figure(history: pd.Series, forecast: pd.DataFrame) -> go.Figure:
    hist_x = np.arange(history.size)
    fc_x = np.arange(history.size, history.size + len(forecast))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=hist_x, y=history.to_numpy(), mode="lines", name="History")
    )
    fig.add_trace(
        go.Scatter(x=fc_x, y=forecast["forecast"], mode="lines", name="Forecast")
    )
    if {"lower", "upper"}.issubset(forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([fc_x, fc_x[::-1]]),
                y=np.concatenate(
                    [forecast["upper"].to_numpy(), forecast["lower"].to_numpy()[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.22)",
                line={"color": "rgba(255,255,255,0)"},
                name="Confidence Band",
            )
        )
    fig.update_layout(
        title="Forecast with Confidence Interval", template="plotly_white", height=520
    )
    return fig


def _volatility_figure(df: pd.DataFrame) -> go.Figure:
    x = np.arange(1, len(df) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=df["volatility"], mode="lines", name="Forecast Volatility")
    )
    if {"lower", "upper"}.issubset(df.columns):
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate(
                    [df["upper"].to_numpy(), df["lower"].to_numpy()[::-1]]
                ),
                fill="toself",
                fillcolor="rgba(255, 127, 14, 0.22)",
                line={"color": "rgba(255,255,255,0)"},
                name="Volatility CI",
            )
        )
    fig.update_layout(
        title="GARCH Volatility Forecast", template="plotly_white", height=460
    )
    return fig


def _research_summary_lines(
    ticker: str,
    conf_level: float,
    model_choice: str,
    inference_results: Dict[str, Dict],
) -> str:
    adf_reject = bool(inference_results["ADF Stationarity"].get("reject_null", False))
    kpss_reject = bool(inference_results["KPSS Stationarity"].get("reject_null", False))
    stability_reject = bool(inference_results["CUSUM Break"].get("reject_null", False))

    stationarity_msg = "Mixed evidence"
    if adf_reject and not kpss_reject:
        stationarity_msg = "Series appears stationary after cleaning"
    elif (not adf_reject) and kpss_reject:
        stationarity_msg = "Series shows non-stationary behavior"

    stability_msg = "Potential structural instability detected"
    if not stability_reject:
        stability_msg = "No strong structural break evidence"

    return "\n".join(
        [
            "### Research Summary",
            f"- Asset: {ticker}",
            f"- Primary forecasting model: {model_choice}",
            f"- Confidence level: {conf_level:.0%}",
            f"- Stationarity conclusion: {stationarity_msg}",
            f"- Stability conclusion: {stability_msg}",
            "- Recommendation: report both inferential evidence and out-of-sample predictive metrics.",
        ]
    )


def main() -> None:
    """Run the Streamlit dashboard entrypoint."""
    st.set_page_config(
        page_title="FinTimeSeries-Pipeline", page_icon="📈", layout="wide"
    )
    st.title("FinTimeSeries-Pipeline")
    st.caption(
        "Financial time series analysis pipeline for forecasting, volatility, and inference"
    )

    cfg = _load_runtime_config("config/params.yaml")
    default_symbol = str(cfg.get("data", {}).get("symbol", "AAPL"))
    default_conf = float(cfg.get("inference", {}).get("confidence_level", 0.95))

    with st.sidebar:
        st.header("Configuration")
        ticker = st.text_input("Stock ticker", value=default_symbol).strip().upper()
        start = st.date_input("Start date", value=date(2018, 1, 1))
        end = st.date_input("End date", value=date(2025, 1, 1))
        model_choice = st.selectbox(
            "Model", options=["ARIMA", "GARCH", "LSTM"], index=0
        )

        st.subheader("Data Cleaning")
        missing_method = st.selectbox(
            "Missing value method",
            options=["forward_fill", "backward_fill", "interpolate", "drop"],
            index=0,
        )
        outlier_threshold = st.slider(
            "Outlier threshold", min_value=1.5, max_value=4.0, value=3.0, step=0.1
        )

        st.subheader("Model Parameters")
        horizon = st.slider(
            "Forecast horizon", min_value=5, max_value=90, value=30, step=5
        )
        conf_level = st.slider(
            "Confidence level",
            min_value=0.80,
            max_value=0.99,
            value=default_conf,
            step=0.01,
        )
        garch_variant = st.selectbox(
            "GARCH variant", options=["GARCH", "EGARCH", "GJR-GARCH"], index=0
        )
        lstm_window = st.slider(
            "LSTM window size", min_value=10, max_value=60, value=20, step=5
        )

    if start >= end:
        st.error("Invalid date range: start date must be earlier than end date.")
        return
    if not ticker:
        st.error("Ticker cannot be empty.")
        return

    with st.spinner("Fetching and cleaning market data..."):
        try:
            df, report = _fetch_and_clean(
                ticker=ticker,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                missing_method=missing_method,
                outlier_threshold=outlier_threshold,
            )
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            return

    if df.empty:
        st.warning("No data available for the chosen ticker and date range.")
        return

    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)

    tabs = st.tabs(
        ["Overview", "Forecast", "Volatility", "Risk", "Inference", "Research"]
    )

    with tabs[0]:
        st.subheader("Market Overview")
        fig = _candlestick_volume_figure(df, ticker=ticker)
        st.plotly_chart(fig, use_container_width=True)
        st.write("Cleaning Report")
        st.json(report)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned data (CSV)",
            data=csv_bytes,
            file_name=f"{ticker}_cleaned.csv",
        )

    with tabs[1]:
        st.subheader("Forecast")
        try:
            with st.spinner(f"Running {model_choice} forecast..."):
                if model_choice == "ARIMA":
                    fc, coeff, selected_order = _run_arima(
                        df, horizon=horizon, conf_level=conf_level
                    )
                    st.success(f"ARIMA auto-selected order: {selected_order}")
                    st.plotly_chart(
                        _forecast_figure(df["Close"], fc), use_container_width=True
                    )
                    st.dataframe(coeff, use_container_width=True)
                    st.download_button(
                        "Download ARIMA forecast",
                        data=fc.to_csv(index=False).encode("utf-8"),
                        file_name=f"{ticker}_arima_forecast.csv",
                    )
                elif model_choice == "LSTM":
                    preds = _run_lstm(
                        df["Close"], horizon=horizon, window_size=lstm_window
                    )
                    fc = pd.DataFrame({"forecast": preds})
                    st.plotly_chart(
                        _forecast_figure(df["Close"], fc), use_container_width=True
                    )
                    st.download_button(
                        "Download LSTM forecast",
                        data=fc.to_csv(index=False).encode("utf-8"),
                        file_name=f"{ticker}_lstm_forecast.csv",
                    )
                else:
                    st.info("Use the Volatility tab for GARCH forecast visualization.")
        except Exception as exc:
            st.error(f"Forecast computation failed: {exc}")

    with tabs[2]:
        st.subheader("Volatility")
        returns = (
            df["Returns"].dropna()
            if "Returns" in df.columns
            else np.log(df["Close"] / df["Close"].shift(1)).dropna()
        )
        try:
            with st.spinner("Running GARCH volatility modeling..."):
                vol_fc, _ = _run_garch(
                    returns,
                    horizon=horizon,
                    conf_level=conf_level,
                    variant=garch_variant,
                )
            st.plotly_chart(_volatility_figure(vol_fc), use_container_width=True)
            st.download_button(
                "Download volatility forecast",
                data=vol_fc.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_volatility_forecast.csv",
            )
        except Exception as exc:
            st.error(f"Volatility modeling failed: {exc}")

    with tabs[3]:
        st.subheader("Risk Metrics")
        returns = (
            df["Returns"].dropna()
            if "Returns" in df.columns
            else np.log(df["Close"] / df["Close"].shift(1)).dropna()
        )
        try:
            with st.spinner("Computing risk metrics..."):
                _, risk = _run_garch(
                    returns,
                    horizon=horizon,
                    conf_level=conf_level,
                    variant=garch_variant,
                )
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Level", f"{risk['confidence_level']:.2%}")
            col2.metric("Value at Risk (VaR)", f"{risk['VaR']:.5f}")
            col3.metric("Expected Shortfall", f"{risk['ExpectedShortfall']:.5f}")
            st.download_button(
                "Download risk report (JSON)",
                data=json.dumps(risk, indent=2),
                file_name=f"{ticker}_risk_report.json",
                mime="application/json",
            )
        except Exception as exc:
            st.error(f"Risk computation failed: {exc}")

    with tabs[4]:
        st.subheader("Statistical Inference")
        try:
            with st.spinner("Running statistical tests..."):
                results, correction = _run_tests(df)
            result_df = pd.DataFrame(results).T
            st.dataframe(result_df, use_container_width=True)

            st.write("Multiple Testing Correction")
            st.json(correction)

            pvals = result_df["p_value"].astype(float).to_numpy()
            fig = go.Figure(
                go.Bar(x=result_df.index.tolist(), y=pvals, name="p-values")
            )
            fig.add_hline(y=0.05, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Inference p-values", template="plotly_white", height=380
            )
            st.plotly_chart(fig, use_container_width=True)

            buffer = StringIO()
            result_df.to_csv(buffer)
            st.download_button(
                "Download inference report",
                data=buffer.getvalue(),
                file_name=f"{ticker}_inference_report.csv",
            )
        except Exception as exc:
            st.error(f"Inference tests failed: {exc}")

    with tabs[5]:
        st.subheader("Research Summary")
        try:
            with st.spinner("Building research-oriented summary..."):
                inference_results, correction = _run_tests(df)

                holdout_size = max(20, int(len(df) * 0.2))
                holdout_size = min(holdout_size, len(df) - 40)
                holdout_metrics = {}

                if holdout_size > 0:
                    train = df.iloc[:-holdout_size].copy()
                    test = df.iloc[-holdout_size:].copy()
                    model = ARIMAModel(conf_level=conf_level)
                    model.fit(train, column="Close", auto_order=True)
                    fc = model.forecast(
                        steps=len(test), confidence_interval=True
                    ).reset_index(drop=True)
                    holdout_metrics = summarize_forecast_metrics(
                        y_true=test["Close"].to_numpy(),
                        y_pred=fc["forecast"].to_numpy(),
                        lower=fc["lower"].to_numpy(),
                        upper=fc["upper"].to_numpy(),
                    )

            st.markdown(
                _research_summary_lines(
                    ticker=ticker,
                    conf_level=conf_level,
                    model_choice=model_choice,
                    inference_results=inference_results,
                )
            )

            st.write("Hypothesis Test Snapshot")
            st.dataframe(pd.DataFrame(inference_results).T, use_container_width=True)
            st.write("Multiple Testing Correction")
            st.json(correction)

            if holdout_metrics:
                st.write("Holdout Forecast Metrics (ARIMA)")
                metrics_df = pd.DataFrame(
                    {
                        "metric": list(holdout_metrics.keys()),
                        "value": list(holdout_metrics.values()),
                    }
                )
                st.dataframe(metrics_df, use_container_width=True)
                st.download_button(
                    "Download research metrics",
                    data=metrics_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{ticker}_research_metrics.csv",
                )
        except Exception as exc:
            st.error(f"Research summary generation failed: {exc}")


if __name__ == "__main__":
    main()
