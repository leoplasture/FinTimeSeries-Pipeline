"""Data cleaning and statistical diagnostics for financial time series."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller


class DataCleaner:
    """Clean, transform, and diagnose financial time series datasets.

    Parameters:
       missing_method (str): Strategy for missing values.
          Supported values: "forward_fill", "backward_fill", "drop", "interpolate".
       outlier_threshold (float): Threshold used by z-score or IQR fences.
       config_path (str | None): Optional YAML config path.

    Example:
       >>> cleaner = DataCleaner(missing_method="forward_fill", outlier_threshold=3.0)
       >>> cleaned_df, report = cleaner.clean_pipeline(df)
       >>> report["is_stationary"]
       False
    """

    def __init__(
        self,
        missing_method: str = "forward_fill",
        outlier_threshold: float = 3.0,
        config_path: Optional[str] = None,
    ) -> None:
        self.missing_method = missing_method
        self.outlier_threshold = outlier_threshold
        self.config = self._load_config(config_path) if config_path else {}
        self._transformations_applied: List[str] = []

        cleaner_cfg = self.config.get("cleaner", {})
        self.missing_method = cleaner_cfg.get("missing_method", self.missing_method)
        self.outlier_threshold = float(
            cleaner_cfg.get("outlier_threshold", self.outlier_threshold)
        )

    def handle_missing_values(
        self, df: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """Handle missing values with the selected strategy.

        Parameters:
           df (pandas.DataFrame): Input DataFrame.
           method (str): Missing-value strategy.

        Returns:
           pandas.DataFrame: DataFrame after missing-value treatment.
        """
        cleaned = df.copy()
        selected = method or self.missing_method

        if selected == "forward_fill":
            cleaned = cleaned.ffill().bfill()
        elif selected == "backward_fill":
            cleaned = cleaned.bfill().ffill()
        elif selected == "drop":
            cleaned = cleaned.dropna(axis=0)
        elif selected == "interpolate":
            cleaned = cleaned.interpolate(method="linear", limit_direction="both")
        else:
            raise ValueError(f"Unsupported missing value method: {selected}")

        self._transformations_applied.append(f"missing:{selected}")
        return cleaned

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str = "Close",
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.Series:
        """Detect outlier rows based on selected method.

        Parameters:
           df (pandas.DataFrame): Input dataset.
           column (str): Column to inspect.
           method (str): "iqr" or "zscore".
           threshold (float): Threshold parameter.

        Returns:
           pandas.Series: Boolean mask where True indicates an outlier.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")

        series = df[column].astype(float)
        selected_threshold = (
            threshold if threshold is not None else self.outlier_threshold
        )

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - selected_threshold * iqr
            upper = q3 + selected_threshold * iqr
            mask = (series < lower) | (series > upper)
        elif method == "zscore":
            std = series.std(ddof=1)
            if np.isclose(std, 0.0):
                mask = pd.Series(False, index=df.index)
            else:
                z = (series - series.mean()) / std
                mask = z.abs() > selected_threshold
        else:
            raise ValueError("method must be either 'iqr' or 'zscore'.")

        self._transformations_applied.append(f"outlier_detect:{method}")
        return mask

    def adjust_for_splits_dividends(
        self, df: pd.DataFrame, ticker: str
    ) -> pd.DataFrame:
        """Align price columns to adjusted prices when available.

        Parameters:
           df (pandas.DataFrame): Price DataFrame.
           ticker (str): Symbol label used in reporting.

        Returns:
           pandas.DataFrame: Price DataFrame adjusted for corporate actions.
        """
        adjusted = df.copy()
        if "Adj_Close" in adjusted.columns and "Close" in adjusted.columns:
            ratio = adjusted["Adj_Close"] / adjusted["Close"].replace(0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            for col in ["Open", "High", "Low", "Close"]:
                if col in adjusted.columns:
                    adjusted[col] = adjusted[col] * ratio
            self._transformations_applied.append(f"adj_prices:{ticker}")
        return adjusted

    def compute_returns(
        self,
        df: pd.DataFrame,
        method: str = "log_return",
        frequency: str = "daily",
    ) -> pd.DataFrame:
        """Compute return series from close prices.

        Parameters:
           df (pandas.DataFrame): Input DataFrame with a Close column.
           method (str): "log_return" or "simple_return".
           frequency (str): "daily", "weekly", or "monthly".

        Returns:
           pandas.DataFrame: DataFrame augmented with a Returns column.
        """
        if "Close" not in df.columns:
            raise KeyError("Close column is required to compute returns.")

        transformed = df.copy()
        if "Date" in transformed.columns:
            transformed["Date"] = pd.to_datetime(transformed["Date"], errors="coerce")
            transformed = transformed.dropna(subset=["Date"]).sort_values("Date")
            transformed = transformed.set_index("Date", drop=False)

        if frequency in {"weekly", "monthly"}:
            rule = "W" if frequency == "weekly" else "M"
            resampled = transformed[["Close"]].resample(rule).last().dropna()
            transformed = transformed.join(
                resampled.rename(columns={"Close": "Close_resampled"}), how="left"
            )
            price_series = transformed["Close_resampled"].fillna(transformed["Close"])
        else:
            price_series = transformed["Close"]

        if method == "log_return":
            transformed["Returns"] = np.log(price_series / price_series.shift(1))
        elif method == "simple_return":
            transformed["Returns"] = price_series.pct_change()
        else:
            raise ValueError("method must be 'log_return' or 'simple_return'.")

        transformed = transformed.drop(columns=["Close_resampled"], errors="ignore")
        self._transformations_applied.append(f"returns:{method}:{frequency}")
        return transformed.reset_index(drop=True)

    def check_stationarity(
        self,
        df: pd.DataFrame,
        column: str = "Close",
        test: str = "adf",
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        """Run stationarity diagnostics.

        Parameters:
           df (pandas.DataFrame): Input DataFrame.
           column (str): Target column for the test.
           test (str): Statistical test name, currently supports "adf".
           significance (float): Decision threshold for p-value.

        Returns:
           dict: Test statistics including `is_stationary` and p-value.
        """
        if test != "adf":
            raise NotImplementedError("Only ADF stationarity test is implemented.")
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")

        series = df[column].dropna().astype(float)
        if len(series) < 10:
            raise ValueError(
                "At least 10 observations are recommended for stationarity testing."
            )

        stat, pvalue, used_lag, nobs, critical_values, _ = adfuller(series)
        result = {
            "test": "adf",
            "statistic": float(stat),
            "p_value": float(pvalue),
            "used_lag": int(used_lag),
            "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in critical_values.items()},
            "is_stationary": bool(pvalue < significance),
        }
        self._transformations_applied.append("stationarity:adf")
        return result

    def run_distribution_tests(
        self,
        df: pd.DataFrame,
        column: str = "Returns",
        lags: int = 10,
    ) -> Dict[str, Any]:
        """Run Ljung-Box and Jarque-Bera diagnostics on a series.

        Parameters:
           df (pandas.DataFrame): Input DataFrame.
           column (str): Column to test.
           lags (int): Number of lags used in Ljung-Box.

        Returns:
           dict: Statistical diagnostics for autocorrelation and normality.
        """
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found.")

        series = df[column].dropna().astype(float)
        if len(series) < max(20, lags + 1):
            raise ValueError("Insufficient observations for diagnostics.")

        lb = acorr_ljungbox(series, lags=[lags], return_df=True)
        jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(series)
        return {
            "ljung_box": {
                "lag": int(lags),
                "statistic": float(lb["lb_stat"].iloc[0]),
                "p_value": float(lb["lb_pvalue"].iloc[0]),
            },
            "jarque_bera": {
                "statistic": float(jb_stat),
                "p_value": float(jb_pvalue),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
            },
        }

    def generate_cleaning_report(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        outlier_mask: pd.Series,
        stationarity_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create standardized cleaning report dictionary.

        Returns:
           dict: Report with shape changes, quality counts, and transformations.
        """
        return {
            "original_shape": tuple(original_df.shape),
            "cleaned_shape": tuple(cleaned_df.shape),
            "missing_count": int(original_df.isna().sum().sum()),
            "outlier_count": int(outlier_mask.sum()),
            "is_stationary": bool(stationarity_result.get("is_stationary", False)),
            "transformations_applied": list(
                dict.fromkeys(self._transformations_applied)
            ),
        }

    def plot_before_after(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        column: str = "Close",
    ) -> go.Figure:
        """Plot before/after cleaning comparison for a selected price column.

        Parameters:
           original_df (pandas.DataFrame): Dataset before cleaning.
           cleaned_df (pandas.DataFrame): Dataset after cleaning.
           column (str): Column to visualize.

        Returns:
           plotly.graph_objects.Figure: Interactive comparison figure.
        """
        fig = go.Figure()
        if column in original_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=original_df.get("Date", original_df.index),
                    y=original_df[column],
                    mode="lines",
                    name="Before",
                )
            )
        if column in cleaned_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=cleaned_df.get("Date", cleaned_df.index),
                    y=cleaned_df[column],
                    mode="lines",
                    name="After",
                )
            )
        fig.update_layout(
            title=f"Before/After Cleaning: {column}",
            xaxis_title="Date",
            yaxis_title=column,
            template="plotly_white",
        )
        return fig

    def clean_pipeline(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        outlier_method: str = "iqr",
        returns_method: str = "log_return",
        frequency: str = "daily",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute an end-to-end cleaning workflow and return report.

        Parameters:
           df (pandas.DataFrame): Raw price series.
           ticker (str): Ticker label for metadata.
           outlier_method (str): Outlier detection method.
           returns_method (str): Return construction method.
           frequency (str): Return frequency.

        Returns:
           tuple[pandas.DataFrame, dict]: Cleaned DataFrame and cleaning report.
        """
        self._transformations_applied = []
        original = df.copy()
        cleaned = self.handle_missing_values(original, method=self.missing_method)
        outlier_mask = self.detect_outliers(
            cleaned,
            column="Close",
            method=outlier_method,
            threshold=self.outlier_threshold,
        )
        cleaned = cleaned.loc[~outlier_mask].copy()
        self._transformations_applied.append(f"outlier_remove:{outlier_method}")
        cleaned = self.adjust_for_splits_dividends(cleaned, ticker=ticker)
        cleaned = self.compute_returns(
            cleaned, method=returns_method, frequency=frequency
        )
        stationarity = self.check_stationarity(cleaned, column="Close", test="adf")
        report = self.generate_cleaning_report(
            original, cleaned, outlier_mask, stationarity
        )
        return cleaned, report

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
