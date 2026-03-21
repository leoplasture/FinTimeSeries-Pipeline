"""Data fetching utilities for financial time series and optional microstructure snapshots."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


class DataFetchError(RuntimeError):
    """Raised when remote data acquisition fails after retries."""


@dataclass
class _RequestConfig:
    """Internal request controls for retry and rate limiting."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    min_interval_seconds: float = 0.35


class DataFetcher:
    """Fetch and standardize financial market data from remote providers.

    Parameters:
       cache_dir (str): Directory where cached API responses are stored.
       use_cache (bool): Whether to read/write cache files.

    Example:
       >>> fetcher = DataFetcher(cache_dir="data/cache", use_cache=True)
       >>> df = fetcher.fetch_stock_data("AAPL", "2020-01-01", "2020-12-31")
       >>> list(df.columns)
       ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']

    Raises:
       DataFetchError: If the provider call fails after max retries.
       ValueError: If date parameters are invalid.
    """

    STANDARD_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj_Close"]

    def __init__(self, cache_dir: str = "data/cache", use_cache: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.request_cfg = _RequestConfig(max_retries=3)
        self._last_call_time = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        source: str = "yfinance",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for an equity ticker.

        Parameters:
           ticker (str): Equity ticker symbol, for example "AAPL".
           start_date (str): Inclusive start date in ISO format YYYY-MM-DD.
           end_date (str): Inclusive end date in ISO format YYYY-MM-DD.
           source (str): Data provider name. Currently supports "yfinance".

        Returns:
           pandas.DataFrame: Standardized OHLCV DataFrame with columns
           [Date, Open, High, Low, Close, Volume, Adj_Close].

        Raises:
           DataFetchError: If fetch fails after 3 retries.
           NotImplementedError: If source is unsupported.
        """
        if source.lower() != "yfinance":
            raise NotImplementedError(f"Unsupported source '{source}'. Use 'yfinance'.")
        return self._fetch_with_retry(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            asset_type="stock",
        )

    def fetch_index_data(
        self, index_name: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch index time series data.

        Parameters:
           index_name (str): Index ticker (for example '^GSPC', '^IXIC').
           start_date (str): Inclusive start date in ISO format YYYY-MM-DD.
           end_date (str): Inclusive end date in ISO format YYYY-MM-DD.

        Returns:
           pandas.DataFrame: Standardized index OHLCV dataset.

        Raises:
           DataFetchError: If provider requests fail repeatedly.
        """
        return self._fetch_with_retry(
            symbol=index_name,
            start_date=start_date,
            end_date=end_date,
            asset_type="index",
        )

    def fetch_crypto_data(
        self,
        start_date: str,
        end_date: str,
        symbol: str = "BTC-USD",
    ) -> pd.DataFrame:
        """Fetch cryptocurrency OHLCV time series.

        Parameters:
           start_date (str): Inclusive start date in ISO format YYYY-MM-DD.
           end_date (str): Inclusive end date in ISO format YYYY-MM-DD.
           symbol (str): Crypto pair symbol, defaults to "BTC-USD".

        Returns:
           pandas.DataFrame: Standardized crypto OHLCV DataFrame.

        Raises:
           DataFetchError: If provider requests fail repeatedly.
        """
        return self._fetch_with_retry(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            asset_type="crypto",
        )

    def fetch_market_microstructure(
        self,
        ticker: str,
        date: str,
        source: str = "alpaca",
    ) -> pd.DataFrame:
        """Fetch optional intraday microstructure snapshot (order-book style fields).

        Parameters:
           ticker (str): Market symbol.
           date (str): Trading date in YYYY-MM-DD format.
           source (str): Market microstructure data source, e.g., "alpaca".

        Returns:
           pandas.DataFrame: DataFrame with representative order-book metrics.

        Raises:
           NotImplementedError: For real API integrations not configured locally.

        Example:
           >>> fetcher = DataFetcher()
           >>> snapshot = fetcher.fetch_market_microstructure("AAPL", "2024-10-01")
           >>> snapshot[["symbol", "bid_ask_spread", "mid_price"]].head(1)
        """
        if source.lower() != "alpaca":
            raise NotImplementedError("Only 'alpaca' placeholder schema is provided.")

        # Placeholder schema to keep downstream pipelines typed and testable.
        snapshot = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp(f"{date} 15:59:00")],
                "symbol": [ticker],
                "best_bid": [100.0],
                "best_ask": [100.05],
                "bid_size": [1200],
                "ask_size": [900],
            }
        )
        snapshot["mid_price"] = (snapshot["best_bid"] + snapshot["best_ask"]) / 2.0
        snapshot["bid_ask_spread"] = snapshot["best_ask"] - snapshot["best_bid"]
        return snapshot

    def _fetch_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        asset_type: str,
    ) -> pd.DataFrame:
        self._validate_dates(start_date, end_date)
        cache_key = f"{asset_type}_{symbol}_{start_date}_{end_date}".replace("^", "IDX")
        cache_path = self.cache_dir / f"{cache_key}.csv"

        if self.use_cache and cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["Date"])
            return self._standardize_columns(cached)

        last_exception: Optional[Exception] = None
        for attempt in range(1, self.request_cfg.max_retries + 1):
            try:
                self._respect_rate_limit()
                raw = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if raw.empty:
                    raise DataFetchError(f"No data returned for symbol '{symbol}'.")

                standardized = self._standardize_columns(raw.reset_index())
                if self.use_cache:
                    standardized.to_csv(cache_path, index=False)
                    self._write_cache_metadata(
                        cache_path, symbol=symbol, asset_type=asset_type
                    )
                return standardized
            except Exception as exc:  # pragma: no cover - network instability branch
                last_exception = exc
                if attempt < self.request_cfg.max_retries:
                    time.sleep(self.request_cfg.retry_delay_seconds * attempt)

        raise DataFetchError(
            f"Failed fetching {asset_type} '{symbol}' after {self.request_cfg.max_retries} attempts."
        ) from last_exception

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_call_time
        min_interval = self.request_cfg.min_interval_seconds
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        rename_map = {"Adj Close": "Adj_Close", "Datetime": "Date"}
        normalized = normalized.rename(columns=rename_map)
        if "Date" not in normalized.columns and "index" in normalized.columns:
            normalized = normalized.rename(columns={"index": "Date"})
        if "Adj_Close" not in normalized.columns:
            normalized["Adj_Close"] = normalized["Close"]

        missing_cols = [
            col for col in self.STANDARD_COLUMNS if col not in normalized.columns
        ]
        if missing_cols:
            raise DataFetchError(f"Fetched data missing columns: {missing_cols}")

        standardized = normalized[self.STANDARD_COLUMNS].copy()
        standardized["Date"] = pd.to_datetime(standardized["Date"], errors="coerce")
        standardized = (
            standardized.dropna(subset=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
        return standardized

    def _validate_dates(self, start_date: str, end_date: str) -> None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts >= end_ts:
            raise ValueError("start_date must be earlier than end_date.")

    def _write_cache_metadata(
        self, cache_path: Path, symbol: str, asset_type: str
    ) -> None:
        metadata: Dict[str, Any] = {
            "symbol": symbol,
            "asset_type": asset_type,
            "fetch_time": pd.Timestamp.utcnow().isoformat(),
            "provider": "yfinance",
            "version": "v1",
        }
        meta_path = cache_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
