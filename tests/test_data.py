"""Unit tests for data acquisition, cleaning, and storage modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create synthetic OHLCV data for repeatable tests."""
    n = 160
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0.1, 1.2, n))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + np.abs(rng.normal(0, 0.6, n)),
            "Low": close - np.abs(rng.normal(0, 0.6, n)),
            "Close": close,
            "Volume": rng.integers(100_000, 2_000_000, n),
            "Adj_Close": close * 0.998,
        }
    )
    df.loc[4, "Close"] = np.nan
    df.loc[15, "Volume"] = np.nan
    df.loc[100, "Close"] = df["Close"].median() + 35.0
    return df


def test_data_fetcher_cache_and_standardization(tmp_path, monkeypatch) -> None:
    """Fetcher should standardize columns and reuse cache on repeated calls."""
    from src.data.fetch import DataFetcher

    calls = {"count": 0}

    def fake_download(symbol, start, end, auto_adjust, progress, threads):
        calls["count"] += 1
        idx = pd.date_range("2023-01-01", periods=12, freq="D")
        return pd.DataFrame(
            {
                "Open": np.linspace(10, 12, 12),
                "High": np.linspace(11, 13, 12),
                "Low": np.linspace(9, 11, 12),
                "Close": np.linspace(10, 12, 12),
                "Adj Close": np.linspace(10, 12, 12),
                "Volume": np.arange(1000, 1012),
            },
            index=idx,
        )

    monkeypatch.setattr("src.data.fetch.yf.download", fake_download)
    fetcher = DataFetcher(cache_dir=str(tmp_path / "cache"), use_cache=True)

    df_1 = fetcher.fetch_stock_data("AAPL", "2023-01-01", "2023-02-01")
    df_2 = fetcher.fetch_stock_data("AAPL", "2023-01-01", "2023-02-01")

    assert (
        calls["count"] == 1
    ), "Expected second fetch to use cache instead of API call."
    assert list(df_1.columns) == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Adj_Close",
    ], "Fetcher output schema is not standardized."
    assert (
        len(df_1) == len(df_2) == 12
    ), "Cached and fetched DataFrame lengths should match."


def test_data_fetcher_invalid_date_range_raises() -> None:
    """Fetcher should reject invalid date ranges."""
    from src.data.fetch import DataFetcher

    fetcher = DataFetcher(use_cache=False)
    with pytest.raises(ValueError, match="start_date"):
        fetcher.fetch_stock_data("AAPL", "2024-01-10", "2024-01-01")


def test_market_microstructure_placeholder_schema() -> None:
    """Microstructure method should return expected placeholder columns."""
    from src.data.fetch import DataFetcher

    fetcher = DataFetcher(use_cache=False)
    snap = fetcher.fetch_market_microstructure("AAPL", "2024-06-01")
    expected_cols = {"symbol", "best_bid", "best_ask", "mid_price", "bid_ask_spread"}
    assert expected_cols.issubset(
        set(snap.columns)
    ), "Microstructure snapshot schema is incomplete."


def test_data_cleaner_pipeline_and_report(sample_ohlcv_df: pd.DataFrame) -> None:
    """Cleaner pipeline should produce returns and a complete report."""
    from src.data.clean import DataCleaner

    cleaner = DataCleaner(missing_method="forward_fill", outlier_threshold=2.5)
    cleaned, report = cleaner.clean_pipeline(
        sample_ohlcv_df, ticker="TEST", outlier_method="iqr"
    )

    required_report_keys = {
        "original_shape",
        "cleaned_shape",
        "missing_count",
        "outlier_count",
        "is_stationary",
        "transformations_applied",
    }
    assert (
        "Returns" in cleaned.columns
    ), "Cleaned DataFrame should include computed returns."
    assert required_report_keys.issubset(
        set(report.keys())
    ), "Cleaning report missing required keys."
    assert (
        report["cleaned_shape"][0] <= report["original_shape"][0]
    ), "Pipeline unexpectedly increased row count."


def test_data_cleaner_distribution_tests(sample_ohlcv_df: pd.DataFrame) -> None:
    """Distribution diagnostics should return Ljung-Box and Jarque-Bera results."""
    from src.data.clean import DataCleaner

    cleaner = DataCleaner()
    cleaned, _ = cleaner.clean_pipeline(sample_ohlcv_df, ticker="TEST")
    stats = cleaner.run_distribution_tests(cleaned, column="Returns", lags=10)
    assert (
        "ljung_box" in stats and "jarque_bera" in stats
    ), "Expected both Ljung-Box and Jarque-Bera outputs."


def test_data_storage_roundtrip_and_backup(
    tmp_path, sample_ohlcv_df: pd.DataFrame
) -> None:
    """Storage layer should support write/query/export/backup flow."""
    pytest.importorskip("sqlalchemy")
    from src.data.storage import DataStorage

    db_path = tmp_path / "finance.db"
    storage = DataStorage(db_path=str(db_path))

    cleaned = sample_ohlcv_df.dropna().copy()
    storage.store_dataframe(
        cleaned, "sample_prices", source="unit-test", cleaning_log="{}", version="v1"
    )

    queried = storage.query_table(
        "sample_prices", date_range=("2022-01-05", "2022-02-20")
    )
    assert not queried.empty, "Query should return rows for valid date range."

    tables = storage.get_table_list()
    assert "sample_prices" in set(
        tables["table_name"].tolist()
    ), "Stored table should appear in table list."

    csv_path = storage.export_to_csv(
        "sample_prices", str(tmp_path / "export" / "prices.csv")
    )
    assert csv_path.exists(), "CSV export file was not created."

    backup_path = storage.backup_database(
        str(tmp_path / "backup" / "finance_backup.db")
    )
    assert backup_path.exists(), "Database backup file was not created."
