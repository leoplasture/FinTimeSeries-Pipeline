"""Command-line entrypoint for running dashboard or batch pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.data.clean import DataCleaner
from src.data.fetch import DataFetcher
from src.data.storage import DataStorage
from src.models.arima import ARIMAModel
from src.utils.config import load_config
from src.utils.experiment import (
    ExperimentRun,
    finalize_experiment_run,
    register_artifact,
    start_experiment_run,
    write_json,
)
from src.utils.metrics import summarize_forecast_metrics


def run_dashboard() -> None:
    """Launch Streamlit dashboard as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/viz/dashboard.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
    ]
    subprocess.run(cmd, check=True)


def _safe_finalize(
    run: Optional[ExperimentRun],
    status: str,
    metrics: Optional[dict] = None,
    notes: Optional[str] = None,
) -> None:
    if run is not None:
        finalize_experiment_run(run=run, status=status, metrics=metrics, notes=notes)


def _walkforward_on_series(
    series: pd.Series,
    order: Tuple[int, int, int],
    initial_train_size: int,
    horizon: int,
    step_size: int,
    max_windows: int,
    auto_order: bool,
    conf_level: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    values = series.dropna().astype(float).reset_index(drop=True)
    if len(values) < initial_train_size + horizon:
        raise ValueError("Series is too short for walk-forward configuration.")

    rows = []
    window_id = 0
    train_end = initial_train_size

    while train_end + horizon <= len(values) and window_id < max_windows:
        train = values.iloc[:train_end]
        test = values.iloc[train_end : train_end + horizon]

        model = ARIMAModel(order=order, conf_level=conf_level)
        model.fit(pd.DataFrame({"Close": train}), column="Close", auto_order=auto_order)
        forecast = model.forecast(steps=horizon, confidence_interval=True)

        metrics = summarize_forecast_metrics(
            y_true=test.to_numpy(),
            y_pred=forecast["forecast"].to_numpy(),
            lower=forecast["lower"].to_numpy(),
            upper=forecast["upper"].to_numpy(),
        )
        metrics.update(
            {
                "window_id": float(window_id),
                "train_size": float(train_end),
                "test_size": float(horizon),
            }
        )
        rows.append(metrics)

        train_end += step_size
        window_id += 1

    detail = pd.DataFrame(rows)
    metric_cols = [
        c for c in detail.columns if c not in {"window_id", "train_size", "test_size"}
    ]
    summary = {f"mean_{c}": float(detail[c].mean()) for c in metric_cols}
    summary["n_windows"] = float(len(detail))
    return detail, summary


def run_batch_pipeline(config_path: str, track_experiment: bool = True) -> None:
    """Run fetch-clean-store pipeline using YAML configuration."""
    cfg = load_config(config_path)
    run: Optional[ExperimentRun] = None

    experiment_cfg = cfg.get("experiment", {})
    tracking_enabled = bool(experiment_cfg.get("enabled", True)) and track_experiment
    experiment_dir = str(experiment_cfg.get("output_dir", "runs"))

    if tracking_enabled:
        run = start_experiment_run(
            project_name=str(
                cfg.get("project", {}).get("name", "FinTimeSeries-Pipeline")
            ),
            mode="pipeline",
            config_path=config_path,
            output_dir=experiment_dir,
            workspace_root=".",
        )

    symbol = cfg.get("data", {}).get("symbol", "AAPL")
    start = cfg.get("data", {}).get("date_range", {}).get("start", "2020-01-01")
    end = cfg.get("data", {}).get("date_range", {}).get("end", "2024-12-31")
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache")
    use_cache = bool(cfg.get("data", {}).get("cache_enabled", True))

    missing_method = cfg.get("cleaning", {}).get("missing_method", "forward_fill")
    outlier_threshold = float(cfg.get("cleaning", {}).get("outlier_threshold", 3.0))
    returns_method = cfg.get("cleaning", {}).get("returns_method", "log_return")

    db_path = cfg.get("database", {}).get("path", "data/finance.db")

    try:
        fetcher = DataFetcher(cache_dir=cache_dir, use_cache=use_cache)
        raw = fetcher.fetch_stock_data(ticker=symbol, start_date=start, end_date=end)

        cleaner = DataCleaner(
            missing_method=missing_method, outlier_threshold=outlier_threshold
        )
        cleaned, report = cleaner.clean_pipeline(
            raw,
            ticker=symbol,
            outlier_method="iqr",
            returns_method=returns_method,
            frequency="daily",
        )

        storage = DataStorage(db_path=db_path)
        table_name = f"{symbol.lower()}_daily"
        storage.store_dataframe(
            cleaned,
            table_name=table_name,
            if_exists="replace",
            source="yfinance",
            cleaning_log=json.dumps(report),
            version="v1",
        )

        output_dir = Path(cfg.get("storage", {}).get("output_dir", "data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / f"{symbol.lower()}_cleaned.csv"
        cleaned.to_csv(output_csv, index=False)

        report_path = output_dir / f"{symbol.lower()}_cleaning_report.json"
        write_json(report_path, report)

        if run is not None:
            run_output = run.run_dir / "artifacts"
            run_output.mkdir(parents=True, exist_ok=True)
            run_csv = run_output / f"{symbol.lower()}_cleaned.csv"
            run_report = run_output / f"{symbol.lower()}_cleaning_report.json"
            cleaned.to_csv(run_csv, index=False)
            write_json(run_report, report)
            register_artifact(run, run_csv)
            register_artifact(run, run_report)

            if "Returns" in cleaned.columns:
                returns = cleaned["Returns"].dropna().astype(float)
                run_metrics = {
                    "rows": int(len(cleaned)),
                    "return_mean": float(returns.mean()) if not returns.empty else 0.0,
                    "return_std": (
                        float(returns.std(ddof=1)) if returns.size > 1 else 0.0
                    ),
                    "start_date": (
                        str(cleaned["Date"].min()) if "Date" in cleaned.columns else ""
                    ),
                    "end_date": (
                        str(cleaned["Date"].max()) if "Date" in cleaned.columns else ""
                    ),
                }
            else:
                run_metrics = {"rows": int(len(cleaned))}
            _safe_finalize(run, status="success", metrics=run_metrics)

        print("Pipeline completed.")
        print(f"Rows: {len(cleaned)}")
        print(f"Saved CSV: {output_csv}")
        print(f"Saved SQL table: {table_name}")
        if run is not None:
            print(f"Experiment run ID: {run.run_id}")
            print(f"Experiment directory: {run.run_dir}")
    except Exception as exc:
        _safe_finalize(run, status="failed", notes=str(exc))
        raise


def run_walkforward_validation(config_path: str, track_experiment: bool = True) -> None:
    """Run walk-forward ARIMA evaluation and export window-level metrics."""
    cfg = load_config(config_path)
    run: Optional[ExperimentRun] = None

    experiment_cfg = cfg.get("experiment", {})
    tracking_enabled = bool(experiment_cfg.get("enabled", True)) and track_experiment
    experiment_dir = str(experiment_cfg.get("output_dir", "runs"))

    if tracking_enabled:
        run = start_experiment_run(
            project_name=str(
                cfg.get("project", {}).get("name", "FinTimeSeries-Pipeline")
            ),
            mode="walkforward",
            config_path=config_path,
            output_dir=experiment_dir,
            workspace_root=".",
        )

    symbol = cfg.get("data", {}).get("symbol", "AAPL")
    start = cfg.get("data", {}).get("date_range", {}).get("start", "2020-01-01")
    end = cfg.get("data", {}).get("date_range", {}).get("end", "2024-12-31")
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache")
    use_cache = bool(cfg.get("data", {}).get("cache_enabled", True))

    missing_method = cfg.get("cleaning", {}).get("missing_method", "forward_fill")
    outlier_threshold = float(cfg.get("cleaning", {}).get("outlier_threshold", 3.0))
    returns_method = cfg.get("cleaning", {}).get("returns_method", "log_return")

    eval_cfg = cfg.get("evaluation", {}).get("walkforward", {})
    initial_train_size = int(eval_cfg.get("initial_train_size", 120))
    horizon = int(eval_cfg.get("horizon", 10))
    step_size = int(eval_cfg.get("step_size", 10))
    max_windows = int(eval_cfg.get("max_windows", 20))

    arima_cfg = cfg.get("models", {}).get("arima", {})
    arima_order = tuple(arima_cfg.get("order", [1, 1, 1]))
    auto_order = bool(arima_cfg.get("auto_select", False))
    conf_level = float(cfg.get("inference", {}).get("confidence_level", 0.95))

    try:
        fetcher = DataFetcher(cache_dir=cache_dir, use_cache=use_cache)
        raw = fetcher.fetch_stock_data(ticker=symbol, start_date=start, end_date=end)

        cleaner = DataCleaner(
            missing_method=missing_method, outlier_threshold=outlier_threshold
        )
        cleaned, _ = cleaner.clean_pipeline(
            raw,
            ticker=symbol,
            outlier_method="iqr",
            returns_method=returns_method,
            frequency="daily",
        )

        detail, summary = _walkforward_on_series(
            series=cleaned["Close"],
            order=(int(arima_order[0]), int(arima_order[1]), int(arima_order[2])),
            initial_train_size=initial_train_size,
            horizon=horizon,
            step_size=step_size,
            max_windows=max_windows,
            auto_order=auto_order,
            conf_level=conf_level,
        )

        output_dir = Path(cfg.get("storage", {}).get("output_dir", "data"))
        output_dir.mkdir(parents=True, exist_ok=True)
        detail_path = output_dir / f"{symbol.lower()}_walkforward_metrics.csv"
        summary_path = output_dir / f"{symbol.lower()}_walkforward_summary.json"
        detail.to_csv(detail_path, index=False)
        write_json(summary_path, summary)

        if run is not None:
            run_output = run.run_dir / "artifacts"
            run_output.mkdir(parents=True, exist_ok=True)
            run_detail = run_output / detail_path.name
            run_summary = run_output / summary_path.name
            detail.to_csv(run_detail, index=False)
            write_json(run_summary, summary)
            register_artifact(run, run_detail)
            register_artifact(run, run_summary)
            _safe_finalize(run, status="success", metrics=summary)

        print("Walk-forward validation completed.")
        print(f"Windows evaluated: {int(summary['n_windows'])}")
        print(f"Saved window metrics: {detail_path}")
        print(f"Saved summary: {summary_path}")
        if run is not None:
            print(f"Experiment run ID: {run.run_id}")
            print(f"Experiment directory: {run.run_dir}")
    except Exception as exc:
        _safe_finalize(run, status="failed", notes=str(exc))
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinTimeSeries-Pipeline entrypoint")
    parser.add_argument(
        "--mode",
        choices=["dashboard", "pipeline", "walkforward"],
        default="dashboard",
        help="Run Streamlit dashboard, batch pipeline, or walk-forward validation",
    )
    parser.add_argument(
        "--config",
        default="config/params.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable experiment tracking for this run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "walkforward":
        run_walkforward_validation(args.config, track_experiment=not args.no_track)
    else:
        run_batch_pipeline(args.config, track_experiment=not args.no_track)
