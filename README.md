# FinTimeSeries-Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-informational)
![Build](https://img.shields.io/badge/build-docker%20ready-success)
![License](https://img.shields.io/badge/license-MIT-green)

## Project Overview

FinTimeSeries-Pipeline is a portfolio-grade financial time series analysis system designed for undergraduate students targeting graduate applications in statistics, data science, and computational finance. It combines rigorous statistical inference, modular software engineering, and practical market modeling into a reproducible workflow.

## Why This Project

This repository is built to demonstrate three advisor-relevant competencies:

- Statistics: confidence intervals, bootstrap inference, unit-root/stability tests, model comparison.
- Engineering: modular architecture, test suite, typed utilities, configuration-driven runtime.
- Finance: ARIMA forecasting, GARCH volatility/risk modeling, market microstructure-ready interfaces.

## Key Features

- Multiple forecasting models: ARIMA, GARCH-family volatility models, LSTM-style baseline.
- Statistical inference toolkit: confidence intervals, coverage analysis, hypothesis testing.
- Interactive Streamlit dashboard with overview, forecast, volatility, risk, and inference tabs.
- Reproducibility stack: requirements, Dockerfile, .dockerignore, YAML config.
- Experiment tracking: run-level manifests and summaries with artifacts and metadata.
- CI automation: GitHub Actions pipeline for install + full pytest validation.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

```python
from src.data.fetch import DataFetcher
from src.data.clean import DataCleaner

df = DataFetcher().fetch_stock_data("AAPL", "2020-01-01", "2024-12-31")
cleaned, report = DataCleaner().clean_pipeline(df, ticker="AAPL")
print(cleaned.tail(3), report["is_stationary"])
```

## Configuration

Runtime parameters live in `config/params.yaml`.

- `data`: default symbol, date range, caching settings.
- `cleaning`: missing value handling, outlier threshold, return construction.
- `models`: ARIMA/GARCH/LSTM hyperparameters.
- `inference`: confidence level and bootstrap setup.
- `database`: SQLite path and backup settings.
- `experiment`: run tracking controls and output directory.

## Reproducible Batch Runs

Run the batch pipeline with tracking enabled by default:

```bash
python run.py --mode pipeline --config config/params.yaml
```

This creates a timestamped run directory under `runs/` containing:

- `manifest.json`: run ID, config path, git commit, and artifact registry.
- `summary.json`: final status and key metrics.
- Artifacts: cleaned CSV and cleaning report.

Disable tracking for quick local checks:

```bash
python run.py --mode pipeline --config config/params.yaml --no-track
```

Run walk-forward validation (rolling-origin backtest for forecasting quality):

```bash
python run.py --mode walkforward --config config/params.yaml
```

Outputs:

- `data/<symbol>_walkforward_metrics.csv`: per-window metrics.
- `data/<symbol>_walkforward_summary.json`: aggregate metrics for reporting.

## Dashboard

Launch locally:

```bash
streamlit run src/viz/dashboard.py
```

Open: `http://localhost:8501`

Dashboard includes:

- Overview: candlestick + volume and cleaning report.
- Forecast: ARIMA/LSTM forecasts with confidence bands.
- Volatility: GARCH forecast and clustering visualization.
- Risk: VaR and Expected Shortfall.
- Inference: stationarity and structural break test outputs.

## Project Structure

```text
FinTimeSeries-Pipeline/
├── src/
│   ├── data/         # fetching, cleaning, storage
│   ├── models/       # ARIMA, GARCH, LSTM baseline
│   ├── inference/    # confidence intervals, hypothesis tests
│   ├── viz/          # Streamlit dashboard
│   └── utils/        # config and metric helpers
├── config/           # YAML runtime parameters
├── notebooks/        # exploration notebooks
├── tests/            # pytest suite
├── Dockerfile
├── requirements.txt
└── run.py
```

## Testing

```bash
python -m pytest -q tests
```

CI is configured in `.github/workflows/ci.yml` and runs tests on push/PR.

Quality gates include:

- `ruff` static checks.
- `mypy` targeted type checks.

## Backtesting Notes

`GARCHModel.backtest(...)` supports:

- `window_mode="rolling"`: fixed lookback window (default).
- `window_mode="expanding"`: recursively expanding estimation window.

Returned metrics include Sharpe ratio, cumulative return, hit rate, and max drawdown.

## Citation and References

Foundational references used for implementation choices:

- Box, Jenkins, Reinsel, Ljung. Time Series Analysis: Forecasting and Control.
- Hamilton. Time Series Analysis.
- Engle (1982), Autoregressive Conditional Heteroscedasticity.
- Bollerslev (1986), Generalized ARCH.
- Hyndman and Athanasopoulos. Forecasting: Principles and Practice.

## License

MIT License.

## Contact

- GitHub: https://github.com/your-username
- Email: your.email@example.com
