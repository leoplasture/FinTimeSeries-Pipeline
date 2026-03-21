"""Setup configuration for packaging FinTimeSeries-Pipeline."""

from setuptools import find_packages, setup


setup(
    name="FinTimeSeries-Pipeline",
    version="0.1.0",
    description="Professional financial time series analysis pipeline.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "statsmodels>=0.14",
        "scikit-learn>=1.2",
        "plotly>=5.15",
        "streamlit>=1.30",
        "PyYAML>=6.0",
        "yfinance>=0.2",
        "SQLAlchemy>=2.0",
        "arch>=6.3",
        "matplotlib>=3.8",
    ],
    python_requires=">=3.9",
)
