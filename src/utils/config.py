"""Configuration management for reproducible financial time-series experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


class ConfigError(ValueError):
    """Raised when configuration content is missing or invalid."""


@dataclass
class AppConfig:
    """Container for typed access to project configuration."""

    raw: Dict[str, Any]

    @property
    def project_name(self) -> str:
        return str(self.raw.get("project", {}).get("name", "FinTimeSeries-Pipeline"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("project", {}).get("seed", 42))

    @property
    def symbol(self) -> str:
        return str(self.raw.get("data", {}).get("symbol", "AAPL"))


def load_config(config_path: str, env_prefix: Optional[str] = None) -> Dict[str, Any]:
    """Load and parse a YAML configuration file with optional env override.

    Parameters:
        config_path (str): Path to YAML config file.
        env_prefix (str | None): Optional environment prefix, e.g., `FTSP_`.

    Returns:
        dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file path does not exist.
        ConfigError: If YAML is invalid or required sections are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML at {config_path}") from exc

    if env_prefix:
        config = apply_env_overrides(config, prefix=env_prefix)

    validate_config(config)
    return config


def save_config(config: Dict[str, Any], output_path: str) -> Path:
    """Persist configuration dictionary to YAML file.

    Parameters:
        config (dict): Config dictionary to save.
        output_path (str): Destination file path.

    Returns:
        pathlib.Path: Written file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=False)
    return path


def validate_config(config: Dict[str, Any]) -> None:
    """Validate presence and basic types for required configuration sections."""
    required_sections = ["project", "data", "model", "inference", "storage"]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    if not isinstance(config.get("project", {}).get("seed", 42), int):
        raise ConfigError("project.seed must be an integer")
    if not isinstance(config.get("data", {}).get("symbol", ""), str):
        raise ConfigError("data.symbol must be a string")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two configuration dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(config: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    """Fetch nested config value by key path.

    Example:
        get_nested(cfg, ["model", "arima_order"], [1, 1, 1])
    """
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def apply_env_overrides(
    config: Dict[str, Any], prefix: str = "FTSP_"
) -> Dict[str, Any]:
    """Apply environment variable overrides using double-underscore path syntax.

    Example:
        FTSP_DATA__SYMBOL=MSFT -> config["data"]["symbol"] = "MSFT"
    """
    result = dict(config)
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix) :].lower().split("__")
        result = _set_nested_value(result, path, _coerce_env_value(env_val))
    return result


def _set_nested_value(
    config: Dict[str, Any], path: Iterable[str], value: Any
) -> Dict[str, Any]:
    updated = dict(config)
    current = updated
    keys = list(path)
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return updated


def _coerce_env_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
