"""Shared helpers: project root, config, trial validation, environment metadata."""
import os
import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Resolve a config path against the repository root."""
    return ROOT / rel_path


def is_valid_trial(response_time_ms: float, min_rt: float, max_rt: float) -> bool:
    return min_rt <= response_time_ms <= max_rt


def environment_metadata() -> dict:
    ram_gb = None
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except (ValueError, OSError, AttributeError):
        pass  # sysconf is unavailable on Windows
    return {
        "python_version": platform.python_version(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Generic CPU",
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
    }
