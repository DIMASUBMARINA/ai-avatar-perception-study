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


def max_rt_for(condition: str, guardrails: dict) -> float:
    """Upper RT limit for a condition: per-condition override, else the global max_valid_rt_ms."""
    overrides = guardrails.get("max_valid_rt_ms_by_condition") or {}
    return overrides.get(condition, guardrails["max_valid_rt_ms"])


def trials_path(cfg: dict = None) -> Path:
    """CSV with human trials: env AVATAR_TRIALS_PATH, else dataset.human_trials_path, else sample_logs_path."""
    env = os.environ.get("AVATAR_TRIALS_PATH")
    if env:
        return Path(env)
    ds = (cfg or load_config())["dataset"]
    return resolve(ds.get("human_trials_path") or ds["sample_logs_path"])


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
