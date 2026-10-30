"""Shared helpers: project root, config, trial validation, environment metadata."""
import os
import platform
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path: str) -> Path:
    """Resolve a config path against the repository root."""
    return ROOT / rel_path


def is_valid_trial(response_time_ms: float, min_rt: float, max_rt: float = None) -> bool:
    """Lower bound is always applied; max_rt=None means no upper limit."""
    return response_time_ms >= min_rt and (max_rt is None or response_time_ms <= max_rt)


def max_rt_for(condition: str, guardrails: dict):
    """Upper RT limit for a condition: per-condition override, else the global max_valid_rt_ms (None = no limit)."""
    overrides = guardrails.get("max_valid_rt_ms_by_condition") or {}
    return overrides.get(condition, guardrails.get("max_valid_rt_ms"))


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


def trial_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Boolean exclusion flags per trial (True = the trial breaks a rule). One column per reason.

    invalid_rt        response time outside min..max (the max may be set per condition)
    invalid_focus     the tab lost focus during the trial (needs a focus_lost column, rule exclude_focus_lost)
    invalid_exposure  3s_limit only: the face stayed on screen longer than the limit + max_exposure_overshoot_ms
                      (needs an exposure_ms column); early answers (shorter exposure) are fine
    Rules whose column is missing (e.g. the pilot file) are skipped.
    """
    g = cfg["guardrails"]
    flags = pd.DataFrame(index=df.index)
    flags["invalid_rt"] = [not is_valid_trial(rt, g["min_valid_rt_ms"], max_rt_for(cond, g))
                           for rt, cond in zip(df["response_time_ms"], df["condition"])]
    if g.get("exclude_focus_lost") and "focus_lost" in df.columns:
        flags["invalid_focus"] = pd.to_numeric(df["focus_lost"], errors="coerce").fillna(0).astype(int) == 1
    else:
        flags["invalid_focus"] = False
    tol = g.get("max_exposure_overshoot_ms")
    if tol is not None and "exposure_ms" in df.columns:
        limit = cfg["experiment"]["viewing_time_limit_ms"]
        exposure = pd.to_numeric(df["exposure_ms"], errors="coerce")
        flags["invalid_exposure"] = ((df["condition"] == "3s_limit") & (exposure > limit + tol)).fillna(False)
    else:
        flags["invalid_exposure"] = False
    return flags
