"""Tests for the web-experiment integration. Run from the repo root: python -m pytest -q"""
import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = ROOT / "tests" / "fixtures" / "web_export_sample.csv"


def _cfg_with(path: Path, **guardrail_overrides) -> dict:
    from common import load_config
    cfg = copy.deepcopy(load_config())
    cfg["dataset"]["human_trials_path"] = str(path)
    cfg["guardrails"].update(guardrail_overrides)
    return cfg


def _run_import(*args):
    return subprocess.run([sys.executable, str(ROOT / "src" / "import_web_data.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_web_export_schema_is_readable_by_pipeline(monkeypatch):
    """A CSV in the web schema (extra columns included) goes through load + aggregate."""
    import numpy as np
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    cfg = _cfg_with(FIXTURE)
    df = p.load_human_trials(cfg)
    assert len(df) == 16 and df["participant_id"].nunique() == 2
    assert set(df["condition"]) == {"3s_limit", "unrestricted"}
    metrics = p.aggregate_human(df, cfg, np.random.default_rng(1))
    assert set(metrics["condition"]) == {"3s_limit", "unrestricted"}


def test_default_config_points_to_pilot_sample(monkeypatch):
    """Tests and the reference README tables rely on the pilot file until the user switches."""
    from common import load_config, trials_path
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    assert trials_path(load_config()) == ROOT / "data" / "sample" / "sample_logs.csv"


def test_env_var_overrides_trials_path(monkeypatch, tmp_path):
    from common import load_config, trials_path
    monkeypatch.setenv("AVATAR_TRIALS_PATH", str(tmp_path / "x.csv"))
    assert trials_path(load_config()) == tmp_path / "x.csv"


def test_per_condition_rt_limit(monkeypatch, tmp_path):
    """Slow answers are invalid by default, valid for a condition with a raised limit."""
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    csv = tmp_path / "t.csv"
    pd.DataFrame({
        "participant_id": ["P1"] * 4, "condition": ["3s_limit", "3s_limit", "unrestricted", "unrestricted"],
        "stimulus_id": list("abcd"), "is_ai": [1, 0, 1, 0], "user_choice": [1, 0, 1, 0],
        "response_time_ms": [1000, 20000, 1000, 20000],
    }).to_csv(csv, index=False)

    base = p.load_human_trials(_cfg_with(csv, max_valid_rt_ms_by_condition={}))
    assert base["is_valid"].tolist() == [True, False, True, False]

    raised = p.load_human_trials(_cfg_with(csv, max_valid_rt_ms_by_condition={"unrestricted": 60000}))
    assert raised["is_valid"].tolist() == [True, False, True, True]


def test_import_copies_valid_file_and_refuses_overwrite(tmp_path):
    dst = tmp_path / "real" / "participant_trials.csv"
    ok = _run_import("--src", FIXTURE, "--dst", dst)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert dst.read_bytes() == FIXTURE.read_bytes()
    again = _run_import("--src", FIXTURE, "--dst", dst)
    assert again.returncode != 0 and "already exists" in again.stdout
    assert _run_import("--src", FIXTURE, "--dst", dst, "--force").returncode == 0


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.drop(columns=["is_ai"]), "missing columns"),
    (lambda d: d.assign(condition="weird"), "unknown condition"),
    (lambda d: d.assign(user_choice=2), "0/1"),
])
def test_import_rejects_bad_files(tmp_path, mutate, expected):
    bad = tmp_path / "bad.csv"
    mutate(pd.read_csv(FIXTURE)).to_csv(bad, index=False)
    res = _run_import("--src", bad, "--dst", tmp_path / "out.csv")
    assert res.returncode != 0 and expected in res.stdout
    assert not (tmp_path / "out.csv").exists()
