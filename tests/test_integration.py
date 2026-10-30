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


def test_configured_trials_file_exists(monkeypatch):
    """human_trials_path must point to the pilot sample or to the imported real data, and the file must exist."""
    from common import load_config, trials_path
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    path = trials_path(load_config())
    assert path in (ROOT / "data" / "sample" / "sample_logs.csv", ROOT / "data" / "real" / "participant_trials.csv")
    assert path.exists(), f"{path} not found: run src/import_web_data.py or fix dataset.human_trials_path"


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

    base = p.load_human_trials(_cfg_with(csv, max_valid_rt_ms=15000, max_valid_rt_ms_by_condition={}))
    assert base["is_valid"].tolist() == [True, False, True, False]

    raised = p.load_human_trials(_cfg_with(csv, max_valid_rt_ms=15000, max_valid_rt_ms_by_condition={"unrestricted": 60000}))
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


def _write(tmp_path, rows):
    csv = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_focus_lost_and_exposure_overshoot_are_flagged(monkeypatch, tmp_path):
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    base = dict(participant_id="P1", stimulus_id="s", is_ai=1, user_choice=1)
    csv = _write(tmp_path, [
        dict(base, condition="3s_limit", response_time_ms=1000, exposure_ms=1000, focus_lost=0),      # ok, early answer
        dict(base, condition="3s_limit", response_time_ms=4000, exposure_ms=3010, focus_lost=0),      # ok, timer hid the face
        dict(base, condition="3s_limit", response_time_ms=4000, exposure_ms=3400, focus_lost=0),      # timer failure
        dict(base, condition="3s_limit", response_time_ms=1500, exposure_ms=1500, focus_lost=1),      # tab switched
        dict(base, condition="unrestricted", response_time_ms=9000, exposure_ms=9000, focus_lost=0),  # ok: no exposure rule here
    ])
    df = p.load_human_trials(_cfg_with(csv))
    assert df["is_valid"].tolist() == [True, True, False, False, True]
    assert df["invalid_exposure"].tolist() == [False, False, True, False, False]
    assert df["invalid_focus"].tolist() == [False, False, False, True, False]
    # rules can be switched off in the config
    off = p.load_human_trials(_cfg_with(csv, exclude_focus_lost=False, max_exposure_overshoot_ms=None))
    assert off["is_valid"].all()


def test_rules_are_skipped_without_columns(monkeypatch):
    """The pilot file has no focus_lost / exposure_ms columns: the rules must not break or exclude anything."""
    import process_data as p
    from common import load_config
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    df = p.load_human_trials(load_config())
    assert not df[["invalid_focus", "invalid_exposure"]].any().any()


def test_no_upper_rt_limit_by_default(monkeypatch, tmp_path):
    """max_valid_rt_ms is null: slow answers are valid, the lower bound still applies."""
    import process_data as p
    from common import load_config
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    assert load_config()["guardrails"]["max_valid_rt_ms"] is None
    base = dict(participant_id="P1", stimulus_id="s", is_ai=1, user_choice=1)
    csv = _write(tmp_path, [
        dict(base, condition="unrestricted", response_time_ms=900000),   # 15 minutes: still valid
        dict(base, condition="3s_limit", response_time_ms=120),          # too fast: invalid
        dict(base, condition="3s_limit", response_time_ms=1500),
    ])
    df = p.load_human_trials(_cfg_with(csv))
    assert df["is_valid"].tolist() == [True, False, True]
