"""Tests for the participant-level analysis. Run from the repo root: python -m pytest -q"""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURE = ROOT / "tests" / "fixtures" / "web_export_sample.csv"


def _synthetic(tmp_path, n_participants, correct_3s, correct_free, trials=20):
    """CSV with n participants; per participant the numbers of correct answers (out of `trials`) in each condition."""
    rows = []
    for k in range(n_participants):
        for cond, n_ok in (("3s_limit", correct_3s[k]), ("unrestricted", correct_free[k])):
            for t in range(trials):
                is_ai = t % 2
                correct = t < n_ok
                rows.append(dict(participant_id=f"P{k:02d}", condition=cond, stimulus_id=f"{cond}{t}", is_ai=is_ai,
                                 user_choice=is_ai if correct else 1 - is_ai,
                                 response_time_ms=1000 + 50 * t if cond == "3s_limit" else 3000 + 100 * t))
    path = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cfg(path):
    from common import load_config
    cfg = copy.deepcopy(load_config())
    cfg["dataset"]["human_trials_path"] = str(path)
    return cfg


def test_participant_table_values(tmp_path, monkeypatch):
    import participant_level as pl
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    path = _synthetic(tmp_path, 2, [15, 10], [18, 20])
    tab = pl.participant_table(p.load_human_trials(_cfg(path)))
    row = tab[(tab.participant_id == "P00") & (tab.condition == "3s_limit")].iloc[0]
    assert row.n_trials == 20 and row.n_valid == 20 and row.accuracy_pct == 75.0
    assert row.accuracy_all_pct == 75.0 and row.tir_pct == 0.0
    # correct = trials t < 15, AI = odd t: AI answered right 7 of 10 -> FNR 30%; real answered right 8 of 10 -> FPR 20%
    assert row.fnr_pct == 30.0 and row.fpr_pct == 20.0


def test_paired_test_detects_clear_difference(tmp_path, monkeypatch):
    import participant_level as pl
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    rng = np.random.default_rng(3)
    n = 14
    c3 = rng.integers(11, 15, n)       # about 55-75% accuracy
    cf = rng.integers(17, 20, n)       # about 85-95% accuracy
    cfg = _cfg(_synthetic(tmp_path, n, c3, cf))
    tab = pl.participant_table(p.load_human_trials(cfg))
    tests = pl.participant_tests(tab, cfg, np.random.default_rng(1))
    row = tests[(tests.analysis.str.startswith("paired")) & (tests.metric == "accuracy_pct")].iloc[0]
    assert row.n == n and row.estimate < 0 and row.p_value < 0.01
    assert row.ci_high < 0, "the participant-level CI of the difference must exclude 0"
    chance = tests[tests.analysis.str.contains("chance") & (tests.condition == "unrestricted")].iloc[0]
    assert chance.p_value < 0.01


def test_too_few_participants_gives_note_not_crash(tmp_path, monkeypatch):
    import participant_level as pl
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    cfg = _cfg(_synthetic(tmp_path, 3, [10, 12, 14], [18, 17, 19]))
    tests = pl.participant_tests(pl.participant_table(p.load_human_trials(cfg)), cfg, np.random.default_rng(1))
    paired = tests[tests.analysis.str.startswith("paired")]
    assert paired["p_value"].isna().all() and paired["note"].str.contains("fewer than").all()


def test_identical_conditions_do_not_crash(tmp_path, monkeypatch):
    import participant_level as pl
    import process_data as p
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    same = [15] * 8
    cfg = _cfg(_synthetic(tmp_path, 8, same, same))
    tests = pl.participant_tests(pl.participant_table(p.load_human_trials(cfg)), cfg, np.random.default_rng(1))
    row = tests[tests.analysis.str.startswith("paired") & (tests.metric == "accuracy_pct")].iloc[0]
    assert row.estimate == 0 and np.isnan(row.p_value) and "zero" in row.note


def test_script_writes_outputs(tmp_path):
    import os
    env = dict(os.environ, AVATAR_TRIALS_PATH=str(FIXTURE))
    res = subprocess.run([sys.executable, str(ROOT / "src" / "participant_level.py"), "--out", str(tmp_path)],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert (tmp_path / "participant_table.csv").exists() and (tmp_path / "participant_level_tests.csv").exists()
    assert len(pd.read_csv(tmp_path / "participant_table.csv")) == 4   # 2 participants x 2 conditions
