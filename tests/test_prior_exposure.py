"""Tests for the prior-exposure question, the sensitivity analysis and the leakage check."""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FIXTURE = ROOT / "tests" / "fixtures" / "web_export_sample.csv"


def _csv(tmp_path, correct_3s, correct_free, prior=None, trials=20):
    """Participants finish one minute apart; correct_* are the numbers of correct answers out of `trials`."""
    rows = []
    for k in range(len(correct_3s)):
        for cond, n_ok in (("3s_limit", correct_3s[k]), ("unrestricted", correct_free[k])):
            for t in range(trials):
                is_ai = t % 2
                rows.append(dict(participant_id=f"P{k:02d}", condition=cond, stimulus_id=f"{cond}{t}", is_ai=is_ai,
                                 user_choice=is_ai if t < n_ok else 1 - is_ai,
                                 response_time_ms=1000 + 50 * t if cond == "3s_limit" else 3000 + 100 * t,
                                 prior_exposure=(prior or ["no"] * len(correct_3s))[k],
                                 submitted_at=f"2026-01-01T10:{k:02d}:00.000Z"))
    path = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cfg(path, **participants):
    from common import load_config
    cfg = copy.deepcopy(load_config())
    cfg["dataset"]["human_trials_path"] = str(path)
    cfg["participants"].update(participants)
    return cfg


def _paired_acc(tests, sample):
    t = tests[(tests["sample"] == sample) & tests["analysis"].str.startswith("paired") & (tests["metric"] == "accuracy_pct")]
    return t.iloc[0]


def test_primary_excludes_prior_yes_and_sensitivity_keeps_everyone(tmp_path, monkeypatch):
    import participant_level as pl
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    c3 = [12, 13, 14, 12, 13, 14, 13, 12, 20, 20]       # the last two know the answers
    cf = [16, 17, 18, 17, 16, 18, 17, 16, 20, 20]
    prior = ["no"] * 8 + ["yes"] * 2
    cfg = _cfg(_csv(tmp_path, c3, cf, prior))
    table, tests, _ = pl.run(cfg, tmp_path / "out")
    assert set(tests["sample"]) == {"primary", "sensitivity: all participants", "check: all participants"}
    assert _paired_acc(tests, "primary").n == 8
    assert _paired_acc(tests, "sensitivity: all participants").n == 10
    assert table.set_index("participant_id").loc[["P08", "P09"], "in_primary"].eq(False).all()
    assert table["in_primary"].sum() == 2 * 8            # two rows (conditions) per participant who stays
    # with an empty exclusion list everybody is in the primary analysis and there is no separate sensitivity run
    _, tests2, _ = pl.run(_cfg(_csv(tmp_path, c3, cf, prior), exclude_prior_exposure=[]), tmp_path / "out2")
    assert _paired_acc(tests2, "primary").n == 10
    assert "sensitivity: all participants" not in set(tests2["sample"])


def test_unsure_is_kept_by_default(tmp_path, monkeypatch):
    import participant_level as pl
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    prior = ["unsure"] * 3 + ["no"] * 3
    table, _, _ = pl.run(_cfg(_csv(tmp_path, [14] * 6, [17] * 6, prior)), tmp_path / "out")
    assert table["in_primary"].all()


def test_leakage_check_detects_rising_accuracy(tmp_path, monkeypatch):
    import participant_level as pl
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    rising = [10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    _, tests, _ = pl.run(_cfg(_csv(tmp_path, rising, rising)), tmp_path / "out")
    chk = tests[tests["sample"] == "check: all participants"].set_index("analysis")
    sp = chk.loc["accuracy vs order of finishing (Spearman rho)"]
    assert sp.estimate > 0.9 and sp.p_value < 0.01
    halves = chk.loc["accuracy second half - first half (Mann-Whitney)"]
    assert halves.estimate > 0 and halves.p_value < 0.05


def test_leakage_check_quiet_when_accuracy_does_not_drift(tmp_path, monkeypatch):
    import participant_level as pl
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    flat = [14, 18, 15, 17, 16, 16, 18, 14, 17, 15, 16, 16]
    _, tests, _ = pl.run(_cfg(_csv(tmp_path, flat, flat)), tmp_path / "out")
    sp = tests[tests["analysis"].str.contains("Spearman")].iloc[0]
    assert sp.p_value > 0.3


def test_leakage_check_needs_enough_participants(tmp_path, monkeypatch):
    import participant_level as pl
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    _, tests, _ = pl.run(_cfg(_csv(tmp_path, [14, 15, 16], [17, 18, 19])), tmp_path / "out")
    chk = tests[tests["sample"] == "check: all participants"]
    assert chk["p_value"].isna().all() and chk["note"].str.contains("fewer than 8").all()


def test_files_without_new_columns_still_work(tmp_path, monkeypatch):
    """The pilot file has neither prior_exposure nor submitted_at: only the primary rows are produced."""
    import participant_level as pl
    from common import load_config
    monkeypatch.delenv("AVATAR_TRIALS_PATH", raising=False)
    cfg = load_config()
    cfg["dataset"]["human_trials_path"] = cfg["dataset"]["sample_logs_path"]   # always the pilot, whatever the real config says
    _, tests, _ = pl.run(cfg, tmp_path / "out")
    assert set(tests["sample"]) == {"primary"}


def test_import_checks_prior_exposure_values(tmp_path):
    script = str(ROOT / "src" / "import_web_data.py")
    good = subprocess.run([sys.executable, script, "--src", str(FIXTURE), "--dst", str(tmp_path / "ok.csv")],
                          capture_output=True, text=True)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "prior_exposure (participants): no=1, unsure=1, yes=0" in good.stdout
    bad = tmp_path / "bad.csv"
    pd.read_csv(FIXTURE).assign(prior_exposure="maybe").to_csv(bad, index=False)
    res = subprocess.run([sys.executable, script, "--src", str(bad), "--dst", str(tmp_path / "out.csv")],
                         capture_output=True, text=True)
    assert res.returncode != 0 and "prior_exposure" in res.stdout and not (tmp_path / "out.csv").exists()
