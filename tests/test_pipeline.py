"""Smoke-тесты пайплайна. Запуск из корня репозитория: python -m pytest -q"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def repo_copy(tmp_path):
    """Копия минимального скелета репозитория, чтобы тесты не трогали ваши данные."""
    dst = tmp_path / "repo"
    (dst / "data").mkdir(parents=True)
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data" / "sample", dst / "data" / "sample")
    shutil.copy(ROOT / "config.yaml", dst / "config.yaml")
    return dst


def _run(repo: Path, script: str, cwd: Path):
    return subprocess.run([sys.executable, str(repo / "src" / script)], cwd=cwd,
                          capture_output=True, text=True)


def test_scripts_run_from_any_directory(repo_copy, tmp_path):
    """Скрипты не должны зависеть от текущей папки."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    for script in ("benchmark.py", "process_data.py", "visualize.py"):
        res = _run(repo_copy, script, elsewhere)
        assert res.returncode == 0, f"{script} failed:\n{res.stderr[-800:]}"
    assert (repo_copy / "data/raw/test_run.jsonl").exists()
    assert (repo_copy / "data/processed/metrics_summary.csv").exists()
    assert len(list((repo_copy / "reports/figures").glob("*.png"))) == 4


def test_d_prime_uses_hautus_correction():
    """При hit=1, FA=0, n=1 коррекция Хаутуса даёт 2*Phi^-1(0.75) ~= 1.349, а не 0."""
    from process_data import calculate_d_prime
    assert calculate_d_prime(1.0, 0.0, 1, 1) == pytest.approx(1.3490, abs=1e-3)
    # n=2: hr=(2+.5)/3, far=.5/3 -> конечное значение, без +/-inf
    assert np.isfinite(calculate_d_prime(1.0, 0.0, 2, 2))


def test_pipeline_is_deterministic(repo_copy, tmp_path):
    """Два запуска обработки на одних данных должны давать одинаковые CSV (фиксированный seed)."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    outs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        code = (f"import sys; sys.path.insert(0, r'{repo_copy / 'src'}'); "
                f"import process_data as p; p.process_and_aggregate(r'{repo_copy / 'data/raw/test_run.jsonl'}', r'{out}')")
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr[-800:]
        outs.append(pd.read_csv(out / "metrics_summary.csv"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_summary_metrics_on_sample(repo_copy):
    """Контрольные значения на пилотном сэмпле: 3s_limit -> Acc 75%, FPR=FNR=25%, TIR 0%."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    assert _run(repo_copy, "process_data.py", repo_copy).returncode == 0
    m = pd.read_csv(repo_copy / "data/processed/metrics_summary.csv").set_index("condition")
    assert m.loc["3s_limit", "accuracy_pct"] == 75.0
    assert m.loc["3s_limit", "fpr_pct"] == 25.0 and m.loc["3s_limit", "fnr_pct"] == 25.0
    assert m.loc["3s_limit", "tir_pct"] == 0.0
    assert m.loc["unrestricted", "accuracy_pct"] == 100.0
