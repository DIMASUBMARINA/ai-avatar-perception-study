"""Pipeline tests. Run from the repo root: python -m pytest -q"""
import json
import os
import shutil
import yaml
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
    """Minimal repo copy so tests never touch real data."""
    dst = tmp_path / "repo"
    (dst / "data").mkdir(parents=True)
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data" / "sample", dst / "data" / "sample")
    shutil.copy(ROOT / "config.yaml", dst / "config.yaml")
    # the tests are built on the pilot file, whatever human_trials_path the real config points to
    cfg_path = dst / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["dataset"]["human_trials_path"] = cfg["dataset"]["sample_logs_path"]
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return dst


# Non-UTF-8 console (e.g. cp932 on Windows): scripts must print ASCII-safe text only
CONSOLE_ENV = {**os.environ, "PYTHONIOENCODING": "cp932"}


def _run(repo: Path, script: str, cwd: Path):
    return subprocess.run([sys.executable, str(repo / "src" / script)], cwd=cwd,
                          capture_output=True, text=True, env=CONSOLE_ENV)


def _process(repo: Path, out: Path):
    code = (f"import sys; sys.path.insert(0, r'{repo / 'src'}'); import process_data as p; "
            f"p.process_and_aggregate(r'{repo / 'data/raw/test_run.jsonl'}', r'{out}')")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


# --- basic functionality ---
def test_scripts_run_from_any_directory(repo_copy, tmp_path):
    """Scripts must not depend on the working directory."""
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    for script in ("benchmark.py", "process_data.py", "visualize.py"):
        res = _run(repo_copy, script, elsewhere)
        assert res.returncode == 0, f"{script} failed:\n{res.stderr[-800:]}"
    assert (repo_copy / "data/raw/test_run.jsonl").exists()
    assert (repo_copy / "data/processed/metrics_summary.csv").exists()
    for fig in ("fig1_latency_vs_load", "fig2_distribution_boxplot"):
        for ext in ("pdf", "png"):
            assert (repo_copy / "reports/figures" / f"{fig}.{ext}").exists()


def test_d_prime_uses_hautus_correction():
    """hit=1, FA=0, n=1 -> 2*Phi^-1(0.75) ~= 1.349, not 0."""
    from process_data import calculate_d_prime
    assert calculate_d_prime(1.0, 0.0, 1, 1) == pytest.approx(1.3490, abs=1e-3)
    assert np.isfinite(calculate_d_prime(1.0, 0.0, 2, 2))


def test_pipeline_is_deterministic(repo_copy, tmp_path):
    """Same input, same seed -> identical CSVs."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    runs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        res = _process(repo_copy, out)
        assert res.returncode == 0, res.stderr[-800:]
        runs.append({n: pd.read_csv(out / f"{n}.csv") for n in ("metrics_summary", "latency_by_load")})
    for name in runs[0]:
        pd.testing.assert_frame_equal(runs[0][name], runs[1][name])


def test_summary_metrics_on_sample(repo_copy):
    """Reference values on the sample data."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    assert _run(repo_copy, "process_data.py", repo_copy).returncode == 0
    m = pd.read_csv(repo_copy / "data/processed/metrics_summary.csv").set_index("condition")
    assert m.loc["3s_limit", "accuracy_pct"] == 75.0
    assert m.loc["3s_limit", "fpr_pct"] == 25.0 and m.loc["3s_limit", "fnr_pct"] == 25.0
    assert m.loc["3s_limit", "tir_pct"] == 0.0
    assert m.loc["unrestricted", "accuracy_pct"] == 100.0


# --- data processing ---
def test_raw_log_has_phases_and_enough_records(repo_copy):
    """Raw log has both phases and at least 100 benchmark records."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    raw = pd.read_json(repo_copy / "data/raw/test_run.jsonl", lines=True)
    assert set(raw["phase"]) == {"warmup", "benchmark"}
    assert (raw["phase"] == "benchmark").sum() >= 100
    assert (repo_copy / "data/raw/run_metadata.json").exists()


def test_warmup_is_discarded_in_aggregation(repo_copy, tmp_path):
    """Aggregates use benchmark runs only."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    out = tmp_path / "out"
    assert _process(repo_copy, out).returncode == 0
    raw = pd.read_json(repo_copy / "data/raw/test_run.jsonl", lines=True)
    lat = pd.read_csv(out / "latency_by_load.csv")
    assert lat["n_runs"].sum() == (raw["phase"] == "benchmark").sum()


def test_latency_ci_brackets_percentiles(repo_copy, tmp_path):
    """Median CI brackets the estimate; p95/p99 are present."""
    assert _run(repo_copy, "benchmark.py", repo_copy).returncode == 0
    out = tmp_path / "out"
    assert _process(repo_copy, out).returncode == 0
    lat = pd.read_csv(out / "latency_by_load.csv")
    assert {"p50_us", "p95_us", "p99_us"} <= set(lat.columns)
    assert (lat["p50_ci95_low_us"] <= lat["p50_us"] + 1e-9).all()
    assert (lat["p50_us"] <= lat["p50_ci95_high_us"] + 1e-9).all()
    assert lat["guardrail_ok"].all()


def test_schema_validation_rejects_bad_log(tmp_path):
    from process_data import load_benchmark_log
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"phase": "benchmark", "load_level": 1}) + "\n")
    with pytest.raises(ValueError):
        load_benchmark_log(bad)


# --- repository state ---
def test_required_task3_files_exist():
    for rel in ("data/raw/test_run.jsonl", "data/processed/metrics_summary.csv", "docs/domain_model.md",
                "notebooks/01_data_analysis.ipynb", "src/process_data.py", "src/visualize.py",
                ".gitignore", "README.md", "config.yaml"):
        assert (ROOT / rel).exists(), f"missing {rel}"


def test_figures_exported_as_pdf_and_300dpi_png():
    from PIL import Image
    for name in ("fig1_latency_vs_load", "fig2_distribution_boxplot"):
        pdf, png = ROOT / "reports/figures" / f"{name}.pdf", ROOT / "reports/figures" / f"{name}.png"
        assert pdf.read_bytes()[:4] == b"%PDF"
        assert round(Image.open(png).info["dpi"][0]) >= 300


def test_gitignore_rules():
    """Deliverables are tracked; caches and environments are ignored."""
    lines = {l.strip() for l in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "data/processed/" not in lines and "reports/figures/" not in lines
    assert any("ipynb_checkpoints" in l for l in lines)
    assert "__pycache__/" in lines and ".venv/" in lines


def test_notebook_is_executed_in_order_without_errors():
    import nbformat
    nb = nbformat.read(ROOT / "notebooks/01_data_analysis.ipynb", 4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 5, "notebook is empty"
    assert [c.execution_count for c in code] == list(range(1, len(code) + 1)), "cells not executed in order"
    assert all(c.outputs for c in code), "every cell must keep its output"
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert any("seed" in c.source.lower() for c in code), "seed must be fixed"
    assert any(c.cell_type == "markdown" for c in nb.cells)


def test_domain_model_has_formal_artifact():
    text = (ROOT / "docs/domain_model.md").read_text(encoding="utf-8")
    assert "$$" in text and "```mermaid" in text
