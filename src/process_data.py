"""Data processing: schema check, warm-up filter, aggregation, bootstrap CIs, guardrail, tests.

Writes to data/processed/: metrics_summary.csv, latency_by_load.csv, statistical_tests.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from common import ROOT, load_config, resolve, trial_flags, trials_path

REQUIRED_LOG_COLUMNS = {
    "phase", "run_id", "load_level", "batch_latency_us", "n_invalid", "error_rate_pct", "runtime_error",
}


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(arrays: tuple, stat_fn, n_boot: int, level: float, rng) -> tuple:
    """Percentile-bootstrap CI. `arrays` are resampled jointly; `stat_fn` maps (n_boot, n) -> (n_boot,)."""
    n = len(arrays[0])
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = stat_fn(*[np.asarray(a)[idx] for a in arrays])
    alpha = (100.0 - level) / 2.0
    lo, hi = np.nanpercentile(samples, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def _mean(x): return x.mean(axis=1)
def _median(x): return np.median(x, axis=1)
def _p(q): return lambda x: np.percentile(x, q, axis=1)
def _acc(ai, ch): return (ai == ch).mean(axis=1)


def _fpr(ai, ch):
    neg = (ai == 0).sum(axis=1)
    fp = ((ai == 0) & (ch == 1)).sum(axis=1)
    return np.where(neg > 0, fp / np.maximum(neg, 1), np.nan)


def _fnr(ai, ch):
    pos = (ai == 1).sum(axis=1)
    fn = ((ai == 1) & (ch == 0)).sum(axis=1)
    return np.where(pos > 0, fn / np.maximum(pos, 1), np.nan)


def calculate_d_prime(hit_rate: float, fa_rate: float, n_signal: int, n_noise: int) -> float:
    """Sensitivity index d' with Hautus log-linear correction: (k + 0.5) / (n + 1)."""
    adjusted_hr = (hit_rate * n_signal + 0.5) / (n_signal + 1.0)
    adjusted_far = (fa_rate * n_noise + 0.5) / (n_noise + 1.0)
    return float(norm.ppf(adjusted_hr) - norm.ppf(adjusted_far))


def iqr_outlier_flag(values: pd.Series, factor: float) -> pd.Series:
    """Flag IQR outliers without removing them."""
    q25, q75 = values.quantile(0.25), values.quantile(0.75)
    iqr = q75 - q25
    return (values < q25 - factor * iqr) | (values > q75 + factor * iqr)


# ----------------------------------------------------------------------------- part 1: system latency
def load_benchmark_log(path=None) -> pd.DataFrame:
    """Load the raw JSONL log and validate its schema."""
    path = path or resolve(load_config()["dataset"]["jsonl_output_path"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/benchmark.py` first.")
    df = pd.read_json(path, lines=True)
    missing = REQUIRED_LOG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Raw log is missing required columns: {sorted(missing)}")
    bad = set(df["phase"].unique()) - {"warmup", "benchmark"}
    if bad:
        raise ValueError(f"Unknown phase values: {sorted(bad)}")
    return df


def filter_warmup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up runs, keep phase == 'benchmark'."""
    return df[df["phase"] == "benchmark"].copy()


def aggregate_latency(df_bench: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    n_boot = cfg["stats"]["bootstrap_iterations"]
    level = cfg["stats"]["ci_level"]
    factor = cfg["stats"]["iqr_factor"]
    rows = []
    for load, g in df_bench.groupby("load_level"):
        y = g["batch_latency_us"].to_numpy()
        row = {
            "load_level": int(load),
            "n_runs": len(y),
            "mean_us": y.mean(),
            "sd_us": y.std(ddof=1),
            "p25_us": np.percentile(y, 25),
            "p50_us": np.percentile(y, 50),
            "p75_us": np.percentile(y, 75),
            "p95_us": np.percentile(y, 95),
            "p99_us": np.percentile(y, 99),
        }
        for name, fn in (("p50", _median), ("p95", _p(95)), ("p99", _p(99))):
            lo, hi = bootstrap_ci((y,), fn, n_boot, level, rng)
            row[f"{name}_ci95_low_us"], row[f"{name}_ci95_high_us"] = lo, hi
        row["n_iqr_outliers"] = int(iqr_outlier_flag(g["batch_latency_us"], factor).sum())
        row["error_rate_pct"] = 100.0 * g["n_invalid"].sum() / g["load_level"].sum()
        row["runtime_errors"] = int(g["runtime_error"].sum())
        row["guardrail_ok"] = bool(row["error_rate_pct"] <= cfg["guardrails"]["max_allowed_tir_percent"])
        rows.append(row)
    return pd.DataFrame(rows).round(3)


# ----------------------------------------------------------------------------- part 2: human perception (pilot)
def load_human_trials(cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df = pd.read_csv(trials_path(cfg))
    flags = trial_flags(df, cfg)
    df = pd.concat([df, flags], axis=1)
    df["is_valid"] = ~flags.any(axis=1)
    df["is_correct"] = (df["is_ai"] == df["user_choice"])
    df["is_iqr_outlier"] = False
    for cond, idx in df.groupby("condition").groups.items():
        df.loc[idx, "is_iqr_outlier"] = iqr_outlier_flag(df.loc[idx, "response_time_ms"], cfg["stats"]["iqr_factor"])
    return df


def aggregate_human(df: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    """Positive class = AI avatar. FP: real judged AI; FN: AI judged real."""
    n_boot, level = cfg["stats"]["bootstrap_iterations"], cfg["stats"]["ci_level"]
    rows = []
    for cond, g in df.groupby("condition"):
        n_total = len(g)
        tir = 100.0 * (~g["is_valid"]).sum() / n_total  # invalid trials are counted, not silently dropped
        v = g[g["is_valid"]]
        ai, ch, rt = v["is_ai"].to_numpy(), v["user_choice"].to_numpy(), v["response_time_ms"].to_numpy()
        tp = int(((ai == 1) & (ch == 1)).sum()); tn = int(((ai == 0) & (ch == 0)).sum())
        fp = int(((ai == 0) & (ch == 1)).sum()); fn = int(((ai == 1) & (ch == 0)).sum())
        hit_rate = tp / (tp + fn) if (tp + fn) else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) else 0.0
        row = {
            "condition": cond, "n_total": n_total, "sample_size": len(v), "tir_pct": tir,
            "accuracy_pct": 100.0 * (tp + tn) / len(v),
            "fpr_pct": 100.0 * fa_rate if (fp + tn) else np.nan,
            "fnr_pct": 100.0 * fn / (tp + fn) if (tp + fn) else np.nan,
            "d_prime": calculate_d_prime(hit_rate, fa_rate, tp + fn, fp + tn),
            "mean_rt_ms": rt.mean(), "std_rt_ms": rt.std(ddof=1) if len(rt) > 1 else np.nan,
            "p25_rt_ms": np.percentile(rt, 25), "median_p50_ms": np.percentile(rt, 50),
            "p75_rt_ms": np.percentile(rt, 75), "p95_rt_ms": np.percentile(rt, 95),
            "p99_rt_ms": np.percentile(rt, 99),
            "n_iqr_outliers": int(v["is_iqr_outlier"].sum()),
        }
        for name, arrays, fn_ in (
            ("mean_rt_ms", (rt,), _mean), ("median_rt_ms", (rt,), _median),
            ("accuracy_pct", (ai, ch), _acc), ("fpr_pct", (ai, ch), _fpr), ("fnr_pct", (ai, ch), _fnr),
        ):
            lo, hi = bootstrap_ci(arrays, fn_, n_boot, level, rng)
            scale = 100.0 if name.endswith("_pct") else 1.0
            row[f"{name}_ci95_low"], row[f"{name}_ci95_high"] = lo * scale, hi * scale
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def run_statistical_tests(df: pd.DataFrame, a: str = "3s_limit", b: str = "unrestricted") -> pd.DataFrame:
    """Compare conditions (illustrative only at this sample size)."""
    va, vb = (df[(df["condition"] == c) & df["is_valid"]] for c in (a, b))
    out = []
    if len(va) and len(vb):
        table = [[int(va["is_correct"].sum()), int((~va["is_correct"]).sum())],
                 [int(vb["is_correct"].sum()), int((~vb["is_correct"]).sum())]]
        _, p_acc = stats.fisher_exact(table)
        out.append({"test": "Fisher exact (accuracy)", "comparison": f"{a} vs {b}",
                    "n_a": len(va), "n_b": len(vb), "statistic": np.nan, "p_value": p_acc})
        u, p_rt = stats.mannwhitneyu(va["response_time_ms"], vb["response_time_ms"], alternative="two-sided")
        out.append({"test": "Mann-Whitney U (response time)", "comparison": f"{a} vs {b}",
                    "n_a": len(va), "n_b": len(vb), "statistic": u, "p_value": p_rt})
    return pd.DataFrame(out).round(4)


# ----------------------------------------------------------------------------- orchestration
def process_and_aggregate(input_path=None, output_dir=None) -> dict:
    """Run the full processing pass; the fixed rng call order makes results reproducible."""
    from pathlib import Path
    cfg = load_config()
    out = Path(output_dir) if output_dir else resolve(cfg["dataset"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["experiment"]["random_seed"])

    print("[STEP 1] Loading raw JSONL log, validating schema, discarding warm-up runs")
    raw = load_benchmark_log(Path(input_path) if input_path else None)
    bench = filter_warmup(raw)
    print(f"         raw={len(raw)}  warmup dropped={len(raw) - len(bench)}  kept={len(bench)}")

    print("[STEP 2] Aggregating system latency (p50/p95/p99 + bootstrap 95% CI)")
    latency = aggregate_latency(bench, cfg, rng)

    print(f"[STEP 3] Aggregating human perception metrics from {trials_path(cfg)}")
    trials = load_human_trials(cfg)
    metrics = aggregate_human(trials, cfg, rng)
    tests = run_statistical_tests(trials)

    metrics.to_csv(out / "metrics_summary.csv", index=False)
    latency.to_csv(out / "latency_by_load.csv", index=False)
    tests.to_csv(out / "statistical_tests.csv", index=False)
    print(f"[EXPORT] metrics_summary.csv, latency_by_load.csv, statistical_tests.csv -> {out}")
    return {"metrics": metrics, "latency": latency, "tests": tests}


if __name__ == "__main__":
    res = process_and_aggregate()
    for name, frame in res.items():
        print(f"\n--- {name} ---")
        print(frame.to_string(index=False))
