"""Participant-level analysis of the human trials (the primary analysis for real data).

Every participant answers in both conditions, so trials of one person are not independent.
Here each participant is one observation: accuracy / FPR / FNR / median RT are computed per participant and condition,
and conditions are compared with paired tests and a participant-level (cluster) bootstrap.

    python src/participant_level.py [--out DIR]

Output (data/processed/): participant_table.csv, participant_level_tests.csv
Exploratory metrics (everything except accuracy) are not corrected for multiple comparisons.

participant_level_tests.csv has a `sample` column:
  primary                       participants without the excluded prior_exposure answers (config participants.exclude_prior_exposure)
  sensitivity: all participants the same tests on everybody, to see whether the exclusion changes the conclusions
  check: all participants       leakage check: accuracy vs. the order in which participants finished (Spearman, first vs second half);
                                a rise over time suggests that answers were passed on
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import load_config, resolve, trials_path
from process_data import load_human_trials

A, B = "3s_limit", "unrestricted"
PAIRED_METRICS = ["accuracy_pct", "fpr_pct", "fnr_pct", "median_rt_ms", "accuracy_all_pct"]
MIN_PAIRS = 6   # fewer participants than this: no p-value is reported (Wilcoxon would be meaningless)


def participant_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant x condition. *_pct metrics use valid trials; accuracy_all_pct uses all trials (sensitivity)."""
    rows = []
    for (pid, cond), g in df.groupby(["participant_id", "condition"]):
        v = g[g["is_valid"]]
        neg, pos = v[v["is_ai"] == 0], v[v["is_ai"] == 1]
        rows.append({
            "participant_id": pid, "condition": cond, "n_trials": len(g), "n_valid": len(v),
            "tir_pct": 100.0 * (1 - len(v) / len(g)),
            "accuracy_pct": 100.0 * v["is_correct"].mean() if len(v) else np.nan,
            "fpr_pct": 100.0 * (neg["user_choice"] == 1).mean() if len(neg) else np.nan,
            "fnr_pct": 100.0 * (pos["user_choice"] == 0).mean() if len(pos) else np.nan,
            "median_rt_ms": v["response_time_ms"].median() if len(v) else np.nan,
            "accuracy_all_pct": 100.0 * g["is_correct"].mean(),
        })
    return pd.DataFrame(rows).round(3)


def _wilcoxon(x: np.ndarray) -> tuple:
    """Wilcoxon signed-rank on differences x; (statistic, p, note)."""
    x = x[~np.isnan(x)]
    if len(x) < MIN_PAIRS:
        return np.nan, np.nan, f"fewer than {MIN_PAIRS} participants, no test"
    if np.all(x == 0):
        return np.nan, np.nan, "all differences are zero"
    res = stats.wilcoxon(x)
    return float(res.statistic), float(res.pvalue), ""


def _cluster_ci(values: np.ndarray, n_boot: int, level: float, rng) -> tuple:
    """Bootstrap CI of the mean over participants (rows of `values` are resampled together)."""
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.nanmean(values[idx], axis=1)
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


def participant_tests(table: pd.DataFrame, cfg: dict, rng) -> pd.DataFrame:
    n_boot, level = cfg["stats"]["bootstrap_iterations"], cfg["stats"]["ci_level"]
    rows = []
    for metric in PAIRED_METRICS:
        wide = table.pivot(index="participant_id", columns="condition", values=metric)
        if A not in wide or B not in wide:
            continue
        # per-condition means over participants
        for cond in (A, B):
            col = wide[cond].dropna().to_numpy()
            lo, hi = _cluster_ci(col, n_boot, level, rng)
            rows.append({"analysis": "condition mean over participants", "metric": metric, "condition": cond,
                         "n": len(col), "estimate": col.mean() if len(col) else np.nan, "ci_low": lo, "ci_high": hi,
                         "statistic": np.nan, "p_value": np.nan, "note": ""})
        # paired difference A - B
        pair = wide[[A, B]].dropna()
        diff = (pair[A] - pair[B]).to_numpy()
        stat, p, note = _wilcoxon(diff)
        lo, hi = _cluster_ci(diff, n_boot, level, rng)
        rows.append({"analysis": f"paired difference {A} - {B}", "metric": metric, "condition": "",
                     "n": len(diff), "estimate": diff.mean() if len(diff) else np.nan, "ci_low": lo, "ci_high": hi,
                     "statistic": stat, "p_value": p, "note": note})
    # accuracy against chance (50%) in each condition
    wide = table.pivot(index="participant_id", columns="condition", values="accuracy_pct")
    for cond in (A, B):
        if cond not in wide:
            continue
        col = wide[cond].dropna().to_numpy()
        stat, p, note = _wilcoxon(col - 50.0)
        rows.append({"analysis": "accuracy vs 50% chance (Wilcoxon)", "metric": "accuracy_pct", "condition": cond,
                     "n": len(col), "estimate": col.mean() if len(col) else np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "statistic": stat, "p_value": p, "note": note})
    return pd.DataFrame(rows).round(4)


def participant_info(df: pd.DataFrame) -> pd.DataFrame:
    """Per participant: prior_exposure answer and the rank of the finishing time (1 = first to finish), when available."""
    g = df.groupby("participant_id")
    out = pd.DataFrame(index=g.size().index)
    if "prior_exposure" in df.columns:
        out["prior_exposure"] = g["prior_exposure"].first()
    if "submitted_at" in df.columns:
        when = pd.to_datetime(g["submitted_at"].min(), utc=True, errors="coerce")
        out["submit_rank"] = when.rank(method="first").astype("Int64")
    return out.reset_index()


def excluded_participants(df: pd.DataFrame, cfg: dict) -> set:
    values = set((cfg.get("participants") or {}).get("exclude_prior_exposure") or [])
    if not values or "prior_exposure" not in df.columns:
        return set()
    first = df.groupby("participant_id")["prior_exposure"].first()
    return set(first[first.isin(values)].index)


def leakage_checks(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Does accuracy drift with the order of finishing? (all participants, both conditions, valid trials)"""
    if "submitted_at" not in df.columns:
        return pd.DataFrame()
    min_n = (cfg.get("participants") or {}).get("min_participants_order_check", 8)
    acc = df[df["is_valid"]].groupby("participant_id")["is_correct"].mean() * 100
    when = pd.to_datetime(df.groupby("participant_id")["submitted_at"].min(), utc=True, errors="coerce")
    d = pd.DataFrame({"acc": acc, "when": when}).dropna().sort_values("when", kind="stable")
    n = len(d)
    base = {"metric": "accuracy_pct", "condition": "", "n": n, "ci_low": np.nan, "ci_high": np.nan}
    rows = []
    if n < min_n:
        note = f"fewer than {min_n} participants, no check"
        rows.append({**base, "analysis": "accuracy vs order of finishing (Spearman rho)", "estimate": np.nan,
                     "statistic": np.nan, "p_value": np.nan, "note": note})
        rows.append({**base, "analysis": "accuracy second half - first half (Mann-Whitney)", "estimate": np.nan,
                     "statistic": np.nan, "p_value": np.nan, "note": note})
        return pd.DataFrame(rows).round(4)
    if d["acc"].nunique() < 2:
        rho, p, note = np.nan, np.nan, "accuracy is identical for all participants"
    else:
        res = stats.spearmanr(np.arange(n), d["acc"].to_numpy())
        rho, p, note = float(res.statistic), float(res.pvalue), ""
    rows.append({**base, "analysis": "accuracy vs order of finishing (Spearman rho)", "estimate": rho,
                 "statistic": np.nan, "p_value": p, "note": note})
    first, second = d["acc"].iloc[: n // 2], d["acc"].iloc[n - n // 2:]
    mw = stats.mannwhitneyu(second, first, alternative="two-sided")
    rows.append({**base, "n": f"{len(first)}+{len(second)}", "analysis": "accuracy second half - first half (Mann-Whitney)",
                 "estimate": float(second.mean() - first.mean()), "statistic": float(mw.statistic),
                 "p_value": float(mw.pvalue), "note": ""})
    return pd.DataFrame(rows).round(4)


def run(cfg: dict = None, out_dir=None) -> tuple:
    cfg = cfg or load_config()
    df = load_human_trials(cfg)
    table = participant_table(df).merge(participant_info(df), on="participant_id", how="left")
    excluded = excluded_participants(df, cfg)
    table["in_primary"] = ~table["participant_id"].isin(excluded)
    seed = cfg["experiment"]["random_seed"]
    parts = [participant_tests(table[table["in_primary"]], cfg, np.random.default_rng(seed)).assign(sample="primary")]
    if excluded:
        parts.append(participant_tests(table, cfg, np.random.default_rng(seed)).assign(sample="sensitivity: all participants"))
    leak = leakage_checks(df, cfg)
    if len(leak):
        parts.append(leak.assign(sample="check: all participants"))
    tests = pd.concat(parts, ignore_index=True)
    tests = tests[["sample"] + [c for c in tests.columns if c != "sample"]]
    out = Path(out_dir) if out_dir else resolve(cfg["dataset"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "participant_table.csv", index=False)
    tests.to_csv(out / "participant_level_tests.csv", index=False)
    return table, tests, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output directory (default: dataset.processed_dir)")
    args = ap.parse_args()
    cfg = load_config()
    print(f"[PARTICIPANT LEVEL] reading {trials_path(cfg)}")
    table, tests, out = run(cfg, args.out)
    print(f"participants: {table['participant_id'].nunique()}  in primary analysis: {int(table['in_primary'].sum())}")
    print(tests.to_string(index=False))
    print(f"[EXPORT] participant_table.csv, participant_level_tests.csv -> {out}")
