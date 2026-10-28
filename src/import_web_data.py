"""Validate the CSV exported by the web experiment and copy it to data/real/.

    python src/import_web_data.py                       # web/data/participant_trials.csv -> data/real/participant_trials.csv
    python src/import_web_data.py --src X.csv --dst Y.csv [--force]

The script does not change config.yaml: after a successful import set
dataset.human_trials_path to the destination file yourself.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from common import ROOT, load_config, max_rt_for, trial_flags

REQUIRED = ["participant_id", "condition", "stimulus_id", "is_ai", "user_choice", "response_time_ms"]
CONDITIONS = {"3s_limit", "unrestricted"}


def validate(df: pd.DataFrame) -> list:
    """Return a list of problems (empty list = file is usable)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    problems = []
    if df.empty:
        return ["file has no rows"]
    if df[REQUIRED].isna().any().any():
        problems.append("empty values in required columns")
    bad_cond = set(df["condition"].dropna().unique()) - CONDITIONS
    if bad_cond:
        problems.append(f"unknown condition values: {sorted(bad_cond)}")
    for col in ("is_ai", "user_choice"):
        if not set(df[col].dropna().unique()) <= {0, 1}:
            problems.append(f"{col} must contain only 0/1")
    if not pd.api.types.is_numeric_dtype(df["response_time_ms"]):
        problems.append("response_time_ms is not numeric")
    return problems


def summarize(df: pd.DataFrame, cfg: dict) -> None:
    g = cfg["guardrails"]
    print(f"participants: {df['participant_id'].nunique()}  trials: {len(df)}")
    both = df.groupby("participant_id")["condition"].nunique()
    if (both < 2).any():
        print(f"WARNING: {(both < 2).sum()} participant(s) have only one condition")
    dup = df.duplicated(["participant_id", "stimulus_id"]).sum()
    if dup:
        print(f"WARNING: {dup} repeated (participant, stimulus) pairs")
    flags = trial_flags(df, cfg)
    for cond, grp in df.groupby("condition"):
        f = flags.loc[grp.index]
        print(f"  {cond:<13} trials={len(grp):<5} excluded={100.0 * f.any(axis=1).mean():.1f}%  "
              f"[rt {100.0 * f['invalid_rt'].mean():.1f}% (limit {g['min_valid_rt_ms']}..{max_rt_for(cond, g)} ms), "
              f"focus {100.0 * f['invalid_focus'].mean():.1f}%, exposure {100.0 * f['invalid_exposure'].mean():.1f}%]")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "web" / "data" / "participant_trials.csv"))
    ap.add_argument("--dst", default=str(ROOT / "data" / "real" / "participant_trials.csv"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing destination file")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        print(f"ERROR: {src} not found")
        return 1
    df = pd.read_csv(src)
    problems = validate(df)
    if problems:
        print("ERROR: " + "; ".join(problems))
        return 1
    summarize(df, load_config())
    if dst.exists() and not args.force:
        print(f"ERROR: {dst} already exists (use --force to overwrite)")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[EXPORT] {dst}")
    print("Next: set dataset.human_trials_path in config.yaml to this file, then run process_data.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
