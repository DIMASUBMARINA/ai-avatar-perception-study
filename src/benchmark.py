"""Collect raw latency measurements (warm-up + benchmark) into data/raw/test_run.jsonl.

The timed function is evaluate_batch, run on batches of sample trials at several load levels.
Only the call itself is timed; batch preparation and file I/O stay outside the timer.
"""
import json
import time

import numpy as np
import pandas as pd

from common import environment_metadata, is_valid_trial, load_config, resolve


def evaluate_batch(batch: list, min_rt: float, max_rt: float) -> tuple:
    """Return (n_invalid, n_correct) for a batch of (is_ai, user_choice, rt) trials."""
    n_invalid = 0
    n_correct = 0
    for is_ai, user_choice, rt in batch:
        if is_valid_trial(rt, min_rt, max_rt):
            if is_ai == user_choice:
                n_correct += 1
        else:
            n_invalid += 1
    return n_invalid, n_correct


def build_pool(config: dict) -> list:
    """Sample CSV trials plus synthetic edge cases from the config."""
    sample = pd.read_csv(resolve(config["dataset"]["sample_logs_path"]))
    pool = [
        (int(r.is_ai), int(r.user_choice), float(r.response_time_ms))
        for r in sample.itertuples(index=False)
    ]
    for case in config["benchmark"].get("edge_cases", []):
        pool.append((int(case["is_ai"]), int(case["user_choice"]), float(case["response_time_ms"])))
    return pool


def run_benchmark() -> list:
    config = load_config()
    seed = config["experiment"]["random_seed"]
    rng = np.random.default_rng(seed)
    bcfg = config["benchmark"]
    min_rt = config["guardrails"]["min_valid_rt_ms"]
    max_rt = config["guardrails"]["max_valid_rt_ms"]

    env = environment_metadata()
    print(f"[INFO] Environment: {env['os_name']} {env['os_release']} | Python {env['python_version']} | CPUs: {env['cpu_count']}")
    pool = build_pool(config)
    print(f"[INFO] Pool size: {len(pool)} trials | seed={seed}")

    records = []  # written to disk after all timing is done
    for load in bcfg["load_levels"]:
        for phase, n_runs in (("warmup", bcfg["warmup_runs"]), ("benchmark", bcfg["benchmark_runs"])):
            for run_id in range(n_runs):
                idx = rng.integers(0, len(pool), size=load)
                batch = [pool[i] for i in idx]  # prepared outside the timer

                runtime_error = False
                t0 = time.perf_counter_ns()
                try:
                    n_invalid, n_correct = evaluate_batch(batch, min_rt, max_rt)
                except Exception:  # record the failure instead of hiding it
                    n_invalid, n_correct, runtime_error = load, 0, True
                t1 = time.perf_counter_ns()

                latency_us = (t1 - t0) / 1000.0
                records.append({
                    "timestamp_ns": time.time_ns(),
                    "phase": phase,
                    "run_id": run_id,
                    "load_level": load,
                    "batch_latency_us": latency_us,
                    "latency_per_trial_us": latency_us / load,
                    "n_invalid": n_invalid,
                    "error_rate_pct": 100.0 * n_invalid / load,
                    "n_correct": n_correct,
                    "runtime_error": runtime_error,
                })

    out_jsonl = resolve(config["dataset"]["jsonl_output_path"])
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    meta = {
        "environment": env,
        "random_seed": seed,
        "warmup_runs_per_level": bcfg["warmup_runs"],
        "benchmark_runs_per_level": bcfg["benchmark_runs"],
        "load_levels": bcfg["load_levels"],
        "pool_size": len(pool),
        "synthetic_edge_cases": bcfg.get("edge_cases", []),
    }
    with open(resolve(config["dataset"]["metadata_path"]), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    n_warm = sum(r["phase"] == "warmup" for r in records)
    print(f"[SUCCESS] {len(records)} records ({n_warm} warmup + {len(records) - n_warm} benchmark) -> {out_jsonl}")

    print("\n--- Inline summary (benchmark phase, batch latency, us) ---")
    print(f"{'load':>6} {'p50':>9} {'p95':>9} {'p99':>9}")
    for load in bcfg["load_levels"]:
        y = [r["batch_latency_us"] for r in records if r["phase"] == "benchmark" and r["load_level"] == load]
        print(f"{load:>6} {np.percentile(y, 50):>9.3f} {np.percentile(y, 95):>9.3f} {np.percentile(y, 99):>9.3f}")
    return records


if __name__ == "__main__":
    run_benchmark()
