import json
import os
import platform
import time
import yaml
import pandas as pd
import numpy as np

def get_environment_metadata():
    """Collection of hardware and software environment metadata for 100% reproducibility."""
    return {
        "python_version": platform.python_version(),
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Generic CPU"
    }

def run_warmup(warmup_runs: int):
    """Warm-up phase to eliminate the cold-start effect."""
    print(f"[INFO] Executing Warm-up Phase ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        _ = np.exp(np.sin(np.linspace(0, 100, 1000)))
    print("[INFO] Warm-up complete. Discarding cold-start measurements.\n")

def run_clean_benchmark():
    print("=== Week 7: Clean Data Collection & Benchmark Pipeline ===")
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    env_meta = get_environment_metadata()
    print(f"[INFO] Environment: {env_meta['os_name']} {env_meta['os_release']} | Python {env_meta['python_version']}")
    
    warmup_runs = config["experiment"].get("warmup_runs", 5)
    run_warmup(warmup_runs)
    
    raw_path = config["dataset"]["raw_logs_path"]
    jsonl_output = config["dataset"].get("jsonl_output_path", "data/raw/test_run.jsonl")
    min_rt = config["guardrails"]["min_valid_rt_ms"]
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Sample data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    os.makedirs(os.path.dirname(jsonl_output), exist_ok=True)
    
    jsonl_entries = []
    
    print(f"[INFO] Processing {len(df)} entries with I/O isolation...")
    for _, row in df.iterrows():
        t_start_ns = time.perf_counter_ns()
        
        is_valid = bool(row["response_time_ms"] >= min_rt)
        is_correct = bool((row["is_ai"] == row["user_choice"])) if is_valid else False
        
        t_end_ns = time.perf_counter_ns()
        eval_overhead_us = (t_end_ns - t_start_ns) / 1000.0
        
        entry = {
            "timestamp_ns": time.time_ns(),
            "participant_id": str(row["participant_id"]),
            "condition": str(row["condition"]),
            "stimulus_id": str(row["stimulus_id"]),
            "is_ai": int(row["is_ai"]),
            "user_choice": int(row["user_choice"]),
            "response_time_ms": float(row["response_time_ms"]),
            "is_valid": is_valid,
            "is_correct": is_correct,
            "eval_overhead_us": eval_overhead_us
        }
        jsonl_entries.append(entry)

    with open(jsonl_output, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"[SUCCESS] Saved {len(jsonl_entries)} clean records to JSONL: {jsonl_output}")
    
    valid_entries = [e for e in jsonl_entries if e["is_valid"]]
    valid_df = pd.DataFrame(valid_entries)
    
    summary = {}
    for condition, group in valid_df.groupby("condition"):
        rts = group["response_time_ms"].values
        tp = ((group["is_ai"] == 1) & (group["user_choice"] == 1)).sum()
        tn = ((group["is_ai"] == 0) & (group["user_choice"] == 0)).sum()
        fp = ((group["is_ai"] == 0) & (group["user_choice"] == 1)).sum()
        fn = ((group["is_ai"] == 1) & (group["user_choice"] == 0)).sum()
        
        total = len(group)
        acc = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        summary[condition] = {
            "sample_count": int(total),
            "accuracy_percent": round(float(acc * 100), 2),
            "fpr_percent": round(float(fpr * 100), 2),
            "fnr_percent": round(float(fnr * 100), 2),
            "latency_metrics_ms": {
                "mean": round(float(np.mean(rts)), 2),
                "median_p50": round(float(np.median(rts)), 2),
                "p95": round(float(np.percentile(rts, 95)), 2),
                "p99": round(float(np.percentile(rts, 99)), 2)
            }
        }
        
    final_output = {
        "metadata": env_meta,
        "summary": summary
    }
    
    print("\n--- Summary Results (Percentiles & Accuracy) ---")
    print(json.dumps(final_output, indent=2))
    
    out_summary = config["dataset"]["processed_output_path"]
    os.makedirs(os.path.dirname(out_summary), exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\n[SUCCESS] Pipeline execution finished. Summary saved to: {out_summary}")

if __name__ == "__main__":
    run_clean_benchmark()