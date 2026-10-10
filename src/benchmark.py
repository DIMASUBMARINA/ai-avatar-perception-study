import yaml
import pandas as pd
import json
import os

def run_benchmark():
    print("=== Running Research Experiment Benchmark Pipeline ===")
    
    # Load configuration
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    raw_path = config["dataset"]["raw_logs_path"]
    print(f"[INFO] Loading raw experiment log sample from: {raw_path}")
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Sample data file not found at {raw_path}")
        
    df = pd.read_csv(raw_path)
    
    # Filter anomalies (guardrails)
    min_rt = config["guardrails"]["min_valid_rt_ms"]
    valid_df = df[df["response_time_ms"] >= min_rt].copy()
    
    # Calculate metrics per condition
    results = {}
    for condition, group in valid_df.groupby("condition"):
        tp = ((group["is_ai"] == 1) & (group["user_choice"] == 1)).sum()
        tn = ((group["is_ai"] == 0) & (group["user_choice"] == 0)).sum()
        fp = ((group["is_ai"] == 1) & (group["user_choice"] == 0)).sum()
        fn = ((group["is_ai"] == 0) & (group["user_choice"] == 1)).sum()
        
        total = len(group)
        acc = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        mean_rt = group["response_time_ms"].mean()
        
        results[condition] = {
            "sample_count": int(total),
            "accuracy": round(acc * 100, 2),
            "fpr_percent": round(fpr * 100, 2),
            "fnr_percent": round(fnr * 100, 2),
            "mean_latency_ms": round(mean_rt, 2)
        }
        
    print("
--- Benchmark Execution Results ---")
    print(json.dumps(results, indent=2))
    
    os.makedirs("data/processed", exist_ok=True)
    out_path = config["dataset"]["processed_output_path"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"
[SUCCESS] Benchmark pipeline completed successfully. Output written to {out_path}")

if __name__ == "__main__":
    run_benchmark()
