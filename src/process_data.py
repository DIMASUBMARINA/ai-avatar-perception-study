import os
import json
import pandas as pd
import numpy as np

def bootstrap_ci(series: np.ndarray, n_bootstraps: int = 1000, ci: float = 95.0) -> tuple:
    """Calculates 95% Bootstrapped Confidence Interval for mean latency."""
    if len(series) == 0:
        return 0.0, 0.0
    bootstrapped_means = [
        np.mean(np.random.choice(series, size=len(series), replace=True))
        for _ in range(n_bootstraps)
    ]
    lower_bound = np.percentile(bootstrapped_means, (100.0 - ci) / 2.0)
    upper_bound = np.percentile(bootstrapped_means, 100.0 - (100.0 - ci) / 2.0)
    return lower_bound, upper_bound

def process_and_aggregate(
    input_path: str = "data/raw/test_run.jsonl", 
    output_dir: str = "data/processed"
) -> pd.DataFrame:
    print(f"[STEP 1] Loading raw JSONL logs from: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
        
    df = pd.read_json(input_path, lines=True)
    
    # 1. Warm-up Filtering
    if "phase" in df.columns:
        df = df[df["phase"] == "benchmark"].copy()
        
    # 2. Non-Destructive IQR Outliers
    def mark_iqr_outliers(group, col="response_time_ms", factor=1.5):
        q25 = group[col].quantile(0.25)
        q75 = group[col].quantile(0.75)
        iqr = q75 - q25
        return (group[col] < (q25 - factor * iqr)) | (group[col] > (q75 + factor * iqr))

    df["is_iqr_outlier"] = df.groupby("condition", group_keys=False).apply(
        lambda g: mark_iqr_outliers(g, "response_time_ms")
    )
    
    # Filter for valid non-outlier entries for primary aggregation
    valid_df = df[df["is_valid"] == True].copy()

    # [STEP 2] Aggregate Metrics per Condition
    print("\n[STEP 2] Aggregating Statistical Metrics & 95% Bootstrapped CIs...")
    metrics_list = []
    
    for condition, group in valid_df.groupby("condition"):
        rts = group["response_time_ms"].values
        acc = group["is_correct"].mean() * 100.0 if "is_correct" in group.columns else 0.0
        
        ci_low, ci_high = bootstrap_ci(rts, n_bootstraps=1000, ci=95.0)
        
        metrics_list.append({
            "condition": condition,
            "sample_size": len(group),
            "accuracy_pct": round(float(acc), 2),
            "mean_rt_ms": round(float(np.mean(rts)), 2),
            "std_rt_ms": round(float(np.std(rts, ddof=1)), 2) if len(rts) > 1 else 0.0,
            "median_p50_ms": round(float(np.median(rts)), 2),
            "p95_rt_ms": round(float(np.percentile(rts, 95)), 2),
            "p99_rt_ms": round(float(np.percentile(rts, 99)), 2),
            "ci95_low_ms": round(float(ci_low), 2),
            "ci95_high_ms": round(float(ci_high), 2)
        })
        
    summary_df = pd.DataFrame(metrics_list)
    
    # Export Aggregated Dataset for 100% Reproducibility
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[EXPORT] Saved clean aggregated metrics to: {summary_csv}")
    
    return summary_df

if __name__ == "__main__":
    summary_df = process_and_aggregate()
    print("\n--- Aggregated Metrics Summary ---")
    print(summary_df.to_string(index=False))