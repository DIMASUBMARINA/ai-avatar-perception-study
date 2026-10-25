import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm

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

def calculate_d_prime(hit_rate: float, fa_rate: float, n_signal: int, n_noise: int) -> float:
    """Calculates Signal Detection Theory Sensitivity Index (d') with Hautus loglinear correction."""
    adjusted_hr = np.clip(hit_rate, 0.5 / max(1, n_signal), 1.0 - 0.5 / max(1, n_signal))
    adjusted_far = np.clip(fa_rate, 0.5 / max(1, n_noise), 1.0 - 0.5 / max(1, n_noise))
    return float(norm.ppf(adjusted_hr) - norm.ppf(adjusted_far))

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
        
    # 2. IQR Outliers marking
    def mark_iqr_outliers(group, col="response_time_ms", factor=1.5):
        q25 = group[col].quantile(0.25)
        q75 = group[col].quantile(0.75)
        iqr = q75 - q25
        return (group[col] < (q25 - factor * iqr)) | (group[col] > (q75 + factor * iqr))

    df["is_iqr_outlier"] = df.groupby("condition", group_keys=False).apply(
        lambda g: mark_iqr_outliers(g, "response_time_ms")
    )

    # [STEP 2] Aggregate Metrics per Condition
    print("\n[STEP 2] Aggregating Statistical Metrics, SDT Index (d'), and 95% CIs...")
    metrics_list = []
    
    for condition, group in df.groupby("condition"):
        total_trials = len(group)
        invalid_trials = len(group[group["is_valid"] == False])
        tir_pct = (invalid_trials / total_trials * 100.0) if total_trials > 0 else 0.0
        
        valid_group = group[group["is_valid"] == True]
        rts = valid_group["response_time_ms"].values
        
        tp = ((valid_group["is_ai"] == 1) & (valid_group["user_choice"] == 1)).sum()
        tn = ((valid_group["is_ai"] == 0) & (valid_group["user_choice"] == 0)).sum()
        fp = ((valid_group["is_ai"] == 0) & (valid_group["user_choice"] == 1)).sum()
        fn = ((valid_group["is_ai"] == 1) & (valid_group["user_choice"] == 0)).sum()
        
        n_valid = len(valid_group)
        acc = (tp + tn) / n_valid * 100.0 if n_valid > 0 else 0.0
        fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
        fnr = (fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        
        hit_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        d_prime = calculate_d_prime(hit_rate, fa_rate, n_signal=tp+fn, n_noise=fp+tn)
        
        ci_low, ci_high = bootstrap_ci(rts, n_bootstraps=1000, ci=95.0)
        
        metrics_list.append({
            "condition": condition,
            "sample_size": n_valid,
            "tir_pct": round(float(tir_pct), 2),
            "accuracy_pct": round(float(acc), 2),
            "fpr_pct": round(float(fpr), 2),
            "fnr_pct": round(float(fnr), 2),
            "d_prime": round(float(d_prime), 3),
            "mean_rt_ms": round(float(np.mean(rts)), 2),
            "std_rt_ms": round(float(np.std(rts, ddof=1)), 2) if len(rts) > 1 else 0.0,
            "median_p50_ms": round(float(np.median(rts)), 2),
            "p95_rt_ms": round(float(np.percentile(rts, 95)), 2),
            "p99_rt_ms": round(float(np.percentile(rts, 99)), 2),
            "ci95_low_ms": round(float(ci_low), 2),
            "ci95_high_ms": round(float(ci_high), 2)
        })
        
    summary_df = pd.DataFrame(metrics_list)
    
    # Export Aggregated Dataset
    os.makedirs(output_dir, exist_ok=True)
    summary_csv = os.path.join(output_dir, "metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[EXPORT] Saved clean aggregated metrics (with SDT d' & FPR/FNR) to: {summary_csv}")
    
    return summary_df

if __name__ == "__main__":
    summary_df = process_and_aggregate()
    print("\n--- Aggregated Metrics Summary ---")
    print(summary_df.to_string(index=False))

