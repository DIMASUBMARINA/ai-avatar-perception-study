import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_scientific_plots(
    raw_path: str = "data/raw/test_run.jsonl",
    summary_path: str = "data/processed/metrics_summary.csv",
    output_dir: str = "reports/figures"
):
    print("[STEP 3 & 4] Generating publication-ready scientific plots...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set Scientific Publication Theme
    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("colorblind")
    
    # Load raw data
    df_raw = pd.read_json(raw_path, lines=True)
    valid_df = df_raw[df_raw["is_valid"] == True].copy()
    
    # -------------------------------------------------------------------------
    # PLOT 1: Response Latency Across Conditions (Bar Chart with 95% CI)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=valid_df,
        x="condition",
        y="response_time_ms",
        hue="condition",
        palette=palette,
        capsize=0.1,
        errorbar=("ci", 95),
        ax=ax
    )
    ax.set_title("Response Latency Across Exposure Limits (Mean ± 95% CI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Response Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(valid_df["response_time_ms"]) * 1.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_comparison_ci.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "latency_comparison_ci.png"), dpi=300)
    plt.close()
    
    # -------------------------------------------------------------------------
    # PLOT 2: Latency Distribution & Outlier Inspection
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=valid_df,
        x="condition",
        y="response_time_ms",
        hue="condition",
        palette=palette,
        boxprops=dict(alpha=0.7),
        ax=ax
    )
    sns.stripplot(
        data=valid_df,
        x="condition",
        y="response_time_ms",
        hue="condition",
        palette="dark:black",
        size=6,
        jitter=0.2,
        ax=ax
    )
    ax.set_title("Response Latency Distribution & Outliers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Response Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(valid_df["response_time_ms"]) * 1.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_distribution_outliers.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "latency_distribution_outliers.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3 (NEW): Detection Accuracy (Acc %) vs 50% Chance Level Baseline (Tests H1)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    acc_df = valid_df.groupby("condition")["is_correct"].mean().reset_index()
    acc_df["accuracy_pct"] = acc_df["is_correct"] * 100.0

    sns.barplot(
        data=acc_df,
        x="condition",
        y="accuracy_pct",
        hue="condition",
        palette=palette,
        ax=ax
    )
    ax.axhline(50.0, color='red', linestyle='--', linewidth=1.5, label='Chance Level Baseline (50%)')
    ax.set_title("Detection Accuracy Across Exposure Limits", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison_ci.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "accuracy_comparison_ci.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 4 (NEW): Error Asymmetry (FPR vs FNR) (Tests H2)
    # -------------------------------------------------------------------------
    records = []
    for condition, group in valid_df.groupby("condition"):
        tp = ((group["is_ai"] == 1) & (group["user_choice"] == 1)).sum()
        tn = ((group["is_ai"] == 0) & (group["user_choice"] == 0)).sum()
        fp = ((group["is_ai"] == 0) & (group["user_choice"] == 1)).sum()
        fn = ((group["is_ai"] == 1) & (group["user_choice"] == 0)).sum()
        
        fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
        fnr = (fn / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        
        records.append({"condition": condition, "error_type": "FPR (Hyper-skepticism)", "rate_pct": fpr})
        records.append({"condition": condition, "error_type": "FNR (Unwarranted Trust)", "rate_pct": fnr})

    err_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=err_df,
        x="condition",
        y="rate_pct",
        hue="error_type",
        palette="muted",
        ax=ax
    )
    ax.set_title("Cognitive Error Asymmetry (FPR vs. FNR)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Error Rate (%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_asymmetry_comparison.pdf"), format="pdf")
    plt.savefig(os.path.join(output_dir, "error_asymmetry_comparison.png"), dpi=300)
    plt.close()

    print(f"[EXPORT] Successfully generated 4 scientific plots in {output_dir}")

if __name__ == "__main__":
    generate_scientific_plots()


