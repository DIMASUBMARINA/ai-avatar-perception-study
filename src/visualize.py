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
    
    # Set Scientific Publication Theme (Rule 3)
    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("colorblind")
    
    # Load raw and aggregated data
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
    
    # Rule 1: Explicit labels with units & zero baseline
    ax.set_title("Response Latency Across Exposure Limits (Mean ± 95% CI)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Response Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(valid_df["response_time_ms"]) * 1.25)
    
    plt.tight_layout()
    plot1_pdf = os.path.join(output_dir, "latency_comparison_ci.pdf")
    plot1_png = os.path.join(output_dir, "latency_comparison_ci.png")
    plt.savefig(plot1_pdf, format="pdf")
    plt.savefig(plot1_png, dpi=300)
    plt.close()
    print(f"[EXPORT] Plot 1 saved to: {plot1_pdf} and {plot1_png}")
    
    # -------------------------------------------------------------------------
    # PLOT 2: Latency Distribution & Outlier Inspection (Boxplot + Stripplot)
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
    
    # Rule 1 & Rule 2: Distribution, outliers & physical units
    ax.set_title("Response Latency Distribution & Outliers", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Viewing Condition", fontsize=11, labelpad=8)
    ax.set_ylabel("Response Latency (ms)", fontsize=11, labelpad=8)
    ax.set_ylim(0, max(valid_df["response_time_ms"]) * 1.25)
    
    plt.tight_layout()
    plot2_pdf = os.path.join(output_dir, "latency_distribution_outliers.pdf")
    plot2_png = os.path.join(output_dir, "latency_distribution_outliers.png")
    plt.savefig(plot2_pdf, format="pdf")
    plt.savefig(plot2_png, dpi=300)
    plt.close()
    print(f"[EXPORT] Plot 2 saved to: {plot2_pdf} and {plot2_png}")

if __name__ == "__main__":
    generate_scientific_plots()