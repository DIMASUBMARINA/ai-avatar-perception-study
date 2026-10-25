"""Publication-ready figures built from data/processed/*.csv and the sample trials.

Rules: labelled axes with units (log scale for wide ranges), 95% CIs on every plot,
colorblind palette with distinct markers/line styles/hatches. Export: .pdf (vector) and .png (300 DPI).
"""
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from common import load_config, resolve
from process_data import load_human_trials

PALETTE = sns.color_palette("colorblind")
HATCHES = ["//", "xx", "..", "\\\\"]
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", ":", "-."]


def _setup():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({"pdf.fonttype": 42, "axes.titleweight": "bold"})


def _save(fig, name: str, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", format="pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)
    return [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]


def fig1_latency_vs_load(lat: pd.DataFrame, out_dir):
    """Latency percentiles vs load; shaded bands are bootstrap 95% CIs."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (q, label) in enumerate((("p50", "Median (p50)"), ("p95", "p95"), ("p99", "p99"))):
        ax.plot(lat["load_level"], lat[f"{q}_us"], color=PALETTE[i], linestyle=LINESTYLES[i],
                marker=MARKERS[i], markersize=7, linewidth=2, label=label)
        ax.fill_between(lat["load_level"], lat[f"{q}_ci95_low_us"], lat[f"{q}_ci95_high_us"],
                        color=PALETTE[i], alpha=0.25, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")  # values span orders of magnitude
    ax.set_xticks(lat["load_level"])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # 1-2-5 ticks per decade, covering the p99 tails
    lo, hi = lat["p50_ci95_low_us"].min(), lat["p99_ci95_high_us"].max()
    decades = range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1)
    ticks = [m * 10.0 ** d for d in decades for m in (1, 2, 5) if lo * 0.8 <= m * 10.0 ** d <= hi * 1.25]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(lo * 0.8, hi * 1.25)
    ax.set_xlabel("Load (trials per batch)")
    ax.set_ylabel("Batch evaluation latency (µs)")
    ax.set_title("Latency vs. Load (median, p95, p99; shaded = 95% bootstrap CI)", fontsize=11)
    ax.legend(title="Percentile", loc="lower right")
    return _save(fig, "fig1_latency_vs_load", out_dir)


def fig2_distribution_boxplot(trials: pd.DataFrame, metrics: pd.DataFrame, out_dir):
    """Response-time distribution per condition with mean and 95% CI."""
    v = trials[trials["is_valid"]]
    conds = list(metrics["condition"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=v, x="condition", y="response_time_ms", order=conds, hue="condition",
                palette=PALETTE[:len(conds)], boxprops=dict(alpha=0.6), showfliers=False, legend=False, ax=ax)
    for i, c in enumerate(conds):
        y = v.loc[v["condition"] == c, "response_time_ms"]
        ax.scatter(np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y)), y, marker=MARKERS[i],
                   color="black", s=36, zorder=3, label=None)
        m = metrics.loc[metrics["condition"] == c].iloc[0]
        ax.errorbar(i + 0.32, m["mean_rt_ms"],
                    yerr=[[m["mean_rt_ms"] - m["mean_rt_ms_ci95_low"]], [m["mean_rt_ms_ci95_high"] - m["mean_rt_ms"]]],
                    fmt="D", color="black", markerfacecolor="white", capsize=5, zorder=4,
                    label="Mean ± 95% bootstrap CI" if i == 0 else None)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{c}\n(n={int(n)})" for c, n in zip(conds, metrics["sample_size"])])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Viewing condition")
    ax.set_ylabel("Response time (ms)")
    ax.set_title("Response Time Distribution by Condition (pilot sample)", fontsize=11)
    ax.legend(loc="upper left")
    return _save(fig, "fig2_distribution_boxplot", out_dir)


def _bars(ax, x, vals, lo, hi, color, hatch, width=0.8, label=None):
    ax.bar(x, vals, width=width, color=color, edgecolor="black", hatch=hatch, label=label)
    ax.errorbar(x, vals, yerr=[np.asarray(vals) - np.asarray(lo), np.asarray(hi) - np.asarray(vals)],
                fmt="none", ecolor="black", capsize=5, linewidth=1.5)


def fig3_accuracy(metrics: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metrics))
    for i, m in metrics.iterrows():
        _bars(ax, x[i], [m["accuracy_pct"]], [m["accuracy_pct_ci95_low"]], [m["accuracy_pct_ci95_high"]],
              PALETTE[i], HATCHES[i], width=0.55)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.5, label="Chance level (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int(n)})" for c, n in zip(metrics["condition"], metrics["sample_size"])])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Viewing condition")
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_title("Detection Accuracy (mean ± 95% bootstrap CI)", fontsize=11)
    ax.legend(loc="lower right")
    return _save(fig, "fig3_accuracy_comparison", out_dir)


def fig4_error_asymmetry(metrics: pd.DataFrame, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(metrics))
    w = 0.36
    for k, (col, label) in enumerate((("fpr_pct", "FPR (real judged as AI)"), ("fnr_pct", "FNR (AI judged as real)"))):
        _bars(ax, x + (k - 0.5) * w, metrics[col], metrics[f"{col}_ci95_low"], metrics[f"{col}_ci95_high"],
              PALETTE[k], HATCHES[k], width=w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={int(n)})" for c, n in zip(metrics["condition"], metrics["sample_size"])])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Viewing condition")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Error Asymmetry FPR vs. FNR (± 95% bootstrap CI)", fontsize=11)
    ax.legend(loc="upper right")
    return _save(fig, "fig4_error_asymmetry", out_dir)


def generate_scientific_plots() -> list:
    cfg = load_config()
    _setup()
    proc = resolve(cfg["dataset"]["processed_dir"])
    out_dir = resolve(cfg["dataset"]["figures_dir"])
    lat = pd.read_csv(proc / "latency_by_load.csv")
    metrics = pd.read_csv(proc / "metrics_summary.csv")
    trials = load_human_trials(cfg)

    files = []
    files += fig1_latency_vs_load(lat, out_dir)
    files += fig2_distribution_boxplot(trials, metrics, out_dir)
    files += fig3_accuracy(metrics, out_dir)
    files += fig4_error_asymmetry(metrics, out_dir)
    print(f"[EXPORT] {len(files)} files (.pdf + .png @300 DPI) -> {out_dir}")
    return files


if __name__ == "__main__":
    generate_scientific_plots()
