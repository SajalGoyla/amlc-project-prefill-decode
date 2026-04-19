"""
plot_results.py — Generate publication-quality benchmark plots.

Creates:
1. Sweep crossover plot (TTFT vs input tokens, both architectures)
2. Bar chart comparing profiles (chat, doc_qa, balanced)
3. ITL distribution (box plots)
4. Throughput comparison

Usage:
    python -m analysis.plot_results --results-dir benchmark/results/ --out analysis/figures/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "collocated": "#4F46E5",     # Indigo
    "disaggregated": "#EC4899",  # Pink
}
MARKERS = {"collocated": "o", "disaggregated": "s"}


def load_results(results_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        sys.exit(1)
    dfs = [pd.read_csv(f) for f in csv_files]
    return pd.concat(dfs, ignore_index=True)


def plot_sweep_crossover(df: pd.DataFrame, out_dir: str, metric: str = "ttft_ms"):
    """Plot the sweep crossover: TTFT vs input tokens for both architectures."""
    sweep_df = df[df["profile"].str.startswith("sweep_")].copy()
    if sweep_df.empty:
        print("  No sweep data — skipping crossover plot")
        return

    sweep_df["input_tokens_target"] = sweep_df["profile"].str.extract(r"sweep_(\d+)").astype(int)
    sweep_df = sweep_df[sweep_df["error"].fillna("") == ""]

    agg = sweep_df.groupby(["arch", "input_tokens_target"]).agg(
        mean=(metric, "mean"),
        std=(metric, "std"),
        p50=(metric, "median"),
    ).reset_index()

    fig, ax = plt.subplots()

    for arch in ["collocated", "disaggregated"]:
        data = agg[agg["arch"] == arch].sort_values("input_tokens_target")
        if data.empty:
            continue
        ax.errorbar(
            data["input_tokens_target"],
            data["mean"],
            yerr=data["std"],
            label=arch.capitalize(),
            color=COLORS.get(arch, "#888"),
            marker=MARKERS.get(arch, "o"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Input Prompt Length (tokens)")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (ms)")
    ax.set_title("Collocated vs Disaggregated: Prompt-Length Sweep")
    ax.legend()
    ax.set_xscale("log")

    # Add crossover annotation if found
    merged = pd.merge(
        agg[agg["arch"] == "collocated"][["input_tokens_target", "mean"]],
        agg[agg["arch"] == "disaggregated"][["input_tokens_target", "mean"]],
        on="input_tokens_target",
        suffixes=("_colo", "_disagg"),
    )
    merged["delta"] = merged["mean_disagg"] - merged["mean_colo"]
    for i in range(1, len(merged)):
        if merged.iloc[i - 1]["delta"] > 0 and merged.iloc[i]["delta"] <= 0:
            x1 = merged.iloc[i - 1]["input_tokens_target"]
            x2 = merged.iloc[i]["input_tokens_target"]
            y1 = merged.iloc[i - 1]["delta"]
            y2 = merged.iloc[i]["delta"]
            crossover = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            ax.axvline(x=crossover, color="#EF4444", linestyle="--", alpha=0.7, label=f"Crossover ≈ {int(crossover)} tokens")
            ax.legend()
            break

    plt.tight_layout()
    filepath = os.path.join(out_dir, "sweep_crossover.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_profile_comparison(df: pd.DataFrame, out_dir: str):
    """Bar chart comparing profiles across architectures."""
    profiles_df = df[~df["profile"].str.startswith("sweep_")].copy()
    profiles_df = profiles_df[profiles_df["error"].fillna("") == ""]

    if profiles_df.empty:
        print("  No profile data — skipping comparison plot")
        return

    metrics = ["ttft_ms", "total_latency_ms", "throughput_tps"]
    titles = ["Time to First Token (ms)", "Total Latency (ms)", "Throughput (tokens/s)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, metric, title in zip(axes, metrics, titles):
        agg = profiles_df.groupby(["arch", "profile"]).agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
        ).reset_index()

        profiles = sorted(agg["profile"].unique())
        x = np.arange(len(profiles))
        width = 0.35

        for i, arch in enumerate(["collocated", "disaggregated"]):
            data = agg[agg["arch"] == arch]
            if data.empty:
                continue
            vals = [data[data["profile"] == p]["mean"].values[0] if len(data[data["profile"] == p]) > 0 else 0 for p in profiles]
            stds = [data[data["profile"] == p]["std"].values[0] if len(data[data["profile"] == p]) > 0 else 0 for p in profiles]
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                yerr=stds,
                label=arch.capitalize(),
                color=COLORS.get(arch, "#888"),
                alpha=0.85,
                capsize=3,
            )

        ax.set_xlabel("Profile")
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in profiles], fontsize=10)
        ax.legend()

    fig.suptitle("Profile Comparison: Collocated vs Disaggregated", fontsize=16, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(out_dir, "profile_comparison.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_itl_distribution(df: pd.DataFrame, out_dir: str):
    """Box plot of ITL (inter-token latency) by arch and profile."""
    profiles_df = df[~df["profile"].str.startswith("sweep_")].copy()
    profiles_df = profiles_df[profiles_df["error"].fillna("") == ""]
    profiles_df = profiles_df[profiles_df["itl_ms_mean"] > 0]

    if profiles_df.empty:
        print("  No ITL data — skipping distribution plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    profiles = sorted(profiles_df["profile"].unique())
    archs = ["collocated", "disaggregated"]
    positions = []
    labels = []
    data = []
    colors = []

    for i, profile in enumerate(profiles):
        for j, arch in enumerate(archs):
            subset = profiles_df[(profiles_df["profile"] == profile) & (profiles_df["arch"] == arch)]
            if not subset.empty:
                data.append(subset["itl_ms_mean"].values)
                positions.append(i * 2.5 + j * 0.7)
                labels.append(f"{profile}\n{arch[:5]}")
                colors.append(COLORS.get(arch, "#888"))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Mean ITL (ms)")
    ax.set_title("Inter-Token Latency Distribution")

    plt.tight_layout()
    filepath = os.path.join(out_dir, "itl_distribution.png")
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--results-dir", default="benchmark/results")
    parser.add_argument("--out", default="analysis/figures")
    parser.add_argument("--metric", default="ttft_ms")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_results(args.results_dir)

    print(f"\n{'='*60}")
    print(f"  Generating Plots")
    print(f"  {len(df)} data points")
    print(f"{'='*60}\n")

    plot_sweep_crossover(df, args.out, args.metric)
    plot_profile_comparison(df, args.out)
    plot_itl_distribution(df, args.out)

    print(f"\nAll plots saved to {args.out}/")


if __name__ == "__main__":
    main()
