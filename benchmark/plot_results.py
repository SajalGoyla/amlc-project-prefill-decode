"""
Plot Benchmark Results
======================
Reads CSV outputs from prompt_sweep.py and concurrency_sweep.py,
then generates publication-quality figures for the report.

Produces:
  1. Crossover Curve   — Compute TTFT vs Prompt Length (both architectures)
  2. Network Overhead  — KV transfer + reconstruction vs Prompt Length
  3. TPOT Stability    — Average TPOT vs Prompt Length
  4. Concurrency Scaling — Throughput vs Concurrency Level
  5. Concurrency TPOT   — Average TPOT vs Concurrency Level
  6. Cost Per Token     — Collocated vs Disaggregated

Usage:
    python benchmark/plot_results.py \
        --collocated-sweep  benchmark_results/sweep_collocated.csv \
        --disaggregated-sweep benchmark_results/sweep_disaggregated.csv \
        --collocated-conc   benchmark_results/concurrency_collocated.csv \
        --disaggregated-conc benchmark_results/concurrency_disaggregated.csv \
        --outdir benchmark_results/figures
"""

import os
import csv
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")       # headless — works on VMs without display
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib")
    raise

# ==============================================================================
# Styling
# ==============================================================================

plt.rcParams.update({
    "figure.figsize":     (9, 5.5),
    "figure.dpi":         200,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "legend.fontsize":    10,
    "lines.linewidth":    2.2,
    "lines.markersize":   7,
})

COLOR_COLLOCATED    = "#2563eb"   # blue
COLOR_DISAGGREGATED = "#dc2626"   # red
COLOR_NETWORK       = "#f59e0b"   # amber
COLOR_RECON         = "#8b5cf6"   # violet

# GCP on-demand pricing (g2-standard-4, 1x L4)
L4_COST_PER_HOUR = 0.81


# ==============================================================================
# Helpers
# ==============================================================================

def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_by(rows, key_col, val_col):
    """Group rows by key_col and aggregate val_col values (median ± std)."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[float(r[key_col])].append(float(r[val_col]))
    keys = sorted(buckets.keys())
    medians = [np.median(buckets[k]) for k in keys]
    stds    = [np.std(buckets[k]) for k in keys]
    return np.array(keys), np.array(medians), np.array(stds)


# ==============================================================================
# Figure 1: Crossover Curve (Compute TTFT vs Prompt Length)
# ==============================================================================

def plot_crossover(coloc_csv, disag_csv, outdir):
    coloc = read_csv(coloc_csv)
    disag = read_csv(disag_csv)

    c_x, c_y, c_err = group_by(coloc, "target_tokens", "compute_ttft_ms")
    d_x, d_y, d_err = group_by(disag, "target_tokens", "compute_ttft_ms")

    fig, ax = plt.subplots()
    ax.errorbar(c_x, c_y, yerr=c_err, fmt="-o", color=COLOR_COLLOCATED,
                label="Collocated (single GPU)", capsize=3)
    ax.errorbar(d_x, d_y, yerr=d_err, fmt="-s", color=COLOR_DISAGGREGATED,
                label="Disaggregated (compute only)", capsize=3)

    # Find approximate crossover
    common = sorted(set(c_x) & set(d_x))
    for i in range(len(common) - 1):
        idx_c1 = list(c_x).index(common[i])
        idx_c2 = list(c_x).index(common[i + 1])
        idx_d1 = list(d_x).index(common[i])
        idx_d2 = list(d_x).index(common[i + 1])
        # Check sign change in (collocated - disaggregated)
        diff1 = c_y[idx_c1] - d_y[idx_d1]
        diff2 = c_y[idx_c2] - d_y[idx_d2]
        if diff1 * diff2 < 0:
            # Linear interpolation for crossover
            frac = abs(diff1) / (abs(diff1) + abs(diff2))
            cross_n = common[i] + frac * (common[i + 1] - common[i])
            ax.axvline(x=cross_n, color="green", linestyle="--", alpha=0.7,
                       label=f"Crossover N ≈ {int(cross_n)} tokens")
            break

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Compute TTFT (ms)")
    ax.set_title("Compute TTFT vs Prompt Length — Crossover Curve")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(outdir, "crossover_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Figure 2: Network Overhead (KV transfer + reconstruction)
# ==============================================================================

def plot_network_overhead(disag_csv, outdir):
    disag = read_csv(disag_csv)

    x_kv, y_kv, e_kv   = group_by(disag, "target_tokens", "kv_transfer_ms")
    x_rc, y_rc, e_rc    = group_by(disag, "target_tokens", "cache_recon_ms")

    fig, ax = plt.subplots()
    ax.bar(x_kv - 15, y_kv, width=30, color=COLOR_NETWORK,
           label="KV Network Transfer", alpha=0.85, yerr=e_kv, capsize=3)
    ax.bar(x_rc + 15, y_rc, width=30, color=COLOR_RECON,
           label="Cache Reconstruction", alpha=0.85, yerr=e_rc, capsize=3)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Disaggregated Overhead — Network + Reconstruction vs Prompt Length")
    ax.legend()

    path = os.path.join(outdir, "network_overhead.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Figure 3: TPOT Stability vs Prompt Length
# ==============================================================================

def plot_tpot_vs_length(coloc_csv, disag_csv, outdir):
    coloc = read_csv(coloc_csv)
    disag = read_csv(disag_csv)

    c_x, c_y, c_e = group_by(coloc, "target_tokens", "tpot_ms")
    d_x, d_y, d_e = group_by(disag, "target_tokens", "tpot_ms")

    fig, ax = plt.subplots()
    ax.errorbar(c_x, c_y, yerr=c_e, fmt="-o", color=COLOR_COLLOCATED,
                label="Collocated", capsize=3)
    ax.errorbar(d_x, d_y, yerr=d_e, fmt="-s", color=COLOR_DISAGGREGATED,
                label="Disaggregated", capsize=3)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("TPOT (ms/token)")
    ax.set_title("TPOT Stability vs Prompt Length")
    ax.legend()

    path = os.path.join(outdir, "tpot_vs_length.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Figure 4: Throughput vs Concurrency
# ==============================================================================

def plot_concurrency_throughput(coloc_csv, disag_csv, outdir):
    coloc = read_csv(coloc_csv)
    disag = read_csv(disag_csv)

    c_x, c_y, c_e = group_by(coloc, "concurrency", "throughput_tps")
    d_x, d_y, d_e = group_by(disag, "concurrency", "throughput_tps")

    fig, ax = plt.subplots()
    ax.errorbar(c_x, c_y, yerr=c_e, fmt="-o", color=COLOR_COLLOCATED,
                label="Collocated", capsize=3)
    ax.errorbar(d_x, d_y, yerr=d_e, fmt="-s", color=COLOR_DISAGGREGATED,
                label="Disaggregated", capsize=3)

    # Find crossover in throughput
    common = sorted(set(c_x) & set(d_x))
    for i in range(len(common) - 1):
        idx_c1 = list(c_x).index(common[i])
        idx_c2 = list(c_x).index(common[i + 1])
        idx_d1 = list(d_x).index(common[i])
        idx_d2 = list(d_x).index(common[i + 1])
        diff1 = d_y[idx_d1] - c_y[idx_c1]
        diff2 = d_y[idx_d2] - c_y[idx_c2]
        if diff1 * diff2 < 0:
            frac = abs(diff1) / (abs(diff1) + abs(diff2))
            cross_c = common[i] + frac * (common[i + 1] - common[i])
            ax.axvline(x=cross_c, color="green", linestyle="--", alpha=0.7,
                       label=f"Crossover ≈ {cross_c:.0f} concurrent")
            break

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Concurrency Level")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(outdir, "concurrency_throughput.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Figure 5: TPOT vs Concurrency
# ==============================================================================

def plot_concurrency_tpot(coloc_csv, disag_csv, outdir):
    coloc = read_csv(coloc_csv)
    disag = read_csv(disag_csv)

    c_x, c_y, c_e = group_by(coloc, "concurrency", "avg_tpot_ms")
    d_x, d_y, d_e = group_by(disag, "concurrency", "avg_tpot_ms")

    fig, ax = plt.subplots()
    ax.errorbar(c_x, c_y, yerr=c_e, fmt="-o", color=COLOR_COLLOCATED,
                label="Collocated", capsize=3)
    ax.errorbar(d_x, d_y, yerr=d_e, fmt="-s", color=COLOR_DISAGGREGATED,
                label="Disaggregated", capsize=3)

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Average TPOT (ms/token)")
    ax.set_title("TPOT Stability Under Concurrent Load")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    path = os.path.join(outdir, "concurrency_tpot.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Figure 6: Cost Per Token
# ==============================================================================

def plot_cost(coloc_csv, disag_csv, outdir):
    coloc = read_csv(coloc_csv)
    disag = read_csv(disag_csv)

    c_x, c_e2e, _ = group_by(coloc, "target_tokens", "e2e_ms")
    d_x, d_e2e, _ = group_by(disag, "target_tokens", "e2e_ms")
    _, c_tok, _    = group_by(coloc, "target_tokens", "tokens")
    _, d_tok, _    = group_by(disag, "target_tokens", "tokens")

    # cost = (e2e_seconds * $/sec) / tokens  (in micro-dollars for readability)
    c_cost = (c_e2e / 1000) * (L4_COST_PER_HOUR / 3600) / np.maximum(c_tok, 1) * 1e6
    d_cost = (d_e2e / 1000) * (2 * L4_COST_PER_HOUR / 3600) / np.maximum(d_tok, 1) * 1e6

    fig, ax = plt.subplots()
    ax.plot(c_x, c_cost, "-o", color=COLOR_COLLOCATED, label="Collocated (1 GPU)")
    ax.plot(d_x, d_cost, "-s", color=COLOR_DISAGGREGATED, label="Disaggregated (2 GPUs)")

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Cost per Token (μ$)")
    ax.set_title("Cost per Output Token — GCP On-Demand Pricing")
    ax.legend()

    path = os.path.join(outdir, "cost_per_token.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description="Generate report figures from benchmark CSVs.")
    p.add_argument("--collocated-sweep",   default=None, help="Collocated prompt-sweep CSV")
    p.add_argument("--disaggregated-sweep", default=None, help="Disaggregated prompt-sweep CSV")
    p.add_argument("--collocated-conc",    default=None, help="Collocated concurrency CSV")
    p.add_argument("--disaggregated-conc", default=None, help="Disaggregated concurrency CSV")
    p.add_argument("--outdir", default="benchmark_results/figures")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 50)
    print("Generating Report Figures")
    print("=" * 50)

    # Prompt-sweep figures (need both CSVs)
    if args.collocated_sweep and args.disaggregated_sweep:
        plot_crossover(args.collocated_sweep, args.disaggregated_sweep, args.outdir)
        plot_tpot_vs_length(args.collocated_sweep, args.disaggregated_sweep, args.outdir)
        plot_cost(args.collocated_sweep, args.disaggregated_sweep, args.outdir)
    else:
        print("  ⚠ Skipping prompt-sweep figures (need both --collocated-sweep and --disaggregated-sweep)")

    # Network overhead (only needs disaggregated)
    if args.disaggregated_sweep:
        plot_network_overhead(args.disaggregated_sweep, args.outdir)
    else:
        print("  ⚠ Skipping network overhead figure (need --disaggregated-sweep)")

    # Concurrency figures (need both CSVs)
    if args.collocated_conc and args.disaggregated_conc:
        plot_concurrency_throughput(args.collocated_conc, args.disaggregated_conc, args.outdir)
        plot_concurrency_tpot(args.collocated_conc, args.disaggregated_conc, args.outdir)
    else:
        print("  ⚠ Skipping concurrency figures (need both --collocated-conc and --disaggregated-conc)")

    print("=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
