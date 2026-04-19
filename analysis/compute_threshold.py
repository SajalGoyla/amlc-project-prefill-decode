"""
compute_threshold.py — Find the crossover threshold N.

Reads benchmark sweep CSVs for both architectures, computes the prompt-length N
where disaggregated serving becomes faster than collocated serving.

Usage:
    python -m analysis.compute_threshold --results-dir benchmark/results/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_dir: str) -> pd.DataFrame:
    """Load all CSV results from the results directory."""
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_dir}")
        sys.exit(1)

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def compute_crossover(df: pd.DataFrame, metric: str = "ttft_ms") -> dict:
    """
    Find the crossover threshold N between collocated and disaggregated.

    Looks at sweep profiles and finds where disaggregated becomes favorable.
    Returns dict with threshold info.
    """
    # Filter to sweep profiles only
    sweep_mask = df["profile"].str.startswith("sweep_")
    sweep_df = df[sweep_mask].copy()

    if sweep_df.empty:
        return {"threshold": None, "error": "No sweep data found"}

    # Extract input token count from profile name
    sweep_df["input_tokens_target"] = sweep_df["profile"].str.extract(r"sweep_(\d+)").astype(int)

    # Filter out errors
    sweep_df = sweep_df[sweep_df["error"] == ""].copy() if "error" in sweep_df.columns else sweep_df

    # Aggregate by arch and input token target
    agg = sweep_df.groupby(["arch", "input_tokens_target"]).agg(
        metric_mean=(metric, "mean"),
        metric_p50=(metric, "median"),
        metric_std=(metric, "std"),
        count=("request_id", "count"),
    ).reset_index()

    collocated = agg[agg["arch"] == "collocated"].sort_values("input_tokens_target")
    disagg = agg[agg["arch"] == "disaggregated"].sort_values("input_tokens_target")

    if collocated.empty or disagg.empty:
        return {"threshold": None, "error": "Need both collocated and disaggregated sweep data"}

    # Merge on input_tokens_target
    merged = pd.merge(
        collocated[["input_tokens_target", "metric_mean"]],
        disagg[["input_tokens_target", "metric_mean"]],
        on="input_tokens_target",
        suffixes=("_colo", "_disagg"),
    )
    merged["delta"] = merged["metric_mean_disagg"] - merged["metric_mean_colo"]

    # Find crossover: where delta changes sign from positive to negative
    # (disagg goes from slower to faster)
    crossover = None
    for i in range(1, len(merged)):
        if merged.iloc[i - 1]["delta"] > 0 and merged.iloc[i]["delta"] <= 0:
            # Linear interpolation
            x1 = merged.iloc[i - 1]["input_tokens_target"]
            x2 = merged.iloc[i]["input_tokens_target"]
            y1 = merged.iloc[i - 1]["delta"]
            y2 = merged.iloc[i]["delta"]
            crossover = x1 + (x2 - x1) * (-y1) / (y2 - y1)
            break

    # Build result
    result = {
        "metric": metric,
        "threshold": round(crossover) if crossover else None,
        "raw_data": merged.to_dict("records"),
        "collocated_summary": collocated[["input_tokens_target", "metric_mean", "metric_std"]].to_dict("records"),
        "disagg_summary": disagg[["input_tokens_target", "metric_mean", "metric_std"]].to_dict("records"),
    }

    if crossover is None:
        # Check if disagg is always faster or always slower
        if (merged["delta"] <= 0).all():
            result["note"] = "Disaggregated is always faster for all tested input lengths"
            result["threshold"] = merged["input_tokens_target"].min()
        elif (merged["delta"] > 0).all():
            result["note"] = "Disaggregated is always slower for all tested input lengths"
            result["threshold"] = None

    return result


def main():
    parser = argparse.ArgumentParser(description="Compute crossover threshold N")
    parser.add_argument("--results-dir", default="benchmark/results", help="Directory with CSV results")
    parser.add_argument("--metric", default="ttft_ms", choices=["ttft_ms", "total_latency_ms", "itl_ms_mean"], help="Metric to analyze")
    parser.add_argument("--output-env", default=".env", help="Write threshold to .env file")
    args = parser.parse_args()

    df = load_results(args.results_dir)

    print(f"\n{'='*60}")
    print(f"  Crossover Analysis ({args.metric})")
    print(f"{'='*60}")
    print(f"  Total rows: {len(df)}")
    print(f"  Architectures: {df['arch'].unique().tolist()}")
    print(f"  Profiles: {df['profile'].unique().tolist()}")

    result = compute_crossover(df, args.metric)

    if result["threshold"]:
        print(f"\n  ✓ Crossover threshold N = {result['threshold']} tokens")
        print(f"    Below {result['threshold']} tokens → collocated is better")
        print(f"    Above {result['threshold']} tokens → disaggregated is better")

        if result.get("note"):
            print(f"    Note: {result['note']}")

        # Update .env
        if os.path.exists(args.output_env):
            with open(args.output_env, "r") as f:
                env_content = f.read()
            if "ROUTING_THRESHOLD=" in env_content:
                import re
                env_content = re.sub(
                    r"ROUTING_THRESHOLD=\d+",
                    f"ROUTING_THRESHOLD={result['threshold']}",
                    env_content,
                )
                with open(args.output_env, "w") as f:
                    f.write(env_content)
                print(f"\n  Updated {args.output_env} with ROUTING_THRESHOLD={result['threshold']}")
    else:
        print(f"\n  ✗ No crossover found")
        if result.get("error"):
            print(f"    Error: {result['error']}")

    # Print raw data table
    if result.get("raw_data"):
        print(f"\n  {'Tokens':<10} {'Colo (ms)':<12} {'Disagg (ms)':<12} {'Δ (ms)':<10}")
        print(f"  {'-'*44}")
        for row in result["raw_data"]:
            print(
                f"  {row['input_tokens_target']:<10} "
                f"{row['metric_mean_colo']:<12.1f} "
                f"{row['metric_mean_disagg']:<12.1f} "
                f"{row['delta']:<10.1f}"
            )

    print()


if __name__ == "__main__":
    main()
