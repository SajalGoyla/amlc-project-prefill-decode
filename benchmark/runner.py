"""
runner.py — Benchmark orchestrator.

Runs workloads against endpoints and writes results to timestamped CSVs.

Usage:
    python -m benchmark.runner \
        --endpoint http://100.x.x.x:8000/v1 \
        --profile chat \
        --arch collocated \
        --num-requests 30 \
        --out benchmark/results/
"""

import argparse
import asyncio
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.client import run_benchmark_requests
from benchmark.workloads import (
    ALL_PROFILES,
    SWEEP_INPUT_TOKENS,
    generate_sweep_workload,
    generate_workload,
)


def results_to_csv(
    results, arch: str, profile_name: str, out_dir: str, run_id: str
) -> str:
    """Write benchmark results to CSV. Returns the file path."""
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{arch}_{profile_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    fieldnames = [
        "timestamp",
        "run_id",
        "arch",
        "profile",
        "request_id",
        "prompt_tokens",
        "output_tokens",
        "ttft_ms",
        "itl_ms_mean",
        "itl_ms_p50",
        "itl_ms_p95",
        "throughput_tps",
        "total_latency_ms",
        "error",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "arch": arch,
                "profile": profile_name,
                "request_id": r.request_id,
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "ttft_ms": round(r.ttft_ms, 2),
                "itl_ms_mean": round(r.itl_ms_mean, 2),
                "itl_ms_p50": round(r.itl_ms_p50, 2),
                "itl_ms_p95": round(r.itl_ms_p95, 2),
                "throughput_tps": round(r.throughput_tps, 2),
                "total_latency_ms": round(r.total_latency_ms, 2),
                "error": r.error or "",
            })

    return filepath


async def run_profile(
    endpoint: str,
    arch: str,
    profile_name: str,
    num_requests: int,
    out_dir: str,
    model: str,
    concurrency: int = 1,
):
    """Run a single profile and save results."""
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if profile_name.startswith("sweep_"):
        # Sweep profile: "sweep_500" → 500 input tokens
        input_tokens = int(profile_name.split("_")[1])
        workload = generate_sweep_workload(input_tokens, num_requests, model)
    else:
        # Standard profile
        profile_map = {p.name: p for p in ALL_PROFILES}
        if profile_name not in profile_map:
            print(f"Unknown profile: {profile_name}")
            print(f"Available: {list(profile_map.keys())} + sweep_<N>")
            return
        profile = profile_map[profile_name]
        workload = generate_workload(profile, num_requests, model)

    print(f"\n{'='*60}")
    print(f"  Running: {profile_name} ({arch})")
    print(f"  Endpoint: {endpoint}")
    print(f"  Requests: {num_requests}, Concurrency: {concurrency}")
    print(f"{'='*60}\n")

    results = await run_benchmark_requests(
        endpoint=endpoint,
        prompts=workload,
        model=model,
        concurrency=concurrency,
    )

    # Print summary
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if successful:
        avg_ttft = sum(r.ttft_ms for r in successful) / len(successful)
        avg_itl = sum(r.itl_ms_mean for r in successful) / len(successful)
        avg_tps = sum(r.throughput_tps for r in successful) / len(successful)
        print(f"  ✓ {len(successful)} successful, {len(failed)} failed")
        print(f"  TTFT:       {avg_ttft:.1f} ms (mean)")
        print(f"  ITL:        {avg_itl:.1f} ms (mean)")
        print(f"  Throughput: {avg_tps:.1f} tokens/s (mean)")
    else:
        print(f"  ✗ All {len(failed)} requests failed")
        if failed:
            print(f"  Error: {failed[0].error}")

    filepath = results_to_csv(results, arch, profile_name, out_dir, run_id)
    print(f"  Saved to: {filepath}\n")
    return filepath


async def run_all_profiles(
    endpoint: str,
    arch: str,
    num_requests: int,
    out_dir: str,
    model: str,
    concurrency: int = 1,
    include_sweep: bool = False,
):
    """Run all standard profiles + optional sweep."""
    for profile in ALL_PROFILES:
        await run_profile(
            endpoint, arch, profile.name, num_requests, out_dir, model, concurrency
        )

    if include_sweep:
        for input_tokens in SWEEP_INPUT_TOKENS:
            await run_profile(
                endpoint, arch, f"sweep_{input_tokens}",
                num_requests, out_dir, model, concurrency,
            )


def main():
    parser = argparse.ArgumentParser(description="AMLIC Benchmark Runner")
    parser.add_argument("--endpoint", required=True, help="vLLM endpoint URL (e.g., http://100.x.x.x:8000/v1)")
    parser.add_argument("--arch", required=True, choices=["collocated", "disaggregated"], help="Architecture being tested")
    parser.add_argument("--profile", default="all", help="Profile name: chat, doc_qa, balanced, sweep_<N>, or 'all'")
    parser.add_argument("--num-requests", type=int, default=30, help="Number of requests per profile")
    parser.add_argument("--out", default="benchmark/results", help="Output directory for CSVs")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent requests")
    parser.add_argument("--include-sweep", action="store_true", help="Include prompt-length sweep")
    args = parser.parse_args()

    if args.profile == "all":
        asyncio.run(run_all_profiles(
            args.endpoint, args.arch, args.num_requests, args.out,
            args.model, args.concurrency, args.include_sweep,
        ))
    else:
        asyncio.run(run_profile(
            args.endpoint, args.arch, args.profile, args.num_requests,
            args.out, args.model, args.concurrency,
        ))


if __name__ == "__main__":
    main()
