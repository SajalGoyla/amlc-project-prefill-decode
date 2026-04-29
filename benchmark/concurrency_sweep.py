"""
Concurrency Sweep Benchmark
============================
Fixes prompt length and output tokens, then ramps up the number of
concurrent requests to show how each architecture scales.

Key insight:
  - Collocated:     single GPU processes requests sequentially → throughput
                    plateaus and per-request TTFT grows linearly with queue.
  - Disaggregated:  prefill + decode are pipelined on separate GPUs →
                    higher throughput because one GPU can prefill request N+1
                    while the other decodes request N.

Metrics per concurrency level:
  - avg / p50 / p99 TPOT
  - avg TTFT (compute only)
  - batch wall-clock time
  - throughput (total tokens / wall-clock seconds)

Usage:
    # Against collocated baseline:
    python benchmark/concurrency_sweep.py \
        --mode collocated \
        --url http://<COLLOCATED_VM_IP>:8000 \
        --output benchmark_results/concurrency_collocated.csv

    # Against disaggregated pipeline:
    python benchmark/concurrency_sweep.py \
        --mode disaggregated \
        --url http://<GATEWAY_IP>:8000 \
        --decode-url http://<DECODE_VM_IP>:8002 \
        --output benchmark_results/concurrency_disaggregated.csv
"""

import os
import sys
import csv
import time
import uuid
import asyncio
import argparse
import logging

import aiohttp
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================

# Concurrency levels to sweep
DEFAULT_CONCURRENCIES = [1, 2, 4, 6, 8, 10]

# Fixed workload per request
DEFAULT_PROMPT_TOKENS = 500
DEFAULT_MAX_OUTPUT    = 128
DEFAULT_REPS          = 3   # full batch repetitions per concurrency level

CHARS_PER_TOKEN = 4
POLL_INTERVAL   = 0.3
POLL_TIMEOUT    = 600.0
STAGGER_MS      = 100      # ms between concurrent request launches

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("concurrency_sweep")

# ==============================================================================
# Prompt Generation
# ==============================================================================

BASE_TEXT = (
    "The field of artificial intelligence has undergone remarkable transformations "
    "over the past several decades, driven by advances in computational power, data "
    "availability, and algorithmic innovation. The transformer architecture has "
    "revolutionized natural language processing by enabling parallel processing of "
    "sequential data through self-attention mechanisms. Large language models have "
    "demonstrated unprecedented capabilities in text generation, reasoning, and "
    "few-shot learning. The deployment of these models at scale presents unique "
    "challenges in inference optimization, memory management, and hardware "
    "utilization that require careful system design. "
)


def generate_prompt(target_tokens: int) -> str:
    target_chars = target_tokens * CHARS_PER_TOKEN
    text = BASE_TEXT
    while len(text) < target_chars + 200:
        text += BASE_TEXT
    return text[:target_chars]


# ==============================================================================
# Single-Request Runners (same logic as prompt_sweep.py)
# ==============================================================================

async def _run_collocated(session, url, prompt, max_tokens, sid):
    payload = {"prompt": prompt, "session_id": sid, "max_new_tokens": max_tokens}
    t0 = time.perf_counter()

    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status != 200:
                return None
    except Exception:
        return None

    deadline = time.perf_counter() + POLL_TIMEOUT
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{url}/metrics/{sid}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "complete":
                        return {
                            "compute_ttft_ms": data.get("compute_ttft_ms", data.get("ttft_ms", 0)),
                            "true_ttft_ms":    data.get("true_ttft_ms", data.get("ttft_ms", 0)),
                            "tpot_ms":         data["tpot_ms"],
                            "tokens":          data["tokens"],
                            "e2e_ms":          (time.perf_counter() - t0) * 1000,
                        }
                    if data.get("status") == "error":
                        return None
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)
    return None


async def _run_disaggregated(session, gw_url, dec_url, prompt, max_tokens, sid):
    payload = {"prompt": prompt, "session_id": sid, "max_new_tokens": max_tokens}
    t0 = time.perf_counter()

    try:
        async with session.post(f"{gw_url}/generate", json=payload) as resp:
            if resp.status != 200:
                return None
    except Exception:
        return None

    # Wait for prefill accepted
    deadline = time.perf_counter() + POLL_TIMEOUT
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{gw_url}/status/{sid}") as resp:
                if resp.status == 200:
                    s = await resp.json()
                    status = s.get("status")
                    if status == "prefill_accepted":
                        break
                    elif status in ("error", "prefill_error", "connection_error"):
                        log.error("Gateway reported error for session %s: %s", sid, s.get("error", "Unknown error"))
                        return None
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)

    # Poll decode metrics
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{dec_url}/metrics/{sid}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "complete":
                        return {
                            "compute_ttft_ms": data.get("forward_time_ms", 0),
                            "true_ttft_ms":    data.get("true_ttft_ms", data.get("forward_time_ms", 0)),
                            "tpot_ms":         data.get("tpot_ms", 0),
                            "kv_transfer_ms":  data.get("kv_receive_time_ms", 0),
                            "tokens":          data.get("tokens", 0),
                            "e2e_ms":          (time.perf_counter() - t0) * 1000,
                        }
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)
    return None


# ==============================================================================
# Concurrency Batch Runner
# ==============================================================================

async def run_batch(session, args, concurrency, prompt, rep):
    """Fire *concurrency* requests with slight stagger and wait for all."""
    tasks = []
    for i in range(concurrency):
        sid = f"conc_{args.mode}_c{concurrency}_r{rep}_i{i}_{uuid.uuid4().hex[:4]}"
        if args.mode == "collocated":
            coro = _run_collocated(session, args.url, prompt, args.max_tokens, sid)
        else:
            coro = _run_disaggregated(
                session, args.url, args.decode_url, prompt, args.max_tokens, sid,
            )
        await asyncio.sleep(STAGGER_MS / 1000.0)
        tasks.append(asyncio.create_task(coro))

    t_batch_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    t_batch_end = time.perf_counter()

    valid = [r for r in results if r is not None]
    wall_clock_s = t_batch_end - t_batch_start

    return valid, wall_clock_s


# ==============================================================================
# Main Sweep
# ==============================================================================

async def run_sweep(args):
    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    prompt = generate_prompt(args.prompt_tokens)

    log.info("=" * 60)
    log.info("CONCURRENCY SWEEP — mode=%s", args.mode)
    log.info("Prompt tokens: %d | Output tokens: %d", args.prompt_tokens, args.max_tokens)
    log.info("Concurrency levels: %s", concurrencies)
    log.info("Reps per level: %d", args.reps)
    log.info("=" * 60)

    csv_rows = []
    timeout = aiohttp.ClientTimeout(total=POLL_TIMEOUT + 120)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for conc in concurrencies:
            for rep in range(1, args.reps + 1):
                log.info("[concurrency=%d] rep %d/%d", conc, rep, args.reps)

                valid, wall_s = await run_batch(session, args, conc, prompt, rep)

                if not valid:
                    log.warning("  → All %d requests failed", conc)
                    continue

                tpots  = [r["tpot_ms"] for r in valid]
                true_ttfts = [r["true_ttft_ms"] for r in valid]
                total_tokens = sum(r["tokens"] for r in valid)
                throughput   = total_tokens / wall_s if wall_s > 0 else 0

                row = {
                    "mode":            args.mode,
                    "concurrency":     conc,
                    "rep":             rep,
                    "successful":      len(valid),
                    "avg_tpot_ms":     round(float(np.mean(tpots)), 2),
                    "p50_tpot_ms":     round(float(np.percentile(tpots, 50)), 2),
                    "p99_tpot_ms":     round(float(np.percentile(tpots, 99)), 2),
                    "avg_true_ttft_ms": round(float(np.mean(true_ttfts)), 2),
                    "total_tokens":    total_tokens,
                    "wall_clock_s":    round(wall_s, 2),
                    "throughput_tps":  round(throughput, 1),
                }
                csv_rows.append(row)

                log.info(
                    "  → ok=%d  avg_tpot=%.1f  avg_true_ttft=%.1f  "
                    "throughput=%.1f tok/s  wall=%.1f s",
                    len(valid), row["avg_tpot_ms"], row["avg_true_ttft_ms"],
                    throughput, wall_s,
                )

                # Cooldown between batches
                await asyncio.sleep(5.0)

    # ----- Write CSV -----
    if not csv_rows:
        log.error("No successful runs.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    log.info("=" * 60)
    log.info("Results saved → %s  (%d rows)", args.output, len(csv_rows))
    log.info("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concurrency sweep benchmark for collocated vs disaggregated."
    )
    parser.add_argument("--mode", required=True, choices=["collocated", "disaggregated"])
    parser.add_argument("--url", required=True)
    parser.add_argument("--decode-url", default=None)
    parser.add_argument(
        "--concurrencies",
        default=",".join(str(c) for c in DEFAULT_CONCURRENCIES),
        help="Comma-separated concurrency levels.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_OUTPUT)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--output", default="benchmark_results/concurrency_results.csv")
    args = parser.parse_args()

    if args.mode == "disaggregated" and not args.decode_url:
        parser.error("--decode-url is required for disaggregated mode.")

    asyncio.run(run_sweep(args))
