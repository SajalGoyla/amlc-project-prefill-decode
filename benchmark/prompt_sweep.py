"""
Prompt-Length Sweep Benchmark
=============================
Sweeps across prompt token lengths and measures:
  - compute_ttft_ms:  Pure GPU prefill time (no network)
  - tpot_ms:          Time Per Output Token
  - kv_transfer_ms:   Network overhead for KV cache transfer (disaggregated only)
  - cache_recon_ms:   Cache reconstruction time on decode GPU (disaggregated only)
  - e2e_ms:           Total end-to-end latency from client perspective
  - tokens_generated: Number of output tokens produced

Outputs a CSV file for plotting the crossover curve.

Usage:
    # Against collocated baseline (run on any machine that can reach the VM):
    python benchmark/prompt_sweep.py \
        --mode collocated \
        --url http://<COLLOCATED_VM_IP>:8000 \
        --output benchmark_results/sweep_collocated.csv

    # Against disaggregated pipeline:
    python benchmark/prompt_sweep.py \
        --mode disaggregated \
        --url http://<GATEWAY_IP>:8000 \
        --decode-url http://<DECODE_VM_IP>:8002 \
        --output benchmark_results/sweep_disaggregated.csv
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
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# Configuration
# ==============================================================================

# Prompt token lengths to sweep
DEFAULT_LENGTHS = [50, 100, 200, 400, 800, 1200, 1600, 2000]

# Fixed output length (isolates prefill/TTFT effect)
DEFAULT_MAX_OUTPUT = 128

# Repetitions per prompt length
DEFAULT_REPS = 5

# Polling
POLL_INTERVAL = 0.3       # seconds between polls
POLL_TIMEOUT  = 600.0     # max seconds to wait per request

# LLaMA-class models average ~4 characters per token
CHARS_PER_TOKEN = 4

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prompt_sweep")

# ==============================================================================
# Prompt Generation
# ==============================================================================

BASE_TEXT = (
    "The field of artificial intelligence has undergone remarkable transformations "
    "over the past several decades, driven by advances in computational power, data "
    "availability, and algorithmic innovation. The transformer architecture, introduced "
    "in the seminal paper Attention Is All You Need, revolutionized natural language "
    "processing by enabling parallel processing of sequential data through self-attention "
    "mechanisms. Large language models built on this architecture have demonstrated "
    "unprecedented capabilities in text generation, reasoning, and few-shot learning. "
    "The deployment of these models at scale presents unique challenges in inference "
    "optimization, memory management, and hardware utilization that require careful "
    "system design and engineering. "
)


def generate_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately *target_tokens* tokens.

    Uses the ~4 chars/token heuristic for LLaMA-class tokenizers.
    """
    target_chars = target_tokens * CHARS_PER_TOKEN
    text = BASE_TEXT
    while len(text) < target_chars + 200:      # overshoot then trim
        text += BASE_TEXT
    return text[:target_chars]


# ==============================================================================
# Collocated Benchmark
# ==============================================================================

async def run_collocated_single(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    session_id: str,
) -> dict | None:
    """Single request against the collocated baseline.

    Returns a dict with timing metrics, or None on failure.
    """
    payload = {
        "prompt": prompt,
        "session_id": session_id,
        "max_new_tokens": max_tokens,
    }

    t0 = time.perf_counter()

    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status != 200:
                log.error("Collocated /generate returned %d", resp.status)
                return None
    except Exception as e:
        log.error("Collocated /generate failed: %s", e)
        return None

    # Poll /metrics/{session_id} until complete
    deadline = time.perf_counter() + POLL_TIMEOUT
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{url}/metrics/{session_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "complete":
                        e2e_ms = (time.perf_counter() - t0) * 1000
                        return {
                            "compute_ttft_ms": data.get("compute_ttft_ms", data.get("ttft_ms", 0)),
                            "tpot_ms":         data.get("tpot_ms", 0),
                            "kv_transfer_ms":  0.0,
                            "cache_recon_ms":  0.0,
                            "tokens":          data.get("tokens", 0),
                            "e2e_ms":          round(e2e_ms, 2),
                        }
                    if data.get("status") == "error":
                        log.error("Collocated error: %s", data)
                        return None
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)

    log.error("Collocated request timed out for session %s", session_id)
    return None


# ==============================================================================
# Disaggregated Benchmark
# ==============================================================================

async def run_disaggregated_single(
    session: aiohttp.ClientSession,
    gateway_url: str,
    decode_url: str,
    prompt: str,
    max_tokens: int,
    session_id: str,
) -> dict | None:
    """Single request against the disaggregated pipeline.

    Collects compute_ttft (forward_time_ms from prefill worker),
    kv_transfer_ms, cache_recon_ms, and tpot_ms from the decode worker
    metrics API.
    """
    payload = {
        "prompt": prompt,
        "session_id": session_id,
        "max_new_tokens": max_tokens,
    }

    t0 = time.perf_counter()

    try:
        async with session.post(f"{gateway_url}/generate", json=payload) as resp:
            if resp.status != 200:
                log.error("Gateway /generate returned %d", resp.status)
                return None
    except Exception as e:
        log.error("Gateway /generate failed: %s", e)
        return None

    # Wait for gateway to report prefill_accepted
    deadline = time.perf_counter() + POLL_TIMEOUT
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{gateway_url}/status/{session_id}") as resp:
                if resp.status == 200:
                    s = await resp.json()
                    status = s.get("status")
                    if status == "prefill_accepted":
                        break
                    elif status in ("error", "prefill_error", "connection_error"):
                        log.error("Gateway reported error for session %s: %s", session_id, s.get("error", "Unknown error"))
                        return None
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)

    # Now poll the Decode worker metrics API for full results
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{decode_url}/metrics/{session_id}") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("status") == "complete":
                        e2e_ms = (time.perf_counter() - t0) * 1000
                        return {
                            "compute_ttft_ms": data.get("forward_time_ms", 0),
                            "tpot_ms":         data.get("tpot_ms", 0),
                            "kv_transfer_ms":  data.get("kv_receive_time_ms", 0),
                            "cache_recon_ms":  data.get("cache_reconstruct_ms", 0),
                            "tokens":          data.get("tokens", 0),
                            "e2e_ms":          round(e2e_ms, 2),
                        }
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)

    log.error("Disaggregated request timed out for session %s", session_id)
    return None


# ==============================================================================
# Main Sweep
# ==============================================================================

async def run_sweep(args):
    lengths  = [int(x) for x in args.lengths.split(",")]
    csv_rows = []

    log.info("=" * 60)
    log.info("PROMPT-LENGTH SWEEP — mode=%s", args.mode)
    log.info("Lengths (tokens): %s", lengths)
    log.info("Output tokens:    %d", args.max_tokens)
    log.info("Repetitions:      %d", args.reps)
    log.info("=" * 60)

    timeout = aiohttp.ClientTimeout(total=POLL_TIMEOUT + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for tgt_tokens in lengths:
            prompt = generate_prompt(tgt_tokens)

            for rep in range(1, args.reps + 1):
                sid = f"sweep_{args.mode}_{tgt_tokens}t_r{rep}_{uuid.uuid4().hex[:6]}"
                log.info(
                    "[%d tokens] rep %d/%d  sid=%s",
                    tgt_tokens, rep, args.reps, sid,
                )

                if args.mode == "collocated":
                    result = await run_collocated_single(
                        session, args.url, prompt, args.max_tokens, sid,
                    )
                else:
                    result = await run_disaggregated_single(
                        session, args.url, args.decode_url,
                        prompt, args.max_tokens, sid,
                    )

                if result is None:
                    log.warning("  → FAILED, skipping")
                    continue

                row = {
                    "mode":            args.mode,
                    "target_tokens":   tgt_tokens,
                    "rep":             rep,
                    "session_id":      sid,
                    "compute_ttft_ms": result["compute_ttft_ms"],
                    "tpot_ms":         result["tpot_ms"],
                    "kv_transfer_ms":  result["kv_transfer_ms"],
                    "cache_recon_ms":  result["cache_recon_ms"],
                    "tokens":          result["tokens"],
                    "e2e_ms":          result["e2e_ms"],
                }
                csv_rows.append(row)

                log.info(
                    "  → compute_ttft=%.1f ms  tpot=%.1f ms  "
                    "kv_net=%.1f ms  recon=%.1f ms  tokens=%d  e2e=%.0f ms",
                    result["compute_ttft_ms"],
                    result["tpot_ms"],
                    result["kv_transfer_ms"],
                    result["cache_recon_ms"],
                    result["tokens"],
                    result["e2e_ms"],
                )

                # Cooldown between reps to let GPU caches settle
                await asyncio.sleep(2.0)

    # ----- Write CSV -----
    if not csv_rows:
        log.error("No successful runs. Nothing to write.")
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fieldnames = list(csv_rows[0].keys())
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
        description="Prompt-length sweep benchmark for collocated vs disaggregated LLM serving."
    )
    parser.add_argument(
        "--mode", required=True, choices=["collocated", "disaggregated"],
        help="Which architecture to benchmark.",
    )
    parser.add_argument(
        "--url", required=True,
        help="Base URL.  Collocated: http://<IP>:8000   Disaggregated: http://<GATEWAY_IP>:8000",
    )
    parser.add_argument(
        "--decode-url", default=None,
        help="(disaggregated only) Decode worker metrics URL, e.g. http://<DECODE_IP>:8002",
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(x) for x in DEFAULT_LENGTHS),
        help="Comma-separated list of target prompt token lengths.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_OUTPUT,
        help="Fixed number of output tokens per request.",
    )
    parser.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help="Repetitions per prompt length.",
    )
    parser.add_argument(
        "--output", default="benchmark_results/sweep_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.mode == "disaggregated" and not args.decode_url:
        parser.error("--decode-url is required for disaggregated mode.")

    asyncio.run(run_sweep(args))
