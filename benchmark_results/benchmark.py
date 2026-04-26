"""
Phase 3: Benchmarking & Validation Script
==========================================
End-to-end benchmarking for the disaggregated LLM serving system.

Measures:
    - TTFT (Time To First Token): Latency from request to first generated token
    - TPOT (Time Per Output Token): Average per-token latency during decode
    - Total E2E Latency: Full round-trip time

Usage:
    python benchmark/benchmark.py
    python benchmark/benchmark.py --gateway-url http://10.128.0.2:8000
    python benchmark/benchmark.py --prompt "Explain quantum computing" --max-tokens 256
"""

import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime

import requests
from dotenv import load_dotenv

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

DEFAULT_GATEWAY_HOST = os.getenv("GATEWAY_HOST", "localhost")
DEFAULT_GATEWAY_PORT = os.getenv("GATEWAY_PORT", "8000")
DEFAULT_GATEWAY_URL = f"http://{DEFAULT_GATEWAY_HOST}:{DEFAULT_GATEWAY_PORT}"

DEFAULT_PROMPT = (
    "You are a helpful AI assistant. Explain the key differences between "
    "supervised learning and unsupervised learning in machine learning. "
    "Provide concrete examples of each."
)

DEFAULT_MAX_TOKENS = 128
POLL_INTERVAL_S = 0.5     # How often to poll session status
POLL_TIMEOUT_S = 300.0    # Max time to wait for generation

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ==============================================================================
# Benchmark Functions
# ==============================================================================


def check_gateway_health(gateway_url: str) -> bool:
    """Verify the Gateway is reachable and healthy."""
    try:
        resp = requests.get(f"{gateway_url}/health", timeout=10)
        if resp.status_code == 200:
            health = resp.json()
            logger.info("Gateway health: %s", json.dumps(health, indent=2))
            return True
        else:
            logger.error("Gateway health check failed: HTTP %d", resp.status_code)
            return False
    except requests.ConnectionError as e:
        logger.error("Cannot reach Gateway at %s: %s", gateway_url, e)
        return False


def submit_request(
    gateway_url: str,
    prompt: str,
    max_new_tokens: int,
    session_id: str | None = None,
) -> dict:
    """
    Submit a generation request to the Gateway.

    Returns the response dict containing the assigned session_id.
    """
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }
    if session_id:
        payload["session_id"] = session_id

    logger.info("Submitting request to %s/generate", gateway_url)
    logger.info("  Prompt length: %d chars", len(prompt))
    logger.info("  Max new tokens: %d", max_new_tokens)

    t_submit = time.perf_counter()
    resp = requests.post(
        f"{gateway_url}/generate",
        json=payload,
        timeout=30,
    )
    t_ack = time.perf_counter()

    if resp.status_code != 200:
        logger.error(
            "Gateway returned HTTP %d: %s", resp.status_code, resp.text
        )
        raise RuntimeError(f"Gateway error: {resp.status_code}")

    result = resp.json()
    ack_latency_ms = (t_ack - t_submit) * 1000

    logger.info("Request acknowledged in %.1f ms", ack_latency_ms)
    logger.info("  Session ID: %s", result.get("session_id"))
    logger.info("  Status: %s", result.get("status"))

    return {
        "response": result,
        "submit_time": t_submit,
        "ack_time": t_ack,
        "ack_latency_ms": ack_latency_ms,
    }


def poll_session_status(
    gateway_url: str,
    session_id: str,
    timeout_s: float = POLL_TIMEOUT_S,
) -> dict:
    """
    Poll the Gateway for session status until completion or timeout.

    Returns the final session status dict.
    """
    t_start = time.perf_counter()
    last_status = None

    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed > timeout_s:
            logger.error(
                "Timeout after %.1f s waiting for session '%s'",
                elapsed,
                session_id,
            )
            return {"status": "timeout", "elapsed_s": elapsed}

        try:
            resp = requests.get(
                f"{gateway_url}/status/{session_id}", timeout=10
            )
            if resp.status_code == 200:
                status = resp.json()
                current_status = status.get("status", "unknown")

                if current_status != last_status:
                    logger.info(
                        "Session '%s' status: %s (%.1f s elapsed)",
                        session_id,
                        current_status,
                        elapsed,
                    )
                    last_status = current_status

                # Terminal states
                if current_status in (
                    "prefill_accepted",
                    "error",
                    "connection_error",
                    "prefill_error",
                ):
                    return status

            elif resp.status_code == 404:
                logger.warning("Session '%s' not found yet...", session_id)
            else:
                logger.warning(
                    "Unexpected status response: HTTP %d", resp.status_code
                )

        except requests.ConnectionError:
            logger.warning("Lost connection to Gateway, retrying...")

        time.sleep(POLL_INTERVAL_S)


def run_benchmark(
    gateway_url: str,
    prompt: str,
    max_new_tokens: int,
    num_runs: int = 1,
) -> dict:
    """
    Run the full benchmark:
      1. Submit request
      2. Measure acknowledgment latency
      3. Poll for completion
      4. Report metrics

    Note: TTFT and TPOT are approximate since the Gateway uses
    fire-and-forget. For precise timing, check the Decode Worker logs.
    """
    results = []

    for run_idx in range(num_runs):
        logger.info("=" * 60)
        logger.info("Benchmark Run %d/%d", run_idx + 1, num_runs)
        logger.info("=" * 60)

        t_e2e_start = time.perf_counter()

        # Step 1: Submit request
        try:
            submit_result = submit_request(
                gateway_url=gateway_url,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.error("Request submission failed: %s", e)
            results.append({"run": run_idx + 1, "status": "submit_failed", "error": str(e)})
            continue

        session_id = submit_result["response"]["session_id"]

        # Step 2: Poll for status
        final_status = poll_session_status(gateway_url, session_id)

        t_e2e_end = time.perf_counter()
        e2e_latency_ms = (t_e2e_end - t_e2e_start) * 1000

        # Step 3: Compile results
        run_result = {
            "run": run_idx + 1,
            "session_id": session_id,
            "ack_latency_ms": round(submit_result["ack_latency_ms"], 2),
            "e2e_latency_ms": round(e2e_latency_ms, 2),
            "final_status": final_status.get("status", "unknown"),
            "prompt_length_chars": len(prompt),
            "max_new_tokens": max_new_tokens,
        }

        results.append(run_result)
        logger.info("Run %d complete: %s", run_idx + 1, json.dumps(run_result, indent=2))

        # Brief pause between runs
        if run_idx < num_runs - 1:
            time.sleep(2.0)

    return {"runs": results, "summary": compute_summary(results)}


def compute_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics from benchmark runs."""
    successful = [r for r in results if r.get("final_status") not in ("submit_failed", "timeout")]

    if not successful:
        return {"status": "all_failed", "num_runs": len(results)}

    ack_latencies = [r["ack_latency_ms"] for r in successful]
    e2e_latencies = [r["e2e_latency_ms"] for r in successful]

    return {
        "num_runs": len(results),
        "num_successful": len(successful),
        "ack_latency_ms": {
            "min": round(min(ack_latencies), 2),
            "max": round(max(ack_latencies), 2),
            "avg": round(sum(ack_latencies) / len(ack_latencies), 2),
        },
        "e2e_latency_ms": {
            "min": round(min(e2e_latencies), 2),
            "max": round(max(e2e_latencies), 2),
            "avg": round(sum(e2e_latencies) / len(e2e_latencies), 2),
        },
        "note": (
            "TTFT and TPOT are measured on the Decode Worker side. "
            "Check decode_worker logs for precise per-token timing. "
            "E2E latency here measures Gateway round-trip only."
        ),
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the disaggregated LLM serving system."
    )
    parser.add_argument(
        "--gateway-url",
        default=DEFAULT_GATEWAY_URL,
        help=f"URL of the API Gateway (default: {DEFAULT_GATEWAY_URL})",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to send for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum new tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of benchmark runs (default: 1)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Disaggregated LLM Serving — Benchmark")
    logger.info("=" * 60)
    logger.info("  Gateway URL:    %s", args.gateway_url)
    logger.info("  Prompt:         %s...", args.prompt[:80])
    logger.info("  Max tokens:     %d", args.max_tokens)
    logger.info("  Num runs:       %d", args.num_runs)
    logger.info("  Timestamp:      %s", datetime.now().isoformat())
    logger.info("=" * 60)

    # Health check
    if not check_gateway_health(args.gateway_url):
        logger.error("Gateway health check failed. Aborting benchmark.")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        gateway_url=args.gateway_url,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        num_runs=args.num_runs,
    )

    # Print final summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("=" * 60)

    # Save results to file
    results_file = os.path.join(
        os.path.dirname(__file__),
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_file)


if __name__ == "__main__":
    main()
