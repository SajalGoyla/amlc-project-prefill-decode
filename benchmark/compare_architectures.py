"""
Collocated vs. Disaggregated Architecture Benchmark
===================================================
A comprehensive testing suite that sequentially launches concurrent traffic
against a single-GPU (Collocated) endpoint and a dual-GPU (Disaggregated) pipeline.

Measures and contrasts:
1. Average TTFT (Time To First Token)
2. Average TPOT (Time Per Output Token)
3. End-to-End Pipeline Latency

Both setups must be running for this script to work!
"""

import sys
import time
import asyncio
import aiohttp
import argparse
import numpy as np

DEFAULT_COLLOCATED_URL = "http://localhost:8000"
DEFAULT_GATEWAY_URL = "http://10.128.0.2:8000"   # Example: Internal IP
DEFAULT_DECODE_URL = "http://localhost:8002"  # Exposed Decode Metrics API

PROMPTS = [
    "Explain the history of the Roman Empire and mapping its conquests.",
    "Describe the architectural differences between Transformer models and RNNs in detail.",
    "What are the economic implications of superintelligence? Speculate freely.",
]

async def benchmark_collocated(session, url, req_id, max_tokens):
    """Hits the collocated baseline model API."""
    payload = {"prompt": PROMPTS[req_id % len(PROMPTS)], "max_new_tokens": max_tokens, "session_id": f"coloc_{req_id}"}
    
    t0 = time.perf_counter()
    async with session.post(f"{url}/generate", json=payload) as resp:
        if resp.status != 200:
            return None
            
    # Poll for metrics natively internally
    while True:
        try:
            async with session.get(f"{url}/metrics/coloc_{req_id}") as metric_resp:
                if metric_resp.status == 200:
                    data = await metric_resp.json()
                    if data.get("status") == "complete":
                        t_end = time.perf_counter()
                        data["e2e_lat"] = (t_end - t0) * 1000
                        return data
        except Exception:
            pass
        await asyncio.sleep(0.5)

async def benchmark_disaggregated(session, gateway_url, decode_url, req_id, max_tokens):
    """Hits the Gateway and polls the Decode worker for finalized TPOT/TTFT."""
    session_id = f"disag_{req_id}"
    payload = {"prompt": PROMPTS[req_id % len(PROMPTS)], "max_new_tokens": max_tokens, "session_id": session_id}
    
    t0 = time.perf_counter()
    async with session.post(f"{gateway_url}/generate", json=payload) as resp:
        if resp.status != 200:
            return None
            
    # Wait until TTFT (Gateway reports prefill complete)
    ttft_ms = None
    while True:
        try:
            async with session.get(f"{gateway_url}/status/{session_id}") as status_resp:
                if status_resp.status == 200:
                    status_data = await status_resp.json()
                    if status_data.get("status") in ("prefill_accepted", "error"):
                        if not ttft_ms:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        break
        except Exception:
            pass
        await asyncio.sleep(0.2)

    # Now wait for Decode Metrics API to expose the finalized TPOT
    while True:
        try:
            async with session.get(f"{decode_url}/metrics/{session_id}") as metric_resp:
                if metric_resp.status == 200:
                    data = await metric_resp.json()
                    if data.get("status") == "complete":
                        t_end = time.perf_counter()
                        data["e2e_lat"] = (t_end - t0) * 1000
                        data["ttft_ms"] = ttft_ms
                        return data
        except Exception:
            # Poll quietly, target Decode might not have session mapped yet
            pass
        await asyncio.sleep(0.5)


async def run_suite(args):
    print("=" * 60)
    print("Disaggregated vs Collocated Scaling Benchmark Architecture")
    print(f"Executing {args.concurrent} concurrent concurrent requests per pipeline.")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. Collocated Test
        if args.skip_collocated:
            print("Skipping Collocated Test...")
            coloc_results = []
        else:
            print(f"\\n--- Running COLLOCATED Baseline Baseline ({args.collocated_url}) ---")
            tasks = []
            for i in range(args.concurrent):
                await asyncio.sleep(0.1)
                tasks.append(asyncio.create_task(benchmark_collocated(session, args.collocated_url, i, args.max_tokens)))
            
            coloc_results = [r for r in await asyncio.gather(*tasks) if r]
            print("Collocated Test Finished.")

        # 2. Disaggregated Test
        print(f"\\n--- Running DISAGGREGATED Baseline ({args.gateway_url}) ---")
        tasks = []
        for i in range(args.concurrent):
            await asyncio.sleep(0.1)
            tasks.append(asyncio.create_task(benchmark_disaggregated(session, args.gateway_url, args.decode_url, i, args.max_tokens)))
        
        disag_results = [r for r in await asyncio.gather(*tasks) if r]
        print("Disaggregated Test Finished.\\n")
        
    # Generate Output
    print("=" * 60)
    print("FINAL COMPARISON REPORT")
    print("=" * 60)
    
    def print_stats(name, results):
        if not results:
            return
        ttft = [r["ttft_ms"] for r in results]
        tpot = [r["tpot_ms"] for r in results]
        e2e = [r["e2e_lat"] for r in results]
        print(f"\\n## {name} Metrics (Concurrent Load: {args.concurrent})")
        print(f"  Avg TTFT (Time-To-First-Token) : {np.mean(ttft):.1f} ms")
        print(f"  Avg TPOT (Time-Per-Output)     : {np.mean(tpot):.1f} ms")
        print(f"  Total Batch Generation Time    : {np.max(e2e)/1000.0:.2f} seconds")
        print(f"  Cumulative System Throughput   : {sum([r['tokens'] for r in results]) / (np.max(e2e)/1000.0):.1f} tokens/sec")

    print_stats("COLLOCATED (Single GPU)", coloc_results)
    if not args.skip_collocated:
        print("\\n  VS\\n")
    print_stats("DISAGGREGATED (Dual GPU ZMQ Tunnel)", disag_results)
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collocated-url", default=DEFAULT_COLLOCATED_URL)
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--decode-url", default=DEFAULT_DECODE_URL)
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--skip-collocated", action="store_true")
    
    asyncio.run(run_suite(parser.parse_args()))
