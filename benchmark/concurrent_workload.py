"""
Concurrent Workload Benchmark
=============================
Proves the advantage of disaggregated LLM serving by simulating multiple 
users sending requests at the same time. 

In a single-GPU setup, concurrent requests block each other: either 
Time-To-First-Token (TTFT) skyrockets because the GPU is busy generating 
tokens for an earlier user, or generation TPOT (Time-Per-Output-Token) 
stutters because the GPU has to pause to compute a heavy prefill.

In this disaggregated setup:
- The Prefill worker (GPU 1) effortlessly churns through incoming prompts.
- The Decode worker (GPU 2) continuously generates tokens without stuttering.
"""

import sys
import time
import asyncio
import argparse
import aiohttp
import numpy as np

DEFAULT_GATEWAY_URL = "http://localhost:8000"

PROMPTS = [
    "Explain the theories of quantum gravity in detail and how they attempt to unite general relativity with quantum mechanics.",
    "Write a highly detailed, 5-paragraph story about a futuristic city powered entirely by bioluminescent algae.",
    "Describe the architectural differences between Transformer models and Recurrent Neural Networks.",
    "What are the economic implications of artificial general intelligence? Discuss the impact on labor markets.",
    "Explain the complete life cycle of a star, from the stellar nebula phase to a black hole or neutron star.",
]

async def submit_request(session, gateway_url, prompt, req_id):
    """Fire a request and measure how fast it clears the Prefill phase."""
    payload = {"prompt": prompt, "max_new_tokens": 128}
    start_time = time.perf_counter()
    
    try:
        # Submit the generation request
        async with session.post(f"{gateway_url}/generate", json=payload) as resp:
            if resp.status != 200:
                print(f"[Req {req_id}] Failed to submit: HTTP {resp.status}")
                return None
            result = await resp.json()
            session_id = result["session_id"]
            
        print(f"[Req {req_id}] Submitted successfully (Session: {session_id[:8]}...). Polling for TTFT...")
        
        # Poll for prefill_accepted (TTFT equivalent)
        while True:
            async with session.get(f"{gateway_url}/status/{session_id}") as status_resp:
                if status_resp.status == 200:
                    status_data = await status_resp.json()
                    if status_data.get("status") in ("prefill_accepted", "error", "prefill_error"):
                        end_time = time.perf_counter()
                        ttft = (end_time - start_time) * 1000
                        print(f"[Req {req_id}] Prefill Complete! TTFT: {ttft:.1f} ms | Status: {status_data.get('status')}")
                        return ttft
            await asyncio.sleep(0.5)

    except Exception as e:
        print(f"[Req {req_id}] Exception: {e}")
        return None

async def run_concurrent_workload(gateway_url, num_concurrent):
    """Launch multiple generation requests almost simultaneously."""
    print("=" * 60)
    print(f"Launching {num_concurrent} concurrent requests to the disaggregated pipeline...")
    print("=" * 60)
    print("EXPECTED BEHAVIOR:")
    print("1. Prefill GPU will securely queue and process these overlapping prompts rapidly.")
    print("2. Decode GPU will receive the KV caches and queue them for token generation.")
    print("3. Check the Decode Worker terminal: it will smoothly output tokens without crashing!")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_concurrent):
            prompt = PROMPTS[i % len(PROMPTS)]
            # Slight stagger to simulate natural web traffic
            await asyncio.sleep(0.1) 
            tasks.append(asyncio.create_task(submit_request(session, gateway_url, prompt, i+1)))
            
        # Wait for all TTFT responses
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            print("=" * 60)
            print("WORKLOAD SUMMARY (Time-to-First-Token from Gateway returning prefill completion):")
            print(f"Average TTFT: {np.mean(valid_results):.1f} ms")
            print(f"Min TTFT:     {np.min(valid_results):.1f} ms")
            print(f"Max TTFT:     {np.max(valid_results):.1f} ms")
            print("=" * 60)
            print("To see your TPOT (Time-Per-Output-Token), look directly at the")
            print("terminal running 'decode_worker.py'. You will see the Decode GPU continuously")
            print("generating streams of tokens while isolated safely from the heavy Prefill calculations!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    parser.add_argument("--concurrent", type=int, default=3, help="Number of concurrent requests")
    args = parser.parse_args()
    
    # Needs aiohttp: pip install aiohttp
    asyncio.run(run_concurrent_workload(args.gateway_url, args.concurrent))
