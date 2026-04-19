"""
client.py — Async OpenAI-compatible benchmark client with TTFT/ITL timing.

Sends streaming chat completion requests and captures:
- TTFT (time to first token)
- ITL (inter-token latency) per token
- Total latency
- Output token count
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp


@dataclass
class RequestResult:
    """Result of a single benchmark request."""
    request_id: int
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float                    # Time to first token
    itl_ms: list[float] = field(default_factory=list)  # Per-token inter-token latency
    total_latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def itl_ms_mean(self) -> float:
        return sum(self.itl_ms) / len(self.itl_ms) if self.itl_ms else 0.0

    @property
    def itl_ms_p50(self) -> float:
        if not self.itl_ms:
            return 0.0
        s = sorted(self.itl_ms)
        return s[len(s) // 2]

    @property
    def itl_ms_p95(self) -> float:
        if not self.itl_ms:
            return 0.0
        s = sorted(self.itl_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def throughput_tps(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_latency_ms / 1000.0)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    prompt_tokens: int,
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> RequestResult:
    """Send a single streaming chat completion request and time it."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,  # Deterministic for benchmarking
    }

    t_start = time.perf_counter()
    t_first_token = None
    t_prev_token = None
    itl_list = []
    output_tokens = 0
    error = None

    try:
        async with session.post(
            f"{endpoint}/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    output_tokens=0,
                    ttft_ms=0,
                    total_latency_ms=0,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = now
                            t_prev_token = now
                        else:
                            itl_list.append((now - t_prev_token) * 1000)
                            t_prev_token = now
                        output_tokens += 1
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        error = "Request timed out (300s)"
    except Exception as e:
        error = str(e)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000
    ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else total_ms

    return RequestResult(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        itl_ms=itl_list,
        total_latency_ms=total_ms,
        error=error,
    )


async def run_benchmark_requests(
    endpoint: str,
    prompts: list[tuple[str, int, int]],  # (prompt_text, prompt_tokens, max_output_tokens)
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    concurrency: int = 1,
) -> list[RequestResult]:
    """Run benchmark requests with controlled concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async with aiohttp.ClientSession() as session:

        async def bounded_request(idx, prompt, prompt_tokens, max_tokens):
            async with semaphore:
                return await send_streaming_request(
                    session, endpoint, prompt, max_tokens, idx, prompt_tokens, model
                )

        tasks = [
            bounded_request(i, p, pt, mt)
            for i, (p, pt, mt) in enumerate(prompts)
        ]
        results = await asyncio.gather(*tasks)

    return list(results)
