"""
router.py — Adaptive Router (FastAPI).

Routes requests to either the collocated or disaggregated endpoint
based on the prompt length threshold N.

Usage:
    uvicorn router.router:app --host 0.0.0.0 --port 7000
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("amlic.router")


def get_threshold() -> int:
    """Get the routing threshold from env, default 500."""
    return int(os.environ.get("ROUTING_THRESHOLD", "500"))


def get_endpoints() -> dict:
    """Get endpoint URLs from environment."""
    return {
        "collocated": os.environ.get("COLLOCATED_URL", "http://localhost:8000/v1"),
        "disaggregated": os.environ.get("DISAGG_PROXY_URL", "http://localhost:9000/v1"),
    }


# Lazy tokenizer
_tokenizer = None

def count_tokens(text: str) -> int:
    """Count tokens using the model tokenizer (cached)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        model = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
        _tokenizer = AutoTokenizer.from_pretrained(
            model, token=os.environ.get("HF_TOKEN")
        )
    return len(_tokenizer.encode(text))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize HTTP clients."""
    endpoints = get_endpoints()
    app.state.clients = {}
    for arch, url in endpoints.items():
        app.state.clients[arch] = httpx.AsyncClient(
            timeout=None,
            base_url=url,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
        )
    app.state.route_stats = {"collocated": 0, "disaggregated": 0}

    logger.info("Router ready: threshold=%d tokens", get_threshold())
    for arch, url in endpoints.items():
        logger.info("  %s → %s", arch, url)
    yield

    for client in app.state.clients.values():
        await client.aclose()


app = FastAPI(
    lifespan=lifespan,
    title="AMLIC Adaptive Router",
    description="Routes LLM requests based on prompt length threshold",
)


def estimate_prompt_length(req_data: dict) -> int:
    """Estimate the prompt token count from the request body."""
    # Chat completion
    messages = req_data.get("messages", [])
    if messages:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return count_tokens(text)

    # Completion
    prompt = req_data.get("prompt", "")
    if isinstance(prompt, str):
        return count_tokens(prompt)
    elif isinstance(prompt, list):
        return sum(count_tokens(p) if isinstance(p, str) else 1 for p in prompt)

    return 0


def select_arch(prompt_tokens: int) -> str:
    """Select architecture based on threshold."""
    threshold = get_threshold()
    if prompt_tokens >= threshold:
        return "disaggregated"
    return "collocated"


async def proxy_request(request: Request, api_path: str):
    """Route and proxy the request."""
    req_data = await request.json()
    prompt_tokens = estimate_prompt_length(req_data)
    arch = select_arch(prompt_tokens)

    app.state.route_stats[arch] += 1
    logger.info(
        "Route: tokens=%d, arch=%s, total_routed=%s",
        prompt_tokens, arch, app.state.route_stats,
    )

    client = app.state.clients.get(arch)
    if not client:
        return {"error": f"No client configured for {arch}"}

    # Add routing metadata header
    headers = {
        "X-AMLIC-Route": arch,
        "X-AMLIC-Tokens": str(prompt_tokens),
    }

    # Stream the response
    async def stream():
        async with client.stream("POST", api_path, json=req_data, headers=headers) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    return StreamingResponse(stream(), media_type="application/json")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request(request, "/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request):
    return await proxy_request(request, "/completions")


@app.get("/health")
async def health():
    threshold = get_threshold()
    endpoints = get_endpoints()
    return {
        "status": "ok",
        "threshold": threshold,
        "endpoints": endpoints,
        "stats": app.state.route_stats,
    }


@app.get("/stats")
async def stats():
    return {
        "threshold": get_threshold(),
        "route_stats": app.state.route_stats,
    }
