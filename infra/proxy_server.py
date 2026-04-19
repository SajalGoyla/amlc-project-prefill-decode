"""
proxy_server.py — Disaggregated prefill/decode proxy.

Adapted from vLLM's tests/v1/kv_connector/nixl_integration/toy_proxy_server.py.
Receives requests, sends to prefill (max_tokens=1) to trigger KV cache production,
then forwards to decode for full generation with KV cache consumption.

Run on: PREFILL VM (or any VM that can reach both prefill and decode)
Usage:
    python infra/proxy_server.py \
        --prefiller-host <PREFILL_TAILSCALE_IP> --prefiller-port 8100 \
        --decoder-host <DECODE_TAILSCALE_IP> --decoder-port 8200 \
        --port 9000
"""

import argparse
import itertools
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("amlic.proxy")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize HTTP client pools for prefill and decode services."""
    app.state.prefill_clients = []
    app.state.decode_clients = []

    for i, (host, port) in enumerate(global_args.prefiller_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.prefill_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=base_url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    for i, (host, port) in enumerate(global_args.decoder_instances):
        base_url = f"http://{host}:{port}/v1"
        app.state.decode_clients.append({
            "client": httpx.AsyncClient(
                timeout=None,
                base_url=base_url,
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                ),
            ),
            "host": host,
            "port": port,
            "id": i,
        })

    app.state.prefill_iter = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iter = itertools.cycle(range(len(app.state.decode_clients)))

    logger.info(
        "Proxy ready: %d prefill, %d decode instances",
        len(app.state.prefill_clients),
        len(app.state.decode_clients),
    )
    yield

    for c in app.state.prefill_clients:
        await c["client"].aclose()
    for c in app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan, title="AMLIC Disagg Proxy")


def get_next_client(app_state, service_type: str):
    if service_type == "prefill":
        idx = next(app_state.prefill_iter)
        return app_state.prefill_clients[idx]
    else:
        idx = next(app_state.decode_iter)
        return app_state.decode_clients[idx]


async def send_prefill_request(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """Send to prefill with max_tokens=1, get back kv_transfer_params."""
    req_data = req_data.copy()
    req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]

    # These args are not supported for prefill
    min_tokens = req_data.pop("min_tokens", None)
    min_completion_tokens = req_data.pop("min_completion_tokens", None)

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        "X-Request-Id": request_id,
    }

    t0 = time.perf_counter()
    response = await client_info["client"].post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    await response.aread()
    prefill_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Prefill done: request=%s, prefill_ms=%.1f, instance=%s:%s",
        request_id[:8],
        prefill_ms,
        client_info["host"],
        client_info["port"],
    )

    # Restore min_tokens for decode
    req_data["min_tokens"] = min_tokens
    req_data["min_completion_tokens"] = min_completion_tokens

    return response, prefill_ms


async def stream_decode_response(
    client_info: dict, endpoint: str, req_data: dict, request_id: str
):
    """Stream response from decode service."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'dummy')}",
        "X-Request-Id": request_id,
    }
    async with client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    """Main disaggregated serving flow: prefill → KV transfer → decode."""
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        # 1. Send to prefill
        prefill_client = get_next_client(request.app.state, "prefill")
        response, prefill_ms = await send_prefill_request(
            prefill_client, api, req_data, request_id
        )

        # 2. Extract KV transfer params
        response_json = response.json()
        await response.aclose()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # 3. Stream from decode
        decode_client = get_next_client(request.app.state, "decode")
        logger.info(
            "Routing to decode: request=%s, decode=%s:%s",
            request_id[:8],
            decode_client["host"],
            decode_client["port"],
        )

        async def generate():
            async for chunk in stream_decode_response(
                decode_client, api, req_data, request_id
            ):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")

    except Exception as e:
        import traceback
        logger.error("Proxy error on %s: %s\n%s", api, e, traceback.format_exc())
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/health")
@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="AMLIC Disaggregated Proxy Server")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--prefiller-host", "--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-port", "--prefiller-ports", type=int, nargs="+", default=[8100])
    parser.add_argument("--decoder-host", "--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-port", "--decoder-ports", type=int, nargs="+", default=[8200])
    args = parser.parse_args()

    if len(args.prefiller_host) != len(args.prefiller_port):
        raise ValueError("Number of prefiller hosts must match ports")
    if len(args.decoder_host) != len(args.decoder_port):
        raise ValueError("Number of decoder hosts must match ports")

    args.prefiller_instances = list(zip(args.prefiller_host, args.prefiller_port))
    args.decoder_instances = list(zip(args.decoder_host, args.decoder_port))
    return args


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn
    logger.info("Starting proxy on %s:%d", global_args.host, global_args.port)
    uvicorn.run(app, host=global_args.host, port=global_args.port)
