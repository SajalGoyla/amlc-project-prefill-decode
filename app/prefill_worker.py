"""
Phase 2B: Prefill Worker
========================
FastAPI server that receives prompts from the Gateway, runs the LLM forward
pass, and streams KV-cache tensors layer-by-layer to the Decode Worker over
ZMQ.

Key Design:
    - Run the forward pass with use_cache=True.
    - Extract past_key_values from the model output (all layers at once).
    - Stream each layer's K/V tensors over ZMQ in background threads.
    - Send PREFILL_COMPLETE signal when all layers are queued.

Endpoints:
    POST /prefill  — Receive a prompt and run prefill
    GET  /health   — Health check
"""

import os
import sys
import time
import pickle
import logging
import threading
import asyncio
from contextlib import asynccontextmanager

import torch
import zmq
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

PREFILL_HOST = os.getenv("PREFILL_WORKER_HOST", "0.0.0.0")
PREFILL_PORT = int(os.getenv("PREFILL_WORKER_PORT", "8001"))

# Fallback to the actual Decode VM Internal IP if .env is missing on the VM
DECODE_WORKER_HOST = os.getenv("DECODE_WORKER_HOST", "10.20.0.2")
ZMQ_PORT = int(os.getenv("ZMQ_PORT", "5555"))

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)

ZMQ_ENDPOINT = f"tcp://{DECODE_WORKER_HOST}:{ZMQ_PORT}"

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prefill_worker")

# ==============================================================================
# Data Models
# ==============================================================================


class PrefillRequest(BaseModel):
    """Incoming request from the Gateway."""
    session_id: str
    prompt: str
    max_new_tokens: int = Field(default=128, ge=1, le=2048)


class PrefillResponse(BaseModel):
    """Acknowledgment response."""
    session_id: str
    status: str
    num_layers_streamed: int
    prefill_time_ms: float
    true_ttft_ms: float


# ==============================================================================
# Global State
# ==============================================================================

model = None
tokenizer = None
zmq_context = None
zmq_socket = None
device = None


def load_model():
    """Load the LLM and tokenizer onto GPU."""
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model '%s' onto device '%s'...", MODEL_NAME, device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = len(model.model.layers)
    logger.info(
        "Model loaded successfully. %d transformer layers, dtype=%s",
        num_layers,
        model.dtype,
    )
    return model, tokenizer


def setup_zmq():
    """Initialize the ZMQ PUSH socket."""
    global zmq_context, zmq_socket

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUSH)

    # High-water mark: buffer up to 100 messages before blocking
    zmq_socket.setsockopt(zmq.SNDHWM, 100)
    # Linger: wait up to 5s for pending messages on close
    zmq_socket.setsockopt(zmq.LINGER, 5000)

    zmq_socket.connect(ZMQ_ENDPOINT)
    logger.info("ZMQ PUSH socket connected to %s", ZMQ_ENDPOINT)


# ==============================================================================
# KV-Cache Streaming (post forward pass)
# ==============================================================================


def _send_kv_layer(
    layer_idx: int,
    session_id: str,
    key_tensor: torch.Tensor,
    value_tensor: torch.Tensor,
    zmq_sock,
    send_lock: threading.Lock,
):
    """
    Background thread function: move K/V tensors to CPU (non-blocking),
    serialize, and send over ZMQ.
    """
    try:
        t_start = time.perf_counter()

        # Non-blocking transfer to CPU
        key_cpu = key_tensor.detach().to("cpu", non_blocking=True)
        value_cpu = value_tensor.detach().to("cpu", non_blocking=True)

        # Synchronize to ensure the transfer is complete before pickling
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Create payload
        payload = {
            "type": "kv_layer",
            "session_id": session_id,
            "layer_idx": layer_idx,
            "key": key_cpu,
            "value": value_cpu,
            "key_shape": list(key_cpu.shape),
            "value_shape": list(value_cpu.shape),
        }

        # Serialize and send (thread-safe)
        data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        with send_lock:
            zmq_sock.send(data, flags=zmq.NOBLOCK)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Sent KV layer %d/%d for session '%s' "
            "(key=%s, value=%s, %.1f KB, %.1f ms)",
            layer_idx,
            31,  # 0-indexed, total will be logged at the end
            session_id,
            list(key_cpu.shape),
            list(value_cpu.shape),
            len(data) / 1024,
            elapsed_ms,
        )

    except zmq.Again:
        logger.warning(
            "ZMQ send buffer full for layer %d, session '%s'. Dropping.",
            layer_idx,
            session_id,
        )
    except Exception as e:
        logger.error(
            "Failed to send KV layer %d for session '%s': %s",
            layer_idx,
            session_id,
            e,
            exc_info=True,
        )


# ==============================================================================
# Core Prefill Logic
# ==============================================================================


def run_prefill(session_id: str, prompt: str, max_new_tokens: int, created_at: float = 0.0) -> dict:
    """
    Execute the prefill phase:
      1. Tokenize the prompt.
      2. Run the forward pass with use_cache=True.
      3. Extract past_key_values from model output.
      4. Stream each layer's K/V tensors over ZMQ.
      5. Send PREFILL_COMPLETE signal.

    Returns a summary dict with timing information.
    """
    global model, tokenizer, zmq_socket

    logger.info(
        "Starting prefill for session '%s' (prompt_len=%d chars, max_new_tokens=%d)",
        session_id,
        len(prompt),
        max_new_tokens,
    )

    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)
    input_ids = inputs["input_ids"]

    logger.info(
        "Tokenized: %d tokens, input_ids shape=%s",
        input_ids.shape[1],
        list(input_ids.shape),
    )

    # Run the forward pass — this is the actual prefill computation
    t_forward_start = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
    except Exception as e:
        logger.error("Forward pass failed for session '%s': %s", session_id, e)
        raise

    t_forward_end = time.perf_counter()
    forward_time_ms = (t_forward_end - t_forward_start) * 1000
    
    # Calculate True TTFT (if created_at is provided, else just fallback to forward_time)
    start_time = created_at if created_at > 0 else t_forward_start
    true_ttft_ms = (t_forward_end - start_time) * 1000

    # =========================================================================
    # Extract past_key_values from model output
    # =========================================================================
    # In HuggingFace Transformers, outputs.past_key_values is either:
    #   - A tuple of tuples: ((k0, v0), (k1, v1), ..., (kN, vN))
    #   - A DynamicCache object (newer versions) with .key_cache / .value_cache
    # We handle both cases.
    # =========================================================================

    past_kv = outputs.past_key_values
    
    if past_kv is None:
        logger.error("No past_key_values in model output! use_cache may not be working.")
        raise RuntimeError("Model did not return past_key_values")

    # Extract KV cache — works for both legacy tuple-of-tuples and new DynamicCache
    try:
        kv_pairs = list(past_kv)
        num_layers = len(kv_pairs)
        logger.info("Extracted past_key_values (type: %s) with %d layers", type(past_kv).__name__, num_layers)
    except Exception as e:
        logger.error("Failed to parse past_key_values of type %s: %s", type(past_kv), e)
        raise RuntimeError(f"Unknown past_key_values format: {type(past_kv)}")

    logger.info(
        "KV cache shape per layer: key=%s, value=%s",
        list(kv_pairs[0][0].shape),
        list(kv_pairs[0][1].shape),
    )

    # =========================================================================
    # Stream KV layers over ZMQ (in background threads for parallelism)
    # =========================================================================

    send_lock = threading.Lock()
    threads = []

    t_stream_start = time.perf_counter()

    for layer_idx, layer_data in enumerate(kv_pairs):
        key_tensor = layer_data[0]
        value_tensor = layer_data[1]
        
        t = threading.Thread(
            target=_send_kv_layer,
            args=(
                layer_idx,
                session_id,
                key_tensor,
                value_tensor,
                zmq_socket,
                send_lock,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Wait for all send threads to complete
    for t in threads:
        t.join(timeout=30.0)
        if t.is_alive():
            logger.warning("Send thread still alive after 30s timeout")

    t_stream_end = time.perf_counter()
    stream_time_ms = (t_stream_end - t_stream_start) * 1000

    logger.info(
        "All %d KV layers sent over ZMQ in %.1f ms",
        num_layers,
        stream_time_ms,
    )

    # Send PREFILL_COMPLETE signal
    complete_payload = {
        "type": "prefill_complete",
        "session_id": session_id,
        "num_layers": num_layers,
        "max_new_tokens": max_new_tokens,
        "num_prompt_tokens": int(input_ids.shape[1]),
        "last_token_id": int(input_ids[0, -1].item()),
        "forward_time_ms": forward_time_ms,
        "true_ttft_ms": true_ttft_ms,
    }
    complete_data = pickle.dumps(complete_payload, protocol=pickle.HIGHEST_PROTOCOL)

    with send_lock:
        zmq_socket.send(complete_data)

    total_time_ms = forward_time_ms + stream_time_ms

    logger.info(
        "Prefill complete for session '%s': %d layers, "
        "forward=%.1f ms, stream=%.1f ms, total=%.1f ms, "
        "%d prompt tokens.",
        session_id,
        num_layers,
        forward_time_ms,
        stream_time_ms,
        total_time_ms,
        input_ids.shape[1],
    )

    return {
        "session_id": session_id,
        "status": "prefill_complete",
        "num_layers_streamed": num_layers,
        "prefill_time_ms": round(total_time_ms, 2),
        "true_ttft_ms": round(true_ttft_ms, 2),
    }


# ==============================================================================
# Background Worker Queue
# ==============================================================================

prefill_queue: asyncio.Queue = asyncio.Queue()

async def prefill_worker_loop():
    """Background task that processes prefill requests sequentially."""
    logger.info("Prefill worker loop started.")
    while True:
        request_data = await prefill_queue.get()
        session_id = request_data["session_id"]
        prompt = request_data["prompt"]
        max_new_tokens = request_data["max_new_tokens"]
        created_at = request_data["created_at"]

        try:
            logger.info("Processing queued prefill for session '%s'", session_id)
            # Run the blocking PyTorch compute in a background thread to keep event loop free
            await asyncio.to_thread(
                run_prefill,
                session_id=session_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                created_at=created_at,
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during prefill for session '%s'", session_id)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception("Prefill failed for session '%s': %s", session_id, e)
        finally:
            prefill_queue.task_done()

# ==============================================================================
# FastAPI Application
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and set up ZMQ on startup."""
    logger.info("=" * 60)
    logger.info("Prefill Worker starting")
    logger.info("  Model: %s", MODEL_NAME)
    logger.info("  ZMQ target: %s", ZMQ_ENDPOINT)
    logger.info("=" * 60)

    load_model()
    setup_zmq()

    # Start the background prefill processing loop
    worker_task = asyncio.create_task(prefill_worker_loop())

    yield

    # Cleanup
    worker_task.cancel()
    logger.info("Prefill Worker shutting down.")
    if zmq_socket:
        zmq_socket.close()
    if zmq_context:
        zmq_context.term()


app = FastAPI(
    title="Disaggregated LLM Serving — Prefill Worker",
    description="Runs the prefill forward pass and streams KV-cache via ZMQ.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "service": "prefill-worker",
        "model": MODEL_NAME,
        "device": str(device) if device else "not loaded",
        "zmq_target": ZMQ_ENDPOINT,
    }


@app.post("/prefill", response_model=PrefillResponse)
async def prefill_endpoint(request: PrefillRequest):
    """
    Receive a prompt from the Gateway, enqueue it for processing,
    and return immediately so the Gateway is not blocked.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Enqueue the request
    await prefill_queue.put({
        "session_id": request.session_id,
        "prompt": request.prompt,
        "max_new_tokens": request.max_new_tokens,
        "created_at": time.perf_counter(),
    })

    logger.info("Enqueued session '%s'. Queue size: %d", request.session_id, prefill_queue.qsize())

    # Return immediately
    return PrefillResponse(
        session_id=request.session_id,
        status="queued",
        num_layers_streamed=0,
        prefill_time_ms=0.0,
        true_ttft_ms=0.0,
    )


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "prefill_worker:app",
        host=PREFILL_HOST,
        port=PREFILL_PORT,
        log_level="info",
        reload=False,
    )
