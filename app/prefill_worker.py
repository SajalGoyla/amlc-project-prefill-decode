"""
Phase 2B: Prefill Worker
========================
FastAPI server that receives prompts from the Gateway, runs the LLM forward
pass, and streams KV-cache tensors layer-by-layer to the Decode Worker over
ZMQ using PyTorch register_forward_hook.

Key Design:
    - Each transformer layer has a forward hook attached.
    - The hook captures the K and V tensors from that layer's output.
    - A background thread moves tensors to CPU (non-blocking) and sends
      them over a ZMQ PUSH socket — this happens DURING the forward pass,
      not after it.
    - After the full forward pass completes, a PREFILL_COMPLETE signal is sent.

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

DECODE_WORKER_HOST = os.getenv("DECODE_WORKER_HOST", "10.128.0.3")
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
# KV-Cache Streaming via Forward Hooks
# ==============================================================================


def create_kv_hook(layer_idx: int, session_id: str, zmq_sock, send_lock: threading.Lock):
    """
    Create a forward hook for a specific transformer layer.

    The hook intercepts the layer's output, extracts Key and Value tensors
    from the attention module's cache, and spawns a background thread to
    serialize and send them over ZMQ.
    """

    def hook_fn(module, input, output):
        """
        Forward hook callback.

        For LLaMA-3 models, the output of each decoder layer is a tuple:
          (hidden_states, present_key_value)
        where present_key_value is a tuple (key_tensor, value_tensor).
        """
        try:
            # output is typically (hidden_states, ) or (hidden_states, present_kv, ...)
            # When use_cache=True, present_key_value is available
            if isinstance(output, tuple) and len(output) >= 2:
                present_kv = output[1]

                if present_kv is not None and isinstance(present_kv, tuple) and len(present_kv) == 2:
                    key_tensor = present_kv[0]
                    value_tensor = present_kv[1]

                    # Spawn a background thread for async CPU transfer + ZMQ send
                    t = threading.Thread(
                        target=_send_kv_layer,
                        args=(
                            layer_idx,
                            session_id,
                            key_tensor,
                            value_tensor,
                            zmq_sock,
                            send_lock,
                        ),
                        daemon=True,
                    )
                    t.start()
                else:
                    logger.warning(
                        "Layer %d: present_kv is None or unexpected format. "
                        "Type: %s",
                        layer_idx,
                        type(present_kv),
                    )
            else:
                logger.warning(
                    "Layer %d: output has unexpected structure. len=%s, type=%s",
                    layer_idx,
                    len(output) if isinstance(output, tuple) else "N/A",
                    type(output),
                )

        except Exception as e:
            logger.error("Hook error at layer %d: %s", layer_idx, e, exc_info=True)

    return hook_fn


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
        logger.debug(
            "Sent KV layer %d for session '%s' "
            "(key=%s, value=%s, payload=%.1f KB, time=%.1f ms)",
            layer_idx,
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


def run_prefill(session_id: str, prompt: str, max_new_tokens: int) -> dict:
    """
    Execute the prefill phase:
      1. Tokenize the prompt.
      2. Attach forward hooks to every transformer layer.
      3. Run the forward pass with use_cache=True.
      4. Send PREFILL_COMPLETE signal.
      5. Clean up hooks.

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

    # Thread-safe lock for ZMQ sends
    send_lock = threading.Lock()

    # Register forward hooks on every transformer layer
    hooks = []
    num_layers = len(model.model.layers)
    for layer_idx, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(
            create_kv_hook(layer_idx, session_id, zmq_socket, send_lock)
        )
        hooks.append(hook)

    logger.info("Registered %d forward hooks.", len(hooks))

    # Run the forward pass
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
        # Clean up hooks even on failure
        for h in hooks:
            h.remove()
        raise

    t_forward_end = time.perf_counter()
    forward_time_ms = (t_forward_end - t_forward_start) * 1000

    # Clean up hooks
    for h in hooks:
        h.remove()
    logger.info("Removed %d forward hooks.", len(hooks))

    # Allow background threads a moment to finish sending
    time.sleep(0.5)

    # Send PREFILL_COMPLETE signal
    complete_payload = {
        "type": "prefill_complete",
        "session_id": session_id,
        "num_layers": num_layers,
        "max_new_tokens": max_new_tokens,
        "num_prompt_tokens": int(input_ids.shape[1]),
        "last_token_id": int(input_ids[0, -1].item()),
        "forward_time_ms": forward_time_ms,
    }
    complete_data = pickle.dumps(complete_payload, protocol=pickle.HIGHEST_PROTOCOL)

    with send_lock:
        zmq_socket.send(complete_data)

    logger.info(
        "Prefill complete for session '%s': %d layers, %.1f ms forward pass, "
        "%d prompt tokens.",
        session_id,
        num_layers,
        forward_time_ms,
        input_ids.shape[1],
    )

    return {
        "session_id": session_id,
        "status": "prefill_complete",
        "num_layers_streamed": num_layers,
        "prefill_time_ms": round(forward_time_ms, 2),
    }


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

    yield

    # Cleanup
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
    Receive a prompt from the Gateway, run the prefill forward pass,
    and stream KV-cache to the Decode Worker.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = run_prefill(
            session_id=request.session_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
        )
        return PrefillResponse(**result)

    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM during prefill for session '%s'", request.session_id)
        torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU out of memory. Try a shorter prompt or reduce max_new_tokens.",
        )
    except Exception as e:
        logger.exception("Prefill failed for session '%s': %s", request.session_id, e)
        raise HTTPException(status_code=500, detail=str(e))


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
