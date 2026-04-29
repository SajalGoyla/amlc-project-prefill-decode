"""
Phase 2C: Decode Worker
=======================
Loads the same LLM, listens for KV-cache layers over ZMQ, and runs
autoregressive token generation once all layers arrive.

Architecture:
    - ZMQ PULL socket receives KV layer tensors streamed from the Prefill Worker.
    - An in-memory dictionary buffers layers keyed by (session_id, layer_idx).
    - On PREFILL_COMPLETE signal, reconstructs the full past_key_values tuple,
      moves it to GPU, and runs model.generate() for the decode phase.

Optional:
    --test-zmq    Run in ZMQ connectivity test mode (no model loading).
"""

import os
import sys
import time
import pickle
import logging
import argparse
import threading
import queue
from collections import defaultdict

import torch
import zmq
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

DECODE_HOST = os.getenv("DECODE_WORKER_HOST", "0.0.0.0")
DECODE_PORT = int(os.getenv("DECODE_WORKER_PORT", "8002"))
ZMQ_PORT = int(os.getenv("ZMQ_PORT", "5555"))

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)

ZMQ_BIND_ADDR = f"tcp://0.0.0.0:{ZMQ_PORT}"

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("decode_worker")

# ==============================================================================
# Global State
# ==============================================================================

model = None
tokenizer = None
device = None

# In-memory KV cache: { session_id: { layer_idx: (key_tensor, value_tensor) } }
kv_cache_store: dict[str, dict[int, tuple]] = defaultdict(dict)

# Session metadata: { session_id: { ... } }
session_metadata: dict[str, dict] = {}

# Per-session KV receive timing: { session_id: { kv_start: float } }
session_timing: dict[str, dict] = {}


# ==============================================================================
# Model Loading
# ==============================================================================


def load_model():
    """Load the same LLM used by the Prefill Worker."""
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


# ==============================================================================
# KV-Cache Reconstruction
# ==============================================================================


def reconstruct_past_key_values(session_id: str, num_layers: int):
    """
    Reconstruct the past_key_values tuple structure expected by HuggingFace.

    HuggingFace expects:
        past_key_values = (
            (key_layer_0, value_layer_0),
            (key_layer_1, value_layer_1),
            ...
            (key_layer_N, value_layer_N),
        )

    Each key/value tensor has shape: (batch_size, num_heads, seq_len, head_dim)
    """
    session_cache = kv_cache_store.get(session_id, {})

    if len(session_cache) != num_layers:
        logger.warning(
            "Session '%s': Expected %d layers but received %d. "
            "Missing layers: %s",
            session_id,
            num_layers,
            len(session_cache),
            sorted(set(range(num_layers)) - set(session_cache.keys())),
        )

    past_key_values = []
    for layer_idx in range(num_layers):
        if layer_idx not in session_cache:
            logger.error(
                "Session '%s': Missing KV cache for layer %d!",
                session_id,
                layer_idx,
            )
            raise ValueError(
                f"Incomplete KV cache: layer {layer_idx} missing for session {session_id}"
            )

        key_tensor, value_tensor = session_cache[layer_idx]

        # Move to GPU
        key_gpu = key_tensor.to(device, non_blocking=True)
        value_gpu = value_tensor.to(device, non_blocking=True)

        past_key_values.append((key_gpu, value_gpu))

    # Synchronize GPU transfers
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Convert to tuple of tuples (legacy format)
    past_key_values = tuple(past_key_values)
    
    # In newer Transformers (e.g. Llama 3), DynamicCache is required instead of raw tuples.
    try:
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(past_key_values):
            cache.update(k, v, layer_idx)
            
        # Swap tuple for the DynamicCache object
        past_key_values = cache
        logger.info(
            "Reconstructed DynamicCache for session '%s': %d layers (format manually built).",
            session_id,
            num_layers,
        )
    except Exception as e:
        logger.info(
            "Could not build DynamicCache (using legacy tuple fallback) for session '%s': %s",
            session_id,
            e
        )

    return past_key_values


# ==============================================================================
# Token Generation (Decode Phase)
# ==============================================================================


def run_decode(
    session_id: str,
    past_key_values,
    last_token_id: int,
    max_new_tokens: int,
    num_prompt_tokens: int,
    forward_time_ms: float = 0.0,
    kv_receive_time_ms: float = 0.0,
    cache_reconstruct_ms: float = 0.0,
) -> dict:
    """
    Run the autoregressive decode phase using the reconstructed KV cache.

    Args:
        session_id: Unique session identifier.
        past_key_values: Reconstructed KV cache from the prefill phase.
        last_token_id: The last token ID from the prompt (used as decode seed).
        max_new_tokens: Maximum number of new tokens to generate.
        num_prompt_tokens: Number of tokens in the original prompt.

    Returns:
        Dict with generated text and timing information.
    """
    logger.info(
        "Starting decode for session '%s' (max_new_tokens=%d, seed_token=%d)",
        session_id,
        max_new_tokens,
        last_token_id,
    )

    # Construct the initial input_ids for decode.
    # We use the last prompt token as the seed since past_key_values
    # already encodes all prior context.
    input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)

    # Build attention mask: 1 for all prompt tokens + 1 for the new token
    attention_mask = torch.ones(
        (1, num_prompt_tokens + 1), dtype=torch.long, device=device
    )

    t_decode_start = time.perf_counter()
    t_first_token = None

    # Generate tokens
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # Greedy decoding for reproducibility
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    t_decode_end = time.perf_counter()

    # Decode the generated tokens (skip the seed token)
    new_token_ids = generated_ids[0, 1:]  # Remove the seed token
    generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    num_generated = len(new_token_ids)

    decode_time_ms = (t_decode_end - t_decode_start) * 1000
    tpot_ms = decode_time_ms / max(num_generated, 1)

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE — Session '%s'", session_id)
    logger.info("-" * 60)
    logger.info("Generated %d tokens in %.1f ms", num_generated, decode_time_ms)
    logger.info("TPOT: %.2f ms/token", tpot_ms)
    logger.info("Output: %s", generated_text[:200] + ("..." if len(generated_text) > 200 else ""))
    logger.info("=" * 60)
    
    # Store metrics for benchmark comparisons securely
    metrics_store[session_id] = {
        "status":"complete",
        "tpot_ms": round(tpot_ms, 2),
        "decode_time_ms": round(decode_time_ms, 2),
        "forward_time_ms": round(forward_time_ms, 2),
        "kv_receive_time_ms": round(kv_receive_time_ms, 2),
        "cache_reconstruct_ms": round(cache_reconstruct_ms, 2),
        "tokens": num_generated,
        "num_prompt_tokens": num_prompt_tokens,
    }

    return {
        "session_id": session_id,
        "generated_text": generated_text,
        "num_tokens_generated": num_generated,
        "decode_time_ms": round(decode_time_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
    }


# ==============================================================================
# Background Decode Worker
# ==============================================================================

import queue
decode_queue = queue.Queue()

def decode_worker_loop():
    """
    Background thread that pulls from the decode_queue and processes
    requests strictly sequentially. This prevents multiple threads from
    hammering the GPU with concurrent model.generate() calls.
    """
    logger.info("Background decode worker loop started.")
    while True:
        task = decode_queue.get()
        sid = task["session_id"]
        
        try:
            # Reconstruct past_key_values (measure reconstruction time)
            t_recon_start = time.perf_counter()
            past_key_values = reconstruct_past_key_values(sid, task["num_layers"])
            cache_reconstruct_ms = (time.perf_counter() - t_recon_start) * 1000
            logger.info("Session '%s': Cache reconstruction: %.1f ms", sid, cache_reconstruct_ms)
        except ValueError as e:
            logger.error("KV reconstruction failed for session '%s': %s", sid, e)
            if sid in kv_cache_store:
                del kv_cache_store[sid]
            decode_queue.task_done()
            continue

        # Run decode
        try:
            result = run_decode(
                session_id=sid,
                past_key_values=past_key_values,
                last_token_id=task["last_token_id"],
                max_new_tokens=task["max_new_tokens"],
                num_prompt_tokens=task["num_prompt_tokens"],
                forward_time_ms=task["prefill_time_ms"],
                kv_receive_time_ms=task["kv_receive_time_ms"],
                cache_reconstruct_ms=cache_reconstruct_ms,
            )
            logger.info(
                "Decode result for session '%s': %d tokens, %.1f ms total",
                sid,
                result["num_tokens_generated"],
                result["decode_time_ms"],
            )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during decode for session '%s'", sid)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.exception("Decode failed for session '%s': %s", sid, e)
        finally:
            # Clean up session from cache and timing
            if sid in kv_cache_store:
                del kv_cache_store[sid]
            if sid in session_timing:
                del session_timing[sid]
            logger.info("Cleaned up KV cache for session '%s'.", sid)
            decode_queue.task_done()

# ==============================================================================
# Background API Server for Metrics
# ==============================================================================

from fastapi import FastAPI, HTTPException
import uvicorn
import threading

# Metrics store: { session_id: { tpot_ms, tokens, decode_time_ms } }
metrics_store = {}

app = FastAPI(title="Decode Worker Metrics API")

@app.get("/metrics/{session_id}")
async def get_metrics(session_id: str):
    if session_id not in metrics_store:
        raise HTTPException(status_code=404, detail="Session metrics not available")
    return metrics_store[session_id]

def run_metrics_api():
    """Run the metrics FastAPI server in a background thread."""
    uvicorn.run(app, host=DECODE_HOST, port=DECODE_PORT, log_level="error")


# ==============================================================================
# ZMQ Listener Loop
# ==============================================================================



def run_zmq_listener(test_mode: bool = False):
    """
    Main loop: listen on ZMQ PULL socket for KV-cache layers from the
    Prefill Worker.

    Message types:
        - "kv_layer": Buffer the K/V tensors for a specific layer.
        - "prefill_complete": Reconstruct KV cache and run decode.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)

    # High-water mark
    sock.setsockopt(zmq.RCVHWM, 100)

    sock.bind(ZMQ_BIND_ADDR)
    logger.info("ZMQ PULL socket bound to %s — waiting for messages...", ZMQ_BIND_ADDR)

    if test_mode:
        logger.info("Running in TEST MODE — no model loaded.")
        logger.info("Waiting for a test message...")
        msg = sock.recv_string()
        logger.info("Received test message: '%s'", msg)
        logger.info("ZMQ connectivity test PASSED!")
        sock.close()
        ctx.term()
        return

    # Main receive loop
    try:
        while True:
            try:
                data = sock.recv()
                payload = pickle.loads(data)

                msg_type = payload.get("type", "unknown")
                session_id = payload.get("session_id", "unknown")

                if msg_type == "kv_layer":
                    layer_idx = payload["layer_idx"]
                    key_tensor = payload["key"]
                    value_tensor = payload["value"]

                    # Track when first KV layer arrives for this session
                    if session_id not in session_timing:
                        session_timing[session_id] = {"kv_start": time.perf_counter()}

                    # Buffer in memory
                    kv_cache_store[session_id][layer_idx] = (key_tensor, value_tensor)

                    logger.debug(
                        "Received KV layer %d for session '%s' "
                        "(key=%s, value=%s). Buffered %d/%s layers.",
                        layer_idx,
                        session_id,
                        payload.get("key_shape"),
                        payload.get("value_shape"),
                        len(kv_cache_store[session_id]),
                        "?",
                    )

                elif msg_type == "prefill_complete":
                    # Calculate KV receive time (network transfer)
                    kv_receive_time_ms = 0.0
                    if session_id in session_timing:
                        kv_receive_time_ms = (time.perf_counter() - session_timing[session_id]["kv_start"]) * 1000
                        logger.info("Session '%s': KV receive time (network): %.1f ms", session_id, kv_receive_time_ms)

                    num_layers = payload["num_layers"]
                    max_new_tokens = payload["max_new_tokens"]
                    num_prompt_tokens = payload["num_prompt_tokens"]
                    last_token_id = payload["last_token_id"]
                    prefill_time_ms = payload.get("forward_time_ms", 0)

                    logger.info(
                        "Received PREFILL_COMPLETE for session '%s': "
                        "%d layers, %d prompt tokens, prefill=%.1f ms",
                        session_id,
                        num_layers,
                        num_prompt_tokens,
                        prefill_time_ms,
                    )

                    # Check we have all layers
                    received_layers = len(kv_cache_store.get(session_id, {}))
                    if received_layers < num_layers:
                        logger.warning(
                            "Session '%s': Only %d/%d layers received. "
                            "Waiting briefly for stragglers...",
                            session_id,
                            received_layers,
                            num_layers,
                        )
                        # Brief wait for any in-flight layers
                        time.sleep(1.0)
                        received_layers = len(kv_cache_store.get(session_id, {}))

                    if received_layers < num_layers:
                        logger.error(
                            "Session '%s': Still only %d/%d layers after waiting. "
                            "Aborting decode.",
                            session_id,
                            received_layers,
                            num_layers,
                        )
                        # Clean up
                        if session_id in kv_cache_store:
                            del kv_cache_store[session_id]
                        continue

                    # Put the decode task into the sequential queue
                    # This frees the ZMQ listener to keep buffering incoming layers
                    decode_queue.put({
                        "session_id": session_id,
                        "num_layers": num_layers,
                        "max_new_tokens": max_new_tokens,
                        "num_prompt_tokens": num_prompt_tokens,
                        "last_token_id": last_token_id,
                        "prefill_time_ms": prefill_time_ms,
                        "kv_receive_time_ms": kv_receive_time_ms,
                    })
                    logger.info("Enqueued decode job for session '%s'. Queue size: %d", session_id, decode_queue.qsize())

                else:
                    logger.warning(
                        "Unknown message type '%s' for session '%s'",
                        msg_type,
                        session_id,
                    )

            except pickle.UnpicklingError as e:
                logger.error("Failed to unpickle message: %s", e)
            except KeyError as e:
                logger.error("Malformed message — missing key: %s", e)
            except Exception as e:
                logger.exception("Error processing ZMQ message: %s", e)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    finally:
        sock.close()
        ctx.term()
        logger.info("ZMQ socket closed. Decode Worker stopped.")


# ==============================================================================
# Entry Point
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Decode Worker — Receives KV cache via ZMQ and generates tokens."
    )
    parser.add_argument(
        "--test-zmq",
        action="store_true",
        help="Run in ZMQ connectivity test mode (no model loading).",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Decode Worker starting")
    logger.info("  Model: %s", MODEL_NAME)
    logger.info("  ZMQ bind: %s", ZMQ_BIND_ADDR)
    logger.info("  Test mode: %s", args.test_zmq)
    logger.info("=" * 60)

    if not args.test_zmq:
        load_model()
        
        # Start background metrics API server
        api_t = threading.Thread(target=run_metrics_api, daemon=True)
        api_t.start()
        logger.info("Background REST API thread started on port %d", DECODE_PORT)

        # Start the dedicated sequential decode thread
        decode_t = threading.Thread(target=decode_worker_loop, daemon=True)
        decode_t.start()

    run_zmq_listener(test_mode=args.test_zmq)


if __name__ == "__main__":
    main()
