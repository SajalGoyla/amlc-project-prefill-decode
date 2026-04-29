"""
Phase 3: Collocated Baseline Worker
===================================
A single-GPU FastAPI server that loads the LLM natively and performs BOTH
prefill and decode phases sequentially. Used for benchmarking purposes.

Measures and returns exactly:
- compute_ttft_ms  (pure prefill GPU time)
- true_ttft_ms     (includes queue wait time — critical for concurrency comparison)
- TPOT (Time Per Output Token)
"""

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

COLLOCATED_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
COLLOCATED_PORT = int(os.getenv("GATEWAY_PORT", "8000"))

MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", None)

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("collocated_worker")

# ==============================================================================
# Global State
# ==============================================================================

model = None
tokenizer = None
device = None

# Metrics store: { session_id: { status, compute_ttft_ms, true_ttft_ms, tpot_ms, tokens } }
metrics_store = {}

# Arrival timestamps: captured IMMEDIATELY in the endpoint handler
# before any queuing or GPU work. This is the key to correct true_ttft_ms.
arrival_times = {}


def load_model():
    """Load the LLM and tokenizer onto GPU."""
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model '%s' onto device '%s'...", MODEL_NAME, device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
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
    logger.info("Collocated Model loaded successfully.")

# ==============================================================================
# Data Models
# ==============================================================================


class GenerateRequest(BaseModel):
    """Client request payload."""
    prompt: str = Field(..., description="The input prompt for generation")
    session_id: str = Field(..., description="Unique session ID.")
    max_new_tokens: int = Field(default=128, ge=1, le=2048)


class GenerateResponse(BaseModel):
    """Acknowledgment response."""
    session_id: str
    status: str


# ==============================================================================
# Synchronous Generation (runs in thread pool via asyncio.to_thread)
# ==============================================================================


def run_generation_sync(session_id: str, prompt: str, max_new_tokens: int):
    """
    Synchronous generation measuring TTFT and TPOT.
    Runs in a thread pool so the event loop stays responsive.
    """
    # Retrieve the arrival timestamp that was captured in the endpoint handler
    created_at = arrival_times.get(session_id, time.perf_counter())

    logger.info("Starting collocated generation for '%s'", session_id)
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(device)

        # Prefill phase
        t_prefill_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values

        t_prefill_end = time.perf_counter()
        compute_ttft_ms = (t_prefill_end - t_prefill_start) * 1000
        true_ttft_ms = (t_prefill_end - created_at) * 1000

        logger.info(
            "Session '%s': Prefill complete. True TTFT: %.1f ms (Compute: %.1f ms, Queue wait: %.1f ms)",
            session_id, true_ttft_ms, compute_ttft_ms, true_ttft_ms - compute_ttft_ms,
        )

        # Decode Phase
        input_ids = inputs["input_ids"]
        last_token_id = int(input_ids[0, -1].item())
        decode_input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)

        num_prompt_tokens = input_ids.shape[1]
        attention_mask = torch.ones((1, num_prompt_tokens + 1), dtype=torch.long, device=device)

        t_decode_start = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=decode_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t_decode_end = time.perf_counter()

        num_generated = generated_ids.shape[1] - 1  # Excluding the seed token
        decode_time_ms = (t_decode_end - t_decode_start) * 1000
        tpot_ms = decode_time_ms / max(num_generated, 1)

        logger.info("Session '%s': Decode complete. %d tokens | TPOT: %.2f ms", session_id, num_generated, tpot_ms)

        # Store metrics
        metrics_store[session_id] = {
            "status": "complete",
            "compute_ttft_ms": round(compute_ttft_ms, 2),
            "true_ttft_ms": round(true_ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "tokens": num_generated,
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        metrics_store[session_id] = {"status": "error", "error": "OOM"}
    except Exception as e:
        logger.exception("Generation error: %s", e)
        metrics_store[session_id] = {"status": "error", "error": str(e)}
    finally:
        # Clean up arrival time
        arrival_times.pop(session_id, None)


# ==============================================================================
# Background Worker Queue (same pattern as disaggregated prefill_worker)
# ==============================================================================

generation_queue: asyncio.Queue = asyncio.Queue()


async def generation_worker_loop():
    """Background task that processes generation requests sequentially via thread pool."""
    logger.info("Generation worker loop started.")
    while True:
        request_data = await generation_queue.get()
        try:
            await asyncio.to_thread(
                run_generation_sync,
                session_id=request_data["session_id"],
                prompt=request_data["prompt"],
                max_new_tokens=request_data["max_new_tokens"],
            )
        except Exception as e:
            logger.exception("Generation worker error: %s", e)
        finally:
            generation_queue.task_done()


# ==============================================================================
# FastAPI Application
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("=" * 60)
    logger.info("Collocated Baseline API starting")
    logger.info("=" * 60)
    load_model()
    # Start the background worker loop
    asyncio.create_task(generation_worker_loop())
    yield
    logger.info("Collocated API shutting down.")


app = FastAPI(title="Collocated LLM Baseline", lifespan=lifespan)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Accept the request and enqueue for background processing."""
    # Capture arrival time IMMEDIATELY — before any queuing
    arrival_times[request.session_id] = time.perf_counter()
    metrics_store[request.session_id] = {"status": "pending"}

    await generation_queue.put({
        "session_id": request.session_id,
        "prompt": request.prompt,
        "max_new_tokens": request.max_new_tokens,
    })

    logger.info("Enqueued session '%s'. Queue size: %d", request.session_id, generation_queue.qsize())
    return GenerateResponse(session_id=request.session_id, status="queued")

@app.get("/metrics/{session_id}")
async def get_metrics(session_id: str):
    """Endpoint to cleanly fetch the precise internal timings for the comparison script."""
    if session_id not in metrics_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return metrics_store[session_id]


if __name__ == "__main__":
    uvicorn.run("collocated_baseline:app", host=COLLOCATED_HOST, port=COLLOCATED_PORT)
