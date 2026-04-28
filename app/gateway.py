"""
Phase 2A: API Gateway
=====================
FastAPI server that accepts client requests and forwards them to the Prefill
Worker using a fire-and-forget background task. Returns an immediate 200 OK
with the session ID so the client doesn't block.

Endpoints:
    POST /generate  — Submit a generation request
    GET  /health    — Health check
    GET  /status/{session_id} — Query generation status (placeholder)
"""

import os
import uuid
import time
import logging
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

# ==============================================================================
# Configuration
# ==============================================================================

load_dotenv()

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8000"))
# Force localhost since Gateway and Prefill Worker are always collocated on the same VM
PREFILL_WORKER_HOST = "127.0.0.1"
PREFILL_WORKER_PORT = int(os.getenv("PREFILL_WORKER_PORT", "8001"))

PREFILL_URL = f"http://{PREFILL_WORKER_HOST}:{PREFILL_WORKER_PORT}/prefill"

# ==============================================================================
# Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gateway")

# ==============================================================================
# Data Models
# ==============================================================================


class GenerateRequest(BaseModel):
    """Client request payload."""
    prompt: str = Field(..., description="The input prompt for generation")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID. Auto-generated if not provided.",
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate",
    )


class GenerateResponse(BaseModel):
    """Immediate acknowledgment response."""
    session_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    prefill_worker_url: str


# ==============================================================================
# Session Tracking
# ==============================================================================

# In-memory session status store (for demo/benchmarking purposes).
# In production, use Redis or a database.
session_store: dict[str, dict] = {}

# ==============================================================================
# Background Task: Forward to Prefill Worker
# ==============================================================================


async def forward_to_prefill(session_id: str, prompt: str, max_new_tokens: int):
    """
    Fire-and-forget async task that forwards the generation request
    to the Prefill Worker. Updates session status on success/failure.
    """
    payload = {
        "session_id": session_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
    }

    session_store[session_id]["status"] = "forwarding"
    session_store[session_id]["forward_start"] = time.time()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            logger.info(
                "Forwarding session '%s' to Prefill Worker at %s",
                session_id,
                PREFILL_URL,
            )
            response = await client.post(PREFILL_URL, json=payload)

            if response.status_code == 200:
                session_store[session_id]["status"] = "prefill_accepted"
                logger.info(
                    "Prefill Worker accepted session '%s' (HTTP %d)",
                    session_id,
                    response.status_code,
                )
            else:
                session_store[session_id]["status"] = "prefill_error"
                session_store[session_id]["error"] = response.text
                logger.error(
                    "Prefill Worker returned HTTP %d for session '%s': %s",
                    response.status_code,
                    session_id,
                    response.text,
                )

    except httpx.ConnectError as e:
        session_store[session_id]["status"] = "connection_error"
        session_store[session_id]["error"] = str(e)
        logger.error(
            "Cannot reach Prefill Worker at %s for session '%s': %s",
            PREFILL_URL,
            session_id,
            e,
        )
    except Exception as e:
        session_store[session_id]["status"] = "error"
        session_store[session_id]["error"] = str(e)
        logger.exception(
            "Unexpected error forwarding session '%s': %s", session_id, e
        )


# ==============================================================================
# FastAPI Application
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("=" * 60)
    logger.info("API Gateway starting")
    logger.info("  Listening on:       %s:%d", GATEWAY_HOST, GATEWAY_PORT)
    logger.info("  Prefill Worker URL: %s", PREFILL_URL)
    logger.info("=" * 60)
    yield
    logger.info("API Gateway shutting down.")


app = FastAPI(
    title="Disaggregated LLM Serving — API Gateway",
    description=(
        "Accepts generation requests and forwards them to the Prefill Worker. "
        "Returns immediately with a session ID for async tracking."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# Endpoints
# ==============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="api-gateway",
        prefill_worker_url=PREFILL_URL,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Submit a generation request.

    The Gateway immediately returns a 200 OK with the session ID.
    The actual generation is handled asynchronously:
      1. This endpoint fires a background task to forward the request
         to the Prefill Worker.
      2. The Prefill Worker streams KV-cache to the Decode Worker via ZMQ.
      3. The Decode Worker runs token generation.
    """
    # Assign or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Initialize session tracking
    session_store[session_id] = {
        "status": "queued",
        "prompt": request.prompt,
        "max_new_tokens": request.max_new_tokens,
        "created_at": time.time(),
    }

    logger.info(
        "Received generation request — session_id='%s', prompt_len=%d, max_tokens=%d",
        session_id,
        len(request.prompt),
        request.max_new_tokens,
    )

    # Fire-and-forget: forward to Prefill Worker in background
    background_tasks.add_task(
        forward_to_prefill,
        session_id=session_id,
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    return GenerateResponse(
        session_id=session_id,
        status="queued",
        message=(
            f"Request accepted. Generation will proceed asynchronously. "
            f"Session ID: {session_id}"
        ),
    )


@app.get("/status/{session_id}")
async def get_session_status(session_id: str):
    """Query the current status of a generation session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session_store[session_id]


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="info",
        reload=False,
    )
