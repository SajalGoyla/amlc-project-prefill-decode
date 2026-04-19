#!/usr/bin/env bash
# =============================================================================
# start_collocated.sh — Run vLLM in standard (collocated) mode
# Run on: L4 VM (prefill VM doubles as collocated baseline)
# =============================================================================
set -euo pipefail

# Load env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

# Activate venv
source "$HOME/amlic-venv/bin/activate"

MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${COLLOCATED_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"

echo "============================================"
echo "  Starting COLLOCATED vLLM server"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  MaxLen:   $MAX_MODEL_LEN"
echo "  GPU Util: $GPU_MEM_UTIL"
echo "============================================"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --disable-log-requests
