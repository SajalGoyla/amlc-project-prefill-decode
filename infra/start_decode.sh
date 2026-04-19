#!/usr/bin/env bash
# =============================================================================
# start_decode.sh — Run vLLM as KV consumer (decode engine)
# Run on: T4 VM (DECODE VM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

source "$HOME/amlic-venv/bin/activate"

MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${DECODE_PORT:-8200}"
# T4 has 16 GB — use lower max-model-len to leave room for KV cache
MAX_MODEL_LEN="${MAX_MODEL_LEN_DECODE:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL_DECODE:-0.85}"
SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5600}"

# Get the Tailscale IP of THIS machine (decode)
DECODE_IP="${VM_DECODE_IP:-$(tailscale ip -4 2>/dev/null || echo '0.0.0.0')}"

echo "============================================"
echo "  Starting DECODE engine (KV consumer)"
echo "  Model:          $MODEL"
echo "  Port:           $PORT"
echo "  MaxLen:         $MAX_MODEL_LEN"
echo "  Side Channel:   $DECODE_IP:$SIDE_CHANNEL_PORT"
echo "============================================"

# NIXL / UCX environment
export VLLM_NIXL_SIDE_CHANNEL_HOST="$DECODE_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"

export UCX_TLS=tcp
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-tailscale0}"

KV_TRANSFER_CONFIG='{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer",
    "kv_rank": 1,
    "kv_parallel_size": 2
}'

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --disable-log-requests
