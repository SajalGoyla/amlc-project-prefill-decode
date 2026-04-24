#!/usr/bin/env bash
# =============================================================================
# start_prefill.sh — Run vLLM as KV producer (prefill engine)
# Run on: L4 VM (PREFILL VM)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

source "$HOME/amlic-venv/bin/activate"

MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PREFILL_PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5600}"

# Get the Tailscale IP of THIS machine (prefill)
PREFILL_IP="${VM_PREFILL_IP:-$(tailscale ip -4 2>/dev/null || echo '0.0.0.0')}"

echo "============================================"
echo "  Starting PREFILL engine (KV producer)"
echo "  Model:          $MODEL"
echo "  Port:           $PORT"
echo "  Side Channel:   $PREFILL_IP:$SIDE_CHANNEL_PORT"
echo "============================================"

# NIXL / UCX environment
export VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_IP"
export VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT"

# Force TCP transport (no RDMA on standard GCP VMs)
export UCX_TLS=tcp
# Bind to Tailscale interface — confirm iface name with `ip addr`
# Common names: tailscale0, or ts0
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-tailscale0}"

KV_TRANSFER_CONFIG='{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer",
    "kv_rank": 0,
    "kv_parallel_size": 2
}'

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
