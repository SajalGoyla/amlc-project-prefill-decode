#!/usr/bin/env bash
# =============================================================================
# start_prefill_docker.sh — Run vLLM (containerized) as KV producer
# Run on: L4 VM (PREFILL VM)
# Uses vllm/vllm-openai image which has NIXL + UCX-CUDA prebuilt.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi

MODEL="${MODEL_NAME:-meta-llama/Llama-3.2-3B-Instruct}"
PORT="${PREFILL_PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"   # match decode (T4 is the constraint)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SIDE_CHANNEL_PORT="${NIXL_SIDE_CHANNEL_PORT:-5600}"
PREFILL_IP="${VM_PREFILL_IP:-$(tailscale ip -4 2>/dev/null || echo '0.0.0.0')}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
CONTAINER_NAME="amlic-prefill"

echo "============================================"
echo "  Starting PREFILL container (KV producer)"
echo "  Image:        vllm/vllm-openai:latest"
echo "  Model:        $MODEL"
echo "  Port:         $PORT"
echo "  Side Channel: $PREFILL_IP:$SIDE_CHANNEL_PORT"
echo "  HF cache:     $HF_CACHE"
echo "============================================"

KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_connector_extra_config":{"enforce_handshake_compat":false}}'

sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

sudo docker run -d --name "$CONTAINER_NAME" \
    --gpus all \
    --network host \
    --ipc host \
    -v "$HF_CACHE":/root/.cache/huggingface \
    -e HF_TOKEN="$HF_TOKEN" \
    -e VLLM_NIXL_SIDE_CHANNEL_HOST="$PREFILL_IP" \
    -e VLLM_NIXL_SIDE_CHANNEL_PORT="$SIDE_CHANNEL_PORT" \
    -e UCX_TLS=tcp \
    -e UCX_NET_DEVICES="${UCX_NET_DEVICES:-tailscale0}" \
    -e VLLM_ATTENTION_BACKEND=FLASHINFER \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype float16 \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"

echo ""
echo "Container started. Tail logs with:"
echo "  sudo docker logs -f $CONTAINER_NAME"
