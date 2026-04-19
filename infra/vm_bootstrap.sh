#!/usr/bin/env bash
# =============================================================================
# vm_bootstrap.sh — Idempotent setup for GCP GPU VMs
# Run on: PREFILL VM, DECODE VM, or COLOCATED VM
# Usage:  bash infra/vm_bootstrap.sh
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  AMLIC VM Bootstrap — $(date)"
echo "============================================"

# ── 1. System packages ──────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv git curl wget htop tmux \
    net-tools iperf3 netcat-openbsd jq

# ── 2. Verify NVIDIA drivers + CUDA ─────────────────────────────────────────
echo "[2/6] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    echo "  On GCP Deep Learning VMs, drivers are pre-installed."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "Checking CUDA toolkit..."
if command -v nvcc &>/dev/null; then
    nvcc --version | grep "release"
else
    echo "WARNING: nvcc not found. CUDA toolkit may not be installed."
    echo "  vLLM bundles its own CUDA kernels, so this may be okay."
fi

# ── 3. Python virtual environment ───────────────────────────────────────────
VENV_DIR="$HOME/amlic-venv"
echo "[3/6] Setting up Python venv at $VENV_DIR..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel -q

# ── 4. Install vLLM + NIXL ─────────────────────────────────────────────────
# Pin versions once they work — update these after Phase 3 succeeds.
VLLM_VERSION="vllm"       # e.g., "vllm==0.8.5" once pinned
NIXL_VERSION="nixl"       # e.g., "nixl==0.1.0" once pinned

echo "[4/6] Installing vLLM..."
pip install "$VLLM_VERSION" -q 2>&1 | tail -5

echo "[5/6] Installing NIXL..."
pip install "$NIXL_VERSION" -q 2>&1 | tail -5 || {
    echo "WARNING: NIXL pip install failed. Trying from GitHub..."
    pip install "git+https://github.com/ai-dynamo/nixl.git" -q 2>&1 | tail -5 || {
        echo "ERROR: NIXL install failed. Will need manual build."
        echo "  See: https://github.com/ai-dynamo/nixl"
    }
}

# ── 5. Install additional dependencies ──────────────────────────────────────
echo "[6/6] Installing additional packages..."
pip install httpx fastapi uvicorn -q

# ── 6. Verify installation ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Verification"
echo "============================================"
echo "Python:   $(python3 --version)"
echo "vLLM:     $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "NIXL:     $(python3 -c 'import nixl; print("OK")' 2>/dev/null || echo 'NOT INSTALLED')"
echo "PyTorch:  $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "CUDA:     $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'NOT AVAILABLE')"
echo "GPU:      $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'NOT AVAILABLE')"
echo ""
echo "venv activate:  source $VENV_DIR/bin/activate"
echo "============================================"
echo "  Bootstrap complete!"
echo "============================================"
