#!/usr/bin/env bash
# =============================================================================
# run_full_benchmark.sh — Runs the complete benchmark suite.
# Run on: Laptop (via SSH tunnels or Tailscale connectivity to VMs)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

NUM_REQUESTS="${1:-30}"
CONCURRENCY="${2:-1}"
OUT_DIR="$PROJECT_DIR/benchmark/results"

COLLOCATED="http://${VM_COLOCATED_IP}:${COLLOCATED_PORT}/v1"
DISAGG="http://${VM_PREFILL_IP}:${PROXY_PORT}/v1"

echo "============================================"
echo "  AMLIC Full Benchmark Suite"
echo "  Collocated:     $COLLOCATED"
echo "  Disaggregated:  $DISAGG"
echo "  Requests/profile: $NUM_REQUESTS"
echo "  Concurrency:    $CONCURRENCY"
echo "============================================"

cd "$PROJECT_DIR"

# ── PHASE 1: Quick health check ─────────────────────────────────────────────
echo ""
echo "[1/4] Health check..."
for url in "$COLLOCATED" "$DISAGG"; do
    if curl -sf "${url%/v1}/health" >/dev/null 2>&1; then
        echo "  ✓ $url"
    else
        echo "  ✗ $url — NOT REACHABLE"
        echo "    Make sure the vLLM servers are running!"
        exit 1
    fi
done

# ── PHASE 2: Standard profiles (collocated) ─────────────────────────────────
echo ""
echo "[2/4] Running collocated benchmarks..."
python -m benchmark.runner \
    --endpoint "$COLLOCATED" \
    --arch collocated \
    --profile all \
    --num-requests "$NUM_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --include-sweep \
    --out "$OUT_DIR"

# ── PHASE 3: Standard profiles (disaggregated) ──────────────────────────────
echo ""
echo "[3/4] Running disaggregated benchmarks..."
python -m benchmark.runner \
    --endpoint "$DISAGG" \
    --arch disaggregated \
    --profile all \
    --num-requests "$NUM_REQUESTS" \
    --concurrency "$CONCURRENCY" \
    --include-sweep \
    --out "$OUT_DIR"

# ── PHASE 4: Analysis ───────────────────────────────────────────────────────
echo ""
echo "[4/4] Computing threshold and generating plots..."
python -m analysis.compute_threshold --results-dir "$OUT_DIR"
python -m analysis.plot_results --results-dir "$OUT_DIR" --out "$PROJECT_DIR/analysis/figures"

echo ""
echo "============================================"
echo "  Benchmark suite complete!"
echo "  Results:  $OUT_DIR/"
echo "  Plots:    $PROJECT_DIR/analysis/figures/"
echo "============================================"
