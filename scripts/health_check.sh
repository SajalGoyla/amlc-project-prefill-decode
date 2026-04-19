#!/usr/bin/env bash
# =============================================================================
# health_check.sh — Quick health check for all services
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

check_service() {
    local name="$1"
    local url="$2"
    if curl -sf "$url" --max-time 5 >/dev/null 2>&1; then
        echo "  ✓ $name ($url)"
    else
        echo "  ✗ $name ($url) — DOWN"
    fi
}

echo "============================================"
echo "  AMLIC Service Health Check"
echo "============================================"

# Collocated
check_service "Collocated" "http://${VM_COLOCATED_IP:-localhost}:${COLLOCATED_PORT:-8000}/health"

# Prefill
check_service "Prefill" "http://${VM_PREFILL_IP:-localhost}:${PREFILL_PORT:-8100}/health"

# Decode
check_service "Decode" "http://${VM_DECODE_IP:-localhost}:${DECODE_PORT:-8200}/health"

# Proxy
check_service "Proxy" "http://${VM_PREFILL_IP:-localhost}:${PROXY_PORT:-9000}/health"

# Router (if running locally)
check_service "Router" "http://localhost:7000/health"

echo "============================================"
