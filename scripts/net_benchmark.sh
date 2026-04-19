#!/usr/bin/env bash
# =============================================================================
# net_benchmark.sh — Network bandwidth/latency test between VMs
# Run on: PREFILL VM (tests connectivity to DECODE VM)
# Usage:  bash scripts/net_benchmark.sh <DECODE_TAILSCALE_IP>
# =============================================================================
set -euo pipefail

DECODE_IP="${1:?Usage: $0 <DECODE_IP>}"
DURATION="${2:-10}"

echo "============================================"
echo "  Tailscale Network Benchmark"
echo "  Source:   $(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}')"
echo "  Target:   $DECODE_IP"
echo "============================================"

# ── 1. Ping latency ─────────────────────────────────────────────────────────
echo ""
echo "[1/3] Ping latency (20 packets)..."
ping -c 20 "$DECODE_IP" 2>/dev/null | tail -1 || echo "  Ping failed"

# ── 2. TCP bandwidth (iperf3) ────────────────────────────────────────────────
echo ""
echo "[2/3] TCP bandwidth (${DURATION}s)..."
echo "  NOTE: Start iperf3 server on decode VM first:"
echo "    iperf3 -s -B $DECODE_IP"
echo ""
iperf3 -c "$DECODE_IP" -t "$DURATION" -J 2>/dev/null | \
    python3 -c "
import json, sys
data = json.load(sys.stdin)
end = data.get('end', {})
sent = end.get('sum_sent', {})
received = end.get('sum_received', {})
print(f'  Sent:     {sent.get(\"bits_per_second\", 0)/1e9:.2f} Gbps')
print(f'  Received: {received.get(\"bits_per_second\", 0)/1e9:.2f} Gbps')
" 2>/dev/null || echo "  iperf3 failed — is the server running on $DECODE_IP?"

# ── 3. Tailscale status ─────────────────────────────────────────────────────
echo ""
echo "[3/3] Tailscale connection info..."
tailscale status 2>/dev/null || echo "  Tailscale not running"
echo ""
tailscale ping "$DECODE_IP" --c 3 2>/dev/null || echo "  Tailscale ping failed"

echo ""
echo "============================================"
echo "  Network benchmark complete"
echo "============================================"
