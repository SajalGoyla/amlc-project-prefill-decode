#!/usr/bin/env bash
# ==============================================================================
# Cloud NAT Setup for VMs Without External IPs
# ==============================================================================
# VMs created with --no-address (due to Columbia org policy) cannot reach
# the internet. Cloud NAT provides outbound internet access (for git clone,
# pip install, apt-get, HuggingFace model downloads) without needing an
# external IP on the VM.
#
# This script creates:
#   1. A Cloud Router in each project/region
#   2. A Cloud NAT gateway attached to each router
#
# Usage:
#   chmod +x infra/setup_nat.sh
#   bash infra/setup_nat.sh
#
# Run this AFTER setup_gcp.sh and BEFORE SSHing into VMs to install deps.
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION — Must match setup_gcp.sh values
# ==============================================================================

# Source the .env file if it exists (to get auto-discovered values)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${REPO_ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
    echo "[INFO] Loading configuration from ${ENV_FILE}"
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
else
    echo "[WARN] No .env file found at ${ENV_FILE}. Using defaults."
fi

# Project IDs (from .env or defaults)
PREFILL_PROJECT="${PREFILL_PROJECT:-coms-6998-amlc}"
DECODE_PROJECT="${DECODE_PROJECT:-coms-6998-hpml-487106}"

# Region (from .env or default)
PREFILL_REGION="${PREFILL_REGION:-us-east4}"
DECODE_REGION="${DECODE_REGION:-us-east4}"

# VPC names (must match setup_gcp.sh)
PREFILL_VPC="prefill-vpc"
DECODE_VPC="decode-vpc"

# Cloud Router names
PREFILL_ROUTER="prefill-router"
DECODE_ROUTER="decode-router"

# Cloud NAT names
PREFILL_NAT="prefill-nat"
DECODE_NAT="decode-nat"

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log()   { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

# ==============================================================================
# STEP 1: Create Cloud Routers
# ==============================================================================

create_routers() {
    log "=== Creating Cloud Routers ==="

    # Prefill project router
    log "Creating router '${PREFILL_ROUTER}' in ${PREFILL_REGION} (project: ${PREFILL_PROJECT})..."
    gcloud compute routers create "${PREFILL_ROUTER}" \
        --project="${PREFILL_PROJECT}" \
        --region="${PREFILL_REGION}" \
        --network="${PREFILL_VPC}" \
        2>/dev/null \
        || warn "Router '${PREFILL_ROUTER}' may already exist, continuing..."

    # Decode project router
    log "Creating router '${DECODE_ROUTER}' in ${DECODE_REGION} (project: ${DECODE_PROJECT})..."
    gcloud compute routers create "${DECODE_ROUTER}" \
        --project="${DECODE_PROJECT}" \
        --region="${DECODE_REGION}" \
        --network="${DECODE_VPC}" \
        2>/dev/null \
        || warn "Router '${DECODE_ROUTER}' may already exist, continuing..."

    log "Cloud Routers created."
}

# ==============================================================================
# STEP 2: Create Cloud NAT Gateways
# ==============================================================================

create_nats() {
    log "=== Creating Cloud NAT Gateways ==="

    # Prefill NAT
    log "Creating NAT '${PREFILL_NAT}' on router '${PREFILL_ROUTER}'..."
    gcloud compute routers nats create "${PREFILL_NAT}" \
        --project="${PREFILL_PROJECT}" \
        --router="${PREFILL_ROUTER}" \
        --region="${PREFILL_REGION}" \
        --auto-allocate-nat-external-ips \
        --nat-all-subnet-ip-ranges \
        2>/dev/null \
        || warn "NAT '${PREFILL_NAT}' may already exist, continuing..."

    # Decode NAT
    log "Creating NAT '${DECODE_NAT}' on router '${DECODE_ROUTER}'..."
    gcloud compute routers nats create "${DECODE_NAT}" \
        --project="${DECODE_PROJECT}" \
        --router="${DECODE_ROUTER}" \
        --region="${DECODE_REGION}" \
        --auto-allocate-nat-external-ips \
        --nat-all-subnet-ip-ranges \
        2>/dev/null \
        || warn "NAT '${DECODE_NAT}' may already exist, continuing..."

    log "Cloud NAT Gateways created."
}

# ==============================================================================
# STEP 3: Verify NAT Setup
# ==============================================================================

verify_nat() {
    log "=== Verifying NAT Configuration ==="

    log "--- Prefill NAT ---"
    gcloud compute routers nats describe "${PREFILL_NAT}" \
        --project="${PREFILL_PROJECT}" \
        --router="${PREFILL_ROUTER}" \
        --region="${PREFILL_REGION}" \
        --format="table(name, natIpAllocateOption, sourceSubnetworkIpRangesToNat)" \
        2>/dev/null \
        || warn "Could not describe Prefill NAT."

    log "--- Decode NAT ---"
    gcloud compute routers nats describe "${DECODE_NAT}" \
        --project="${DECODE_PROJECT}" \
        --router="${DECODE_ROUTER}" \
        --region="${DECODE_REGION}" \
        --format="table(name, natIpAllocateOption, sourceSubnetworkIpRangesToNat)" \
        2>/dev/null \
        || warn "Could not describe Decode NAT."
}

# ==============================================================================
# MAIN
# ==============================================================================

main() {
    log "============================================================"
    log "Cloud NAT Setup for Disaggregated LLM Serving"
    log "============================================================"
    log "Prefill: project=${PREFILL_PROJECT}, region=${PREFILL_REGION}, vpc=${PREFILL_VPC}"
    log "Decode:  project=${DECODE_PROJECT}, region=${DECODE_REGION}, vpc=${DECODE_VPC}"
    log "============================================================"

    create_routers
    create_nats
    verify_nat

    log ""
    log "============================================================"
    log "✓ Cloud NAT setup complete!"
    log "============================================================"
    log ""
    log "Your VMs can now reach the internet for:"
    log "  - git clone"
    log "  - pip install"
    log "  - apt-get update/install"
    log "  - HuggingFace model downloads"
    log ""
    log "Note: NAT may take 1-2 minutes to become active."
    log "Test from a VM with:  curl -s https://ifconfig.me"
    log "============================================================"
}

main "$@"
