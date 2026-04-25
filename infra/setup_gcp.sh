#!/usr/bin/env bash
# ==============================================================================
# Phase 1: GCP Infrastructure & Networking Setup
# ==============================================================================
# This script provisions two GCP instances in separate projects, connected
# via VPC Peering, for disaggregated LLM serving (Prefill + Decode).
#
# KEY FEATURE: Automatically sweeps regions/zones to find available L4 GPUs.
# Prefers placing both VMs in the SAME region for lowest latency.
# Writes discovered metadata (IPs, zones, etc.) to a .env file.
#
# Usage:
#   chmod +x infra/setup_gcp.sh
#   bash infra/setup_gcp.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated for both projects
#   - Sufficient GPU quota (1x L4 per project) — the script will find where
#   - Billing enabled on both projects
# ==============================================================================

set -euo pipefail

# ==============================================================================
# CONFIGURATION — Edit these variables before running
# ==============================================================================

# GCP Project IDs
PREFILL_PROJECT="coms-6998-amlc"
DECODE_PROJECT="coms-6998-hpml-487106"

# VPC Names
PREFILL_VPC="prefill-vpc"
DECODE_VPC="decode-vpc"

# Subnet Name Prefixes (final name = prefix-region, e.g. prefill-subnet-us-east4)
PREFILL_SUBNET_PREFIX="prefill-subnet"
DECODE_SUBNET_PREFIX="decode-subnet"

# VPC Peering Names
PEERING_PREFILL_TO_DECODE="prefill-to-decode"
PEERING_DECODE_TO_PREFILL="decode-to-prefill"

# Compute Instance Names
PREFILL_VM="prefill-worker-vm"
DECODE_VM="decode-worker-vm"

# Machine Configuration
MACHINE_TYPE="g2-standard-4"
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
IMAGE_PROJECT="deeplearning-platform-release"
# IMAGE_FAMILY is auto-discovered at runtime — see discover_image_family()
IMAGE_FAMILY=""

# Firewall
INTERNAL_CIDR="10.0.0.0/8"
FIREWALL_PREFILL="allow-internal-prefill"
FIREWALL_DECODE="allow-internal-decode"

# ZMQ Port
ZMQ_PORT=5555

# Output .env file path (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${REPO_ROOT}/.env"

# ==============================================================================
# Candidate regions to sweep, ordered by preference.
# US-East first (closest to Columbia/NYC), then US, then Europe, then Asia.
# ==============================================================================
CANDIDATE_REGIONS=(
    "us-east4"
    "us-east1"
    "us-central1"
    "us-west1"
    "us-west4"
    "us-south1"
    "europe-west1"
    "europe-west4"
    "europe-west2"
    "europe-west3"
    "asia-east1"
    "asia-southeast1"
    "asia-northeast1"
    "asia-south1"
    "me-west1"
    "northamerica-northeast1"
    "southamerica-east1"
)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log()   { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        error "gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    log "gcloud CLI found: $(gcloud --version 2>/dev/null | head -1)"
}

# ==============================================================================
# STEP 0a: Discover the latest Deep Learning VM image family
# ==============================================================================
# GCP deprecated "pytorch-latest-gpu". Image families are now versioned:
#   pytorch-2-9-cu129-ubuntu-2404-nvidia-580
# This function auto-discovers the newest pytorch image family so we never
# hardcode a name that might disappear.
# ==============================================================================

discover_image_family() {
    log "=== Discovering latest PyTorch GPU image family ==="

    # List all image families matching pytorch, pick the latest one
    local families
    families=$(gcloud compute images list \
        --project="${IMAGE_PROJECT}" \
        --no-standard-images \
        --format="value(family)" \
        --sort-by="~creationTimestamp" \
        --limit=200 2>/dev/null | tr -d '\r' | grep -i "^pytorch" | sort -u -r | head -5)

    if [[ -z "${families}" ]]; then
        error "Could not find any PyTorch image families in ${IMAGE_PROJECT}!"
        error "Falling back to a common known family name."
        # Hardcoded fallback — adjust if this also becomes stale
        IMAGE_FAMILY="pytorch-2-4-cu124-debian-11-py310"
        warn "Using fallback: ${IMAGE_FAMILY}"
        return
    fi

    log "Available PyTorch image families (top 5, newest first):"
    echo "${families}" | while read -r fam; do
        log "  - ${fam}"
    done

    # Pick the first (newest) one
    IMAGE_FAMILY=$(echo "${families}" | head -1)

    log "Selected image family: ${IMAGE_FAMILY}"
    log ""
}

# ==============================================================================
# STEP 0b: GPU Zone Discovery
# ==============================================================================
# Sweeps candidate regions to find zones where nvidia-l4 is available.
#
# Strategy:
#   1. For each region (in preference order), list zones and check if the
#      nvidia-l4 accelerator type exists there for each project.
#   2. OPTIMIZATION: If both projects have L4 in the SAME region, pick that
#      region immediately (lowest latency via intra-region VPC peering).
#   3. If no shared region, fall back to best available zone per project.
# ==============================================================================

# Results — set by discover_gpu_zones()
PREFILL_ZONE=""
PREFILL_REGION=""
DECODE_ZONE=""
DECODE_REGION=""
PREFILL_SUBNET_NAME=""
DECODE_SUBNET_NAME=""

get_zones_in_region() {
    local project="$1"
    local region="$2"
    gcloud compute zones list \
        --project="${project}" \
        --filter="region:${region} AND status=UP" \
        --format="value(name)" 2>/dev/null | tr -d '\r' || true
}

check_gpu_in_zone() {
    local project="$1"
    local zone="$2"
    local gpu_type="$3"

    local result
    result=$(gcloud compute accelerator-types list \
        --project="${project}" \
        --filter="zone:${zone} AND name:${gpu_type}" \
        --format="value(name)" 2>/dev/null | tr -d '\r' || true)

    if [[ -n "${result}" ]]; then
        return 0
    fi
    return 1
}

discover_gpu_zones() {
    log "=== GPU Zone Discovery ==="
    log "Sweeping ${#CANDIDATE_REGIONS[@]} candidate regions for ${GPU_TYPE}..."
    log ""

    # Track which zones work per project, and per-region hits
    local -a prefill_valid_zones=()
    local -a decode_valid_zones=()
    local -A prefill_region_to_zone=()
    local -A decode_region_to_zone=()

    for region in "${CANDIDATE_REGIONS[@]}"; do
        log "Scanning region: ${region} ..."

        local zones
        zones=$(get_zones_in_region "${PREFILL_PROJECT}" "${region}")

        if [[ -z "${zones}" ]]; then
            log "  (no active zones, skipping)"
            continue
        fi

        for zone in ${zones}; do
            # --- Check Prefill project ---
            if [[ -z "${prefill_region_to_zone[${region}]+_}" ]]; then
                if check_gpu_in_zone "${PREFILL_PROJECT}" "${zone}" "${GPU_TYPE}"; then
                    log "  [PREFILL] ✓ ${GPU_TYPE} available in ${zone}"
                    prefill_valid_zones+=("${zone}")
                    prefill_region_to_zone["${region}"]="${zone}"
                else
                    log "  [PREFILL] ✗ ${zone}"
                fi
            fi

            # --- Check Decode project ---
            if [[ -z "${decode_region_to_zone[${region}]+_}" ]]; then
                if check_gpu_in_zone "${DECODE_PROJECT}" "${zone}" "${GPU_TYPE}"; then
                    log "  [DECODE]  ✓ ${GPU_TYPE} available in ${zone}"
                    decode_valid_zones+=("${zone}")
                    decode_region_to_zone["${region}"]="${zone}"
                else
                    log "  [DECODE]  ✗ ${zone}"
                fi
            fi

            # Early exit for this region if both found
            if [[ -n "${prefill_region_to_zone[${region}]+_}" ]] && \
               [[ -n "${decode_region_to_zone[${region}]+_}" ]]; then
                break
            fi
        done

        # =================================================================
        # OPTIMIZATION: If we found BOTH projects in this region, stop
        # the entire sweep — this is the best possible placement.
        # =================================================================
        if [[ -n "${prefill_region_to_zone[${region}]+_}" ]] && \
           [[ -n "${decode_region_to_zone[${region}]+_}" ]]; then
            log ""
            log "  ★ SHARED REGION FOUND: ${region}"
            log "    Both projects have ${GPU_TYPE} here — optimal placement!"
            log "    Stopping sweep early."
            PREFILL_ZONE="${prefill_region_to_zone[${region}]}"
            PREFILL_REGION="${region}"
            DECODE_ZONE="${decode_region_to_zone[${region}]}"
            DECODE_REGION="${region}"
            break
        fi
    done

    # If we didn't find a shared region, fall back to first valid zone per project
    if [[ -z "${PREFILL_ZONE}" ]]; then
        if [[ ${#prefill_valid_zones[@]} -eq 0 ]]; then
            error "No zone found with ${GPU_TYPE} for project '${PREFILL_PROJECT}'!"
            error "Request quota at: https://console.cloud.google.com/iam-admin/quotas?project=${PREFILL_PROJECT}"
            exit 1
        fi
        PREFILL_ZONE="${prefill_valid_zones[0]}"
        PREFILL_REGION=$(echo "${PREFILL_ZONE}" | rev | cut -d'-' -f2- | rev)
    fi
    if [[ -z "${DECODE_ZONE}" ]]; then
        if [[ ${#decode_valid_zones[@]} -eq 0 ]]; then
            error "No zone found with ${GPU_TYPE} for project '${DECODE_PROJECT}'!"
            error "Request quota at: https://console.cloud.google.com/iam-admin/quotas?project=${DECODE_PROJECT}"
            exit 1
        fi
        DECODE_ZONE="${decode_valid_zones[0]}"
        DECODE_REGION=$(echo "${DECODE_ZONE}" | rev | cut -d'-' -f2- | rev)

        warn "No shared region found. Using separate regions:"
        warn "  Prefill: ${PREFILL_ZONE} (${PREFILL_REGION})"
        warn "  Decode:  ${DECODE_ZONE} (${DECODE_REGION})"
        warn "  Cross-region VPC peering will have higher latency."
    fi

    log ""
    log "============================================"
    log "  GPU ZONE DISCOVERY RESULTS"
    log "============================================"
    log "  Prefill VM → zone: ${PREFILL_ZONE} (region: ${PREFILL_REGION})"
    log "  Decode  VM → zone: ${DECODE_ZONE} (region: ${DECODE_REGION})"
    if [[ "${PREFILL_REGION}" == "${DECODE_REGION}" ]]; then
        log "  Placement:  SAME REGION ✓ (optimal)"
    else
        log "  Placement:  CROSS-REGION ⚠ (higher latency)"
    fi
    log "============================================"
    log ""
}

# ==============================================================================
# STEP 1: Create Custom VPCs + Subnets
# ==============================================================================

create_vpcs() {
    log "=== Creating Custom VPCs & Subnets ==="

    # --- Prefill VPC (global) ---
    log "Creating VPC '${PREFILL_VPC}' in project '${PREFILL_PROJECT}'..."
    gcloud compute networks create "${PREFILL_VPC}" \
        --project="${PREFILL_PROJECT}" \
        --subnet-mode=custom \
        --bgp-routing-mode=regional \
        --description="Prefill VPC for disaggregated LLM serving" \
        2>/dev/null \
        || warn "VPC '${PREFILL_VPC}' may already exist, continuing..."

    # Prefill Subnet in discovered region
    PREFILL_SUBNET_NAME="${PREFILL_SUBNET_PREFIX}-${PREFILL_REGION}"
    log "Creating subnet '${PREFILL_SUBNET_NAME}' in region '${PREFILL_REGION}'..."
    gcloud compute networks subnets create "${PREFILL_SUBNET_NAME}" \
        --project="${PREFILL_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --region="${PREFILL_REGION}" \
        --range="10.10.0.0/20" \
        2>/dev/null \
        || warn "Subnet '${PREFILL_SUBNET_NAME}' may already exist, continuing..."

    # --- Decode VPC (global) ---
    log "Creating VPC '${DECODE_VPC}' in project '${DECODE_PROJECT}'..."
    gcloud compute networks create "${DECODE_VPC}" \
        --project="${DECODE_PROJECT}" \
        --subnet-mode=custom \
        --bgp-routing-mode=regional \
        --description="Decode VPC for disaggregated LLM serving" \
        2>/dev/null \
        || warn "VPC '${DECODE_VPC}' may already exist, continuing..."

    # Decode Subnet in discovered region
    DECODE_SUBNET_NAME="${DECODE_SUBNET_PREFIX}-${DECODE_REGION}"
    log "Creating subnet '${DECODE_SUBNET_NAME}' in region '${DECODE_REGION}'..."
    gcloud compute networks subnets create "${DECODE_SUBNET_NAME}" \
        --project="${DECODE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --region="${DECODE_REGION}" \
        --range="10.20.0.0/20" \
        2>/dev/null \
        || warn "Subnet '${DECODE_SUBNET_NAME}' may already exist, continuing..."

    # If cross-region, create extra subnets for routing
    if [[ "${PREFILL_REGION}" != "${DECODE_REGION}" ]]; then
        warn "Cross-region detected. Creating additional subnets for routing..."

        gcloud compute networks subnets create "${DECODE_SUBNET_PREFIX}-${PREFILL_REGION}" \
            --project="${DECODE_PROJECT}" \
            --network="${DECODE_VPC}" \
            --region="${PREFILL_REGION}" \
            --range="10.21.0.0/20" \
            2>/dev/null \
            || warn "Cross-region decode subnet may already exist..."

        gcloud compute networks subnets create "${PREFILL_SUBNET_PREFIX}-${DECODE_REGION}" \
            --project="${PREFILL_PROJECT}" \
            --network="${PREFILL_VPC}" \
            --region="${DECODE_REGION}" \
            --range="10.11.0.0/20" \
            2>/dev/null \
            || warn "Cross-region prefill subnet may already exist..."
    fi

    log "VPCs and subnets created."
}

# ==============================================================================
# STEP 2: Create Firewall Rules
# ==============================================================================

create_firewall_rules() {
    log "=== Creating Firewall Rules ==="

    gcloud compute firewall-rules create "${FIREWALL_PREFILL}" \
        --project="${PREFILL_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp,udp,icmp \
        --source-ranges="${INTERNAL_CIDR}" \
        --description="Allow internal traffic for disaggregated serving" \
        2>/dev/null \
        || warn "Firewall '${FIREWALL_PREFILL}' may already exist..."

    gcloud compute firewall-rules create "${FIREWALL_DECODE}" \
        --project="${DECODE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --direction=INGRESS \
        --action=ALLOW \
        --rules=tcp,udp,icmp \
        --source-ranges="${INTERNAL_CIDR}" \
        --description="Allow internal traffic for disaggregated serving" \
        2>/dev/null \
        || warn "Firewall '${FIREWALL_DECODE}' may already exist..."

    gcloud compute firewall-rules create "allow-ssh-prefill" \
        --project="${PREFILL_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --direction=INGRESS --action=ALLOW \
        --rules=tcp:22 --source-ranges="0.0.0.0/0" \
        --description="Allow SSH to Prefill VM" \
        2>/dev/null \
        || warn "SSH rule for Prefill may already exist..."

    gcloud compute firewall-rules create "allow-ssh-decode" \
        --project="${DECODE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --direction=INGRESS --action=ALLOW \
        --rules=tcp:22 --source-ranges="0.0.0.0/0" \
        --description="Allow SSH to Decode VM" \
        2>/dev/null \
        || warn "SSH rule for Decode may already exist..."

    # IAP (Identity-Aware Proxy) rules — required for SSH when VMs have no 
    # external IP (Columbia org policy blocks external IPs).
    # IAP's IP range is 35.235.240.0/20.
    log "Creating IAP tunnel firewall rules..."
    gcloud compute firewall-rules create "allow-iap-prefill" \
        --project="${PREFILL_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --direction=INGRESS --action=ALLOW \
        --rules=tcp:22,tcp:3389 \
        --source-ranges="35.235.240.0/20" \
        --description="Allow IAP tunnel SSH access (no external IP)" \
        2>/dev/null \
        || warn "IAP rule for Prefill may already exist..."

    gcloud compute firewall-rules create "allow-iap-decode" \
        --project="${DECODE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --direction=INGRESS --action=ALLOW \
        --rules=tcp:22,tcp:3389 \
        --source-ranges="35.235.240.0/20" \
        --description="Allow IAP tunnel SSH access (no external IP)" \
        2>/dev/null \
        || warn "IAP rule for Decode may already exist..."

    log "Firewall rules created."
}

# ==============================================================================
# STEP 3: Establish Bidirectional VPC Peering
# ==============================================================================
# NOTE: Under a university workspace (Columbia), organizational policies may
# block cross-project VPC peering:
#   constraints/compute.restrictVpcPeering
#
# If this fails, the error output below gives IT-ready details for override.
# ==============================================================================

create_vpc_peering() {
    log "=== Establishing Bidirectional VPC Peering ==="

    PREFILL_NETWORK_URI="projects/${PREFILL_PROJECT}/global/networks/${PREFILL_VPC}"
    DECODE_NETWORK_URI="projects/${DECODE_PROJECT}/global/networks/${DECODE_VPC}"

    # Direction 1: Prefill → Decode
    log "Peering '${PEERING_PREFILL_TO_DECODE}': ${PREFILL_VPC} → ${DECODE_VPC}..."
    if ! gcloud compute networks peerings create "${PEERING_PREFILL_TO_DECODE}" \
        --project="${PREFILL_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --peer-project="${DECODE_PROJECT}" \
        --peer-network="${DECODE_VPC}" \
        --export-custom-routes \
        --import-custom-routes 2>&1; then

        error "============================================================"
        error "VPC PEERING FAILED: Prefill → Decode"
        error "Likely: constraints/compute.restrictVpcPeering"
        error ""
        error "Send this to IT for override:"
        error "  Prefill Network: ${PREFILL_NETWORK_URI}"
        error "  Decode Network:  ${DECODE_NETWORK_URI}"
        error "  Prefill Project: ${PREFILL_PROJECT}"
        error "  Decode Project:  ${DECODE_PROJECT}"
        error "============================================================"
    fi

    # Direction 2: Decode → Prefill
    log "Peering '${PEERING_DECODE_TO_PREFILL}': ${DECODE_VPC} → ${PREFILL_VPC}..."
    if ! gcloud compute networks peerings create "${PEERING_DECODE_TO_PREFILL}" \
        --project="${DECODE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --peer-project="${PREFILL_PROJECT}" \
        --peer-network="${PREFILL_VPC}" \
        --export-custom-routes \
        --import-custom-routes 2>&1; then

        error "============================================================"
        error "VPC PEERING FAILED: Decode → Prefill"
        error "Same org policy issue. See above for IT details."
        error "============================================================"
    fi

    # Verify
    log "Verifying peering status..."
    gcloud compute networks peerings list \
        --project="${PREFILL_PROJECT}" --network="${PREFILL_VPC}" \
        --format="table(name, network, peerNetwork, state)" \
        || warn "Could not verify Prefill peering status."

    gcloud compute networks peerings list \
        --project="${DECODE_PROJECT}" --network="${DECODE_VPC}" \
        --format="table(name, network, peerNetwork, state)" \
        || warn "Could not verify Decode peering status."

    log "VPC Peering setup done."
}

# ==============================================================================
# STEP 4: Provision Compute Instances
# ==============================================================================

create_instances() {
    log "=== Provisioning Compute Instances ==="
    log "  Prefill: ${PREFILL_ZONE} (subnet: ${PREFILL_SUBNET_NAME})"
    log "  Decode:  ${DECODE_ZONE} (subnet: ${DECODE_SUBNET_NAME})"

    # Write the startup script to a temporary file to avoid gcloud CLI 
    # metadata parsing errors (which split on commas if passed inline).
    local STARTUP_SCRIPT_FILE=".startup_script.sh"
    cat > "${STARTUP_SCRIPT_FILE}" << 'EOF'
#!/bin/bash
set -e
echo "[startup] Installing system deps..."
apt-get update -qq && apt-get install -y -qq netcat-openbsd

echo "[startup] Installing Python deps..."
pip install --quiet fastapi "uvicorn[standard]" httpx torch transformers accelerate pyzmq python-dotenv requests pydantic

echo "[startup] Verifying GPU..."
nvidia-smi || echo "[startup] WARNING: nvidia-smi failed"
python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")"
echo "[startup] Done."
EOF

    # --- Create Prefill VM ---
    log "Creating Prefill VM '${PREFILL_VM}' in ${PREFILL_ZONE}..."
    if ! gcloud compute instances create "${PREFILL_VM}" \
        --project="${PREFILL_PROJECT}" \
        --zone="${PREFILL_ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type=pd-balanced \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --network="${PREFILL_VPC}" \
        --subnet="${PREFILL_SUBNET_NAME}" \
        --no-address \
        --metadata-from-file="startup-script=${STARTUP_SCRIPT_FILE}" \
        --scopes="cloud-platform" \
        --tags="prefill-worker" 2>&1; then

        error "Failed to create Prefill VM in ${PREFILL_ZONE}."
        error "Quota URL: https://console.cloud.google.com/iam-admin/quotas?project=${PREFILL_PROJECT}"
        rm -f "${STARTUP_SCRIPT_FILE}"
        exit 1
    fi

    # --- Create Decode VM ---
    log "Creating Decode VM '${DECODE_VM}' in ${DECODE_ZONE}..."
    if ! gcloud compute instances create "${DECODE_VM}" \
        --project="${DECODE_PROJECT}" \
        --zone="${DECODE_ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
        --maintenance-policy=TERMINATE \
        --restart-on-failure \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type=pd-balanced \
        --image-family="${IMAGE_FAMILY}" \
        --image-project="${IMAGE_PROJECT}" \
        --network="${DECODE_VPC}" \
        --subnet="${DECODE_SUBNET_NAME}" \
        --no-address \
        --metadata-from-file="startup-script=${STARTUP_SCRIPT_FILE}" \
        --scopes="cloud-platform" \
        --tags="decode-worker" 2>&1; then

        error "Failed to create Decode VM in ${DECODE_ZONE}."
        error "Quota URL: https://console.cloud.google.com/iam-admin/quotas?project=${DECODE_PROJECT}"
        rm -f "${STARTUP_SCRIPT_FILE}"
        exit 1
    fi

    # Clean up the temp script file
    rm -f "${STARTUP_SCRIPT_FILE}"

    log "Instances created. Waiting 10s for network init..."
    sleep 10
}

# ==============================================================================
# STEP 5: Extract VM Metadata → Write .env
# ==============================================================================

write_env_file() {
    log "=== Extracting VM metadata and writing .env ==="

    # Fetch internal IPs (no external IPs due to org policy)
    PREFILL_INTERNAL_IP=$(gcloud compute instances describe "${PREFILL_VM}" \
        --project="${PREFILL_PROJECT}" --zone="${PREFILL_ZONE}" \
        --format="value(networkInterfaces[0].networkIP)" 2>/dev/null | tr -d '\r')

    DECODE_INTERNAL_IP=$(gcloud compute instances describe "${DECODE_VM}" \
        --project="${DECODE_PROJECT}" --zone="${DECODE_ZONE}" \
        --format="value(networkInterfaces[0].networkIP)" 2>/dev/null | tr -d '\r')

    log "  Prefill internal IP: ${PREFILL_INTERNAL_IP}"
    log "  Decode  internal IP: ${DECODE_INTERNAL_IP}"

    local placement="same-region"
    [[ "${PREFILL_REGION}" != "${DECODE_REGION}" ]] && placement="cross-region"

    cat > "${ENV_FILE}" <<ENVEOF
# ==============================================================================
# Disaggregated LLM Serving — Environment Configuration
# ==============================================================================
# AUTO-GENERATED by infra/setup_gcp.sh on $(date -u +'%Y-%m-%dT%H:%M:%SZ')
# Placement mode: ${placement}
# ==============================================================================

# --- GCP Infrastructure (auto-discovered) ---
PREFILL_PROJECT=${PREFILL_PROJECT}
DECODE_PROJECT=${DECODE_PROJECT}
PREFILL_ZONE=${PREFILL_ZONE}
DECODE_ZONE=${DECODE_ZONE}
PREFILL_REGION=${PREFILL_REGION}
DECODE_REGION=${DECODE_REGION}
PREFILL_VM_NAME=${PREFILL_VM}
DECODE_VM_NAME=${DECODE_VM}
PLACEMENT_MODE=${placement}

# --- VM Network Addresses (auto-discovered) ---
# No external IPs — Columbia org policy blocks them.
# Use IAP tunneling for SSH: gcloud compute ssh --tunnel-through-iap
PREFILL_INTERNAL_IP=${PREFILL_INTERNAL_IP}
DECODE_INTERNAL_IP=${DECODE_INTERNAL_IP}

# --- HuggingFace (FILL THIS IN MANUALLY) ---
HF_TOKEN=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

# --- Model ---
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct

# --- API Gateway ---
GATEWAY_HOST=0.0.0.0
GATEWAY_PORT=8000

# --- Prefill Worker ---
PREFILL_WORKER_HOST=${PREFILL_INTERNAL_IP}
PREFILL_WORKER_PORT=8001

# --- Decode Worker ---
DECODE_WORKER_HOST=${DECODE_INTERNAL_IP}
DECODE_WORKER_PORT=8002

# --- ZMQ KV-Cache Streaming ---
ZMQ_PORT=${ZMQ_PORT}

# --- SSH Quick Commands (via IAP tunnel — no external IP) ---
# gcloud compute ssh ${PREFILL_VM} --project=${PREFILL_PROJECT} --zone=${PREFILL_ZONE} --tunnel-through-iap
# gcloud compute ssh ${DECODE_VM} --project=${DECODE_PROJECT} --zone=${DECODE_ZONE} --tunnel-through-iap
ENVEOF

    log ""
    log "✓ .env written to: ${ENV_FILE}"
    log ""
    log "========== .env contents =========="
    cat "${ENV_FILE}"
    log "==================================="
}

# ==============================================================================
# MAIN
# ==============================================================================

main() {
    log "============================================================"
    log "Disaggregated LLM Serving — GCP Infrastructure Setup"
    log "============================================================"
    log "Prefill Project: ${PREFILL_PROJECT}"
    log "Decode Project:  ${DECODE_PROJECT}"
    log "GPU Target:      ${MACHINE_TYPE} + ${GPU_COUNT}x ${GPU_TYPE}"
    log "Regions to scan: ${#CANDIDATE_REGIONS[@]}"
    log "============================================================"

    check_gcloud

    # Step 0a: Discover the latest PyTorch GPU image family
    discover_image_family

    # Step 0b: Sweep regions to find L4 GPU zones
    discover_gpu_zones

    # Steps 1-4: Build infra in discovered zones
    create_vpcs
    create_firewall_rules
    create_vpc_peering
    create_instances

    # Step 5: Extract IPs and write .env
    write_env_file

    log ""
    log "============================================================"
    log "✓ INFRASTRUCTURE SETUP COMPLETE"
    log "============================================================"
    log "  Prefill VM: ${PREFILL_ZONE} (${PREFILL_INTERNAL_IP})"
    log "  Decode  VM: ${DECODE_ZONE} (${DECODE_INTERNAL_IP})"
    if [[ "${PREFILL_REGION}" == "${DECODE_REGION}" ]]; then
        log "  ✓ Same-region placement — optimal latency"
    else
        warn "  ⚠ Cross-region — expect higher latency"
    fi
    log ""
    log "Next steps:"
    log "  1. Edit .env → fill in HF_TOKEN"
    log "  2. SSH into VMs (via IAP tunnel — no external IP):"
    log "     gcloud compute ssh ${PREFILL_VM} --project=${PREFILL_PROJECT} --zone=${PREFILL_ZONE} --tunnel-through-iap"
    log "     gcloud compute ssh ${DECODE_VM} --project=${DECODE_PROJECT} --zone=${DECODE_ZONE} --tunnel-through-iap"
    log "  3. Clone repo + copy .env to each VM"
    log "  4. Start: decode_worker → prefill_worker → gateway"
    log "  5. Run:   python benchmark/benchmark.py"
    log "============================================================"
}

main "$@"
