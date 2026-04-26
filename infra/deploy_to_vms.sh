#!/usr/bin/env bash
# ==============================================================================
# Deploy App Code + Dependencies to Both VMs
# ==============================================================================
# Runs from your LOCAL machine. Copies files to each VM via IAP tunnel,
# installs Python deps, and configures .env.
#
# Usage:
#   bash infra/deploy_to_vms.sh
#
# Prerequisites:
#   - VMs are running (setup_gcp.sh completed)
#   - Cloud NAT is active (setup_nat.sh completed)
#   - .env file exists in repo root
# ==============================================================================

set -euo pipefail

# Load .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
ENV_FILE="${REPO_ROOT}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] No .env file found at ${ENV_FILE}. Run setup_gcp.sh first."
    exit 1
fi
source "${ENV_FILE}"

# Files to copy
APP_FILES=(
    "app/gateway.py"
    "app/prefill_worker.py"
    "app/decode_worker.py"
    "app/collocated_baseline.py"
    "benchmark/benchmark.py"
    "benchmark/concurrent_workload.py"
    "benchmark/compare_architectures.py"
    "requirements.txt"
    ".env"
)

log()   { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $*"; }
warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  $*" >&2; }
error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }

deploy_to_vm() {
    local project="$1"
    local zone="$2"
    local vm_name="$3"
    local role="$4"  # "prefill" or "decode"

    log "=== Deploying to ${role} VM: ${vm_name} (${project}/${zone}) ==="

    # Create directory structure on VM
    log "Creating directories on ${vm_name}..."
    gcloud compute ssh "${vm_name}" \
        --project="${project}" \
        --zone="${zone}" \
        --tunnel-through-iap \
        --command="mkdir -p ~/project/app ~/project/benchmark" \
        2>/dev/null

    # Copy files
    for file in "${APP_FILES[@]}"; do
        local src="${REPO_ROOT}/${file}"
        if [[ -f "${src}" ]]; then
            log "  Copying ${file}..."
            gcloud compute scp "${src}" \
                "${vm_name}:~/project/${file}" \
                --project="${project}" \
                --zone="${zone}" \
                --tunnel-through-iap \
                2>/dev/null
        else
            warn "  File not found: ${src}, skipping."
        fi
    done

    # Install Python dependencies
    log "Installing Python dependencies on ${vm_name}..."
    gcloud compute ssh "${vm_name}" \
        --project="${project}" \
        --zone="${zone}" \
        --tunnel-through-iap \
        --command="cd ~/project && pip install -q -r requirements.txt" \
        2>/dev/null

    # Verify GPU
    log "Verifying GPU on ${vm_name}..."
    gcloud compute ssh "${vm_name}" \
        --project="${project}" \
        --zone="${zone}" \
        --tunnel-through-iap \
        --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU not ready yet'" \
        2>/dev/null

    log "✓ ${role} VM deployment complete."
    log ""
}

main() {
    log "============================================================"
    log "Deploying to Both VMs"
    log "============================================================"

    deploy_to_vm "${PREFILL_PROJECT}" "${PREFILL_ZONE}" "${PREFILL_VM_NAME}" "prefill"
    deploy_to_vm "${DECODE_PROJECT}" "${DECODE_ZONE}" "${DECODE_VM_NAME}" "decode"

    log "============================================================"
    log "✓ Deployment complete on both VMs!"
    log "============================================================"
    log ""
    log "Next: SSH in and start services (in order):"
    log ""
    log "  1. Decode VM (start first):"
    log "     gcloud compute ssh ${DECODE_VM_NAME} --project=${DECODE_PROJECT} --zone=${DECODE_ZONE} --tunnel-through-iap"
    log "     cd ~/project && python app/decode_worker.py"
    log ""
    log "  2. Prefill VM:"
    log "     gcloud compute ssh ${PREFILL_VM_NAME} --project=${PREFILL_PROJECT} --zone=${PREFILL_ZONE} --tunnel-through-iap"
    log "     cd ~/project && python app/prefill_worker.py"
    log ""
    log "  3. Gateway (on Prefill VM, different terminal):"
    log "     cd ~/project && python app/gateway.py"
    log "============================================================"
}

main "$@"
