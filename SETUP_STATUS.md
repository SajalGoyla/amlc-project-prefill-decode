# AMLIC Setup Status & Bring-up Notes

> Snapshot from the initial infrastructure bring-up. Captures what is deployed,
> what works end-to-end, and where the disaggregated path is currently blocked.

## TL;DR

- **Both VMs are provisioned, on Tailscale, and have all dependencies installed.**
- **Collocated baseline works end-to-end** on the L4 (verified with a real chat completion).
- **Disaggregated path is blocked** by a NIXL handshake incompatibility between L4 (Ada / FlashAttention2 / bf16) and T4 (Turing / FlashInfer / fp16). NIXL strictly validates that producer and consumer KV cache layouts match, and the two GPU generations produce different in-memory shapes.
- VMs are **stopped** to save cost; everything resumes idempotently on `gcloud compute instances start`.

## Deployed Infrastructure

### VMs

| Role | Name | Project | Zone | GPU | Tailscale IP | External IP |
|------|------|---------|------|-----|--------------|-------------|
| Prefill + Collocated | `amlic-prefill` | `static-factor-485822-d6` (Madhav) | `us-central1-a` | NVIDIA L4 24 GB | `100.95.140.121` | `34.67.216.155` |
| Decode | `amlic-decode` | `amlic-proj` (friend) | `us-west4-b` | NVIDIA T4 16 GB | `100.87.203.47` | (none — Cloud NAT only) |

Boot images: `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (Deep Learning VM), 150 GB pd-balanced, NVIDIA driver 580 preinstalled.

### Networking

- **Tailscale tailnet** (`madhavtibrewal92@`) joins both VMs across the two GCP projects. VM↔VM ping ~50 ms avg RTT, ~0.56 Gbps measured TCP throughput (direct connection, no DERP relay).
- **L4 firewall** (`amlic-allow` in Madhav's project) opens TCP `8000, 8100, 9000, 5600` to `0.0.0.0/0`.
- **T4 firewall** (`amlic-allow` in friend's project) opens TCP `8200, 5600`.
- **Cloud NAT** (`amlic-router` + `amlic-nat`) in `us-west4` for the T4's outbound internet (the friend's project has an org policy `compute.vmExternalIpAccess` that blocks external IPs on instances).
- T4 SSH is via IAP tunnel (`gcloud compute ssh ... --tunnel-through-iap`).

### Software state on both VMs

- Python 3.10 venv at `~/amlic-venv` with `vllm==0.19.1`, `nixl`, PyTorch 2.10+cu128, NIXL Python bindings.
- Project repo at `~/amlic/` (rsynced from laptop, `.sh` files line-ending-stripped).
- Llama-3.2-3B-Instruct cached at `~/.cache/huggingface/hub/...` (via `hf download`).
- Docker 29.4.1 + nvidia-container-toolkit configured.
- `vllm/vllm-openai:latest` Docker image pulled (~9.6 GB compressed).
- Tailscale 1.96.4 installed and authed to the tailnet.

### Local laptop state

- `.env` filled in with both VM Tailscale IPs, HF token, ports, etc. (`.env` is gitignored.)
- `gcloud` SDK authenticated as `madhavtibrewal92@gmail.com` with **owner** role on both GCP projects.

## What Works

### 1. Collocated vLLM on L4 (proven)

Bare-metal, native install. Smoke test request returned a real generation:

```bash
curl http://100.95.140.121:8000/v1/chat/completions \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":30}'
# → "Hello, it's nice to meet you." (10 completion tokens, 41 prompt tokens)
```

Health endpoint at `/health` returns `200`. This is the baseline arm of the experiment and is ready for benchmarking via `benchmark/runner.py --arch collocated`.

Started with [`infra/start_collocated.sh`](infra/start_collocated.sh) (native, no container).

### 2. Both Docker containers can start independently

`infra/start_prefill_docker.sh` on L4 and `infra/start_decode_docker.sh` on T4 both bring the vLLM API server to `200 /health`. The model loads in ~5–10 sec when pre-cached. Issue is only at first inter-engine request (see Blocker below).

## What's Blocked

### NIXL handshake fails on KV cache layout between L4 and T4

When the proxy (`infra/proxy_server.py`) forwards a prefill response's `kv_transfer_params` to the decode engine, decode tries to handshake with the prefill via NIXL's side-channel and crashes with:

```
RuntimeError: NIXL compatibility hash mismatch.
Local:  3fc700da... (T4 / FlashInfer / fp16)
Remote: 161228f0... (L4 / FlashAttention2 / bf16)
Prefill and decode instances have incompatible configurations.
```

After disabling the strict hash check via `kv_connector_extra_config: {enforce_handshake_compat: false}` and forcing `--dtype float16 --max-model-len 4096` on both engines, the handshake gets one step further before crashing on a deeper assertion:

```
AssertionError: Remote P worker KV layer cache must be of shape
[2, N, local_kv_heads*tp_ratio, page_size, head_dim] and same dtype.
```

**Root cause.** L4 (Ada Lovelace, compute capability 8.9) auto-selects FlashAttention 2; T4 (Turing, compute capability 7.5) cannot run FA2 and falls back to FlashInfer. The two backends produce different in-memory KV cache shapes, and NIXL's design assumption is a homogeneous GPU pool — it strictly validates layouts at the kernel level.

Setting `VLLM_ATTENTION_BACKEND=FLASHINFER` as a container env var **was not honored** by vLLM 0.x in `vllm/vllm-openai:latest` — the L4 still picked `FLASH_ATTN` from its potential backends list. The CLI flag (`--attention-backend FLASHINFER`) has not yet been tried; it may or may not work given vLLM still has a code path that auto-detects FA2 availability.

### Why a side note on UCX is also relevant

NIXL inside the `vllm/vllm-openai:latest` image emits this warning on startup:

```
W ucx_utils.cpp:590] memory is detected as host, check that UCX is configured with CUDA support
```

…meaning the image's bundled UCX wasn't built with CUDA support, so NIXL falls back to a "Hybrid Memory Allocator" (host bounce buffers instead of zero-copy GPU registration). This is functional but adds a host-DMA roundtrip per KV transfer. Not a blocker on its own — the layout mismatch is the actual blocker.

## What Was Tried

In rough chronological order:

1. **Native vLLM install via `pip install vllm` + `pip install nixl`.** vLLM ran fine collocated. NIXL crashed on engine init with `nixlBackendError: NIXL_ERR_BACKEND` and the explicit error `VRAM memory is detected as host by UCX. UCX is likely not configured with CUDA support.` → pip-installed `nixl` ships UCX without CUDA support.
2. **Switched to `vllm/vllm-openai:latest` Docker image** (NVIDIA-blessed prebuilt). Same UCX-CUDA warning, but vLLM 0.x's NIXL connector now uses `Hybrid Memory Allocator` and continues instead of crashing — so engines start cleanly.
3. **Sent first disaggregated request through the proxy.** Decode engine crashed with `NIXL compatibility hash mismatch` on its first try to handshake the prefill.
4. **Disabled hash check + matched dtype/max-model-len.** Decode now passes the hash check but fails on the deeper `Remote P worker KV layer cache must be of shape ...` assertion.
5. **Tried `VLLM_ATTENTION_BACKEND=FLASHINFER` env var on both containers.** Container env shows it's set (`docker exec amlic-prefill env | grep VLLM`), but vLLM still picks `FLASH_ATTN` on L4. Env var is being ignored by vLLM 0.x in this image.

## Open Paths Forward

In rough order of effort vs. probability of success:

### Path A — Switch to `LMCacheConnector` (the README's documented Plan B)

Use Redis as the KV transport. The KV blocks get serialized at the application layer (not the kernel layer), so heterogeneous GPUs work naturally. ~30 min: install Redis on one VM, point the other at it, change `kv_connector` to `LMCacheConnector` in the start scripts. Same end-to-end *experiment* — only the transport changes. Recommended.

### Path B — Force a universal attention backend

Try `--attention-backend FLASHINFER` (or `--attention-backend TRITON_ATTN`) as a CLI flag on both engines. If vLLM's CLI flag overrides the FA2 auto-detection on L4, the KV layouts may align and NIXL handshake may succeed. Unknown risk: vLLM might still produce subtly different cache shapes between Ada and Turing even with the same backend name.

### Path C — Homogeneous GPU pool

Provision a second L4 (or two T4s) on either project. Eliminates the layout mismatch entirely but defeats the project's interesting "heterogeneous tier" pitch. Use only as a last resort.

### Path D — Patch NIXL connector to relax the layout check

The vLLM source has the assertion at `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py:2140`. Could be patched to insert a layout-conversion step. Real engineering work, not a config change. Out of scope for the project deadline.

## Quick Resume Commands

```bash
# Restart both VMs
gcloud compute instances start amlic-prefill --zone=us-central1-a
gcloud compute instances start amlic-decode --zone=us-west4-b --project=amlic-proj

# Tail collocated startup
gcloud compute ssh amlic-prefill --zone=us-central1-a --command="cd ~/amlic && bash infra/start_collocated.sh"

# Restart Docker disagg attempt
gcloud compute ssh amlic-prefill --zone=us-central1-a --command="cd ~/amlic && bash infra/start_prefill_docker.sh"
gcloud compute ssh amlic-decode --zone=us-west4-b --project=amlic-proj --tunnel-through-iap --command="cd ~/amlic && bash infra/start_decode_docker.sh"

# Stop everything (cost saver)
gcloud compute instances stop amlic-prefill --zone=us-central1-a
gcloud compute instances stop amlic-decode --zone=us-west4-b --project=amlic-proj
```

## Cost Notes

- L4 (`g2-standard-4`, on-demand): **$0.71/hr** when running, ~$0.02/hr when stopped (boot disk only).
- T4 (`n1-standard-4` + 1× T4): **$0.35/hr** when running, ~$0.02/hr when stopped.
- Cloud NAT: ~$0.045/hr + egress; trivial for this workload.
- Total when both running: **~$1.06/hr GPU + ~$0.05/hr networking ≈ $1.11/hr**.
- Stop both whenever idle for >15 min.
