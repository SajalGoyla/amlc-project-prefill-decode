# Disaggregated LLM Serving: Prefill-Decode Splitting over ZMQ

A distributed LLM inference system that disaggregates the **Prefill** and **Decode** phases across two separate GCP instances connected via VPC Peering and ZeroMQ.

## Architecture

```
┌──────────┐     HTTP      ┌─────────────────┐     ZMQ PUSH/PULL      ┌─────────────────┐
│  Client  │──────────────▶│  API Gateway    │                        │                 │
│          │               │  (gateway.py)   │──────HTTP──────────────▶│  Prefill Worker │
└──────────┘               └─────────────────┘                        │  (prefill_      │
                                                                      │   worker.py)    │
                                                                      └────────┬────────┘
                                                                               │
                                                                    ZMQ (TCP)  │  KV-cache
                                                                    layer-by-  │  streaming
                                                                    layer      │
                                                                               ▼
                           ┌─────────────────┐                        ┌─────────────────┐
                           │  Benchmark      │                        │  Decode Worker  │
                           │  (benchmark.py) │                        │  (decode_       │
                           └─────────────────┘                        │   worker.py)    │
                                                                      └─────────────────┘
```

### How It Works

1. **Client** sends a prompt to the **API Gateway**.
2. **Gateway** forwards the request to the **Prefill Worker** (fire-and-forget) and returns `200 OK` immediately.
3. **Prefill Worker** runs the forward pass through LLaMA-3-8B-Instruct. Using `register_forward_hook`, it **streams KV-cache tensors layer-by-layer** over a ZMQ PUSH socket as soon as each layer completes — no waiting for the full forward pass.
4. **Decode Worker** listens on a ZMQ PULL socket, buffers incoming layer tensors, and upon receiving the `PREFILL_COMPLETE` signal, reconstructs `past_key_values` and runs `model.generate()` to produce output tokens.

### Why ZMQ?

The two GCP instances live in **different GCP projects** under a university workspace. Standard GPU interconnects (NVLink, NVSwitch) and NCCL are not available across projects. VPC Peering provides L3 routing, and ZMQ provides a lightweight, high-performance async messaging layer over TCP.

## Repository Structure

```
.
├── infra/
│   └── setup_gcp.sh           # Phase 1: GCP infrastructure provisioning
├── app/
│   ├── gateway.py              # Phase 2A: FastAPI API Gateway
│   ├── prefill_worker.py       # Phase 2B: Prefill Worker with hook-based KV streaming
│   └── decode_worker.py        # Phase 2C: Decode Worker with KV cache reconstruction
├── benchmark/
│   └── benchmark.py            # Phase 3: End-to-end benchmarking script
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── Project Proposal.pdf
└── README.md
```

## Prerequisites

- Two GCP projects with GPU quota for `g2-standard-4` (NVIDIA L4)
- `gcloud` CLI authenticated for both projects
- Python 3.10+
- A HuggingFace token with access to `meta-llama/Meta-Llama-3-8B-Instruct`

## Quick Start

### 1. Provision GCP Infrastructure

```bash
# Edit variables at the top of the script
vim infra/setup_gcp.sh

# Run the setup
chmod +x infra/setup_gcp.sh
bash infra/setup_gcp.sh
```

### 2. Deploy to VMs

SSH into each VM and clone the repo:

```bash
# On BOTH VMs:
gcloud compute ssh <vm-name> --project=<project-id> --zone=<zone>

git clone <your-repo-url> ~/project && cd ~/project
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
vim .env  # Fill in HF_TOKEN, DECODE_WORKER_IP, etc.
```

### 3. Start the Services (in order)

```bash
# Terminal 1 — Decode Worker (start FIRST, it listens for connections)
# On the DECODE VM:
python app/decode_worker.py

# Terminal 2 — Prefill Worker
# On the PREFILL VM:
python app/prefill_worker.py

# Terminal 3 — API Gateway (can run on either VM or locally)
python app/gateway.py
```

### 4. Test It

```bash
# From any machine with network access to the Gateway:
python benchmark/benchmark.py
```

## Environment Variables

| Variable | Description | Example |
|---|---|---|
| `HF_TOKEN` | HuggingFace access token | `hf_abc123...` |
| `PREFILL_WORKER_HOST` | IP/hostname of the Prefill VM | `10.128.0.2` |
| `PREFILL_WORKER_PORT` | Port for Prefill FastAPI server | `8001` |
| `DECODE_WORKER_HOST` | IP/hostname of the Decode VM | `10.128.0.3` |
| `DECODE_WORKER_PORT` | Port for Decode FastAPI server | `8002` |
| `ZMQ_PORT` | Port for ZMQ KV-cache streaming | `5555` |
| `GATEWAY_HOST` | IP/hostname for the API Gateway | `0.0.0.0` |
| `GATEWAY_PORT` | Port for Gateway FastAPI server | `8000` |

## End-to-End Testing Guide

See the detailed walkthrough below for step-by-step instructions.

### Step 1: Verify Network Connectivity

```bash
# From Prefill VM, ping Decode VM internal IP:
ping <DECODE_INTERNAL_IP>

# From Decode VM, ping Prefill VM internal IP:
ping <PREFILL_INTERNAL_IP>

# Test ZMQ port is reachable (from Prefill VM):
nc -zv <DECODE_INTERNAL_IP> 5555
```

### Step 2: Smoke Test (without GPU)

You can run a quick connectivity test without loading models:

```bash
# On Decode VM — start in test mode
python app/decode_worker.py --test-zmq

# On Prefill VM — send a test message
python -c "import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.PUSH); s.connect('tcp://<DECODE_IP>:5555'); s.send_string('hello'); print('sent')"
```

### Step 3: Full End-to-End Run

```bash
# 1. Start Decode Worker (on Decode VM)
python app/decode_worker.py

# 2. Start Prefill Worker (on Prefill VM)
python app/prefill_worker.py

# 3. Start Gateway (on Prefill VM or separate machine)
python app/gateway.py

# 4. Run benchmark (from any machine)
python benchmark/benchmark.py --gateway-url http://<GATEWAY_IP>:8000
```

### Step 4: Interpret Results

The benchmark script will print:
- **TTFT (Time To First Token)**: Latency from request submission to first token generation
- **TPOT (Time Per Output Token)**: Average latency per generated token
- **Total E2E Latency**: Full round-trip time
- **Tokens Generated**: Number of output tokens

## Troubleshooting

| Issue | Solution |
|---|---|
| VPC Peering blocked | Check GCP org policies; request IT override for `constraints/compute.restrictVpcPeering` |
| ZMQ connection timeout | Verify firewall rules allow TCP on port 5555; check VPC peering status |
| CUDA OOM | Reduce `max_new_tokens` or use `torch.float16` (already default) |
| HuggingFace gated model | Ensure `HF_TOKEN` has accepted the LLaMA-3 license agreement |

## License

Academic use — Columbia University COMS 6998 (AMLC) course project.
