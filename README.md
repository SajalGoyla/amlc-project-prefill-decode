# AMLC Project: Prefill-Decode Disaggregated Inference Serving

A distributed LLM inference system proving that disaggregating the **Prefill** and **Decode** phases across two physically separated GPU servers (connected via VPC Peering and ZeroMQ) prevents Long-Prompt Interference and radically secures generation TPOT (Time-Per-Output-Token) scale at the cost of upfront networking/serialization latency.

By bypassing standard frameworks and streaming raw PyTorch KV caches over a ZMQ TCP socket, we successfully served `meta-llama/Meta-Llama-3-8B-Instruct` across two GCP L4 VM instances connected over a custom VPC Peering fabric. The results demonstrate an ~80x improvement in True TTFT under load while keeping TPOT rock-solid.

## 🏗️ Architecture

```text
┌─────────────────┐       ┌───────────────────────────────────┐
│  Client / User  │──────▶│         API Gateway (FastAPI)     │
└─────────────────┘       │             Port: 8000            │
                          └─────────────────┬─────────────────┘
                                            │ HTTP POST (Prompt)
                          ┌─────────────────▼─────────────────┐
                          │          Prefill Worker           │
                          │   (L4 GPU - 24GB, Port: 8001)     │
                          │   Generates past_key_values       │
                          └─────────────────┬─────────────────┘
                                            │ ZMQ PUSH (TCP: 5555)
                                            │ Streams layer tensors 
                          ┌─────────────────▼─────────────────┐
                          │           Decode Worker           │
                          │   (L4 GPU - 24GB, Port: 8002)     │
                          │    Rebuilds DynamicCache ->       │
                          │    Autoregressive Generation      │
                          └───────────────────────────────────┘
```

## 📁 Code Structure

```text
├── app/
│   ├── collocated_baseline.py  # Unified single-GPU worker for baseline measurements
│   ├── gateway.py              # FastAPI gateway routing requests asynchronously
│   ├── prefill_worker.py       # Prompt prefill & ZMQ tensor streaming
│   └── decode_worker.py        # ZMQ PULL socket & autoregressive decoding
├── benchmark/
│   ├── benchmark.py            # End-to-end latency sanity checker
│   ├── compare_architectures.py # Definitive test for metric offsets (TTFT vs TPOT)
│   ├── concurrency_sweep.py    # Analyzes varying concurrent loads
│   ├── concurrent_workload.py  # asyncio bombardier for concurrency scaling
│   ├── plot_results.py         # Visualizes CSV outputs
│   └── prompt_sweep.py         # Tests serialization impact across prompt lengths
└── infra/
    ├── setup_gcp.sh            # Automates VPC, Firewalls, Peering, and VM creation
    └── setup_nat.sh            # Provisions Cloud NAT for VMs without external IPs
```

## ⚙️ Setup Instructions

### End-to-End Infrastructure Setup (GCP)
To achieve the 1ms RTT required for efficient ZMQ streaming, the architecture relies on **VPC Peering** between two isolated networks. We provide automated scripts to handle this entire process, or you can perform the steps manually.

#### Option 1: Automated Setup (Recommended)
The repository includes robust infrastructure automation scripts in the `infra/` directory.

1. **Authenticate gcloud**: Ensure you have the `gcloud` CLI installed and authenticated.
2. **Configure Projects**: Open `infra/setup_gcp.sh` and set `PREFILL_PROJECT` and `DECODE_PROJECT` to your active GCP project IDs.
3. **Run the Provisioning Script**:
   ```bash
   bash infra/setup_gcp.sh
   ```
   *What this script does:*
   - **GPU Sweeping**: Scans GCP regions to find zones with available `nvidia-l4` GPUs, intelligently attempting to place both VMs in the **same region** to guarantee minimal latency.
   - **VPC & Firewalls**: Creates custom VPC networks (`prefill-vpc` and `decode-vpc`), subnets, and configures firewall rules to allow internal traffic and IAP SSH access.
   - **VPC Peering**: Establishes bidirectional VPC network peering between the two projects, bypassing public internet routing.
   - **VM Creation**: Deploys `g2-standard-4` instances with L4 GPUs and PyTorch images. The instances are created without external IPs (`--no-address`) to comply with strict organizational security policies.
   - **.env Generation**: Automatically generates a `.env` file in your local project root mapped with the discovered internal IPs.

4. **Run the NAT Setup**:
   Because the VMs lack external IPs, they require a Cloud NAT gateway to download HuggingFace models and pip packages.
   ```bash
   bash infra/setup_nat.sh
   ```

5. **SSH into the VMs**:
   Connect using Identity-Aware Proxy (IAP) tunnels:
   ```bash
   gcloud compute ssh prefill-worker-vm --project=YOUR_PREFILL_PROJECT --zone=YOUR_ZONE --tunnel-through-iap
   gcloud compute ssh decode-worker-vm --project=YOUR_DECODE_PROJECT --zone=YOUR_ZONE --tunnel-through-iap
   ```

#### Option 2: Manual Setup via GCP Console
If you prefer to configure the environment manually:
1. **Create Custom VPCs**: Go to VPC Network and create `prefill-vpc` and `decode-vpc`. Ensure their subnet CIDR blocks **do not overlap** (e.g., `10.10.0.0/20` and `10.20.0.0/20`).
2. **Establish VPC Peering**: 
   - In the Prefill project, go to VPC Network Peering -> Create Connection. Target the Decode project ID and `decode-vpc`.
   - In the Decode project, create the reciprocal peering connection targeting the Prefill project and `prefill-vpc`.
3. **Configure Firewalls**: Create Ingress rules in both VPCs allowing traffic from `10.0.0.0/8` for all ports, and allow `35.235.240.0/20` on port 22 to permit IAP SSH connections.
4. **Deploy Cloud NAT**: Go to Cloud Router -> Create Router -> Cloud NAT for both VPCs. This grants outbound internet access.
5. **Create Instances**: Launch two `g2-standard-4` instances (one in each VPC). Attach 1x NVIDIA L4 GPU, select a PyTorch Deep Learning OS image, and expand the Networking tab to set "External IPv4 address" to **None**.

### Application Installation
Once connected to **both** VMs via SSH, clone the repository and install requirements:
```bash
git clone <YOUR_REPOSITORY_URL> ~/amlc-project-prefill-decode
cd ~/amlc-project-prefill-decode
pip3 install -r requirements.txt
```

Configure your environment variables:
```bash
cp .env.example .env
nano .env 
```
*Crucial*: Ensure you set your `HF_TOKEN` to a valid Hugging Face token with LLaMA-3 access. On the Prefill VM, `DECODE_WORKER_HOST` must be accurately configured to point to the internal VPC IP of the Decode VM.

## 🚀 How to Run

### Phase 1: Obtaining the Collocated Baseline
1. **On your Decode VM**, launch the unified benchmarking baseline:
```bash
python3 app/collocated_baseline.py
```
2. **On your Prefill VM**, run the benchmark client to bombard the baseline node:
```bash
python3 benchmark/compare_architectures.py \
   --concurrent 3 \
   --collocated-url http://<DECODE_INTERNAL_IP>:8000 \
   --skip-disaggregated
```

### Phase 2: Launching the Dual-GPU Disaggregated Rig
*Note: Ensure `collocated_baseline.py` is fully terminated (Ctrl+C) on the Decode VM before proceeding to avoid Out-Of-Memory crashes, as L4 GPUs are limited to 24GB VRAM.*

1. **Start the Decode Worker (Decode VM)**
```bash
python3 app/decode_worker.py
```

2. **Start the Prefill Engine Array (Prefill VM)**
In Terminal 1 (Gateway):
```bash
python3 app/gateway.py
```
In Terminal 2 (Prefill Worker):
```bash
python3 app/prefill_worker.py
```

3. **Generate Final Metrics Overlay (Prefill VM)**
Run the benchmarker isolating only the disaggregated logic:
```bash
python3 benchmark/compare_architectures.py \
   --concurrent 3 \
   --gateway-url http://localhost:8000 \
   --decode-url http://<DECODE_INTERNAL_IP>:8002 \
   --skip-collocated
```

## 🧩 How It Works
1. **Collocated**: A single GPU runs both prefill (processing the prompt) and decode (generating tokens). Under concurrent load, prefill and decode contend for GPU compute cycles causing head-of-line blocking.
2. **Disaggregated**: GPU 1 runs prefill only. After prefill, the raw PyTorch KV cache (`past_key_values`) is extracted from the model and serialized over a ZeroMQ socket to GPU 2. This eliminates prefill-decode contention but adds KV network transfer overhead.
3. **Gateway**: A lightweight FastAPI gateway (`app/gateway.py`) accepts synchronous `/generate` requests and delegates them as background tasks to the Prefill worker, preventing the client connection from blocking.

## 📊 Key Results & Findings

- **Concurrent Load Resiliency**: Under a concurrent load of 10 requests, the collocated TTFT degrades to an unusable 37.5 seconds due to destructive prefill/decode interference. Conversely, the disaggregated system stays highly responsive at ~0.4 seconds, an **~80x latency improvement**.
- **Stable TPOT**: Time-Per-Output-Token remains rock-solid at ~64ms across both architectures, validating that architectural decoupling does not harm generation speed.
- **The Network/Serialization Tax**: Moving KV cache layers across a 1ms RTT VPC network utilizing Python Pickle serialization scales linearly from ~30ms at 50 prompt tokens to ~350ms at 2000 prompt tokens. While VPC peering minimized network constraints, standard tensor serialization formats still pose a substantial transfer overhead.

## 🛠️ Troubleshooting

- **CUDA Out Of Memory (OOM)**: The NVIDIA L4 GPU has 24GB VRAM. If you run the `decode_worker.py` while `collocated_baseline.py` is still alive, the GPU will crash. Ensure you explicitly kill the baseline process before initializing the disaggregated worker.
- **High TTFT (Slow Connection)**: If your TTFT is significantly lagging, verify that your `.env` is utilizing **internal IPs** for `DECODE_WORKER_HOST` and not external IPs. Using external IPs routes traffic out of Google Cloud and breaks the 1ms RTT peering advantage.
- **ZMQ Connection Timeouts**: Ensure your custom VPC firewalls actively allow TCP traffic on port `5555` (used by ZeroMQ). Additionally, confirm that your VPC Peering status is marked as `ACTIVE` in both GCP projects.
- **HuggingFace 401 Unauthorized**: Ensure your `HF_TOKEN` in the `.env` file is valid and your HuggingFace account explicitly has granted access to use `meta-llama/Meta-Llama-3-8B-Instruct`.
