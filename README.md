# AMLIC Project: Prefill-Decode Disaggregated Inference Serving

A distributed LLM inference system proving that disaggregating the **Prefill** and **Decode** phases across two physically separated GPU servers (connected via VPC Peering and ZeroMQ) prevents Long-Prompt Interference and radically secures generation TPOT (Time-Per-Output-Token) scale at the cost of upfront networking/serialization latency.

By bypassing standard frameworks and streaming raw PyTorch KV caches over a ZMQ TCP socket, we successfully served `meta-llama/Meta-Llama-3-8B-Instruct` across two GCP L4 VM instances connected over a custom VPC Peering fabric. The results demonstrate an ~80x improvement in True TTFT under load while keeping TPOT rock-solid.

## 📁 Code Structure

- **`app/`**: Contains the core serving nodes.
  - `collocated_baseline.py`: A unified single-GPU FastAPI worker performing both prefill and decode natively. Used to measure the baseline.
  - `gateway.py`: A lightweight FastAPI gateway routing client requests asynchronously to the Prefill worker.
  - `prefill_worker.py`: Runs the prompt prefill phase, extracts `past_key_values` from the forward pass, and streams them tensor-by-layer over ZMQ to the Decode worker.
  - `decode_worker.py`: Maintains a ZMQ PULL socket, buffers incoming KV tensors, reconstructs the HuggingFace `DynamicCache`, and runs `.generate()`.
- **`benchmark/`**: Analytics and load testing suite.
  - Scripts including `compare_architectures.py`, `concurrency_sweep.py`, and `prompt_sweep.py` bombard the endpoints and calculate True TTFT, TPOT, network overhead, and end-to-end latencies.
- **`infra/`**: Helper scripts (`setup_gcp.sh`, `setup_nat.sh`) necessary for automating the GCP and custom VPC peering architecture to guarantee 1ms RTT cross-node networking.

## ⚙️ Setup Instructions

### Prerequisites
- Two distinct GCP VMs with **NVIDIA L4 GPUs** (24GB VRAM each).
- Both VMs must be connected via a peered VPC internal network (required to hit sub-millisecond / 1ms RTT).
- Python 3.10+ environment.
- A valid Hugging Face account token with access to `meta-llama/Meta-Llama-3-8B-Instruct`.

### Installation
Clone the repository and install requirements on **both** VMs:
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
*Crucial*: Ensure you set your `HF_TOKEN`. On the Prefill VM, `DECODE_WORKER_HOST` must be accurately configured to point to the internal VPC IP of the Decode VM.

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

## 📊 Key Results & Findings

- **Concurrent Load Resiliency**: Under a concurrent load of 10 requests, the collocated TTFT degrades to an unusable 37.5 seconds due to destructive prefill/decode interference. Conversely, the disaggregated system stays highly responsive at ~0.4 seconds, an **~80x latency improvement**.
- **Stable TPOT**: Time-Per-Output-Token remains rock-solid at ~64ms across both architectures, validating that architectural decoupling does not harm generation speed.
- **The Network/Serialization Tax**: Moving KV cache layers across a 1ms RTT VPC network utilizing Python Pickle serialization scales linearly from ~30ms at 50 prompt tokens to ~350ms at 2000 prompt tokens. While VPC peering minimized network constraints, standard tensor serialization formats still pose a substantial transfer overhead.
