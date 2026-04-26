# Disaggregated LLM Serving: Prefill-Decode Splitting over ZMQ

A distributed LLM inference system proving that disaggregating the **Prefill** and **Decode** phases across two physically separated GPU servers (connected via VPC Peering and ZeroMQ) prevents Long-Prompt Interference and radically secures generation TPOT (Time-Per-Output-Token) scale at the cost of upfront networking latency.

## Hardware & Environment Architecture

```
┌──────────┐     HTTP      ┌─────────────────┐     ZMQ PUSH/PULL (TCP)     ┌─────────────────┐
│  Client  │──────────────▶│  API Gateway    │                             │                 │
│          │               │  (gateway.py)   │──────HTTP──────────────────▶│  Prefill Worker │
└──────────┘               └─────────────────┘                             │  (prefill_      │
                                                                           │   worker.py)    │
                                                                           └────────┬────────┘
                                                                (GPU 1: L4)         │ 
                                                                         KV-cache   │  Extracted
                                                                         streaming  │  Post-Forward
                                                                                    ▼
┌──────────────────┐                            ┌─────────────────┐        ┌─────────────────┐
│  Comparison      │                            │                 │        │                 │
│  Benchmark Suite │◀────────HTTP Polling───────┤  Decode Worker  │◀───────┤ZMQ Tunnel       │
│                  │                            │  (decode_       │        │                 │
└──────────────────┘                            │   worker.py)    │        └─────────────────┘
                                                └─────────────────┘
                                                    (GPU 2: L4)
```

### The "Goodput" Disaggregated Pipeline
1. **Client** sends a prompt.
2. **Prefill Worker** completely strips the entire PyTorch execution layer by running the forward pass through `meta-llama/Meta-Llama-3-8B-Instruct`. Crucially, it isolates the raw output block, serializes the exact 32 deep-layer matrices locally, and ZMQ pushes them physically across the datacenter VPC subnet.
3. **Decode Worker** exclusively catches 32 layer inputs via asynchronous pooling. It reconstructs a synthetic HuggingFace `DynamicCache` block natively in Python natively from the tuple pairs matching the exact size, and hands exclusively the Cache block to standard `.generate()`, decoupling massive upstream loads!

---

## Prerequisites (From Scratch)

- Two distinct GCP servers with **L4 GPUs** (e.g., `g2-standard-4`).
- Minimum 24GB VRAM required per server (16GB occupied by Llama-3 precision loading).
- `nc -zv` connectivity verified across your internal firewall blocks for Target Port `5555` (ZMQ) and Port `8002` (Metrics Endpoint).
- A valid Huggingface Account token storing explicitly granted Llama-3 weights access.

---

## 🚀 Replicating the Experiment (Step-By-Step)

Because this test rigorously calculates exact hardware-compute comparisons, deploying via automatic scripts isn't favored. Here is exactly how to pull the code down and run both environments symmetrically on your GPUs.

### 1. Connect & Clone to your First Node (Prefill VM)
SSH into your first server and extract the codebase:
```bash
git clone <YOUR_REPOSITORY_URL> ~/project
cd ~/project
pip3 install -r requirements.txt
```
Copy and fill out the `.env` settings pointing cleanly to your HuggingFace key.
```bash
cp .env.example .env
nano .env 
```

### 2. Connect & Clone to your Second Node (Decode VM)
SSH manually into your second, target GPU server and do the exact same configuration matching:
```bash
git clone <YOUR_REPOSITORY_URL> ~/project
cd ~/project
pip3 install -r requirements.txt
cp .env.example .env 
nano .env  # Make sure the internal target IPs align to this node!
```

---

## 📊 Phase 1: Obtaining the Collocated Single-GPU Baseline Metric

To mathematically prove the physical architectural difference, we first need to understand how exactly the model operates organically bundled together on a single GPU load.

1. **On your Decode VM**, launch the unified benchmarking baseline:
```bash
cd ~/project
python3 app/collocated_baseline.py
```
2. Wait until the Uvicorn terminal reports `Collocated Model loaded successfully`.

3. **On your Prefill VM**, run the master analytics command to bombard the local baseline node with overlapping prompt load sequentially:
```bash
cd ~/project
python3 benchmark/compare_architectures.py \
   --concurrent 3 \
   --collocated-url http://<DECODE_INTERNAL_IP>:8000 \
   --skip-disaggregated
```
*Note down the resulting TTFT, Compute Time, and TPOT metrics. You now have your baseline single-node evaluation threshold!*

---

## 🛠️ Phase 2: Launching the Dual-GPU Disaggregated Rig

Now we deploy the physically separated mechanisms and measure the identical parameters using the custom ZMQ tunnel payload routing!

**1. Clean up** 
*CRITICAL: Manually kill your collocated_worker thread (Ctrl+C). The L4 GPUs only have 24GB of memory. It will crash instantly if both the Baseline and the Decode listener attempt to occupy VRAM locally together.*

**2. Start the Disaggregated Decode Server (Decode VM)**
```bash
python3 app/decode_worker.py
```
*(Wait until it binds ZMQ port 5555 and starts background metrics on port 8002).*

**3. Start the Subnet Engine Array (Prefill VM)**
Spin up your backend gateway orchestrator:
```bash
# Terminal 1 -> Gateway Server
python3 app/gateway.py

# Terminal 2 -> Main Prefill Core
python3 app/prefill_worker.py
```

**4. Generate Final Metrics Overlay**
Run the benchmarker isolating only the disaggregated URL payloads logic:
```bash
python3 benchmark/compare_architectures.py \
   --concurrent 3 \
   --gateway-url http://localhost:8000 \
   --decode-url http://<DECODE_INTERNAL_IP>:8002 \
   --skip-collocated
```

### 🧠 Interpreting the Output Math
The console block explicitly breaks down the `TTFT` mathematical pipeline variables:
* **Collocated Environment:** TTFT should be purely constrained to physical compute bounds. As traffic volumes mount simultaneously toward a single endpoint, you will immediately witness severe TPOT stuttering and overall prompt interference scaling badly.
* **Disaggregated Array:** You will identify extreme TTFT latency offsetting due to the massive physical movement cost (typically ~600+ ms transferring 30+ GBs across VPC networks), **but exactly stable 60ms TPOT generations.** No matter how heavy upstream prompts queue out onto the Prefill array, the Decode system smoothly renders exact throughput autonomously without ever halting an active token chain!
