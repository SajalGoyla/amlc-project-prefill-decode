# AMLIC — Adaptive Multi-tier LLM Inference on Cloud

> Columbia COMS 6998 (Cloud Computing) — Final Project

**Adaptive LLM Inference via Prefill/Decode Disaggregation on Heterogeneous GPUs**

## Overview

This project benchmarks **collocated** vs **disaggregated** (prefill/decode) LLM serving using [vLLM](https://github.com/vllm-project/vllm) with the [NIXL](https://github.com/ai-dynamo/nixl) connector over a Tailscale overlay network. The goal is to find the empirical **prompt-length crossover threshold N**, below which collocated serving is faster and above which disaggregated serving wins.

### Architecture

```
┌─────────────────┐         ┌──────────────────────────────────┐
│   User / Demo   │         │         Adaptive Router          │
│   (Streamlit)   │────────▶│  token_count ≥ N? → disagg      │
│                 │         │  token_count < N? → collocated   │
└─────────────────┘         └──────────┬───────────┬───────────┘
                                       │           │
                            ┌──────────▼──┐   ┌────▼───────────────────────┐
                            │ Collocated  │   │   Disaggregated Proxy     │
                            │  vLLM (L4)  │   │   (FastAPI)               │
                            │  Port 8000  │   │   Port 9000               │
                            └─────────────┘   └────┬──────────┬───────────┘
                                                   │          │
                                          ┌────────▼──┐  ┌────▼────────┐
                                          │  Prefill  │  │   Decode    │
                                          │ vLLM (L4) │  │  vLLM (T4) │
                                          │ Port 8100 │──│  Port 8200  │
                                          │ KV Prod.  │  │  KV Cons.  │
                                          └───────────┘  └────────────┘
                                              NIXL/UCX over Tailscale
```

### Hardware

| Role | GPU | VRAM | VM Type | 
|------|-----|------|---------|
| Prefill + Collocated | NVIDIA L4 | 24 GB | GCP g2-standard-4 |
| Decode | NVIDIA T4 | 16 GB | GCP n1-standard-4 + T4 |

### Model

- `meta-llama/Llama-3.2-3B-Instruct` (3B params, fits in 16 GB with room for KV cache)

## Repository Structure

```
Project/
├── .env.example           # Environment template
├── pyproject.toml         # Python project config
├── README.md              # This file
│
├── infra/                 # VM infrastructure scripts
│   ├── vm_bootstrap.sh    # Idempotent VM setup
│   ├── start_collocated.sh
│   ├── start_prefill.sh
│   ├── start_decode.sh
│   └── proxy_server.py    # Disaggregated proxy
│
├── benchmark/             # Benchmarking framework
│   ├── client.py          # Async OpenAI-compatible client
│   ├── workloads.py       # Workload profiles + sweep
│   ├── runner.py          # Benchmark orchestrator
│   ├── prompts/           # Source text for prompt generation
│   └── results/           # CSV results (gitignored, except .gitkeep)
│
├── analysis/              # Data analysis
│   ├── compute_threshold.py  # Find crossover N
│   ├── plot_results.py    # Generate plots
│   └── figures/           # Output plots
│
├── router/                # Adaptive router
│   └── router.py          # FastAPI router
│
├── demo/                  # Streamlit demo
│   └── app.py             # Interactive chatbot + comparison
│
└── scripts/               # Helper scripts
    ├── run_full_benchmark.sh
    ├── health_check.sh
    └── net_benchmark.sh
```

## Quick Start

### 1. Setup (Laptop)

```bash
# Clone and install local dependencies
cd Project/
cp .env.example .env
# Edit .env with your HF_TOKEN, VM IPs, etc.

pip install -e ".[dev]"
```

### 2. Setup (VMs)

```bash
# On each VM:
scp -r infra/ user@VM_IP:~/amlic/
ssh user@VM_IP
cd ~/amlic
bash infra/vm_bootstrap.sh
```

### 3. Install Tailscale (both VMs)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
# Note the Tailscale IPs and update .env
```

### 4. Start Services

```bash
# ON PREFILL VM (L4):
bash infra/start_collocated.sh   # Collocated baseline
bash infra/start_prefill.sh      # Prefill engine (in another tmux)
python infra/proxy_server.py \
    --prefiller-host $(tailscale ip -4) --prefiller-port 8100 \
    --decoder-host <DECODE_TAILSCALE_IP> --decoder-port 8200

# ON DECODE VM (T4):
bash infra/start_decode.sh
```

### 5. Run Benchmarks

```bash
# ON LAPTOP:
bash scripts/run_full_benchmark.sh 30 1
# Or run individual profiles:
python -m benchmark.runner --endpoint http://<IP>:8000/v1 --arch collocated --profile chat
```

### 6. Analyze Results

```bash
python -m analysis.compute_threshold --results-dir benchmark/results/
python -m analysis.plot_results --results-dir benchmark/results/
```

### 7. Launch Demo

```bash
streamlit run demo/app.py
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token (ms) — user-perceived responsiveness |
| **ITL** | Inter-token latency (ms) — streaming smoothness |
| **Throughput** | Output tokens per second |
| **Total Latency** | End-to-end request latency (ms) |

## Workload Profiles

| Profile | Input Tokens | Output Tokens | Use Case |
|---------|-------------|---------------|----------|
| `chat` | 50 | 500 | Short prompt, long generation |
| `doc_qa` | 2000 | 100 | Long document, short answer |
| `balanced` | 500 | 200 | Medium both |
| `sweep_N` | Variable | 100 | Prompt-length sweep for threshold |

## Fallback Plan

If NIXL/UCX over Tailscale proves unstable:
1. Switch to **LMCacheConnector** (uses Redis for KV transfer)
2. Install Redis on both VMs
3. Update `kv_connector` in start scripts

## License

Academic project — Columbia University COMS 6998.
