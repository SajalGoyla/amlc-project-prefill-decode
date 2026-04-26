# Disaggregated LLM Serving Infrastructure - Handoff Context

**Date Updated**: April 2026
**Target Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
**Infrastructure**: Google Cloud Platform (GCP). Two isolated `g2-standard-4` VMs (1x NVIDIA L4 GPU each) spread across two distinct projects with VPC Network Peering.

---

## 1. Project Objective
The goal is to implement and benchmark a true **Disaggregated LLM Serving** framework. In normal inference (collocated), a single GPU manages both the "spiky", compute-heavy **Prefill phase** and the memory-bandwidth heavy **Decode phase**. 
This project separates them: 
- **GPU 1 (Prefill Worker)** computes the initial prompt and generates the raw KV cache.
- **GPU 2 (Decode Worker)** consumes the KV cache over a network socket and streams autogressive tokens. 
By decoupling these, we prevent massive user prompts from causing Time-Per-Output-Token (TPOT) spikes for users currently actively generating tokens in the decoding batch.

---

## 2. Codebase Architecture

The application runs heavily on Python, PyTorch, FastApi, and ZeroMQ (ZMQ) for rapid peer-to-peer data streaming.

### Core Files
- **`app/gateway.py`**: A FastAPI routing layer. Accepts synchronous user HTTP `/generate` requests and delegates them as background tasks to the Prefill worker to prevent blocking.
- **`app/prefill_worker.py`**: Houses the LLM. Executes the forward pass (`use_cache=True`). It extracts the full KV state directly from `outputs.past_key_values` and streams all 32 transformer layers securely over a ZMQ push socket (Port `5555`) to the Decode VM via background threading multiplexing.
- **`app/decode_worker.py`**: Contains an identical clone of the LLM. Runs a background ZMQ pull socket. Once 32 KV layers arrive, it dynamically instantiates a generic `transformers.cache_utils.DynamicCache` object, aggressively injects the transferred layer tensors utilizing `.update(k,v)`, and slides the completed object straight into `model.generate()`.
- **`app/collocated_baseline.py`**: A standard, single-GPU FastAPI runner mapping the entire procedure natively. Built purely to serve as a strict baseline for analyzing comparative benchmarks against the Disaggregated network setup overhead.

### Benchmarking Suite (`benchmark/`)
- **`benchmark.py`**: Basic sanity checker plotting simple end-to-end Gateway latencies.
- **`concurrent_workload.py`**: `asyncio`-operated bombardier script targeting the HTTP endpoint strictly to prove concurrency scaling without tanking generation TPOT.
- **`compare_architectures.py`**: The definitive test file. Utilizes automated metrics probing across both the Collocated API and Disaggregated architectures to calculate programmatic metric offsets (TTFT vs TPOT) securely.

### Infrastructure Automation (`infra/`)
- **`setup_gcp.sh`**: Completely automates VPC routing generation, Cloud NATs (For internet egress access), and intelligently probes exact regions looking for L4 GPU quotas.
- VMs utilize `--no-address` flags strict compliance—traffic runs exclusively via IAP proxies tunnels.

---

## 3. Key Technical Decisions & Bug Fixes Solved

If you are modifying the pipeline, you **must understand** how the cache object is managed to prevent crashing:
- **Hook Avoidance**: We explicitly do *not* use `register_forward_hook` to slice KV values inside `prefill_worker` because the newest huggingface `transformers` releases utilize dynamic cache objects internally which do not yield valid matrices inside standard tuple module outputs during individual layer evaluations. We extract post-forward directly.
- **DynamicCache Structure Reconstruction**: `transformers` frequently update their cache backend methods. The decode worker used to look for `DynamicCache.from_legacy_cache()` or `cache.key_cache`. **Do not do this**. To rebuild safely across versions, `decode_worker.py` iterates `enumerate(past_key_values)` locally and mounts layers inside the cache framework manually via `cache.update(key_tensor, value_tensor, layer_idx)`.

---

## 4. Current State & Next Steps

The entire disaggregated data pipeline functions seamlessly. The user has successfully transmitted overlapping massive 512-token generations across the VPC securely maintaining an isolated ~65ms local TPOT.

**Right Now**:
The User is transitioning into the final testing phase: **Evaluating the metrics**. 
They are currently running `app/collocated_baseline.py` on the Decode VM and `app/gateway.py` / `app/prefill_worker.py` on the Prefill VM and pointing `benchmark/compare_architectures.py` at the respective HTTP endpoints to generate the comparative report graphs to prove system viability!

If instructed to assist with debugging the benchmarking loops or fixing potential scaling issues, refer directly to `benchmark/compare_architectures.py` execution flows!
