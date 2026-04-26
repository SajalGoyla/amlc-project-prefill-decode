#!/usr/bin/env python3
"""
amlic_benchmark.py — Collocated vs Disaggregated latency benchmark.

Adapted for this project's custom polling API (not OpenAI-compatible).

Metrics collected
-----------------
  collocated:    ttft_ms (server-side), tpot_ms (server-side),
                 e2e_ms (client wall-clock), tokens
  disaggregated: ttft_ms (client: POST → prefill_accepted on gateway),
                 tpot_ms (server-side from decode worker),
                 forward_ms (GPU prefill compute time, from decode worker),
                 kv_overhead_ms (ttft_ms − forward_ms = network + queue),
                 e2e_ms (client wall-clock), tokens

Usage
-----
  # Collocated only (run on decode VM or any machine with access)
  python benchmark/amlic_benchmark.py --mode collocated \\
      --collocated-url http://<DECODE_IP>:8000 --runs 3 --max-tokens 128

  # Disaggregated only
  python benchmark/amlic_benchmark.py --mode disaggregated \\
      --gateway-url http://<PREFILL_IP>:8000 \\
      --decode-url http://<DECODE_IP>:8002 --runs 3 --max-tokens 128

  # Both (sequential), produces side-by-side comparison table
  python benchmark/amlic_benchmark.py --mode both \\
      --collocated-url http://<DECODE_IP>:8000 \\
      --gateway-url http://<PREFILL_IP>:8000 \\
      --decode-url http://<DECODE_IP>:8002

  # Run a subset of prompts (comma-separated labels)
  python benchmark/amlic_benchmark.py --mode both --prompts P1_tiny,P3_medium,P5_xlarge
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

# ==============================================================================
# Prompts — 7 sizes from tiny to very long
# ==============================================================================

PROMPTS = [
    {
        "label": "P1_tiny",
        "text": "Explain what a neural network is in simple terms.",
    },
    {
        "label": "P2_small",
        "text": (
            "Explain the difference between supervised and unsupervised learning "
            "in machine learning, giving two concrete examples of each type of "
            "approach and explaining what kinds of problems each is best suited "
            "to solve."
        ),
    },
    {
        "label": "P3_medium",
        "text": (
            "You are an expert machine learning engineer. Explain in detail how "
            "the transformer architecture works, covering the self-attention "
            "mechanism, multi-head attention, positional encoding, the encoder "
            "and decoder stacks, layer normalization, feed-forward networks, and "
            "how all these components combine to process sequential data. Include "
            "the mathematical intuition behind scaled dot-product attention."
        ),
    },
    {
        "label": "P4_large",
        "text": (
            "You are an expert in the history of computing, mathematics, and "
            "artificial intelligence. Provide an exhaustive and deeply technical "
            "account of the intellectual lineage of modern large language models, "
            "beginning with Alan Turing and McCulloch-Pitts (1943-1950), through "
            "backpropagation (1986), support vector machines (1990s), ImageNet "
            "and AlexNet (2012), word2vec (2013), seq2seq and Bahdanau attention "
            "(2014-2015), the Transformer (Vaswani et al., 2017), BERT and GPT "
            "(2018-2019), scaling laws (Kaplan et al., 2020), RLHF and "
            "InstructGPT, and the current frontier of multimodal and reasoning "
            "models."
        ),
    },
    {
        "label": "P5_xlarge",
        "text": (
            "You are a professor teaching a graduate seminar on distributed "
            "systems and ML infrastructure. Write a comprehensive lecture "
            "covering: (1) GPU memory bandwidth as a bottleneck in autoregressive "
            "decoding and why it worsens at scale; (2) KV cache growth with "
            "sequence length and batch size; (3) prefill-decode disaggregation "
            "motivation from DistServe and Splitwise with theoretical performance "
            "benefits; (4) engineering challenges of KV cache transfer between "
            "nodes; (5) homogeneous vs heterogeneous GPU configurations; (6) "
            "production disaggregated serving at Meta, LinkedIn, and Mistral "
            "using vLLM; (7) the role of NIXL, UCX, and LMCache; (8) open "
            "research questions on dynamic routing and the crossover threshold "
            "N* where disaggregation becomes beneficial."
        ),
    },
    {
        "label": "P6_long",
        "text": (
            "You are a world-class expert in computer architecture, distributed "
            "systems, ML infrastructure, and GPU programming. Write an exhaustive "
            "technical treatise covering: (1) the von Neumann bottleneck in "
            "modern GPU architectures and bandwidth numbers for H100 NVLink vs "
            "PCIe vs DDR5; (2) arithmetic intensity of prefill vs decode phases "
            "derived from first principles (exact FLOP counts per attention layer "
            "as a function of S, D, H), explaining why prefill is compute-bound "
            "and decode is memory-bandwidth-bound; (3) PagedAttention in vLLM "
            "and how continuous batching eliminates fragmentation; (4) the NIXL "
            "library architecture using UCX and why compatibility hash validation "
            "fails for heterogeneous GPU architectures; (5) KV cache transfer "
            "approaches — NVLink, RDMA/InfiniBand, TCP/Ethernet, Redis — with "
            "theoretical bandwidth, typical latency, and minimum prompt length "
            "crossover for each approach."
        ),
    },
    {
        "label": "P7_vlong",
        "text": (
            "You are simultaneously a professor of computer science specializing "
            "in distributed systems, a GPU architect at NVIDIA, a researcher at "
            "a leading AI lab working on LLM inference optimization, and a cloud "
            "infrastructure engineer who has deployed large language models at "
            "scale. Write the most comprehensive possible technical document "
            "covering: the complete history of parallel computing from Flynn "
            "taxonomy through SIMD through GPGPU to modern tensor cores; the "
            "full mathematical framework for transformer attention from Bahdanau "
            "through scaled dot-product through MHA through GQA with memory "
            "access patterns and operational intensity in FLOPS per byte; why "
            "prefill and decode have fundamentally different computational "
            "characteristics with full roofline model analysis; the complete "
            "vLLM design including PagedAttention, continuous batching, chunked "
            "prefill, preemption, tensor parallelism, and pipeline parallelism; "
            "the disaggregated prefill-decode architecture (DistServe, Splitwise) "
            "with the full mathematical model for when disaggregation is "
            "beneficial and derivation of the crossover threshold N*; NIXL and "
            "UCX engineering challenges for heterogeneous GPUs and what a "
            "heterogeneous-compatible transfer module would require; the LMCache "
            "architecture in complete detail; and a complete performance model "
            "for two L4 GPUs connected via Redis over a WireGuard VPN with 50ms "
            "RTT, deriving expected TTFT as a function of prompt length."
        ),
    },
]

POLL_INTERVAL_S = 0.3   # seconds between status/metrics polls
MAX_POLL_S      = 300.0  # give up after this many seconds


# ==============================================================================
# Single-request measurement helpers
# ==============================================================================

def _error(msg: str) -> dict:
    return {
        "status": f"failed: {msg}",
        "ttft_ms": None, "tpot_ms": None, "e2e_ms": None,
        "tokens": None, "forward_ms": None, "kv_overhead_ms": None,
    }


async def measure_collocated(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """
    Submit to collocated baseline; poll /metrics/{id} until complete.
    TTFT and TPOT are measured server-side and returned directly.
    """
    session_id = str(uuid.uuid4())
    payload = {"prompt": prompt, "session_id": session_id, "max_new_tokens": max_tokens}

    t_start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/generate", json=payload, timeout=timeout)
        if resp.status_code != 200:
            return _error(f"generate HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.RequestError as e:
        return _error(f"generate connection: {e}")

    deadline = time.perf_counter() + MAX_POLL_S
    while time.perf_counter() < deadline:
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            mresp = await client.get(f"{url}/metrics/{session_id}", timeout=10.0)
            if mresp.status_code == 200:
                data = mresp.json()
                status = data.get("status", "")
                if status == "complete":
                    e2e_ms = (time.perf_counter() - t_start) * 1000
                    return {
                        "status":         "success",
                        "ttft_ms":        data.get("ttft_ms"),
                        "tpot_ms":        data.get("tpot_ms"),
                        "e2e_ms":         round(e2e_ms, 2),
                        "tokens":         data.get("tokens"),
                        "forward_ms":     None,
                        "kv_overhead_ms": None,
                    }
                if status == "error":
                    return _error(f"server: {data.get('error', 'unknown')}")
        except Exception:
            pass

    return _error("timeout waiting for collocated metrics")


async def measure_disaggregated(
    client: httpx.AsyncClient,
    gateway_url: str,
    decode_url: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """
    Submit to gateway; poll gateway /status/{id} until prefill_accepted (= TTFT),
    then poll decode /metrics/{id} until complete (= TPOT + forward_ms).
    kv_overhead_ms = ttft_ms − forward_ms captures network + queue cost.
    """
    session_id = str(uuid.uuid4())
    payload = {"prompt": prompt, "session_id": session_id, "max_new_tokens": max_tokens}

    t_start = time.perf_counter()
    try:
        resp = await client.post(f"{gateway_url}/generate", json=payload, timeout=timeout)
        if resp.status_code != 200:
            return _error(f"generate HTTP {resp.status_code}: {resp.text[:200]}")
    except httpx.RequestError as e:
        return _error(f"generate connection: {e}")

    # Phase 1: wait for prefill_accepted — this is the client-measured TTFT
    ttft_ms: float | None = None
    deadline = time.perf_counter() + MAX_POLL_S
    while time.perf_counter() < deadline:
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            sresp = await client.get(f"{gateway_url}/status/{session_id}", timeout=10.0)
            if sresp.status_code == 200:
                data = sresp.json()
                status = data.get("status", "")
                if status == "prefill_accepted":
                    ttft_ms = (time.perf_counter() - t_start) * 1000
                    break
                if status in ("prefill_error", "connection_error", "error"):
                    return _error(f"prefill failed with status='{status}'")
        except Exception:
            pass
    else:
        return _error("timeout waiting for prefill_accepted")

    # Phase 2: wait for decode metrics
    deadline = time.perf_counter() + MAX_POLL_S
    while time.perf_counter() < deadline:
        await asyncio.sleep(POLL_INTERVAL_S)
        try:
            mresp = await client.get(f"{decode_url}/metrics/{session_id}", timeout=10.0)
            if mresp.status_code == 200:
                data = mresp.json()
                status = data.get("status", "")
                if status == "complete":
                    e2e_ms = (time.perf_counter() - t_start) * 1000
                    forward_ms = data.get("forward_time_ms")
                    kv_overhead = (
                        round(ttft_ms - forward_ms, 2)
                        if (ttft_ms is not None and forward_ms is not None)
                        else None
                    )
                    return {
                        "status":         "success",
                        "ttft_ms":        round(ttft_ms, 2),
                        "tpot_ms":        data.get("tpot_ms"),
                        "e2e_ms":         round(e2e_ms, 2),
                        "tokens":         data.get("tokens"),
                        "forward_ms":     forward_ms,
                        "kv_overhead_ms": kv_overhead,
                    }
                if status == "error":
                    return _error(f"decode: {data.get('error', 'unknown')}")
        except Exception:
            pass

    return _error("timeout waiting for decode metrics")


# ==============================================================================
# Run a full condition (all prompts × N runs)
# ==============================================================================

async def run_condition(
    mode: str,
    urls: dict,
    prompts: list[dict],
    runs: int,
    max_tokens: int,
    warmup: int,
    timeout: float,
) -> list[dict]:
    results = []

    async with httpx.AsyncClient() as client:

        async def _measure(prompt_text: str) -> dict:
            if mode == "collocated":
                return await measure_collocated(
                    client, urls["collocated"], prompt_text, max_tokens, timeout
                )
            return await measure_disaggregated(
                client, urls["gateway"], urls["decode"], prompt_text, max_tokens, timeout
            )

        # Warmup on first prompt only
        for _ in range(warmup):
            p = prompts[0]
            print(f"  [warmup] {p['label']}...", end=" ", flush=True)
            t0 = time.perf_counter()
            await _measure(p["text"])
            print(f"{int((time.perf_counter() - t0) * 1000)}ms")

        print()

        for idx, p in enumerate(prompts):
            label = p["label"]
            print(f"[{idx + 1}/{len(prompts)}] {label}")
            run_results = []

            for run_num in range(1, runs + 1):
                r = await _measure(p["text"])
                run_results.append(r)

                ttft_d  = f"{r['ttft_ms']:.0f}ms"  if r["ttft_ms"]  is not None else "N/A"
                tpot_d  = f"{r['tpot_ms']:.1f}ms"  if r["tpot_ms"]  is not None else "N/A"
                e2e_d   = f"{r['e2e_ms']:.0f}ms"   if r["e2e_ms"]   is not None else "N/A"
                extra   = (
                    f"  fwd={r['forward_ms']:.0f}ms  xfer={r['kv_overhead_ms']:.0f}ms"
                    if r.get("forward_ms") is not None
                    else ""
                )
                print(
                    f"  Run {run_num}: TTFT={ttft_d}  TPOT={tpot_d}  E2E={e2e_d}"
                    f"{extra}  [{r['status']}]"
                )

                results.append({
                    "condition":      mode,
                    "prompt_label":   label,
                    "run":            run_num,
                    "status":         r["status"],
                    "ttft_ms":        r["ttft_ms"],
                    "tpot_ms":        r["tpot_ms"],
                    "e2e_ms":         r["e2e_ms"],
                    "tokens":         r["tokens"],
                    "forward_ms":     r["forward_ms"],
                    "kv_overhead_ms": r["kv_overhead_ms"],
                })

            good = [x for x in run_results if x["status"] == "success"]
            if good:
                def _m(key):
                    vals = [x[key] for x in good if x[key] is not None]
                    return statistics.mean(vals) if vals else None

                m_ttft = _m("ttft_ms")
                m_tpot = _m("tpot_ms")
                m_e2e  = _m("e2e_ms")
                m_fwd  = _m("forward_ms")
                m_xfer = _m("kv_overhead_ms")

                parts = [
                    f"TTFT={m_ttft:.0f}ms" if m_ttft is not None else "TTFT=N/A",
                    f"TPOT={m_tpot:.1f}ms" if m_tpot is not None else "TPOT=N/A",
                    f"E2E={m_e2e:.0f}ms"   if m_e2e  is not None else "E2E=N/A",
                ]
                if m_fwd is not None:
                    parts.append(f"fwd={m_fwd:.0f}ms  xfer={m_xfer:.0f}ms")
                print(f"  Mean:  {'  '.join(parts)}")
            else:
                print(f"  All {runs} runs failed.")
            print()

    return results


# ==============================================================================
# Output helpers
# ==============================================================================

def _stats(values: list) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(clean), 2),
        "std":  round(statistics.stdev(clean), 2) if len(clean) > 1 else 0.0,
        "min":  round(min(clean), 2),
        "max":  round(max(clean), 2),
    }


def write_results(
    results: list[dict],
    condition: str,
    output_dir: str,
) -> tuple[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = str(out / f"{condition}_{ts}.csv")
    json_path = str(out / f"{condition}_{ts}_summary.json")

    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    by_label: dict[str, list] = {}
    for r in results:
        by_label.setdefault(r["prompt_label"], []).append(r)

    per_prompt = {}
    for label, rows in by_label.items():
        good = [r for r in rows if r["status"] == "success"]
        per_prompt[label] = {
            "n_runs":         len(rows),
            "success_rate":   round(len(good) / len(rows), 3),
            "ttft_ms":        _stats([r["ttft_ms"]        for r in good]),
            "tpot_ms":        _stats([r["tpot_ms"]        for r in good]),
            "e2e_ms":         _stats([r["e2e_ms"]         for r in good]),
            "tokens":         _stats([r["tokens"]         for r in good]),
            "forward_ms":     _stats([r["forward_ms"]     for r in good]),
            "kv_overhead_ms": _stats([r["kv_overhead_ms"] for r in good]),
        }

    summary = {"condition": condition, "timestamp": ts, "results": per_prompt}
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def _fmt_mean(section: dict | None) -> str:
    if not section or section.get("mean") is None:
        return "N/A"
    return f"{section['mean']:.1f}"


def print_summary_table(json_path: str):
    with open(json_path) as f:
        s = json.load(f)

    cond = s["condition"].upper()
    cols = ["prompt_label", "ttft_ms", "tpot_ms", "e2e_ms", "tokens", "forward_ms", "kv_overhead_ms", "ok%"]
    widths = [16, 10, 10, 10, 8, 12, 15, 6]

    def _row(vals: list) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print(f"\n{'='*95}")
    print(f"  {cond} SUMMARY")
    print(f"{'='*95}")
    print(_row(cols))
    print("-" * 95)
    for label in sorted(s["results"]):
        r = s["results"][label]
        print(_row([
            label,
            _fmt_mean(r.get("ttft_ms")),
            _fmt_mean(r.get("tpot_ms")),
            _fmt_mean(r.get("e2e_ms")),
            _fmt_mean(r.get("tokens")),
            _fmt_mean(r.get("forward_ms")),
            _fmt_mean(r.get("kv_overhead_ms")),
            f"{r['success_rate']:.0%}",
        ]))


def print_comparison(coloc_json: str, disag_json: str):
    with open(coloc_json) as f:
        c = json.load(f)
    with open(disag_json) as f:
        d = json.load(f)

    labels = sorted(set(c["results"]) | set(d["results"]))

    print(f"\n{'='*115}")
    print("  COLLOCATED vs DISAGGREGATED — HEAD-TO-HEAD COMPARISON")
    print(f"{'='*115}")
    hdr = (
        f"{'prompt':<16}  "
        f"{'TTFT_coloc':>12}  {'TTFT_disag':>12}  "
        f"{'TPOT_coloc':>12}  {'TPOT_disag':>12}  "
        f"{'fwd_ms':>8}  {'kv_xfer_ms':>12}"
    )
    print(hdr)
    print("-" * 115)

    for label in labels:
        cr = c["results"].get(label, {})
        dr = d["results"].get(label, {})
        ttft_c = _fmt_mean(cr.get("ttft_ms"))
        ttft_d = _fmt_mean(dr.get("ttft_ms"))
        tpot_c = _fmt_mean(cr.get("tpot_ms"))
        tpot_d = _fmt_mean(dr.get("tpot_ms"))
        fwd    = _fmt_mean(dr.get("forward_ms"))
        xfer   = _fmt_mean(dr.get("kv_overhead_ms"))
        print(
            f"{label:<16}  "
            f"{ttft_c:>12}  {ttft_d:>12}  "
            f"{tpot_c:>12}  {tpot_d:>12}  "
            f"{fwd:>8}  {xfer:>12}"
        )

    print(f"{'='*115}")
    print("  All values in milliseconds (mean across runs).")
    print("  kv_xfer_ms = TTFT_disag − fwd_ms  (network transfer + queue overhead)")
    print(f"{'='*115}")


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="AMLIC ZMQ benchmark — collocated vs disaggregated latency comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--mode", choices=["collocated", "disaggregated", "both"], default="both",
        help="Which architecture(s) to benchmark",
    )
    ap.add_argument("--collocated-url", default="http://localhost:8000",
                    help="Collocated baseline URL (host:port)")
    ap.add_argument("--gateway-url",    default="http://localhost:8000",
                    help="Disaggregated gateway URL (prefill VM host:8000)")
    ap.add_argument("--decode-url",     default="http://localhost:8002",
                    help="Decode worker metrics URL (decode VM host:8002)")
    ap.add_argument("--runs",       type=int,   default=3,     help="Timed runs per prompt")
    ap.add_argument("--max-tokens", type=int,   default=128,   help="max_new_tokens per request")
    ap.add_argument("--warmup",     type=int,   default=1,     help="Warmup runs (not recorded)")
    ap.add_argument("--timeout",    type=float, default=300.0, help="Per-request HTTP timeout (s)")
    ap.add_argument("--output",     default="benchmark/results/", help="Output directory for CSV/JSON")
    ap.add_argument(
        "--prompts", default="all",
        help="Comma-separated prompt labels to run, e.g. P1_tiny,P3_medium,P5_xlarge, or 'all'",
    )
    args = ap.parse_args()

    prompt_map = {p["label"]: p for p in PROMPTS}
    if args.prompts == "all":
        selected = PROMPTS
    else:
        selected = [prompt_map[lbl] for lbl in args.prompts.split(",") if lbl in prompt_map]
        unknown = [lbl for lbl in args.prompts.split(",") if lbl not in prompt_map]
        if unknown:
            print(f"WARNING: unknown prompt labels skipped: {unknown}")
        if not selected:
            print("ERROR: no valid prompts selected.")
            return

    urls = {
        "collocated": args.collocated_url,
        "gateway":    args.gateway_url,
        "decode":     args.decode_url,
    }
    modes = ["collocated", "disaggregated"] if args.mode == "both" else [args.mode]

    json_paths: dict[str, str] = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode.upper()}")
        if mode == "collocated":
            print(f"  URL:     {urls['collocated']}")
        else:
            print(f"  Gateway: {urls['gateway']}")
            print(f"  Decode:  {urls['decode']}")
        print(f"  Prompts: {len(selected)}  Runs: {args.runs}  MaxTokens: {args.max_tokens}  Warmup: {args.warmup}")
        print(f"{'='*60}\n")

        results = asyncio.run(run_condition(
            mode=mode,
            urls=urls,
            prompts=selected,
            runs=args.runs,
            max_tokens=args.max_tokens,
            warmup=args.warmup,
            timeout=args.timeout,
        ))

        csv_path, json_path = write_results(results, mode, args.output)
        json_paths[mode] = json_path
        print_summary_table(json_path)
        print(f"\n  Saved: {csv_path}")
        print(f"         {json_path}")

    if args.mode == "both" and "collocated" in json_paths and "disaggregated" in json_paths:
        print_comparison(json_paths["collocated"], json_paths["disaggregated"])


if __name__ == "__main__":
    main()
