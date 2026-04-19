"""
app.py — Streamlit demo: "Adaptive LLM Router" showcase.

Demonstrates the router in action with live latency comparison.

Usage:
    streamlit run demo/app.py
"""

import json
import os
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive LLM Router — AMLIC Demo",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    router_url = st.text_input(
        "Router URL",
        value=os.environ.get("ROUTER_URL", "http://localhost:7000/v1"),
        help="URL of the adaptive router",
    )
    collocated_url = st.text_input(
        "Collocated URL (direct)",
        value=os.environ.get("COLLOCATED_URL", "http://localhost:8000/v1"),
    )
    disagg_url = st.text_input(
        "Disaggregated URL (direct)",
        value=os.environ.get("DISAGG_PROXY_URL", "http://localhost:9000/v1"),
    )
    threshold = st.number_input(
        "Routing Threshold (tokens)",
        value=int(os.environ.get("ROUTING_THRESHOLD", "500")),
        min_value=10,
        max_value=8000,
        step=50,
    )
    model_name = st.text_input(
        "Model",
        value=os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
    )
    max_tokens = st.slider("Max Output Tokens", 10, 1000, 200)
    compare_mode = st.checkbox("Compare Mode", value=True, help="Send to both endpoints and compare latency")

    st.divider()
    st.caption("Columbia COMS 6998 — Cloud Computing Project")

# ── Main area ───────────────────────────────────────────────────────────────
st.title("🧠 Adaptive LLM Inference Router")
st.markdown(
    "This demo routes requests to either a **collocated** or **disaggregated** "
    "serving backend based on the prompt length threshold **N**."
)

# Token count display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Threshold N", f"{threshold} tokens")
with col2:
    st.metric("Architecture", "Collocated" if threshold > 1000 else "Adaptive")
with col3:
    st.metric("Model", model_name.split("/")[-1])

st.divider()

# ── Chat interface ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            st.caption(msg["metadata"])

# User input
user_prompt = st.chat_input("Type your prompt here...")

if user_prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Estimate tokens
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        token_count = len(tok.encode(user_prompt))
    except Exception:
        token_count = len(user_prompt.split()) * 1.3  # rough estimate

    route = "disaggregated" if token_count >= threshold else "collocated"

    # Send request
    with st.chat_message("assistant"):
        import httpx

        if compare_mode:
            # Compare both endpoints
            results = {}
            for arch, url in [("collocated", collocated_url), ("disaggregated", disagg_url)]:
                try:
                    t0 = time.perf_counter()
                    with httpx.Client(timeout=120) as client:
                        resp = client.post(
                            f"{url}/chat/completions",
                            json={
                                "model": model_name,
                                "messages": [{"role": "user", "content": user_prompt}],
                                "max_tokens": max_tokens,
                                "temperature": 0.0,
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        latency = (time.perf_counter() - t0) * 1000
                        content = data["choices"][0]["message"]["content"]
                        results[arch] = {"content": content, "latency_ms": latency}
                except Exception as e:
                    results[arch] = {"content": f"Error: {e}", "latency_ms": 0}

            # Display comparison
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🏠 Collocated")
                if "collocated" in results:
                    st.markdown(results["collocated"]["content"])
                    st.caption(f"⏱️ {results['collocated']['latency_ms']:.0f} ms")
            with c2:
                st.subheader("🔀 Disaggregated")
                if "disaggregated" in results:
                    st.markdown(results["disaggregated"]["content"])
                    st.caption(f"⏱️ {results['disaggregated']['latency_ms']:.0f} ms")

            # Winner
            if all(r.get("latency_ms", 0) > 0 for r in results.values()):
                winner = min(results, key=lambda k: results[k]["latency_ms"])
                delta = abs(results["collocated"]["latency_ms"] - results["disaggregated"]["latency_ms"])
                st.success(f"🏆 **{winner.capitalize()}** wins by {delta:.0f} ms | Tokens: {int(token_count)} | Router would pick: **{route}**")

            chosen_content = results.get(route, {}).get("content", "")
            metadata = f"Tokens: {int(token_count)} | Route: {route} | Threshold: {threshold}"
        else:
            # Single endpoint via router
            target_url = router_url
            try:
                t0 = time.perf_counter()
                with httpx.Client(timeout=120) as client:
                    resp = client.post(
                        f"{target_url}/chat/completions",
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": user_prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.0,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    latency = (time.perf_counter() - t0) * 1000
                    chosen_content = data["choices"][0]["message"]["content"]
                    st.markdown(chosen_content)
                    metadata = f"⏱️ {latency:.0f} ms | Tokens: {int(token_count)} | Route: {route}"
                    st.caption(metadata)
            except Exception as e:
                chosen_content = f"Error: {e}"
                metadata = f"Error | Tokens: {int(token_count)} | Route: {route}"
                st.error(chosen_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": chosen_content,
        "metadata": metadata,
    })

# ── Results viewer ──────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 View Benchmark Results", expanded=False):
    results_dir = Path(__file__).parent.parent / "benchmark" / "results"
    import glob
    csv_files = sorted(glob.glob(str(results_dir / "*.csv")))
    if csv_files:
        import pandas as pd
        selected = st.selectbox("Select results file", csv_files, format_func=lambda x: Path(x).name)
        if selected:
            df = pd.read_csv(selected)
            st.dataframe(df, use_container_width=True)

            # Quick stats
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Mean TTFT", f"{df['ttft_ms'].mean():.1f} ms")
            with c2:
                st.metric("Mean Throughput", f"{df['throughput_tps'].mean():.1f} tok/s")
            with c3:
                st.metric("Requests", len(df))
    else:
        st.info("No benchmark results found. Run benchmarks first!")

with st.expander("📈 View Plots", expanded=False):
    figures_dir = Path(__file__).parent.parent / "analysis" / "figures"
    for img_path in sorted(figures_dir.glob("*.png")):
        st.image(str(img_path), caption=img_path.stem.replace("_", " ").title())
