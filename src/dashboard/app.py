"""
AI Evaluation Pipeline - Complete Web Dashboard
A production-ready web interface for benchmarking LLMs.
"""
import json
import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st
from plotly import express as px
from plotly import graph_objects as go

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:12000")

st.set_page_config(
    page_title="AI Evaluation Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

def check_api() -> bool:
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except:
        return False

def get_info() -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/info", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def get_datasets() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/datasets", timeout=5)
        return r.json() if r.status_code == 200 else []
    except:
        return []

def get_evals() -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/evaluations", timeout=5)
        return r.json().get("runs", []) if r.status_code == 200 else []
    except:
        return []

def get_status(run_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/evaluations/{run_id}/status", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def get_summary(run_id: str) -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}/evaluations/{run_id}/summary", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except:
        return {}

def get_results(run_id: str) -> list:
    try:
        r = requests.get(f"{API_BASE_URL}/evaluations/{run_id}/results", params={"limit": 100}, timeout=10)
        return r.json().get("results", []) if r.status_code == 200 else []
    except:
        return []

def upload_ds(file, name: str) -> dict:
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        r = requests.post(f"{API_BASE_URL}/datasets/upload", files=files, data={"name": name}, timeout=30)
        return {"ok": r.status_code == 200, "data": r.json() if r.status_code == 200 else {}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def start_eval(dataset: str, models: list, metrics: bool = True, concurrent: int = 5) -> dict:
    try:
        payload = {"dataset_name": dataset, "models": models, "calculate_metrics": metrics, "max_concurrent": concurrent}
        r = requests.post(f"{API_BASE_URL}/evaluations", json=payload, timeout=10)
        return {"ok": r.status_code == 200, "data": r.json() if r.status_code == 200 else {}}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    st.markdown('<h1 class="main-header">🤖 AI Evaluation Pipeline</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Benchmark LLMs with automated quality metrics | Reduce validation time by 80%</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        global API_BASE_URL
        API_BASE_URL = st.text_input("API Base URL", value=API_BASE_URL)
        
        if st.button("🔄 Test Connection"):
            st.success("✅ Connected!") if check_api() else st.error("❌ Not connected")
        
        st.markdown("---")
        st.markdown("### 🔑 API Keys")
        st.text_input("OpenAI Key", type="password", key="openai")
        st.text_input("Anthropic Key", type="password", key="anthropic")
        st.text_input("Groq Key", type="password", key="groq")
        
        st.markdown("---")
        st.markdown("**AI Evaluation Pipeline v1.0**\n\nBenchmark LLMs with automated quality metrics.")
    
    if not check_api():
        st.error("⚠️ Cannot connect to API. Ensure it's running on port 12000.")
        st.code("python -m uvicorn src.api.routes:app --host 0.0.0.0 --port 12000")
        return
    
    tabs = st.tabs(["⚙️ Config", "📁 Datasets", "🚀 Evaluate", "📊 Results"])
    
    with tabs[0]:
        st.markdown("### 🤖 Select Models")
        info = get_info()
        models = info.get("available_models", [])
        selected = []
        
        if models:
            by_provider = {}
            for m in models:
                by_provider.setdefault(m.get("provider", "unk"), []).append(m)
            
            cols = st.columns(len(by_provider))
            for i, (p, ms) in enumerate(by_provider.items()):
                with cols[i]:
                    st.markdown(f"**{p.upper()}**")
                    for m in ms:
                        if st.checkbox(m.get("display_name", m["name"]), value=True, key=f"m_{m['name']}"):
                            selected.append(m["name"])
            
            st.session_state.selected_models = selected
            st.success(f"Selected: {len(selected)} models")
        else:
            st.warning("No models available")
    
    with tabs[1]:
        st.markdown("### 📁 Datasets")
        datasets = get_datasets()
        
        if datasets:
            df = pd.DataFrame([{"Name": d["name"], "Items": d["item_count"], "Ground Truth": "✅" if d.get("has_ground_truth") else "❌"} for d in datasets])
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 📤 Upload")
        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Dataset file", type=["csv", "json", "jsonl"])
        with col2:
            name = st.text_input("Name", placeholder="my_dataset")
        
        if file:
            try:
                if file.name.endswith(".csv"):
                    df = pd.read_csv(file)
                elif file.name.endswith(".jsonl"):
                    df = pd.read_json(file.getvalue().decode(), lines=True)
                else:
                    df = pd.read_json(file.getvalue().decode())
                st.dataframe(df.head(3), use_container_width=True, hide_index=True)
                st.markdown(f"**{len(df)} rows** | Columns: {list(df.columns)}")
                
                if "id" not in df.columns or "prompt" not in df.columns:
                    st.error("Missing required columns: id, prompt")
                else:
                    st.success("✅ Valid format")
            except Exception as e:
                st.error(f"Error: {e}")
        
        if file and st.button("Upload", type="primary"):
            result = upload_ds(file, name or file.name.split(".")[0])
            if result.get("ok"):
                st.success("Uploaded!")
                st.rerun()
            else:
                st.error(result.get("error", "Failed"))
    
    with tabs[2]:
        st.markdown("### 🚀 Run Evaluation")
        datasets = get_datasets()
        selected_models = st.session_state.get("selected_models", [])
        
        if datasets:
            ds = st.selectbox("Dataset", [d["name"] for d in datasets])
            st.markdown(f"Items: {next((d['item_count'] for d in datasets if d['name'] == ds), 0)}")
        else:
            st.warning("No datasets")
            ds = None
        
        metrics = st.checkbox("Calculate Quality Metrics", value=True)
        concurrent = st.slider("Concurrent Requests", 1, 20, 5)
        
        st.markdown("---")
        
        if st.button("▶️ Start Evaluation", type="primary", disabled=not ds or not selected_models):
            result = start_eval(ds, selected_models, metrics, concurrent)
            if result.get("ok"):
                run_id = result["data"].get("run_id")
                st.session_state.run_id = run_id
                st.success(f"Started! `{run_id}`")
            else:
                st.error(result.get("error", "Failed"))
        
        if st.session_state.get("run_id"):
            status = get_status(st.session_state.run_id)
            if status:
                cols = st.columns(4)
                cols[0].metric("Status", status.get("status", "?"))
                cols[1].metric("Progress", f"{status.get('progress', 0):.1f}%")
                cols[2].metric("Success", status.get("successful", 0))
                cols[3].metric("Failed", status.get("failed", 0))
                st.progress(status.get("progress", 0) / 100)
                
                if status.get("status") == "running":
                    time.sleep(2)
                    st.rerun()
                elif status.get("status") == "completed":
                    st.success("🎉 Done!")
    
    with tabs[3]:
        st.markdown("### 📊 Results")
        evals = [e for e in get_evals() if e.get("status") == "completed"]
        
        if not evals:
            st.info("No completed evaluations")
            return
        
        run = st.selectbox("Evaluation", [e["id"] for e in evals], format_func=lambda x: next((e["name"] for e in evals if e["id"] == x), x), key="result_run")
        
        if run:
            summary = get_summary(run)
            results = get_results(run)
            
            if summary and summary.get("models"):
                data = []
                for name, ms in summary["models"].items():
                    m = ms["metrics"]
                    data.append({"Model": name, "Eval": ms.get("successful", 0), "Accuracy": m.get("accuracy", {}).get("mean", 0), "Faithful": m.get("faithfulness", {}).get("mean", 0), "Latency": m.get("latency_ms", {}).get("mean", 0), "Cost": m.get("cost_usd", {}).get("mean", 0)})
                
                st.dataframe(pd.DataFrame(data).style.format({"Accuracy": "{:.3f}", "Faithful": "{:.3f}", "Latency": "{:.1f}", "Cost": "${:.6f}"}), use_container_width=True, hide_index=True)
                
                viz = st.tabs(["📈 Charts", "🏆 Winner"])
                
                with viz[0]:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(x=list(summary["models"].keys()), y=[m["metrics"]["accuracy"]["mean"] for m in summary["models"].values()], title="Accuracy", labels={"x": "Model", "y": "Accuracy"}, color=list(summary["models"].keys()))
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.bar(x=list(summary["models"].keys()), y=[m["metrics"]["latency_ms"]["mean"] for m in summary["models"].values()], title="Latency (ms)", labels={"x": "Model", "y": "ms"}, color=list(summary["models"].keys()))
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                with viz[1]:
                    winner = max(summary["models"].items(), key=lambda x: x[1]["metrics"]["composite_score"]["mean"])
                    w = winner[1]["metrics"]
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🥇 Best: {winner[0]}</h3>
                        <p>Score: {w['composite_score']['mean']:.3f}</p>
                        <ul>
                            <li>Accuracy: {w['accuracy']['mean']:.3f}</li>
                            <li>Faithfulness: {w['faithfulness']['mean']:.3f}</li>
                            <li>Latency: {w['latency_ms']['mean']:.1f}ms</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                if results:
                    df = pd.DataFrame(results)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), f"results_{run[:8]}.csv", "text/csv")

if __name__ == "__main__":
    main()
