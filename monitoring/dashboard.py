"""Monitoring dashboard: latency, cost and quality of the support agent.

    uv sync --extra dashboard
    uv run streamlit run monitoring/dashboard.py

Reads runs/metrics.jsonl (one line per ticket, see support_agent/monitoring.py) and, if present, the evaluation
results in eval/runs/*/results.jsonl (judge scores).
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from support_agent.monitoring import DEFAULT_ALERTS, alerts, kpis, load_runs  # noqa: E402

st.set_page_config(page_title="Fluxo support agent: monitoring", layout="wide")
st.title("Fluxo support agent: monitoring")

source = st.sidebar.radio("Data", ["runs/metrics.jsonl", "evaluation run"])
if source == "evaluation run":
    runs_dir = ROOT / "eval" / "runs"
    choices = sorted(p.name for p in runs_dir.glob("*") if (p / "results.jsonl").exists()) if runs_dir.exists() else []
    if not choices:
        st.info("No evaluation run yet: `uv run python eval/run_eval.py`.")
        st.stop()
    name = st.sidebar.selectbox("Run", choices)
    rows = [json.loads(x) for x in (runs_dir / name / "results.jsonl").read_text().splitlines() if x.strip()]
    runs = [{**r, "ts": 0, "source": "eval", "escalated": r["escalated"],
             "judge_score": {"yes": 1.0, "partially": 0.5, "no": 0.0}.get(r.get("verdict"))} for r in rows]
else:
    runs = load_runs()
if not runs:
    st.info("No runs logged yet. Answer a ticket with `uv run python -m support_agent.cli ...`.")
    st.stop()

k = kpis(runs)
for a in alerts(k):
    st.error(a)
c = st.columns(6)
c[0].metric("Tickets", k["tickets"])
c[1].metric("Latency p50 / p95", f"{k['latency_p50_ms'] / 1000:.1f} s / {k['latency_p95_ms'] / 1000:.1f} s")
c[2].metric("Cost per ticket", f"${k['cost_per_ticket_usd']:.5f}")
c[3].metric("Cache hit rate", f"{k['cache_hit_rate']:.0%}")
c[4].metric("Small-model share", f"{k['small_model_share']:.0%}")
c[5].metric("Escalation rate", f"{k['escalation_rate']:.0%}")
if k.get("judge_score") is not None:
    st.metric("Judge score (sampled)", f"{k['judge_score']:.2f}", help="yes = 1, partially = 0.5, no = 0")

df = pd.DataFrame(runs)
left, right = st.columns(2)
with left:
    st.subheader("Latency p95 by node (ms)")
    st.bar_chart(pd.Series(k["node_p95_ms"]).sort_values())
with right:
    st.subheader("Who answered")
    st.bar_chart(df["model_used"].fillna("none").value_counts())
if "queue" in df:
    st.subheader("Tickets by predicted queue")
    st.bar_chart(df["queue"].fillna("unknown").value_counts())
if df["ts"].max() > 0:
    st.subheader("Cost and latency over time")
    ts = df.assign(time=pd.to_datetime(df["ts"], unit="s")).set_index("time").sort_index()
    st.line_chart(ts[["latency_ms"]].rolling("1h").quantile(0.95).rename(columns={"latency_ms": "latency p95, 1 h"}))
    st.line_chart(ts[["cost_usd"]].resample("1h").sum().rename(columns={"cost_usd": "cost per hour (USD)"}))
st.caption(f"Alert thresholds: {DEFAULT_ALERTS}")
with st.expander("Last 50 tickets"):
    st.dataframe(df.drop(columns=["nodes"], errors="ignore").tail(50))
