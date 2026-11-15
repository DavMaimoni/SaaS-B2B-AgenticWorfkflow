"""Monitoring: one JSON line per answered ticket in runs/metrics.jsonl, and the KPIs computed from those lines.

The same functions serve the CLI report (monitoring/report.py), the Streamlit dashboard and the CI thresholds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from support_agent.config import RUNS_DIR

# Alert thresholds: set them from the reference measurement (see the project doc, "Cadrage produit").
DEFAULT_ALERTS = {
    "latency_p95_ms": 15000,      # alert if above
    "cost_per_ticket_usd": 0.002,  # alert if above
    "small_model_share": 0.40,     # alert if below
    "escalation_rate": 0.35,       # alert if above
    "judge_score": 0.70,           # alert if below (sampled traffic)
}


def run_record(state: dict, source: str, ticket_id: str | None = None, **extra) -> dict:
    rows = state.get("metrics", [])
    tri = state.get("triage", {}) or {}
    return {
        "ts": time.time(), "source": source, "ticket_id": ticket_id,
        "status": state.get("status"), "model_used": state.get("model_used"), "route": state.get("route"),
        "cache_hit": bool(state.get("cache_hit")), "queue": tri.get("queue"), "priority": tri.get("priority"),
        "type": tri.get("type"), "confidence": tri.get("confidence"),
        "escalated": state.get("status") == "escalated",
        "validation_errors": state.get("validation_errors", []),
        "latency_ms": round(sum(r["latency_ms"] for r in rows), 1),
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
        "tokens_in": sum(r["tokens_in"] for r in rows), "tokens_out": sum(r["tokens_out"] for r in rows),
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "nodes": [{k: r[k] for k in ("node", "model", "latency_ms", "cost_usd")} for r in rows],
        **extra,
    }


def log_run(state: dict, source: str, ticket_id: str | None = None, path: Path | None = None, **extra) -> dict:
    record = run_record(state, source, ticket_id, **extra)
    path = path or RUNS_DIR / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_runs(path: Path | None = None) -> list[dict]:
    path = path or RUNS_DIR / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def kpis(runs: list[dict]) -> dict:
    if not runs:
        return {"tickets": 0}
    lat = np.array([r["latency_ms"] for r in runs])
    cost = np.array([r["cost_usd"] for r in runs])
    answered_by_model = [r for r in runs if r["model_used"] in ("small", "large")]
    judged = [r["judge_score"] for r in runs if r.get("judge_score") is not None]
    per_node: dict[str, list[float]] = {}
    for r in runs:
        for n in r.get("nodes", []):
            per_node.setdefault(n["node"], []).append(n["latency_ms"])
    return {
        "tickets": len(runs),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "cost_per_ticket_usd": round(float(cost.mean()), 6),
        "cost_total_usd": round(float(cost.sum()), 6),
        "cache_hit_rate": round(sum(r["cache_hit"] for r in runs) / len(runs), 3),
        "small_model_share": round(sum(r["model_used"] == "small" for r in answered_by_model)
                                   / max(1, len(answered_by_model)), 3),
        "escalation_rate": round(sum(r["escalated"] for r in runs) / len(runs), 3),
        "judge_score": round(float(np.mean(judged)), 3) if judged else None,
        "node_p95_ms": {k: round(float(np.percentile(v, 95)), 1) for k, v in per_node.items()},
    }


def alerts(k: dict, limits: dict | None = None) -> list[str]:
    limits = {**DEFAULT_ALERTS, **(limits or {})}
    out = []
    if k.get("tickets", 0) == 0:
        return out
    if k["latency_p95_ms"] > limits["latency_p95_ms"]:
        out.append(f"latency p95 {k['latency_p95_ms']} ms > {limits['latency_p95_ms']} ms")
    if k["cost_per_ticket_usd"] > limits["cost_per_ticket_usd"]:
        out.append(f"cost per ticket {k['cost_per_ticket_usd']} USD > {limits['cost_per_ticket_usd']} USD")
    if k["small_model_share"] < limits["small_model_share"]:
        out.append(f"small-model share {k['small_model_share']} < {limits['small_model_share']} "
                   "(small model down or threshold too strict?)")
    if k["escalation_rate"] > limits["escalation_rate"]:
        out.append(f"escalation rate {k['escalation_rate']} > {limits['escalation_rate']}")
    if k.get("judge_score") is not None and k["judge_score"] < limits["judge_score"]:
        out.append(f"judge score {k['judge_score']} < {limits['judge_score']}")
    return out
