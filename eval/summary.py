"""Aggregate a results.jsonl into the metrics of the report, check them against thresholds, compare two runs.

    uv run python eval/summary.py check eval/runs/latest --baseline eval/baseline.json --thresholds eval/thresholds.json
    uv run python eval/summary.py compare eval/runs/base eval/runs/lora
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.metrics import (VERDICT_SCORE, accuracy, bootstrap_ci, confusion, judge_score, macro_f1,  # noqa: E402
                          paired_bootstrap)

LABELS = ("queue", "priority", "type")


def _rate(rows: list[dict], pred) -> float | None:
    return round(sum(1 for r in rows if pred(r)) / len(rows), 4) if rows else None


def summarize(results: list[dict]) -> dict:
    s: dict = {"n": len(results), "error_rate": _rate(results, lambda r: bool(r.get("error")))}
    for label in LABELS:
        gold = [r["expected"][label] for r in results]
        pred = [(r.get("pred") or {}).get(label) for r in results]
        s[f"{label}_acc"] = round(accuracy(gold, pred), 4)
        s[f"{label}_macro_f1"] = round(macro_f1(gold, pred), 4)
    s["triage_exact"] = _rate(results, lambda r: all((r.get("pred") or {}).get(k) == r["expected"][k] for k in LABELS))
    s["confusion"] = {"queue": confusion([r["expected"]["queue"] for r in results],
                                         [(r.get("pred") or {}).get("queue") for r in results])}

    judged = [r for r in results if r.get("verdict")]
    if judged:
        scores = [VERDICT_SCORE[r["verdict"]] for r in judged]
        s["judge_score"] = round(judge_score([r["verdict"] for r in judged]), 4)
        s["judge_score_ci95"] = [round(x, 4) for x in bootstrap_ci(scores)]
        s["judge_cost_usd"] = round(sum(r.get("judge_cost_usd", 0) for r in judged), 6)
    by_cat = {}
    for cat in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == cat]
        jr = [r["verdict"] for r in rows if r.get("verdict")]
        by_cat[cat] = {"n": len(rows), "judge_score": round(judge_score(jr), 4) if jr else None,
                       "escalated": _rate(rows, lambda r: r["escalated"]),
                       "cache_hit": _rate(rows, lambda r: r["cache_hit"])}
    s["by_category"] = by_cat

    cat = lambda c: [r for r in results if r["category"] == c]  # noqa: E731
    s["escalation_recall_critical"] = _rate(cat("critical"), lambda r: r["escalated"])
    s["escalation_rate_routine"] = _rate(cat("routine"), lambda r: r["escalated"])
    s["redirect_accuracy"] = _rate(cat("out_of_scope"), lambda r: r["status"] == "redirected")
    s["leak_rate"] = _rate(results, lambda r: r.get("leaked"))
    s["injection_judge_score"] = by_cat.get("injection", {}).get("judge_score")
    s["cache_hit_rate_paraphrase"] = _rate(cat("paraphrase"), lambda r: r["cache_hit"])
    s["cache_hit_rate_other"] = _rate([r for r in results if r["category"] != "paraphrase"], lambda r: r["cache_hit"])

    by_model = [r for r in results if r.get("model_used") in ("small", "large")]
    s["small_model_share"] = _rate(by_model, lambda r: r["model_used"] == "small")
    lat = np.array([r["latency_ms"] for r in results]) if results else np.array([0.0])
    s["latency_p50_ms"] = round(float(np.percentile(lat, 50)), 1)
    s["latency_p95_ms"] = round(float(np.percentile(lat, 95)), 1)
    s["cost_per_ticket_usd"] = round(float(np.mean([r["cost_usd"] for r in results])), 8) if results else None
    s["llm_calls_per_ticket"] = round(float(np.mean([r["llm_calls"] for r in results])), 3) if results else None
    return s


FLAT_KEYS = ["n", "queue_acc", "priority_acc", "type_acc", "triage_exact", "queue_macro_f1", "judge_score",
             "escalation_recall_critical", "escalation_rate_routine", "redirect_accuracy", "leak_rate",
             "injection_judge_score", "cache_hit_rate_paraphrase", "cache_hit_rate_other", "small_model_share",
             "latency_p50_ms", "latency_p95_ms", "cost_per_ticket_usd", "llm_calls_per_ticket", "error_rate"]


def write_summary(summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in FLAT_KEYS:
            w.writerow([k, summary.get(k)])


def check(summary: dict, baseline: dict | None, thresholds: dict) -> list[str]:
    """Return the failed checks (empty list = pass)."""
    failures = []
    for metric, rule in thresholds.get("absolute", {}).items():
        value = summary.get(metric)
        if value is None:
            continue                           # metric not measured on this triple set (no such category)
        if "min" in rule and value < rule["min"]:
            failures.append(f"{metric} = {value} < {rule['min']}")
        if "max" in rule and value > rule["max"]:
            failures.append(f"{metric} = {value} > {rule['max']}")
    rel = thresholds.get("relative", {})
    if baseline and rel:
        tol = rel.get("tolerance", 0.02)
        for metric in rel.get("higher_is_better", []):
            if summary.get(metric) is not None and baseline.get(metric) is not None:
                if summary[metric] < baseline[metric] - tol:
                    failures.append(f"{metric} = {summary[metric]} < baseline {baseline[metric]} - {tol}")
        for metric, factor in rel.get("lower_is_better_factor", {}).items():
            if summary.get(metric) is not None and baseline.get(metric):
                if summary[metric] > baseline[metric] * factor:
                    failures.append(f"{metric} = {summary[metric]} > baseline {baseline[metric]} x {factor}")
    return failures


def _load(path: Path) -> list[dict]:
    path = path / "results.jsonl" if path.is_dir() else path
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def compare(a_path: Path, b_path: Path) -> dict:
    a = {r["id"]: r for r in _load(a_path)}
    b = {r["id"]: r for r in _load(b_path)}
    ids = sorted(set(a) & set(b))
    out = {"n_common": len(ids)}
    ja = [VERDICT_SCORE.get(a[i].get("verdict"), 0.0) for i in ids if a[i].get("verdict") and b[i].get("verdict")]
    jb = [VERDICT_SCORE.get(b[i].get("verdict"), 0.0) for i in ids if a[i].get("verdict") and b[i].get("verdict")]
    if ja:
        out["judge_score"] = paired_bootstrap(ja, jb)
    exact = lambda r: float(all((r.get("pred") or {}).get(k) == r["expected"][k] for k in LABELS))  # noqa: E731
    out["triage_exact"] = paired_bootstrap([exact(a[i]) for i in ids], [exact(b[i]) for i in ids]) if ids else None
    out["cost_per_ticket_usd"] = {"a": float(np.mean([a[i]["cost_usd"] for i in ids])),
                                  "b": float(np.mean([b[i]["cost_usd"] for i in ids]))} if ids else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("run", type=Path)
    c.add_argument("--baseline", type=Path)
    c.add_argument("--thresholds", type=Path, default=Path(__file__).parent / "thresholds.json")
    p = sub.add_parser("compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = ap.parse_args()

    if args.cmd == "check":
        summary = json.loads((args.run / "summary.json" if args.run.is_dir() else args.run).read_text())
        baseline = json.loads(args.baseline.read_text()) if args.baseline and args.baseline.exists() else None
        failures = check(summary, baseline, json.loads(args.thresholds.read_text()))
        if failures:
            print("FAILED:\n- " + "\n- ".join(failures))
            sys.exit(1)
        print("All evaluation checks passed.")
    else:
        print(json.dumps(compare(args.a, args.b), indent=2))


if __name__ == "__main__":
    main()
