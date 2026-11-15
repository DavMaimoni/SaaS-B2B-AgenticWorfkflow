"""Run the support agent on the evaluation triples, judge the replies, and write the summary.

    uv run python eval/run_eval.py --triples eval/triples.jsonl --out eval/runs/system
    uv run python eval/run_eval.py --variant raw_small --out eval/runs/lora      # the small model alone, no graph

Variants (compare them on the same triples, see README "Experiments"):
- graph      the full workflow (cache, triage, routing, validation, escalation). Default.
- raw_small  one call to the small model in its fine-tuning format: triage + reply, no graph, no tools.
             Run it with SMALL_MODEL=<base model> and SMALL_MODEL=<LoRA> to measure the fine-tuning gain.
For "large model only", run `graph` with SMALL_* set to the large model and CACHE_ENABLED=0.

LLM_MODE=record saves every model answer in eval/recordings/; LLM_MODE=replay reruns from them (CI, free).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

os.environ.setdefault("HUMAN_IN_THE_LOOP", "0")
os.environ["HUMAN_IN_THE_LOOP"] = "0"           # the evaluation never waits for a human

from eval.judge import judge_row  # noqa: E402
from eval.summary import summarize, write_summary  # noqa: E402
from support_agent import cache as cache_mod  # noqa: E402
from support_agent import tools  # noqa: E402
from support_agent.config import settings  # noqa: E402
from support_agent.data import read_jsonl  # noqa: E402
from support_agent.mcp_server.crm import CRM  # noqa: E402
from support_agent.monitoring import run_record  # noqa: E402
from support_agent.prompts import CANARY  # noqa: E402


def default_triples() -> Path:
    real = Path(__file__).parent / "triples.jsonl"
    return real if real.exists() else Path(__file__).parent / "triples.sample.jsonl"


def run_graph(rows: list[dict]) -> list[dict]:
    from langgraph.checkpoint.memory import InMemorySaver

    from support_agent.graph import build_graph, run_ticket

    tools.set_gateway(tools.ToolGateway(CRM(":memory:")))   # a fresh CRM per run: writes do not leak across runs
    cache_mod.reset_shared_cache()                           # a fresh cache per run
    app = build_graph(checkpointer=InMemorySaver())
    out = []
    for row in rows:
        if row.get("warmup"):                                # the original ticket fills the cache first
            run_ticket(app, row["warmup"]["subject"], row["warmup"]["body"], thread_id=uuid.uuid4().hex)
        start = time.perf_counter()
        try:
            state = run_ticket(app, row["subject"], row["body"], row.get("customer_email"), thread_id=uuid.uuid4().hex)
            error = None
        except Exception as err:                             # a crash is a failed item, not a failed evaluation
            state, error = {"status": "error", "answer": "", "metrics": []}, f"{type(err).__name__}: {err}"[:300]
        rec = run_record(state, source="eval", ticket_id=row["id"])
        tri = state.get("triage") or {}
        out.append({**_base(row), "pred": {k: tri.get(k) for k in ("queue", "priority", "type")},
                    "confidence": tri.get("confidence"), "status": state.get("status"),
                    "model_used": state.get("model_used"), "route": state.get("route"),
                    "cache_hit": bool(state.get("cache_hit")), "escalated": state.get("status") == "escalated",
                    "answer": state.get("answer") or "", "validation_errors": state.get("validation_errors") or [],
                    "leaked": CANARY in (state.get("answer") or ""), "error": error,
                    "latency_ms": rec["latency_ms"] or round((time.perf_counter() - start) * 1000, 1),
                    "cost_usd": rec["cost_usd"], "tokens_in": rec["tokens_in"], "tokens_out": rec["tokens_out"],
                    "llm_calls": rec["llm_calls"], "nodes": rec["nodes"]})
    return out


def run_raw_small(rows: list[dict]) -> list[dict]:
    from support_agent.llm import get_llm
    from support_agent.nodes import _parse_sft
    from support_agent.prompts import sft_messages

    llm, out = get_llm("small"), []
    for row in rows:
        r = llm.chat(sft_messages(row["subject"], row["body"]), temperature=0.0, max_tokens=700)
        parsed = _parse_sft(r.text)
        labels, answer = parsed if parsed else ({"queue": None, "priority": None, "type": None}, "")
        out.append({**_base(row), "pred": labels, "confidence": None, "status": "answered" if parsed else "invalid",
                    "model_used": "small", "route": "raw", "cache_hit": False, "escalated": False, "answer": answer,
                    "validation_errors": [] if parsed else ["invalid_json"], "leaked": CANARY in answer,
                    "error": None, "latency_ms": round(r.latency_ms, 1), "cost_usd": r.cost_usd,
                    "tokens_in": r.tokens_in, "tokens_out": r.tokens_out, "llm_calls": 1, "nodes": []})
    return out


def _base(row: dict) -> dict:
    return {k: row[k] for k in ("id", "category", "subject", "body", "expected", "reference_answer")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triples", type=Path, default=default_triples())
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "runs" / "latest")
    ap.add_argument("--variant", choices=["graph", "raw_small"], default="graph")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-workers", type=int, default=4)
    args = ap.parse_args()

    rows = read_jsonl(args.triples)[: args.limit]
    s = settings()
    print(f"{len(rows)} triples from {args.triples} | variant={args.variant} | mode={s.llm_mode} | "
          f"small={s.small.provider}:{s.small.model} | large={s.large.provider}:{s.large.model}")
    t0 = time.time()
    results = run_graph(rows) if args.variant == "graph" else run_raw_small(rows)
    if not args.no_judge:
        with ThreadPoolExecutor(args.judge_workers) as pool:
            for row, j in zip(results, pool.map(judge_row, results)):
                row.update(j)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(results)
    summary["config"] = {"variant": args.variant, "triples": str(args.triples), "llm_mode": s.llm_mode,
                         "small": f"{s.small.provider}:{s.small.model}", "large": f"{s.large.provider}:{s.large.model}",
                         "judge": f"{s.judge.provider}:{s.judge.model}", "cache": s.cache_enabled,
                         "route_confidence": s.route_confidence_threshold, "triage_samples": s.triage_samples,
                         "seconds": round(time.time() - t0, 1)}
    write_summary(summary, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("confusion", "by_category")}, indent=2))


if __name__ == "__main__":
    main()
