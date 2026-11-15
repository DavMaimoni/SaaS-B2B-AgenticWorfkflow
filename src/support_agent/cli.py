"""Answer one ticket from the command line.

    uv run python -m support_agent.cli "Charged twice" "Our card was charged twice this month" --email admin@quanta-labs.example.com

With HUMAN_IN_THE_LOOP=1 an escalated ticket pauses; approve it with --resume THREAD_ID [--answer "..."].
Every run appends its metrics to runs/metrics.jsonl (read by monitoring/report.py and the dashboard).
"""

import argparse
import json
import uuid

from langgraph.checkpoint.sqlite import SqliteSaver

from support_agent.config import RUNS_DIR
from support_agent.graph import build_graph, resume, run_ticket
from support_agent.monitoring import log_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?", default="")
    ap.add_argument("body", nargs="?", default="")
    ap.add_argument("--email")
    ap.add_argument("--thread")
    ap.add_argument("--resume", metavar="THREAD_ID")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--answer")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    import sqlite3
    saver = SqliteSaver(sqlite3.connect(RUNS_DIR / "checkpoints.sqlite", check_same_thread=False))
    app = build_graph(checkpointer=saver)
    if args.resume:
        out = resume(app, args.resume, approve=not args.reject, answer=args.answer)
        thread = args.resume
    else:
        thread = args.thread or uuid.uuid4().hex[:12]
        out = run_ticket(app, args.subject, args.body, args.email, thread_id=thread)
    if out.get("__interrupt__"):
        print(json.dumps({"thread_id": thread, "paused_for_human": out["__interrupt__"][0].value}, indent=2))
        return
    log_run(out, source="cli", ticket_id=thread)
    print(json.dumps({k: out.get(k) for k in ("status", "model_used", "cache_hit", "triage", "answer", "escalation")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
