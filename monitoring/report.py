"""Text monitoring report from runs/metrics.jsonl (written by the CLI and by eval/run_eval.py --log).

    uv run python monitoring/report.py                  # all runs
    uv run python monitoring/report.py --since-hours 24 --source cli
Exit code 2 when an alert fires, so a cron job or CI step can notify someone.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from support_agent.monitoring import alerts, kpis, load_runs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path)
    ap.add_argument("--since-hours", type=float)
    ap.add_argument("--source")
    args = ap.parse_args()
    runs = load_runs(args.file)
    if args.since_hours:
        runs = [r for r in runs if r["ts"] >= time.time() - args.since_hours * 3600]
    if args.source:
        runs = [r for r in runs if r["source"] == args.source]
    k = kpis(runs)
    print(json.dumps(k, indent=2))
    fired = alerts(k)
    for a in fired:
        print("ALERT:", a)
    sys.exit(2 if fired else 0)


if __name__ == "__main__":
    main()
