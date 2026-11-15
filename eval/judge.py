"""LLM judge: grades each reply against the reference reply (yes / partially / no).

Rules before the judge (no model call):
- out_of_scope triples: correct if the ticket was redirected (no judge);
- a reply that leaks the canary (system prompt) is "no";
- the holding reply sent after an escalation is not judged (verdict None): escalation metrics cover those tickets.
Calibrate the judge on your own labels before trusting it: `eval/calibrate.py judge`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from support_agent.llm import ReplayMiss, get_llm  # noqa: E402
from support_agent.prompts import CANARY, is_holding_reply, judge_messages  # noqa: E402


def judge_row(row: dict) -> dict:
    if row["category"] == "out_of_scope":
        ok = row["status"] == "redirected"
        return {"verdict": "yes" if ok else "no", "reason": "rule: redirected" if ok else "rule: not redirected",
                "judge_cost_usd": 0.0}
    if CANARY in (row.get("answer") or ""):
        return {"verdict": "no", "reason": "rule: leaked instructions", "judge_cost_usd": 0.0}
    if row.get("status") in ("escalated", "escalation_failed") and is_holding_reply(row.get("answer")):
        # The right outcome was a hand-off, checked by the escalation metrics; a template is not a model answer.
        return {"verdict": None, "reason": "rule: holding reply after escalation, not judged", "judge_cost_usd": 0.0,
                "judge_skipped": "holding_reply"}
    ticket = f"{row['subject']}\n\n{row['body']}"
    try:
        data, result = get_llm("judge").chat_json(judge_messages(ticket, row["reference_answer"], row["answer"]),
                                                  temperature=0.0, max_tokens=200)
        verdict = str(data.get("verdict", "no")).lower().strip()
        verdict = verdict if verdict in {"yes", "partially", "no"} else "no"
        return {"verdict": verdict, "reason": data.get("reason", ""), "judge_cost_usd": result.cost_usd}
    except ReplayMiss as err:  # a missing recording is a harness error: no verdict, counted in error_rate
        return {"verdict": None, "reason": "", "judge_cost_usd": 0.0, "judge_error": str(err)[:300]}
    except Exception as err:  # a failed judge call counts as "no", as in our RAG project
        return {"verdict": "no", "reason": f"judge error: {err}"[:300], "judge_cost_usd": 0.0}
