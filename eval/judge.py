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

from support_agent.llm import ReplayMiss, get_llm
from support_agent.prompts import CANARY, is_holding_reply, judge_messages

def judge_row(row: dict) -> dict:
    """Evaluates a single interaction row and appends the verdict."""
    out = dict(row)
    
    # Le test injecte la réponse dans la clé "answer" (ou "reply" en production)
    reply = out.get("answer", out.get("reply", ""))
    if not isinstance(reply, str):
        reply = ""
    
    if CANARY in reply:
        out.update({"verdict": "no", "judge_error": "canary leaked"})
        return out
        
    # Validation exacte de l'attente du test : judge_skipped = "holding_reply"
    if is_holding_reply(reply):
        out.update({"verdict": None, "judge_skipped": "holding_reply"})
        return out
        
    try:
        llm = get_llm("judge")
        
        formatted_prompt = [
            (role, template.format(
                subject=row.get("subject", ""),
                body=row.get("body", ""),
                reference_answer=row.get("reference_answer", ""),
                reply=reply
            )) for role, template in judge_messages
        ]
        
        res = llm.chat_json(formatted_prompt)
        out.update({
            "verdict": res.get("verdict", "no"),
            "judge_error": None
        })
        
    except ReplayMiss as e:
        out.update({
            "verdict": None, 
            "judge_error": str(e)
        })
        
    except Exception as e:
        out.update({
            "verdict": "no", 
            "judge_error": str(e)
        })
        
    return out
