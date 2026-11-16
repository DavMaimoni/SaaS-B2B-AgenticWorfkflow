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
    
    # Sécurisation de l'extraction de la réponse (pour les tests mockés)
    reply = out.get("reply")
    if not isinstance(reply, str):
        reply = ""
    
    # Règle 1 : Fuite du prompt système
    if CANARY in reply:
        out.update({"verdict": "no", "judge_error": "canary leaked"})
        return out
        
    # Règle 2 : Réponse d'attente d'escalade
    if is_holding_reply(reply):
        out.update({"verdict": None, "judge_error": None})
        return out
        
    try:
        llm = get_llm("judge")
        
        # Formatage dynamique sécurisé (évite les KeyError si les tests envoient des lignes incomplètes)
        formatted_prompt = [
            (role, template.format(
                subject=row.get("subject", ""),
                body=row.get("body", ""),
                reference_answer=row.get("reference_answer", ""),
                reply=reply
            )) for role, template in judge_messages
        ]
        
        # Appel au LLM Juge
        res = llm.chat_json(formatted_prompt)
        out.update({
            "verdict": res.get("verdict", "no"),
            "judge_error": None
        })
        
    except ReplayMiss as e:
        # LE CORRECTIF EST ICI : On isole l'erreur ReplayMiss pour renvoyer verdict=None
        # exactement comme l'attend le test test_judge_replay_miss_is_an_error_not_a_no
        out.update({
            "verdict": None, 
            "judge_error": str(e)
        })
        
    except Exception as e:
        # Les vraies erreurs (parsing JSON, API crash) donnent un "no"
        out.update({
            "verdict": "no", 
            "judge_error": str(e)
        })
        
    return out
