"""The graph's nodes. Each one returns the keys it changes plus one `metrics` row (latency, model, tokens, cost).

Shared resources (the MCP gateway, the semantic cache, the model clients) are module-level singletons so the
graph stays serialisable for the checkpointer; tests swap them with `tools.set_gateway` and `cache.reset_shared_cache`.
"""

from __future__ import annotations

import re
import time
from collections import Counter

from pydantic import ValidationError

from support_agent import prompts
from support_agent.cache import CacheEntry, shared_cache
from support_agent.config import CACHEABLE_QUEUES, CACHEABLE_TYPES, OUT_OF_SCOPE_QUEUES, settings
from support_agent.llm import LLMResult, get_llm, parse_json
from support_agent.routing import choose_route
from support_agent.schemas import Draft, Triage
from support_agent.tools import ToolCallError, gateway

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)",
    r"disregard (your|the) (instructions|rules|guidelines)",
    r"(reveal|print|show|repeat) (your|the) (system )?(prompt|instructions|rules)",
    r"you are now", r"act as (a|an) ", r"developer mode", r"jailbreak",
    r"(system|assistant)\s*:", r"</?(system|instructions)>",
]
# Requests that commit money, data or security: never answered by the small model alone (see routing.py).
SENSITIVE_PATTERNS = [
    r"\brefund", r"chargeback", r"charged twice", r"double charge", r"cancel (my|our) (contract|subscription)",
    r"delete (all )?(my|our) (data|account|workspace)", r"data (breach|leak|loss)", r"security incident",
    r"hacked", r"unauthori[sz]ed (access|login)", r"legal action", r"lawyer", r"gdpr request",
]
PLACEHOLDER = re.compile(r"(<[a-z_ ]+>|\{\{[^}]+\}\}|\[(name|customer name|your name|company)\])", re.I)
HOLDING_REPLY = ("Thank you for contacting Fluxo support. Your request has been passed to a specialist on our team, "
                 "who will get back to you shortly. Your ticket reference is {ticket}.")
FALLBACK_TRIAGE = {"queue": "Customer Service", "priority": "medium", "type": "Problem"}


def _thread(config) -> str:
    return ((config or {}).get("configurable") or {}).get("thread_id", "default")


def _row(node: str, start: float, results: list[LLMResult] = (), model: str | None = None, **extra) -> dict:
    return {"node": node, "model": model, "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "tokens_in": sum(r.tokens_in for r in results), "tokens_out": sum(r.tokens_out for r in results),
            "cost_usd": round(sum(r.cost_usd for r in results), 8), "llm_calls": len(results), **extra}


def ticket_text(state: dict) -> str:
    return f"{state.get('subject', '')}\n\n{state.get('body', '')}".strip()


# ---- 1. semantic cache -----------------------------------------------------------------------------------------
def cache_lookup(state: dict) -> dict:
    start = time.perf_counter()
    if not settings().cache_enabled:
        return {"cache_hit": False, "metrics": [_row("cache_lookup", start, cache_hit=False)]}
    hit = shared_cache().lookup(ticket_text(state))
    if not hit:
        return {"cache_hit": False, "metrics": [_row("cache_lookup", start, cache_hit=False)]}
    return {"cache_hit": True, "cache_similarity": round(hit.similarity, 4), "answer": hit.entry.answer,
            "triage": {**hit.entry.triage, "confidence": 1.0, "valid": True},
            "draft": {"answer": hit.entry.answer, "cited_articles": hit.entry.cited_articles, "needs_human": False},
            "model_used": "cache", "status": "answered", "route": "cache",
            "metrics": [_row("cache_lookup", start, model="cache", cache_hit=True)]}


# ---- 2. context from the MCP server ----------------------------------------------------------------------------
def retrieve(state: dict, config=None) -> dict:
    start = time.perf_counter()
    gw, thread = gateway(), _thread(config)
    try:
        articles = gw.call("search_kb", {"query": ticket_text(state)[:2000], "top_k": settings().kb_top_k}, thread)
    except ToolCallError:
        articles = []
    account = None
    if state.get("customer_email"):
        account = gw.try_call("get_account", {"email": state["customer_email"]}, thread)
    low = ticket_text(state).lower()
    injection = any(re.search(p, low) for p in INJECTION_PATTERNS)
    sensitive = any(re.search(p, low) for p in SENSITIVE_PATTERNS)
    return {"articles": articles, "account": account, "injection_suspected": injection, "sensitive": sensitive,
            "metrics": [_row("retrieve", start, n_articles=len(articles), account_found=account is not None)]}


# ---- 3. triage + draft by the small model (the LoRA) -----------------------------------------------------------
def _parse_sft(text: str) -> tuple[dict, str] | None:
    try:
        raw = parse_json(text)
        labels = Triage(**{k: raw.get(k) for k in ("queue", "priority", "type")}).model_dump(include={"queue", "priority", "type"})
        return labels, str(raw.get("answer", "")).strip()
    except (ValueError, ValidationError, TypeError):
        return None


def triage(state: dict) -> dict:
    """Self-consistency: n samples; confidence = share of samples that agree with the majority labels."""
    start = time.perf_counter()
    llm, n = get_llm("small"), max(1, settings().triage_samples)
    messages = prompts.sft_messages(state.get("subject", ""), state.get("body", ""), state.get("articles"))
    results, parsed = [], []
    for i in range(n):
        try:
            r = llm.chat(messages, temperature=0.0 if i == 0 else 0.7, sample=i, max_tokens=700)
        except Exception as err:  # the small model is down: everything goes to the large model
            return {"triage": {**FALLBACK_TRIAGE, "in_scope": True, "confidence": 0.0, "valid": False,
                               "error": str(err)[:200]}, "small_answer": "",
                    "route": choose_route({"valid": False}, state.get("injection_suspected", False),
                                          sensitive=state.get("sensitive", False)),
                    "metrics": [_row("triage", start, results, model="small", error="small_model_unavailable")]}
        results.append(r)
        parsed.append(_parse_sft(r.text))
    valid = [p for p in parsed if p]
    if not valid:
        labels, answer, confidence, ok = dict(FALLBACK_TRIAGE), "", 0.0, False
    else:
        votes = Counter(tuple(sorted(p[0].items())) for p in valid)
        majority, count = votes.most_common(1)[0]
        labels = dict(majority)
        confidence = count / n
        answer = next((p[1] for p in parsed if p and p[0] == labels), valid[0][1])
        ok = parsed[0] is not None
    tri = {**labels, "in_scope": labels["queue"] not in OUT_OF_SCOPE_QUEUES, "confidence": round(confidence, 3),
           "valid": ok}
    route = choose_route(tri, state.get("injection_suspected", False), sensitive=state.get("sensitive", False))
    return {"triage": tri, "small_answer": answer, "route": route,
            "metrics": [_row("triage", start, results, model="small", confidence=tri["confidence"], route=route)]}


# ---- 4a. small-model reply ---------------------------------------------------------------------------------------
def use_small_answer(state: dict) -> dict:
    start = time.perf_counter()
    draft = {"answer": state.get("small_answer", ""), "cited_articles": [], "needs_human": False}
    return {"draft": draft, "model_used": "small", "metrics": [_row("use_small_answer", start, model="small")]}


# ---- 4b. large-model reply -----------------------------------------------------------------------------------------
def generate_large(state: dict) -> dict:
    start = time.perf_counter()
    labels = {k: state.get("triage", {}).get(k) for k in ("queue", "priority", "type")}
    messages = prompts.answer_messages(state.get("subject", ""), state.get("body", ""), labels,
                                       state.get("articles", []), state.get("account"))
    attempts = state.get("attempts_large", 0) + 1
    try:
        r = get_llm("large").chat(messages, temperature=0.0, max_tokens=900)
    except Exception as err:
        return {"draft": {"answer": "", "cited_articles": [], "needs_human": True}, "model_used": "large",
                "attempts_large": attempts,
                "metrics": [_row("generate_large", start, model="large", error=str(err)[:200])]}
    try:
        draft = Draft(**parse_json(r.text)).model_dump()
    except (ValueError, ValidationError, TypeError):
        draft = {"answer": "", "cited_articles": [], "needs_human": True, "parse_error": True}
    return {"draft": draft, "model_used": "large", "attempts_large": attempts,
            "metrics": [_row("generate_large", start, [r], model="large")]}


# ---- 4c. out of scope ----------------------------------------------------------------------------------------------
def out_of_scope(state: dict) -> dict:
    start = time.perf_counter()
    queue = state.get("triage", {}).get("queue")
    answer = prompts.OUT_OF_SCOPE_REPLY.get(queue, prompts.OUT_OF_SCOPE_REPLY["Human Resources"])
    return {"answer": answer, "draft": {"answer": answer, "cited_articles": [], "needs_human": False},
            "model_used": "template", "status": "redirected",
            "metrics": [_row("out_of_scope", start, model="template")]}


# ---- 5. validation ------------------------------------------------------------------------------------------------
def check_draft(draft: dict, articles: list[dict]) -> list[str]:
    errors = []
    answer = (draft or {}).get("answer", "") or ""
    if draft.get("parse_error"):
        errors.append("invalid_json")
    if len(answer.strip()) < 20:
        errors.append("empty_or_too_short")
    if len(answer) > 4000:
        errors.append("too_long")
    if prompts.CANARY in answer or "TASK:" in answer:
        errors.append("leaked_instructions")
    if PLACEHOLDER.search(answer):
        errors.append("placeholder")
    known = {a["id"] for a in articles or []}
    unknown = [c for c in draft.get("cited_articles", []) if c not in known]
    if unknown:
        errors.append("unknown_citation:" + ",".join(unknown))
    return errors


def validate(state: dict) -> dict:
    start = time.perf_counter()
    errors = check_draft(state.get("draft", {}), state.get("articles", []))
    return {"validation_errors": errors,
            "metrics": [_row("validate", start, errors=errors, model_checked=state.get("model_used"))]}


# ---- 6. escalation (MCP writes + optional human approval) ---------------------------------------------------------
def escalate(state: dict, config=None) -> dict:
    start = time.perf_counter()
    gw, thread = gateway(), _thread(config)
    tri = state.get("triage", {})
    draft_ok = not state.get("validation_errors")
    reasons = []
    if state.get("validation_errors"):
        reasons.append("invalid draft: " + ", ".join(state["validation_errors"]))
    if tri.get("priority") == "high":
        reasons.append("high priority")
    if state.get("draft", {}).get("needs_human"):
        reasons.append("model asked for a human")
    if state.get("sensitive"):
        reasons.append("sensitive request (money, data or security)")
    if tri.get("queue") == "Service Outages and Maintenance":
        reasons.append("possible outage")
    reason = "; ".join(reasons) or "escalation"
    record = {}
    try:
        ticket = gw.call("create_ticket", {
            "subject": (state.get("subject") or "Support request")[:300],
            "summary": (ticket_text(state) + "\n\n--- draft ---\n" + state.get("draft", {}).get("answer", ""))[:4000],
            "queue": tri.get("queue", "Customer Service"), "priority": tri.get("priority", "medium"),
            "account_email": state.get("customer_email")}, thread)
        esc = gw.call("escalate_to_human", {"ticket_id": ticket["id"], "reason": reason[:1000],
                                            "queue": tri.get("queue", "Customer Service")}, thread)
        record = {"ticket_id": ticket["id"], "escalation_id": esc["id"], "team": esc["team"], "reason": reason}
    except ToolCallError as err:
        record = {"error": str(err), "reason": reason}

    answer = state.get("draft", {}).get("answer", "") if draft_ok else ""
    if settings().human_in_the_loop:
        from langgraph.types import interrupt
        decision = interrupt({"ticket": record, "draft": answer, "reason": reason,
                              "resume_with": {"approve": True, "answer": "<optional edited reply>"}})
        decision = decision if isinstance(decision, dict) else {"approve": bool(decision)}
        if decision.get("answer"):
            answer = decision["answer"]
        elif not decision.get("approve"):
            answer = ""
    if not answer:
        answer = HOLDING_REPLY.format(ticket=record.get("ticket_id", "pending"))
    return {"answer": answer, "status": "escalated", "escalation": record,
            "metrics": [_row("escalate", start, reason=reason, ticket_id=record.get("ticket_id"))]}


# ---- 7. cache write + finalize ------------------------------------------------------------------------------------
def is_cacheable(state: dict) -> bool:
    tri = state.get("triage", {})
    return (settings().cache_enabled and not state.get("cache_hit") and state.get("status") != "escalated"
            and tri.get("queue") in CACHEABLE_QUEUES and tri.get("type") in CACHEABLE_TYPES
            and tri.get("priority") != "high" and not state.get("account")
            and not state.get("injection_suspected") and not state.get("sensitive")
            and not state.get("validation_errors"))


def cache_write(state: dict) -> dict:
    start = time.perf_counter()
    answer = state.get("draft", {}).get("answer", "")
    stored = False
    if is_cacheable(state) and answer:
        tri = state["triage"]
        shared_cache().add(CacheEntry(text=ticket_text(state), answer=answer,
                                      triage={k: tri[k] for k in ("queue", "priority", "type")},
                                      cited_articles=state.get("draft", {}).get("cited_articles", [])))
        stored = True
    return {"answer": answer, "status": "answered", "metrics": [_row("cache_write", start, stored=stored)]}


def finalize(state: dict) -> dict:
    start = time.perf_counter()
    return {"status": state.get("status") or "answered", "answer": state.get("answer", ""),
            "metrics": [_row("finalize", start, status=state.get("status") or "answered",
                             model_used=state.get("model_used"))]}
