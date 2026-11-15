"""The support workflow as a LangGraph state machine.

START → cache_lookup ─hit─────────────────────────────────────────────────────────────→ finalize → END
              └─miss→ retrieve (MCP) → triage (small model, n samples)
                        ├─ small ────────→ use_small_answer ─┐
                        ├─ large ────────→ generate_large ───┤
                        └─ out_of_scope ─→ out_of_scope ─────┼──────────────────────→ finalize
                                                             ▼
                                                          validate
                          invalid small reply → generate_large (1 retry)
                          invalid large reply, needs a human, high priority → escalate (MCP writes, interrupt) → finalize
                          ok → cache_write → finalize
"""

from __future__ import annotations

import uuid

from langgraph.graph import END, START, StateGraph

from support_agent import nodes
from support_agent.routing import route_after_cache, route_after_triage, route_after_validate
from support_agent.state import SupportState, TicketInput


def build_graph(checkpointer=None, store=None):
    g = StateGraph(SupportState, input_schema=TicketInput)
    for name, fn in [("cache_lookup", nodes.cache_lookup), ("retrieve", nodes.retrieve), ("triage", nodes.triage),
                     ("use_small_answer", nodes.use_small_answer), ("generate_large", nodes.generate_large),
                     ("out_of_scope", nodes.out_of_scope), ("validate", nodes.validate),
                     ("escalate", nodes.escalate), ("cache_write", nodes.cache_write), ("finalize", nodes.finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "cache_lookup")
    g.add_conditional_edges("cache_lookup", route_after_cache, ["retrieve", "finalize"])
    g.add_edge("retrieve", "triage")
    g.add_conditional_edges("triage", route_after_triage, ["use_small_answer", "generate_large", "out_of_scope"])
    g.add_edge("use_small_answer", "validate")
    g.add_edge("generate_large", "validate")
    g.add_conditional_edges("validate", route_after_validate, ["generate_large", "escalate", "cache_write"])
    g.add_edge("out_of_scope", "finalize")
    g.add_edge("escalate", "finalize")
    g.add_edge("cache_write", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, store=store)


def run_ticket(app, subject: str, body: str, customer_email: str | None = None, thread_id: str | None = None,
               recursion_limit: int = 25) -> dict:
    """Run one ticket; returns the final state (with `__interrupt__` if a human approval is pending)."""
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": recursion_limit}
    payload = {"subject": subject, "body": body}
    if customer_email:
        payload["customer_email"] = customer_email
    return app.invoke(payload, config)


def resume(app, thread_id: str, approve: bool = True, answer: str | None = None) -> dict:
    """Resume a ticket paused at the human approval step (HUMAN_IN_THE_LOOP=1)."""
    from langgraph.types import Command
    decision = {"approve": approve, **({"answer": answer} if answer else {})}
    return app.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})


def summarize_metrics(state: dict) -> dict:
    """Totals over the per-node metrics rows of one run."""
    rows = state.get("metrics", [])
    return {"latency_ms": round(sum(r["latency_ms"] for r in rows), 1),
            "cost_usd": round(sum(r["cost_usd"] for r in rows), 8),
            "tokens_in": sum(r["tokens_in"] for r in rows), "tokens_out": sum(r["tokens_out"] for r in rows),
            "llm_calls": sum(r["llm_calls"] for r in rows), "nodes": [r["node"] for r in rows]}
