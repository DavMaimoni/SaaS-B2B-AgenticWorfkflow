"""The graph state. Nodes return only the keys they change; `metrics` is appended to (one row per node)."""

import operator
from typing import Annotated, TypedDict


class TicketInput(TypedDict, total=False):
    subject: str
    body: str
    customer_email: str | None
    ticket_id: str


class SupportState(TicketInput, total=False):
    # context
    articles: list[dict]            # help-center articles from the MCP server
    account: dict | None            # CRM account, when the customer email is known
    injection_suspected: bool
    sensitive: bool                 # refund, data deletion, security...: large model + escalation
    # triage (small model)
    triage: dict                    # queue, priority, type, in_scope, confidence, valid
    small_answer: str               # the reply the small model drafted in the same call
    route: str                      # small | large | out_of_scope
    # answer
    draft: dict                     # answer, cited_articles, needs_human
    model_used: str                 # small | large | cache | template
    attempts_large: int
    validation_errors: list[str]
    # outcome
    answer: str
    status: str                     # answered | escalated | redirected
    cache_hit: bool
    cache_similarity: float
    escalation: dict | None         # ticket and escalation records created through MCP
    metrics: Annotated[list[dict], operator.add]
