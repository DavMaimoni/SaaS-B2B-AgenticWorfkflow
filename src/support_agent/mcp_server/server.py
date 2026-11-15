"""The Fluxo support MCP server: help-center search, CRM reads, and two idempotent writes.

    uv run python -m support_agent.mcp_server.server        # stdio, for any MCP client (Claude Desktop, Inspector...)

The graph reaches it through `support_agent.tools.ToolGateway`, in-process by default or over stdio
(MCP_TRANSPORT=stdio). Hardening follows course lab 2: annotations, constrained argument types, typed errors,
and an `idempotency_key` on writes that is set by the harness, never by the model.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from support_agent.config import CRM_PATH, OUT_OF_SCOPE_QUEUES, PRIORITIES, QUEUES
from support_agent.mcp_server.crm import CRM
from support_agent.mcp_server.kb import knowledge_base

READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False, openWorldHint=False)

Query = Annotated[str, Field(min_length=2, max_length=2000, description="Search words or the ticket text")]
ArticleId = Annotated[str, Field(pattern=r"^KB-[A-Z]+-\d{2}$", description="Article id, e.g. KB-BILL-03")]
Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", description="Customer account email")]
AccountId = Annotated[str, Field(pattern=r"^ACC-\d{4}$", description="Account id, e.g. ACC-1001")]
TicketId = Annotated[str, Field(pattern=r"^TCK-\d{5}$", description="Ticket id, e.g. TCK-00001")]
QueueName = Literal[tuple(QUEUES + OUT_OF_SCOPE_QUEUES)]  # type: ignore[valid-type]
PriorityName = Literal[tuple(PRIORITIES)]  # type: ignore[valid-type]
IdemKey = Annotated[str | None, Field(description="Set by the harness; never by the model")]


def tool_error(code: str, message: str, retryable: bool = False, outcome_unknown: bool = False) -> ToolError:
    """A typed error: `message` is for the model, `retryable` and `outcome_unknown` are for the harness."""
    return ToolError(json.dumps({"code": code, "message": message,
                                 "retryable": retryable, "outcome_unknown": outcome_unknown}))


def build_server(crm: CRM | None = None) -> MCPServer:
    """A server bound to one CRM (tests pass an in-memory CRM; the default is data/crm.sqlite)."""
    crm = crm or CRM(os.getenv("CRM_PATH", CRM_PATH))
    kb = knowledge_base()
    mcp = MCPServer("fluxo-support", instructions="Fluxo customer support tools: help center, CRM and tickets.")

    @mcp.tool(annotations=READ)
    def search_kb(query: Query, top_k: Annotated[int, Field(ge=1, le=5)] = 3) -> list[dict]:
        """Search the Fluxo help center. Returns up to top_k articles (id, title, text, score), best first.
        Returns [] when nothing is relevant: then do not invent an answer."""
        return kb.search(query, top_k=top_k)

    @mcp.tool(annotations=READ)
    def get_article(article_id: ArticleId) -> dict:
        """Full text of one help-center article."""
        article = kb.get(article_id)
        if not article:
            raise tool_error("not_found", f"No article {article_id}. Use search_kb to find ids.")
        return article.as_dict()

    @mcp.tool(annotations=READ)
    def get_account(email: Email) -> dict:
        """The customer's account: plan, billing cycle, seats, status (active | read_only), renewal date, region."""
        account = crm.get_account(email)
        if not account:
            raise tool_error("not_found", f"No account for {email}. Ask the customer for their account email.")
        return account

    @mcp.tool(annotations=READ)
    def list_invoices(account_id: AccountId) -> list[dict]:
        """The account's last invoices, newest first (id, issued_on, amount_usd, status paid | failed)."""
        return crm.list_invoices(account_id)

    @mcp.tool(annotations=WRITE)
    def create_ticket(subject: Annotated[str, Field(min_length=1, max_length=300)],
                      summary: Annotated[str, Field(min_length=1, max_length=4000)],
                      queue: QueueName, priority: PriorityName,
                      account_email: Email | None = None, idempotency_key: IdemKey = None) -> dict:
        """Open a ticket for the level-2 team. Repeating the exact same call is safe: it never creates two tickets."""
        if not idempotency_key:
            raise tool_error("missing_key", "create_ticket needs an idempotency_key (set by the harness).")
        return crm.create_ticket(subject, summary, queue, priority, account_email, idempotency_key)

    @mcp.tool(annotations=WRITE)
    def escalate_to_human(ticket_id: TicketId, reason: Annotated[str, Field(min_length=3, max_length=1000)],
                          queue: QueueName, idempotency_key: IdemKey = None) -> dict:
        """Hand a ticket to a human team (billing, security, sre, engineering or support-l2, chosen from the queue).
        Repeating the exact same call is safe."""
        if not idempotency_key:
            raise tool_error("missing_key", "escalate_to_human needs an idempotency_key (set by the harness).")
        try:
            return crm.escalate(ticket_id, reason, queue, idempotency_key)
        except KeyError as e:
            raise tool_error("not_found", str(e))

    @mcp.resource("kb://articles")
    def list_articles() -> str:
        """Index of the help center: one line per article."""
        return "\n".join(f"{a.id}: {a.title}" for a in kb.articles)

    @mcp.resource("kb://article/{article_id}")
    def article_resource(article_id: str) -> str:
        """One help-center article as markdown."""
        a = kb.get(article_id)
        return f"# {a.title}\n\n{a.text}" if a else f"No article {article_id}"

    @mcp.prompt()
    def support_reply(subject: str, body: str) -> str:
        """Draft a reply to a customer ticket using the help center."""
        return (f"Find the relevant Fluxo help-center articles with search_kb, then draft a polite reply to this "
                f"ticket. Cite the article ids. Do not invent policies.\n\nSubject: {subject}\n\n{body}")

    return mcp


if __name__ == "__main__":
    if sys.stdin.isatty():
        print("Fluxo MCP server on stdio, waiting for a client (Ctrl+C to stop).", file=sys.stderr)
    build_server().run()
