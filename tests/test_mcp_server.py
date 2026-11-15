import asyncio

import pytest
from mcp import Client

from support_agent.mcp_server.crm import CRM
from support_agent.mcp_server.server import build_server
from support_agent.tools import ToolCallError, ToolGateway


def test_tools_are_annotated_and_constrained():
    async def go():
        async with Client(build_server(CRM(":memory:"))) as c:
            return {t.name: t for t in (await c.list_tools()).tools}
    tools = asyncio.run(go())
    assert set(tools) == {"search_kb", "get_article", "get_account", "list_invoices", "create_ticket",
                          "escalate_to_human"}
    for name in ("search_kb", "get_article", "get_account", "list_invoices"):
        assert tools[name].annotations.read_only_hint is True
    for name in ("create_ticket", "escalate_to_human"):
        assert tools[name].annotations.read_only_hint is False
        assert tools[name].annotations.idempotent_hint is True
        assert "idempotency_key" in tools[name].input_schema["properties"]
    assert "pattern" in tools["get_article"].input_schema["properties"]["article_id"]
    assert "enum" in tools["create_ticket"].input_schema["properties"]["queue"]


def test_search_finds_the_right_article():
    gw = ToolGateway(CRM(":memory:"))
    hits = gw.call("search_kb", {"query": "API answers 429 Too Many Requests", "top_k": 3})
    assert hits[0]["id"] == "KB-API-01"
    hits = gw.call("search_kb", {"query": "SAML error for every user since this morning", "top_k": 3})
    assert hits[0]["id"] == "KB-SEC-01"


def test_typed_errors_reach_the_graph():
    gw = ToolGateway(CRM(":memory:"))
    with pytest.raises(ToolCallError) as err:
        gw.call("get_account", {"email": "nobody@unknown.example.com"})
    assert err.value.error["code"] == "not_found"
    assert gw.try_call("get_account", {"email": "nobody@unknown.example.com"}, default="none") == "none"


def test_writes_are_idempotent():
    crm = CRM(":memory:")
    gw = ToolGateway(crm)
    args = {"subject": "API down", "summary": "500 on every call", "queue": "Technical Support", "priority": "high"}
    first = gw.call("create_ticket", args, thread_id="t1")
    second = gw.call("create_ticket", args, thread_id="t1")   # same thread + args = same key: a replay
    assert first["id"] == second["id"] and second["replayed"] is True
    assert len(crm.tickets()) == 1
    other = gw.call("create_ticket", args, thread_id="t2")     # another conversation: a new ticket
    assert other["id"] != first["id"]
    esc = gw.call("escalate_to_human", {"ticket_id": first["id"], "reason": "outage", "queue": "Technical Support"},
                  thread_id="t1")
    assert esc["team"] == "engineering"
    assert crm.tickets()[0]["status"] == "escalated"


def test_account_and_invoices():
    gw = ToolGateway(CRM(":memory:"))
    acc = gw.call("get_account", {"email": "admin@lumen-retail-group.example.com"})
    assert acc["status"] == "read_only"
    invoices = gw.call("list_invoices", {"account_id": acc["id"]})
    assert invoices[0]["status"] == "failed"
