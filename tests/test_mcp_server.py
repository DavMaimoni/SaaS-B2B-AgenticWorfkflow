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


def test_knowledge_base_follows_article_edits(tmp_path):
    from support_agent.config import KB_DIR
    from support_agent.mcp_server.kb import article_versions, knowledge_base
    for name in ("KB-PROD-01.md", "KB-BILL-01.md"):
        (tmp_path / name).write_text((KB_DIR / name).read_text(encoding="utf-8"), encoding="utf-8")
    before = article_versions(tmp_path)
    assert "Zanzibarite" not in knowledge_base(tmp_path).get("KB-PROD-01").text
    path = tmp_path / "KB-PROD-01.md"
    path.write_text(path.read_text(encoding="utf-8") + "\nZanzibarite templates are supported.\n", encoding="utf-8")
    after = article_versions(tmp_path)
    assert after["KB-PROD-01"] != before["KB-PROD-01"] and after["KB-BILL-01"] == before["KB-BILL-01"]
    assert "Zanzibarite" in knowledge_base(tmp_path).get("KB-PROD-01").text      # index rebuilt, no restart
    assert knowledge_base(tmp_path).search("Zanzibarite templates")[0]["id"] == "KB-PROD-01"


def test_slow_tool_times_out_and_the_gateway_recovers():
    from mcp.server.mcpserver import MCPServer
    srv = MCPServer("slow")

    @srv.tool()
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    @srv.tool()
    def fast() -> str:
        return "ok"

    gw = ToolGateway(server=srv, timeout=0.5, max_read_retries=0)
    with pytest.raises(ToolCallError) as err:
        gw.call("slow", {})
    assert err.value.error["code"] == "timeout" and err.value.error["retryable"] is True
    assert gw.call("fast", {}) == "ok"                     # a new session was opened
    gw.close()


def test_transport_errors_are_typed_and_retried(monkeypatch):
    from support_agent import tools
    gw = ToolGateway(CRM(":memory:"))
    real, failures = tools._Session.call, []

    def flaky(self, name, args, timeout):
        if not failures:
            failures.append(name)
            raise ConnectionError("server process died")
        return real(self, name, args, timeout)
    monkeypatch.setattr(tools._Session, "call", flaky)
    assert gw.call("search_kb", {"query": "invoice", "top_k": 1})
    assert [c["ok"] for c in gw.calls] == [False, True]
    assert gw.calls[0]["error"]["code"] == "transport_error"


def test_locked_crm_is_a_retryable_error():
    import sqlite3

    class LockedOnce(CRM):
        locked = True

        def get_account(self, email):
            if LockedOnce.locked:
                LockedOnce.locked = False
                raise sqlite3.OperationalError("database is locked")
            return super().get_account(email)

    gw = ToolGateway(LockedOnce(":memory:"))
    assert gw.call("get_account", {"email": "admin@quanta-labs.example.com"})["id"].startswith("ACC-")
    assert gw.calls[0]["error"]["code"] == "unavailable" and gw.calls[1]["ok"]


def test_stdio_keeps_one_server_process_for_many_calls():
    gw = ToolGateway(transport="stdio")
    try:
        gw.call("search_kb", {"query": "invoice", "top_k": 1})
        session = gw._session
        gw.call("search_kb", {"query": "API rate limit", "top_k": 1})
        assert gw._session is session and session.alive
    finally:
        gw.close()
    assert gw._session is None
