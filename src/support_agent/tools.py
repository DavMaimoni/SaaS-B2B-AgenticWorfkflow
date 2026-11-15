"""The graph's door to the MCP server.

`ToolGateway.call(name, args)` returns the tool's result as plain Python, or raises `ToolCallError` with the typed
error the server sent (code, retryable, outcome_unknown). Reads that fail with `retryable` are retried; writes get
an idempotency key computed by the harness (same thread + tool + arguments = same key), never by a model.

Transport (MCP_TRANSPORT):
- `inprocess` (default): the server object runs in this process; fast, used by tests and the evaluation.
- `stdio`: the server is launched as a subprocess, exactly as an external MCP client would use it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mcp import Client, StdioServerParameters

from support_agent.mcp_server.crm import CRM
from support_agent.mcp_server.server import build_server

WRITE_TOOLS = {"create_ticket", "escalate_to_human"}


class ToolCallError(RuntimeError):
    def __init__(self, tool: str, error: dict):
        self.tool, self.error = tool, error
        super().__init__(f"{tool}: {error.get('code')}: {error.get('message')}")


def idempotency_key(thread_id: str, tool: str, args: dict) -> str:
    raw = json.dumps([thread_id, tool, args], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _parse_error(text: str) -> dict:
    try:
        return json.loads(text[text.index("{"):])
    except ValueError:
        return {"code": "error", "message": text, "retryable": False, "outcome_unknown": False}


def _parse_result(result) -> object:
    if result.is_error:
        raise ToolCallError("tool", _parse_error(result.content[0].text if result.content else ""))
    sc = result.structured_content
    if sc is not None:
        return sc["result"] if set(sc) == {"result"} else sc
    if not result.content:
        return None
    if len(result.content) == 1:
        text = result.content[0].text
        try:
            return json.loads(text)
        except ValueError:
            return text
    return [json.loads(c.text) for c in result.content]


class ToolGateway:
    def __init__(self, crm: CRM | None = None, transport: str | None = None, max_read_retries: int = 2):
        self.transport = transport or os.getenv("MCP_TRANSPORT", "inprocess")
        self.crm = crm
        self.max_read_retries = max_read_retries
        self._server = build_server(crm) if self.transport == "inprocess" else None
        self._pool = ThreadPoolExecutor(max_workers=1)   # one event loop thread: safe inside Jupyter / LangGraph
        self._lock = threading.Lock()
        self.calls: list[dict] = []                      # every call, for the trace and the evaluation

    def _target(self):
        if self.transport == "stdio":
            env = {**os.environ}
            return StdioServerParameters(command=sys.executable, args=["-m", "support_agent.mcp_server.server"], env=env)
        return self._server

    async def _acall(self, name: str, args: dict):
        async with Client(self._target()) as client:
            return await client.call_tool(name, args)

    def call(self, name: str, args: dict, thread_id: str = "default"):
        args = dict(args)
        if name in WRITE_TOOLS and "idempotency_key" not in args:
            args["idempotency_key"] = idempotency_key(thread_id, name, args)
        # Writes may be retried too: the idempotency key makes a replay harmless (course lab 2).
        attempts = 1 + self.max_read_retries
        last: ToolCallError | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            result = self._pool.submit(asyncio.run, self._acall(name, args)).result()
            ms = (time.perf_counter() - start) * 1000
            try:
                value = _parse_result(result)
                self.calls.append({"tool": name, "args": args, "ok": True, "ms": round(ms, 1)})
                return value
            except ToolCallError as err:
                err.tool = name
                self.calls.append({"tool": name, "args": args, "ok": False, "error": err.error, "ms": round(ms, 1)})
                last = err
                if not err.error.get("retryable"):
                    break
                time.sleep(0.2 * (attempt + 1))
        raise last  # type: ignore[misc]

    def try_call(self, name: str, args: dict, thread_id: str = "default", default=None):
        """Like `call`, but returns `default` on a tool error (for optional context such as the account)."""
        try:
            return self.call(name, args, thread_id)
        except ToolCallError:
            return default


_gateway: ToolGateway | None = None


def gateway() -> ToolGateway:
    global _gateway
    if _gateway is None:
        _gateway = ToolGateway()
    return _gateway


def set_gateway(gw: ToolGateway | None) -> None:
    global _gateway
    _gateway = gw
