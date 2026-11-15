"""The graph's door to the MCP server.

`ToolGateway.call(name, args)` returns the tool's result as plain Python, or raises `ToolCallError` with a typed
error (code, retryable, outcome_unknown): either the one the server sent, or one built here for a transport failure
(`timeout`, `transport_error`). Calls whose error is `retryable` are retried; writes get an idempotency key computed
by the harness (same thread + tool + arguments = same key), never by a model, so retrying a write is harmless.

Transport (MCP_TRANSPORT):
- `inprocess` (default): the server object runs in this process; fast, used by tests and the evaluation.
- `stdio`: the server runs as a subprocess, exactly as an external MCP client would use it.
Either way ONE client session stays open on a background event loop (one subprocess in stdio mode, not one per
call). Every call has a timeout (MCP_TIMEOUT, seconds); after a timeout or a transport error the session is closed
and the next call opens a new one.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time

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


class _Session:
    """One MCP client session served by a task on a private event loop thread. Calls are queued and run one at a
    time (the CRM serialises writes anyway); a caller waits at most `timeout` seconds for its result."""

    def __init__(self, target, timeout: float):
        self.alive = True
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-session")
        self._thread.start()
        ready: concurrent.futures.Future = concurrent.futures.Future()
        asyncio.run_coroutine_threadsafe(self._serve(target, ready), self._loop)
        try:
            ready.result(timeout=timeout)
        except BaseException:
            self.close()
            raise

    async def _serve(self, target, ready: concurrent.futures.Future) -> None:
        self._task = asyncio.current_task()
        self._queue = asyncio.Queue()
        try:
            async with Client(target) as client:
                ready.set_result(None)
                while True:
                    item = await self._queue.get()
                    if item is None:
                        return
                    name, args, fut = item
                    if fut.done():                     # the caller already gave up (timeout)
                        continue
                    try:
                        result = await client.call_tool(name, args)
                    except Exception as err:
                        if not fut.done():
                            fut.set_exception(err)
                    else:
                        if not fut.done():
                            fut.set_result(result)
        except BaseException as err:
            if not ready.done():
                ready.set_exception(err if isinstance(err, Exception) else ConnectionError("MCP session cancelled"))
            if not isinstance(err, (Exception, asyncio.CancelledError)):
                raise
        finally:
            self.alive = False
            while self._queue is not None and not self._queue.empty():
                item = self._queue.get_nowait()
                if item and not item[2].done():
                    item[2].set_exception(ConnectionError("MCP session closed"))

    def call(self, name: str, args: dict, timeout: float):
        if not self.alive or self._queue is None:
            raise ConnectionError("MCP session closed")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, args, fut))
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            raise TimeoutError(f"no answer from the MCP server after {timeout:g} s") from None

    def close(self, wait: float = 3.0) -> None:
        """Stop the session task (this ends the stdio subprocess), then the loop thread."""
        self.alive = False

        async def stop():
            task = self._task
            if task and not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=wait)

        if self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(stop(), self._loop).result(timeout=wait + 1)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=wait)


class ToolGateway:
    def __init__(self, crm: CRM | None = None, transport: str | None = None, max_read_retries: int = 2,
                 timeout: float | None = None, server=None):
        self.transport = transport or os.getenv("MCP_TRANSPORT", "inprocess")
        self.crm = crm
        self.max_read_retries = max_read_retries
        self.timeout = timeout if timeout is not None else float(os.getenv("MCP_TIMEOUT", 30))
        # `server` lets tests plug in their own MCP server object (e.g. one with a slow tool).
        self._server = server or (build_server(crm) if self.transport == "inprocess" else None)
        self._session: _Session | None = None
        self._lock = threading.Lock()
        self.calls: list[dict] = []                      # every call, for the trace and the evaluation
        atexit.register(self.close)

    def _target(self):
        if self.transport == "stdio" and self._server is None:
            return StdioServerParameters(command=sys.executable, args=["-m", "support_agent.mcp_server.server"],
                                         env={**os.environ})
        return self._server

    def _get_session(self) -> _Session:
        with self._lock:
            if self._session is None or not self._session.alive:
                self._session = _Session(self._target(), self.timeout)
            return self._session

    def _drop_session(self) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session:
            session.close()

    def close(self) -> None:
        self._drop_session()

    def call(self, name: str, args: dict, thread_id: str = "default"):
        args = dict(args)
        if name in WRITE_TOOLS and "idempotency_key" not in args:
            args["idempotency_key"] = idempotency_key(thread_id, name, args)
        # Writes may be retried too: the idempotency key makes a replay harmless (course lab 2).
        attempts = 1 + self.max_read_retries
        last: ToolCallError | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                value = _parse_result(self._get_session().call(name, args, self.timeout))
            except ToolCallError as err:
                err.tool = name
                last = err
            except Exception as err:                     # timeout, dead subprocess, closed session...
                self._drop_session()
                last = ToolCallError(name, {
                    "code": "timeout" if isinstance(err, TimeoutError) else "transport_error",
                    "message": f"{type(err).__name__}: {err}"[:300], "retryable": True,
                    "outcome_unknown": name in WRITE_TOOLS})
            else:
                self.calls.append({"tool": name, "args": args, "ok": True,
                                   "ms": round((time.perf_counter() - start) * 1000, 1)})
                return value
            self.calls.append({"tool": name, "args": args, "ok": False, "error": last.error,
                               "ms": round((time.perf_counter() - start) * 1000, 1)})
            if not last.error.get("retryable"):
                break
            if attempt + 1 < attempts:
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
    """Replace the shared gateway; the previous one is closed (its session and subprocess end)."""
    global _gateway
    if _gateway is not None and _gateway is not gw:
        _gateway.close()
    _gateway = gw
