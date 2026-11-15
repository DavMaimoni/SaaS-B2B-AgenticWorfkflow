"""Every test runs offline: mock models, in-memory CRM, temporary runs/ folder, fresh singletons."""

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    for name in ("SMALL", "LARGE", "JUDGE"):
        monkeypatch.setenv(f"{name}_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("CACHE_ENABLED", "1")
    monkeypatch.setenv("CACHE_PATH", "")
    monkeypatch.setenv("HUMAN_IN_THE_LOOP", "0")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CRM_PATH", ":memory:")

    from support_agent import cache, llm, tools
    from support_agent.mcp_server.crm import CRM
    llm.reset_clients()
    cache.reset_shared_cache()
    tools.set_gateway(tools.ToolGateway(CRM(":memory:")))
    yield
    tools.set_gateway(None)
    cache.reset_shared_cache()
    llm.reset_clients()
