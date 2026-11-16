import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """
    Configures the global test environment.
    Mocks environment variables and prevents actual API calls.
    """
    for name in ("SMALL", "LARGE", "JUDGE"):
        monkeypatch.setenv(f"{name}_PROVIDER", "mock")
    
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("CACHE_ENABLED", "1")
    monkeypatch.setenv("CACHE_PATH", "")
    monkeypatch.setenv("HUMAN_IN_THE_LOOP", "0")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CRM_PATH", ":memory:")

    # Import at runtime to ensure env vars are applied
    from support_agent import llm
    
    # Reset only the LLM clients (shared cache no longer exists in B2B)
    llm.reset_clients()
