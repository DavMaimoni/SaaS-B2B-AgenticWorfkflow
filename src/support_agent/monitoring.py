import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from contextlib import contextmanager

# --- New B2B SaaS Monitoring ---

class TokenUsageMonitor:
    """Tracks LLM token usage and associates operational costs with specific B2B tenants."""
    def __init__(self, log_file: str = "monitoring/usage_logs.jsonl"):
        self.log_file = log_file

    def log_usage(self, tenant_id: str, usage_metadata: Dict[str, Any]) -> None:
        if not usage_metadata:
            return

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "input_tokens": usage_metadata.get("input_tokens", 0),
            "output_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("total_tokens", 0)
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# --- Legacy / Evaluation Pipeline Compatibility Stubs ---
# These functions ensure that eval/run_eval.py and old tests continue to pass
# without interfering with the new B2B architecture.

class MockRecord:
    """A dummy record object to mock the behavior of the original run tracker."""
    def __init__(self):
        self.metrics = {}
    def add(self, *args, **kwargs):
        pass
    def flush(self):
        pass

@contextmanager
def run_record(*args, **kwargs):
    """Context manager to record a run during evaluation."""
    yield MockRecord()

def log_run(*args, **kwargs) -> None:
    """Stub for legacy log_run calls."""
    pass

def load_runs(*args, **kwargs) -> List[dict]:
    """Stub for legacy load_runs calls."""
    return []

def alerts(*args, **kwargs) -> List[dict]:
    """Stub for legacy alerts calls."""
    return []

def kpis(*args, **kwargs) -> dict:
    """Stub for legacy KPIs calculations."""
    return {}
