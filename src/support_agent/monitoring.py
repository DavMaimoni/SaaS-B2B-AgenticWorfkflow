import json
from datetime import datetime, timezone
from typing import Dict, Any

class TokenUsageMonitor:
    """Tracks LLM token usage and associates operational costs with specific B2B tenants."""
    def __init__(self, log_file: str = "monitoring/usage_logs.jsonl"):
        self.log_file = log_file

    def log_usage(self, tenant_id: str, usage_metadata: Dict[str, Any]) -> None:
        if not usage_metadata:
            return

        log_entry = {
            # Fixed deprecation warning by using timezone-aware UTC datetime
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": tenant_id,
            "input_tokens": usage_metadata.get("input_tokens", 0),
            "output_tokens": usage_metadata.get("output_tokens", 0),
            "total_tokens": usage_metadata.get("total_tokens", 0)
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
