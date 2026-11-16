import os
import json
from support_agent.monitoring import TokenUsageMonitor

def test_token_usage_monitor(tmp_path):
    log_file = tmp_path / "usage_logs.jsonl"
    monitor = TokenUsageMonitor(log_file=str(log_file))
    
    usage_data = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
    monitor.log_usage("tenant_123", usage_data)
    
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        logs = [json.loads(line) for line in f]
    
    assert len(logs) == 1
    assert logs[0]["tenant_id"] == "tenant_123"
    assert logs[0]["total_tokens"] == 30
