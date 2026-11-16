import hashlib
import time
from typing import Optional, Dict, Any

def generate_secure_cache_key(tenant_id: str, query: str) -> str:
    """Generates a cryptographically isolated cache key per client."""
    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    return f"tenant_{tenant_id}_q_{query_hash}"

class SemanticTenantCache:
    """In-memory semantic cache manager with strict B2B tenant isolation."""
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache_store: Dict[str, Dict[str, Any]] = {}

    def get_cached_response(self, tenant_id: str, query: str) -> Optional[str]:
        key = generate_secure_cache_key(tenant_id, query)
        
        if key in self.cache_store:
            entry = self.cache_store[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                del self.cache_store[key]
        
        return None

    def set_cached_response(self, tenant_id: str, query: str, response: str) -> None:
        key = generate_secure_cache_key(tenant_id, query)
        self.cache_store[key] = {
            "response": response,
            "timestamp": time.time()
        }
