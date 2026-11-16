import hashlib
import time
from typing import Optional, Dict, Any

def generate_secure_cache_key(tenant_id: str, query: str) -> str:
    """
    Generates a cryptographically isolated cache key per client.
    Ensures that identical queries from different tenants do not share cached responses.
    """
    query_hash = hashlib.sha256(query.lower().strip().encode()).hexdigest()
    return f"tenant_{tenant_id}_q_{query_hash}"

class SemanticTenantCache:
    """
    In-memory semantic cache manager with strict B2B tenant isolation and TTL expiration.
    """
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        # In a production environment, replace this dict with Redis or an in-memory Vector DB (e.g., Chroma/FAISS)
        self.cache_store: Dict[str, Dict[str, Any]] = {}

    def get_cached_response(self, tenant_id: str, query: str) -> Optional[str]:
        """
        Retrieves a cached response if it exists and is within the TTL window.
        """
        key = generate_secure_cache_key(tenant_id, query)
        
        if key in self.cache_store:
            entry = self.cache_store[key]
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                return entry["response"]
            else:
                # TTL Invalidation: Remove expired cache entry
                del self.cache_store[key]
        
        # Fallback logic for vector similarity search (e.g., threshold > 0.95) would go here
        return None

    def set_cached_response(self, tenant_id: str, query: str, response: str) -> None:
        """
        Stores a generated response in the cache with the current timestamp.
        """
        key = generate_secure_cache_key(tenant_id, query)
        self.cache_store[key] = {
            "response": response,
            "timestamp": time.time()
        }
