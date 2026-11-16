from support_agent.cache import SemanticTenantCache, generate_secure_cache_key

def test_secure_cache_key():
    key1 = generate_secure_cache_key("tenant_A", "Reset password")
    key2 = generate_secure_cache_key("tenant_B", "Reset password")
    assert key1 != key2

def test_semantic_tenant_cache():
    cache = SemanticTenantCache(ttl_seconds=3600)
    cache.set_cached_response("tenant_A", "Hello", "Response A")
    
    assert cache.get_cached_response("tenant_A", "Hello") == "Response A"
    assert cache.get_cached_response("tenant_B", "Hello") is None
