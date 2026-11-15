import time

from support_agent.cache import CacheEntry, SemanticCache

TICKET = "How do I save a project as a template so my team can reuse it?"


def entry(text=TICKET, cited=("KB-PROD-01",)):
    return CacheEntry(text=text, answer="Use Save as template.", triage={"queue": "Product Support"},
                      cited_articles=list(cited))


def test_hit_on_same_request_with_greetings_and_miss_on_other_request():
    c = SemanticCache(threshold=0.85)
    c.add(entry())
    assert c.lookup("Hello, how do I save a project as a template so my team can reuse it? Thanks!")
    assert c.lookup("Our invoice shows the wrong company name, can you fix it?") is None


def test_trap_pairs_stay_below_a_strict_threshold():
    c = SemanticCache(threshold=0.92)
    c.add(entry("How do I add seats to our plan?"))
    assert c.lookup("How do I remove seats from our plan?") is None


def test_invalidation_by_article_and_ttl():
    c = SemanticCache(threshold=0.85, ttl_seconds=3600)
    c.add(entry())
    c.add(entry("Where can I download my invoices?", cited=("KB-BILL-02",)))
    assert c.invalidate_article("KB-PROD-01") == 1
    assert len(c) == 1
    c.entries[0].created_at = time.time() - 7200
    assert c.lookup("Where can I download my invoices?") is None
    assert len(c) == 0


def test_save_and_load(tmp_path):
    c = SemanticCache(threshold=0.85)
    c.add(entry())
    c.save(tmp_path / "cache.jsonl")
    d = SemanticCache(threshold=0.85)
    d.load(tmp_path / "cache.jsonl")
    assert d.lookup(TICKET).entry.answer == "Use Save as template."


def test_entries_built_on_a_changed_or_deleted_article_are_dropped():
    c = SemanticCache(threshold=0.85)
    c.add(CacheEntry(text=TICKET, answer="a", triage={}, article_versions={"KB-PROD-01": "v1"}))
    c.add(CacheEntry(text="Where can I download my invoices?", answer="b", triage={},
                     article_versions={"KB-BILL-02": "v1"}))
    assert c.invalidate_stale({"KB-PROD-01": "v1", "KB-BILL-02": "v1"}) == 0
    assert c.invalidate_stale({"KB-PROD-01": "v2", "KB-BILL-02": "v1"}) == 1     # edited
    assert c.invalidate_stale({}) == 1                                          # deleted
    assert len(c) == 0


def test_cache_with_a_path_survives_a_restart(tmp_path):
    path = tmp_path / "cache.jsonl"
    c = SemanticCache(threshold=0.85, path=path)
    c.add(entry())
    assert SemanticCache(threshold=0.85, path=path).lookup(TICKET).entry.answer == "Use Save as template."
    c.invalidate_article("KB-PROD-01")
    assert len(SemanticCache(threshold=0.85, path=path)) == 0                   # the invalidation is saved too
