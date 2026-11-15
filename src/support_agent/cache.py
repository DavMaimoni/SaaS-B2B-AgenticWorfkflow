"""Semantic cache: answer a ticket that is (almost) the same as one already answered, without any model call.

Rules (see the project doc):
- only generic answers are stored (cacheable queue and type, no account data, not escalated);
- a hit needs cosine similarity >= threshold (calibrate it with `eval/calibrate.py cache`);
- entries expire after a TTL, and an entry is dropped as soon as one of the KB articles it was built on changes
  (each entry stores the content hash of those articles: `invalidate_stale`);
- with a `path` (CACHE_PATH), every change is written to a JSONL file, so the cache survives a restart.
FAISS is used when installed (inner product on normalised vectors = cosine), numpy otherwise.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from support_agent.config import settings
from support_agent.embeddings import get_embedder


@dataclass
class CacheEntry:
    text: str
    answer: str
    triage: dict
    cited_articles: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # {article id: content hash} of every article the answer may depend on (cited or given as context).
    article_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float


class SemanticCache:
    def __init__(self, threshold: float | None = None, ttl_seconds: int | None = None, embedder=None,
                 path: Path | str | None = None):
        s = settings()
        self.threshold = s.cache_threshold if threshold is None else threshold
        self.ttl = s.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        self.embedder = embedder or get_embedder()
        self.entries: list[CacheEntry] = []
        self.vectors: np.ndarray | None = None
        self._faiss = None
        self.path = Path(path) if path else None
        if self.path:
            self.load(self.path)

    def __len__(self) -> int:
        return len(self.entries)

    # -- lookup / write ----------------------------------------------------------------------------------------
    def best(self, text: str) -> CacheHit | None:
        """Most similar live entry, whatever its similarity (used for calibration)."""
        self._expire()
        if not self.entries:
            return None
        q = self.embedder.encode([text])[0]
        if self._faiss is not None:
            sims, idx = self._faiss.search(q[None, :], 1)
            i, sim = int(idx[0][0]), float(sims[0][0])
        else:
            sims = self.vectors @ q
            i = int(np.argmax(sims))
            sim = float(sims[i])
        return CacheHit(self.entries[i], sim)

    def lookup(self, text: str) -> CacheHit | None:
        hit = self.best(text)
        return hit if hit and hit.similarity >= self.threshold else None

    def add(self, entry: CacheEntry) -> None:
        vec = self.embedder.encode([entry.text])
        self.entries.append(entry)
        self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])
        self._rebuild_index()
        self._persist()

    # -- invalidation --------------------------------------------------------------------------------------------
    def invalidate_article(self, article_id: str) -> int:
        """Drop every entry built on this article. Returns how many were dropped."""
        keep = [i for i, e in enumerate(self.entries)
                if article_id not in e.cited_articles and article_id not in e.article_versions]
        return self._keep(keep) if len(keep) != len(self.entries) else 0

    def invalidate_stale(self, current: dict[str, str]) -> int:
        """Drop every entry built on an article whose content changed or that was deleted.
        `current` is {article id: content hash} for the help center now (see `mcp_server.kb.article_versions`)."""
        keep = [i for i, e in enumerate(self.entries)
                if all(current.get(aid) == h for aid, h in e.article_versions.items())]
        return self._keep(keep) if len(keep) != len(self.entries) else 0

    def clear(self) -> None:
        self.entries, self.vectors = [], None
        self._rebuild_index()
        self._persist()

    def _expire(self) -> None:
        if self.ttl <= 0 or not self.entries:
            return
        now = time.time()
        keep = [i for i, e in enumerate(self.entries) if now - e.created_at < self.ttl]
        if len(keep) != len(self.entries):
            self._keep(keep)

    def _keep(self, keep: list[int]) -> int:
        dropped = len(self.entries) - len(keep)
        self.entries = [self.entries[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else None
        self._rebuild_index()
        self._persist()
        return dropped

    def _rebuild_index(self) -> None:
        self._faiss = None
        if self.vectors is None:
            return
        try:
            import faiss
        except ImportError:
            return
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors.astype(np.float32))
        self._faiss = index

    # -- persistence ---------------------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    def load(self, path: Path) -> None:
        """Replace the entries with the file's (expired ones are dropped on the next lookup)."""
        path = Path(path)
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]             if path.exists() else []
        self.entries = [CacheEntry(**r) for r in rows]
        self.vectors = self.embedder.encode([e.text for e in self.entries]) if self.entries else None
        self._rebuild_index()

    def _persist(self) -> None:
        if self.path:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            self.save(tmp)
            tmp.replace(self.path)                     # atomic: a crash never leaves a half-written cache


_shared: SemanticCache | None = None


def shared_cache() -> SemanticCache:
    """The process-wide cache used by the graph."""
    global _shared
    if _shared is None:
        _shared = SemanticCache(path=settings().cache_path)
    return _shared


def reset_shared_cache() -> None:
    global _shared
    _shared = None
