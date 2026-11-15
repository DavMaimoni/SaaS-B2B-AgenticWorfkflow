"""The Fluxo help center: markdown articles in data/kb, searched with TF-IDF (word + character n-grams).

TF-IDF is enough for 17 short articles and needs no model download. To reuse the hybrid BM25 + dense + rerank
retriever of our RAG project, replace `search` and keep its signature.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from support_agent.config import KB_DIR


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    tags: tuple[str, ...]
    text: str

    def as_dict(self, max_chars: int | None = None) -> dict:
        text = self.text if max_chars is None else self.text[:max_chars]
        return {"id": self.id, "title": self.title, "tags": list(self.tags), "text": text}


def _parse(path: Path) -> Article:
    raw = path.read_text(encoding="utf-8")
    meta, body = {}, raw
    match = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.DOTALL)
    if match:
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
        body = match.group(2)
    tags = tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip())
    return Article(meta.get("id", path.stem), meta.get("title", path.stem), tags, body.strip())


class KnowledgeBase:
    def __init__(self, directory: Path = KB_DIR):
        self.articles = [_parse(p) for p in sorted(Path(directory).glob("*.md"))]
        self.by_id = {a.id: a for a in self.articles}
        docs = [f"{a.title} {a.title} {' '.join(a.tags)} {a.text}" for a in self.articles]
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][:top_k]
        return [{**self.articles[i].as_dict(), "score": round(float(scores[i]), 4)}
                for i in order if scores[i] >= min_score]

    def get(self, article_id: str) -> Article | None:
        return self.by_id.get(article_id)


_versions_memo: dict[Path, tuple[tuple[int, int], str, str]] = {}   # path -> ((mtime_ns, size), id, hash)


def article_versions(directory: Path = KB_DIR) -> dict[str, str]:
    """{article id: content hash}. Only files whose mtime or size changed are re-read, so it is cheap per ticket.
    The semantic cache compares it with the versions stored in each entry to drop answers built on old articles."""
    out = {}
    for path in sorted(Path(directory).glob("*.md")):
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
        memo = _versions_memo.get(path)
        if not memo or memo[0] != stamp:
            raw = path.read_bytes()
            memo = (stamp, _parse(path).id, hashlib.sha256(raw).hexdigest()[:16])
            _versions_memo[path] = memo
        out[memo[1]] = memo[2]
    return out


@lru_cache(maxsize=4)
def _knowledge_base(directory: Path, fingerprint: tuple) -> KnowledgeBase:
    return KnowledgeBase(directory)


def knowledge_base(directory: Path = KB_DIR) -> KnowledgeBase:
    """The help center, rebuilt when an article is added, removed or edited."""
    return _knowledge_base(Path(directory), tuple(sorted(article_versions(directory).items())))
