"""Load and clean the Tobi-Bueck customer-support-tickets dataset.

Source: https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets (CC BY-NC 4.0).
Columns: subject, body, answer, type, queue, priority, language, version, tag_1..tag_8.
The card documents 10 queues and 3 priorities, the viewer shows 52 queue values and 5 priority values: every row
whose labels are not in our label set is dropped, and the counts are reported in data/processed/stats.json.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from support_agent.config import OUT_OF_SCOPE_QUEUES, PRIORITIES, QUEUES, TYPES
from support_agent.embeddings import normalize_text

HF_DATASET = "Tobi-Bueck/customer-support-tickets"
ALL_QUEUES = QUEUES + OUT_OF_SCOPE_QUEUES
PRIORITY_MAP = {"low": "low", "medium": "medium", "high": "high", "critical": "high", "urgent": "high",
                "very low": "low", "very_low": "low", "very high": "high", "very_high": "high"}
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 11)]


def load_raw(csv: Path | None = None) -> pd.DataFrame:
    """From a local CSV/JSONL export if given, else from the Hugging Face Hub (needs the `data` extra)."""
    if csv is not None:
        csv = Path(csv)
        return pd.read_json(csv, lines=True) if csv.suffix == ".jsonl" else pd.read_csv(csv)
    from datasets import load_dataset
    return load_dataset(HF_DATASET, split="train").to_pandas()


def _clean(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\\n", "\n")
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize(df: pd.DataFrame, language: str | None = "en") -> tuple[pd.DataFrame, dict]:
    """Keep rows with known labels and usable text; return the clean frame and a report of what was dropped."""
    report: dict = {"rows_in": len(df)}
    df = df.copy()
    for col in ("subject", "body", "answer"):
        df[col] = df[col].map(_clean) if col in df else ""
    df["queue"] = df["queue"].map(_clean)
    df["type"] = df["type"].map(lambda t: _clean(t).title())
    df["priority"] = df["priority"].map(lambda p: PRIORITY_MAP.get(_clean(p).lower(), "?"))

    if language and "language" in df:
        report["dropped_language"] = int((df["language"] != language).sum())
        df = df[df["language"] == language]
    report["unknown_queues"] = dict(Counter(df.loc[~df["queue"].isin(ALL_QUEUES), "queue"]).most_common(20))
    report["unknown_priorities"] = int((df["priority"] == "?").sum())
    report["unknown_types"] = dict(Counter(df.loc[~df["type"].isin(TYPES), "type"]).most_common(10))
    df = df[df["queue"].isin(ALL_QUEUES) & df["priority"].isin(PRIORITIES) & df["type"].isin(TYPES)]
    short = (df["body"].str.len() < 20) | (df["answer"].str.len() < 20)
    report["dropped_short"] = int(short.sum())
    df = df[~short]

    # Exact and near-exact duplicates (same normalised subject+body, or same normalised first 200 body chars).
    norm = (df["subject"] + " " + df["body"]).map(normalize_text)
    head = df["body"].map(lambda b: normalize_text(b)[:200])
    dup = norm.duplicated() | head.duplicated()
    report["dropped_duplicates"] = int(dup.sum())
    df = df[~dup]

    tags = [c for c in TAG_COLUMNS if c in df]
    df["tags"] = df[tags].apply(lambda r: [t for t in r if isinstance(t, str) and t], axis=1) if tags else [[]] * len(df)
    df["id"] = (df["subject"] + df["body"]).map(lambda s: "T" + hashlib.sha1(s.encode()).hexdigest()[:10])
    keep = ["id", "subject", "body", "answer", "queue", "priority", "type", "language", "tags"]
    df = df[[c for c in keep if c in df]].reset_index(drop=True)
    report["rows_out"] = len(df)
    report["queues"] = dict(Counter(df["queue"]).most_common())
    report["priorities"] = dict(Counter(df["priority"]).most_common())
    report["types"] = dict(Counter(df["type"]).most_common())
    return df, report


def stratified_split(df: pd.DataFrame, val: float = 0.1, test: float = 0.1, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split each (queue, priority) group with the same ratios, so rare labels appear in every split."""
    rng = random.Random(seed)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, (q, p) in enumerate(zip(df["queue"], df["priority"])):
        groups[(q, p)].append(i)
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for idx in groups.values():
        rng.shuffle(idx)
        n_test = round(len(idx) * test)
        n_val = round(len(idx) * val)
        if len(idx) >= 3:          # tiny groups: at least one example in test when possible
            n_test = max(n_test, 1)
        parts["test"] += idx[:n_test]
        parts["val"] += idx[n_test:n_test + n_val]
        parts["train"] += idx[n_test + n_val:]
    return {k: df.iloc[sorted(v)].reset_index(drop=True) for k, v in parts.items()}


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
