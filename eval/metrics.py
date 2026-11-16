"""Scoring functions, no model calls: label accuracy and macro-F1, judge score, bootstrap intervals."""

from __future__ import annotations

import random
from collections import Counter

VERDICT_SCORE = {"yes": 1.0, "partially": 0.5, "no": 0.0}


def accuracy(gold: list, pred: list) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")


def macro_f1(gold, pred):
    """Calculates macro F1 score while safely filtering out None values."""
    valid_pairs = [(g, p) for g, p in zip(gold, pred) if g is not None and p is not None]
    if not valid_pairs:
        return 0.0
    
    g_filtered, p_filtered = zip(*valid_pairs)
    
    labels = sorted(set(g_filtered) | set(p_filtered))
    if not labels:
        return 0.0

    f1_scores = []
    for label in labels:
        tp = sum(1 for g, p in zip(g_filtered, p_filtered) if g == label and p == label)
        fp = sum(1 for g, p in zip(g_filtered, p_filtered) if g != label and p == label)
        fn = sum(1 for g, p in zip(g_filtered, p_filtered) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

def confusion(gold: list, pred: list) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = {}
    for g, p in zip(gold, pred):
        out.setdefault(g, Counter())[p] += 1
    return {g: dict(c) for g, c in out.items()}


def judge_score(verdicts: list[str]) -> float:
    scores = [VERDICT_SCORE.get(v, 0.0) for v in verdicts]
    return sum(scores) / len(scores) if scores else float("nan")


def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Mean of (b - a) over the same items, with a 95 % interval. a and b must be aligned item by item."""
    assert len(a) == len(b) and a, "paired bootstrap needs two aligned, non-empty lists"
    diffs = [y - x for x, y in zip(a, b)]
    low, high = bootstrap_ci(diffs, n=n, seed=seed)
    return {"mean_diff": sum(diffs) / len(diffs), "ci95": (low, high), "n": len(diffs)}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[label] * cb[label] for label in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
