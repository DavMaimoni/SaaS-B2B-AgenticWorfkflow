"""Calibrate the three thresholds the system depends on, from data rather than guesses.

    uv run python eval/calibrate.py judge  --run eval/runs/system --labels eval/judge_labels.jsonl
    uv run python eval/calibrate.py cache  --triples eval/triples.jsonl --test data/processed/test.jsonl
    uv run python eval/calibrate.py router --run eval/runs/lora_samples

judge   agreement and Cohen's kappa between the LLM judge and your own labels (40 items, 20 each).
        Label file: one JSON per line {"id": "E012", "verdict": "yes|partially|no", "labeler": "A"}.
cache   cosine similarity of paraphrase pairs (must hit) vs nearest different tickets and hand-written traps
        (must not hit); recommends the lowest threshold with zero false hits.
router  accuracy of the small model's triage by confidence level (run the graph variant with the LoRA as small
        model); recommends ROUTE_CONFIDENCE.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from eval.metrics import cohen_kappa  # noqa: E402
from support_agent.data import read_jsonl  # noqa: E402

CACHE_TRAPS = [  # near-identical wording, different request: a cache hit here would send a wrong answer
    ("How do I cancel my subscription?", "How do I pause my subscription?"),
    ("How do I add seats to our plan?", "How do I remove seats from our plan?"),
    ("How do I export a project to CSV?", "How do I import a CSV into a project?"),
    ("Can I upgrade from Pro to Enterprise?", "Can I downgrade from Enterprise to Pro?"),
    ("How do I enable two-factor authentication?", "How do I disable two-factor authentication?"),
    ("Where can I download last month's invoice?", "Where can I download next month's invoice?"),
    ("How do I invite a member to the workspace?", "How do I remove a member from the workspace?"),
    ("How do I create an API token?", "How do I revoke an API token?"),
    ("Can we switch to annual billing?", "Can we switch to monthly billing?"),
    ("How do I turn on email notifications?", "How do I turn off email notifications?"),
]


def calibrate_judge(run: Path, labels_path: Path) -> dict:
    results = {r["id"]: r for r in read_jsonl(run / "results.jsonl" if run.is_dir() else run)}
    labels = read_jsonl(labels_path)
    pairs = [(lab["verdict"], results[lab["id"]]["verdict"]) for lab in labels
             if lab["id"] in results and results[lab["id"]].get("verdict")]
    human, judge = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"n": len(pairs), "agreement": round(sum(h == j for h, j in pairs) / max(1, len(pairs)), 3),
            "cohen_kappa": round(cohen_kappa(human, judge), 3),
            "confusion_human_to_judge": {h: dict(Counter(j for hh, j in pairs if hh == h)) for h in sorted(set(human))},
            "advice": "kappa >= 0.6 is usually considered substantial; below that, rewrite the judge prompt or "
                      "change the judge model, then recheck."}


def calibrate_cache(triples_path: Path, test_path: Path | None, max_negatives: int = 200) -> dict:
    from support_agent.embeddings import get_embedder
    emb = get_embedder()
    pos = []
    for r in read_jsonl(triples_path):
        if r.get("warmup"):
            a, b = emb.encode([f"{r['warmup']['subject']}\n\n{r['warmup']['body']}", f"{r['subject']}\n\n{r['body']}"])
            pos.append(float(a @ b))
    neg = []
    for a_text, b_text in CACHE_TRAPS:
        a, b = emb.encode([a_text, b_text])
        neg.append(float(a @ b))
    if test_path and Path(test_path).exists():
        tickets = read_jsonl(test_path)[:max_negatives]
        vecs = emb.encode([f"{t['subject']}\n\n{t['body']}" for t in tickets])
        sims = vecs @ vecs.T
        for i in range(len(tickets)):
            sims[i, i] = -1
            neg.append(float(sims[i].max()))      # nearest OTHER ticket: a different request by construction
    table = []
    for th in [x / 100 for x in range(70, 100)]:
        table.append({"threshold": th, "paraphrase_hit_rate": round(sum(p >= th for p in pos) / max(1, len(pos)), 3),
                      "false_hits": sum(n >= th for n in neg)})
    safe = [row for row in table if row["false_hits"] == 0]
    best = max(safe, key=lambda row: (row["paraphrase_hit_rate"], -row["threshold"])) if safe else None
    return {"n_positive_pairs": len(pos), "n_negative_pairs": len(neg),
            "max_negative_similarity": round(max(neg), 4) if neg else None,
            "recommended_threshold": best["threshold"] if best else None,
            "hit_rate_at_recommended": best["paraphrase_hit_rate"] if best else None,
            "table": table}


def calibrate_router(run: Path, target: float = 0.9) -> dict:
    rows = [r for r in read_jsonl(run / "results.jsonl" if run.is_dir() else run) if r.get("confidence") is not None]
    by_conf: dict[float, list[bool]] = defaultdict(list)
    for r in rows:
        ok = all((r.get("pred") or {}).get(k) == r["expected"][k] for k in ("queue", "priority"))
        by_conf[round(r["confidence"], 2)].append(ok)
    table = [{"confidence": c, "n": len(v), "queue_and_priority_acc": round(sum(v) / len(v), 3)}
             for c, v in sorted(by_conf.items())]
    ok_levels = [t["confidence"] for t in table if t["queue_and_priority_acc"] >= target]
    return {"target_accuracy": target, "table": table,
            "recommended_ROUTE_CONFIDENCE": min(ok_levels) if ok_levels else None,
            "note": "Tickets below the recommended confidence go to the large model."}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--run", type=Path, required=True)
    j.add_argument("--labels", type=Path, default=Path(__file__).parent / "judge_labels.jsonl")
    c = sub.add_parser("cache")
    c.add_argument("--triples", type=Path, default=Path(__file__).parent / "triples.jsonl")
    c.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "test.jsonl")
    r = sub.add_parser("router")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--target", type=float, default=0.9)
    args = ap.parse_args()
    if args.cmd == "judge":
        out = calibrate_judge(args.run, args.labels)
    elif args.cmd == "cache":
        out = calibrate_cache(args.triples, args.test)
        out["table"] = [row for row in out["table"] if row["threshold"] * 100 % 2 == 0]
    else:
        out = calibrate_router(args.run, args.target)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
