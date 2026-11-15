"""Build the LoRA training files from the train / val splits.

    uv run python finetune/build_sft.py                      # -> finetune/data/{train,val}.jsonl
    uv run python finetune/build_sft.py --max-per-queue 3000 # cap the biggest queues

Each line is a chat example in the exact format the graph uses at run time (`prompts.sft_messages`):
system = SFT_SYSTEM, user = the ticket, assistant = {"queue", "priority", "type", "answer"} as JSON.
Dataset placeholders such as <tel_num> or <name> are removed from the answers (the validator would reject a reply
containing them); an answer left too short after cleaning is dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from support_agent.config import DATA_DIR  # noqa: E402
from support_agent.data import read_jsonl  # noqa: E402
from support_agent.prompts import sft_messages  # noqa: E402

PLACEHOLDER = re.compile(r"<[^<>\n]{1,40}>")


def clean_answer(answer: str) -> str | None:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    kept = [s for s in sentences if not PLACEHOLDER.search(s)]
    text = " ".join(kept).strip()
    return text if len(text) >= 40 else None


def to_example(t: dict) -> dict | None:
    answer = clean_answer(t["answer"])
    if not answer:
        return None
    target = {"queue": t["queue"], "priority": t["priority"], "type": t["type"], "answer": answer}
    messages = sft_messages(t["subject"], t["body"]) + [
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}]
    return {"id": t["id"], "messages": messages}


def build(rows: list[dict], max_per_queue: int | None, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_queue: dict[str, list[dict]] = defaultdict(list)
    for t in rows:
        by_queue[t["queue"]].append(t)
    out, dropped = [], 0
    for queue, items in by_queue.items():
        rng.shuffle(items)
        for t in items[: max_per_queue or len(items)]:
            ex = to_example(t)
            if ex:
                out.append(ex)
            else:
                dropped += 1
    rng.shuffle(out)
    queues = Counter(json.loads(e["messages"][-1]["content"])["queue"] for e in out)
    return out, {"examples": len(out), "dropped_placeholder_or_short": dropped, "queues": dict(queues)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=DATA_DIR / "processed")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--max-per-queue", type=int)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in ("train", "val"):
        examples, stats = build(read_jsonl(args.processed / f"{split}.jsonl"), args.max_per_queue, args.seed)
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        report[split] = stats
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
