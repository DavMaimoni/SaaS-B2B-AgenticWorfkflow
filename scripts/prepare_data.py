"""Step 1 of the pipeline: load the Tobi-Bueck tickets, clean the labels, dedupe, split train/val/test.

    uv run python scripts/prepare_data.py                     # from the Hugging Face Hub (needs: uv sync --extra data)
    uv run python scripts/prepare_data.py --csv tickets.csv   # from a local export
    uv run python scripts/prepare_data.py --sample            # the 48 hand-written demo tickets (offline)

Writes data/processed/{train,val,test}.jsonl and data/processed/stats.json.
"""

import argparse
import json
from pathlib import Path

from support_agent.config import DATA_DIR
from support_agent.data import load_raw, normalize, stratified_split, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, help="local CSV or JSONL export of the dataset")
    ap.add_argument("--sample", action="store_true", help="use data/sample_tickets.jsonl (offline demo)")
    ap.add_argument("--language", default="en", help="keep one language ('' = all)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "processed")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = DATA_DIR / "sample_tickets.jsonl" if args.sample else args.csv
    df, report = normalize(load_raw(source), language=args.language or None)
    splits = stratified_split(df, seed=args.seed)
    for name, part in splits.items():
        write_jsonl(part, args.out / f"{name}.jsonl")
    report["splits"] = {k: len(v) for k, v in splits.items()}
    report["source"] = str(source or "huggingface:Tobi-Bueck/customer-support-tickets")
    (args.out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
