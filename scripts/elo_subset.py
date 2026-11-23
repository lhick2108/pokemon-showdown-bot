#!/usr/bin/env python
"""Hard-link the human replay records at or above an Elo floor into a new tree.

    python scripts/elo_subset.py --src data/human/pc1 --out data/human/pc1-e1600 --min-elo 1600

Reads only each record's header line (``elo`` is a bucket such as ``1600-1799`` or ``1800+``, or a
plain rating; the leading digits are the floor) and links ``<split>/<shard>/<file>`` under --out, so
the trainers (train.py, prior_train.py --human) can take an Elo-filtered set without copying data.
``policy/`` is skipped. Existing links are kept, so the script resumes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def elo_floor(value) -> int:
    m = re.match(r"\s*(\d+)", str(value or ""))
    return int(m.group(1)) if m else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-elo", type=int, required=True)
    p.add_argument("--splits", default="train,test")
    p.add_argument("--since-year", type=int, default=0,
                   help="also require the header month_year (e.g. March2025) to be this year or later")
    args = p.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    kept, seen = Counter(), Counter()
    for split in args.splits.split(","):
        for path in (src / split).rglob("*.jsonl.gz"):
            seen[split] += 1
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    head = json.loads(handle.readline())
            except (OSError, EOFError, json.JSONDecodeError):
                continue
            if elo_floor(head.get("elo")) < args.min_elo:
                continue
            if args.since_year and int(str(head.get("month_year") or "0")[-4:] or 0) < args.since_year:
                continue
            dest = out / path.relative_to(src)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, dest)
            kept[split] += 1
            if seen[split] % 50000 == 0:
                print(f"{split}: {seen[split]} scanned, {kept[split]} kept", flush=True)
    (out / "subset.json").write_text(json.dumps(
        {"src": str(src), "min_elo": args.min_elo, "since_year": args.since_year, "seen": seen, "kept": kept}, indent=1))
    print("done", dict(seen), dict(kept), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
