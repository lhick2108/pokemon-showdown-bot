"""Report on a ladder A/B log written by `scripts/ladder.py --arm ...`.

    python scripts/ab_report.py [data/ladder/ab.jsonl]

Per arm: games, wins, win rate with Wilson 95% CI, mean opponent rating; the difference
between arms with a 95% CI; a logistic regression win ~ arm + (opp - our rating)/400
(arm effect in win-probability points at equal ratings); games needed per arm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from showdown_bot import ab_stats

REPO = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("log", type=Path, nargs="?", default=REPO / "data" / "ladder" / "ab.jsonl")
    args = p.parse_args(argv)
    if not args.log.is_file():
        print(f"error: {args.log} not found", file=sys.stderr)
        return 2
    print(ab_stats.format_report(ab_stats.read_records(args.log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
