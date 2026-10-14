"""Report on a ladder A/B log written by `scripts/ladder.py --arm ...`.

    python scripts/ab_report.py [data/ladder/ab.jsonl] [--logs DIR ...] [--proxy-opp-rating]

Missing opponent_rating / rating_before are backfilled from the |player| lines of the
per-game logs (default: */logs dirs beside the ab.jsonl).

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
    p.add_argument("--logs", type=Path, action="append", default=None, metavar="DIR",
                   help="protocol-log dir (<battle_tag>.log); repeatable. Default: every "
                        "*/logs dir beside the ab.jsonl")
    p.add_argument("--proxy-opp-rating", action="store_true",
                   help="if an opponent rating is still unknown, use our rating_before")
    args = p.parse_args(argv)
    if not args.log.is_file():
        print(f"error: {args.log} not found", file=sys.stderr)
        return 2
    records = ab_stats.read_records(args.log)
    dirs = args.logs if args.logs else ab_stats.default_log_dirs(args.log)
    records = ab_stats.backfill_ratings(records, dirs, proxy_opp_rating=args.proxy_opp_rating)
    print(ab_stats.format_report(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
