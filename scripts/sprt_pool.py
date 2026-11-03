"""Pool paired league games from several runs of the same matchup into one SPRT.

    python scripts/sprt_pool.py data/league/sprt-stack16:stack16:best16 data/league/sprt-v5-gen7-16w:v5:gen7
    python scripts/sprt_pool.py ... --sprt 0,21 --max-games 800 --watch 120 --stop-file data/league/stop-A

Each argument is RUN_DIR:CANDIDATE:BASELINE (the labels used in that run). Pairs are the two
games sharing a team_pair within one run; unfinished pairs are left out. Prints the pooled
pentanomial counts, LLR, Elo and verdict (H1 / H0 / max / continue). With --watch it re-reads
every N seconds until a verdict, then writes --stop-file (if given) and exits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from showdown_bot import sprt
from showdown_bot.league import sprt_run_counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", metavar="RUN_DIR:CANDIDATE:BASELINE")
    p.add_argument("--sprt", default="0,21")
    p.add_argument("--max-games", type=int, default=1200, help="pooled game cap (verdict 'max')")
    p.add_argument("--watch", type=float, default=0, help="poll every N s until a verdict")
    p.add_argument("--stop-file", type=Path, default=None, help="touched when a verdict is reached")
    args = p.parse_args()
    cfg = sprt.parse_sprt(args.sprt, max_games=args.max_games)
    while True:
        total = [0] * 5
        for spec in args.runs:
            run, cand, base = spec.rsplit(":", 2)
            c = sprt_run_counts(Path(run), cand, base)
            total = [a + b for a, b in zip(total, c)]
            print(f"  {run}: pairs {c}", flush=True)
        verdict = sprt.decide(total, cfg)
        print(f"{time.strftime('%H:%M:%S')} pooled: {sprt.summary_text(total, cfg)} -> {verdict or 'continue'}",
              flush=True)
        if verdict or not args.watch:
            if verdict and args.stop_file:
                args.stop_file.write_text(f"{verdict} {sprt.summary_text(total, cfg)}\n", encoding="utf-8")
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
