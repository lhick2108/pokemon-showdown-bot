"""Hypothetical-position augmentation: fork recorded truth states, edit them, play them out.

    python -m uv run python scripts/fork_augment.py --source "data/selfplay/gen*" \
        --out data/forks/run1 --forks 2000 --workers 8 --time-ms 50
    python -m uv run python -m showdown_bot.train --data data/forks/run1 --data data/selfplay/gen4 ...

Engine-only (no Showdown server): each fork takes an exact "truth" State from the self-play
records (at most ``--per-game`` per battle, so rows stay decorrelated), perturbs it (HP, status,
stat stages, hazards, screens, weather, terrain) and plays it to the end with one
perfect-information MCTS per turn serving both sides, moves drawn with the bot's mixed rule.
Output is the normal self-play schema (``docs/selfplay-data.md``) with header
``source: "fork"`` and the edits under ``fork``. Per-fork seeds depend only on ``--seed`` and
the fork index; the edits and the source choice are reproducible, the playout is as
reproducible as the (time-limited, unseeded) engine search.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

from showdown_bot import fork_augment as fa

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", default=[str(REPO_ROOT / "data" / "selfplay" / "gen*")],
                   help="self-play run dirs / files / globs to fork from (default data/selfplay/gen*)")
    p.add_argument("--out", type=Path, required=True, help="output directory (e.g. data/forks/<run>)")
    p.add_argument("--forks", type=int, default=1000, help="number of forks to attempt")
    p.add_argument("--per-game", type=int, default=2, help="max forks per source battle")
    p.add_argument("--min-turn", type=int, default=1, help="earliest source turn to fork from")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-ms", type=int, default=50, help="search budget per move (one search/turn)")
    p.add_argument("--iterations", type=int, default=0,
                   help="fixed MCTS iterations per move instead of --time-ms (0 = time-limited)")
    p.add_argument("--evaluator", type=Path, default=None, help="learned .pev leaf evaluator")
    p.add_argument("--leaf-scale", type=float, default=1.0, help="learned leaf scale k")
    p.add_argument("--max-turns", type=int, default=150)
    p.add_argument("--on-cap", choices=("drop", "tie"), default="drop",
                   help="what to do with a game that hits --max-turns")
    for f in dataclasses.fields(fa.EditConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=float, default=f.default,
                       help=f"edit config (default {f.default})")
    return p


def resolve_sources(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        out.extend(hits if hits else [pattern])
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    edit = fa.EditConfig(**{
        f.name: getattr(args, f.name) for f in dataclasses.fields(fa.EditConfig)
    })
    play = fa.PlayConfig(
        time_ms=args.time_ms, iterations=args.iterations,
        evaluator=str(args.evaluator.resolve()) if args.evaluator else None,
        leaf_scale=args.leaf_scale, max_turns=args.max_turns, on_cap=args.on_cap,
    )
    rng = random.Random(args.seed)
    print("sampling source decisions...", file=sys.stderr)
    sources = fa.sample_sources(
        resolve_sources(args.source), args.forks, args.per_game, rng, min_turn=args.min_turn
    )
    if not sources:
        print("no usable source decisions (need records with truth states)", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = fa.make_jobs(sources, args.seed, args.out, edit, play, run_id=f"s{args.seed}")
    print(f"{len(jobs)} forks from {len({s['battle_tag'] for s in sources})} battles, "
          f"{args.workers} workers, {args.time_ms} ms/move", file=sys.stderr)
    tally: Counter = Counter()
    steps = 0
    started = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(fa.run_fork, jobs), 1):
            if res.get("ok"):
                tally[res["result"]] += 1
                steps += res["steps"]
            elif res.get("dropped"):
                tally["dropped"] += 1
            else:
                tally["error"] += 1
                print(f"\nfork {res['index']}: {res.get('error')}", file=sys.stderr)
            elapsed = time.time() - started
            written = sum(tally[r] for r in ("win", "loss", "tie"))
            print(
                f"\r[{n}/{len(jobs)}] written {written} (win {tally['win']} loss {tally['loss']} "
                f"tie {tally['tie']}) dropped {tally['dropped']} errors {tally['error']} | "
                f"{60 * n / max(elapsed, 1e-9):.1f} forks/min, avg {steps / max(written, 1):.1f} turns",
                end="", file=sys.stderr, flush=True,
            )
    print(file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
