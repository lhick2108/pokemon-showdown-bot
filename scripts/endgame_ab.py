"""Engine-only A/B of endgame mode: expectiminimax (A) vs plain MCTS (B) on endgame positions.

    python scripts/endgame_ab.py --source "data/selfplay/gen6" --positions 200 --workers 6 \
        --a-spec "a=search:endgame=safest:eg_ms=1500" --b-spec "b=search:time_ms=1500" \
        --max-alive 5 --time-ms 1500 --seed 1

Positions: the first full-information truth state of each self-play game (read-only) with
total alive Pokemon <= --max-alive (both actives alive, nobody forced to switch). Each
position is played twice with the seats swapped, A against B, both choosing every move by
search on the full state (no hidden information: this isolates endgame search quality), the
turn outcomes sampled with the engine's instruction percentages, up to --max-turns turns
(a capped game scores 0.5). A's score per position is the mean of its two games; the report
gives that mean with a normal and a bootstrap 95% CI over positions.

Bot specs reuse the league syntax (label=search[-net]:key=value...): endgame=off|safest|nash,
eg_alive, eg_side, eg_ms, eg_worlds (unused here: one true world), time_ms (MCTS ms per move),
evaluator, scale, mode (mixed|argmax). --time-ms is the default for B's time_ms and for A's
time_ms and eg_ms. Note the endgame search uses the stock evaluation; evaluator/scale only
affect the MCTS fallback and bot B.
Run with --workers 1 for a quick in-process smoke test.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", nargs="+", default=[str(REPO_ROOT / "data" / "selfplay" / "gen*")],
                   help="self-play run dirs / files / globs (read-only)")
    p.add_argument("--positions", type=int, default=100, help="number of start positions")
    p.add_argument("--max-alive", type=int, default=5,
                   help="position filter: total alive on both sides <= N (default 5)")
    p.add_argument("--max-side", type=int, default=0, help="also each side <= N alive (0 = off)")
    p.add_argument("--workers", type=int, default=1, help="game processes (1 = in-process)")
    p.add_argument("--a-spec", default="a=search:endgame=safest",
                   help="bot A spec (default: endgame safest)")
    p.add_argument("--b-spec", default="b=search", help="bot B spec (default: plain MCTS)")
    p.add_argument("--time-ms", type=int, default=1000,
                   help="default ms per move for MCTS and for A's endgame search (default 1000)")
    p.add_argument("--max-turns", type=int, default=60, help="turn limit per game (capped = tie)")
    p.add_argument("--seed", type=int, default=0)
    return p


def make_config(text: str, time_ms: int):
    """SearchConfig of a bot spec (league syntax), with --time-ms as the time defaults."""
    from showdown_bot import league
    from showdown_bot.players.search import SearchConfig

    spec = league.parse_spec(text)
    if spec.type not in league.SEARCH_BOTS:
        raise SystemExit(f"{text!r}: only search / search-net bots are supported")
    fields = league.search_config_fields(spec)
    fields.setdefault("search_time_ms", time_ms)
    if "endgame_time_ms" not in fields:
        fields["endgame_time_ms"] = time_ms
    cfg = dataclasses.replace(SearchConfig(), **fields)
    if cfg.mode == "nash":
        raise SystemExit(f"{text!r}: mode=nash is not supported here (mixed|argmax)")
    return cfg


def resolve(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        out.extend(hits if hits else [pattern])
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from showdown_bot import endgame_eval as ev

    a_cfg = make_config(args.a_spec, args.time_ms)
    b_cfg = make_config(args.b_spec, args.time_ms)
    print(f"A: {args.a_spec} -> endgame={a_cfg.endgame} eg_alive={a_cfg.endgame_max_alive} "
          f"eg_ms={a_cfg.endgame_time_ms} mcts_ms={a_cfg.search_time_ms}", file=sys.stderr)
    print(f"B: {args.b_spec} -> endgame={b_cfg.endgame} mcts_ms={b_cfg.search_time_ms}",
          file=sys.stderr)
    rng = random.Random(args.seed)
    positions = ev.extract_positions(
        resolve(args.source), args.positions, args.max_alive, rng, args.max_side
    )
    if not positions:
        print("no endgame positions found (need records with truth states)", file=sys.stderr)
        return 1
    print(f"{len(positions)} positions (alive counts: "
          f"{sorted({tuple(p['alive']) for p in positions})}), {args.workers} workers",
          file=sys.stderr)
    tasks = [
        {"index": i, "a_side": side, "state": pos["state"], "a_cfg": a_cfg, "b_cfg": b_cfg,
         "max_turns": args.max_turns, "seed": args.seed}
        for i, pos in enumerate(positions) for side in (1, 2)
    ]
    results: list[dict] = []
    started = time.time()

    def progress(n: int, res: dict) -> None:
        elapsed = time.time() - started
        print(f"\r[{n}/{len(tasks)}] {60 * n / max(elapsed, 1e-9):.1f} games/min"
              + (f"  ERROR {res['error']}" if not res.get("ok") else ""),
              end="", file=sys.stderr, flush=True)

    if args.workers <= 1:
        for n, task in enumerate(tasks, 1):
            res = ev.run_task(task)
            results.append(res)
            progress(n, res)
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for n, res in enumerate(pool.imap_unordered(ev.run_task, tasks), 1):
                results.append(res)
                progress(n, res)
    print(file=sys.stderr)

    ok = [r for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    pair = ev.paired_scores(results)
    summary = ev.summarize(list(pair.values()), seed=args.seed)
    wins = sum(r["score"] == 1.0 for r in ok)
    ties = sum(r["score"] == 0.5 for r in ok)
    losses = sum(r["score"] == 0.0 for r in ok)
    by_seat = {s: [r["score"] for r in ok if r["a_side"] == s] for s in (1, 2)}
    depths = [d for r in ok for d in r["a_depths"]]
    print(f"games: {len(ok)} ok, {len(errors)} errors, "
          f"{sum(r['capped'] for r in ok)} hit the turn cap (scored 0.5)")
    print(f"A  W/T/L: {wins}/{ties}/{losses}")
    for side, scores in by_seat.items():
        if scores:
            print(f"A as side {side}: mean {sum(scores) / len(scores):.3f} over {len(scores)} games")
    print(f"A paired score over {summary['n']} positions: {summary['mean']:.3f} "
          f"(se {summary['se']:.3f}); normal 95% CI [{summary['lo']:.3f}, {summary['hi']:.3f}]; "
          f"bootstrap 95% CI [{summary['boot_lo']:.3f}, {summary['boot_hi']:.3f}]")
    if depths:
        print(f"A expectiminimax decisions: {len(depths)}, mean depth {sum(depths) / len(depths):.1f}, "
              f"fallbacks to MCTS: {sum(r['a_fallbacks'] for r in ok)}")
    print(f"mean game length {sum(r['steps'] for r in ok) / max(len(ok), 1):.1f} turns; "
          f"wall time {time.time() - started:.0f} s")
    for r in errors[:5]:
        print(f"error (position {r['index']}, A side {r['a_side']}): {r.get('error')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
