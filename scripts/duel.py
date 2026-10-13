"""Duel: search with a learned evaluator (search-net) vs stock search, at equal compute.

    python -m uv run python scripts/duel.py --evaluator models/x.pev --games 100 --pairs 4

Both sides get the same SearchConfig (time per world, worlds, threads, mode)
except for ``evaluator`` and, if asked, the engine selection mode
(--net-selection / --stock-selection, gamma --selection-gamma) and the final
move rule (--net-mode / --stock-mode: mixed, argmax or nash); with
--stock-uses-evaluator both sides use the same evaluator, so e.g.
``--net-selection rm --stock-selection ucb1 --stock-uses-evaluator`` isolates
the selection rule; each side's config is passed to its player as a
constructor argument, so the process-wide SHOWDOWN_SEARCH_* env vars are not
used. The two sides of every pair share one MCTS process pool, so CPU
contention is symmetric. Needs the local server on 127.0.0.1:8000
(scripts/start_server.ps1). Prints the net's win rate with a Wilson 95% CI and
each side's measured MCTS iterations per second of search budget.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import secrets
import sys
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench
from showdown_bot.players.search import (
    MODES, SELECTIONS, SearchConfig, SearchNetPlayer, SearchPlayer, get_pool,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--evaluator", type=Path, default=None,
                   help="learned .pev file; omit to duel two stock searches (no evaluator on "
                        "either side), e.g. --net-mode nash --stock-mode mixed")
    p.add_argument("--leaf-scale", type=float, default=1.0,
                   help="learned leaf scale k: score = sigmoid(k * dz); net side only")
    p.add_argument("--net-selection", choices=SELECTIONS, default="ucb1",
                   help="engine tree policy of the net side (default ucb1)")
    p.add_argument("--stock-selection", choices=SELECTIONS, default="ucb1",
                   help="engine tree policy of the stock side (default ucb1)")
    p.add_argument("--net-mode", choices=MODES, default="mixed",
                   help="final move rule of the net side (default mixed; nash needs --threads 1)")
    p.add_argument("--stock-mode", choices=MODES, default="mixed",
                   help="final move rule of the stock side (default mixed)")
    p.add_argument("--net-inference", choices=("on", "off"), default="on",
                   help="589.28 opponent set inference on the net side (default on)")
    p.add_argument("--stock-inference", choices=("on", "off"), default="on",
                   help="589.28 opponent set inference on the stock side (default on)")
    p.add_argument("--net-damage-inference", choices=("on", "off"), default="on",
                   help="damage-roll part of the net side's inference (default on)")
    p.add_argument("--stock-damage-inference", choices=("on", "off"), default="on",
                   help="damage-roll part of the stock side's inference (default on)")
    p.add_argument("--net-eval-lowhp", type=float, default=0.0,
                   help="589.32: low-HP evaluation knee for the net side (0 = stock; needs the 589.32 engine wheel)")
    p.add_argument("--stock-eval-lowhp", type=float, default=0.0,
                   help="589.32: low-HP evaluation knee for the stock side (0 = stock)")
    p.add_argument("--net-eval-boost-scale", type=float, default=1.0,
                   help="589.32: multiplier on stat-boost eval terms for the net side (1 = stock)")
    p.add_argument("--stock-eval-boost-scale", type=float, default=1.0,
                   help="589.32: multiplier on stat-boost eval terms for the stock side (1 = stock)")
    p.add_argument("--net-tendency", choices=("on", "off"), default="off",
                   help="589.40 opponent tendency model on the net side (default off)")
    p.add_argument("--stock-tendency", choices=("on", "off"), default="off",
                   help="589.40 opponent tendency model on the stock side (default off)")
    p.add_argument("--tendency-kappa", type=float, default=None, help="prior pseudo-observations")
    p.add_argument("--tendency-delta", type=float, default=None, help="max loss vs Nash")
    for side in ("net", "stock"):
        p.add_argument(f"--{side}-escalate", action="store_true",
                       help=f"589.47: {side} side re-searches longer when the top two options are flat")
        p.add_argument(f"--{side}-escalate-gap", type=float, default=0.02,
                       help="flat when the top two mean scores differ by less than this")
        p.add_argument(f"--{side}-escalate-factor", type=float, default=4.0,
                       help="escalated per-world budget as a multiple of --search-time-ms")
        p.add_argument(f"--{side}-escalate-steps", type=int, default=1)
        p.add_argument(f"--{side}-escalate-cap-ms", type=int, default=3000,
                       help="hard wall-clock cap of one decision's search")
        p.add_argument(f"--{side}-prune-noop", action="store_true",
                       help=f"589.47: {side} side drops moves the engine proves have no effect")
        p.add_argument(f"--{side}-prune-noop-loose", action="store_true",
                       help="with --prune-noop: ignore opponent switches when proving a no-op "
                            "(also prunes status into an already-statused target)")
        p.add_argument(f"--{side}-tera-margin", type=float, default=0.0,
                       help="589.47: X-tera needs a mean this much above X, else it folds into X (0 = off)")
    p.add_argument("--selection-gamma", type=float, default=0.1,
                   help="exploration rate in (0, 1] for exp3/rm sides")
    p.add_argument("--stock-uses-evaluator", action="store_true",
                   help="give the second side the same evaluator/leaf scale (isolates selection)")
    p.add_argument("--opponent-evaluator", type=Path, default=None,
                   help="learned .pev for the second side (e.g. the v1 net when --evaluator is a "
                        "v2 net); overrides --stock-uses-evaluator")
    p.add_argument("--opponent-leaf-scale", type=float, default=1.0,
                   help="leaf scale of --opponent-evaluator (default 1.0)")
    p.add_argument("--games", type=bench._positive_int, default=100)
    p.add_argument("--pairs", type=bench._positive_int, default=4, help="parallel pairs")
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--search-time-ms", type=bench._positive_int, default=100,
                   help="per-world search time, identical for both sides")
    p.add_argument("--parallelism", type=bench._positive_int, default=1)
    p.add_argument("--worlds", type=bench._positive_int, default=None)
    p.add_argument("--threads", type=bench._positive_int, default=1)
    p.add_argument("--workers", type=bench._positive_int, default=os.cpu_count() or 4,
                   help="shared MCTS process pool size")
    p.add_argument("--connect-timeout", type=bench._positive_seconds, default=30)
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=900)
    return p


def _tendency_fields(args, flag: str) -> dict:
    fields = {"tendency": flag == "on"}
    if args.tendency_kappa is not None:
        fields["tendency_kappa"] = args.tendency_kappa
    if args.tendency_delta is not None:
        fields["tendency_delta"] = args.tendency_delta
    return fields


def _quality(args, side: str) -> dict:
    """SearchConfig fields of the 589.47 decision-quality options for one side."""
    get = lambda name: getattr(args, f"{side}_{name}")  # noqa: E731
    return {
        "escalate": get("escalate"), "escalate_gap": get("escalate_gap"),
        "escalate_factor": get("escalate_factor"), "escalate_steps": get("escalate_steps"),
        "escalate_cap_ms": get("escalate_cap_ms"), "prune_noop": get("prune_noop"),
        "noop_vs_switches": not get("prune_noop_loose"),
        "tera_margin": get("tera_margin"),
    }


async def _duel_pair(args, share: int, pool):
    options = {
        "battle_format": args.format,
        "max_concurrent_battles": 1,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR,
        "open_timeout": args.connect_timeout,
    }
    base = SearchConfig(
        search_time_ms=args.search_time_ms, parallelism=args.parallelism,
        worlds=args.worlds, threads=args.threads, selection_gamma=args.selection_gamma,
    )
    evaluator_fields = ({"evaluator": str(args.evaluator.resolve()),
                         "evaluator_scale": args.leaf_scale} if args.evaluator else {})
    net_cls = SearchNetPlayer if args.evaluator else SearchPlayer
    net = net_cls(
        account_configuration=AccountConfiguration("dn" + secrets.token_hex(5), None),
        config=SearchConfig(**{**base.__dict__, **evaluator_fields,
                          "selection": args.net_selection,
                          "inference": args.net_inference == "on",
                          "damage_inference": args.net_damage_inference == "on",
                          "mode": args.net_mode,
                          "eval_lowhp_knee": args.net_eval_lowhp,
                          "eval_boost_scale": args.net_eval_boost_scale,
                          **_tendency_fields(args, args.net_tendency),
                          **_quality(args, "net")}),
        executor=pool, **options,
    )
    if args.opponent_evaluator:
        stock_eval_fields = {"evaluator": str(args.opponent_evaluator.resolve()),
                             "evaluator_scale": args.opponent_leaf_scale}
    elif args.stock_uses_evaluator and args.evaluator:
        stock_eval_fields = evaluator_fields
    else:
        stock_eval_fields = {}
    stock_cls = SearchNetPlayer if stock_eval_fields else SearchPlayer
    stock = stock_cls(
        account_configuration=AccountConfiguration("ds" + secrets.token_hex(5), None),
        config=SearchConfig(**{
            **base.__dict__, **stock_eval_fields,
            "selection": args.stock_selection,
            "inference": args.stock_inference == "on",
            "damage_inference": args.stock_damage_inference == "on",
            "mode": args.stock_mode,
            "eval_lowhp_knee": args.stock_eval_lowhp,
            "eval_boost_scale": args.stock_eval_boost_scale,
            **_tendency_fields(args, args.stock_tendency),
            **_quality(args, "stock"),
        }),
        executor=pool, **options,
    )
    try:
        await bench._wait_for_player_logins([net, stock], args.connect_timeout)
        await asyncio.wait_for(
            net.battle_against(stock, n_battles=share), timeout=args.game_timeout * share
        )
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in (net, stock)))
    return net, stock


async def _run(args):
    pool = get_pool(args.workers)
    shares = bench.split_games(args.games, args.pairs)
    tasks = [asyncio.ensure_future(_duel_pair(args, s, pool)) for s in shares]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not (math.isfinite(args.leaf_scale) and args.leaf_scale > 0):
        print(f"--leaf-scale must be finite and > 0, got {args.leaf_scale}", file=sys.stderr)
        return 2
    if not (math.isfinite(args.selection_gamma) and 0 < args.selection_gamma <= 1):
        print(f"--selection-gamma must be in (0, 1], got {args.selection_gamma}", file=sys.stderr)
        return 2
    if "ucb1" != args.net_selection or "ucb1" != args.stock_selection:
        if args.threads > 1:
            print("exp3/rm selection needs --threads 1", file=sys.stderr)
            return 2
    if "on" in (args.net_tendency, args.stock_tendency) and args.threads > 1:
        print("tendency needs --threads 1", file=sys.stderr)
        return 2
    if "nash" in (args.net_mode, args.stock_mode) and args.threads > 1:
        print("nash mode needs --threads 1", file=sys.stderr)
        return 2
    for ev in (args.evaluator, args.opponent_evaluator):
        if ev is not None and not ev.is_file():
            print(f"no such evaluator: {ev}", file=sys.stderr)
            return 2
    if not (math.isfinite(args.opponent_leaf_scale) and args.opponent_leaf_scale > 0):
        print(f"--opponent-leaf-scale must be finite and > 0, got {args.opponent_leaf_scale}",
              file=sys.stderr)
        return 2
    start = time.perf_counter()
    try:
        pairs = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"duel error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start
    n = sum(net.n_finished_battles for net, _ in pairs)
    wins = sum(net.n_won_battles for net, _ in pairs)
    lo, hi = bench.wilson_interval(wins, n)
    print("\n=== duel: search-net vs search ===")
    if args.evaluator:
        print(f"evaluator: {args.evaluator} (leaf scale {args.leaf_scale:g}, "
              f"{'both sides' if args.stock_uses_evaluator else 'net side only'})")
    else:
        print("evaluator: none (stock search on both sides)")
    if args.opponent_evaluator:
        print(f"opponent evaluator: {args.opponent_evaluator} "
              f"(leaf scale {args.opponent_leaf_scale:g})")
    print(f"selection: net={args.net_selection} stock={args.stock_selection} "
          f"(gamma {args.selection_gamma:g} for exp3/rm); "
          f"mode: net={args.net_mode} stock={args.stock_mode}; "
          f"tendency: net={args.net_tendency} stock={args.stock_tendency}")
    for side in ("net", "stock"):
        q = _quality(args, side)
        if q["escalate"] or q["prune_noop"] or q["tera_margin"]:
            print(f"589.47 {side}: {q}")
    print(f"compute (both sides): {args.search_time_ms} ms/world, parallelism {args.parallelism}, "
          f"worlds {args.worlds or '2 x parallelism (4x early)'}, threads {args.threads}; "
          f"{len(pairs)} parallel pairs, pool of {args.workers}")
    print(f"search-net wins: {wins}/{n} ({wins / max(n, 1):.1%}); Wilson 95% CI: [{lo:.1%}, {hi:.1%}]")
    for label, idx in (("search-net", 0), ("search    ", 1)):
        iters = sum(p[idx].stats.iterations for p in pairs)
        ms = sum(p[idx].stats.search_ms for p in pairs)
        fb = sum(p[idx].stats.fallbacks for p in pairs)
        st = [p[idx].stats for p in pairs]
        print(f"{label}: decisions={sum(x.searched for x in st)} escalations={sum(x.escalations for x in st)} "
              f"noop_pruned={sum(x.noop_pruned for x in st)} tera_folded={sum(x.tera_folded for x in st)}")
        werr = sum(p[idx].stats.world_errors for p in pairs)
        print(f"{label}: {iters / (ms / 1000) if ms else 0:,.0f} iterations per second of "
              f"per-world budget ({iters:,} iterations over {ms / 1000:.0f} s); "
              f"fallbacks={fb} world_errors={werr}")
    print(f"wall time: {elapsed:.0f} s")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
