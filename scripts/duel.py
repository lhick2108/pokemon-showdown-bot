"""Duel: search with a learned evaluator (search-net) vs stock search, at equal compute.

    python -m uv run python scripts/duel.py --evaluator models/x.pev --games 100 --pairs 4

Both sides get the same SearchConfig (time per world, worlds, threads, mode)
except for ``evaluator``; each side's config is passed to its player as a
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
import os
import secrets
import sys
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench
from showdown_bot.players.search import SearchConfig, SearchNetPlayer, SearchPlayer, get_pool


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--evaluator", type=Path, required=True, help="learned .pev file")
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
        worlds=args.worlds, threads=args.threads,
    )
    net = SearchNetPlayer(
        account_configuration=AccountConfiguration("dn" + secrets.token_hex(5), None),
        config=SearchConfig(**{**base.__dict__, "evaluator": str(args.evaluator.resolve())}),
        executor=pool, **options,
    )
    stock = SearchPlayer(
        account_configuration=AccountConfiguration("ds" + secrets.token_hex(5), None),
        config=base, executor=pool, **options,
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
    if not args.evaluator.is_file():
        print(f"no such evaluator: {args.evaluator}", file=sys.stderr)
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
    print(f"evaluator: {args.evaluator}")
    print(f"compute (both sides): {args.search_time_ms} ms/world, parallelism {args.parallelism}, "
          f"worlds {args.worlds or '2 x parallelism (4x early)'}, threads {args.threads}; "
          f"{len(pairs)} parallel pairs, pool of {args.workers}")
    print(f"search-net wins: {wins}/{n} ({wins / max(n, 1):.1%}); Wilson 95% CI: [{lo:.1%}, {hi:.1%}]")
    for label, idx in (("search-net", 0), ("search    ", 1)):
        iters = sum(p[idx].stats.iterations for p in pairs)
        ms = sum(p[idx].stats.search_ms for p in pairs)
        fb = sum(p[idx].stats.fallbacks for p in pairs)
        werr = sum(p[idx].stats.world_errors for p in pairs)
        print(f"{label}: {iters / (ms / 1000) if ms else 0:,.0f} iterations per second of "
              f"per-world budget ({iters:,} iterations over {ms / 1000:.0f} s); "
              f"fallbacks={fb} world_errors={werr}")
    print(f"wall time: {elapsed:.0f} s")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
