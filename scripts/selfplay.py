"""Record self-play games (search vs search) for expert-iteration training data.

    python -m uv run python scripts/selfplay.py --games 20 --pairs 4
    python -m uv run python scripts/selfplay.py --vs foulplay --games 20 --pairs 2

Needs the local Showdown server on 127.0.0.1:8000 (scripts\\start_server.ps1).
Output: data/selfplay/<run-id>/*.jsonl.gz (schema: docs/selfplay-data.md).

Search vs search runs ``--pairs`` pairs in this process, each a pair of
SearchPlayers with unique usernames playing its share of the games one at a
time; both sides record, plus an exact "truth" state per decision. With
``--vs foulplay`` the existing bench machinery launches Foul Play subprocesses
and only our side records (no truth: Foul Play is not in this process).
All search work shares one process pool of ``--workers`` processes.
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

from showdown_bot import bench, dataset, recorder
from showdown_bot.players.search import SearchConfig, SearchPlayer, get_pool, total_stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--games", type=bench._positive_int, default=20, help="total games")
    p.add_argument("--pairs", type=bench._positive_int, default=4, help="parallel pairs")
    p.add_argument("--vs", choices=("search", "foulplay"), default="search")
    p.add_argument("--run-id", default=time.strftime("run-%Y%m%d-%H%M%S"))
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "selfplay")
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--search-time-ms", type=bench._positive_int, default=100)
    p.add_argument("--parallelism", type=bench._positive_int, default=1,
                   help="per-player compute: worlds = 2 x parallelism (Foul Play's rule)")
    p.add_argument("--worlds", type=bench._positive_int, default=None, help="fixed worlds per decision")
    p.add_argument("--threads", type=bench._positive_int, default=1)
    p.add_argument("--workers", type=bench._positive_int, default=os.cpu_count() or 4,
                   help="shared MCTS process pool size (search vs search)")
    p.add_argument("--no-truth", action="store_true", help="do not record exact-state truth")
    p.add_argument("--connect-timeout", type=bench._positive_seconds, default=30)
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=900)
    return p


async def _selfplay_pair(args, run_dir: Path, rec: recorder.GameRecorder, share: int, pool) -> list:
    options = {
        "battle_format": args.format,
        "max_concurrent_battles": 1,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR,
        "open_timeout": args.connect_timeout,
    }
    config = SearchConfig(
        search_time_ms=args.search_time_ms,
        parallelism=args.parallelism,
        worlds=args.worlds,
        threads=args.threads,
    )
    players = [
        SearchPlayer(
            account_configuration=AccountConfiguration("sp" + secrets.token_hex(5), None),
            config=config,
            executor=pool,
            recorder=rec,
            **options,
        )
        for _ in range(2)
    ]
    try:
        await bench._wait_for_player_logins(players, args.connect_timeout)
        await asyncio.wait_for(
            players[0].battle_against(players[1], n_battles=share),
            timeout=args.game_timeout * share,
        )
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in players))
    return players


async def _run_selfplay(args, run_dir: Path, rec: recorder.GameRecorder) -> None:
    pool = get_pool(args.workers)
    shares = bench.split_games(args.games, args.pairs)
    tasks = [asyncio.ensure_future(_selfplay_pair(args, run_dir, rec, s, pool)) for s in shares]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_foulplay(args, run_dir: Path) -> None:
    # SearchPlayer reads these when bench instantiates it.
    os.environ[recorder.RECORD_DIR_ENV] = str(run_dir)
    os.environ["SHOWDOWN_SEARCH_TIME_MS"] = str(args.search_time_ms)
    os.environ["SHOWDOWN_SEARCH_PARALLELISM"] = str(args.parallelism)
    os.environ["SHOWDOWN_SEARCH_THREADS"] = str(args.threads)
    if args.worlds:
        os.environ["SHOWDOWN_SEARCH_WORLDS"] = str(args.worlds)
    await bench._play_against_foulplay(
        "search", bench.FOULPLAY_NAME, args.games, args.format, args.connect_timeout,
        args.search_time_ms, args.parallelism, args.threads, args.game_timeout,
        fp_instances=args.pairs,
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.out / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = recorder.GameRecorder(run_dir, truth=not args.no_truth and args.vs == "search")
    print(f"recording to {run_dir}", flush=True)
    start = time.perf_counter()
    try:
        if args.vs == "search":
            asyncio.run(_run_selfplay(args, run_dir, rec))
        else:
            asyncio.run(_run_foulplay(args, run_dir))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"selfplay error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    stats = dataset.collect_statistics(run_dir)
    games, hours = stats["games"], elapsed / 3600
    positions = stats["worlds"] + stats["truth_states"]
    print("\n=== selfplay summary ===")
    print(f"wall time:            {elapsed:.1f} s")
    battles = len(stats["battle_tags"])
    print(f"battles finished:     {battles} of {args.games} requested ({games} game files)")
    print(f"decisions:            {stats['decisions']}")
    print(f"worlds recorded:      {stats['worlds']}")
    print(f"truth states:         {stats['truth_states']}")
    print(f"positions/hour:       {positions / hours:,.0f} (worlds+truth); "
          f"{stats['worlds'] / hours:,.0f} worlds; {stats['decisions'] / hours:,.0f} decisions")
    if battles:
        print(f"disk:                 {stats['bytes'] / 1e6:.2f} MB total; "
              f"{stats['bytes'] / 1e6 / battles * 1000:.1f} MB per 1,000 games")
    print(f"search stats:         {total_stats().summary()}")
    return 0 if games else 1


if __name__ == "__main__":
    raise SystemExit(main())
