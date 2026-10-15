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

``--pool a=x.pev:scale=1.5,stock`` draws side B's evaluator per game from a pool (the
opponent's label lands in the header config as ``opponent`` for side A, ``label`` for B, its
full spec as ``spec``). Entries take any league search-spec key, so an AlphaStar-style
exploiter that learns side A's habits within a game is
``--pool exploit=models/x.pev:scale=1.5:mode=nash:tendency=on`` (see parse_pool).
``--full-frac P --fast-time-ms T`` is playout cap randomization: a decision gets the full
search and is recorded with prob P, else a T ms search that is not recorded (search vs
search only; the Foul Play path ignores it).
All search work shares one process pool of ``--workers`` processes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import dataclasses
import hashlib
import os
import random
import secrets
import sys
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench, dataset, recorder
from showdown_bot.players.search import SearchConfig, SearchPlayer, get_pool, inference_summary, total_stats

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class PoolEntry:
    """One pool opponent: ``fields`` are SearchConfig overrides on top of side A's config."""

    label: str
    evaluator: str | None = None
    scale: float = 1.0
    fields: dict = dataclasses.field(default_factory=dict)
    spec: str = "stock"


def parse_pool(text: str) -> list[PoolEntry]:
    """Parse ``--pool``: comma-separated ``stock`` or ``label=path.pev[:key=value...]``.

    The keys are the league's search-spec keys (``league.SPEC_KEYS['search']``, parsed and
    validated by ``league.parse_spec``): ``scale``, ``mode`` (mixed|argmax|nash), ``tendency``
    (on|off, with ``tkappa``/``tdelta``; needs ``mode=nash``), ``time_ms``, ``worlds``,
    ``escalate``, ``ev2``/``scale2`` (evaluator ensemble), ... The path may be ``stock`` (no
    evaluator) to vary only the other options. A Windows drive colon in the path is fine.
    Example exploiter:
    ``exploit=models/gen2-res-searchabs-mlp16.pev:scale=1.5:mode=nash:tendency=on``
    """
    from showdown_bot import league

    entries: list[PoolEntry] = []
    for raw in (part.strip() for part in text.split(",")):
        if not raw:
            continue
        if raw == "stock":
            entries.append(PoolEntry("stock"))
            continue
        if "=" not in raw:
            raise ValueError(f"bad pool spec {raw!r}: want 'stock' or 'label=path[:key=value...]'")
        label, rest = raw.split("=", 1)
        label = label.strip()
        parts = league._split_spec_params(rest)
        path = parts[0] if "=" not in parts[0] else ""
        options = parts[1:] if path else parts
        if not label or not path:
            raise ValueError(f"bad pool spec {raw!r}: empty label or path")
        text_spec = f"{label}=search" + ("" if path == "stock" else f":evaluator={path}")
        text_spec += "".join(":" + o for o in options)
        spec = league.parse_spec(text_spec)  # validates keys and values
        fields = league.search_config_fields(spec)
        if fields.get("tendency") and fields.get("mode") != "nash":
            raise ValueError(f"pool spec {raw!r}: tendency=on needs mode=nash")
        entries.append(PoolEntry(label, fields.get("evaluator"), fields.get("evaluator_scale", 1.0),
                                 fields, raw))
    if not entries:
        raise ValueError("empty pool")
    if len({e.label for e in entries}) != len(entries):
        raise ValueError("duplicate pool labels")
    return entries


def parse_weights(text: str | None, n: int) -> list[float]:
    if text is None:
        return [1.0] * n
    weights = [float(w) for w in text.split(",") if w.strip()]
    if len(weights) != n or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"--pool-weights needs {n} non-negative weights with a positive sum")
    return weights


def sample_pool(entries: list[PoolEntry], weights: list[float], rng: random.Random) -> PoolEntry:
    return rng.choices(entries, weights=weights, k=1)[0]


def entry_config(base: SearchConfig, entry: PoolEntry) -> SearchConfig:
    """Side B's config for a pool entry: side A's config with the entry's spec applied.

    Without options an entry only swaps the evaluator/scale (as before); ``spec`` is the full
    entry spec and lands in the game header next to ``label``.
    """
    fields = {"evaluator": entry.evaluator, "evaluator_scale": entry.scale, **entry.fields}
    return dataclasses.replace(base, **fields, label=entry.label, opponent="main", spec=entry.spec)


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
    p.add_argument("--evaluator", default=None,
                   help="learned .pev leaf evaluator for both players (default: stock evaluate())")
    p.add_argument("--leaf-scale", type=float, default=1.0, help="leaf scale for --evaluator")
    p.add_argument("--pool", default=None,
                   help="opponent pool for side B (search vs search): comma-separated 'stock' or "
                        "label=evaluator_path[:key=value...] (keys of the league search spec: scale, mode, "
                        "tendency, tkappa, tdelta, time_ms, worlds, escalate, ev2, scale2, ...); "
                        "side A stays the --evaluator config. Exploiter example: "
                        "exploit=models/x.pev:scale=1.5:mode=nash:tendency=on")
    p.add_argument("--pool-weights", default=None, help="comma-separated sampling weights (default uniform)")
    p.add_argument("--full-frac", type=float, default=1.0,
                   help="playout cap randomization: P(full recorded search) per decision (1.0 = off)")
    p.add_argument("--fast-time-ms", type=bench._positive_int, default=50,
                   help="search time of the unrecorded fast decisions (with --full-frac < 1)")
    p.add_argument("--seed", type=int, default=None, help="seed pool draws and full/fast draws")
    p.add_argument("--workers", type=bench._positive_int, default=os.cpu_count() or 4,
                   help="shared MCTS process pool size (search vs search)")
    p.add_argument("--inference", choices=("on", "off"), default="on",
                   help="589.28 opponent set inference (off = pre-589.28 baseline sampling)")
    p.add_argument("--damage-inference", choices=("on", "off"), default="on",
                   help="damage-roll part of the inference (needs --inference on)")
    p.add_argument("--no-truth", action="store_true", help="do not record exact-state truth")
    p.add_argument("--export-nodes", type=int, default=0,
                   help="bead rnm: also record up to K sampled interior MCTS nodes per searched world "
                        "(visit-weighted; state, visits, search value) as extra value targets "
                        "(needs --threads 1 and an engine with export_nodes, 0.0.48+sdb.9; 0 = off)")
    p.add_argument("--prior", default=None, help="policy prior .ppol for PUCT selection (both sides)")
    p.add_argument("--c-puct", type=float, default=1.0, help="PUCT exploration constant with --prior")
    p.add_argument("--export-min-visits", type=int, default=16,
                   help="--export-nodes: only nodes with at least this many visits are eligible")
    p.add_argument("--connect-timeout", type=bench._positive_seconds, default=30)
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=900)
    return p


def hash_seed(seed: str, key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}/{key}".encode()).digest()[:8], "big")


async def _selfplay_pair(
    args, run_dir: Path, rec: recorder.GameRecorder, share: int, pool, index: int = 0
) -> list:
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
        evaluator=args.evaluator,
        evaluator_scale=args.leaf_scale,
        inference=args.inference == "on",
        damage_inference=args.damage_inference == "on",
        full_frac=args.full_frac,
        fast_time_ms=args.fast_time_ms,
        export_nodes=args.export_nodes,
        export_min_visits=args.export_min_visits,
        prior=args.prior,
        c_puct=args.c_puct,
        label="main",
    )
    entries = parse_pool(args.pool) if args.pool else [PoolEntry("main", args.evaluator, args.leaf_scale)]
    weights = parse_weights(args.pool_weights, len(entries))
    seed = None if args.seed is None else f"{args.seed}-{index}"
    pick_rng = random.Random(seed)

    def make(cfg, k):
        return SearchPlayer(
            account_configuration=AccountConfiguration("sp" + secrets.token_hex(5), None),
            config=cfg,
            executor=pool,
            recorder=rec,
            seed=None if seed is None else hash_seed(seed, k),
            **options,
        )

    player_a = make(config, "a")
    # one side-B player per pool entry (a pool of one entry == plain self-play)
    side_b = [
        make(entry_config(config, e), e.label)
        for e in entries
    ]
    players = [player_a, *side_b]
    try:
        await bench._wait_for_player_logins(players, args.connect_timeout)
        for _ in range(share):
            entry = sample_pool(entries, weights, pick_rng)
            player_a.config = dataclasses.replace(player_a.config, opponent=entry.label)
            await asyncio.wait_for(
                player_a.battle_against(side_b[entries.index(entry)], n_battles=1),
                timeout=args.game_timeout,
            )
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in players))
    return players


async def _run_selfplay(args, run_dir: Path, rec: recorder.GameRecorder) -> None:
    pool = get_pool(args.workers)
    shares = bench.split_games(args.games, args.pairs)
    tasks = [asyncio.ensure_future(_selfplay_pair(args, run_dir, rec, s, pool, i)) for i, s in enumerate(shares)]
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
    if args.export_nodes < 0 or args.export_min_visits < 1 or (args.export_nodes and args.threads > 1):
        print("--export-nodes needs K >= 0, --export-min-visits >= 1 and --threads 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.full_frac <= 1.0:
        print("--full-frac must be in [0, 1]", file=sys.stderr)
        return 2
    if args.pool:
        try:
            parse_weights(args.pool_weights, len(parse_pool(args.pool)))
        except ValueError as exc:
            print(f"selfplay error: {exc}", file=sys.stderr)
            return 2
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
    print(f"inference stats:      {inference_summary()}")
    return 0 if games else 1


if __name__ == "__main__":
    raise SystemExit(main())
