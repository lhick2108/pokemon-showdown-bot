"""Upper-bound experiment (bead 589.27): how much would perfect set knowledge be worth?

    python scripts/oracle_duel.py --level items --games 100 --pairs 3

An "oracle" SearchPlayer is told its opponent's real item/ability/tera type (and with
``--level full`` the real moves) for every REVEALED opposing Pokemon, read from the
opponent's own battle object (both players live in this process). The unrevealed
slots stay sampled. It plays the stock SearchPlayer at identical compute. The oracle's
score above 50% bounds what any set-inference improvement (Foul Play's speed, damage
and item-exclusion inference) can buy; it is NOT a deployable player.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import math
import os
import secrets

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench, engine_state, recorder
from showdown_bot.players.search import SearchConfig, SearchPlayer, get_pool

ORACLES: dict[str, str] = {}  # username -> level
_ORIGINAL = engine_state.sample_opponent_team


def _real_value(mon):
    item = mon.item if mon.item and mon.item != "unknown_item" else ""
    tera = getattr(mon.tera_type, "name", None) or mon.tera_type
    return item, (mon.ability or mon.base_ability), (str(tera).lower() if tera else None)


def oracle_sample(battle, trackers, sampler, rng, **kwargs):
    slots = _ORIGINAL(battle, trackers, sampler, rng, **kwargs)
    level = ORACLES.get(getattr(battle, "player_username", ""))
    if level is None:
        return slots
    peer = recorder.peer_battle(battle)
    if peer is None:
        return slots
    by_species = {engine_state.to_id(m.species): m for m in peer.team.values()}
    for slot in slots:
        if slot.mon is None or slot.disguised:
            continue
        real = by_species.get(slot.species_id)
        if real is None:
            continue
        item, ability, tera = _real_value(real)
        changes = {"item": item, "ability": ability, "tera_type": tera or slot.sampled.tera_type}
        if level == "full":
            changes["moves"] = tuple(real.moves.keys())
        slot.sampled = dataclasses.replace(slot.sampled, **changes)
    return slots


engine_state.sample_opponent_team = oracle_sample


async def pair(args, share, pool):
    options = {
        "battle_format": "gen9randombattle", "max_concurrent_battles": 1,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR, "open_timeout": 30,
    }
    config = SearchConfig(search_time_ms=args.time_ms)
    players = [
        SearchPlayer(account_configuration=AccountConfiguration("or" + secrets.token_hex(5), None),
                     config=config, executor=pool, **options)
        for _ in range(2)
    ]
    for p in players:
        recorder.register_player(p)
    oracle, stock = players
    ORACLES[oracle.username] = args.level
    try:
        await bench._wait_for_player_logins(players, 30)
        await asyncio.wait_for(oracle.battle_against(stock, n_battles=share), timeout=1200 * share)
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in players))
    return oracle.n_won_battles, stock.n_won_battles, oracle.n_finished_battles


def wilson(wins, n, z=1.96):
    if not n:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


async def run(args):
    pool = get_pool(args.workers)
    shares = bench.split_games(args.games, args.pairs)
    results = await asyncio.gather(*(pair(args, s, pool) for s in shares))
    won = sum(r[0] for r in results)
    lost = sum(r[1] for r in results)
    n = won + lost
    lo, hi = wilson(won, n)
    print(f"level={args.level} time={args.time_ms}ms: oracle {won} - {lost} stock "
          f"({100 * won / max(1, n):.1f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%) over {n} decisive games")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("items", "full"), default="items")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--time-ms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 4))
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
