"""Tendency model vs scripted exploitable opponents (bead 589.40).

    python scripts/biased_duel.py --bias bait --tendency on --games 100 --pairs 6 --workers 6

Our side is a SearchPlayer (tendency on/off, final-move mode ``--mode`` when off); the
opponent is a ``BiasedSearchPlayer`` (bias bait: status/Protect whenever we hold Sucker
Punch-like moves; attack: always attacks). Prints our win rate with a Wilson 95% CI, plus
the same for games where the bias triggered at least once. Needs the local server on
127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import time

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench
from showdown_bot.players.biased import BIASES, BiasedSearchPlayer
from showdown_bot.players.search import MODES, SearchConfig, SearchPlayer, get_pool


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bias", choices=BIASES, required=True)
    p.add_argument("--tendency", choices=("on", "off"), default="off")
    p.add_argument("--mode", choices=MODES, default="mixed", help="our final-move rule when tendency is off")
    p.add_argument("--tendency-kappa", type=float, default=None)
    p.add_argument("--tendency-delta", type=float, default=None)
    p.add_argument("--games", type=bench._positive_int, default=100)
    p.add_argument("--pairs", type=bench._positive_int, default=6)
    p.add_argument("--workers", type=bench._positive_int, default=6)
    p.add_argument("--search-time-ms", type=bench._positive_int, default=100)
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=900)
    return p


async def _pair(args, share, pool):
    options = {
        "battle_format": args.format, "max_concurrent_battles": 1,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR, "open_timeout": 30,
    }
    fields = {"tendency": args.tendency == "on"}
    if args.tendency_kappa is not None:
        fields["tendency_kappa"] = args.tendency_kappa
    if args.tendency_delta is not None:
        fields["tendency_delta"] = args.tendency_delta
    us = SearchPlayer(
        account_configuration=AccountConfiguration("bu" + secrets.token_hex(5), None),
        config=SearchConfig(search_time_ms=args.search_time_ms, mode=args.mode, **fields),
        executor=pool, **options,
    )
    them = BiasedSearchPlayer(
        bias=args.bias,
        account_configuration=AccountConfiguration("bb" + secrets.token_hex(5), None),
        config=SearchConfig(search_time_ms=args.search_time_ms),
        executor=pool, **options,
    )
    them.peer = us
    try:
        await bench._wait_for_player_logins([us, them], 30)
        await asyncio.wait_for(us.battle_against(them, n_battles=share), timeout=args.game_timeout * share)
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in (us, them)))
    return us, them


async def _run(args):
    pool = get_pool(args.workers)
    tasks = [asyncio.ensure_future(_pair(args, s, pool)) for s in bench.split_games(args.games, args.pairs)]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def main(argv=None):
    args = _parser().parse_args(argv)
    start = time.perf_counter()
    pairs = asyncio.run(_run(args))
    n = sum(u.n_finished_battles for u, _ in pairs)
    wins = sum(u.n_won_battles for u, _ in pairs)
    lo, hi = bench.wilson_interval(wins, n)
    trig = sum(t.trigger_turns for _, t in pairs)
    biased = sum(t.biased_turns for _, t in pairs)
    tw = tn = 0
    for u, t in pairs:
        for tag, battle in u.battles.items():
            if battle.finished and tag in t.triggered_games:
                tn += 1
                tw += bool(battle.won)
    print(f"\n=== biased duel: bias={args.bias} tendency={args.tendency} mode={args.mode} ===")
    print(f"our win rate: {wins}/{n} ({wins / max(n, 1):.1%}); Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
    if tn:
        l2, h2 = bench.wilson_interval(tw, tn)
        print(f"games where the bias triggered: {tw}/{tn} ({tw / tn:.1%}) CI [{l2:.1%}, {h2:.1%}]")
    print(f"opponent trigger turns={trig} biased turns={biased}; "
          f"our decisions with evidence={sum(u.tendency_decisions for u, _ in pairs)}; "
          f"fallbacks={sum(u.stats.fallbacks for u, _ in pairs)} wall={time.perf_counter() - start:.0f}s")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
