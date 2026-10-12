"""Compare poke-env's *public view* of a Pokemon with the owner's own battle object.

    python scripts/tracker_diff.py --games 20 [--time-ms 10]

Two in-process SearchPlayers play each other (needs the Showdown server on
127.0.0.1:8000). At every decision each player's view of the opponent (HP fraction,
boosts, status and counters, effects, ability/item, known moves and PP, side
conditions, weather, fields) is compared with what the opponent's own battle object
knows. Anything the viewer has WRONG (as opposed to merely not knowing yet) is
counted as a tracker bug and a few examples are printed (bead 589.27, task 3).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
from collections import Counter, defaultdict

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench, recorder
from showdown_bot.players.search import SearchConfig, SearchPlayer, get_pool

BAD: Counter = Counter()
TOTAL: Counter = Counter()
EXAMPLES: dict[str, list[str]] = defaultdict(list)


def note(key: str, ok: bool, example: str = "") -> None:
    TOTAL[key] += 1
    if not ok:
        BAD[key] += 1
        if len(EXAMPLES[key]) < 4:
            EXAMPLES[key].append(example)


def name_of(value) -> str:
    return str(getattr(value, "name", value)).lower()


def compare_mon(view, real, turn, tag) -> None:
    """``view``: the viewer's poke-env Pokemon; ``real``: the owner's Pokemon."""
    if view is None or real is None:
        return
    label = f"{tag}:{view.species}"
    if real.max_hp:
        real_frac = real.current_hp / real.max_hp
        view_frac = view.current_hp_fraction
        # the viewer sees whole percents (opponent) or exact hp (own side)
        note(f"{tag}.hp_fraction(>1.5%)", abs(real_frac - view_frac) <= 0.015,
             f"{label} t{turn} view={view_frac:.3f} real={real_frac:.3f}")
    note(f"{tag}.status", name_of(view.status) == name_of(real.status),
         f"{label} t{turn} view={view.status} real={real.status}")
    if real.status is not None and view.status == real.status:
        note(f"{tag}.status_counter", (view.status_counter or 0) == (real.status_counter or 0),
             f"{label} t{turn} {real.status} view={view.status_counter} real={real.status_counter}")
    vb = {k: v for k, v in view.boosts.items() if v}
    rb = {k: v for k, v in real.boosts.items() if v}
    note(f"{tag}.boosts", vb == rb, f"{label} t{turn} view={vb} real={rb}")
    ve = {name_of(e) for e in view.effects}
    re_ = {name_of(e) for e in real.effects}
    note(f"{tag}.effects(set)", ve == re_, f"{label} t{turn} extra={ve - re_} missing={re_ - ve}")
    both = [e for e in view.effects if e in real.effects]
    note(f"{tag}.effect_counters", all((view.effects[e] or 0) == (real.effects[e] or 0) for e in both),
         f"{label} t{turn} " + str({name_of(e): (view.effects[e], real.effects[e]) for e in both
                                    if (view.effects[e] or 0) != (real.effects[e] or 0)}))
    # known information must be right
    if view.item not in (None, "", "unknown_item"):
        note(f"{tag}.item_known_wrong", view.item == real.item,
             f"{label} t{turn} view={view.item} real={real.item}")
    if view.item is None or view.item == "":
        note(f"{tag}.item_marked_gone_wrong", real.item in (None, ""),
             f"{label} t{turn} view=gone real={real.item}")
    if view.ability:
        note(f"{tag}.ability_known_wrong", view.ability == real.ability,
             f"{label} t{turn} view={view.ability} real={real.ability}")
    note(f"{tag}.terastallized", bool(view.is_terastallized) == bool(real.is_terastallized),
         f"{label} t{turn}")
    if view.is_terastallized and real.is_terastallized:
        note(f"{tag}.tera_type", view.tera_type == real.tera_type,
             f"{label} t{turn} view={view.tera_type} real={real.tera_type}")
    note(f"{tag}.fainted", bool(view.fainted) == bool(real.fainted), f"{label} t{turn}")
    for move_id, move in view.moves.items():
        rmove = real.moves.get(move_id)
        note(f"{tag}.known_move_not_real", rmove is not None,
             f"{label} t{turn} view_move={move_id} real_moves={list(real.moves)}")
        if rmove is not None:
            note(f"{tag}.move_pp", move.current_pp == rmove.current_pp,
                 f"{label} t{turn} {move_id} view={move.current_pp} real={rmove.current_pp}")


def compare_battles(viewer, owner, tag) -> None:
    """viewer's view of owner's team, with owner = the opponent's own battle object."""
    turn = viewer.turn
    compare_mon(viewer.opponent_active_pokemon, owner.active_pokemon, turn, f"{tag}.active")
    for ident, mon in viewer.opponent_team.items():
        if mon is viewer.opponent_active_pokemon:
            continue
        real = owner.team.get(ident)
        if real is None:
            note(f"{tag}.bench_identifier_missing", False, f"t{turn} {ident} not in {list(owner.team)}")
            continue
        compare_mon(mon, real, turn, f"{tag}.bench")
    vs = {name_of(k): v for k, v in viewer.opponent_side_conditions.items()}
    rs = {name_of(k): v for k, v in owner.side_conditions.items()}
    note(f"{tag}.side_conditions(set)", set(vs) == set(rs), f"t{turn} view={vs} real={rs}")
    note(f"{tag}.side_condition_values", all(vs[k] == rs[k] for k in set(vs) & set(rs)),
         f"t{turn} view={vs} real={rs}")
    vw, rw = {name_of(k): v for k, v in viewer.weather.items()}, {name_of(k): v for k, v in owner.weather.items()}
    note(f"{tag}.weather", vw == rw, f"t{turn} view={vw} real={rw}")
    vf, rf = {name_of(k): v for k, v in viewer.fields.items()}, {name_of(k): v for k, v in owner.fields.items()}
    note(f"{tag}.fields", vf == rf, f"t{turn} view={vf} real={rf}")


class DiffPlayer(SearchPlayer):
    async def choose_move(self, battle):
        peer = recorder.peer_battle(battle)
        if peer is not None and peer.turn == battle.turn:
            try:
                compare_battles(battle, peer, "opp_as_seen")  # my view of their mons
                compare_battles(peer, battle, "own_as_seen_by_peer")  # their view of mine
            except Exception as exc:  # keep playing
                note("compare_crash", False, repr(exc))
        return await super().choose_move(battle)


async def run(args) -> None:
    pool = get_pool(args.workers)
    options = {
        "battle_format": "gen9randombattle",
        "max_concurrent_battles": 1,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR,
        "open_timeout": 30,
    }
    config = SearchConfig(search_time_ms=args.time_ms, worlds=2)
    players = [
        DiffPlayer(account_configuration=AccountConfiguration("td" + secrets.token_hex(5), None),
                   config=config, executor=pool, **options)
        for _ in range(2)
    ]
    for p in players:
        recorder.register_player(p)
    try:
        await bench._wait_for_player_logins(players, 30)
        await asyncio.wait_for(players[0].battle_against(players[1], n_battles=args.games),
                               timeout=900 * args.games)
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in players))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--time-ms", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run(args))
    print(f"{'check':46s} {'n':>7s} {'wrong':>7s}")
    for key in sorted(TOTAL):
        flag = "  <--" if BAD[key] else ""
        print(f"{key:46s} {TOTAL[key]:7d} {100 * BAD[key] / TOTAL[key]:6.2f}%{flag}")
    for key, examples in EXAMPLES.items():
        print(f"\n{key}:")
        for example in examples:
            print("   ", example)


if __name__ == "__main__":
    main()
