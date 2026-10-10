"""Fuzz the engine with random-but-valid states and short MCTS runs.

Covers the inputs that used to panic (sleep_turns above the engine's maximum,
Encore without a known last move) and the items added in 0.0.48+sdb.2
(Light Clay, Leppa Berry, Scope Lens). Exits non-zero if any search panics.

    python -m uv run python scripts/engine_fuzz.py [--states 2000] [--ms 20] [--seed 1]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

from poke_engine import (
    Move,
    Pokemon,
    Side,
    SideConditions,
    State,
    monte_carlo_tree_search,
)
from spike_engine import TEAM_ONE, TEAM_TWO  # scripts/ is sys.path[0] when run as a script

ITEMS = [
    "none", "leftovers", "lightclay", "leppaberry", "scopelens", "lifeorb", "sitrusberry",
    "choicescarf", "focussash", "lumberry", "chestoberry", "heavydutyboots", "unknownthing",
]
STATUSES = ["none", "none", "none", "sleep", "sleep", "sleep", "burn", "paralyze", "poison", "toxic"]
MOVE_POOL = [
    "tackle", "slash", "reflect", "lightscreen", "auroraveil", "encore", "sleeptalk", "rest",
    "spore", "earthquake", "closecombat", "protect", "stealthrock", "taunt", "uturn",
    "shadowball", "recover", "swordsdance", "thunderwave", "frostbreath",
]
LAST_USED = ["move:none", "move:none", "switch:0", "switch:1", "move:0", "move:1", "move:2", "move:3"]
WEATHERS = ["none", "none", "hail", "snow", "sun", "rain"]


def random_pokemon(rng: random.Random, roster_entry: tuple, force_sleep: bool) -> Pokemon:
    name, types, ability, _item, tera_type, stats, _moves, nature = roster_entry
    hp, attack, defense, special_attack, special_defense, speed = stats
    status = rng.choice(STATUSES)
    if force_sleep:
        status = "sleep"
    maxhp = hp
    current = rng.choice([maxhp, maxhp, rng.randint(1, maxhp)])
    moves = []
    for move_id in rng.sample(MOVE_POOL, 4):
        pp = rng.choice([32, 16, 8, 5, 1, 0])
        moves.append(Move(id=move_id, pp=pp, disabled=rng.random() < 0.05))
    return Pokemon(
        id=name,
        level=100,
        types=types,
        base_types=types,
        hp=current,
        maxhp=maxhp,
        ability=ability,
        base_ability=ability,
        item=rng.choice(ITEMS),
        nature=nature,
        attack=attack,
        defense=defense,
        special_attack=special_attack,
        special_defense=special_defense,
        speed=speed,
        status=status,
        sleep_turns=rng.randint(0, 10) if status == "sleep" else 0,
        rest_turns=rng.choice([0, 0, 0, 1, 2]) if status == "sleep" else 0,
        tera_type=tera_type,
        moves=moves,
    )


def random_side(rng: random.Random, roster: tuple[tuple, ...]) -> Side:
    mons = []
    for i, entry in enumerate(roster):
        mons.append(random_pokemon(rng, entry, force_sleep=(i == 0 and rng.random() < 0.5)))
    volatiles = set()
    if rng.random() < 0.4:
        volatiles.add("encore")
    if rng.random() < 0.1:
        volatiles.add("taunt")
    return Side(
        pokemon=mons,
        volatile_statuses=volatiles,
        last_used_move=rng.choice(LAST_USED),
        side_conditions=SideConditions(
            reflect=rng.choice([0, 0, rng.randint(1, 8)]),
            light_screen=rng.choice([0, 0, rng.randint(1, 8)]),
            aurora_veil=rng.choice([0, 0, rng.randint(1, 8)]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=2000)
    parser.add_argument("--ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    panics: Counter[str] = Counter()
    ok = 0
    encore_no_last = 0
    sleepy_high = 0
    started = time.perf_counter()
    for n in range(args.states):
        side_one = random_side(rng, TEAM_ONE)
        side_two = random_side(rng, TEAM_TWO)
        for side in (side_one, side_two):
            active = side.pokemon[0]
            if "encore" in side.volatile_statuses and side.last_used_move in ("move:none", "switch:0", "switch:1"):
                encore_no_last += 1
            if active.status == "sleep" and active.sleep_turns > 3:
                sleepy_high += 1
        state = State(
            side_one=side_one,
            side_two=side_two,
            weather=rng.choice(WEATHERS),
        )
        try:
            monte_carlo_tree_search(state, duration_ms=args.ms, iterations=0, threads=1)
            ok += 1
        except BaseException as exc:  # pyo3 PanicException derives from BaseException
            if isinstance(exc, KeyboardInterrupt):
                raise
            panics[f"{type(exc).__name__}: {exc}"] += 1
            if sum(panics.values()) <= 3:
                print(f"state {n} failed: {type(exc).__name__}: {exc}\n{state.to_string()}", file=sys.stderr)
    elapsed = time.perf_counter() - started
    print(
        f"states={args.states} ok={ok} panics={sum(panics.values())} "
        f"(encore-without-last-move sides={encore_no_last}, sleep>3 sides={sleepy_high}) "
        f"in {elapsed:.1f}s"
    )
    for message, count in panics.most_common():
        print(f"  {count}x {message}")
    return 1 if panics else 0


if __name__ == "__main__":
    raise SystemExit(main())
