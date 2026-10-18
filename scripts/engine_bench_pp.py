"""MCTS throughput with and without full PP tracking (sdb.22, 589.68.4).

Builds a few fixed states from the spike teams (scripts/spike_engine.py): the opening position, a
mid-game position (hurt / statused mons, sun up) and a late-game 3v2 position. Each state is searched
in two modes, interleaved so machine noise hits both equally:

  legacy    Move(pp=16)              engine tracks PP only below 10 (the behaviour up to sdb.21)
  tracked   Move(pp=24, max_pp=24)   engine decrements PP on every move (sdb.22 with known max PP)

Single thread, stock evaluate() (no net), iterations per second = total_visits / process CPU time (robust against other load on the box).

    python scripts/engine_bench_pp.py [--seconds 2] [--reps 5]

With a wheel older than sdb.22 `max_pp` does not exist: only the legacy rows are produced (the
baseline for the comparison).
"""

from __future__ import annotations

import argparse
import statistics
import time

from poke_engine import Move, Pokemon, Side, State, monte_carlo_tree_search
from spike_engine import TEAM_ONE, TEAM_TWO  # scripts/ is sys.path[0] when run as a script

try:
    Move(id="tackle", max_pp=8)
    HAS_MAX_PP = True
except TypeError:
    HAS_MAX_PP = False


def make_side(roster, tracked: bool, hp_frac=None, status=None, alive=6) -> Side:
    hp_frac = hp_frac or {}
    status = status or {}
    pokemon = []
    for i, (name, types, ability, item, tera, stats, moves, nature) in enumerate(roster):
        hp, atk, dfn, spa, spd, spe = stats
        cur = 0 if i >= alive else max(1, int(hp * hp_frac.get(i, 1.0)))
        if tracked:
            mv = [Move(id=m, pp=24, max_pp=24) for m in moves]
        else:
            mv = [Move(id=m, pp=16) for m in moves]
        pokemon.append(Pokemon(
            id=name, level=100, types=types, base_types=types, hp=cur, maxhp=hp, ability=ability,
            base_ability=ability, item=item, nature=nature, attack=atk, defense=dfn,
            special_attack=spa, special_defense=spd, speed=spe, status=status.get(i, "none"),
            tera_type=tera, moves=mv,
        ))
    return Side(pokemon=pokemon)


def make_states(tracked: bool) -> dict[str, State]:
    opening = State(side_one=make_side(TEAM_ONE, tracked), side_two=make_side(TEAM_TWO, tracked))
    mid = State(
        side_one=make_side(TEAM_ONE, tracked, hp_frac={0: 0.6, 1: 0.4, 2: 0.8}, status={1: "burn"}),
        side_two=make_side(TEAM_TWO, tracked, hp_frac={0: 0.5, 1: 0.9, 3: 0.3}, status={0: "toxic"}),
    )
    late = State(
        side_one=make_side(TEAM_ONE, tracked, hp_frac={0: 0.35, 1: 0.7, 2: 0.2}, alive=3),
        side_two=make_side(TEAM_TWO, tracked, hp_frac={0: 0.25, 1: 0.6}, alive=2),
    )
    return {"opening": opening, "midgame": mid, "late 3v2": late}


def run(state: State, seconds: float) -> float:
    started = time.process_time()
    result = monte_carlo_tree_search(state, duration_ms=int(seconds * 1000), iterations=0, threads=1)
    return result.total_visits / (time.process_time() - started)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    modes = [("legacy", False)] + ([("tracked", True)] if HAS_MAX_PP else [])
    states = {name: make_states(tr) for name, tr in modes}
    run(states["legacy"]["opening"], 1)  # warm-up
    rates: dict[tuple[str, str], list[float]] = {}
    for _ in range(args.reps):
        for sname in states["legacy"]:
            for mname, _tr in modes:
                rates.setdefault((sname, mname), []).append(run(states[mname][sname], args.seconds))

    print(f"single thread, stock evaluate(), {args.seconds}s x {args.reps} interleaved reps (median iterations/s)")
    print(f"{'state':<10} " + " ".join(f"{m:>12}" for m, _ in modes) + ("   tracked/legacy" if HAS_MAX_PP else ""))
    for sname in states["legacy"]:
        med = [statistics.median(rates[(sname, m)]) for m, _ in modes]
        line = f"{sname:<10} " + " ".join(f"{v:>12,.0f}" for v in med)
        if HAS_MAX_PP:
            line += f"   {med[1] / med[0]:>8.3f}x"
        print(line)


if __name__ == "__main__":
    main()
