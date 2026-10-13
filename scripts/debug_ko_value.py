"""Why does the search undervalue a guaranteed KO?  (bead 589.43)

    python scripts/debug_ko_value.py LOG --from-record RECORDS --turn 19 [--time-ms 300] [--worlds 3]
        [--move thunderbolt] [--vs-move leafstorm] [--evaluator X.pev]

For each recorded world of the decision prints
  1. the opponent's whole team (hp, item, moves, status) -- what a KO brings in;
  2. a brute-force one-turn payoff matrix (our moves x their moves): the expected stock
     evaluation after the turn (all chance branches), and, where the opponent's active faints,
     the value after its best forced replacement (depth 2, replacement chosen adversarially);
  3. the MCTS root statistics (visits, mean score) and joint matrix for the same world.
Evaluations are side one's view; the MCTS score is sigmoid-ish of (eval - root eval).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replay_log as rl  # noqa: E402
from poke_engine import State, generate_instructions, monte_carlo_tree_search
from poke_engine import evaluate as stock_evaluate  # noqa: E402


def after(state, instrs):
    return state.apply_instructions(instrs)


def alive_switches(side):
    return [p.id.lower() for i, p in enumerate(side.pokemon)
            if i != int(side.active_index) and p.hp > 0]


def turn_value(state, mine, theirs, depth2=True):
    """(expected eval after one turn, expected eval after the opponent's best replacement,
    P(opp active faints))."""
    mine = mine.removeprefix("switch ")
    theirs = theirs.removeprefix("switch ")
    outs = generate_instructions(state, mine, theirs)
    total = tot2 = pko = 0.0
    for o in outs:
        p = o.percentage / 100.0
        s = state.apply_instructions(o)
        v = stock_evaluate(s)
        total += p * v
        opp = s.side_two
        if depth2 and opp.pokemon[int(opp.active_index)].hp <= 0 and any(
            q.hp > 0 for q in opp.pokemon
        ):
            pko += p
            best = None
            for sw in alive_switches(opp):
                for o2 in generate_instructions(s, "none", sw):
                    pass
                v2 = sum(o2.percentage / 100.0 * stock_evaluate(s.apply_instructions(o2))
                         for o2 in generate_instructions(s, "none", sw))
                best = v2 if best is None else min(best, v2)
            tot2 += p * (best if best is not None else v)
        else:
            tot2 += p * v
    return total, tot2, pko


def print_team(state):
    side = state.side_two
    for i, p in enumerate(side.pokemon):
        mark = "*" if i == int(side.active_index) else " "
        print(f"   {mark}{p.id:14s} hp={p.hp:4d}/{p.maxhp:4d} st={p.status} item={p.item} "
              f"ab={p.ability} tera={p.tera_type} moves={[m.id for m in p.moves]}")


def drop_mon(state, species: str):
    """Counterfactual: the state with opponent ``species`` fainted (hp 0)."""
    raw = state.to_string().split("/")
    mons = raw[1].split("=")
    for k, text in enumerate(mons):
        fields = text.split(",")
        if len(fields) > 7 and fields[0].upper() == species.upper():
            fields[6] = "0"
        mons[k] = ",".join(fields)
    raw[1] = "=".join(mons)
    return State.from_string("/".join(raw))


def maximin(matrix, rows, cols):
    """Pure maximin and (if scipy) the mixed value of a payoff matrix for the row player."""
    pure = max((min(matrix[r][c] for c in cols), r) for r in rows)
    try:
        import numpy as np
        from scipy.optimize import linprog

        a = np.array([[matrix[r][c] for c in cols] for r in rows])
        shift = a.min()
        a = a - shift + 1.0
        res = linprog(np.ones(len(rows)), A_ub=-a.T, b_ub=-np.ones(len(cols)),
                      bounds=[(0, None)] * len(rows))
        v = 1.0 / res.x.sum()
        strat = {rows[i]: round(float(res.x[i] * v), 3) for i in range(len(rows)) if res.x[i] * v > 0.005}
        return pure, v + shift - 1.0, strat
    except Exception:
        return pure, None, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--from-record", required=True)
    ap.add_argument("--turn", type=int, required=True)
    ap.add_argument("--time-ms", type=int, default=300)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--evaluator", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-mcts", action="store_true")
    ap.add_argument("--drop", action="append", default=[], metavar="SPECIES",
                    help="counterfactual: also search the world with this opponent mon fainted "
                         "(does the KO stop being undervalued without that counter?)")
    args = ap.parse_args()

    found = rl.find_decision(args.from_record, args.turn, Path(args.log).stem)
    if found is None:
        raise SystemExit("no recorded decision")
    _, _, decision = found
    for wi, world in enumerate(decision["worlds"][: args.worlds]):
        state = State.from_string(world["state"])
        print(f"\n=== world {wi}: root eval {stock_evaluate(state):.1f}")
        rl.describe_state(state)
        print("  opponent team:")
        print_team(state)
        if args.no_mcts:
            res = None
        else:
            kw = {"return_joint": True}
            if args.evaluator:
                from showdown_bot.players._worker import get_evaluator

                kw["evaluator"] = get_evaluator(args.evaluator, args.scale)
            res = monte_carlo_tree_search(state, args.time_ms, threads=1, **kw)
        mine = [o.move_choice for o in res.side_one] if res else [m for m in
                [x.id.lower() for x in state.side_one.pokemon[int(state.side_one.active_index)].moves]]
        theirs = [o.move_choice for o in res.side_two] if res else []
        if not theirs:
            theirs = [x.id for x in state.side_two.pokemon[int(state.side_two.active_index)].moves]
        d1 = {m: {} for m in mine}
        d2 = {m: {} for m in mine}
        ko = {m: {} for m in mine}
        for m in mine:
            for t in theirs:
                try:
                    d1[m][t], d2[m][t], ko[m][t] = turn_value(state, m, t)
                except BaseException as e:  # illegal pairing / engine panic
                    print(f"   !! {m} vs {t}: {e!r}")
                    d1[m][t] = d2[m][t] = 0.0
                    ko[m][t] = 0.0
        print("  exact 1-turn: E[eval] depth1 | after opp best replacement (depth2) | P(KO)")
        for name, mat in (("depth1", d1), ("depth2", d2)):
            print(f"   [{name}] cols: {theirs}")
            for m in mine:
                print(f"     {m:14s} " + " ".join(f"{d1[m][t] if name == 'depth1' else d2[m][t]:8.1f}" for t in theirs)
                      + "   KO: " + " ".join(f"{ko[m][t]:.2f}" for t in theirs))
            pure, mixed, strat = maximin(mat, mine, theirs)
            print(f"     maximin pure={pure[0]:.1f} via {pure[1]}; Nash value={mixed}; mix={strat}")
        for species in args.drop:
            alt = monte_carlo_tree_search(drop_mon(state, species), args.time_ms, threads=1)
            tot = max(sum(o.visits for o in alt.side_one), 1)
            print(f"  without {species}: " + "  ".join(
                f"{o.move_choice}={o.visits / tot:.2f}/{o.total_score / max(o.visits, 1):.3f}"
                for o in alt.side_one if o.visits / tot > 0.01))
        if res:
            print(f"  MCTS ({args.time_ms} ms): total visits {res.total_visits}")
            for o in res.side_one:
                print(f"     {o.move_choice:14s} visits={o.visits:6d} mean={o.total_score / max(o.visits, 1):.3f}")
            cells = sorted(res.joint, key=lambda c: (c[0], c[1]))
            k2 = len(res.side_two)
            names2 = [o.move_choice for o in res.side_two]
            print("   joint (our row x their col): visits/mean")
            print("     " + " ".join(f"{n[:9]:>12s}" for n in names2))
            by = {(i, j): (v, m) for i, j, v, m in cells}
            for i, o in enumerate(res.side_one):
                print(f"     {o.move_choice[:12]:12s} " + " ".join(
                    f"{by[(i, j)][0]:5d}/{by[(i, j)][1]:.2f}" if (i, j) in by else f"{'-':>10s}"
                    for j in range(k2)))


if __name__ == "__main__":
    main()
