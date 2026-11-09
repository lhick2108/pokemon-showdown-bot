"""One-ply evaluation probe: what do the evaluators say about the two lines of a puzzle?

For every puzzle, apply (our candidate first action) x (the foe's greedy reply) with the
engine's own turn rules, then score the resulting positions with the stock evaluate() and
the learned net (each branch weighted by its probability).  Both numbers are
"side one's advantage"; a candidate that the evaluator ranks above the intended answer
shows where a shallow search is misled.

    python scripts/puzzle_eval_probe.py [--only S01,S02]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import puzzles as P  # noqa: E402


def lines(pz, evaluator, rng):
    import poke_engine as pe
    st = pe.State.from_string(pz["state"])
    P.FORCED_PREF[:] = pz.get("pref", [])
    foe = P.policy_greedy(pe, st, 2, rng)
    rows = {}
    for a in P._legal(st, 1):
        total_stock = total_net = 0.0
        ins = pe.generate_instructions(st, a, foe)
        for i in ins:
            after = st.apply_instructions(i)
            # a fainted active mon is replaced for free by our preferred switch-in, as in play
            if P._must_switch(after.side_one) and P._alive(after.side_one):
                legal = P._legal(after, 1)
                pref = [p for p in P.FORCED_PREF if p in legal]
                nxt = pref[0] if pref else legal[0]
                after2 = after.apply_instructions(pe.generate_instructions(after, nxt, "none")[0])
                after = after2
            w = i.percentage / 100.0
            total_stock += w * pe.evaluate(after)
            if evaluator is not None:
                total_net += w * evaluator.evaluate(after)
        rows[a] = (total_stock, total_net if evaluator is not None else None)
    return foe, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--theme", default="")
    args = ap.parse_args()
    import poke_engine as pe
    ev = pe.Evaluator.load(str(P.MODEL), leaf_scale=P.MODEL_SCALE) if P.MODEL.is_file() else None
    rng = random.Random(0)
    for pz in P.load_puzzles():
        if args.only and pz["id"] not in args.only.split(","):
            continue
        if args.theme and pz["theme"] != args.theme:
            continue
        foe, rows = lines(pz, ev, rng)
        acc = set(pz["accept"])
        print(f"{pz['id']} {pz['kind']} foe={foe} accept={sorted(acc)}")
        for a, (s, n) in sorted(rows.items(), key=lambda kv: -kv[1][0]):
            tag = "*" if P.norm(a) in acc else " "
            print(f"   {tag} {a:<18} stock={s:8.1f}" + (f" net={n:8.3f}" if n is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
