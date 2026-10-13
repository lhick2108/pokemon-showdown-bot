"""Build and finalize the puzzle files from search-rollout evidence (bead 589.32).

Pipeline (each step is a mechanical filter; none looks at what the bot plays):

1. candidates: random positions from the mon library (scripts/puzzle_defs.py for the stall
   family, the generator described in scripts/build_sacrifice.py for the sacrifice family);
2. stage 1: ``puzzles.py searchroll --pool`` plays every first action out with MCTS vs MCTS
   (20 games per action) and ``stage2-pool`` keeps the ones where one side of the intended
   dichotomy (recovering vs not, staying vs switching) leads by >= 0.25;
3. stage 2: the same positions are re-rolled with a fresh sample (more games) and
   ``finalize`` keeps those where the intended group still leads by >= 0.20; accept sets are
   recomputed from the stage 2 values.

    python scripts/finalize_puzzles.py stage2-pool --sac pool_sac_out.json --stall a.json b.json --out pool2.json
    python scripts/puzzles.py searchroll --pool pool2.json --out pool2_out.json --n 50 --ms 30
    python scripts/finalize_puzzles.py finalize pool2_out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import puzzles as P  # noqa: E402

REC = {m.lower() for m in P.RECOVERY}


def _group(theme: str, entry: dict, values: dict) -> tuple[dict, dict]:
    """(values of the 'positive' group, values of the other group)."""
    if theme == "stall":
        pos = {a: v for a, v in values.items() if a in REC}
    else:
        pos = {a: v for a, v in values.items() if a not in entry["pref"]}
    return pos, {a: v for a, v in values.items() if a not in pos}


def stage2_pool(args) -> int:
    out = []
    thr = 0.25
    if args.sac:
        for x in json.loads(Path(args.sac).read_text()):
            pos, oth = _group("sacrifice", x, x["verify_search"]["values"])
            gap = max(pos.values()) - max(oth.values())
            if abs(gap) >= thr:
                out.append(dict(x, kind="positive" if gap > 0 else "control", stage1_gap=round(gap, 2)))
    for f in args.stall or []:
        for x in json.loads(Path(f).read_text()):
            pos, oth = _group("stall", x, x["verify_search"]["values"])
            if not pos or not oth:
                continue
            gap = max(pos.values()) - max(oth.values())
            if abs(gap) >= thr:
                out.append(dict(x, kind="positive" if gap > 0 else "control", stage1_gap=round(gap, 2)))
    for i, x in enumerate(out):
        x["id"] = f"{x['theme'][:2].upper()}{i:03d}"
        x.pop("verify_search", None)
        x["accept"], x["reject"] = [], []  # filled in by finalize
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{len(out)} candidates: " + ", ".join(
        f"{t}/{k} {sum(1 for x in out if x['theme'] == t and x['kind'] == k)}"
        for t in ("stall", "sacrifice") for k in ("positive", "control")))
    return 0


def _describe(x: dict) -> tuple[str, str]:
    if x["theme"] == "stall":
        hp = int(round(x["hp"] * 100))
        desc = (f"{x['r']} ({hp}%: {', '.join(x['rm'])}) vs {x['item'].title()} {x['o']} ({', '.join(x['om'])})")
        if x["kind"] == "positive":
            return (desc + "; each hit is under 50%, so recovery outheals the hits and the foe's Life Orb "
                    "recoil does the rest",
                    "keep recovering: the foe cannot break through and loses 10% per attack")
        return (desc + "; the foe hits for over 50%",
                "recovery cannot outheal the hits, so keep-recovering loses: attack or switch")
    desc = (f"{x['a']} ({int(round(x['hp_a'] * 100))}%: {', '.join(m.lower() for m in x['am'])}) is slower than and "
            f"sure to be KOed by {x['o']} ({', '.join(m.lower() for m in x['om'])}); bench {x['b']}, {x['c']}; "
            f"foe also has {', '.join(x['extra'])}")
    if x["kind"] == "positive":
        return desc, ("the low mon dies anyway: staying in to attack/set up/status gives a free switch-in "
                      "instead of taking the hit on a healthy mon")
    return desc, "do not sacrifice by reflex: keeping the low mon alive (switching) is clearly better here"


def finalize(args) -> int:
    keep = {"stall": {"positive": [], "control": []}, "sacrifice": {"positive": [], "control": []}}
    for x in json.loads(Path(args.pool_out).read_text()):
        vals = x["verify_search"]["values"]
        pos, oth = _group(x["theme"], x, vals)
        if not pos or not oth:
            continue
        gp, go = max(pos.values()), max(oth.values())
        if x["kind"] == "positive":
            good, bad, gap = pos, oth, gp - go
        else:
            good, bad, gap = oth, pos, go - gp
        if gap < args.min_gap:
            continue
        best = max(good.values())
        x["accept"] = sorted(a for a, v in good.items() if v >= max(bad.values()) + 0.10)
        x["reject"] = sorted(bad)
        x["desc"], x["rationale"] = _describe(x)
        x["verify_search"]["gap"] = round(gap, 2)
        keep[x["theme"]][x["kind"]].append((gap, x))
    for theme, groups in keep.items():
        entries = []
        for kind, items in groups.items():
            items.sort(key=lambda t: -t[0])
            for i, (gap, x) in enumerate(items[:args.max_per_class], 1):
                x["id"] = f"{'T' if theme == 'stall' and kind == 'positive' else 'C' if theme == 'stall' else 'S' if kind == 'positive' else 'K'}{i:02d}"
                entries.append({k: x[k] for k in ("id", "theme", "kind", "desc", "rationale", "state", "accept", "reject",
                                                  "verify_search") + (("pref",) if "pref" in x else ())})
        (P.PUZZLE_DIR / f"{theme}.json").write_text(json.dumps(entries, indent=1))
        print(theme, {k: len(v) for k, v in groups.items()}, "->", len(entries), "written")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stage2-pool")
    a.add_argument("--sac")
    a.add_argument("--stall", nargs="*")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=stage2_pool)
    b = sub.add_parser("finalize")
    b.add_argument("pool_out")
    b.add_argument("--min-gap", type=float, default=0.20)
    b.add_argument("--max-per-class", type=int, default=14)
    b.set_defaults(fn=finalize)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
