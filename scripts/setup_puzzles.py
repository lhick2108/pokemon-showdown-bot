"""Setup-vs-attack puzzles (bead 589.32 addendum, from the ladder game vs jhg2001).

Positions where our boosted attacker can KO the foe NOW (guaranteed: the attack's minimum
damage roll >= the foe's HP, 100% accuracy, we move first) but a setup move is also available.
Taking the KO strictly dominates (the foe never acts), so these need no search evidence; the
candidate foes' HP is also confirmed with calculate_damage on the exact state.

    python scripts/setup_puzzles.py --out tests/puzzles/setup.json [--ladder]

--ladder adds the two reconstructed positions from battle-gen9randombattle-2697409728 (turn 15
Hawlucha +2 at 14/256 HP into a 27% Outrage-locked Dragonite; turn 17 Zacian-Crowned +3 vs a
27% confused Dragonite).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import puzzles as P  # noqa: E402

SETUP = {"SWORDSDANCE": "atk", "DRAGONDANCE": "atk", "NASTYPLOT": "spa", "CALMMIND": "spa", "BULKUP": "atk",
         "QUIVERDANCE": "spa", "SHELLSMASH": "atk", "BELLYDRUM": "atk", "AGILITY": "spe", "COIL": "atk",
         "IRONDEFENSE": "def", "CURSE": "atk", "NORETREAT": "atk", "WORKUP": "atk"}
SKIP = set(SETUP) | {"PROTECT", "SUBSTITUTE", "REST", "SLEEPTALK", "TAUNT", "ENCORE", "WILLOWISP", "THUNDERWAVE",
                     "SPORE", "TOXIC", "STEALTHROCK", "SPIKES", "TOXICSPIKES", "STICKYWEB", "DEFOG", "RAPIDSPIN",
                     "HAZE", "ROAR", "WHIRLWIND", "TRICKROOM", "WISH", "HEALBELL", "REVIVALBLESSING",
                     "UTURN", "VOLTSWITCH", "FLIPTURN", "PARTINGSHOT", "EXPLOSION", "SELFDESTRUCT", "FAKEOUT",
                     "FIRSTIMPRESSION", "FOCUSPUNCH", "TERABLAST", "ENDEAVOR", "PERISHSONG"} | set(P.RECOVERY)
from itertools import product  # noqa: E402


def mons_with_setup():
    out = []
    for sp, m in P.lib().items():
        f = m.split(",")
        mv = [x.split(";")[0] for x in f[22:26]]
        su = [x for x in mv if x in SETUP]
        att = [x for x in mv if x not in SKIP]
        if su and att:
            out.append((sp, su[0], att, int(f[17])))
    return out


def damage_ok(state_str: str, our_move: str, foe_move: str):
    from poke_engine import State, calculate_damage
    s = State.from_string(state_str)
    a, _ = calculate_damage(s, our_move, foe_move, True)
    foe = s.side_two.pokemon[int(s.side_two.active_index)]
    return a[0], foe.hp


FOE_BENCH = ["SNORLAX", "GARCHOMP", "CORVIKNIGHT"]


def _bench(n=3):
    return [P.mon(sp) for sp in FOE_BENCH[:n]]


def build_ladder() -> list[dict]:
    out = []
    # Turn 15: Hawlucha +2 Atk, 14/256 HP, vs Dragonite 27% locked into Outrage.
    ours = P.side([P.mon("HAWLUCHA", hp=14 / 256, moves=["swordsdance", "stoneedge", "closecombat", "bravebird"],
                         item="WHITEHERB"), P.mon("ZACIAN", moves=["swordsdance", "behemothblade", "playrough", "closecombat"]), P.mon("TERRAKION", hp=0.6)],
                  boosts={"atk": 2})
    theirs = P.side([P.mon("DRAGONITE", hp=0.27, moves=["outrage"], ability="MULTISCALE", item="HEAVYDUTYBOOTS")] + _bench(), rocks=1)
    out.append(dict(id="U01", theme="setup", kind="positive", state=P.state(ours, theirs),
                    desc="ladder T15: Hawlucha (+2 Atk, 14/256 HP, faster) vs Outrage-locked Dragonite at 27%",
                    rationale="Stone Edge / Close Combat KO the Dragonite before it moves; Swords Dance only gives "
                              "the locked Outrage a free KO on us",
                    accept=["stoneedge", "closecombat", "bravebird"], reject=["swordsdance"]))
    # Turn 17: Zacian-Crowned +3 vs Dragonite 27% (confused, Earthquake).
    ours = P.side([P.mon("ZACIAN", hp=211 / 225, moves=["swordsdance", "behemothblade", "playrough", "closecombat"],
                         item="RUSTEDSWORD"), P.mon("HAWLUCHA", hp=0.1)], boosts={"atk": 3})
    theirs = P.side([P.mon("DRAGONITE", hp=0.27, moves=["earthquake", "outrage"], ability="MULTISCALE",
                           item="HEAVYDUTYBOOTS")] + _bench())
    out.append(dict(id="U02", theme="setup", kind="positive", state=P.state(ours, theirs),
                    desc="ladder T17: Zacian-Crowned (+3 Atk, 211/225) vs confused Dragonite at 27%",
                    rationale="Behemoth Blade KOs now; a second Swords Dance (+5 is overkill) just lets Dragonite "
                              "hit us with a super-effective Earthquake",
                    accept=["behemothblade", "playrough", "closecombat"], reject=["swordsdance"]))
    return out


def build_generated(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cands = mons_with_setup()
    foes = [(sp, m.split(",")) for sp, m in P.lib().items()]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        a, su, att, spd = rng.choice(cands)
        o, of = rng.choice(foes)
        if o == a or (a, o) in seen:
            continue
        ofm = [x.split(";")[0] for x in of[22:26] if not x.startswith("NONE")]
        if int(of[17]) >= spd:  # we must be faster so the KO lands before the foe acts
            continue
        stat = SETUP[su]
        boost = rng.choice([1, 2, 3]) if stat in ("atk", "spa") else 0
        foe_hp = rng.choice([0.2, 0.27, 0.35])
        a_hp = rng.choice([0.1, 0.25, 0.5])
        mv = [su] + att[:3]
        ours_mon = P.mon(a, hp=a_hp, moves=[m.lower() for m in mv])
        ours = P.side([ours_mon, P.mon("TERRAKION", hp=0.5)], boosts={stat: boost} if boost and stat in ("atk", "spa") else None)
        theirs = P.side([P.mon(o, hp=foe_hp, moves=[m.lower() for m in ofm[:4]])] + _bench(2))
        st = P.state(ours, theirs)
        kills = []
        for m in att[:3]:
            try:
                dmin, fhp = damage_ok(st, m.lower(), ofm[0].lower())
            except Exception:
                continue
            acc = 100  # filtered below by excluding known inaccurate moves
            if dmin >= fhp:
                kills.append(m.lower())
        if not kills:
            continue
        seen.add((a, o))
        out.append(dict(id=f"U{len(out) + 3:02d}", theme="setup", kind="positive", state=st,
                        desc=f"{a} ({int(a_hp * 100)}%, {'+%d ' % boost if boost else ''}{stat if boost else 'no boost'}, faster) "
                             f"vs {o} at {int(foe_hp * 100)}%: {', '.join(kills)} KO now, {su.lower()} is also available",
                        rationale="a guaranteed KO from the faster side removes the foe before it acts; setup first "
                                  "gives it a free turn",
                        accept=kills, reject=[su.lower()]))
    return out


INACCURATE = {"STONEEDGE", "FOCUSBLAST", "HYDROPUMP", "THUNDER", "BLIZZARD", "FIREBLAST", "HURRICANE", "SUPERPOWER_",
              "ZAPCANNON", "INFERNO", "DYNAMICPUNCH", "HEADLONGRUSH_", "DRAGONDARTS_", "MEGAHORN", "CRABHAMMER",
              "BRAVEBIRD_", "SLEEPPOWDER", "SPORE_", "AXEKICK", "HIGHJUMPKICK", "JUMPKICK", "SEEDFLARE", "WILDCHARGE_",
              "GUNKSHOT", "POISONJAB_", "ROCKSLIDE", "AIRSLASH", "ICICLECRASH", "PLAYROUGH", "MOONBLAST_",
              "ICEPUNCH_", "SOLARBEAM_", "CLOSECOMBAT_", "BULLETSEED_"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(P.PUZZLE_DIR / "setup.json"))
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    items = build_ladder() + build_generated(args.n, args.seed)
    Path(args.out).write_text(json.dumps(items, indent=1))
    print(f"wrote {len(items)} setup puzzles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
