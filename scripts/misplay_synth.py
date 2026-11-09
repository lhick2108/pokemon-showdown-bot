"""Synthetic candidates for misplay categories the ladder logs hardly contain (Encore, Palafin pivot).

    python scripts/misplay_synth.py --out cands_synth.json

Writes candidates in the ``misplay_suite.py mine`` format.  Nothing here is evidence: every
candidate still has to pass ``misplay_suite.py verify`` / ``playout`` before it is kept, and
most are dropped (the position generator is mechanical: it varies species and boosts and lets
the played-out check decide, as scripts/puzzle_gen did for the older themes).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import puzzles as P  # noqa: E402

LAST_MOVE_IDX = 21  # position of ``last_used_move`` in a side's tail (after the six Pokemon)


def with_last_move(side: str, index: int) -> str:
    parts = side.split("=")
    parts[6 + LAST_MOVE_IDX] = f"move:{index}"
    return "=".join(parts)


# foe, its moves (index 0 = the move it just used), boosts after that move, hp, item
SETUPPERS = [
    ("DRAGONITE", ("dragondance", "outrage", "earthquake", "extremespeed"), {"atk": 1, "spe": 1}),
    ("GARCHOMP", ("swordsdance", "earthquake", "outrage", "stoneedge"), {"atk": 2}),
    ("SCIZOR", ("swordsdance", "bulletpunch", "knockoff", "roost"), {"atk": 2}),
    ("GYARADOS", ("dragondance", "waterfall", "earthquake", "icefang"), {"atk": 1, "spe": 1}),
    ("SALAMENCE", ("dragondance", "outrage", "earthquake", "roost"), {"atk": 1, "spe": 1}),
    ("HYDREIGON", ("nastyplot", "darkpulse", "dracometeor", "flashcannon"), {"spa": 2}),
    ("CLEFABLE", ("calmmind", "moonblast", "softboiled", "flamethrower"), {"spa": 1, "spd": 1}),
]
RECOVERERS = [
    ("BLISSEY", ("softboiled", "seismictoss", "thunderwave", "toxic"), {}),
    ("TOXAPEX", ("recover", "liquidation", "toxic", "haze"), {}),
    ("SLOWBROGALAR", ("slackoff", "scald", "psychic", "toxic"), {}),
]
# our Encore users: moves (Encore first), item
ENCORERS = [
    ("SABLEYE", ("encore", "willowisp", "knockoff", "recover")),
    ("CLEFABLE", ("encore", "moonblast", "softboiled", "calmmind")),
    ("MEW", ("encore", "psychic", "willowisp", "softboiled")),
    ("INCINEROAR", ("encore", "knockoff", "uturn", "flareblitz")),
    ("GRIMMSNARL", ("encore", "spiritbreak", "knockoff", "thunderwave")),
]
BENCH = ["CORVIKNIGHT", "KINGAMBIT", "GHOLDENGO", "LANDORUSTHERIAN", "TOXAPEX", "GARCHOMP", "ROTOM"]


def encore_candidates(rng: random.Random, n: int) -> list[dict]:
    out = []
    pairs = list(itertools.product(ENCORERS, SETUPPERS + RECOVERERS))
    rng.shuffle(pairs)
    for (me, mm), (foe, fm, boosts) in pairs:
        if len(out) >= n:
            break
        hp = rng.choice([1.0, 0.85, 0.6])
        ours = [P.mon(me, hp=rng.choice([1.0, 0.8]), moves=list(mm), item="LEFTOVERS")]
        for b in rng.sample(BENCH, 2):
            ours.append(P.mon(b, hp=rng.choice([1.0, 0.7])))
        theirs = [P.mon(foe, hp=hp, moves=list(fm), item="LEFTOVERS")]
        for b in rng.sample(BENCH, 2):
            theirs.append(P.mon(b, hp=rng.choice([1.0, 0.7])))
        state = P.state(P.side(ours), with_last_move(P.side(theirs, boosts=boosts), 0))
        out.append({"cat": "encore", "accept": ["encore"], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": "encore"},
                    "states": [state],
                    "desc": f"{me} (Encore) vs {foe} {hp:.0%}, which just used {fm[0]}"})
    return out


PALAFIN_FOES = ["TOXAPEX", "CORVIKNIGHT", "GHOLDENGO", "LANDORUSTHERIAN", "ROTOM", "KINGAMBIT", "GARCHOMP"]


def pivot_candidates(rng: random.Random, n: int) -> list[dict]:
    out = []
    for foe in rng.sample(PALAFIN_FOES, len(PALAFIN_FOES)) * 3:
        if len(out) >= n:
            break
        hp = rng.choice([1.0, 0.8, 0.55])
        ours = [P.mon("PALAFIN", hp=hp, moves=["flipturn", "jetpunch", "wavecrash", "closecombat"], item="CHOICEBAND"
                      if rng.random() < 0.0 else "LEFTOVERS")]
        for b in rng.sample(BENCH, 2):
            ours.append(P.mon(b, hp=rng.choice([1.0, 0.8])))
        theirs = [P.mon(foe, hp=rng.choice([1.0, 0.7]))]
        for b in rng.sample(BENCH, 2):
            theirs.append(P.mon(b, hp=rng.choice([1.0, 0.7])))
        state = P.state(P.side(ours), P.side(theirs))
        out.append({"cat": "pivot", "accept": ["flipturn"], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": "wavecrash"},
                    "states": [state], "desc": f"PALAFIN {hp:.0%} (Zero) vs {foe}"})
    return out


def asleep(mon: str, turns: int = 1) -> str:
    f = mon.split(",")
    f[18], f[19] = "SLEEP", str(turns)
    return ",".join(f)


# (foe, moves) that put a Pokemon to sleep
SLEEPERS = [
    ("BRELOOM", ("spore", "machpunch", "bulletseed", "rocktomb")),
    ("LILLIGANT", ("sleeppowder", "quiverdance", "gigadrain", "hurricane")),
    ("HYPNO", ("hypnosis", "psychic", "foulplay", "thunderwave")),
    ("VIVILLONGARDEN", ("sleeppowder", "hurricane", "quiverdance", "energyball")),
]
SLEEP_VICTIMS = [("GARCHOMP", ("earthquake", "outrage", "stoneedge", "swordsdance")),
                 ("DRAGONITE", ("dragondance", "outrage", "earthquake", "extremespeed")),
                 ("GHOLDENGO", ("makeitrain", "shadowball", "nastyplot", "recover")),
                 ("KINGAMBIT", ("kowtowcleave", "suckerpunch", "ironhead", "swordsdance"))]
SLEEP_BENCH = ["CORVIKNIGHT", "TOXAPEX", "LANDORUSTHERIAN", "ROTOM", "BLISSEY"]


def sleep_candidates(rng: random.Random, n: int) -> list[dict]:
    """Our sleeping, nearly dead Pokemon faces a hit it cannot survive; the foe still holds a
    sleep move.  Switching it out keeps it alive and asleep (Sleep Clause shield), staying
    lets it faint and frees the foe to sleep another of ours."""
    out = []
    combos = list(itertools.product(SLEEPERS, SLEEP_VICTIMS))
    rng.shuffle(combos)
    for (foe, fm), (me, mm) in combos:
        if len(out) >= n:
            break
        benches = rng.sample(SLEEP_BENCH, 2)
        ours = [asleep(P.mon(me, hp=rng.choice([0.15, 0.25]), moves=list(mm)), 1)]
        ours += [P.mon(b, hp=1.0) for b in benches]
        theirs = [P.mon(foe, hp=1.0, moves=list(fm))]
        theirs += [P.mon(b, hp=rng.choice([1.0, 0.8])) for b in rng.sample(SLEEP_BENCH, 2)]
        out.append({"cat": "sleep_clause", "accept": [b.lower() for b in benches], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": mm[0]},
                    "states": [P.state(P.side(ours), P.side(theirs))],
                    "desc": f"our sleeping {me} 15-25% vs {foe} ({fm[0]}); bench {benches}"})
    return out


ROLE_PHYS = [("GARCHOMP", ("earthquake", "outrage", "stoneedge", "swordsdance")),
             ("KINGAMBIT", ("kowtowcleave", "suckerpunch", "ironhead", "swordsdance")),
             ("SCIZOR", ("swordsdance", "bulletpunch", "knockoff", "roost")),
             ("DRAGONITE", ("dragondance", "outrage", "earthquake", "extremespeed"))]
ROLE_SPEC = [("HYDREIGON", ("nastyplot", "darkpulse", "dracometeor", "flashcannon")),
             ("GARDEVOIR", ("calmmind", "moonblast", "psychic", "focusblast")),
             ("ROTOM", ("voltswitch", "thunderbolt", "willowisp", "painsplit")),
             ("LATIOS", ("dracometeor", "psychic", "calmmind", "recover"))]
WEAK_ACTIVE = [("TOXTRICITY", ("boomburst", "sludgebomb", "overdrive", "shiftgear")),
               ("MEOWSCARADA", ("flowertrick", "knockoff", "uturn", "swordsdance")),
               ("CINDERACE", ("pyroball", "uturn", "gunkshot", "highjumpkick"))]


def role_candidates(rng: random.Random, n: int) -> list[dict]:
    """Our weakened active must leave; the right wall (Blissey vs special, Corviknight vs
    physical) is on the bench next to the wrong one.  Intended: send the wall that matches
    the attacker, do not spend the other as a punching bag."""
    out = []
    combos = list(itertools.product(WEAK_ACTIVE, ROLE_PHYS + ROLE_SPEC))
    rng.shuffle(combos)
    for (me, mm), (foe, fm) in combos:
        if len(out) >= n:
            break
        phys = (foe, fm) in ROLE_PHYS
        right, wrong = ("CORVIKNIGHT", "BLISSEY") if phys else ("BLISSEY", "CORVIKNIGHT")
        extra = [b for b in BENCH if b not in (right, wrong, foe)]
        ours = [P.mon(me, hp=0.3, moves=list(mm)), P.mon(right, hp=1.0), P.mon(wrong, hp=1.0)]
        ours += [P.mon(b, hp=1.0) for b in rng.sample(extra, 2)]
        theirs = [P.mon(foe, hp=1.0, moves=list(fm))] + [P.mon(b, hp=0.8) for b in rng.sample(extra, 3)]
        out.append({"cat": "role", "accept": [right.lower()], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": wrong.lower()},
                    "states": [P.state(P.side(ours), P.side(theirs))],
                    "desc": f"{me} 30% vs {'physical' if phys else 'special'} {foe}: {right} is the matching wall, {wrong} is not"})
    return out


# --------------------------------------------------------------------------- KO / heal loops

# our Pokemon: (species, moves, role).  "tank" has a recovery move; "attacker" has none.
TANKS = [
    ("SLOWBROGALAR", ("slackoff", "shellsidearm", "psychic", "toxic")),
    ("PORYGON2", ("recover", "terablast", "shadowball", "thunderwave")),
    ("ARBOLIVA", ("strengthsap", "energyball", "hypervoice", "earthpower")),
    ("HIPPOWDON", ("slackoff", "earthquake", "stoneedge", "whirlwind")),
    ("MILOTIC", ("recover", "scald", "icebeam", "dragontail")),
    ("REUNICLUS", ("recover", "psychic", "focusblast", "calmmind")),
    ("CLEFABLE", ("softboiled", "moonblast", "flamethrower", "calmmind")),
]
ATTACKERS = [
    ("GARCHOMP", ("earthquake", "stoneedge", "outrage", "swordsdance")),
    ("KINGAMBIT", ("ironhead", "suckerpunch", "kowtowcleave", "swordsdance")),
    ("ORTHWORM", ("bodypress", "heavyslam", "stealthrock", "irondefense")),
    ("HYDREIGON", ("darkpulse", "dracometeor", "flashcannon", "nastyplot")),
    ("LANDORUSTHERIAN", ("earthquake", "stoneedge", "uturn", "stealthrock")),
    ("SCIZOR", ("bulletpunch", "knockoff", "swordsdance", "roost")),
]
FOES = [
    ("GHOLDENGO", ("makeitrain", "shadowball", "nastyplot", "recover")),
    ("CORVIKNIGHT", ("bravebird", "ironhead", "roost", "bulkup")),
    ("DRAGONITE", ("dragondance", "outrage", "earthquake", "roost")),
    ("GARCHOMP", ("earthquake", "outrage", "stoneedge", "swordsdance")),
    ("KINGAMBIT", ("kowtowcleave", "suckerpunch", "ironhead", "swordsdance")),
    ("LATIOS", ("dracometeor", "psychic", "calmmind", "recover")),
    ("TOXAPEX", ("recover", "liquidation", "toxic", "haze")),
    ("HYDREIGON", ("nastyplot", "darkpulse", "dracometeor", "flashcannon")),
]


def _ko_split(state_string: str):
    """(KO moves, non-KO damaging moves) of our active vs an idle foe, by the engine."""
    import poke_engine as pe
    from misplay_suite import DAMAGING_EXCLUDE, active, exp_damage_frac, ko_prob

    st = pe.State.from_string(state_string)
    ko, soft = [], []
    for m in active(st.side_one).moves:
        if m.id == "NONE" or m.id in DAMAGING_EXCLUDE:
            continue
        if ko_prob(pe, st, m.id.lower()) >= 0.9:
            ko.append(m.id.lower())
        elif exp_damage_frac(pe, st, m.id.lower()) >= 0.05:
            soft.append(m.id.lower())
    return ko, soft


def ko_candidates(rng: random.Random, n: int, tank: bool) -> list[dict]:
    """Our Pokemon can KO the weakened foe this turn: the bot is supposed to take it instead of
    healing (tank) or using the slightly weaker attack / setting up (attacker)."""
    out, tries = [], 0
    pool = TANKS if tank else ATTACKERS
    while len(out) < n and tries < 40 * n:
        tries += 1
        me, mm = rng.choice(pool)
        foe, fm = rng.choice(FOES)
        if foe == me:
            continue
        my_hp = rng.choice([0.5, 0.7, 0.9] if tank else [0.6, 1.0])
        f_hp = rng.choice([0.15, 0.25, 0.35, 0.45])
        ours = [P.mon(me, hp=my_hp, moves=list(mm), item="LEFTOVERS")]
        ours += [P.mon(b, hp=rng.choice([1.0, 0.8])) for b in rng.sample(BENCH, 2)]
        theirs = [P.mon(foe, hp=f_hp, moves=list(fm), item="LEFTOVERS")]
        theirs += [P.mon(b, hp=rng.choice([1.0, 0.8])) for b in rng.sample(BENCH, 2)]
        state = P.state(P.side(ours), P.side(theirs))
        ko, soft = _ko_split(state)
        if not ko or (not tank and not soft):
            continue
        out.append({"cat": "heal_loop" if tank else "ko_value", "accept": ko, "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": mm[0] if tank else (soft[0] if soft else mm[0])},
                    "states": [state],
                    "desc": f"{me} {my_hp:.0%} vs {foe} {f_hp:.0%}: {', '.join(ko)} KO"
                            f"{' (alternative: ' + mm[0] + ')' if tank else ' (weaker: ' + ', '.join(soft) + ')'}"})
    return out


def heal_control_candidates(rng: random.Random, n: int) -> list[dict]:
    """Controls: our tank is low and the foe is healthy, so healing IS right (guards against a
    net that learned 'always attack')."""
    out, tries = [], 0
    while len(out) < n and tries < 60 * n:
        tries += 1
        me, mm = rng.choice(TANKS)
        foe, fm = rng.choice(FOES)
        if foe == me:
            continue
        my_hp = rng.choice([0.2, 0.3, 0.4])
        f_hp = rng.choice([0.7, 0.9, 1.0])
        ours = [P.mon(me, hp=my_hp, moves=list(mm), item="LEFTOVERS")]
        ours += [P.mon(b, hp=rng.choice([1.0, 0.8])) for b in rng.sample(BENCH, 2)]
        theirs = [P.mon(foe, hp=f_hp, moves=list(fm), item="LEFTOVERS")]
        theirs += [P.mon(b, hp=rng.choice([1.0, 0.8])) for b in rng.sample(BENCH, 2)]
        state = P.state(P.side(ours), P.side(theirs))
        ko, soft = _ko_split(state)
        if ko:
            continue
        out.append({"cat": "heal_loop", "kind": "control", "accept": [mm[0]], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": mm[0]},
                    "states": [state],
                    "desc": f"(control) {me} {my_hp:.0%} vs healthy {foe} {f_hp:.0%}: no KO, heal with {mm[0]}"})
    return out


# --------------------------------------------------------------------------- Taunt / Toxic

TAUNTERS = [
    ("TORNADUSTHERIAN", ("taunt", "bleakwindstorm", "knockoff", "uturn")),
    ("MIGHTYENA", ("taunt", "crunch", "suckerpunch", "playrough")),
    ("GRENINJA", ("taunt", "hydropump", "icebeam", "uturn")),
]
PASSIVES = [
    ("BLISSEY", ("softboiled", "seismictoss", "thunderwave", "toxic")),
    ("TOXAPEX", ("recover", "liquidation", "toxic", "haze")),
    ("SLOWBROGALAR", ("slackoff", "scald", "psychic", "toxic")),
    ("CLEFABLE", ("calmmind", "moonblast", "softboiled", "flamethrower")),
    ("DRAGONITE", ("dragondance", "outrage", "earthquake", "roost")),
    ("GARCHOMP", ("swordsdance", "earthquake", "outrage", "stoneedge")),
    ("HYDREIGON", ("nastyplot", "darkpulse", "dracometeor", "flashcannon")),
]


def taunt_candidates(rng: random.Random, n: int) -> list[dict]:
    out = []
    pairs = list(itertools.product(TAUNTERS, PASSIVES))
    rng.shuffle(pairs)
    for (me, mm), (foe, fm) in pairs:
        if len(out) >= n:
            break
        ours = [P.mon(me, hp=rng.choice([1.0, 0.8]), moves=list(mm), item="LEFTOVERS")]
        ours += [P.mon(b, hp=1.0) for b in rng.sample(BENCH, 2)]
        theirs = [P.mon(foe, hp=rng.choice([1.0, 0.7]), moves=list(fm), item="LEFTOVERS")]
        theirs += [P.mon(b, hp=1.0) for b in rng.sample(BENCH, 2)]
        out.append({"cat": "taunt", "accept": ["taunt"], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": mm[1]},
                    "states": [P.state(P.side(ours), P.side(theirs))],
                    "desc": f"{me} (Taunt) vs {foe} ({', '.join(m for m in fm if m in RECOVERY_SETUP)})"})
    return out


RECOVERY_SETUP = {"softboiled", "recover", "slackoff", "calmmind", "dragondance", "swordsdance", "nastyplot", "roost",
                  "bulkup"}

TOXICERS = [
    ("TOXAPEX", ("toxic", "recover", "liquidation", "haze")),
    ("BLISSEY", ("toxic", "softboiled", "seismictoss", "thunderwave")),
    ("SLOWBROGALAR", ("toxic", "slackoff", "shellsidearm", "psychic")),
    ("MILOTIC", ("toxic", "recover", "scald", "icebeam")),
]
TOXIC_TARGETS = [
    ("GARCHOMP", ("earthquake", "outrage", "stoneedge", "swordsdance")),
    ("KINGAMBIT", ("kowtowcleave", "suckerpunch", "ironhead", "swordsdance")),
    ("HYDREIGON", ("darkpulse", "dracometeor", "flashcannon", "uturn")),
    ("DRAGONITE", ("extremespeed", "outrage", "earthquake", "dragondance")),
    ("GHOLDENGO", ("makeitrain", "shadowball", "thunderwave", "nastyplot")),
    ("LATIOS", ("dracometeor", "psychic", "calmmind", "recover")),
]


def toxic_candidates(rng: random.Random, n: int) -> list[dict]:
    out = []
    pairs = list(itertools.product(TOXICERS, TOXIC_TARGETS))
    rng.shuffle(pairs)
    for (me, mm), (foe, fm) in pairs:
        if len(out) >= n:
            break
        ours = [P.mon(me, hp=rng.choice([1.0, 0.8]), moves=list(mm), item="LEFTOVERS")]
        ours += [P.mon(b, hp=1.0) for b in rng.sample(BENCH, 2)]
        theirs = [P.mon(foe, hp=1.0, moves=list(fm), item="LEFTOVERS")]
        theirs += [P.mon(b, hp=1.0) for b in rng.sample(BENCH, 2)]
        out.append({"cat": "toxic", "accept": ["toxic"], "reject": [], "fold_tera": True,
                    "source": {"game": "synthetic", "turn": 0, "chosen": mm[2]},
                    "states": [P.state(P.side(ours), P.side(theirs))],
                    "desc": f"{me} (Toxic) vs healthy {foe}"})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefix", default="s")
    p.add_argument("--encore", type=int, default=40)
    p.add_argument("--pivot", type=int, default=16)
    p.add_argument("--sleep", type=int, default=12)
    p.add_argument("--ko", type=int, default=0, help="KO-instead-of-weaker-move candidates")
    p.add_argument("--heal", type=int, default=0, help="KO-instead-of-heal candidates")
    p.add_argument("--heal-control", type=int, default=0)
    p.add_argument("--taunt", type=int, default=0)
    p.add_argument("--toxic", type=int, default=0)
    p.add_argument("--role", type=int, default=12)
    a = p.parse_args(argv)
    rng = random.Random(a.seed)
    cands = (encore_candidates(rng, a.encore) + pivot_candidates(rng, a.pivot)
             + sleep_candidates(rng, a.sleep) + role_candidates(rng, a.role)
             + ko_candidates(rng, a.ko, tank=False) + ko_candidates(rng, a.heal, tank=True)
             + taunt_candidates(rng, a.taunt) + toxic_candidates(rng, a.toxic)
             + heal_control_candidates(rng, a.heal_control))
    for i, c in enumerate(cands):
        c["id"] = f"{a.prefix}{i:03d}"
    Path(a.out).write_text(json.dumps(cands, indent=1))
    print(len(cands), "candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
