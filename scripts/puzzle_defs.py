"""Puzzle definitions for bead 589.32 (built into tests/puzzles/*.json by ``puzzles.py build``).

Two themes, each with positives (a known better move) and controls where the obvious
heuristic answer is wrong:

* ``stall``: our mon has a recovery move and the foe holds Life Orb (10% recoil per hit)
  and does under 50% per hit: keep recovering.  Controls: the foe does over 50% per hit
  (stalling loses; attack or switch instead).
* ``sacrifice``: our mon is low and will faint anyway: stay in and use a move (hazard,
  status, chip) and take the free switch-in, instead of switching a healthy mon into a hit.
  Controls: the switch-in is cheap / the low mon is still needed, so switching is right.

None of this is a rule for the player.  Every puzzle was kept only if an independent
rollout (``puzzles.py rollouts``: play the line to the end with fixed simple policies, no
search and no evaluation function) says the intended answer is clearly the best first
action; candidates were generated mechanically and the decisive ones kept, so puzzles do
not encode what the bot happens to play.
"""

from __future__ import annotations

import puzzles as P


def stall_state(r, hp, rm, o, om, item="LIFEORB", obench=(), bench=True, o_hp=1.0):
    ours = [P.mon(r, hp=hp, moves=list(rm), item="LEFTOVERS")]
    if bench:
        ours.append(P.mon("TERRAKION", hp=0.1))
    theirs = [P.mon(o, hp=o_hp, moves=list(om), item=item)]
    for sp, mv in obench:
        theirs.append(P.mon(sp, moves=list(mv), item="LIFEORB"))
    return P.state(P.side(ours), P.side(theirs))


OB_TWO = (("SPECTRIER", ("shadowball", "darkpulse")), ("LUMINEON", ("hydropump", "icebeam")))

# (id, our mon, hp, our moves, foe, foe moves, foe bench)
STALL_POS = [
    ("T01", "TOXAPEX", .35, ("recover", "liquidation"), "LUMINEON", ("alluringvoice", "hydropump", "icebeam"), ()),
    ("T02", "TOXAPEX", .45, ("recover", "liquidation"), "LUMINEON", ("alluringvoice", "hydropump", "icebeam"), ()),
    ("T03", "DUDUNSPARCE", .35, ("roost", "earthquake", "headbutt"), "SPECTRIER", ("shadowball", "darkpulse", "drainingkiss"), ()),
    ("T04", "DUDUNSPARCE", .45, ("roost", "earthquake", "headbutt"), "SPECTRIER", ("shadowball", "darkpulse", "drainingkiss"), ()),
    ("T05", "DUDUNSPARCE", .55, ("roost", "earthquake", "headbutt"), "SPECTRIER", ("shadowball", "darkpulse", "drainingkiss"), ()),
    ("T06", "MILOTIC", .35, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), ()),
    ("T07", "MILOTIC", .45, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), ()),
    ("T08", "MILOTIC", .55, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), ()),
    ("T09", "MILOTIC", .35, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), OB_TWO),
    ("T10", "MILOTIC", .45, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), OB_TWO),
    ("T11", "MILOTIC", .55, ("recover", "icebeam", "scald"), "GRENINJA", ("hydropump", "darkpulse", "icebeam"), OB_TWO),
    ("T12", "NOCTOWL", .35, ("roost", "hypervoice", "hurricane"), "SPECTRIER", ("shadowball", "darkpulse", "drainingkiss"), ()),
    ("T13", "NOCTOWL", .45, ("roost", "hypervoice", "hurricane"), "SPECTRIER", ("shadowball", "darkpulse", "drainingkiss"), ()),
    ("T14", "SINISTCHA", .35, ("strengthsap", "shadowball", "matchagotcha"), "ELECTRODEHISUI", ("thunderbolt", "gigadrain"), ()),
    ("T15", "SWANNA", .35, ("roost", "hydropump", "bravebird"), "DUGTRIOALOLA", ("earthquake", "suckerpunch", "ironhead"), ()),
]

RECOVERY_MOVES = {"recover", "roost", "slackoff", "softboiled", "moonlight", "synthesis", "strengthsap", "morningsun",
                  "milkdrink"}


def stall_positives():
    out = []
    for pid, r, hp, rm, o, om, ob in STALL_POS:
        out.append({
            "id": pid, "theme": "stall", "kind": "positive",
            "desc": f"{r} ({int(hp * 100)}%: {', '.join(rm)}) vs Life Orb {o} ({', '.join(om)}), "
                    f"{'+2 Life Orb foes in the back' if ob else 'alone'}; each hit is under 50%",
            "rationale": "recoil of 10% per attack drains the foe while recovery outheals its hits; "
                         "a turn spent attacking leaves us in range",
            "state": stall_state(r, hp, rm, o, om, obench=ob),
            "accept": [m for m in rm if m in RECOVERY_MOVES],
            "pref": [],
        })
    return out


# controls: foe hits over 50%, stalling loses.  (id, mon, hp, moves, foe, foe moves, accept)
STALL_CTRL = [
    ("C01", "HYDRAPPLE", .5, ("recover", "ficklebeam", "earthpower"), "ZACIAN", ("playrough", "wildcharge", "closecombat"), ("terrakion",)),
    ("C02", "JUMPLUFF", .9, ("strengthsap", "acrobatics"), "MELOETTA", ("closecombat", "tripleaxel", "relicsong"), ("acrobatics",)),
    ("C03", "GOGOAT", .9, ("milkdrink", "earthquake", "hornleech"), "REUNICLUS", ("focusblast", "psychic"), ("earthquake", "hornleech")),
    ("C04", "ALCREMIEMINTCREAM", .5, ("recover", "alluringvoice"), "SCEPTILE", ("leafstorm", "focusblast", "shedtail"), ("terrakion",)),
    ("C05", "HOOH", .7, ("recover", "earthquake", "bravebird"), "REUNICLUS", ("focusblast", "psychic"), ("bravebird",)),
    ("C06", "OGERPONWELLSPRING", .9, ("synthesis", "woodhammer", "ivycudgel"), "MEW", ("closecombat", "psychicfangs"), ("woodhammer", "ivycudgel")),
    ("C07", "HIPPOWDON", .9, ("slackoff", "stoneedge", "earthquake"), "AZELF", ("psychic", "fireblast"), ("terrakion",)),
    ("C08", "ILLUMISE", .7, ("roost", "bugbuzz"), "VICTREEBEL", ("poisonjab", "suckerpunch", "powerwhip"), ("bugbuzz",)),
    ("C09", "ALCREMIEMINTCREAM", .5, ("recover", "alluringvoice"), "POLIWRATH", ("closecombat", "liquidation", "raindance"), ("alluringvoice",)),
    ("C10", "SLOWKINGGALAR", .7, ("slackoff", "chillyreception", "psychicnoise"), "MEWTWO", ("fireblast", "psystrike", "aurasphere"), ("terrakion",)),
    ("C11", "MOLTRES", .5, ("roost", "scorchingsands", "fireblast"), "ZACIAN", ("playrough", "wildcharge", "closecombat"), ("fireblast",)),
    ("C12", "NOIVERN", .5, ("roost", "flamethrower", "hurricane"), "ELECTRODEHISUI", ("thunderbolt", "gigadrain"), ("terrakion",)),
]


def stall_controls():
    out = []
    for pid, r, hp, rm, o, om, acc in STALL_CTRL:
        out.append({
            "id": pid, "theme": "stall", "kind": "control",
            "desc": f"{r} ({int(hp * 100)}%: {', '.join(rm)}) vs Life Orb {o} ({', '.join(om)}); hits are over 50%",
            "rationale": "recovery cannot outheal the hits, so keep-recovering loses; attack or switch",
            "state": stall_state(r, hp, rm, o, om),
            "accept": list(acc),
            "reject": [m for m in rm if m in RECOVERY_MOVES],
            "pref": [],
        })
    return out


def hits(state_str: str) -> dict:
    """Per foe move: (min, max) damage as a fraction of our active's max HP (we do nothing)."""
    from poke_engine import State, calculate_damage
    s = State.from_string(state_str)
    me = s.side_one.pokemon[int(s.side_one.active_index)]
    foe = s.side_two.pokemon[int(s.side_two.active_index)]
    mymove = [m.id.lower() for m in me.moves if m.id != "NONE"][0]
    out = {}
    for m in foe.moves:
        if m.id == "NONE":
            continue
        _, b = calculate_damage(s, mymove, m.id.lower(), False)
        out[m.id.lower()] = (round(b[0] / me.maxhp, 2), round(b[1] / me.maxhp, 2))
    return out


def all_puzzles():
    out = {"stall": stall_positives() + stall_controls()}
    try:
        import sacrifice_defs
        out["sacrifice"] = sacrifice_defs.puzzles()
    except ImportError:
        pass
    return out
