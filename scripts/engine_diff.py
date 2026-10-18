"""Differential test: Rust engine (poke-engine fork) vs the real Showdown simulator, turn by turn.

    python scripts/engine_diff.py run --games 100 --seed 1 --out scratch/ed.json
    python scripts/engine_diff.py report scratch/ed.json [--md docs/engine-diff-report.md]
    python scripts/engine_diff.py repro 1005:7          # print one turn (battle seed : turn) in detail

``run`` plays seeded gen9randombattle games in the local Showdown simulator (``scripts/engine_diff.js``,
in-process, no server), both sides choosing random legal actions. Before every turn the engine State is
built from the simulator's TRUE full state (both teams, exact stats / HP / PP / boosts / volatiles / side
conditions / weather ...), the same action pair is run through ``generate_instructions`` (plus the forced
switches that follow within the turn), and the simulator's actual result is compared with every engine
branch. A turn "diverges" when NO branch reproduces the simulator's result. See docs/engine-diff.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import poke_engine as pe  # noqa: E402

NODE_DRIVER = REPO / "scripts" / "engine_diff.js"
MAX_BRANCHES = 20000
COMPARABLE = ("match", "pp_only", "diverge")

# ----------------------------------------------------------------------------- id mappings


def to_id(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


STATUS_TO_ENGINE = {"": "none", "brn": "burn", "frz": "freeze", "par": "paralyze", "psn": "poison",
                    "slp": "sleep", "tox": "toxic"}
WEATHER_TO_ENGINE = {"sunnyday": "sun", "raindance": "rain", "sandstorm": "sand", "hail": "hail",
                     "snowscape": "snow", "desolateland": "harshsun", "primordialsea": "heavyrain"}
TERRAINS = {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
BOOSTS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
ENGINE_BOOST_ATTR = {"atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
                     "spd": "special_defense_boost", "spe": "speed_boost", "accuracy": "accuracy_boost",
                     "evasion": "evasion_boost"}

# sim volatile id -> engine volatile id (identity when the names agree; only listed ones are compared)
PERSISTENT_VOLATILES = {
    "aquaring", "attract", "charge", "confusion", "curse", "defensecurl", "disable", "embargo", "encore",
    "flashfire", "focusenergy", "foresight", "gastroacid", "healblock", "imprison", "ingrain", "laserfocus",
    "leechseed", "lockedmove", "magnetrise", "minimize", "miracleeye", "mustrecharge", "nightmare",
    "noretreat", "octolock", "powertrick", "saltcure", "slowstart", "smackdown", "substitute", "syrupbomb",
    "tarshot", "taunt", "telekinesis", "throatchop", "torment", "yawn", "partiallytrapped", "stockpile",
    "uproar", "destinybond", "rage", "autotomize",
}
# sim volatiles that are bookkeeping / turn-scoped / not state worth comparing
IGNORED_VOLATILES = {
    "stall", "choicelock", "flinch", "followme", "helpinghand", "roost", "magiccoat", "snatch", "endure",
    "ragepowder", "electrify", "spotlight", "protect", "banefulbunker", "kingsshield", "spikyshield",
    "silktrap", "burningbulwark", "obstruct", "trapped", "trapper", "gem", "metronome", "fakeout",
    "dragoncheer", "bide", "mimic", "commanded", "commanding", "lockon", "mindreader", "uturnedmon",
    "furycutter", "iceball", "rollout", "partialtrappinglock", "protosynthesisfromsun", "substitutebroken",
    "echoedvoice", "stellar", "ability:", "item:", "bindmove", "glaiverush", "shelltrap", "beakblast",
    "focuspunch", "ragepowder", "allyswitch", "quash", "maxguard", "transform", "supremeoverlord",
}
CHARGING = {"fly", "dig", "dive", "bounce", "phantomforce", "shadowforce", "skydrop", "solarbeam",
            "solarblade", "skullbash", "razorwind", "skyattack", "meteorbeam", "electroshot", "geomancy",
            "freezeshock", "iceburn"}
# volatile-like features whose counters we map by GUESS (semantics inferred, not verified)
# engine-internal volatiles that carry no battle meaning on their own (types are compared directly)
ENGINE_BOOKKEEPING = {"typechange", "sparklingaria"}
GUESS_FEATURES = ("sleep", "rest", "confusion", "encore", "taunt", "lockedmove", "yawn", "slowstart",
                  "toxic", "wish", "weather_turns", "terrain_turns", "screen_turns", "tailwind_turns",
                  "trickroom_turns", "perish", "substitute")

_VOL_OK: dict[str, bool] = {}


def _wheel_supports_sdb22() -> tuple[bool, bool]:
    """(Heal Block / Throat Chop durations, Move.max_pp) available in the loaded poke_engine wheel (>= sdb.22)."""
    dur = mx = False
    try:
        pe.VolatileStatusDurations(healblock=1)
        dur = True
    except TypeError:
        pass
    try:
        pe.Move(id="tackle", max_pp=8)
        mx = True
    except TypeError:
        pass
    return dur, mx


ENGINE_HAS_DURATIONS, ENGINE_HAS_MAX_PP = _wheel_supports_sdb22()


def engine_supports_volatile(name: str) -> bool:
    ok = _VOL_OK.get(name)
    if ok is None:
        try:
            mon = pe.Pokemon(id="pikachu", moves=[pe.Move(id="tackle")])
            s = pe.State(side_one=pe.Side(pokemon=[mon], volatile_statuses={name}), side_two=pe.Side(pokemon=[mon]))
            back = pe.State.from_string(s.to_string())
            ok = name.upper() in {x.upper() for x in back.side_one.volatile_statuses}
        except BaseException:
            ok = False
        _VOL_OK[name] = ok
    return ok


# ----------------------------------------------------------------------------- state building


class Built:
    """Engine State built from a simulator snapshot plus bookkeeping."""

    def __init__(self, state, uid_index, unrep, unknown_ids, guess, conv_errors):
        self.state = state
        self.uid_index = uid_index  # [uid -> engine index] per side
        self.unrep = unrep  # features the engine cannot represent at all (sorted list of strings)
        self.unknown_ids = unknown_ids  # ids the engine does not know (decay to "none")
        self.guess = guess  # features whose counter mapping is a guess
        self.conv_errors = conv_errors  # builder self-check failures (converter bugs)


def sim_volatiles_to_engine(mon: dict, unrep: set, guess: set) -> tuple[set, dict, int]:
    """Engine volatile set + durations + substitute hp for a sim active mon."""
    vols: set[str] = set()
    dur = {"confusion": 0, "encore": 0, "lockedmove": 0, "slowstart": 0, "taunt": 0, "yawn": 0}
    sub_hp = 0
    for vid, st in mon["volatiles"].items():
        st = st or {}
        name = None
        if vid in ("protosynthesis", "quarkdrive"):
            best = st.get("bestStat")
            name = f"{vid}{best}" if best else None
        elif vid == "twoturnmove":
            mv = st.get("move")
            name = mv if mv in CHARGING else None
            if name is None:
                unrep.add(f"volatile:twoturnmove:{mv}")
        elif vid == "perishsong":
            d = st.get("duration")
            name = f"perish{d}" if d in (1, 2, 3, 4) else None
            guess.add("perish")
        elif vid == "transform":
            name = "transformed"
        elif vid in IGNORED_VOLATILES or vid.startswith(("ability:", "item:")):
            continue
        else:
            name = vid
        if name is None:
            continue
        if not engine_supports_volatile(name):
            unrep.add(f"volatile:{name}")
            continue
        vols.add(name)
        if vid == "substitute":
            sub_hp = int(st.get("hp") or 0)
            guess.add("substitute")
        elif vid == "confusion":
            guess.add("confusion")
        elif vid == "encore":
            dur["encore"] = max(0, min(2, 3 - int(st.get("duration", 3))))
            guess.add("encore")
        elif vid == "taunt":
            dur["taunt"] = max(0, min(2, 3 - int(st.get("duration", 3))))
            guess.add("taunt")
        elif vid == "lockedmove":
            td = st.get("trueDuration")  # sim: total lock length 2-3, decremented each residual
            dur["lockedmove"] = max(0, min(2, 3 - int(td))) if td is not None else 0
            guess.add("lockedmove")
        elif vid == "slowstart":
            dur["slowstart"] = max(1, int(st.get("duration", 5)))
            guess.add("slowstart")
        elif vid == "yawn":
            dur["yawn"] = 1
            guess.add("yawn")
        elif vid == "healblock" and ENGINE_HAS_DURATIONS:
            dur["healblock"] = max(1, int(st.get("duration") or 5))
        elif vid == "throatchop" and ENGINE_HAS_DURATIONS:
            dur["throatchop"] = max(1, int(st.get("duration") or 2))
    if mon.get("abilityOnce") and engine_supports_volatile("typechange"):
        vols.add("typechange")
    return vols, dur, sub_hp


def build_pokemon(m: dict, req_moves: list | None, unrep: set, guess: set):
    status = STATUS_TO_ENGINE.get(m["status"], "none")
    ss = m["statusState"]
    sleep_turns = rest_turns = 0
    if status == "sleep":
        t, s0 = ss.get("time"), ss.get("startTime")
        if ss.get("src") == "rest":
            rest_turns = max(1, int(t or 1))
            guess.add("rest")
        else:
            sleep_turns = max(0, min(3, int((s0 or 0) - (t or 0))))
            guess.add("sleep")
    if status == "toxic":
        guess.add("toxic")
    moves = []
    forced = None
    if req_moves is not None and len(req_moves) < len(m["moves"]):
        forced = req_moves[0]["id"] if req_moves else None
    for i, mv in enumerate(m["moves"]):
        disabled = bool(mv["disabled"])
        if req_moves is not None:
            if forced is not None:
                disabled = mv["id"] != forced
            elif i < len(req_moves):
                disabled = bool(req_moves[i]["disabled"])
        mkw = {"max_pp": max(1, min(127, int(mv["maxpp"])))} if ENGINE_HAS_MAX_PP and mv.get("maxpp") else {}
        moves.append(pe.Move(id=to_id(mv["id"]), pp=max(-128, min(127, int(mv["pp"]))), disabled=disabled, **mkw))
    while len(moves) < 4:
        moves.append(pe.Move(id="none", pp=0, disabled=True))
    types = [to_id(t) for t in m["types"] if t != "???"] or ["typeless"]
    base_types = [to_id(t) for t in m["baseTypes"]] or ["typeless"]
    types = (types + ["typeless"])[:2]
    base_types = (base_types + ["typeless"])[:2]
    st = m["stats"]
    kw = dict(
        id=to_id(m["sid"]), level=m["level"], types=tuple(types), base_types=tuple(base_types),
        hp=m["hp"], maxhp=m["maxhp"], ability=to_id(m["ability"]) or "none",
        base_ability=to_id(m["baseAbility"]) or "none", item=to_id(m["item"]) or "none",
        attack=st["atk"], defense=st["def"], special_attack=st["spa"], special_defense=st["spd"],
        speed=st["spe"], status=status, sleep_turns=sleep_turns, rest_turns=rest_turns,
        weight_kg=float(m["weightkg"]), terastallized=bool(m["terastallized"]),
        tera_type=to_id(m["teraType"]) or "typeless", moves=moves, times_attacked=int(m["timesAttacked"]),
    )
    try:
        p = pe.Pokemon(**kw)
    except TypeError:  # older wheels have no times_attacked
        kw.pop("times_attacked")
        p = pe.Pokemon(**kw)
    return p, forced


def build_state(snap: dict, requests: list | None) -> Built:
    """Engine State from a simulator snapshot (``requests``: per-side request info, pre-turn only)."""
    unrep: set[str] = set()
    guess: set[str] = set()
    sides = []
    uid_index = []
    req_forced = [None, None]
    for k, sd in enumerate(snap["sides"]):
        mons = sd["mons"]
        req = requests[k]["moves"] if requests else None
        pokes = []
        for i, m in enumerate(mons):
            p, forced = build_pokemon(m, req if i == 0 else None, unrep, guess)
            if i == 0:
                req_forced[k] = forced
            pokes.append(p)
        uid_index.append({m["uid"]: i for i, m in enumerate(mons)})
        act = mons[0]
        kw: dict = {}
        vols, dur, sub_hp = sim_volatiles_to_engine(act, unrep, guess) if not act["fainted"] else (set(), {}, 0)
        if vols:
            kw["volatile_statuses"] = vols
        if dur:
            kw["volatile_status_durations"] = pe.VolatileStatusDurations(**dur)
        if sub_hp:
            kw["substitute_health"] = sub_hp
        for b in BOOSTS:
            if act["boosts"].get(b):
                kw[ENGINE_BOOST_ATTR[b]] = max(-6, min(6, int(act["boosts"][b])))
        lm = act.get("lastMove")
        ids = [to_id(mv["id"]) for mv in act["moves"]]
        if lm is None or act.get("activeTurns", 1) == 0:
            kw["last_used_move"] = "switch:0"
        elif to_id(lm) in ids:
            kw["last_used_move"] = f"move:{ids.index(to_id(lm))}"
        else:
            kw["last_used_move"] = "move:none"
        sc = {}
        for cid, st in sd["sideConditions"].items():
            st = st or {}
            layers = int(st.get("layers", 1) or 1)
            dur_left = int(st.get("duration", 0) or 0)
            if cid == "stealthrock":
                sc["stealth_rock"] = 1
            elif cid == "spikes":
                sc["spikes"] = layers
            elif cid == "toxicspikes":
                sc["toxic_spikes"] = layers
            elif cid == "stickyweb":
                sc["sticky_web"] = 1
            elif cid in ("reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "mist", "luckychant"):
                key = {"lightscreen": "light_screen", "auroraveil": "aurora_veil", "luckychant": "lucky_chant"}.get(cid, cid)
                sc[key] = max(1, dur_left)
                guess.add("tailwind_turns" if cid == "tailwind" else "screen_turns")
            else:
                unrep.add(f"sidecond:{cid}")
        stage = act["statusState"].get("stage")
        if act["status"] == "tox":
            sc["toxic_count"] = int(stage or 0)
        if "stall" in act["volatiles"]:
            sc["protect"] = int((act["volatiles"]["stall"] or {}).get("counter", 0) or 0)
        kw["side_conditions"] = pe.SideConditions(**sc)
        w = sd["slotConditions"].get("wish")
        if w:
            # Showdown stores startingTurn (not a countdown): the wish lands at the end of turn startingTurn + 2
            left = int(w.get("startingTurn", snap["turn"]) or 0) + 3 - int(snap["turn"])
            kw["wish"] = (max(1, min(2, left)), int(w.get("hp", 0) or 0))
            guess.add("wish")
        for sid in sd["slotConditions"]:
            if sid != "wish":
                unrep.add(f"slotcond:{sid}")
        if req_forced[k] in ("recharge",):
            pass
        sides.append(pe.Side(pokemon=pokes, active_index="0", **kw))
    wk = {}
    w = snap.get("weather") or ""
    if w:
        if w in WEATHER_TO_ENGINE:
            wk["weather"] = WEATHER_TO_ENGINE[w]
            wk["weather_turns_remaining"] = 0 if w in ("desolateland", "primordialsea") else int(snap["weatherDuration"] or 0)
            guess.add("weather_turns")
        else:
            unrep.add(f"weather:{w}")
    t = snap.get("terrain") or ""
    if t:
        if t in TERRAINS:
            wk["terrain"] = t
            wk["terrain_turns_remaining"] = int(snap["terrainDuration"] or 0)
            guess.add("terrain_turns")
        else:
            unrep.add(f"terrain:{t}")
    for pid, st in snap["pseudoWeather"].items():
        if pid == "trickroom":
            wk["trick_room"] = True
            wk["trick_room_turns_remaining"] = int((st or {}).get("duration", 0) or 0)
            guess.add("trickroom_turns")
        elif pid.endswith(("mod", "clause")):
            continue  # format rules, not battle state
        else:
            unrep.add(f"field:{pid}")
    raw = pe.State(side_one=sides[0], side_two=sides[1], **wk)
    state = pe.State.from_string(raw.to_string())
    unknown = find_unknown_ids(raw, state)
    conv = convert_self_check(snap, state)
    return Built(state, uid_index, sorted(unrep), unknown, sorted(guess), conv)


def find_unknown_ids(sent, got) -> list[str]:
    out = []
    for name in ("side_one", "side_two"):
        for a, b in zip(getattr(sent, name).pokemon, getattr(got, name).pokemon):
            if to_id(a.id) != to_id(b.id):
                out.append(f"species:{to_id(a.id)}")
            if to_id(a.ability) != to_id(b.ability):
                out.append(f"ability:{to_id(a.ability)}")
            if to_id(a.item) != to_id(b.item):
                out.append(f"item:{to_id(a.item)}")
            for ma, mb in zip(a.moves, b.moves):
                if ma.id and to_id(ma.id) != to_id(mb.id):
                    out.append(f"move:{to_id(ma.id)}")
    return sorted(set(out))


def convert_self_check(snap: dict, state) -> list[str]:
    """Builder bugs: values that went in but do not come back out (stats, HP, PP, active index)."""
    errs = []
    for k, name in enumerate(("side_one", "side_two")):
        side = getattr(state, name)
        if str(side.active_index) not in ("0", "PokemonIndex.P0") and "0" not in str(side.active_index):
            errs.append("active_index")
        for m, p in zip(snap["sides"][k]["mons"], side.pokemon):
            if p.hp != m["hp"] or p.maxhp != m["maxhp"]:
                errs.append("hp")
            if p.attack != m["stats"]["atk"] or p.speed != m["stats"]["spe"]:
                errs.append("stats")
            for mv, em in zip(m["moves"], p.moves):
                if to_id(em.id) == to_id(mv["id"]) and em.pp != max(-128, min(127, mv["pp"])):
                    errs.append("pp")
    return sorted(set(errs))


# ----------------------------------------------------------------------------- actions


def engine_choice(action: dict | None, built: Built, k: int, req: dict | None, mon0: dict | None = None) -> str:
    if action is None:
        return "none"
    if action["type"] == "switch":
        idx = built.uid_index[k].get(action["uid"])
        side = built.state.side_one if k == 0 else built.state.side_two
        return to_id(side.pokemon[idx].id)  # generate_instructions takes the bare name (no "switch " prefix)
    mid = to_id(action["id"])
    return mid + "-tera" if action.get("tera") else mid


# ----------------------------------------------------------------------------- engine read + compare


def read_engine(state) -> dict:
    out = {"sides": []}
    for name in ("side_one", "side_two"):
        s = getattr(state, name)
        pk = s.pokemon
        sc = s.side_conditions
        out["sides"].append({
            "active": int(str(s.active_index).split(".")[-1][-1]) if hasattr(s, "active_index") else 0,
            "mons": [{
                "hp": p.hp, "status": str(p.status).lower(), "item": to_id(p.item), "ability": to_id(p.ability),
                "tera": bool(p.terastallized), "types": [to_id(t) for t in p.types if to_id(t) != "typeless"],
                "pp": {to_id(mv.id): mv.pp for mv in p.moves if to_id(mv.id) != "none"},
            } for p in pk],
            "vols": {v.lower() for v in s.volatile_statuses},
            "boosts": {b: getattr(s, ENGINE_BOOST_ATTR[b]) for b in BOOSTS},
            "sc": {"spikes": sc.spikes, "toxic_spikes": sc.toxic_spikes, "stealth_rock": sc.stealth_rock,
                   "sticky_web": sc.sticky_web, "reflect": sc.reflect, "light_screen": sc.light_screen,
                   "aurora_veil": sc.aurora_veil, "tailwind": sc.tailwind, "safeguard": sc.safeguard,
                   "mist": sc.mist, "lucky_chant": sc.lucky_chant},
            "sub_hp": s.substitute_health,
        })
    out["weather"] = str(state.weather).lower().split(".")[-1]
    out["weather_turns"] = state.weather_turns_remaining
    out["terrain"] = str(state.terrain).lower().split(".")[-1]
    out["trick_room"] = bool(state.trick_room)
    return out


def sim_expected(post: dict, pre_built: Built) -> dict:
    """Simulator post-state normalised into the engine's vocabulary (same keys as ``read_engine``)."""
    out = {"sides": []}
    for k, sd in enumerate(post["sides"]):
        mons = []
        by_uid = {}
        for m in sd["mons"]:
            by_uid[m["uid"]] = m
        act = sd["mons"][0]
        vols = set()
        for vid, st in act["volatiles"].items():
            st = st or {}
            if vid in ("protosynthesis", "quarkdrive"):
                best = st.get("bestStat")
                vid = f"{vid}{best}" if best else vid
            elif vid == "twoturnmove":
                vid = st.get("move") or vid
            elif vid == "perishsong":
                vid = f"perish{st.get('duration')}"
            elif vid == "transform":
                vid = "transformed"
            if vid not in IGNORED_VOLATILES and not vid.startswith(("ability:", "item:")) and not act["fainted"]:
                vols.add(vid)
        sc = {"spikes": 0, "toxic_spikes": 0, "stealth_rock": 0, "sticky_web": 0, "reflect": 0, "light_screen": 0,
              "aurora_veil": 0, "tailwind": 0, "safeguard": 0, "mist": 0, "lucky_chant": 0}
        for cid, st in sd["sideConditions"].items():
            st = st or {}
            key = {"stealthrock": "stealth_rock", "toxicspikes": "toxic_spikes", "stickyweb": "sticky_web",
                   "lightscreen": "light_screen", "auroraveil": "aurora_veil", "luckychant": "lucky_chant"}.get(cid, cid)
            if key in sc:
                sc[key] = int(st.get("layers", 1) or 1) if key in ("spikes", "toxic_spikes") else (
                    1 if key in ("stealth_rock", "sticky_web") else int(st.get("duration", 0) or 0))
        out["sides"].append({
            "mons_by_uid": by_uid, "active_uid": act["uid"], "vols": vols,
            "boosts": {b: act["boosts"].get(b, 0) for b in BOOSTS}, "sc": sc,
        })
    w = post.get("weather") or ""
    out["weather"] = WEATHER_TO_ENGINE.get(w, w or "none")
    out["terrain"] = post.get("terrain") or "none"
    out["trick_room"] = "trickroom" in post["pseudoWeather"]
    return out


def hp_ok(hp0: int, e_hp: int, s_hp: int, maxhp: int = 0) -> bool:
    de, ds = hp0 - e_hp, hp0 - s_hp
    return abs(de - ds) <= max(3.0, 0.11 * max(abs(de), abs(ds)), 0.03 * maxhp)


def compare(pre_built: Built, pre: dict, post: dict, eng: dict, exp: dict) -> list[dict]:
    """Mismatches between one engine branch (``eng``) and the simulator result (``exp``)."""
    mm: list[dict] = []
    unknown = {u.split(":", 1)[1] for u in pre_built.unknown_ids}

    def add(field, side, uid, detail, **extra):
        mm.append({"field": field, "side": side, "uid": uid, "detail": detail, **extra})

    for k in (0, 1):
        es, xs = eng["sides"][k], exp["sides"][k]
        pre_mons = pre["sides"][k]["mons"]
        for i, pm in enumerate(pre_mons):
            uid = pm["uid"]
            e = es["mons"][i]
            s = xs["mons_by_uid"].get(uid)
            if s is None:
                continue
            e_alive, s_alive = e["hp"] > 0, s["hp"] > 0
            if e_alive != s_alive:
                add("faint", k, uid, f"engine hp {e['hp']} vs sim {s['hp']} (pre {pm['hp']}/{pm['maxhp']})",
                    ratio=None, e_hp=e["hp"], s_hp=s["hp"], hp0=pm["hp"])
            elif s_alive and not hp_ok(pm["hp"], e["hp"], s["hp"], pm["maxhp"]):
                de, ds = pm["hp"] - e["hp"], pm["hp"] - s["hp"]
                add("hp", k, uid, f"dmg engine {de} vs sim {ds} (pre {pm['hp']}/{pm['maxhp']})",
                    ratio=(de / ds) if ds else None, e_hp=e["hp"], s_hp=s["hp"], hp0=pm["hp"])
            if s_alive and e_alive:
                if STATUS_TO_ENGINE.get(s["status"], s["status"]) != e["status"]:
                    add("status", k, uid, f"engine {e['status']} vs sim {s['status'] or 'none'}")
                if pm["item"] and "item:" + to_id(pm["item"]) not in pre_built.unknown_ids or not pm["item"]:
                    if (to_id(s["item"]) or "none") != e["item"] and to_id(s["item"]) not in unknown:
                        add("item", k, uid, f"engine {e['item']} vs sim {to_id(s['item']) or 'none'}")
                if (to_id(s["ability"]) or "none") != e["ability"] and to_id(s["ability"]) not in unknown:
                    add("ability", k, uid, f"engine {e['ability']} vs sim {to_id(s['ability'])}")
                if bool(s["terastallized"]) != e["tera"]:
                    add("tera", k, uid, f"engine {e['tera']} vs sim {s['terastallized'] or False}")
                st_types = sorted(to_id(t) for t in s["types"] if t != "???")
                if not s["terastallized"] and sorted(e["types"]) != st_types and not e["tera"]:
                    add("types", k, uid, f"engine {sorted(e['types'])} vs sim {st_types}")
            for mv in s["moves"]:
                mid = to_id(mv["id"])
                if mid in e["pp"] and mid not in unknown and s_alive:
                    simpp = max(-128, min(127, mv["pp"]))
                    if e["pp"][mid] != simpp:
                        add("pp", k, uid, f"{mid}: engine {e['pp'][mid]} vs sim {simpp}", move=mid)
        if exp["sides"][k]["active_uid"] != pre_mons[es["active"]]["uid"]:
            add("active", k, exp["sides"][k]["active_uid"],
                f"engine active {pre_mons[es['active']]['uid']} vs sim {exp['sides'][k]['active_uid']}")
        elif exp["sides"][k]["mons_by_uid"][xs["active_uid"]]["hp"] > 0:
            for b in BOOSTS:
                if es["boosts"][b] != xs["boosts"][b]:
                    add(f"boost.{b}", k, xs["active_uid"], f"engine {es['boosts'][b]} vs sim {xs['boosts'][b]}")
            ev, sv = es["vols"] - ENGINE_BOOKKEEPING, xs["vols"]
            for v in sorted(ev - sv):
                if engine_supports_volatile(v):
                    add("volatile.extra_in_engine", k, xs["active_uid"], v, name=v)
            for v in sorted(sv - ev):
                if engine_supports_volatile(v):
                    add("volatile.missing_in_engine", k, xs["active_uid"], v, name=v)
        for key in xs["sc"]:
            ev_, sv_ = es["sc"][key], xs["sc"][key]
            if key in ("spikes", "toxic_spikes", "stealth_rock", "sticky_web"):
                if ev_ != sv_:
                    add(f"sidecond.{key}", k, None, f"engine {ev_} vs sim {sv_}", name=key)
            elif (ev_ > 0) != (sv_ > 0):
                add(f"sidecond.{key}", k, None, f"engine {ev_} vs sim {sv_}", name=key)
            elif ev_ != sv_:
                add(f"sidecond_turns.{key}", k, None, f"engine {ev_} vs sim {sv_}", name=key)
    if eng["weather"] != exp["weather"]:
        add("weather", None, None, f"engine {eng['weather']} vs sim {exp['weather']}")
    if eng["terrain"] != exp["terrain"]:
        add("terrain", None, None, f"engine {eng['terrain']} vs sim {exp['terrain']}")
    if eng["trick_room"] != exp["trick_room"]:
        add("trickroom", None, None, f"engine {eng['trick_room']} vs sim {exp['trick_room']}")
    return mm


# ----------------------------------------------------------------------------- one turn


def engine_branches(state, c1: str, c2: str):
    try:
        return pe.generate_instructions(state, c1, c2), None
    except BaseException as exc:  # pyo3 PanicException derives from BaseException
        if isinstance(exc, KeyboardInterrupt):
            raise
        return None, f"{type(exc).__name__}: {exc}"


def run_turn_engine(rec: dict, built: Built):
    """Apply the turn's actions (and in-turn forced switches) to the engine.

    Returns (list of (probability, final_state), error)."""
    pre = rec["pre"]
    c = [engine_choice(rec["actions"][k], built, k, rec["requests"][k]) for k in (0, 1)]
    reqs = rec["requests"]
    for k in (0, 1):
        mv = reqs[k]["moves"]
        if len(mv) == 1 and mv[0]["id"] == "recharge":
            c[k] = "none"  # the engine models recharge as the mustrecharge volatile + a no-op choice
        elif len(mv) == 1 and mv[0]["id"] == "struggle":
            return None, "skip: struggle (engine has no struggle choice)", c
    branches, err = engine_branches(built.state, c[0], c[1])
    if err:
        return None, f"{err} [choices {c}]", c
    if len(branches) > MAX_BRANCHES:
        return None, f"too many branches ({len(branches)})", c
    frontier = [(float(b.percentage) / 100.0, built.state.apply_instructions(b)) for b in branches]
    for fu in rec["followups"]:
        nxt = []
        for prob, st in frontier:
            fb = Built.__new__(Built)
            fb.state, fb.uid_index = st, built.uid_index
            fc = [engine_choice(fu[k], fb, k, None) for k in (0, 1)]
            # Pivot (U-turn, Flip Turn, ...): the engine flags the pivoter's force_switch and expects the OTHER side's
            # saved move as its choice for the follow-up step, not "none".
            for k in (0, 1):
                me, other = (st.side_one, st.side_two) if k == 0 else (st.side_two, st.side_one)
                if fu[k] is None and other.force_switch and fu[1 - k] is not None:
                    saved = str(me.switch_out_move_second_saved_move).lower()
                    fc[k] = saved if saved not in ("", "none") else "none"
            bs, err = engine_branches(st, fc[0], fc[1])
            if err:
                return None, f"followup {err} [choices {fc}]", c
            for b in bs:
                nxt.append((prob * float(b.percentage) / 100.0, st.apply_instructions(b)))
            if len(nxt) > MAX_BRANCHES:
                return None, "too many branches (follow-up)", c
        frontier = nxt
        c.append("|".join(x or "-" for x in [engine_choice(fu[k], fb, k, None) if fu[k] else None for k in (0, 1)]))
    return frontier, None, c


# ----------------------------------------------------------------------------- log attribution

FROM_RE = re.compile(r"\[from\] ([^|\]]+)")


def parse_log(log: list[str]) -> list[dict]:
    """Events: {side, kind, cause, arg}. cause = [from] source or the move being executed."""
    events = []
    cur = None
    for line in log:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        kind = parts[1]
        if kind == "move" and len(parts) > 3:
            cur = to_id(parts[3])
            events.append({"side": 0 if parts[2].startswith("p1") else 1, "kind": "move", "cause": cur, "arg": cur})
            continue
        if kind == "upkeep":
            cur = None
            continue
        if kind in ("turn", "switch", "drag"):
            if kind != "turn":
                cur = None
        if len(parts) < 3 or not parts[2].startswith(("p1", "p2")):
            if kind in ("-weather", "-fieldstart", "-fieldend"):
                m = FROM_RE.search(line)
                events.append({"side": None, "kind": kind, "cause": to_id(m.group(1)) if m else (cur or "?"),
                               "arg": to_id(parts[2]) if len(parts) > 2 else ""})
            elif kind in ("-sidestart", "-sideend") and len(parts) > 3:
                events.append({"side": 0 if parts[2].startswith("p1") else 1, "kind": kind,
                               "cause": cur or "?", "arg": to_id(parts[3].split(":")[-1])})
            continue
        side = 0 if parts[2].startswith("p1") else 1
        m = FROM_RE.search(line)
        cause = to_id(m.group(1)) if m else (cur or "residual")
        arg = to_id(parts[3]) if len(parts) > 3 else ""
        events.append({"side": side, "kind": kind, "cause": cause, "arg": arg, "line": line})
    return events


FIELD_EVENT_KINDS = {
    "hp": ("-damage", "-heal", "-sethp", "faint"), "faint": ("-damage", "faint", "-heal"),
    "status": ("-status", "-curestatus", "-cureteam"), "item": ("-item", "-enditem"),
    "ability": ("-ability", "-endability", "-activate"), "tera": ("-terastallize",),
    "types": ("-start", "-terastallize", "-end"), "active": ("switch", "drag"),
}


def cause_of(mm: dict, events: list[dict], pre: dict | None = None) -> str:
    f = mm["field"]
    side = mm["side"]
    if f == "pp":
        return mm.get("move", "?")
    if f.startswith("boost."):
        kinds = ("-boost", "-unboost", "-setboost", "-clearallboost", "-clearboost", "-swapboost", "-copyboost",
                 "-invertboost", "-clearnegativeboost", "-clearpositiveboost")
    elif f.startswith("volatile"):
        kinds = ("-start", "-end", "-activate", "-singlemove", "-singleturn")
    elif f.startswith("sidecond"):
        kinds = ("-sidestart", "-sideend")
    elif f in ("weather", "terrain", "trickroom"):
        kinds = ("-weather", "-fieldstart", "-fieldend")
    else:
        kinds = FIELD_EVENT_KINDS.get(f, ())
    cands = [e for e in events if e["kind"] in kinds and (side is None or e["side"] in (side, None))]
    if f.startswith(("volatile", "sidecond")) and mm.get("name"):
        narrowed = [e for e in cands if mm["name"] in e.get("arg", "") or mm["name"] in e.get("line", "").lower().replace(" ", "")]
        cands = narrowed or cands
    if f in ("hp", "faint", "status", "item", "ability", "tera", "types") and mm.get("uid") is not None:
        pass
    if not cands:
        # the simulator logged nothing: the ENGINE invented the change. Name the mon's item/ability as the suspects.
        if pre is not None and mm.get("uid") is not None and mm.get("side") is not None:
            for pm in pre["sides"][mm["side"]]["mons"]:
                if pm["uid"] == mm["uid"]:
                    return f"engine-only[item={to_id(pm['item']) or 'none'},ability={to_id(pm['ability'])}]"
        return "(no sim event)"
    cnt = collections.Counter(e["cause"] for e in cands if e["cause"] not in ("residual",))
    if cnt:
        return cnt.most_common(1)[0][0]
    return "residual"


# ----------------------------------------------------------------------------- tags


def turn_tags(rec: dict, events: list[dict]) -> list[str]:
    pre = rec["pre"]
    tags = set()
    for k in (0, 1):
        a = pre["sides"][k]["mons"][0]
        tags.add("species:" + a["sid"])
        tags.add("ability:" + to_id(a["ability"]))
        if a["item"]:
            tags.add("item:" + to_id(a["item"]))
        if a["terastallized"]:
            tags.add("tera:active_terastallized")
        act = rec["actions"][k]
        if act["type"] == "move":
            tags.add("move:" + to_id(act["id"]))
            if act.get("tera"):
                tags.add("tera:use_this_turn")
        else:
            tags.add("action:switch")
    for e in events:
        if e["kind"] == "move":
            tags.add("move:" + e["cause"])
    if pre.get("weather"):
        tags.add("weather:" + pre["weather"])
    if pre.get("terrain"):
        tags.add("terrain:" + pre["terrain"])
    for line in rec["log"]:
        m = re.search(r"\[from\] (ability|item): ([^|\]]+)", line)
        if m:
            tags.add(f"{m.group(1)}:{to_id(m.group(2))}")
        if line.startswith("|-crit"):
            tags.add("event:crit")
    return sorted(tags)


# ----------------------------------------------------------------------------- run


def find_showdown_dir() -> Path:
    cands = []
    if os.environ.get("SHOWDOWN_BOT_SERVER_DIR"):
        cands.append(Path(os.environ["SHOWDOWN_BOT_SERVER_DIR"]))
    cands.append(REPO / "vendor" / "pokemon-showdown")
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        if common:
            cands.append((REPO / common).resolve().parent / "vendor" / "pokemon-showdown")
    except OSError:
        pass
    for c in cands:
        if (c / "dist" / "sim" / "battle.js").exists():
            return c
    raise SystemExit("Showdown checkout with dist/ not found; set SHOWDOWN_BOT_SERVER_DIR (tried: %s)" % cands)


def node_records(games: int, seed: int, max_turns: int):
    cmd = ["node", str(NODE_DRIVER), str(find_showdown_dir()).replace("\\", "/"), str(games), str(seed), str(max_turns)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line:
            yield json.loads(line)
    proc.wait()


def analyse_turn(rec: dict) -> dict:
    """Run one recorded turn through the engine and classify. Returns a result row (JSON-able)."""
    pre, post = rec["pre"], rec["post"]
    row = {"seed": rec["seed"], "game": rec["game"], "turn": rec["turn"], "ended": rec["ended"]}
    built = build_state(pre, rec["requests"])
    row["unrep"], row["unknown_ids"], row["guess"], row["conv_errors"] = built.unrep, built.unknown_ids, built.guess, built.conv_errors
    events = parse_log(rec["log"])
    row["tags"] = turn_tags(rec, events)
    row["actions"] = [rec["actions"][k] for k in (0, 1)]
    frontier, err, choices = run_turn_engine(rec, built)
    row["choices"] = choices
    if err and err.startswith("skip"):
        row["status"] = "skipped"
        row["error"] = err
        return row
    if err:
        row["status"] = "engine_error"
        row["error"] = err
        return row
    exp = sim_expected(post, built)
    row["branches"] = len(frontier)
    best = None
    best_key = None
    best_eng = None
    for prob, st in frontier:
        eng = read_engine(st)
        mm = compare(built, pre, post, eng, exp)
        key = (sum(1 for m in mm if m["field"] != "pp"), len(mm), -prob)
        if best is None or key < best_key:
            best, best_key, best_prob, best_eng = mm, key, prob, eng
            if not mm:
                break
    row["best_prob"] = best_prob
    if rec["ended"]:
        # the simulator stops at the winning faint: no end-of-turn effects (Wish, Leftovers, ...) after it
        best = [m for m in best if m["field"] in ("faint", "pp")]
    for m in best:
        m["cause"] = cause_of(m, events, pre)
    row["mismatches"] = best
    # PP observations: (pre pp, engine delta, sim delta) for every slot either side touched
    obs = []
    for k in (0, 1):
        for i, pm in enumerate(pre["sides"][k]["mons"]):
            s_mon = exp["sides"][k]["mons_by_uid"].get(pm["uid"])
            for mv in pm["moves"]:
                mid = to_id(mv["id"])
                e_pp = best_eng["sides"][k]["mons"][i]["pp"].get(mid)
                s_pp = next((x["pp"] for x in s_mon["moves"] if to_id(x["id"]) == mid), None) if s_mon else None
                if e_pp is None or s_pp is None:
                    continue
                if e_pp != mv["pp"] or s_pp != mv["pp"]:
                    obs.append([mv["pp"], mv["pp"] - e_pp, mv["pp"] - s_pp])
    row["pp_obs"] = obs
    row["dmg_obs"] = damage_observation(rec, pre, best_eng, exp)
    non_pp = [m for m in best if m["field"] != "pp"]
    row["status"] = "diverge" if non_pp else ("pp_only" if best else "match")
    return row


def damage_observation(rec: dict, pre: dict, eng: dict, exp: dict) -> dict | None:
    """Clean single-target damage sample: exactly one mon lost HP in BOTH engine and simulator, nothing healed.

    Returns engine/sim damage and the attacker/defender context, used for the damage-ratio tables."""
    changed_e, changed_s = [], []
    for k in (0, 1):
        for i, pm in enumerate(pre["sides"][k]["mons"]):
            e_hp = eng["sides"][k]["mons"][i]["hp"]
            s_mon = exp["sides"][k]["mons_by_uid"].get(pm["uid"])
            s_hp = s_mon["hp"] if s_mon else pm["hp"]
            if e_hp != pm["hp"]:
                changed_e.append((k, i, pm["hp"] - e_hp))
            if s_hp != pm["hp"]:
                changed_s.append((k, i, pm["hp"] - s_hp))
    if len(changed_e) != 1 or len(changed_s) != 1 or changed_e[0][:2] != changed_s[0][:2]:
        return None
    k, i, de = changed_e[0]
    ds = changed_s[0][2]
    if de <= 0 or ds <= 0 or i != 0:
        return None
    if any(line.startswith("|-heal|") for line in rec["log"]):  # Leftovers, Recover, drain, Wish...: net HP is not damage
        return None
    d, a = pre["sides"][k]["mons"][0], pre["sides"][1 - k]["mons"][0]
    act = rec["actions"][1 - k]
    if act["type"] != "move":
        return None
    return {"de": de, "ds": ds, "move": to_id(act["id"]), "tera_atk": bool(act.get("tera") or a["terastallized"]),
            "def": d["sid"], "def_ab": to_id(d["ability"]), "def_item": to_id(d["item"]), "def_tera": bool(d["terastallized"]),
            "atk": a["sid"], "atk_ab": to_id(a["ability"]), "atk_item": to_id(a["item"]),
            "crit": any(line.startswith("|-crit") for line in rec["log"])}


def cmd_run(args) -> int:
    rows = []
    t0 = time.time()
    games_seen = set()
    for rec in node_records(args.games, args.seed, args.max_turns):
        if "error" in rec:
            print("sim error:", rec["error"][:300], file=sys.stderr)
            continue
        games_seen.add(rec["game"])
        try:
            row = analyse_turn(rec)
        except BaseException as exc:  # a converter crash must not kill the run
            if isinstance(exc, KeyboardInterrupt):
                raise
            row = {"seed": rec["seed"], "game": rec["game"], "turn": rec["turn"], "status": "converter_crash",
                   "error": f"{type(exc).__name__}: {exc}", "tags": [], "unrep": [], "unknown_ids": [], "guess": [], "conv_errors": []}
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} turns, {len(games_seen)} games, {time.time()-t0:.0f}s", file=sys.stderr)
    out = {"games": args.games, "seed": args.seed, "rows": rows, "elapsed_s": time.time() - t0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{len(rows)} turns from {len(games_seen)} games in {time.time()-t0:.0f}s -> {args.out}", file=sys.stderr)
    print(render_report(out, top=args.top))
    return 0


# ----------------------------------------------------------------------------- report


# which guessed counter mappings could by themselves explain a mismatch of a given kind
GUESS_RELEVANCE = {
    "volatile.extra_in_engine:confusion": ("confusion",), "volatile.missing_in_engine:confusion": ("confusion",),
    "volatile.extra_in_engine:encore": ("encore",), "volatile.missing_in_engine:encore": ("encore",),
    "volatile.extra_in_engine:taunt": ("taunt",), "volatile.missing_in_engine:taunt": ("taunt",),
    "volatile.extra_in_engine:lockedmove": ("lockedmove",), "volatile.missing_in_engine:lockedmove": ("lockedmove",),
    "volatile.extra_in_engine:slowstart": ("slowstart",), "volatile.missing_in_engine:slowstart": ("slowstart",),
    "volatile.extra_in_engine:yawn": ("yawn",), "volatile.missing_in_engine:yawn": ("yawn",),
    "volatile.extra_in_engine:substitute": ("substitute",), "volatile.missing_in_engine:substitute": ("substitute",),
    "status": ("sleep", "rest", "toxic"), "weather": ("weather_turns",), "terrain": ("terrain_turns",),
    "trickroom": ("trickroom_turns",),
}


def normalise_cause(c: str) -> str:
    for pre in ("ability", "item", "move"):
        if c.startswith(pre) and len(c) > len(pre) + 2 and not c.startswith("moveset"):
            return f"{pre}:{c[len(pre):]}"
    return c


def converter_risk(field: str, cause: str, rows: list) -> str:
    """'guess' when most of the category's turns contain a counter mapping that is a guess (see GUESS_FEATURES)."""
    rel = GUESS_RELEVANCE.get(field)
    if field.startswith("sidecond_turns"):
        rel = ("screen_turns", "tailwind_turns")
    if field.startswith("hp") and "wish" in cause:
        rel = ("wish",)
    if not rel:
        return ""
    hit = sum(1 for r in rows if any(g in r["guess"] for g in rel))
    return "guess" if hit * 2 >= len(rows) else ""


def pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def render_report(data: dict, top: int = 25) -> str:
    rows = data["rows"]
    n = len(rows)
    st = collections.Counter(r["status"] for r in rows)
    clean = [r for r in rows if r["status"] in COMPARABLE and not r["unrep"] and not r["unknown_ids"] and not r["conv_errors"]]
    lossy = [r for r in rows if r["status"] in COMPARABLE and (r["unrep"] or r["unknown_ids"])]
    div_all = [r for r in rows if r["status"] == "diverge"]
    div_clean = [r for r in clean if r["status"] == "diverge"]
    out = []
    w = out.append
    w(f"# Engine vs Showdown differential report\n")
    w(f"- games: {data['games']} (seed0 {data['seed']}), turns analysed: {n}, wall time {data['elapsed_s']:.0f}s")
    w(f"- turn status: " + ", ".join(f"{k}={v}" for k, v in st.most_common()))
    w(f"- **overall divergence (all comparable turns): {len(div_all)}/{st['match']+st['pp_only']+st['diverge']} = {pct(len(div_all), st['match']+st['pp_only']+st['diverge'])}** (PP-only differences are not counted here, see the PP section)")
    w(f"- clean turns (no unrepresentable features, no unknown ids): {len(clean)}; divergence {len(div_clean)}/{len(clean)} = {pct(len(div_clean), len(clean))}")
    w(f"- lossy turns (engine cannot represent something in the pre-state or lacks an id): {len(lossy)} ({pct(len(lossy), n)}); divergence there {sum(1 for r in lossy if r['status']=='diverge')}/{len(lossy)} = {pct(sum(1 for r in lossy if r['status']=='diverge'), len(lossy))}")
    bc = [r["branches"] for r in rows if "branches" in r]
    if bc:
        w(f"- engine branches per turn: median {statistics.median(bc)}, p95 {sorted(bc)[int(0.95*len(bc))]}, max {max(bc)}")
    ce = collections.Counter(e for r in rows for e in r["conv_errors"])
    w(f"- converter self-check failures: {dict(ce) or 'none'}")
    guess_note = collections.Counter(g for r in rows for g in r["guess"])
    w("")
    # categories
    cat = collections.defaultdict(list)
    for r in rows:
        if r["status"] != "diverge":
            continue
        seen = set()
        for m in r["mismatches"]:
            if m["field"] == "pp":
                continue  # PP has its own section (the engine does not decrement PP above 9)
            label = m["field"] + (":" + m["name"] if m.get("name") else "")
            key = (label, m["cause"])
            if key in seen:
                continue
            seen.add(key)
            cat[key].append(r)
    ranked = sorted(cat.items(), key=lambda kv: -len(kv[1]))
    w("## Divergence categories (best-matching branch's mismatches; one count per turn per category)\n")
    w("| # | field | mechanic (sim cause) | turns | % of turns | converter risk | example (seed:turn) |")
    w("|---|---|---|---|---|---|---|")
    for i, ((field, cause), rs) in enumerate(ranked[:top], 1):
        ex = ", ".join(f"{r['seed']}:{r['turn']}" for r in rs[:3])
        w(f"| {i} | {field} | {cause} | {len(rs)} | {pct(len(rs), n)} | {converter_risk(field, cause, rs)} | {ex} |")
    w("")
    # field-only
    byfield = collections.Counter()
    for (field, _c), rs in cat.items():
        pass
    fcount = collections.defaultdict(set)
    vol_names = collections.Counter()
    mech = collections.defaultdict(set)
    for r in rows:
        if r["status"] != "diverge":
            continue
        for m in r["mismatches"]:
            if m["field"] == "pp":
                continue
            f = m["field"]
            base = f.split(".")[0] if f.startswith(("boost", "sidecond_turns")) else f
            fcount[base].add((r["seed"], r["turn"]))
            if f.startswith("volatile"):
                vol_names[(f.split(".")[1], m["name"])] += 1
            mech[normalise_cause(m["cause"])].add((r["seed"], r["turn"]))
    w("## By mechanic (cause attributed from the simulator log, merged across fields)\n")
    w("| mechanic | turns | % of turns |")
    w("|---|---|---|")
    for c, ts in sorted(mech.items(), key=lambda kv: -len(kv[1]))[:top]:
        w(f"| {c} | {len(ts)} | {pct(len(ts), n)} |")
    w("")
    w("## By field\n")
    w("| field | turns | % of turns |")
    w("|---|---|---|")
    for f, ts in sorted(fcount.items(), key=lambda kv: -len(kv[1])):
        w(f"| {f} | {len(ts)} | {pct(len(ts), n)} |")
    w("")
    w("volatile mismatches by name (mismatch count): " + ", ".join(f"{d}:{nm} {c}" for (d, nm), c in vol_names.most_common(20)))
    w("")
    # tag lift
    base = len(div_all) / max(1, len(rows))
    inv = collections.Counter()
    dv = collections.Counter()
    for r in rows:
        if r["status"] not in COMPARABLE:
            continue
        for t in r["tags"]:
            inv[t] += 1
            if r["status"] == "diverge":
                dv[t] += 1
    lift = [(t, inv[t], dv[t]) for t in inv if inv[t] >= 8]
    lift.sort(key=lambda x: -(x[2] - base * x[1]))
    w(f"## Mechanics most associated with divergence (tag involved in the turn; base rate {pct(len(div_all), len(rows))})\n")
    w("| tag | turns involved | diverged | rate | excess diverged turns |")
    w("|---|---|---|---|---|")
    for t, a, b in lift[:top]:
        w(f"| {t} | {a} | {b} | {pct(b, a)} | {b - base*a:.1f} |")
    w("")
    # unrepresentable / unknown
    un = collections.Counter(x for r in rows for x in r["unrep"])
    uk = collections.Counter(x for r in rows for x in r["unknown_ids"])
    w("## Engine cannot represent / does not know (converter limits that are ENGINE gaps)\n")
    w("unrepresentable features (turns): " + (", ".join(f"{k} {v}" for k, v in un.most_common(25)) or "none"))
    w("\nunknown ids (turns): " + (", ".join(f"{k} {v}" for k, v in uk.most_common(40)) or "none"))
    w("")
    w("counter mappings that are GUESSES (present in pre-state, turns): " + ", ".join(f"{k} {v}" for k, v in guess_note.most_common()))
    errs = collections.Counter(r["error"][:90] for r in rows if r["status"] in ("engine_error", "converter_crash"))
    if errs:
        w("\n## Engine panics / errors / converter crashes\n")
        for e, c in errs.most_common(10):
            w(f"- {c}x {e}")
    w("")
    w(damage_section(rows))
    w("")
    w(suspect_section(rows))
    return "\n".join(out)


def damage_section(rows, min_n: int = 8, top: int = 20) -> str:
    """Engine/simulator damage ratio on clean single-target damage turns, overall and by context."""
    obs = [r["dmg_obs"] for r in rows if r.get("dmg_obs") and not r["dmg_obs"]["crit"]]
    out = ["## Damage model check (clean single-target damage turns, no crit)", ""]
    w = out.append
    if not obs:
        return "\n".join(out + ["no samples"])
    ratios = [o["de"] / o["ds"] for o in obs]
    bad = sum(1 for x in ratios if not 0.85 <= x <= 1.15)
    w(f"- samples: {len(obs)}; median engine/sim ratio {statistics.median(ratios):.3f}; outside [0.85, 1.15]: {bad} ({pct(bad, len(obs))}); "
      f"outside [0.7, 1.4]: {sum(1 for x in ratios if not 0.7 <= x <= 1.4)}; ratio >= 1.8 or <= 0.55: {sum(1 for x in ratios if x >= 1.8 or x <= 0.55)}")
    w("- the engine uses 0.925 x max roll (mean of the simulator's 0.85..1.00 rolls), so a correct model gives ratio ~1.0 with spread about +-8%")
    w("")
    groups = collections.defaultdict(list)
    for o, x in zip(obs, ratios):
        for key in ("def", "def_ab", "def_item", "atk", "atk_ab", "atk_item", "move"):
            groups[f"{key}={o[key]}"].append(x)
        groups[f"def_tera={o['def_tera']}"].append(x)
        groups[f"atk_tera={o['tera_atk']}"].append(x)
    rows_ = []
    for g, xs in groups.items():
        if len(xs) >= min_n:
            med = statistics.median(xs)
            nb = sum(1 for x in xs if not 0.85 <= x <= 1.15)
            rows_.append((nb - 0.04 * len(xs), g, len(xs), med, nb))
    rows_.sort(reverse=True)
    w(f"Contexts (n >= {min_n}) with the most out-of-band samples (ranked by outliers beyond the 4% base rate):")
    w("")
    w("| context | n | median engine/sim | outside [0.85,1.15] |")
    w("|---|---|---|---|")
    for _d, g, n_, med, b in rows_[:top]:
        w(f"| {g} | {n_} | {med:.2f} | {b} ({pct(b, n_)}) |")
    return "\n".join(out)


def damage_ratio_groups(rows):
    """For diverging/matching turns where exactly the damage fields differ, group |engine/sim| damage ratios."""
    groups = collections.defaultdict(list)
    for r in rows:
        if r["status"] != "diverge":
            continue
        for m in r["mismatches"]:
            if m["field"] == "hp" and m.get("ratio"):
                for t in r["tags"]:
                    groups[t].append(m["ratio"])
    return groups


def suspect_section(rows) -> str:
    out = ["## Known suspects\n"]
    w = out.append

    def stat(tag_filter, label):
        inv = [r for r in rows if r["status"] in COMPARABLE and tag_filter(r["tags"])]
        dv = [r for r in inv if r["status"] == "diverge"]
        hp = [m for r in dv for m in r["mismatches"] if m["field"] in ("hp", "faint")]
        ratios = [m["ratio"] for m in hp if m.get("ratio")]
        med = statistics.median(ratios) if ratios else None
        w(f"- **{label}**: involved turns {len(inv)}, diverged {len(dv)} ({pct(len(dv), len(inv))}); "
          f"hp/faint mismatches {len(hp)}" + (f", median engine/sim damage ratio {med:.2f} (n={len(ratios)})" if med else ""))
        return inv, dv, ratios

    tp = lambda tags: "species:terapagosterastal" in tags or "species:terapagosstellar" in tags
    stat(tp, "Terapagos-Terastal/Stellar on field")
    for ab in ("protosynthesis", "quarkdrive", "supremeoverlord", "flashfire"):
        stat(lambda tags, ab=ab: "ability:" + ab in tags, f"ability {ab} on an active")
    npp = sum(1 for r in rows if r["status"] in COMPARABLE)
    pp_turns = sum(1 for r in rows if r["status"] in COMPARABLE and any(m["field"] == "pp" for m in r["mismatches"]))
    w(f"- **PP tracking**: PP differs from the simulator in {pp_turns} of {npp} turns ({pct(pp_turns, npp)}). "
      "Per pre-turn PP bucket, for slots whose PP changed in either engine or simulator (engine decrement vs simulator decrement):")
    w("")
    w("| pre-turn PP | slots | sim decremented | engine decremented | engine matches sim |")
    w("|---|---|---|---|---|")
    buckets = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        for pre_pp, de, ds in r.get("pp_obs", []):
            b = "1-9" if 1 <= pre_pp <= 9 else ("10-15" if pre_pp < 16 else ("16-31" if pre_pp < 32 else "32+"))
            x = buckets[b]
            x[0] += 1
            x[1] += ds != 0
            x[2] += de != 0
            x[3] += de == ds
    for b in ("1-9", "10-15", "16-31", "32+"):
        if b in buckets:
            x = buckets[b]
            w(f"| {b} | {x[0]} | {x[1]} | {x[2]} | {x[3]} ({pct(x[3], x[0])}) |")
    return "\n".join(out)


def cmd_report(args) -> int:
    data = json.loads(Path(args.file).read_text())
    text = render_report(data, top=args.top)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


# ----------------------------------------------------------------------------- targeted suspect probes


def _set(species, ability, moves, item="", **kw):
    return {"species": species, "ability": ability, "item": item, "moves": moves, **kw}


FILLER = [_set("Magikarp", "Swift Swim", ["Splash"]) for _ in range(5)]
SNORLAX = _set("Snorlax", "Thick Fat", ["Splash"], item="")
SEEDS = list(range(1, 31))


def suspect_scenarios() -> list[dict]:
    """Single-mechanic probes: (name, attacker side p1, defender side p2, scripted turns). Measured turn = last."""
    sc = []

    def add(name, p1lead, p2lead, script, **kw):
        sc.append({"name": name, "p1": [p1lead] + FILLER, "p2": [p2lead] + FILLER, "script": script, "seeds": SEEDS, **kw})

    tera = _set("Terapagos", "Tera Shift", ["Splash"])  # the custom-game sim does not validate learnsets
    for mv, atk in (("Karate Chop", _set("Machamp", "No Guard", ["Karate Chop"])),
                    ("Earthquake", _set("Garchomp", "Rough Skin", ["Earthquake"])),
                    ("Hyper Voice", _set("Salamence", "Intimidate", ["Hyper Voice"]))):
        add(f"tera_shell: {mv} vs Terapagos-Terastal (full HP)", atk, tera, [["move 1", "move 1"]])
        add(f"control: {mv} vs Snorlax", atk, SNORLAX, [["move 1", "move 1"]])
    # Booster Energy abilities (attacker)
    for ab_name, mon, ab, mv in (("protosynthesis", "Roaring Moon", "Protosynthesis", "Earthquake"),
                                 ("quark_drive", "Iron Valiant", "Quark Drive", "Knock Off")):
        add(f"{ab_name}: {mon} Booster Energy {mv} vs Snorlax", _set(mon, ab, [mv], item="Booster Energy"), SNORLAX, [["move 1", "move 1"]])
        add(f"{ab_name} control: {mon} (no item) {mv} vs Snorlax", _set(mon, ab, [mv]), SNORLAX, [["move 1", "move 1"]])
    # Protosynthesis from sun on the DEFENDING side (boosts the best defensive stat if it is Def/SpD)
    add("protosynthesis (sun, defender Iron Hands-like Scream Tail): Earthquake vs Scream Tail",
        _set("Garchomp", "Rough Skin", ["Earthquake"]), _set("Scream Tail", "Protosynthesis", ["Splash"], item="Booster Energy"),
        [["move 1", "move 1"]])
    # Supreme Overlord
    for n in (0, 2, 5):
        add(f"supreme_overlord: Kingambit with {n} fainted allies, Kowtow Cleave vs Snorlax",
            _set("Kingambit", "Supreme Overlord", ["Kowtow Cleave"]), SNORLAX, [["move 1", "move 1"]], faint=[n, 0])
    # Flash Fire: step 1 opponent Flamethrower into Heatran (activates), step 2 Heatran Flamethrower vs Blissey
    blissey = _set("Blissey", "Natural Cure", ["Flamethrower", "Splash"])
    heatran = _set("Heatran", "Flash Fire", ["Flamethrower", "Stealth Rock"])
    add("flash_fire: Heatran after absorbing Flamethrower, Flamethrower vs Blissey", heatran, blissey,
        [["move 2", "move 1"], ["move 1", "move 2"]])
    add("flash_fire control: Heatran (not activated) Flamethrower vs Blissey", heatran, blissey,
        [["move 2", "move 2"], ["move 1", "move 2"]])
    return sc


def cmd_suspects(args) -> int:
    import tempfile

    scs = suspect_scenarios()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(scs, fh)
        path = fh.name
    cmd = ["node", str(NODE_DRIVER), str(find_showdown_dir()).replace("\\", "/"), "scenario", path]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    by_name: dict[str, list] = collections.defaultdict(list)
    for line in proc.stdout.splitlines():
        rec = json.loads(line)
        if "error" in rec:
            print("sim error:", rec["error"][:200], file=sys.stderr)
            continue
        by_name[rec["scenario"]].append(rec)
    lines = ["## Targeted suspect probes (30 seeds each, fixed teams, last scripted turn measured)", "",
             "Engine damage = probability-weighted over all engine branches (includes crit / miss branches); simulator = mean over seeds.", "",
             "| scenario | n | sim mean dmg | sim min-max | engine expected dmg | engine/sim |", "|---|---|---|---|---|---|"]
    for sc in scs:
        recs = by_name.get(sc["name"], [])
        last = len(sc["script"]) - 1
        recs = [r for r in recs if r["step"] == last]
        sims, engs = [], []
        for rec in recs:
            built = build_state(rec["pre"], rec["requests"])
            frontier, err, _c = run_turn_engine(rec, built)
            if err:
                print("engine error:", sc["name"], err, file=sys.stderr)
                continue
            hp0 = rec["pre"]["sides"][1]["mons"][0]["hp"]
            sims.append(hp0 - rec["post"]["sides"][1]["mons"][0]["hp"])
            engs.append(sum(p * (hp0 - read_engine(st)["sides"][1]["mons"][0]["hp"]) for p, st in frontier))
        if not sims:
            lines.append(f"| {sc['name']} | 0 | - | - | - | - |")
            continue
        sm, em = statistics.mean(sims), statistics.mean(engs)
        lines.append(f"| {sc['name']} | {len(sims)} | {sm:.1f} | {min(sims)}-{max(sims)} | {em:.1f} | {em / sm if sm else float('nan'):.2f} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
    return 0


def brief_mon(m: dict) -> str:
    vols = ",".join(f"{k}{'' if not v else v}" for k, v in m["volatiles"].items())
    boosts = {k: v for k, v in m["boosts"].items() if v}
    ss = {k: v for k, v in m["statusState"].items() if v is not None}
    return (f"{m['sid']} L{m['level']} hp {m['hp']}/{m['maxhp']} {m['status'] or '-'}{ss if m['status'] else ''} "
            f"ab={m['ability']} item={m['item'] or '-'} tera={m['teraType']}/{m['terastallized'] or '-'} types={m['types']} "
            f"boosts={boosts} vols=[{vols}] pp={[(x['id'], x['pp']) for x in m['moves']]}")


def cmd_repro(args) -> int:
    seed, turn = (int(x) for x in args.target.split(":"))
    for rec in node_records(1, seed, args.max_turns):
        if rec.get("turn") != turn:
            continue
        pre, post = rec["pre"], rec["post"]
        built = build_state(pre, rec["requests"])
        print(f"=== battle seed {seed} turn {turn}  weather={pre['weather']} terrain={pre['terrain']} field={list(pre['pseudoWeather'])}")
        for k in (0, 1):
            sd = pre["sides"][k]
            print(f"-- side {k+1} PRE  conds={ {c: v for c, v in sd['sideConditions'].items()} }")
            print("   active:", brief_mon(sd["mons"][0]))
            print("   bench :", [(m["uid"], m["hp"]) for m in sd["mons"][1:]])
        print("actions:", rec["actions"], "followups:", rec["followups"])
        print("unrepresentable:", built.unrep, "unknown ids:", built.unknown_ids, "guess:", built.guess)
        frontier, err, choices = run_turn_engine(rec, built)
        print("engine choices:", choices, "error:", err)
        exp = sim_expected(post, built)
        for k in (0, 1):
            sd = post["sides"][k]
            print(f"-- side {k+1} POST (simulator)  conds={ {c: v for c, v in sd['sideConditions'].items()} }")
            print("   active:", brief_mon(sd["mons"][0]))
        if frontier:
            ranked = sorted(frontier, key=lambda x: -x[0])
            print(f"engine branches: {len(frontier)}; showing up to 8 by probability")
            for prob, st in ranked[:8]:
                e = read_engine(st)
                mm = compare(built, pre, post, e, exp)
                print(f" p={prob:.4f} act hp {[ (e['sides'][k]['mons'][e['sides'][k]['active']]['hp']) for k in (0,1)]} "
                      f"vols={[sorted(e['sides'][k]['vols']) for k in (0,1)]} mismatches={[(m['field'], m['detail']) for m in mm]}")
        print(chr(10).join(rec["log"]))
        return 0
    print("turn not found")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--games", type=int, default=20)
    r.add_argument("--seed", type=int, default=1000, help="battle i uses seed [seed+i,2,3,4]")
    r.add_argument("--max-turns", type=int, default=120)
    r.add_argument("--out", default="scratch/engine_diff.json")
    r.add_argument("--top", type=int, default=25)
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("file")
    p.add_argument("--md")
    p.add_argument("--top", type=int, default=25)
    p.set_defaults(fn=cmd_report)
    sp = sub.add_parser("suspects", help="targeted single-mechanic probes (Tera Shell, Protosynthesis, ...)")
    sp.add_argument("--md")
    sp.set_defaults(fn=cmd_suspects)
    q = sub.add_parser("repro")
    q.add_argument("target", help="BATTLE_SEED:TURN, e.g. 1005:7")
    q.add_argument("--max-turns", type=int, default=120)
    q.set_defaults(fn=cmd_repro)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
