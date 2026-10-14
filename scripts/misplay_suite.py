"""Misplay regression suite: positions that probe the bot's recurring ladder misplays.

    python scripts/misplay_suite.py mine   --records DIR --out cands.json     # candidate positions from ladder records
    python scripts/misplay_suite.py verify --cands cands.json --out v.json    # long-search evidence per candidate
    python scripts/misplay_suite.py run --evaluator models/x.pev --scale 1.5 --time-ms 100,1000 --worlds 4

The suite (tests/puzzles/misplays/misplays.json) is a list of puzzles:

    {"id", "cat", "desc", "why", "states": [engine State strings (opponent worlds)],
     "accept": [...], "reject": [...], "fold_tera": true, "source": {...}, "evidence": {...}}

A puzzle passes when the bot's *argmax* first move (visit shares summed over the searched
worlds) is in ``accept`` (or, with no ``accept``, not in ``reject``).  Choices are compared by
base move / switch target (``switch x`` -> ``x``); with ``fold_tera`` false the ``-tera``
suffix is kept, so the tera puzzles can ask for "this move *without* terastallizing".
These are diagnostics, not targets: each position was only kept when a long search or a
played-out check supports the intended answer (see ``evidence``), and the suite is meant to
measure whether a new value net fixes the misplays, never to be trained on.

``run`` reports the pass rate per category.  One *sample* is ``--worlds`` searches (the
puzzle's states cycled over the worlds); ``--reps`` samples are averaged.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "puzzles" / "misplays" / "misplays.json"
sys.path.insert(0, str(ROOT / "src"))

RECOVERY = {"RECOVER", "ROOST", "SLACKOFF", "SOFTBOILED", "MOONLIGHT", "SYNTHESIS", "MORNINGSUN",
            "STRENGTHSAP", "WISH", "MILKDRINK", "SHOREUP", "HEALORDER", "REST", "JUNGLEHEALING",
            "LUNARBLESSING", "LIFEDEW", "SYRUPBOMB", "HEALPULSE"}
SETUP = {"SWORDSDANCE", "NASTYPLOT", "DRAGONDANCE", "CALMMIND", "QUIVERDANCE", "BULKUP", "SHELLSMASH",
         "COIL", "IRONDEFENSE", "AGILITY", "ROCKPOLISH", "WORKUP", "GROWTH", "HONECLAWS", "TAILGLOW",
         "VICTORYDANCE", "CURSE", "AMNESIA", "ACIDARMOR", "COSMICPOWER", "SHIFTGEAR", "FILLETAWAY",
         "TIDYUP", "CLANGOROUSSOUL"}
HAZARD_STATUS = {"STEALTHROCK", "SPIKES", "TOXICSPIKES", "STICKYWEB", "THUNDERWAVE", "WILLOWISP", "TOXIC",
                 "SPORE", "SLEEPPOWDER", "HYPNOSIS", "YAWN", "GLARE", "LEECHSEED", "SUBSTITUTE"}
SLEEP_MOVES = {"SPORE", "SLEEPPOWDER", "HYPNOSIS", "YAWN", "LOVELYKISS", "SING", "DARKVOID", "GRASSWHISTLE",
               "RELICSONG"}
DAMAGING_EXCLUDE = RECOVERY | SETUP | HAZARD_STATUS | {"PROTECT", "DETECT", "TAUNT", "ENCORE", "WHIRLWIND",
                                                      "ROAR", "HEALBELL", "AROMATHERAPY", "REFLECT", "LIGHTSCREEN",
                                                      "AURORAVEIL", "TRICKROOM", "DEFOG", "RAPIDSPIN", "MIRRORCOAT",
                                                      "COUNTER", "DESTINYBOND", "NONE", "TRICK", "SWITCHEROO"}


def norm(choice: str, fold_tera: bool = True) -> str:
    if fold_tera and choice.endswith("-tera"):
        choice = choice[:-5]
    return choice[7:] if choice.startswith("switch ") else choice


# --------------------------------------------------------------------------- state helpers

def active(side):
    return side.pokemon[int(side.active_index)]


def ko_prob(pe, st, move: str) -> float:
    """P(the foe's active faints) if we use ``move`` and the foe does nothing."""
    foe = active(st.side_two)
    total = 0.0
    for ins in pe.generate_instructions(st, move, "none"):
        after = st.apply_instructions(ins)
        if active(after.side_two).hp <= 0 and foe.hp > 0:
            total += ins.percentage / 100.0
    return total


def exp_damage_frac(pe, st, move: str) -> float:
    foe = active(st.side_two)
    total = 0.0
    for ins in pe.generate_instructions(st, move, "none"):
        after = st.apply_instructions(ins)
        total += ins.percentage / 100.0 * max(0, foe.hp - active(after.side_two).hp)
    return total / max(1, foe.maxhp)


# --------------------------------------------------------------------------- mining

def iter_decisions(records: str):
    for path in sorted(glob.glob(str(Path(records) / "*.jsonl.gz"))):
        game = None
        prev = None
        for line in gzip.open(path, "rt"):
            rec = json.loads(line)
            if rec.get("type") == "game":
                game = rec["battle_tag"]
            elif rec.get("type") == "decision":
                yield game, rec, prev
                prev = rec["chosen"]


def mine(records: str) -> list[dict]:
    import poke_engine as pe

    cands: list[dict] = []
    for game, d, prev in iter_decisions(records):
        states = [w["state"] for w in d["worlds"]]
        st = pe.State.from_string(states[0])
        me, foe = active(st.side_one), active(st.side_two)
        if me.hp <= 0 or foe.hp <= 0 or st.side_one.force_switch:
            continue
        chosen = d["chosen"]
        base = norm(chosen)
        tera_used = "terastallize" in (d.get("order") or "")
        mymoves = [m.id for m in me.moves if m.id != "NONE" and m.pp > 0 and not m.disabled]
        foemoves = {m.id for m in foe.moves}
        hp, fhp = me.hp / me.maxhp, foe.hp / foe.maxhp
        src = {"game": game, "turn": int(d["turn"]), "chosen": chosen}

        def add(cat, accept=(), reject=(), desc="", fold_tera=True):
            cands.append({"cat": cat, "source": src, "states": states, "accept": list(accept),
                          "reject": list(reject), "fold_tera": fold_tera, "desc": desc})

        label = f"{me.id} {hp:.0%} vs {foe.id} {fhp:.0%}"
        # --- 1: a KO is available but the bot did something else (heal, setup, weaker attack)
        ko_moves = []
        for m in mymoves:
            if m in DAMAGING_EXCLUDE:
                continue
            p = ko_prob(pe, st, m.lower())
            if p >= 0.5:
                ko_moves.append(m.lower())
        if ko_moves and base not in ko_moves and not chosen.startswith("switch"):
            cat = ("heal_loop" if base.upper() in RECOVERY else
                   "sub_loop" if base == "substitute" else "ko_value")
            add(cat, accept=ko_moves, desc=f"{label}: {', '.join(ko_moves)} KOs (idle foe); bot played {chosen}")
        # --- recovery / substitute loops with no KO in hand
        elif base.upper() in RECOVERY and hp >= 0.45 and fhp <= 0.6:
            add("heal_loop", reject=[base], desc=f"{label}: bot healed at {hp:.0%} while the foe is at {fhp:.0%}")
        elif base == "substitute" and prev == "substitute":
            add("sub_loop", reject=["substitute"], desc=f"{label}: second Substitute in a row")
        # --- 4: Toxic available on a healthy bulky target
        if "TOXIC" in mymoves and foe.status == "NONE" and fhp >= 0.7:
            types = {t.upper() for t in foe.types}
            if not types & {"POISON", "STEEL"}:
                add("toxic", accept=["toxic"], desc=f"{label}: Toxic on an unstatused target; bot played {chosen}")
        # --- 5: Taunt / Encore
        if "TAUNT" in mymoves and foemoves & (SETUP | RECOVERY | HAZARD_STATUS):
            add("taunt", accept=["taunt"], desc=f"{label}: foe has {sorted(foemoves & (SETUP | RECOVERY | HAZARD_STATUS))}; bot played {chosen}")
        if "ENCORE" in mymoves and st.side_two.last_used_move and st.side_two.last_used_move.upper().replace("MOVE:", "") in (SETUP | RECOVERY | HAZARD_STATUS):
            add("encore", accept=["encore"], desc=f"{label}: foe just used {st.side_two.last_used_move}; bot played {chosen}")
        # --- 6: Tera spent
        if tera_used:
            add("tera", reject=[chosen], fold_tera=False, desc=f"{label}: bot terastallized with {base}")
        # --- 7: sleep clause shield: one of our benched/active mons is asleep and the foe holds sleep moves
        asleep = [m.id for m in st.side_one.pokemon if m.status == "SLEEP" and m.hp > 0]
        foe_sleep = [m.id for m in st.side_two.pokemon if m.hp > 0 and any(mv.id in SLEEP_MOVES for mv in m.moves)]
        if asleep and foe_sleep:
            add("sleep_clause", desc=f"{label}: our {asleep} asleep, foe sleep users {foe_sleep}; bot played {chosen}")
        # --- 9: Palafin never pivots out
        if me.id == "PALAFIN":
            add("pivot", accept=["flipturn"] if "FLIPTURN" in mymoves else [], reject=[],
                desc=f"{label}: Palafin (Zero form) on the field; bot played {chosen}")
        # --- 8: role: Blissey / Chansey available on the bench as a special wall
        bench_walls = [m.id for i, m in enumerate(st.side_one.pokemon)
                       if i != int(st.side_one.active_index) and m.hp > 0 and m.id in ("BLISSEY", "CHANSEY")]
        if bench_walls:
            add("role", desc=f"{label}: {bench_walls} on our bench; bot played {chosen}")
    return cands


# --------------------------------------------------------------------------- search

_EV: dict = {}


def _search(args):
    from showdown_bot.players._worker import search_world

    state, ms, evaluator, scale = args
    return search_world(state, ms, 1, evaluator, scale)


def policy(results: list[tuple], fold_tera: bool = True) -> dict[str, float]:
    pol: dict[str, float] = {}
    usable = [r for r in results if r[0] > 0 and r[1]]
    for r in usable:
        tot = sum(v for _, v, _ in r[1]) or r[0]
        for c, v, _ in r[1]:
            k = norm(c, fold_tera)
            pol[k] = pol.get(k, 0.0) + v / tot / max(1, len(usable))
    return pol


def passes(pol: dict[str, float], pz: dict) -> bool:
    if not pol:
        return False
    best = max(pol, key=pol.get)
    if pz.get("accept"):
        return best in set(pz["accept"])
    if pz.get("reject"):
        return best not in set(pz["reject"])
    return True


def search_puzzles(pool, puzzles, ms: int, worlds: int, reps: int, evaluator, scale):
    """Per puzzle: list of ``reps`` policies (each from ``worlds`` searches)."""
    jobs = []
    for pz in puzzles:
        for r in range(reps):
            for w in range(worlds):
                jobs.append((pz["states"][(r * worlds + w) % len(pz["states"])], ms, evaluator, scale))
    flat = list(pool.map(_search, jobs, chunksize=1))
    out, k = [], 0
    for pz in puzzles:
        pols = []
        for _ in range(reps):
            pols.append(policy(flat[k:k + worlds], pz.get("fold_tera", True)))
            k += worlds
        out.append(pols)
    return out


def spec_label(evaluator, scale) -> str:
    return f"{Path(evaluator).stem if evaluator else 'stock'}@{scale}"


def run_suite(puzzles, evaluator, scale, times, worlds, reps, workers):
    """Returns ``{ms: {id: pass_fraction}}``."""
    res: dict[int, dict[str, float]] = {}
    with ProcessPoolExecutor(workers) as pool:
        for ms in times:
            t0 = time.time()
            pols = search_puzzles(pool, puzzles, ms, worlds, reps, evaluator, scale)
            res[ms] = {pz["id"]: sum(passes(p, pz) for p in ps) / len(ps) for pz, ps in zip(puzzles, pols)}
            print(f"# {spec_label(evaluator, scale)} {ms} ms done in {time.time() - t0:.0f}s", flush=True)
    return res


def _row_of(p: dict) -> str:
    return p["cat"] + (" (ctl)" if p.get("kind") == "control" else "")


def table(puzzles, res) -> str:
    cats = sorted({_row_of(p) for p in puzzles})
    times = list(res)
    lines = [f"{'category':<16}{'n':>4}" + "".join(f"{str(ms) + 'ms':>9}" for ms in times)]
    for c in cats + ["ALL"]:
        ids = [p["id"] for p in puzzles if c == "ALL" or _row_of(p) == c]
        row = f"{c:<16}{len(ids):>4}"
        for ms in times:
            row += f"{sum(res[ms][i] for i in ids) / len(ids):>9.2f}"
        lines.append(row)
    return "\n".join(lines)


def load_suite(path: Path = SUITE, cats: str = "", tier: str = "validated") -> list[dict]:
    puzzles = json.loads(Path(path).read_text())
    want = {c for c in cats.split(",") if c}
    return [p for p in puzzles if (not want or p["cat"] in want) and tier in ("all", p.get("tier", "validated"))]


# --------------------------------------------------------------------------- played-out check
#
# Independent evidence: force the first action, then let MCTS (stock evaluation, short budget)
# choose every later move for BOTH sides until one side has no Pokemon left, and count wins.
# Unlike scripts/puzzles.py searchroll the continuation keeps Terastallization (a puzzle about
# *spending* the Tera needs the later turns to be able to use it).  Damage in the engine's
# instruction generation is the same as in the search (rolls averaged except kill/crit
# branches), so this checks the *value judgement*, not the roll model.

def _swap(state_string: str) -> str:
    parts = state_string.split("/")
    return "/".join([parts[1], parts[0]] + parts[2:])


def _alive(side) -> int:
    return sum(1 for m in side.pokemon if m.hp > 0)


def _must_switch(side) -> bool:
    return bool(side.force_switch) or active(side).hp == 0


def _choose(pe, st, side: int, ms: int, rng) -> str:
    s = st if side == 1 else pe.State.from_string(_swap(st.to_string()))
    res = pe.monte_carlo_tree_search(s, ms)
    opts = [(o.move_choice, int(o.visits)) for o in res.side_one]
    best = max(v for _, v in opts)
    keep = [(c, v) for c, v in opts if v >= 0.75 * best]
    pick = rng.choices([c for c, _ in keep], weights=[v for _, v in keep])[0]
    return "none" if pick.lower() == "no move" else pick


def playout(args) -> float:
    """Win rate (draw 0.5) of ``first`` followed by MCTS-vs-MCTS play. args=(state, first, n, ms, seed)."""
    import random

    import poke_engine as pe

    state_string, first, n, ms, seed = args
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(n):
        st = pe.State.from_string(state_string)
        for turn in range(45):
            if _alive(st.side_one) == 0 or _alive(st.side_two) == 0:
                break
            f1, f2 = _must_switch(st.side_one), _must_switch(st.side_two)
            if f1 and not f2:
                m1, m2 = _choose(pe, st, 1, ms, rng), "none"
            elif f2 and not f1:
                m1, m2 = "none", _choose(pe, st, 2, ms, rng)
            else:
                m1 = first if turn == 0 else _choose(pe, st, 1, ms, rng)
                m2 = _choose(pe, st, 2, ms, rng)
            ins = pe.generate_instructions(st, m1.removeprefix("switch "), m2.removeprefix("switch "))
            r, acc, pick = rng.random() * 100.0, 0.0, ins[-1]
            for i in ins:
                acc += i.percentage
                if r < acc:
                    pick = i
                    break
            st = st.apply_instructions(pick)
        a, b = _alive(st.side_one), _alive(st.side_two)
        wins += 1.0 if (b == 0 and a > 0) else (0.5 if (a > 0 and b > 0) or (a == 0 and b == 0) else 0.0)
    return wins / n


def playout_actions(pz: dict) -> list[str]:
    """Which first actions to play out: the intended ones, the live choice and the long search's top."""
    top = [c for c in (pz.get("long", {}).get("net") or {})]
    fold = pz.get("fold_tera", True)
    chosen = norm(pz["source"]["chosen"], fold)
    if pz.get("accept"):
        intended = [c for c in top if c in pz["accept"]][:2] or pz["accept"][:2]
        others = [c for c in [chosen] + top[:2] if c not in pz["accept"]]
        return list(dict.fromkeys(intended + others))[:4]
    others = [c for c in top if c not in pz["reject"]][:2]  # reject-type: the rejected action vs the best others
    return list(dict.fromkeys(pz["reject"][:1] + others))


def _full_name(state_string: str, short: str, fold_tera: bool) -> str:
    """Engine choice string for a normalised action (``toxapex`` -> ``switch toxapex``)."""
    import poke_engine as pe

    res = pe.monte_carlo_tree_search(pe.State.from_string(state_string), 10, iterations=200)
    for o in res.side_one:
        if norm(o.move_choice, fold_tera) == short and not (fold_tera and o.move_choice.endswith("-tera")):
            return o.move_choice
    for o in res.side_one:
        if norm(o.move_choice, fold_tera) == short:
            return o.move_choice
    raise KeyError(short)


def cmd_playout(a):
    cands = json.loads(Path(a.cands).read_text())
    if a.only:
        cands = [c for c in cands if c["id"] in a.only.split(",") or c["cat"] in a.only.split(",")]
    seen: Counter = Counter()
    sel = []
    for c in cands:
        key = (c["cat"], c["source"]["game"], c["desc"].split(" ")[0])
        seen[key] += 1
        if a.max_per_seq and seen[key] > a.max_per_seq:
            continue
        sel.append(c)
    jobs, owners = [], []
    for k, pz in enumerate(sel):
        fold = pz.get("fold_tera", True)
        for act in playout_actions(pz):
            full = _full_name(pz["states"][0], act, fold)
            jobs.append((pz["states"][0], full, a.n, a.ms, zlib.crc32((pz["id"] + act).encode()) + a.seed))
            owners.append((k, act))
    print(f"{len(sel)} candidates, {len(jobs)} playout jobs", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        vals = list(pool.map(playout, jobs, chunksize=1))
    for (k, act), v in zip(owners, vals):
        sel[k].setdefault("playout", {"n": a.n, "ms": a.ms, "values": {}})["values"][act] = round(v, 3)
    Path(a.out).write_text(json.dumps(sel, indent=1))
    for pz in sel:
        print(pz["id"], pz["cat"], pz["desc"][:60], pz.get("playout", {}).get("values"), flush=True)
    return 0


# --------------------------------------------------------------------------- building the suite

def long_agrees(pz: dict) -> bool | None:
    """Do the long searches (gen2 net and stock evaluation) both put the intended answer on top?"""
    lg = pz.get("long") or {}
    if not (pz.get("accept") or pz.get("reject")) or not lg.get("net") or not lg.get("stock"):
        return None
    return all(passes({k: v for k, v in lg[name].items()}, pz) for name in ("net", "stock"))


def playout_gap(pz: dict) -> float | None:
    """Best intended action's played-out win rate minus the best other evaluated action's."""
    vals = (pz.get("playout") or {}).get("values") or {}
    if pz.get("accept"):
        good = [v for k, v in vals.items() if k in pz["accept"]]
        bad = [v for k, v in vals.items() if k not in pz["accept"]]
    else:  # reject-type: the intended answer is anything but the rejected action
        good = [v for k, v in vals.items() if k not in pz["reject"]]
        bad = [v for k, v in vals.items() if k in pz["reject"]]
    if not good or not bad:
        return None
    return max(good) - max(bad)


def evidence_of(pz: dict, confirm: dict | None, min_gap: float, confirm_gap: float) -> dict | None:
    """The puzzle's validation record, or None when neither check supports the intended answer."""
    ev: dict = {}
    gap = playout_gap(pz)
    if gap is not None and gap <= -0.1:
        return None  # the played-out check contradicts the intended answer
    lc = pz.get("long_cfg") or {}
    if long_agrees(pz) and lc.get("seconds", 0) * lc.get("worlds", 0) >= 8:
        ev["long_search"] = {"seconds": lc["seconds"], "worlds": lc["worlds"],
                             "net": pz["long"]["net"], "stock": pz["long"]["stock"]}
    if gap is not None and gap >= min_gap:
        if confirm is None or (playout_gap(confirm) or -1) >= confirm_gap:
            ev["playout"] = {**pz["playout"], "gap": round(gap, 3),
                             "confirm": None if confirm is None else {**confirm["playout"], "gap": round(playout_gap(confirm), 3)}}
    return ev or None


def describe_why(pz: dict, ev: dict) -> str:
    bits = []
    if "long_search" in ev:
        ls = ev["long_search"]
        bits.append(f"a {ls['seconds']:g} s x {ls['worlds']}-world search with the gen2 net and with the stock evaluation "
                    f"both pick {', '.join(sorted(pz.get('accept') or ['not ' + '/'.join(pz['reject'])]))}")
    if "playout" in ev:
        p = ev["playout"]
        vals = ", ".join(f"{k} {v:.2f}" for k, v in sorted(p["values"].items(), key=lambda kv: -kv[1]))
        bits.append(f"played out with MCTS vs MCTS ({p['n']} games, {p['ms']} ms/move, stock eval, first move forced): "
                    f"{vals} (gap {p['gap']:+.2f})")
    return "; ".join(bits)


def cmd_build(a):
    cands = [c for path in a.cands for c in json.loads(Path(path).read_text())]
    confirms = {c["id"]: c for path in a.confirm for c in json.loads(Path(path).read_text())}
    review = set(a.review.split(",")) if a.review else set()
    caps = {k: int(v) for k, v in (kv.split("=") for kv in a.long_cap.split(",") if kv)}
    # positions that only the long search supports are plentiful; cap them per category, real
    # ladder positions first, then a seeded shuffle (never a choice by how the bot fares on them)
    import random as _random
    order = sorted(range(len(cands)), key=lambda i: (cands[i]["source"]["game"] == "synthetic",
                                                     _random.Random(f"{a.seed}:{cands[i]['id']}").random()))
    taken: Counter = Counter()
    chosen_ids = set()
    for i in order:
        pz = cands[i]
        ev = evidence_of(pz, confirms.get(pz["id"]) if a.confirm else None, a.min_gap, a.confirm_gap)
        if ev is None:
            continue
        if "playout" not in ev and pz["id"] not in review:
            key = pz["cat"] + ("_ctl" if pz.get("kind") == "control" else "")
            if taken[key] >= caps.get(key, 10**6):
                continue
            taken[key] += 1
        chosen_ids.add(pz["id"])
    kept, dropped = [], Counter()
    counters: Counter = Counter()
    for pz in cands:
        ev = evidence_of(pz, confirms.get(pz["id"]) if a.confirm else None, a.min_gap, a.confirm_gap)
        if ev is not None and pz["id"] not in chosen_ids:
            ev = None
        tier = "validated"
        if ev is None and pz["id"] in review:
            # a position from the human game review that neither check can confirm: kept as a
            # separate tier (reported on its own, never mixed into the validated pass rates)
            tier = "review"
            ev = {"review": {"long_search": pz.get("long"), "playout": pz.get("playout"),
                             "note": "from the ladder-game review; neither the 2 s x 4-world search nor the "
                                     "played-out check (same evaluator on both sides) supports it"}}
        if ev is None:
            dropped[pz["cat"]] += 1
            continue
        counters[pz["cat"]] += 1
        kept.append({
            "id": f"{pz['cat']}-{counters[pz['cat']]:02d}", "cat": pz["cat"], "kind": pz.get("kind", "positive"),
            "tier": tier, "desc": pz["desc"],
            "why": describe_why(pz, ev) if tier == "validated" else
            "human review: the intended move is what a player would choose here; the bot's own long search disagrees",
            "states": pz["states"][: a.max_states],
            "accept": pz.get("accept", []), "reject": pz.get("reject", []), "fold_tera": pz.get("fold_tera", True),
            "source": pz["source"], "evidence": ev,
        })
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(kept, indent=1))
    print("kept", len(kept), dict(Counter((p["cat"], p["tier"]) for p in kept)), "dropped", dict(dropped))
    return 0


def cmd_mine(a):
    cands = mine(a.records)
    for i, c in enumerate(cands):
        c["id"] = f"c{i:03d}"
    Path(a.out).write_text(json.dumps(cands, indent=1))
    print(len(cands), Counter(c["cat"] for c in cands))
    return 0


def cmd_verify(a):
    """Long search of every candidate with each evaluator; stores the policy distribution."""
    cands = json.loads(Path(a.cands).read_text())
    if a.only:
        cands = [c for c in cands if c["id"] in a.only.split(",") or c["cat"] in a.only.split(",")]
    specs = [("net", a.evaluator, a.scale), ("stock", None, 1.0)]
    with ProcessPoolExecutor(a.workers) as pool:
        for label, ev, sc in specs:
            pols = search_puzzles(pool, cands, int(a.seconds * 1000), a.worlds, 1, ev, sc)
            for c, ps in zip(cands, pols):
                c.setdefault("long", {})[label] = {k: round(v, 3) for k, v in sorted(ps[0].items(), key=lambda kv: -kv[1])[:5]}
                c["long_cfg"] = {"seconds": a.seconds, "worlds": a.worlds}
            print("# verified", label, flush=True)
    Path(a.out).write_text(json.dumps(cands, indent=1))
    return 0


def cmd_run(a):
    puzzles = load_suite(Path(a.suite), a.cats, a.tier)
    if a.mode != "argmax":
        raise SystemExit("only mode=argmax is implemented")
    times = [int(t) for t in a.time_ms.split(",")]
    res = run_suite(puzzles, a.evaluator, a.scale, times, a.worlds, a.reps, a.workers)
    print(f"\n{spec_label(a.evaluator, a.scale)} worlds={a.worlds} reps={a.reps} mode={a.mode} tier={a.tier}")
    for tier in ("validated", "review"):
        part = [p for p in puzzles if p.get("tier", "validated") == tier]
        if part:
            print(f"\n[{tier}]")
            print(table(part, res))
    if a.out:
        Path(a.out).write_text(json.dumps({"spec": {"evaluator": a.evaluator, "scale": a.scale, "worlds": a.worlds,
                                                    "reps": a.reps, "mode": a.mode}, "pass": res}, indent=1))
    if a.failures:
        for ms in times:
            print(f"\nfailing at {ms} ms:", [i for i, v in res[ms].items() if v < 0.5])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--records", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=cmd_mine)
    v = sub.add_parser("verify")
    v.add_argument("--cands", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--only", default="")
    v.add_argument("--seconds", type=float, default=3.0)
    v.add_argument("--worlds", type=int, default=4)
    v.add_argument("--evaluator", default=None)
    v.add_argument("--scale", type=float, default=1.5)
    v.add_argument("--workers", type=int, default=4)
    v.set_defaults(fn=cmd_verify)
    pl = sub.add_parser("playout")
    pl.add_argument("--cands", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--only", default="")
    pl.add_argument("--n", type=int, default=30)
    pl.add_argument("--ms", type=int, default=30)
    pl.add_argument("--max-per-seq", type=int, default=2,
                    help="keep at most this many candidates per (category, game, our species)")
    pl.add_argument("--workers", type=int, default=4)
    pl.add_argument("--seed", type=int, default=0, help="offsets the playout RNG (use a new one to confirm)")
    pl.set_defaults(fn=cmd_playout)
    b = sub.add_parser("build")
    b.add_argument("--cands", action="append", required=True, help="verified/played-out candidate files")
    b.add_argument("--confirm", action="append", default=[],
                   help="a second, independent playout of the same ids; a playout-only puzzle must clear --confirm-gap there too")
    b.add_argument("--out", default=str(SUITE))
    b.add_argument("--min-gap", type=float, default=0.25)
    b.add_argument("--confirm-gap", type=float, default=0.15)
    b.add_argument("--max-states", type=int, default=4)
    b.add_argument("--long-cap", default="", help="per-category cap on long-search-only puzzles, cat=n,cat=n")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--review", default="", help="candidate ids kept as the unvalidated 'review' tier")
    b.set_defaults(fn=cmd_build)
    r = sub.add_parser("run")
    r.add_argument("--suite", default=str(SUITE))
    r.add_argument("--cats", default="")
    r.add_argument("--tier", choices=("validated", "review", "all"), default="validated",
                   help="validated = positions a long search or a played-out check supports (default); "
                        "review = positions from the human game review that no check confirms")
    r.add_argument("--evaluator", default=None, help="learned leaf evaluator .pev (default: stock evaluation)")
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--time-ms", default="100,1000")
    r.add_argument("--worlds", type=int, default=4)
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--mode", default="argmax")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", default="")
    r.add_argument("--failures", action="store_true")
    r.set_defaults(fn=cmd_run)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
