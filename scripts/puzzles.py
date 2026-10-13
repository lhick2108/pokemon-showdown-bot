"""Decision-quality puzzles (bead 589.32): state builder, evidence tools and pass-rate runner.

    python scripts/puzzles.py run --times 100,300,1000,3000 --worlds 10   # bot pass rates
    python scripts/puzzles.py run --knee 0.35 ...                         # with the 589.32 eval variant
    python scripts/puzzles.py searchroll --pool pool.json --out out.json  # evidence for candidates
    python scripts/puzzles.py rollouts --n 60                             # cheap scripted-policy screen
    python scripts/puzzles.py verify --seconds 8                          # deep search of every puzzle
    python scripts/finalize_puzzles.py ...                                # candidate pools -> tests/puzzles

Puzzles (tests/puzzles/{stall,sacrifice}.json) are plain engine State strings (the opponent's
sets are fully known, so no sampling noise) plus the set of acceptable first moves.  ``run``
samples the final move exactly as ``SearchPlayer`` does (2 worlds, visit shares averaged,
``mixed`` rule with the 0.75 threshold) and reports the expected pass rate, computed in closed
form from the mixed rule over every pair of searched worlds.

Why the evidence is what it is: a puzzle's intended answer is only kept if playing the line out
with MCTS-vs-MCTS (``searchroll``, both sides on the stock evaluation, 30 ms per move, the
first action forced) makes the intended group of actions win clearly more often (>= 0.20 in
a fresh sample) than the other group.  Scripted-policy rollouts (``rollouts``: greedy foe that
never switches) were found to disagree with that often, so they are only used as a screen.
A puzzle is a diagnostic, not a target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUZZLE_DIR = ROOT / "tests" / "puzzles"
MODEL = Path(r"C:\Users\Lewis\pokemon-showdown-bot\models\gen2-res-searchabs-mlp16.pev")
MODEL_SCALE = 1.5
MIXED_THRESHOLD = 0.75

_LIB: dict[str, str] | None = None
FORCED_PREF: list[str] = []  # our preferred replacement order after a faint (rollouts)


def lib() -> dict[str, str]:
    global _LIB
    if _LIB is None:
        _LIB = json.loads((PUZZLE_DIR / "monlib.json").read_text())
    return _LIB


# --------------------------------------------------------------------------- builder

def mon(species: str, hp: float = 1.0, item: str | None = None, ability: str | None = None,
        moves: list[str] | None = None, status: str | None = None, pp: int = 16) -> str:
    """One engine Pokemon string from the species library with overrides.

    ``hp`` is a fraction of max HP (0 = fainted).  ``moves`` replaces the move list.
    """
    f = lib()[species.upper()].split(",")
    maxhp = int(f[7])
    f[6] = str(0 if hp <= 0 else max(1, round(hp * maxhp)))
    if item is not None:
        f[10] = item.upper()
    if ability is not None:
        f[8] = f[9] = ability.upper()
    if status is not None:
        f[18] = status.upper()
    if moves is not None:
        ms = [m.upper() for m in moves] + ["NONE"] * (4 - len(moves))
        for i, m in enumerate(ms[:4]):
            f[22 + i] = f"{m};false;{0 if m == 'NONE' else (8 if m in RECOVERY else pp)}"
    return ",".join(f)


RECOVERY = {"RECOVER", "ROOST", "SLACKOFF", "SOFTBOILED", "MOONLIGHT", "SYNTHESIS", "MORNINGSUN",
            "STRENGTHSAP", "WISH", "MILKDRINK", "SHOREUP", "HEALORDER", "REST"}


def _tail() -> tuple[list[str], list[str]]:
    # a clean (no boosts / volatiles / hazards) side tail and the state tail
    side_tail = ["0", "0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0;0", "", "0;0;0;0;0;0", "0", "0", "0", "0",
                 "0", "0", "0", "0", "0", "0", "0", "0", "false", "NONE", "false", "false", "false",
                 "switch:0", "false"]
    return side_tail, ["NONE;0", "NONE;0", "false;0", "false"]


def side(mons: list[str], active: int = 0, *, rocks: int = 0, spikes: int = 0, tspikes: int = 0,
         web: int = 0, boosts: dict[str, int] | None = None, vs: str = "") -> str:
    """A side string: ``mons`` (up to 6, padded with fainted copies), hazards and boosts."""
    mons = list(mons)
    pad = mons[0].split(",")
    pad[6] = "0"
    pad[18] = "NONE"
    while len(mons) < 6:
        mons.append(",".join(pad))
    tail, _ = _tail()
    tail[0] = str(active)
    sc = tail[1].split(";")
    sc[12], sc[13], sc[14], sc[17] = str(spikes), str(rocks), str(web), str(tspikes)
    tail[1] = ";".join(sc)
    tail[2] = vs
    for k, v in (boosts or {}).items():  # atk def spa spd spe
        tail[{"atk": 5, "def": 6, "spa": 7, "spd": 8, "spe": 9}[k]] = str(v)
    return "=".join(mons + tail)


def state(ours: str, theirs: str) -> str:
    return "/".join([ours, theirs] + _tail()[1])


# --------------------------------------------------------------------------- search

def _worker_search(args):
    sys.path.insert(0, str(ROOT / "src"))
    from showdown_bot.players._worker import search_world
    st, ms, ev, scale, knee, bscale = args
    return search_world(st, ms, 1, ev, scale, "ucb1", 0.1, False, knee, bscale)


def norm(choice: str) -> str:
    """Fold ``-tera`` into the base move and ``switch x`` into the species id ``x``."""
    choice = choice[:-5] if choice.endswith("-tera") else choice
    return choice[7:] if choice.startswith("switch ") else choice


def policy_of(results: list[tuple]) -> dict[str, float]:
    """Visit-share average over worlds, ``-tera`` variants folded into the base choice."""
    pol: dict[str, float] = {}
    usable = [r for r in results if r[0] > 0 and r[1]]
    for r in usable:
        tot = sum(v for _, v, _ in r[1]) or r[0]
        for c, v, _ in r[1]:
            pol[norm(c)] = pol.get(norm(c), 0.0) + v / tot / len(usable)
    return pol


def mixed_probs(pol: dict[str, float], threshold: float = MIXED_THRESHOLD) -> dict[str, float]:
    best = max(pol.values())
    keep = {c: m for c, m in pol.items() if m >= best * threshold}
    z = sum(keep.values())
    return {c: m / z for c, m in keep.items()}


def pass_rate(world_results: list[tuple], accept: set[str], reject: set[str]) -> tuple[float, dict]:
    """Expected pass probability over all pairs of worlds (the player searches 2 worlds)."""
    total, n, dist = 0.0, 0, {}
    for a, b in itertools.combinations(range(len(world_results)), 2):
        probs = mixed_probs(policy_of([world_results[a], world_results[b]]))
        for c, p in probs.items():
            dist[c] = dist.get(c, 0.0) + p
        ok = sum(p for c, p in probs.items() if (c in accept if accept else c not in reject))
        total += ok
        n += 1
    return total / n, {c: v / n for c, v in sorted(dist.items(), key=lambda kv: -kv[1])}


def mean_values(world_results: list[tuple]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in world_results:
        for c, v, s in r[1]:
            if v:
                acc.setdefault(norm(c), []).append(s / v)
    return {c: sum(x) / len(x) for c, x in acc.items()}


def load_puzzles(path: Path | None = None) -> list[dict]:
    out: list[dict] = []
    for f in sorted((path or PUZZLE_DIR).glob("*.json")):
        if f.name == "monlib.json" or f.name.startswith("results"):
            continue
        out.extend(json.loads(f.read_text()))
    return out


def search_many(pool, states: list[str], ms: int, worlds: int, evaluator: str | None,
                scale: float, knee: float = 0.0, bscale: float = 1.0) -> list[list[tuple]]:
    jobs = [(s, ms, evaluator, scale, knee, bscale) for s in states for _ in range(worlds)]
    flat = list(pool.map(_worker_search, jobs, chunksize=1))
    return [flat[i * worlds:(i + 1) * worlds] for i in range(len(states))]


# --------------------------------------------------------------------------- rollouts
#
# Evaluator-independent check of a puzzle's intended answer: play the game out to the end
# with fixed, simple continuation policies (no search, no evaluation function) and compare
# the win rate of each first action.  Both sides use the same simple policy families, so
# a first action wins here only because of what the engine's rules (damage, recoil,
# healing, switching) really do.

def _legal(st, side: int) -> list[str]:
    s = st.side_one if side == 1 else st.side_two
    active = int(s.active_index)
    mons = list(s.pokemon)
    switches = [m.id.lower() for i, m in enumerate(mons) if i != active and m.hp > 0]
    if s.force_switch or mons[active].hp == 0:
        return switches
    cur = mons[active]
    moves = [m.id.lower() for m in cur.moves if m.id != "NONE" and m.pp > 0 and not m.disabled]
    return moves + switches


def _switch_ids(st, side: int) -> set[str]:
    s = st.side_one if side == 1 else st.side_two
    active = int(s.active_index)
    return {m.id.lower() for i, m in enumerate(s.pokemon) if i != active and m.hp > 0}


def _must_switch(s) -> bool:
    return bool(s.force_switch) or s.pokemon[int(s.active_index)].hp == 0


def _alive(s) -> int:
    return sum(1 for m in s.pokemon if m.hp > 0)


def _exp_damage(pe, st, side: int, move: str) -> float:
    """Expected fraction of the foe's max HP removed by ``move`` this turn (foe idle)."""
    if side == 1:
        foe = st.side_two.pokemon[int(st.side_two.active_index)]
        ins = pe.generate_instructions(st, move, "none")
    else:
        foe = st.side_one.pokemon[int(st.side_one.active_index)]
        ins = pe.generate_instructions(st, "none", move)
    total = 0.0
    for i in ins:
        after = st.apply_instructions(i)
        f = after.side_two if side == 1 else after.side_one
        hp = f.pokemon[int(f.active_index)].hp
        total += i.percentage / 100.0 * max(0, foe.hp - hp)
    return total / max(1, foe.maxhp)


# moves a naive "max expected damage" rollout policy would otherwise abuse (they end the user)
KAMIKAZE = {"EXPLOSION", "SELFDESTRUCT", "MISTYEXPLOSION", "FINALGAMBIT", "MEMENTO", "HEALINGWISH",
            "LUNARDANCE", "DESTINYBOND", "PERISHSONG", "CURSE"}


def policy_greedy(pe, st, side, rng, heal_below=None, heal_moves=()):
    """Highest expected damage; optionally heal when below ``heal_below`` of max HP."""
    legal = [m for m in _legal(st, side) if m not in _switch_ids(st, side)] or _legal(st, side)
    legal = [m for m in legal if m.upper() not in KAMIKAZE] or legal
    s = st.side_one if side == 1 else st.side_two
    if _must_switch(s):
        pref = [p for p in FORCED_PREF if p in legal] if side == 1 else []
        return pref[0] if pref else (rng.choice(legal) if legal else "none")
    cur = s.pokemon[int(s.active_index)]
    if heal_below is not None and cur.hp < heal_below * cur.maxhp:
        for m in legal:
            if m.upper() in heal_moves:
                return m
    best, best_v = None, -1.0
    for m in legal:
        if m.upper() in heal_moves:
            continue
        v = _exp_damage(pe, st, side, m)
        if v > best_v:
            best, best_v = m, v
    return best or (legal[0] if legal else "none")


def policy_random_attacker(pe, st, side, rng):
    """Uniform over the moves that do damage (the foe model that is not a max-damage bot)."""
    legal = [m for m in _legal(st, side) if m not in _switch_ids(st, side)]
    legal = [m for m in legal if m.upper() not in KAMIKAZE] or legal
    s = st.side_one if side == 1 else st.side_two
    if _must_switch(s) or not legal:
        return policy_greedy(pe, st, side, rng)
    dmg = [m for m in legal if _exp_damage(pe, st, side, m) > 0.01]
    return rng.choice(dmg or legal)


def policy_stall(pe, st, side, rng):
    """Always use a recovery move if the active mon has one, else greedy."""
    s = st.side_one if side == 1 else st.side_two
    if not _must_switch(s):
        for m in _legal(st, side):
            if m.upper() in RECOVERY and m.upper() != "WISH":
                return m
    return policy_greedy(pe, st, side, rng)


def rollout(state_str: str, first: str, ours, theirs, n: int, seed: int = 0,
            max_turns: int = 60, their_first: str | None = None) -> float:
    """Win rate (draw = 0.5) of ``first`` followed by policy ``ours`` against ``theirs``."""
    import random

    import poke_engine as pe
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(n):
        st = pe.State.from_string(state_str)
        turn = 0
        while turn < max_turns:
            if _alive(st.side_one) == 0 or _alive(st.side_two) == 0:
                break
            f1, f2 = _must_switch(st.side_one), _must_switch(st.side_two)
            m1 = first if turn == 0 else ours(pe, st, 1, rng)
            if f1 and m1 not in _switch_ids(st, 1):
                m1 = rng.choice(_legal(st, 1) or ["none"])
            m2 = (their_first if (turn == 0 and their_first) else theirs(pe, st, 2, rng))
            if f2 and m2 not in _switch_ids(st, 2):
                m2 = rng.choice(_legal(st, 2) or ["none"])
            if f1 and not f2:
                m2 = "none"
            if f2 and not f1:
                m1 = "none"
            ins = pe.generate_instructions(st, m1, m2)
            r, acc = rng.random() * 100.0, 0.0
            pick = ins[-1]
            for i in ins:
                acc += i.percentage
                if r < acc:
                    pick = i
                    break
            st = st.apply_instructions(pick)
            turn += 1
        a, b = _alive(st.side_one), _alive(st.side_two)
        wins += 1.0 if (b == 0 and a > 0) else (0.5 if (a > 0 and b > 0) or (a == 0 and b == 0) else 0.0)
    return wins / n


def rollout_table(state_str: str, n: int = 200, heal_below: float = 0.6, foe: str = "greedy") -> dict:
    """Win rate of every legal first action under three continuation policies.

    Opponent: greedy highest-damage attacker.  Ours: ``greedy`` (always attack),
    ``stall`` (always recover when possible) and ``smart`` (attack, recover below
    ``heal_below``).  Rows are first actions.
    """
    import functools

    import poke_engine as pe
    st = pe.State.from_string(state_str)
    opp = policy_greedy if foe == "greedy" else policy_random_attacker
    conts = {
        "greedy": lambda pe_, s, side, rng: policy_greedy(pe_, s, side, rng),
        "stall": policy_stall,
        "smart": lambda pe_, s, side, rng: policy_greedy(pe_, s, side, rng, heal_below, tuple(RECOVERY)),
    }
    table: dict[str, dict[str, float]] = {}
    for a1 in _legal(st, 1):
        table[a1] = {k: rollout(state_str, a1, c, opp, n, seed=zlib.crc32(a1.encode())) for k, c in conts.items()}
    return table


# --------------------------------------------------------------------------- search rollouts
#
# Second, stronger check: play the line out with the engine's own MCTS choosing every later
# move for BOTH sides (stock evaluation, short budget).  The first action is forced; the
# result is the real win rate of that action when both players then play sensibly.  It still
# shares the bot's evaluation function, but only for the continuation, so a horizon/eval
# bias at the root is replaced by actual game outcomes.

def swap_sides(state_str: str) -> str:
    parts = state_str.split("/")
    return "/".join([parts[1], parts[0]] + parts[2:])


def _mcts_choice(pe, st, side: int, ms: int, rng) -> str:
    s = st if side == 1 else pe.State.from_string(swap_sides(st.to_string()))
    res = pe.monte_carlo_tree_search(s, ms)
    opts = [(norm(o.move_choice), int(o.visits)) for o in res.side_one]
    best = max(v for _, v in opts)
    keep = [(c, v) for c, v in opts if v >= MIXED_THRESHOLD * best]
    pick = rng.choices([c for c, _ in keep], weights=[v for _, v in keep])[0]
    return "none" if pick.lower() == "no move" else pick


def search_rollout(args) -> float:
    """Win rate (draw 0.5) of ``first`` followed by MCTS-vs-MCTS play. args=(state, first, n, ms, seed)."""
    import random
    sys.path.insert(0, str(ROOT / "src"))
    import poke_engine as pe
    state_str, first, n, ms, seed = args
    rng = random.Random(seed)
    wins = 0.0
    for _ in range(n):
        st = pe.State.from_string(state_str)
        for turn in range(40):
            if _alive(st.side_one) == 0 or _alive(st.side_two) == 0:
                break
            f1, f2 = _must_switch(st.side_one), _must_switch(st.side_two)
            if f1 and not f2:
                m1 = _mcts_choice(pe, st, 1, ms, rng)
                m2 = "none"
            elif f2 and not f1:
                m1 = "none"
                m2 = _mcts_choice(pe, st, 2, ms, rng)
            else:
                m1 = first if turn == 0 else _mcts_choice(pe, st, 1, ms, rng)
                m2 = _mcts_choice(pe, st, 2, ms, rng)
            ins = pe.generate_instructions(st, m1, m2)
            r, acc = rng.random() * 100.0, 0.0
            pick = ins[-1]
            for i in ins:
                acc += i.percentage
                if r < acc:
                    pick = i
                    break
            st = st.apply_instructions(pick)
        a, b = _alive(st.side_one), _alive(st.side_two)
        wins += 1.0 if (b == 0 and a > 0) else (0.5 if (a > 0 and b > 0) or (a == 0 and b == 0) else 0.0)
    return wins / n


def cmd_searchroll(args) -> int:
    import poke_engine as pe
    if args.pool:  # an arbitrary list of {id, state, ...} candidates; results go to --out, nothing is written back
        files, data = [], {}
        puzzles = json.loads(Path(args.pool).read_text())
        args.write = False
    else:
        files = [f for f in sorted(PUZZLE_DIR.glob("*.json")) if f.name != "monlib.json" and not f.name.startswith("results")]
        data = {f: json.loads(f.read_text()) for f in files}
        puzzles = [p for f in files for p in data[f]
                   if (not args.only or p["id"] in args.only.split(",")) and (not args.theme or p["theme"] == args.theme)]
    jobs, owners = [], []
    for k, pz in enumerate(puzzles):
        st = pe.State.from_string(pz["state"])
        for a in _legal(st, 1):
            jobs.append((pz["state"], a, args.n, args.ms, zlib.crc32((pz["id"] + a).encode())))
            owners.append((k, a))
    with ProcessPoolExecutor(args.workers) as pool:
        vals = list(pool.map(search_rollout, jobs, chunksize=1))
    per = {}
    for (k, a), v in zip(owners, vals):
        per.setdefault(k, {})[a] = round(v, 2)
    for k, pz in enumerate(puzzles):
        pz["verify_search"] = {"values": per[k], "n": args.n, "ms": args.ms}
        if pz.get("accept") or pz.get("reject"):
            good, bad = best_values(pz, {a: {"x": v} for a, v in per[k].items()})
            print(f"{pz['id']:<5}{good - bad:+.2f} intended={good:.2f} other={bad:.2f} {per[k]}", flush=True)
        else:
            print(f"{pz['id']:<5} {per[k]}", flush=True)
    if args.pool:
        Path(args.out).write_text(json.dumps(puzzles, indent=1))
    if args.write:
        for f in files:
            f.write_text(json.dumps(data[f], indent=1))
    return 0


# --------------------------------------------------------------------------- commands

def _evals(args):
    out = [("stock", None, 1.0)]
    if args.net and MODEL.is_file():
        out.append(("net", str(MODEL), MODEL_SCALE))
    return out


def _rollout_job(args):
    sys.path.insert(0, str(ROOT / "src"))
    pz, n, foe = args
    FORCED_PREF[:] = pz.get("pref", [])
    return rollout_table(pz["state"], n, foe=foe)


def best_values(pz: dict, table: dict) -> tuple[float, float]:
    """(best value over intended actions, best value over the others), each the max over our
    continuation policies."""
    accept, reject = set(pz["accept"]), set(pz.get("reject", []))
    good = [max(v.values()) for a, v in table.items() if (a in accept if accept else a not in reject)]
    bad = [max(v.values()) for a, v in table.items() if not (a in accept if accept else a not in reject)]
    return (max(good) if good else 0.0), (max(bad) if bad else 0.0)


def verdict(pz: dict, table: dict, margin: float = 0.03) -> tuple[bool, str]:
    """Intended set must be (within margin) the best first action under smart and stall."""
    accept, reject = set(pz["accept"]), set(pz.get("reject", []))
    ok, why = True, []
    for cont in ("smart", "stall"):
        good = [v[cont] for a, v in table.items() if (a in accept if accept else a not in reject)]
        bad = [v[cont] for a, v in table.items() if not (a in accept if accept else a not in reject)]
        if not good or not bad:
            continue
        gap = max(good) - max(bad)
        why.append(f"{cont}:{gap:+.2f}")
        if gap < -margin:
            ok = False
    return ok, " ".join(why)


def tier(pz: dict) -> str:
    """'robust': intended action beats the rest by >= 0.10 against both foe models;
    'verified': by >= 0.10 against the max-damage foe and not reversed against the random-attack foe;
    'unverified': anything else (kept in the files, excluded from the headline numbers)."""
    v = pz.get("verify")
    if not v:
        return "unverified"
    accept, reject = set(pz["accept"]), set(pz.get("reject", []))

    def gap(vals):
        good = [x for a, x in vals.items() if (a in accept if accept else a not in reject)]
        bad = [x for a, x in vals.items() if not (a in accept if accept else a not in reject)]
        return (max(good) if good else 0.0) - (max(bad) if bad else 0.0)

    g, r = gap(v["foe_greedy"]), gap(v["foe_random"])
    if g >= 0.10 and r >= 0.10:
        return "robust"
    if g >= 0.10 and r >= -0.02:
        return "verified"
    return "unverified"


def cmd_rollouts(args) -> int:
    """Roll out every puzzle against two foe models; --write stores the tables in the JSON."""
    files = [f for f in sorted(PUZZLE_DIR.glob("*.json")) if f.name != "monlib.json" and not f.name.startswith("results")]
    data = {f: json.loads(f.read_text()) for f in files}
    puzzles = [p for f in files for p in data[f] if not args.only or p["id"] in args.only.split(",")]
    jobs = [(p, args.n, foe) for p in puzzles for foe in ("greedy", "random")]
    with ProcessPoolExecutor(args.workers) as pool:
        tables = list(pool.map(_rollout_job, jobs))
    for k, pz in enumerate(puzzles):
        tg, tr = tables[2 * k], tables[2 * k + 1]
        row, robust = [], True
        for name, tb in (("greedy", tg), ("random", tr)):
            good, bad = best_values(pz, tb)
            row.append(f"{name}: intended={good:.2f} other={bad:.2f}")
            robust &= (good - bad >= 0.10)
        pz["verify"] = {"foe_greedy": {a: round(max(v.values()), 2) for a, v in tg.items()},
                        "foe_random": {a: round(max(v.values()), 2) for a, v in tr.items()},
                        "robust": robust, "n": args.n}
        print(f"{pz['id']:<5}{'ROBUST' if robust else 'weak  '} " + " | ".join(row), flush=True)
    if args.write:
        for f in files:
            f.write_text(json.dumps(data[f], indent=1))
    return 0


def cmd_verify(args) -> int:
    puzzles = load_puzzles()
    if args.only:
        puzzles = [p for p in puzzles if p["id"] in args.only.split(",")]
    with ProcessPoolExecutor(args.workers) as pool:
        for label, ev, scale in _evals(args):
            res = search_many(pool, [p["state"] for p in puzzles], int(args.seconds * 1000),
                              args.worlds, ev, scale)
            for p, wr in zip(puzzles, res):
                rate, dist = pass_rate(wr, set(p["accept"]), set(p.get("reject", [])))
                vals = mean_values(wr)
                top = ", ".join(f"{c}:{v:.2f}" for c, v in list(dist.items())[:4])
                vtxt = ", ".join(f"{c}={v:.2f}" for c, v in sorted(vals.items(), key=lambda kv: -kv[1])[:4])
                print(f"{p['id']:<10}{label:<6} pass={rate:.2f} dist[{top}] val[{vtxt}]", flush=True)
    return 0


def cmd_run(args) -> int:
    puzzles = load_puzzles()
    if args.only:
        puzzles = [p for p in puzzles if p["id"] in args.only.split(",")]
    if args.tier != "all":
        keep = {"robust": {"robust"}, "verified": {"robust", "verified"}}[args.tier]
        puzzles = [p for p in puzzles if tier(p) in keep]
    print(f"{len(puzzles)} puzzles (tier {args.tier})", flush=True)
    times = [int(t) for t in args.times.split(",")]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for label, ev, scale in _evals(args):
            for ms in times:
                t0 = time.time()
                res = search_many(pool, [p["state"] for p in puzzles], ms, args.worlds, ev, scale, args.knee, args.boost_scale)
                for p, wr in zip(puzzles, res):
                    rate, dist = pass_rate(wr, set(p["accept"]), set(p.get("reject", [])))
                    rows.append({"id": p["id"], "theme": p["theme"], "kind": p["kind"],
                                 "eval": label, "ms": ms, "pass": rate, "dist": dist})
                print(f"# {label} {ms}ms done in {time.time() - t0:.0f}s", flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1))
    # table: theme/kind x time x eval
    groups = sorted({(r["theme"], r["kind"]) for r in rows})
    print("\ntheme/kind          " + "".join(f"{l}@{ms:<6}" for l, _, _ in _evals(args) for ms in times))
    for th, kd in groups:
        cells = []
        for l, _, _ in _evals(args):
            for ms in times:
                xs = [r["pass"] for r in rows if (r["theme"], r["kind"], r["eval"], r["ms"]) == (th, kd, l, ms)]
                cells.append(f"{sum(xs) / len(xs):>10.2f}" if xs else " " * 10)
        print(f"{th + '/' + kd:<20}" + "".join(cells))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("verify", cmd_verify), ("run", cmd_run), ("rollouts", cmd_rollouts),
                     ("searchroll", cmd_searchroll)):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--worlds", type=int, default=8)
        s.add_argument("--only", default="")
        s.add_argument("--no-net", dest="net", action="store_false")
        if name == "rollouts":
            s.add_argument("--n", type=int, default=100)
            s.add_argument("--write", action="store_true")
        elif name == "searchroll":
            s.add_argument("--n", type=int, default=40)
            s.add_argument("--ms", type=int, default=40)
            s.add_argument("--theme", default="")
            s.add_argument("--pool", default="", help="JSON list of candidate puzzles instead of tests/puzzles")
            s.add_argument("--out", default="searchroll_pool_out.json")
            s.add_argument("--write", action="store_true")
        elif name == "verify":
            s.add_argument("--seconds", type=float, default=8.0)
        else:
            s.add_argument("--times", default="100,300,1000,3000")
            s.add_argument("--knee", type=float, default=0.0, help="589.32 low-HP eval knee (needs the 589.32 wheel)")
            s.add_argument("--boost-scale", type=float, default=1.0, help="589.32 stat-boost eval multiplier")
            s.add_argument("--tier", choices=("all", "verified", "robust"), default="all")
            s.add_argument("--out", default="puzzle_results.json")
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
