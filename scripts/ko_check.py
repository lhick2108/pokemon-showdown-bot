"""How often does the Rust engine's KO model disagree with Showdown's exact damage rolls? (589.58)

    python scripts/ko_check.py validate                 # harness self-checks
    python scripts/ko_check.py run --n 3000 --out ko.jsonl --seed 1
    python scripts/ko_check.py analyze ko.jsonl

Setup (once):  cd scratch/tools && npm install @smogon/calc   (node_modules is gitignored)

Sampling: recorded decisions (ladder + selfplay), world 0 of each decision, both
actives, every damaging move of each attacker.  Two strata are drawn:
``uniform`` (every situation equally likely) and ``near`` (engine max damage is
0.6-1.6x the defender's current HP -- where KO calls actually matter).

For each situation we compute
  engine_k   KO rolls out of 16 the search would use.  This replicates
             genx/generate_instructions.rs (max*0.85 + i*max*0.01 in f32, strict
             ``> hp``; all-16 when floor(max*0.85) >= hp; 0 when max < hp) on the
             engine's own calculate_damage max.  ``--engine-gi`` cross-checks the
             rule against the real generate_instructions KO probability.
  exact_k    number of @smogon/calc's 16 exact rolls that are >= current HP.
  engine_k_on_calc_max
             the engine's roll *model* applied to calc's max roll: isolates the
             0.85..1.00 linear-roll error from errors in the engine's max damage.
Crits are excluded (both sides branch the same way on crit rate); multi-hit and
fixed-damage moves are excluded from the headline numbers.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA_ROOTS = [
    Path(r"C:\Users\Lewis\pokemon-showdown-bot\data\ladder\gen7\records"),
    Path(r"C:\Users\Lewis\pokemon-showdown-bot\data\selfplay\gen8"),
]
NODE_HELPER = REPO / "scripts" / "ko_calc.mjs"
F32 = np.float32


# ----------------------------------------------------------------- engine rule
def engine_kill_rolls(max_dmg: int, hp: int) -> int:
    """KO rolls (of 16) the search uses for a non-crit hit -- mirrors the Rust."""
    if max_dmg <= 0 or hp <= 0:
        return 0
    if max_dmg < hp:
        return 0
    min_dmg = int(F32(max_dmg) * F32(0.85))
    if min_dmg >= hp:
        return 16
    inc = F32(max_dmg) * F32(0.01)
    dmg = F32(max_dmg) * F32(0.85)
    n = 0
    for _ in range(16):
        if dmg > F32(hp):
            n += 1
        dmg = F32(dmg + inc)
    return n


def exact_kill_rolls(rolls: list[int], hp: int) -> int:
    return sum(1 for r in rolls if r >= hp)


# ------------------------------------------------------------ state -> inputs
def _boosts(side, which: str) -> dict:
    return {
        "atk": side.attack_boost, "def": side.defense_boost, "spa": side.special_attack_boost,
        "spd": side.special_defense_boost, "spe": side.speed_boost,
    }


def mon_dict(side, mon) -> dict:
    types = [t for t in mon.types if t != "TYPELESS"]
    return {
        "species": mon.id, "level": mon.level, "ability": mon.ability, "item": mon.item,
        "status": mon.status, "boosts": _boosts(side, ""), "hp": mon.hp,
        "stats": {"hp": mon.maxhp, "atk": mon.attack, "def": mon.defense,
                  "spa": mon.special_attack, "spd": mon.special_defense, "spe": mon.speed},
        "types": types, "tera": mon.tera_type if mon.terastallized else None,
        "weightkg": mon.weight_kg,
    }


def side_conds(side) -> dict:
    sc = side.side_conditions
    return {k: int(getattr(sc, k) > 0) for k in ("reflect", "light_screen", "aurora_veil", "tailwind")}


def situations_from_state(state, meta: dict):
    """Yield one dict per (attacker side, damaging move) in this state."""
    sides = (state.side_one, state.side_two)
    for a in (0, 1):
        sa, sd = sides[a], sides[1 - a]
        atk, dfn = sa.pokemon[int(sa.active_index)], sd.pokemon[int(sd.active_index)]
        if atk.hp <= 0 or dfn.hp <= 0 or "SUBSTITUTE" in sd.volatile_statuses:
            continue
        field = {"weather": state.weather, "terrain": state.terrain,
                 "atkSide": side_conds(sa), "defSide": side_conds(sd)}
        for mv in atk.moves:
            mid = mv.id.lower()
            if mid in ("none", "switch"):
                continue
            yield {
                "meta": meta, "state": state, "a": a, "move": mid,
                "atk": mon_dict(sa, atk), "def": mon_dict(sd, dfn), "field": field,
            }


def engine_max(state, a: int, move: str) -> tuple[int, int] | None:
    try:
        if a == 0:
            r = _pe().calculate_damage(state, move, "none", True)[0]
        else:
            r = _pe().calculate_damage(state, "none", move, False)[1]
    except ValueError:
        return None
    return (r[0], r[1]) if len(r) >= 2 else None


_PE = None


def _pe():
    global _PE
    if _PE is None:
        import poke_engine

        _PE = poke_engine
    return _PE


# ------------------------------------------------------------------- sampling
def iter_decision_states(files, rng: random.Random, per_game: int):
    from showdown_bot import recorder

    for f in files:
        try:
            g = recorder.read_game(f)
        except Exception:
            continue
        decs = [d for d in g["decisions"] if d.get("worlds")]
        rng.shuffle(decs)
        for d in decs[:per_game]:
            try:
                st = _pe().State.from_string(d["worlds"][0]["state"])
            except Exception:
                continue
            yield st, {"game": Path(f).name, "turn": d.get("turn")}


def collect(n_uniform: int, n_near: int, seed: int, per_game: int = 3, max_games: int = 700):
    rng = random.Random(seed)
    files = []
    for root in DATA_ROOTS:
        fs = sorted(root.glob("*.jsonl.gz"))
        rng.shuffle(fs)
        files += fs[: max_games // 2 if root == DATA_ROOTS[1] else len(fs)]
    rng.shuffle(files)
    pool_u, pool_n = [], []
    for st, meta in iter_decision_states(files, rng, per_game):
        for sit in situations_from_state(st, meta):
            em = engine_max(st, sit["a"], sit["move"])
            if not em or em[0] <= 0:
                continue
            sit["engine_max"], sit["engine_crit"] = em
            hp = sit["def"]["hp"]
            pool_u.append(sit)
            if 0.6 * hp <= em[0] <= 1.6 * hp:
                pool_n.append(sit)
        if len(pool_u) > 6 * n_uniform and len(pool_n) > 3 * n_near:
            break
    rng.shuffle(pool_u)
    rng.shuffle(pool_n)
    out = [dict(s, stratum="uniform") for s in pool_u[:n_uniform]]
    out += [dict(s, stratum="near") for s in pool_n[:n_near]]
    return out


# ------------------------------------------------------------------ evaluation
def run_calc(sits):
    payload = [{"atk": s["atk"], "def": s["def"], "move": s["move"], "field": s["field"]} for s in sits]
    p = subprocess.run(["node", str(NODE_HELPER)], input=json.dumps(payload), capture_output=True,
                       text=True, cwd=REPO)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)


def gi_ko_prob(sit):
    """Real generate_instructions KO probability on a *cleaned* copy of the state (no items/abilities/
    statuses/weather, so only the damage + crit branching is in play).  Returns (P(faint | move hit),
    engine KO rolls recomputed on the cleaned state) or None."""
    pe = _pe()
    if sit["field"]["weather"] != "NONE" or sit["field"]["terrain"] != "NONE":
        return None
    a = sit["a"]
    sides = sit["state"].to_string().split("/")
    for i in (0, 1):
        parts = sides[i].split("=")
        for j in range(6):
            f = parts[j].split(",")
            f[8], f[9], f[10], f[18] = "NONE", "NONE", "NONE", "NONE"
            parts[j] = ",".join(f)
        sides[i] = "=".join(parts)
    st = pe.State.from_string("/".join(sides))
    mv = sit["move"]
    em = engine_max(st, a, mv)
    if not em:
        return None
    try:
        ins = pe.generate_instructions(st, mv, "none") if a == 0 else pe.generate_instructions(st, "none", mv)
    except ValueError:
        return None
    dside = (lambda s: s.side_two) if a == 0 else (lambda s: s.side_one)
    d0 = dside(st)
    hp0 = d0.pokemon[int(d0.active_index)].hp
    ko = hit = 0.0
    for si in ins:
        d = dside(st.apply_instructions(si))
        h = d.pokemon[int(d.active_index)].hp
        if h < hp0:
            hit += si.percentage
            if h <= 0:
                ko += si.percentage
    return (ko / hit, engine_kill_rolls(em[0], hp0)) if hit else None


def evaluate(sits, check_gi: bool):
    calc = run_calc(sits)
    recs = []
    for s, c in zip(sits, calc):
        hp = s["def"]["hp"]
        r = {
            "stratum": s["stratum"], "move": s["move"], "atk_species": s["atk"]["species"],
            "def_species": s["def"]["species"], "atk_ability": s["atk"]["ability"],
            "def_ability": s["def"]["ability"], "atk_item": s["atk"]["item"], "def_item": s["def"]["item"],
            "weather": s["field"]["weather"], "terrain": s["field"]["terrain"],
            "tera": bool(s["atk"]["tera"] or s["def"]["tera"]), "status": s["atk"]["status"],
            "screens": any(s["field"]["defSide"][k] for k in ("reflect", "light_screen", "aurora_veil")),
            "hp": hp, "engine_max": s["engine_max"], "meta": s["meta"],
            "multihit": bool(c.get("multihit")), "excluded": None,
        }
        if not c.get("rolls"):
            r["excluded"] = "fixed" if "fixed" in c else (c.get("error") or "norolls")
        else:
            rolls = c["rolls"]
            r["calc_max"] = max(rolls)
            r["exact_k"] = exact_kill_rolls(rolls, hp)
            r["engine_k"] = engine_kill_rolls(s["engine_max"], hp)
            r["engine_k_on_calc_max"] = engine_kill_rolls(max(rolls), hp)
            if r["multihit"]:
                r["excluded"] = "multihit"
            if check_gi and r["excluded"] is None and r["move"] and hp > 0:
                pass
        recs.append(r)
    return recs


# --------------------------------------------------------------------- report
STATEFUL = {"PROTOSYNTHESIS", "QUARKDRIVE", "ELECTROMORPHOSIS", "SLOWSTART", "SUPREMEOVERLORD", "TOXICBOOST",
            "FLASHFIRE", "TERASHELL", "ZEROTOHERO", "DEFEATIST", "STAKEOUT", "ANGERSHELL"}


def classify(k: int) -> str:
    return "never" if k == 0 else ("guaranteed" if k == 16 else "partial")


def summarize(recs, label: str, key_engine="engine_k") -> dict:
    ok = [r for r in recs if r["excluded"] is None]
    n = len(ok)
    if not n:
        return {"label": label, "n": 0}
    diff = [r for r in ok if classify(r[key_engine]) != classify(r["exact_k"])]
    rel = [r for r in ok if 0 < r[key_engine] < 16 or 0 < r["exact_k"] < 16 or classify(r[key_engine]) != classify(r["exact_k"])]
    rel = [r for r in rel if r[key_engine] > 0 or r["exact_k"] > 0]
    mae = statistics.mean(abs(r[key_engine] - r["exact_k"]) / 16 for r in rel) if rel else 0.0
    bias = statistics.mean((r[key_engine] - r["exact_k"]) / 16 for r in rel) if rel else 0.0
    over = sum(1 for r in rel if r[key_engine] > r["exact_k"])
    under = sum(1 for r in rel if r[key_engine] < r["exact_k"])
    trans = collections.Counter(classify(r["exact_k"]) + "->" + classify(r[key_engine]) for r in diff)
    return {"label": label, "n": n, "class_diff": len(diff), "class_diff_pct": 100 * len(diff) / n,
            "n_ko_relevant": len(rel), "mae_ko_prob": mae, "mean_signed_diff": bias,
            "engine_over": over, "engine_under": under, "transitions(exact->engine)": dict(trans)}


def offenders(recs, field: str, key_engine="engine_k", top=10, min_n=5):
    ok = [r for r in recs if r["excluded"] is None and (r[key_engine] > 0 or r["exact_k"] > 0)]
    by = collections.defaultdict(list)
    for r in ok:
        by[r[field]].append(r)
    rows = []
    for k, rs in by.items():
        if len(rs) < min_n:
            continue
        err = sum(abs(r[key_engine] - r["exact_k"]) / 16 for r in rs)
        sg = statistics.mean((r[key_engine] - r["exact_k"]) / 16 for r in rs)
        cd = sum(1 for r in rs if classify(r[key_engine]) != classify(r["exact_k"]))
        rows.append((err, k, len(rs), err / len(rs), sg, cd))
    rows.sort(reverse=True)
    return rows[:top]


def analyze(path: str):
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    for r in recs:
        if r["excluded"] is None and (r["atk_ability"] in STATEFUL or r["def_ability"] in STATEFUL):
            r["excluded"] = "stateful-ability (calc cannot see in-battle ability state)"
    ex = collections.Counter(r["excluded"] for r in recs if r["excluded"])
    print(f"{len(recs)} situations; excluded: {dict(ex)}")
    for strat in ("uniform", "near", None):
        sub = [r for r in recs if strat is None or r["stratum"] == strat]
        for lab, key in (("engine (as shipped)", "engine_k"), ("roll model only (on calc max)", "engine_k_on_calc_max")):
            s = summarize(sub, f"{strat or 'all'} / {lab}", key)
            print(json.dumps(s, indent=None))
    ok = [r for r in recs if r["excluded"] is None and r["calc_max"] > 0]
    ratios = [r["engine_max"] / r["calc_max"] for r in ok]
    if ratios:
        q = np.quantile(ratios, [0.01, 0.1, 0.5, 0.9, 0.99])
        print("engine_max/calc_max quantiles 1/10/50/90/99%:", np.round(q, 3).tolist(),
              " exact-equal:", f"{100 * sum(1 for r in ok if r['engine_max'] == r['calc_max']) / len(ok):.1f}%",
              " |dev|>5%:", f"{100 * sum(1 for x in ratios if abs(x - 1) > 0.05) / len(ratios):.1f}%")
    gi = [r for r in recs if r.get("gi_ko") is not None and 0 < r["gi_k"] < 16]
    if gi:
        def match(r):  # normal crit 1/24, high-crit-ratio moves 1/8, guaranteed crit 1
            return any(abs(r["gi_ko"] - ((1 - c) * r["gi_k"] / 16 + c)) < 0.005 for c in (1 / 24, 1 / 8, 0.5, 1.0))
        bad = [r for r in gi if not match(r)]
        print(f"generate_instructions cross-check (cleaned state, partial-KO cases): {len(gi)} checked, "
              f"{len(bad)} do not match the replicated rule + a standard crit rate")
        for r in bad[:5]:
            print("   ", r["move"], r["gi_k"], round(r["gi_ko"], 3))
    for field in ("move", "atk_ability", "def_ability", "atk_item", "def_item", "weather", "def_species"):
        print(f"\n-- worst by {field} (KO-relevant cases; abs err sum, n, mean abs, mean signed(+=engine over), class diffs)")
        for row in offenders([r for r in recs if r["stratum"] == "near"] or recs, field):
            print("  %-18s n=%-4d sum=%.2f mean=%.3f signed=%+.3f cd=%d" % (row[1], row[2], row[0], row[3], row[4], row[5]))


# ------------------------------------------------------------------- validate
def ref_rolls(level, bp, atk, dfn, stab, eff):
    base = (((2 * level) // 5 + 2) * bp * atk // dfn) // 50 + 2
    out = []
    for r in range(85, 101):
        d = base * r // 100
        if stab:
            x = d * 1.5
            d = math.ceil(x) if x - math.floor(x) > 0.5 else math.floor(x)
        d = math.floor(d * eff)
        out.append(max(d, 1))
    return out


def validate(n_check: int = 2500):
    print("engine rule vs hand-computed examples")
    assert engine_kill_rolls(100, 100) == 0          # max == hp: strict '>' never counts the top roll
    assert engine_kill_rolls(100, 101) == 0
    assert engine_kill_rolls(100, 85) == 16         # min roll 85 >= hp: guaranteed
    assert engine_kill_rolls(100, 86) == 14         # rolls 86..100 kill (15) but strict > drops the equal roll
    assert engine_kill_rolls(100, 80) == 16          # floor(85) >= 80
    assert engine_kill_rolls(200, 180) == 10        # 170..200 step 2, strictly > 180
    print("  ok (note: max==hp -> engine says 0 KO rolls though top roll KOs)")
    print("sanity: calc vs plain Gen 9 formula on neutral situations")
    sits = collect(n_check * 4, 0, seed=7, per_game=4, max_games=400)
    calc = run_calc(sits)
    neutral_items = {"NONE", "LEFTOVERS", "HEAVYDUTYBOOTS", "BOOSTERENERGY", "SITRUSBERRY", "ROCKYHELMET", "COVERTCLOAK", "LUMBERRY", "CHESTOBERRY", "SHELLBELL"}
    neutral_abil = {"PRESSURE", "INTIMIDATE", "NATURALCURE", "SERENEGRACE", "INNERFOCUS", "PRANKSTER", "STURDY", "INFILTRATOR", "STORMDRAIN", "DANCER", "EARLYBIRD", "STICKYHOLD", "COMPETITIVE", "DEFIANT", "TANGLINGHAIR", "ROUGHSKIN", "REGENERATOR", "SHEDSKIN", "OBLIVIOUS", "OWNTEMPO", "CLEARBODY", "MAGICGUARD", "UNAWARE_X", "SYNCHRONIZE", "SOUNDPROOF_X", "KEENEYE", "COMPETITIVE", "TELEPATHY", "TRACE_X", "STAMINA", "BERSERK_X", "VOLTABSORB_X", "ARENATRAP", "SHADOWTAG", "MAGNETPULL", "CURSEDBODY", "CUTECHARM", "RATTLED", "SANDVEIL_X", "VITALSPIRIT", "WHITESMOKE", "LIQUIDOOZE", "ILLUMINATE", "RUNAWAY", "PICKUP", "RUNAWAY", "FRISK", "UNBURDEN_X", "LIGHTNINGROD_X"}
    tested = mism = emis = 0
    for s, c in zip(sits, calc):
        if not c.get("rolls") or c["multihit"] or c["bp"] <= 0 or s["atk"]["tera"] or s["def"]["tera"]:
            continue
        a, d, f = s["atk"], s["def"], s["field"]
        if (a["item"] not in neutral_items or d["item"] not in neutral_items or a["ability"] not in neutral_abil
                or d["ability"] not in neutral_abil or a["status"] != "NONE" or f["weather"] != "NONE"
                or f["terrain"] != "NONE" or any(f["defSide"].values())
                or any(v for v in a["boosts"].values()) or any(v for v in d["boosts"].values())
                or c["bp"] in (0,) or s["move"] in ("bodypress", "foulplay", "psyshock", "psystrike", "secretsword", "facade",
                                                    "hex", "venoshock", "acrobatics", "knockoff", "gyroball", "electroball",
                                                    "lowkick", "heavyslam", "heatcrash", "grassknot", "eruption", "waterspout",
                                                    "judgment", "terablast", "weatherball", "boltbeak", "fishiousrend", "payback",
                                                    "avalanche", "revenge", "brine", "storedpower", "powertrip", "rage", "hardpress")):
            continue
        physical = c["cat"] == "Physical"
        A = a["stats"]["atk" if physical else "spa"]
        D = d["stats"]["def" if physical else "spd"]
        stab = c["type"].upper() in a["types"]
        ref = ref_rolls(a["level"], c["bp"], A, D, stab, c["eff"])
        tested += 1
        if ref != c["rolls"]:
            mism += 1
            if mism <= 3:
                print("  MISMATCH", s["move"], a, d, c["bp"], c["eff"], "ref", ref[-1], "calc", c["rolls"][-1], "eng", s.get("engine_max"))
        em = s.get("engine_max")
        if em and abs(em - ref[-1]) > 1:
            emis += 1
    print(f"  {tested} neutral cases: calc != plain formula in {mism}; engine max differs by >1 in {emis}")
    print("hand-check printing 3 cases")
    for s, c in list(zip(sits, calc))[:3]:
        if c.get("rolls"):
            print(" ", s["move"], s["atk"]["species"], "->", s["def"]["species"], "hp", s["def"]["hp"],
                  "calc rolls", c["rolls"][0], "..", c["rolls"][-1], "engine max", s["engine_max"],
                  "engine_k", engine_kill_rolls(s["engine_max"], s["def"]["hp"]),
                  "exact_k", exact_kill_rolls(c["rolls"], s["def"]["hp"]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=3000, help="uniform-stratum size")
    r.add_argument("--n-near", type=int, default=3000)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", required=True)
    r.add_argument("--engine-gi", type=int, default=300, help="cross-check this many cases with generate_instructions")
    an = sub.add_parser("analyze")
    an.add_argument("path")
    args = ap.parse_args()
    if args.cmd == "validate":
        validate()
    elif args.cmd == "run":
        sits = collect(args.n, args.n_near, args.seed)
        recs = evaluate(sits, check_gi=False)
        if args.engine_gi:
            chk = [s for s, r in zip(sits, recs) if r["excluded"] is None and 0 < r["engine_k"] < 16][: args.engine_gi]
            r_by_id = {id(s): r for s, r in zip(sits, recs)}
            for s in chk:
                g = gi_ko_prob(s)
                if g:
                    r_by_id[id(s)]["gi_ko"], r_by_id[id(s)]["gi_k"] = g
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(recs)} -> {args.out}")
        analyze(args.out)
    else:
        analyze(args.path)


if __name__ == "__main__":
    main()
