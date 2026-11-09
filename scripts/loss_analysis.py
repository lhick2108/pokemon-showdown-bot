"""Loss analysis for a ladder run (data/ladder/<run>): semi-automatic error classification.

Joins three sources per game:
  * games.jsonl           result / opponent rating
  * logs/<tag>.log        Showdown protocol log (we strip the |request| lines)
  * records/<tag>__*.jsonl.gz  per-decision search records (16 worlds, root value per world,
                          opp_facts, chosen action)

Per game it builds a turn timeline (our mean root value, events in the turn), finds the
decisive turn (largest drop in our value), measures world-model surprises (a fact revealed in
the log - move / item / ability / tera type - that was present in <= 2 of the 16 sampled
worlds at the last decision before the reveal), luck events (crits, misses, flinch, full para),
tera usage, switch-in damage, and writes a classified table.

    python scripts/loss_analysis.py --run C:/.../data/ladder/w16 --summary
    python scripts/loss_analysis.py --run C:/.../data/ladder/w16 --game battle-gen9randombattle-2699990136
    python scripts/loss_analysis.py --run ... --dump out.json     # per-game features as json

Pure python, single thread, no engine.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys

SURPRISE_MAX_FRAC = 2 / 16  # fact present in <= this share of worlds counts as a surprise
DROP_DECISIVE = 0.12  # min single-turn value drop to call a turn decisive


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------------------------------------
# log parsing
# ---------------------------------------------------------------------------------------------

def read_log(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.rstrip("\n") for ln in f if not ln.startswith("|request|")]


def mon_name(ident: str) -> tuple[str, str]:
    # "p2a: Name" -> ("p2", "name")
    side = ident[:2]
    name = ident.split(": ", 1)[1] if ": " in ident else ident
    return side, norm(name)


def hp_frac(cond: str) -> float | None:
    cond = cond.split(" ")[0]
    if "fnt" in cond or cond.startswith("0"):
        return 0.0 if "fnt" in cond else None
    if "/" in cond:
        a, b = cond.split("/")
        try:
            return float(a) / float(b)
        except ValueError:
            return None
    return None


class Turn:
    def __init__(self, n):
        self.n = n
        self.lines: list[str] = []
        self.ev = collections.defaultdict(list)  # kind -> list of tuples


def parse_log(lines: list[str], me: str):
    """Return (turns dict n->Turn, facts, meta). me is 'p1' or 'p2'."""
    opp = "p2" if me == "p1" else "p1"
    turns: dict[int, Turn] = {0: Turn(0)}
    cur = turns[0]
    # opp reveals: first turn each fact was seen
    reveal: dict[tuple, int] = {}  # (species, kind, value) -> turn
    species_of: dict[str, str] = {}  # side+nick -> species
    hp: dict[tuple[str, str], float] = {}
    active: dict[str, str] = {}
    tera_used = {me: None, opp: None}
    winner = None
    last_mv = None
    for ln in lines:
        p = ln.split("|")
        if len(p) < 2:
            continue
        k = p[1]
        if k == "turn":
            n = int(p[2])
            cur = turns.setdefault(n, Turn(n))
            continue
        cur.lines.append(ln)
        if k in ("switch", "drag"):
            side, nick = mon_name(p[2])
            sp = norm(p[3].split(",")[0])
            species_of[side + nick] = sp
            prev = active.get(side[:2])
            active[side[:2]] = side + nick
            f = hp_frac(p[4]) if len(p) > 4 else None
            cur.ev["switch"].append((side[:2], sp, f, k))
            if f is not None:
                hp[(side[:2], sp)] = f
        elif k == "move":
            side, nick = mon_name(p[2])
            sp = species_of.get(side + nick, nick)
            mv = norm(p[3])
            last_mv = (side[:2], sp, mv)
            cur.ev["move"].append((side[:2], sp, mv, " ".join(p[4:])))
            if side[:2] == opp:
                reveal.setdefault((sp, "move", mv), cur.n)
        elif k == "-damage":
            side, nick = mon_name(p[2])
            sp = species_of.get(side + nick, nick)
            f = hp_frac(p[3])
            old = hp.get((side[:2], sp), 1.0)
            if f is not None:
                cur.ev["damage"].append((side[:2], sp, old, f, ln))
                hp[(side[:2], sp)] = f
        elif k == "-heal":
            side, nick = mon_name(p[2])
            sp = species_of.get(side + nick, nick)
            f = hp_frac(p[3])
            if f is not None:
                hp[(side[:2], sp)] = f
        elif k == "faint":
            side, nick = mon_name(p[2])
            sp = species_of.get(side + nick, nick)
            cur.ev["faint"].append((side[:2], sp))
            hp[(side[:2], sp)] = 0.0
        elif k == "-crit":
            side, nick = mon_name(p[2])
            cur.ev["crit"].append((side[:2], species_of.get(side + nick, nick)))
        elif k == "-miss":
            side, nick = mon_name(p[2])  # attacker
            cur.ev["miss"].append((side[:2], species_of.get(side + nick, nick)))
        elif k == "cant":
            side, nick = mon_name(p[2])
            cur.ev["cant"].append((side[:2], species_of.get(side + nick, nick), p[3] if len(p) > 3 else ""))
        elif k == "-terastallize":
            side, nick = mon_name(p[2])
            sp = species_of.get(side + nick, nick)
            cur.ev["tera"].append((side[:2], sp, norm(p[3])))
            tera_used[side[:2]] = cur.n
            if side[:2] == opp:
                reveal.setdefault((sp, "tera", norm(p[3])), cur.n)
        elif k == "-status":
            side, nick = mon_name(p[2])
            cur.ev["status"].append((side[:2], species_of.get(side + nick, nick), p[3]))
        elif k in ("-boost", "-unboost"):
            side, nick = mon_name(p[2])
            amt = int(p[4]) * (1 if k == "-boost" else -1)
            cur.ev["boost"].append((side[:2], species_of.get(side + nick, nick), p[3], amt))
        elif k == "-sidestart":
            cur.ev["sidestart"].append((p[2][:2], norm(p[3].replace("move:", ""))))
        elif k in ("-item", "-enditem"):
            side, nick = mon_name(p[2])
            if k == "-enditem" and "Knock Off" in ln and len(p) > 3 and p[3].startswith("Choice"):
                cur.ev["choice_ko"].append((side[:2], norm(p[3]), last_mv[2] if last_mv else ""))
            if side[:2] == opp:
                sp = species_of.get(side + nick, nick)
                reveal.setdefault((sp, "item", norm(p[3])), cur.n)
        elif k == "-ability":
            side, nick = mon_name(p[2])
            if side[:2] == opp:
                sp = species_of.get(side + nick, nick)
                reveal.setdefault((sp, "ability", norm(p[3])), cur.n)
        elif k in ("-immune", "-fail", "-notarget", "-block") and last_mv and last_mv[0] == me:
            cur.ev["waste"].append((last_mv[1], last_mv[2], k))
        elif k == "-activate" and last_mv and last_mv[0] == me and len(p) > 3 and "Protect" in p[3] + (p[4] if len(p) > 4 else ""):
            cur.ev["waste"].append((last_mv[1], last_mv[2], "protected"))
        elif k == "-supereffective":
            side, nick = mon_name(p[2])
            cur.ev["se"].append((side[:2],))
        elif k == "win":
            winner = p[2]
        # generic "[from] item/ability: X" attribution (owner = [of] mon, else the line's mon)
        for mm in re.finditer(r"\[from\] (item|ability): ([^|\[]+)", ln):
            of = re.search(r"\[of\] (p[12][ab]: [^|]+)", ln)
            ident = of.group(1) if of else (p[2] if len(p) > 2 and re.match(r"p[12][ab]: ", p[2]) else None)
            if not ident:
                continue
            side_, nick_ = mon_name(ident)
            if side_[:2] == opp:
                sp_ = species_of.get(side_ + nick_, nick_)
                reveal.setdefault((sp_, mm.group(1), norm(mm.group(2))), cur.n)
    return turns, reveal, tera_used, winner


# ---------------------------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------------------------

def parse_state_opp(state: str):
    """Opp-side mons of an engine state string: list of dict(species,item,ability,tera,moves)."""
    sides = state.split("/")
    out = []
    # side2 is the opponent; its mons separated by '='
    s2 = sides[1] if len(sides) > 1 else ""
    for chunk in s2.split("="):
        f = chunk.split(",")
        if len(f) < 20 or not f[1].isdigit():
            continue
        moves = [norm(x.split(";")[0]) for x in f if ";" in x and x.split(";")[0].isalpha()]
        out.append({
            "species": norm(f[0]), "ability": norm(f[8]), "item": norm(f[10]),
            "tera": norm(f[-1]), "moves": moves,
        })
    return out


def load_records(path_glob: str):
    files = glob.glob(path_glob)
    if not files:
        return None
    dec = []
    for ln in gzip.open(files[0], "rt", encoding="utf-8"):
        d = json.loads(ln)
        if d.get("type") == "game":
            continue
        dec.append(d)
    return dec


def mean_value(d) -> float | None:
    vals = [w["avg_score"] for w in d.get("worlds", []) if w.get("avg_score") is not None]
    return statistics.fmean(vals) if vals else None


def world_presence(d, species: str, kind: str, value: str) -> float | None:
    """Share of the decision's worlds in which opp `species` has `kind`==value."""
    n = hit = 0
    for w in d.get("worlds", []):
        st = w.get("state")
        if not st:
            continue
        for m in parse_state_opp(st):
            if m["species"] == species:
                n += 1
                if kind == "move":
                    hit += value in m["moves"]
                else:
                    hit += m[kind] == value
                break
    if n == 0:
        return None
    return hit / n


# ---------------------------------------------------------------------------------------------
# per game analysis
# ---------------------------------------------------------------------------------------------

def analyse(run: str, g: dict):
    tag = g["battle_tag"]
    logp = os.path.join(run, "logs", tag + ".log")
    recs = load_records(os.path.join(run, "records", tag + "__*.jsonl.gz"))
    if not os.path.exists(logp) or not recs:
        return None
    lines = read_log(logp)
    me = "p1"
    for ln in lines[:40]:
        if ln.startswith("|player|") and "porygonzplaya" in ln:
            me = ln.split("|")[2]
    opp = "p2" if me == "p1" else "p1"
    turns, reveal, tera_used, winner = parse_log(lines, me)
    first_dec: dict[int, dict] = {}
    last_before: dict[int, dict] = {}
    for d in recs:
        if d["type"] != "decision":
            continue
        first_dec.setdefault(d["turn"], d)
    # values by turn (first decision of turn)
    vt = {t: mean_value(d) for t, d in first_dec.items()}
    ts = sorted(t for t in vt if vt[t] is not None)
    drops = []
    for a, b in zip(ts, ts[1:]):
        drops.append((vt[b] - vt[a], a))
    # final turn: value -> 0/1 outcome not included
    worst = min(drops) if drops else (0.0, None)
    # --- surprises: facts whose presence at the last decision before reveal was tiny
    surprises = []
    for (sp, kind, val), tr in reveal.items():
        # last decision at or before turn tr (decision at tr is made before the reveal happens)
        cand = [t for t in first_dec if t <= tr]
        if not cand:
            continue
        d = first_dec[max(cand)]
        # skip if the mon is not in the worlds yet (not revealed before)
        pres = world_presence(d, sp, kind, val)
        if pres is None:
            continue
        surprises.append({"species": sp, "kind": kind, "value": val, "turn": tr, "presence": round(pres, 3)})
    big = [s for s in surprises if s["presence"] <= SURPRISE_MAX_FRAC]
    tera_presence = [s["presence"] for s in surprises if s["kind"] == "tera"]
    pres_by_kind = collections.defaultdict(list)
    for s_ in surprises:
        pres_by_kind[s_["kind"]].append(s_["presence"])
    # --- per-turn luck / events
    ev_tot = collections.Counter()
    for n, t in turns.items():
        for s, sp in t.ev["crit"]:
            ev_tot["crit_on_us" if s == me else "crit_on_opp"] += 1
        for s, sp in t.ev["miss"]:
            ev_tot["our_miss" if s == me else "opp_miss"] += 1
        for s, sp, why in t.ev["cant"]:
            if s == me:
                ev_tot["we_cant_" + norm(why.split(":")[0])] += 1
            else:
                ev_tot["opp_cant_" + norm(why.split(":")[0])] += 1
    # --- switch-in punishments (our switch-ins that lost >=50% hp or fainted the same turn)
    bad_switch = 0
    sw_total = 0
    for n, t in turns.items():
        for s, sp, f, kind in t.ev["switch"]:
            if s != me or n == 0:
                continue
            sw_total += 1
            lost = 0.0
            fainted = False
            for ds, dsp, old, new, _ in t.ev["damage"]:
                if ds == me and dsp == sp:
                    lost += max(0.0, old - new)
            for fs, fsp in t.ev["faint"]:
                if fs == me and fsp == sp:
                    fainted = True
            if lost >= 0.5 or fainted:
                bad_switch += 1
    # loops: runs of the same (our mon, our move) with the opponent mon unchanged
    seq = []
    for n in sorted(turns):
        if n == 0:
            continue
        t = turns[n]
        mv = [e for e in t.ev["move"] if e[0] == me]
        sw = [e for e in t.ev["switch"] if e[0] == me]
        seq.append((n, (mv[0][1], mv[0][2]) if mv and not sw else ("switch", "")))
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1] and seq[i][1][0] != "switch":
            j += 1
        if j - i + 1 >= 5:
            runs.append((seq[i][0], j - i + 1, seq[i][1]))
        i = j + 1
    loop_turns = sum(r[1] for r in runs)
    # loops: runs of the same (our mon, our move)
    seq = []
    for n in sorted(turns):
        if n == 0:
            continue
        t = turns[n]
        mv = [e for e in t.ev["move"] if e[0] == me]
        sw = [e for e in t.ev["switch"] if e[0] == me]
        seq.append((n, (mv[0][1], mv[0][2]) if mv and not sw else ("switch", "")))
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][1] == seq[i][1] and seq[i][1][0] != "switch":
            j += 1
        if j - i + 1 >= 5:
            runs.append((seq[i][0], j - i + 1, seq[i][1]))
        i = j + 1
    loop_turns = sum(r[1] for r in runs)
    waste = [(n, e) for n, t in turns.items() for e in t.ev["waste"]]
    n_our_moves = sum(1 for t in turns.values() for e in t.ev["move"] if e[0] == me)
    status_us = sum(1 for t in turns.values() for e in t.ev["status"] if e[0] == me)
    status_opp = sum(1 for t in turns.values() for e in t.ev["status"] if e[0] == opp)
    choice_ko = sum(len(t.ev["choice_ko"]) for t in turns.values())
    # --- tera
    our_tera = [(n, e) for n, t in turns.items() for e in t.ev["tera"] if e[0] == me]
    opp_tera = [(n, e) for n, t in turns.items() for e in t.ev["tera"] if e[0] == opp]
    fainted_us = sum(1 for t in turns.values() for e in t.ev["faint"] if e[0] == me)
    fainted_opp = sum(1 for t in turns.values() for e in t.ev["faint"] if e[0] == opp)
    # setup: opp positive boosts
    opp_boost = sum(a for t in turns.values() for s, sp, st, a in t.ev["boost"] if s == opp and a > 0)
    # decisive turn explain
    dturn = worst[1]
    expl = []
    if dturn is not None and worst[0] <= -DROP_DECISIVE:
        for n in (dturn,):
            t = turns.get(n)
            if not t:
                continue
            if any(s == me for s, _ in t.ev["crit"]):
                expl.append("crit_on_us")
            if any(s == me for s, _ in t.ev["miss"]):
                expl.append("our_miss")
            if any(s == me for s, _, _ in t.ev["cant"]):
                expl.append("we_cant")
            if any(s == opp for s, _, _ in t.ev["status"]) is False and any(s == me for s, _, _ in t.ev["status"]):
                expl.append("status_on_us")
            if any(e[0] == opp for e in t.ev["tera"]):
                expl.append("opp_tera")
            if any(e[0] == me for e in t.ev["tera"]):
                expl.append("our_tera")
            if any(s == me and k == "switch" for s, _, _, k in t.ev["switch"]):
                expl.append("we_switched")
            if any(s == me for s, _ in t.ev["faint"]):
                expl.append("we_lost_mon")
            if any(s == opp and a > 0 for s, _, _, a in t.ev["boost"]):
                expl.append("opp_boost")
            if any(s == me for s, _ in t.ev["sidestart"]):
                expl.append("hazard_on_us")
        if any(s["turn"] in (dturn, dturn - 1) for s in big):
            expl.append("world_surprise")
    decs = [d for d in recs if d["type"] == "decision"]
    n_slow = sum(1 for d in decs if (d.get("compute") or {}).get("wall_ms", 0) > 1500)
    n_short = sum(1 for d in decs if (d.get("compute") or {}).get("worlds_searched", 16) < (d.get("compute") or {}).get("worlds_requested", 16))
    our_species = sorted({norm(m.group(1)) for ln in lines if ln.startswith("|switch|" + me) for m in [re.match(r"\|switch\|[^|]+\|([^,|]+)", ln)] if m})
    opp_species = sorted({norm(m.group(1)) for ln in lines if ln.startswith("|switch|" + opp) for m in [re.match(r"\|switch\|[^|]+\|([^,|]+)", ln)] if m})
    # point of no return: last turn with value >= 0.45 (the outcome of that turn flipped the game)
    good = [t for t, v in vt.items() if v is not None and v >= 0.45]
    pnr = max(good) if good else None
    pnr_flags = []
    if pnr is not None and g["result"] == "loss" and pnr in turns:
        T0 = turns[pnr]
        T1 = turns.get(pnr + 1)
        blk = [T0] + ([T1] if T1 else [])
        if any(s_ == me for b in blk for s_, _ in b.ev["crit"]):
            pnr_flags.append("crit_on_us")
        if any(s_ == me for b in blk for s_, _ in b.ev["miss"]):
            pnr_flags.append("our_miss")
        if any(s_ == me for b in blk for s_, _, _ in b.ev["cant"]):
            pnr_flags.append("we_cant")
        if any(s_ == me for s_, _, _ in T0.ev["status"]):
            pnr_flags.append("status_on_us")
        if any(e[0] == opp for e in T0.ev["tera"]):
            pnr_flags.append("opp_tera")
        if any(e[0] == me for e in T0.ev["tera"]):
            pnr_flags.append("our_tera")
        if any(s_ == opp and a_ > 0 for s_, _, _, a_ in T0.ev["boost"]):
            pnr_flags.append("opp_boost")
        # opp has a boosted mon on the field at the pnr (cumulative positive boosts since its last switch-in)
        if any(s_ == me and k_ == "switch" for s_, _, _, k_ in T0.ev["switch"]):
            pnr_flags.append("we_switched")
        if any(s_ == me for s_, _ in T0.ev["faint"]):
            pnr_flags.append("we_lost_mon")
        if any(sp_ and t_ in (pnr, pnr - 1) for t_ in [x["turn"] for x in big] for sp_ in [1]):
            pnr_flags.append("world_surprise")
        # a switch-in of ours that fainted/lost >= 50% in T0
        for s_, sp_, f_, k_ in T0.ev["switch"]:
            if s_ == me:
                lost = sum(max(0.0, o - n_) for ds, dsp, o, n_, _ in T0.ev["damage"] if ds == me and dsp == sp_)
                if lost >= 0.5 or any(fs == me and fsp == sp_ for fs, fsp in T0.ev["faint"]):
                    pnr_flags.append("switch_in_punished")
                    break
    chosen_by_turn = {t: d.get("chosen") for t, d in first_dec.items()}
    shares = {}
    for t, d in first_dec.items():
        agg = collections.Counter()
        for w in d.get("worlds", []):
            for m in w.get("moves", []):
                agg[m["choice"]] += m["visits"]
        tot = sum(agg.values()) or 1
        shares[t] = [(c, round(v / tot, 2)) for c, v in agg.most_common(3)]
    return {
        "tag": tag, "result": g["result"], "turns": g["turns"], "opp": g.get("opponent"),
        "opp_rating": g.get("opponent_rating"), "rating": g.get("rating_before"), "me": me,
        "values": {t: round(v, 3) for t, v in vt.items() if v is not None},
        "v1": vt.get(1), "vmax": max((vt[t] for t in ts if t >= 3), default=None),
        "worst_drop": round(worst[0], 3), "worst_turn": dturn, "expl": expl,
        "waste": [(n, *e) for n, e in waste], "pnr": pnr, "pnr_flags": pnr_flags, "v_pnr": vt.get(pnr), "v_pnr1": vt.get((pnr or 0) + 1), "runs": runs, "loop_turns": loop_turns, "runs": runs, "loop_turns": loop_turns, "n_our_moves": n_our_moves, "status_us": status_us, "status_opp": status_opp, "choice_ko": choice_ko, "n_dec": len(decs), "n_slow": n_slow, "n_short": n_short, "our_species": our_species, "opp_species": opp_species, "surprises": big, "tera_presence": tera_presence, "pres_by_kind": dict(pres_by_kind), "n_surprise": len(big), "n_reveals": len(surprises),
        "events": dict(ev_tot), "bad_switch": bad_switch, "sw_total": sw_total,
        "our_tera": [(n, e[1], e[2]) for n, e in our_tera], "opp_tera": [(n, e[1], e[2]) for n, e in opp_tera],
        "fainted_us": fainted_us, "fainted_opp": fainted_opp, "opp_boost": opp_boost,
        "chosen": chosen_by_turn, "shares": shares,
    }


# ---------------------------------------------------------------------------------------------
# timeline for manual reading
# ---------------------------------------------------------------------------------------------

def timeline(run: str, g: dict, window: int = 0) -> str:
    tag = g["battle_tag"]
    res = analyse(run, g)
    lines = read_log(os.path.join(run, "logs", tag + ".log"))
    me = res["me"]
    turns, reveal, _, _ = parse_log(lines, me)
    out = [f"== {tag} {g['result']} vs {g.get('opponent')} ({g.get('opponent_rating')}) me={me} turns={g['turns']}"]
    wt = res["worst_turn"]
    if window < 0:  # point of no return: last turn with value >= 0.45
        good = [t for t, v in res["values"].items() if v >= 0.45]
        wt = max(good) if good else 1
        window = 1
    out.append("vals: " + " ".join(f"{t}:{v:.2f}" for t, v in sorted(res["values"].items())))
    for n in sorted(turns):
        if window and wt is not None and not (wt - window <= n <= wt + 1):
            continue
        if n == 0:
            continue
        t = turns[n]
        v = res["values"].get(n)
        ch = res["chosen"].get(n)
        bits = []
        for e in t.ev["switch"]:
            bits.append(f"{'US' if e[0]==me else 'OPP'} in:{e[1]}")
        for e in t.ev["move"]:
            bits.append(f"{'us' if e[0]==me else 'opp'} {e[1]}:{e[2]}")
        for e in t.ev["tera"]:
            bits.append(f"TERA {'us' if e[0]==me else 'opp'} {e[1]}={e[2]}")
        for e in t.ev["crit"]:
            bits.append(f"CRIT on {'us' if e[0]==me else 'opp'}")
        for e in t.ev["miss"]:
            bits.append(f"MISS by {'us' if e[0]==me else 'opp'}")
        for e in t.ev["cant"]:
            bits.append(f"CANT {'us' if e[0]==me else 'opp'}:{e[2]}")
        for e in t.ev["faint"]:
            bits.append(f"FAINT {'us' if e[0]==me else 'opp'}:{e[1]}")
        for e in t.ev["status"]:
            bits.append(f"STATUS {'us' if e[0]==me else 'opp'}:{e[2]}")
        first_hp: dict = {}
        last_hp: dict = {}
        for e in t.ev["damage"]:
            first_hp.setdefault((e[0], e[1]), e[2])
            last_hp[(e[0], e[1])] = e[3]
        for (s_, sp_), old in first_hp.items():
            bits.append(f"hp {'us' if s_==me else 'opp'}-{sp_} {old*100:.0f}>{last_hp[(s_, sp_)]*100:.0f}")
        out.append(f"T{n:>2} v={v if v is not None else '--'} chose={ch} {res['shares'].get(n)} | " + "; ".join(bits)[:420])
    out.append("surprises: " + json.dumps(res["surprises"]))
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------

# Hand-reviewed games (tag suffix -> (class, note)). Everything else is classified by rules.
MANUAL = {
    "2700189152": ("futile_moves", "113 turns: Glare x46 into an already-paralysed Giratina, Headbutt x20 into a Ghost, EQ into Corviknight"),
    "hs230knwpw": ("futile_moves", "Substitute fails x11 (sub already up) in a Sub / Leech Seed / Sleep Powder loop"),
    "2700102093": ("futile_moves", "88 turns, Thunder Wave fails x6; passive stall"),
    "2700092169": ("tactical_choice_unlock", "Knock Off removed Choice Band from a Heracross locked into a resisted EQ; it then Close Combat KOd Meganium"),
}

LUCK_FLAGS = {"crit_on_us", "our_miss", "we_cant"}


def primary_class(r) -> str:
    if r["result"] != "loss":
        return ""
    suffix = r["tag"][-10:]
    if suffix in MANUAL:
        return MANUAL[suffix][0]
    f = set(r["pnr_flags"])
    if f & LUCK_FLAGS:
        return "luck_at_flip_turn (crit / miss / full-para)"
    if len(r["waste"]) >= 8:
        return "futile_moves"
    if "world_surprise" in f:
        return "world_model_surprise"
    if "opp_tera" in f:
        return "opp_tera_flip"
    if "opp_boost" in f:
        return "opp_setup_flip"
    if "switch_in_punished" in f:
        return "switch_in_punished"
    if "status_on_us" in f:
        return "status_on_us_flip"
    if "we_lost_mon" in f:
        return "lost_exchange (mon lost at the flip turn, nothing unusual)"
    return "no_single_event (gradual / matchup)"


def luck_index(r) -> int:
    e = r["events"]
    cant_us = sum(v for k, v in e.items() if k.startswith("we_cant"))
    cant_opp = sum(v for k, v in e.items() if k.startswith("opp_cant"))
    return (e.get("crit_on_opp", 0) - e.get("crit_on_us", 0)) + (e.get("opp_miss", 0) - e.get("our_miss", 0)) + (cant_opp - cant_us)


def extra_sections(rows, W, Ls, hi):
    out = []
    wr = lambda g: (sum(r["result"] == "win" for r in g) / len(g)) if g else float("nan")
    out.append("\nluck index per game = (our crits - their crits) + (their misses - our misses) + (their lost turns - ours)")
    for lo, hi_ in ((-99, -2), (-2, 0), (0, 1), (1, 2), (2, 99)):
        g = [r for r in rows if lo <= luck_index(r) < hi_]
        out.append(f"  luck in [{lo},{hi_}) n={len(g):>3} winrate={wr(g):.2f}")
    out.append("\nwin rate by game length (turns)")
    for lo, hi_ in ((0, 16), (16, 25), (25, 35), (35, 50), (50, 999)):
        g = [r for r in rows if lo <= r["turns"] < hi_]
        out.append(f"  turns [{lo},{hi_}) n={len(g):>3} winrate={wr(g):.2f}")
    out.append("\nswitch rate (our switches / turn), wasted moves, status")
    def rate(g, f, d=lambda r: r["turns"]):
        return sum(f(r) for r in g) / max(1, sum(d(r) for r in g))
    for nm, g in (("win", W), ("loss", Ls)):
        out.append(f"  {nm}: switches/turn {rate(g, lambda r: r['sw_total']):.3f}; bad switch-ins/switch {rate(g, lambda r: r['bad_switch'], lambda r: r['sw_total']):.3f}; "
                   f"wasted moves/our move {rate(g, lambda r: len(r['waste']), lambda r: r['n_our_moves']):.3f}; status on us/game {statistics.fmean(r['status_us'] for r in g):.2f}, on opp/game {statistics.fmean(r['status_opp'] for r in g):.2f}")
    wc = collections.Counter((w[3] if len(w) > 3 else "?") for r in rows for w in r["waste"])
    out.append(f"  wasted-move kinds (all games): {dict(wc)}")
    wm = collections.Counter((w[2], w[3]) for r in Ls for w in r["waste"])
    out.append("  most common (move, why) wasted in losses: " + ", ".join(f"{k[0]}:{k[1]}x{v}" for k, v in wm.most_common(12)))
    out.append(f"  Knock Off removing a Choice item: {sum(r['choice_ko'] for r in rows)} times in {sum(1 for r in rows if r['choice_ko'])} games")
    out.append("\nrepeat loops (same mon + same move >= 5 turns in a row)")
    for nm, g in (("win", W), ("loss", Ls)):
        out.append(f"  {nm}: games with a loop {sum(1 for r in g if r['runs'])}/{len(g)}; loop turns per game {statistics.fmean(r['loop_turns'] for r in g):.2f}; "
                   f"loop turns / turns {sum(r['loop_turns'] for r in g)/sum(r['turns'] for r in g):.3f}")
    lm = collections.Counter(r_[2][1] for r in rows for r_ in r["runs"])
    out.append("  loop moves: " + ", ".join(f"{k}x{v}" for k, v in lm.most_common(14)))
    out.append("\nhigh-value losses: games where our value reached >= thr at some turn, and won/lost")
    for thr in (0.8, 0.9, 0.95):
        g = [r for r in rows if max(r["values"].values(), default=0) >= thr]
        out.append(f"  max value >= {thr}: n={len(g)} lost={sum(r['result']=='loss' for r in g)} winrate={wr(g):.2f}")
    out.append("\ncompute: decisions slow(>1.5s)/short-world per game")
    out.append(f"  slow decisions {sum(r['n_slow'] for r in rows)} of {sum(r['n_dec'] for r in rows)}; decisions with fewer worlds than requested {sum(r['n_short'] for r in rows)}")
    return out


def summary(run: str, games: list[dict]) -> str:
    rows = []
    for g in games:
        r = analyse(run, g)
        if r:
            r["class"] = primary_class(r)
            rows.append(r)
    out = []
    W = [r for r in rows if r["result"] == "win"]
    Ls = [r for r in rows if r["result"] == "loss"]
    out.append(f"games analysed: {len(rows)} (wins {len(W)}, losses {len(Ls)})")

    def hi(r):
        return (r["opp_rating"] or 0) >= 2100
    for name, grp in (("all", rows), ("opp>=2100", [r for r in rows if hi(r)]), ("opp<2100", [r for r in rows if not hi(r)])):
        w = sum(r["result"] == "win" for r in grp)
        out.append(f"  {name}: n={len(grp)} winrate={w/len(grp):.3f}" if grp else f"  {name}: n=0")
    # rating buckets
    out.append("\nwin rate by opp rating bucket")
    for lo, hi_ in ((0, 1900), (1900, 2100), (2100, 2250), (2250, 2400), (2400, 9999)):
        grp = [r for r in rows if lo <= (r["opp_rating"] or 0) < hi_]
        if grp:
            out.append(f"  [{lo},{hi_}) n={len(grp)} wr={sum(r['result']=='win' for r in grp)/len(grp):.2f}")

    def per_game(key_fn, grp):
        return statistics.fmean(key_fn(r) for r in grp) if grp else float("nan")
    out.append("\nper-game base rates (wins / losses)")
    keys = {
        "crit_on_us": lambda r: r["events"].get("crit_on_us", 0), "crit_on_opp": lambda r: r["events"].get("crit_on_opp", 0),
        "our_miss": lambda r: r["events"].get("our_miss", 0), "opp_miss": lambda r: r["events"].get("opp_miss", 0),
        "we_cant_*": lambda r: sum(v for k, v in r["events"].items() if k.startswith("we_cant")),
        "opp_cant_*": lambda r: sum(v for k, v in r["events"].items() if k.startswith("opp_cant")),
        "world surprises": lambda r: r["n_surprise"], "reveals": lambda r: r["n_reveals"],
        "bad switch-ins": lambda r: r["bad_switch"], "switches": lambda r: r["sw_total"],
        "we tera (any)": lambda r: len(r["our_tera"]), "opp tera (any)": lambda r: len(r["opp_tera"]),
        "opp +boosts": lambda r: r["opp_boost"], "v turn1": lambda r: r["v1"] or 0.5,
        "turns": lambda r: r["turns"],
    }
    for k, fn in keys.items():
        out.append(f"  {k:<18} win {per_game(fn, W):6.2f}   loss {per_game(fn, Ls):6.2f}   (hi-rated losses {per_game(fn, [r for r in Ls if hi(r)]):6.2f})")
    # tera usage
    out.append("\ntera usage")
    for name, grp in (("all", rows), ("opp>=2100", [r for r in rows if hi(r)])):
        used = [r for r in grp if r["our_tera"]]
        unused = [r for r in grp if not r["our_tera"]]
        out.append(f"  {name}: tera used n={len(used)} wr={sum(r['result']=='win' for r in used)/max(1,len(used)):.2f}; unused n={len(unused)} wr={sum(r['result']=='win' for r in unused)/max(1,len(unused)):.2f}")
        tt = collections.defaultdict(list)
        for r in used:
            t = r["our_tera"][0][0]
            tt["<=2" if t <= 2 else "3-6" if t <= 6 else "7-12" if t <= 12 else ">12"].append(r["result"] == "win")
        out.append("    first-tera turn bucket -> n, wr: " + ", ".join(f"{k}:{len(v)},{sum(v)/len(v):.2f}" for k, v in sorted(tt.items())))
    # opp tera
    ot = [r for r in rows if r["opp_tera"]]
    out.append(f"  opp teras: games {len(ot)} wr(us)={sum(r['result']=='win' for r in ot)/max(1,len(ot)):.2f}; opp no tera n={len(rows)-len(ot)} wr(us)={sum(r['result']=='win' for r in rows if not r['opp_tera'])/max(1,len(rows)-len(ot)):.2f}")
    # value calibration
    out.append("\ncalibration: our value at turn t (mean over 16 worlds) vs win rate")
    for t in (1, 3, 5, 8, 12):
        pts = [(r["values"].get(t), r["result"] == "win") for r in rows if r["values"].get(t) is not None]
        for lo, hi_ in ((0, .3), (.3, .45), (.45, .55), (.55, .7), (.7, 1.01)):
            g_ = [w for v, w in pts if lo <= v < hi_]
            if len(g_) >= 5:
                out.append(f"  t={t:>2} v in [{lo:.2f},{hi_:.2f}) n={len(g_):>3} win={sum(g_)/len(g_):.2f}")
    # classes
    out.append("\nprimary class of decisive turn, losses (all / opp>=2100)")
    ca = collections.Counter(r["class"] for r in Ls)
    ch = collections.Counter(r["class"] for r in Ls if hi(r))
    nh = sum(1 for r in Ls if hi(r))
    for k, v in ca.most_common():
        out.append(f"  {k:<42} {v:>3} ({v/len(Ls):.0%})   hi {ch.get(k,0):>3} ({ch.get(k,0)/max(1,nh):.0%})")
    out.append(f"  (losses {len(Ls)}, hi-rated losses {nh})")
    # multi-label explanation counts
    out.append("\nmulti-label events at decisive turn (losses with a decisive drop)")
    cc = collections.Counter(x for r in Ls if r["worst_drop"] <= -DROP_DECISIVE for x in r["expl"])
    for k, v in cc.most_common():
        out.append(f"  {k:<18} {v}")
    # surprise effect
    out.append("\nsurprise reveal kinds (losses / wins), per game")
    for res_, grp in (("loss", Ls), ("win", W)):
        c = collections.Counter(s["kind"] for r in grp for s in r["surprises"])
        out.append(f"  {res_}: " + ", ".join(f"{k}={v/len(grp):.2f}" for k, v in c.items()))
    out.extend(extra_sections(rows, W, Ls, hi))
    out.append("\nworld presence of facts at first reveal (mean share of the 16 worlds containing the fact)")
    for kind in ("move", "item", "ability", "tera"):
        for nm, grp in (("win", W), ("loss", Ls)):
            xs = [x for r in grp for x in r["pres_by_kind"].get(kind, [])]
            if xs:
                out.append(f"  {kind:<8} {nm:<5} n={len(xs):>4} mean={statistics.fmean(xs):.2f} frac<=1/8={sum(x <= 0.125 for x in xs)/len(xs):.3f} frac==0={sum(x == 0 for x in xs)/len(xs):.3f}")
    out.append("\nopp tera types: bot's world presence (tera surprise, our losses)")
    for r in Ls:
        for s in r["surprises"]:
            if s["kind"] == "tera":
                out.append(f"  {r['tag'][-10:]} T{s['turn']} {s['species']} tera {s['value']} presence {s['presence']}")
    # outcome vs fainted
    out.append("\nclose losses (opp mons left at the end)")
    rem = collections.Counter(6 - r["fainted_opp"] for r in Ls)
    out.append("  " + ", ".join(f"{k} left: {v}" for k, v in sorted(rem.items())))
    out.append("\nper loss listing: tag opp_rating turns worst_drop@turn class expl")
    for r in sorted(Ls, key=lambda r: -(r["opp_rating"] or 0)):
        out.append(f"  {r['tag'][-10:]} {r['opp_rating']} t={r['turns']:>2} v1={r['v1'] and round(r['v1'],2)} drop={r['worst_drop']}@{r['worst_turn']} {r['class']} {','.join(r['expl'])} bad_sw={r['bad_switch']} surpr={r['n_surprise']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--game")
    ap.add_argument("--dump")
    ap.add_argument("--window", type=int, default=0)
    a = ap.parse_args()
    games = [json.loads(l) for l in open(os.path.join(a.run, "games.jsonl"), encoding="utf-8")]
    sys.stdout.reconfigure(encoding="utf-8")
    if a.game:
        g = next(x for x in games if x["battle_tag"].endswith(a.game))
        print(timeline(a.run, g, a.window))
    if a.summary:
        print(summary(a.run, games))
    if a.dump:
        rows = [analyse(a.run, g) for g in games]
        json.dump([r for r in rows if r], open(a.dump, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
