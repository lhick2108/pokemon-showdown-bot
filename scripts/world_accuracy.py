"""How close are our sampled worlds to the exact "truth" state? (bead 589.27)

Reads self-play recordings (docs/selfplay-data.md): every searched decision stores
the sampled worlds that were searched and, when both players ran in one process, the
exact opposing team ("truth"). For each decision this script lines up the revealed
opposing Pokemon in each world with the truth and reports per-field mismatch rates.

    python scripts/world_accuracy.py data/selfplay/gen0 [--max-files N] [--top 12]
    python scripts/world_accuracy.py <dir> --by [turn reveals onfield] [--gaps]   # bead 589.34

``--by`` prints error rates bucketed by turn, by facts revealed about the Pokemon (moves seen, item known,
ability known) and by turns on the field; recordings made after 589.34 carry the tracker's own view
(``opp_facts``: revealed facts, number of consistent sets, entropy, candidate groups), older ones fall back to
proxies (moves seen = PP used, "known" = every world agrees). ``--gaps`` adds the worst species/items, confidently
wrong beliefs and the sets the tracker wrongly ruled out. Ditto is skipped (its truth holds the copied set), and
turn-1 truth states carry no moves (peer request not yet seen), so read turn-1 move columns with care.

Definitions
-----------
* ``revealed``: world slot i is revealed if every world of the decision agrees on the
  species of slots 0..i (the opponent's seen Pokemon come first, in the same order in
  every world; unrevealed slots are re-drawn per world). A small number of unrevealed
  slots can agree by chance (~1/150 per slot per extra world); that only dilutes rates.
* Truth is matched to a world slot by species id.
* ``known moves`` are the truth moves whose PP is below the move's maximum: the truth
  builder takes PP from poke-env's record of moves the opponent was seen to use, so a
  used move shows reduced PP. A known move missing from a world is "revealed info lost".
* ``confident-wrong``: all worlds of the decision agree on a value (the model
  "believes" it) and it differs from truth.
* Fields built from the public battle view (HP fraction, boosts, status, side
  conditions, weather/terrain and counters, volatiles) use the same code for the world
  and the truth, so they cannot differ except through the sampled set (max HP, move
  order). They are still reported: any mismatch on OUR side, or a non-zero mismatch on
  a public field, is a bug (nondeterminism / ordering) rather than inference error.
  They are NOT compared with the opponent's own view (the recording only stores the
  public view), so tracker bugs (wrong counters, lost boosts) need live_tracker-style
  checks; see the bead notes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import poke_engine as pe

BOOST_KEYS = (
    "attack_boost", "defense_boost", "special_attack_boost", "special_defense_boost",
    "speed_boost", "accuracy_boost", "evasion_boost",
)
STAT_KEYS = ("attack", "defense", "special_attack", "special_defense", "speed")
MON_PUBLIC = ("status", "sleep_turns", "rest_turns", "terastallized", "level")

CHOICE = {"choiceband", "choicespecs", "choicescarf"}
ITEM_CLASS = {
    "choiceband": "choice-atk", "choicespecs": "choice-atk", "choicescarf": "choice-scarf",
    "lifeorb": "lifeorb", "assaultvest": "bulk(AV/eviolite)", "eviolite": "bulk(AV/eviolite)",
    "leftovers": "recovery", "blacksludge": "recovery", "sitrusberry": "recovery",
    "heavydutyboots": "boots", "rockyhelmet": "rockyhelmet", "none": "none",
}


def item_class(item: str) -> str:
    return ITEM_CLASS.get(str(item).lower(), "other")


def public(obj) -> dict:
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        out[name] = value
    return out


def boost_mult(stage: int) -> float:
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)


def eff_speed(mon, side) -> float:
    speed = mon.speed * boost_mult(int(side.speed_boost))
    if mon.item == "CHOICESCARF" or str(mon.item).lower() == "choicescarf":
        speed *= 1.5
    if str(mon.status).lower().endswith("paralyze"):
        speed *= 0.5
    return speed


def moves_of(mon) -> dict[str, int]:
    return {str(m.id).lower(): int(m.pp) for m in mon.moves if str(m.id).lower() != "none"}


class Counts:
    """mismatch / total counters keyed by field, split by group."""

    def __init__(self) -> None:
        self.bad: Counter = Counter()
        self.total: Counter = Counter()
        self.err_sum: defaultdict = defaultdict(float)
        self.confident_wrong: Counter = Counter()
        self.confident_total: Counter = Counter()
        self.pairs: defaultdict = defaultdict(Counter)

    def add(self, key: str, mismatch: bool, err: float | None = None) -> None:
        self.total[key] += 1
        if mismatch:
            self.bad[key] += 1
        if err is not None:
            self.err_sum[key] += err

    def add_conf(self, key: str, all_agree_wrong: bool, all_agree: bool) -> None:
        if all_agree:
            self.confident_total[key] += 1
            if all_agree_wrong:
                self.confident_wrong[key] += 1


def parse(text: str):
    return pe.State.from_string(text)


def analyse_decision(decision: dict, counts: dict[str, Counts], stats: Counter, bd: Breakdown | None = None) -> None:
    truth_text = (decision.get("truth") or {}).get("state")
    worlds = decision.get("worlds") or []
    if not truth_text or len(worlds) < 2:
        return
    truth = parse(truth_text)
    ws = [parse(w["state"]) for w in worlds]
    stats["decisions"] += 1
    stats["worlds"] += len(ws)

    # ---- revealed prefix length: species agree across all worlds -------------
    n_slots = min(len(w.side_two.pokemon) for w in ws)
    revealed = 0
    for i in range(n_slots):
        species = {str(w.side_two.pokemon[i].id).lower() for w in ws}
        if len(species) == 1 and str(ws[0].side_two.pokemon[i].id).lower() not in ("none", ""):
            revealed += 1
        else:
            break
    stats["revealed_slots"] += revealed
    t_by_species = {str(m.id).lower(): m for m in truth.side_two.pokemon}
    t_active = truth.side_two.pokemon[int(truth.side_two.active_index)]
    if bd is not None:
        observe_breakdown(bd, decision, ws, truth, revealed, t_by_species)

    # ---- field / side level (every world vs truth) ---------------------------
    for w in ws:
        c = counts["field"]
        c.add("weather", w.weather != truth.weather)
        c.add("weather_turns", w.weather_turns_remaining != truth.weather_turns_remaining)
        c.add("terrain", w.terrain != truth.terrain)
        c.add("terrain_turns", w.terrain_turns_remaining != truth.terrain_turns_remaining)
        c.add("trick_room", (w.trick_room, w.trick_room_turns_remaining) != (truth.trick_room, truth.trick_room_turns_remaining))
        for side_name, tag in (("side_one", "our"), ("side_two", "opp")):
            ws_side, ts_side = getattr(w, side_name), getattr(truth, side_name)
            c = counts[f"side:{tag}"]
            for key in BOOST_KEYS:
                c.add("boosts", getattr(ws_side, key) != getattr(ts_side, key))
            sc_w, sc_t = public(ws_side.side_conditions), public(ts_side.side_conditions)
            for key in sorted(set(sc_w) | set(sc_t)):
                c.add(f"cond:{key}", sc_w.get(key) != sc_t.get(key))
            c.add("side_conditions_any", sc_w != sc_t)
            c.add("volatiles", set(ws_side.volatile_statuses) != set(ts_side.volatile_statuses))
            dw, dt = public(ws_side.volatile_status_durations), public(ts_side.volatile_status_durations)
            c.add("volatile_durations", dw != dt)
            c.add("substitute_health", ws_side.substitute_health != ts_side.substitute_health)
            c.add("force_switch/trapped", (ws_side.force_switch, ws_side.force_trapped) != (ts_side.force_switch, ts_side.force_trapped))
            # last used move compared by move id when it is a move index
            def lum(side):
                text = str(side.last_used_move)
                if text.startswith("move:") and text != "move:none":
                    try:
                        return str(side.pokemon[int(side.active_index)].moves[int(text[5:])].id).lower()
                    except Exception:
                        return "?"
                return text
            c.add("last_used_move", lum(ws_side) != lum(ts_side))
            # active mon public state
            wa, ta = ws_side.pokemon[int(ws_side.active_index)], ts_side.pokemon[int(ts_side.active_index)]
            if str(wa.id).lower() == str(ta.id).lower():
                for key in MON_PUBLIC:
                    c.add(f"active:{key}", getattr(wa, key) != getattr(ta, key))

    # ---- our own bench / active vs truth (should be identical) ----------------
    for w in ws:
        c = counts["our_mons"]
        for wm, tm in zip(w.side_one.pokemon, truth.side_one.pokemon):
            for key in ("id", "hp", "maxhp", "item", "ability", "status", "tera_type", "terastallized",
                        "attack", "defense", "special_attack", "special_defense", "speed"):
                c.add(key, getattr(wm, key) != getattr(tm, key))
            c.add("moves(set+pp)", moves_of(wm) != moves_of(tm))

    # ---- unrevealed slots: species hit rate and set quality when the species is right
    revealed_ids = {str(ws[0].side_two.pokemon[i].id).lower() for i in range(revealed)}
    t_unrevealed = {s for s in t_by_species if s not in revealed_ids and s not in ("none", "")}
    for w in ws:
        c = counts["opp_unrevealed"]
        for i in range(revealed, len(w.side_two.pokemon)):
            wm = w.side_two.pokemon[i]
            sid = str(wm.id).lower()
            if sid in ("none", ""):
                continue
            c.add("species hit (in real unrevealed set)", sid not in t_unrevealed)
            t_mon = t_by_species.get(sid)
            if t_mon is not None and sid in t_unrevealed:
                for key in ("item", "ability"):
                    c.add(key, str(getattr(wm, key)).lower() != str(getattr(t_mon, key)).lower())
                c.add("moves recall (of 4)", False,
                      err=len(set(moves_of(wm)) & set(moves_of(t_mon))) / max(1, len(moves_of(t_mon))))

    # ---- opponent revealed mons ----------------------------------------------
    our_active = truth.side_one.pokemon[int(truth.side_one.active_index)]
    for i in range(revealed):
        group = "opp_active" if i == 0 else "opp_bench"
        t_mon = t_by_species.get(str(ws[0].side_two.pokemon[i].id).lower())
        if i == 0 and str(t_active.id).lower() != str(ws[0].side_two.pokemon[0].id).lower():
            counts["opp_active"].add("species (shown!=real)", True)
            stats["active_species_wrong"] += 1
            continue
        if t_mon is None:
            counts[group].add("species (not on real team)", True)
            continue
        c = counts[group]
        c.add("species (not on real team)", False)
        t_moves = moves_of(t_mon)
        known = {m for m, pp in t_moves.items() if pp < _max_pp(m)}
        # per-field, per-world comparison
        agree: dict[str, set] = defaultdict(set)
        for w in ws:
            wm = w.side_two.pokemon[i]
            for key in ("item", "ability", "tera_type", "level"):
                wv, tv = str(getattr(wm, key)).lower(), str(getattr(t_mon, key)).lower()
                if key == "tera_type" and tv == "typeless":
                    continue  # recordings made before the truth-builder fix lack the real tera type
                c.add(key, wv != tv)
                agree[key].add(wv)
                if key == "item" and wv != tv:
                    c.pairs["item"][(tv, wv)] += 1
                if key == "ability" and wv != tv:
                    c.pairs["ability"][(tv, wv)] += 1
                if key == "tera_type" and wv != tv:
                    c.pairs["tera_type"][(tv, wv)] += 1
            w_item, t_item = str(wm.item).lower(), str(t_mon.item).lower()
            c.add("item class (offense/bulk/speed/recovery/boots/..)", item_class(w_item) != item_class(t_item))
            c.add("choice item vs not", (w_item in CHOICE) != (t_item in CHOICE))
            c.add("scarf vs not", (w_item == "choicescarf") != (t_item == "choicescarf"))
            if known:
                # a Pokemon that has used a move since entering is choice-locked in FP's worlds
                # (the "mismatch" column is a SHARE here, not an error rate)
                c.add("share: world has choice item | a move known", w_item in CHOICE)
                c.add("share: truth has choice item | a move known", t_item in CHOICE)
                if len(known) >= 2:
                    c.add("share: world has choice item | >=2 moves known", w_item in CHOICE)
                    c.add("share: truth has choice item | >=2 moves known", t_item in CHOICE)
            w_moves = moves_of(wm)
            overlap = len(set(w_moves) & set(t_moves))
            c.add("moveset exact (4/4)", overlap != len(t_moves) or len(w_moves) != len(t_moves))
            c.add("move slots wrong (of 4)", False, err=len(t_moves) - overlap)
            if known:
                c.add("known move lost", not known <= set(w_moves))
                if not known <= set(w_moves):
                    stats["known_move_lost_examples"] += 1
            unk_t = set(t_moves) - known
            if unk_t:
                c.add("unrevealed-move recall", False, err=len(unk_t & set(w_moves)) / len(unk_t))
            agree["moves"].add(frozenset(w_moves))
            for key in STAT_KEYS:
                wv, tv = getattr(wm, key), getattr(t_mon, key)
                c.add(key, wv != tv, err=abs(wv - tv) / max(1, tv))
            # HP: fraction (public) and absolute points (depend on the sampled max HP)
            wf = wm.hp / max(1, wm.maxhp)
            tf = t_mon.hp / max(1, t_mon.maxhp)
            c.add("maxhp", wm.maxhp != t_mon.maxhp, err=abs(wm.maxhp - t_mon.maxhp) / max(1, t_mon.maxhp))
            c.add("hp fraction", abs(wf - tf) > 1e-9, err=abs(wf - tf))
            c.add("hp points", wm.hp != t_mon.hp, err=abs(wm.hp - t_mon.hp))
            for key in MON_PUBLIC:
                c.add(key, getattr(wm, key) != getattr(t_mon, key))
            c.add("types", (wm.types, ) != (t_mon.types, ))
            if i == 0:
                w_fast = eff_speed(wm, w.side_two) > eff_speed(our_active, w.side_one)
                t_fast = eff_speed(t_mon, truth.side_two) > eff_speed(our_active, truth.side_one)
                c.add("speed order vs us (boosts/scarf/par)", w_fast != t_fast)
        for key, values in agree.items():
            if len(values) != 1:
                continue
            value = next(iter(values))
            if key == "moves":
                wrong = not known <= set(value) if known else False
                c.add_conf("moves (known lost)", wrong, bool(known))
                continue
            c.add_conf(key, value != str(getattr(t_mon, key)).lower(), True)


TURN_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 10, "7-10"), (11, 20, "11-20"), (21, 10**9, "21+"))
FIELD_BUCKETS = ((0, 0, "0 (bench)"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 8, "5-8"), (9, 10**9, "9+"))


def bucket(value: int, table) -> str:
    for lo, hi, name in table:
        if lo <= value <= hi:
            return name
    return table[-1][2]


class Agg:
    """Per-bucket sums over (revealed mon x world) observations."""

    def __init__(self) -> None:
        self.v: Counter = Counter()

    def __getitem__(self, key: str) -> float:
        return self.v[key]


class Breakdown:
    """Accuracy per bucket along turn / reveals / turns-on-field (bead 589.34)."""

    def __init__(self) -> None:
        self.tables: dict[str, dict[str, Agg]] = defaultdict(lambda: defaultdict(Agg))
        self.gap_item: Counter = Counter()
        self.gap_item_total: Counter = Counter()
        self.gap_species_total: Counter = Counter()
        self.gap_species_bad: Counter = Counter()
        self.gap_conf: Counter = Counter()
        self.gap_conf_pair: Counter = Counter()
        self.gap_ability_pair: Counter = Counter()
        self.gap_moves: Counter = Counter()
        self.have_facts = 0
        self.no_facts = 0
        self.onfield: dict[tuple, set] = defaultdict(set)
        # residual ambiguity among well-revealed mons (>=2 moves seen, turn >= 7), from tracker facts
        self.amb_n = 0
        # truth set missing from the tracker's candidates ("unsupported")
        self.unsup_total = 0
        self.unsup_comp: Counter = Counter()
        self.unsup_item: Counter = Counter()
        self.unsup_abil: Counter = Counter()
        self.unsup_moves: Counter = Counter()
        self.amb_vary: Counter = Counter()
        self.amb_species_n: Counter = Counter()
        self.amb_species_h: defaultdict = defaultdict(float)
        self.amb_species_vary: defaultdict = defaultdict(Counter)

    def add(self, dims: dict[str, str], obs: dict[str, float]) -> None:
        for dim, key in dims.items():
            agg = self.tables[dim][key]
            for name, val in obs.items():
                agg.v[name] += val


def observe_breakdown(bd: Breakdown, decision: dict, ws, truth, revealed: int, t_by_species) -> None:
    facts = decision.get("opp_facts") or []
    turn = int(decision.get("turn") or 0)
    game = decision.get("_game", "")
    if revealed:
        bd.onfield[(game, str(ws[0].side_two.pokemon[0].id).lower())].add(turn)
    n_worlds = len(ws)
    for i in range(revealed):
        sid = str(ws[0].side_two.pokemon[i].id).lower()
        t_mon = t_by_species.get(sid)
        if t_mon is None or sid == "ditto":
            continue  # Ditto: the truth state holds the transformed mon's moves/ability (measurement artefact)
        t_moves = moves_of(t_mon)
        known = {m for m, pp in t_moves.items() if pp < _max_pp(m)}
        fact = facts[i] if i < len(facts) and "moves" in facts[i] else None
        t_item, t_abil, t_tera = (str(getattr(t_mon, k)).lower() for k in ("item", "ability", "tera_type"))
        t_tera_done = bool(t_mon.terastallized)
        items = [str(w.side_two.pokemon[i].item).lower() for w in ws]
        abils = [str(w.side_two.pokemon[i].ability).lower() for w in ws]
        w_moves = [moves_of(w.side_two.pokemon[i]) for w in ws]
        w_tera = [str(w.side_two.pokemon[i].tera_type).lower() for w in ws]
        item_agree = len(set(items)) == 1
        abil_agree = len(set(abils)) == 1
        moves_agree = len({frozenset(m) for m in w_moves}) == 1
        sig = {(items[k], abils[k], frozenset(w_moves[k])) for k in range(n_worlds)}
        if fact is not None:
            bd.have_facts += 1
            moves_seen = len(fact["moves"])
            item_known = fact["item"] is not None
            abil_known = fact["ability"] is not None
            suffix = ""
        else:
            bd.no_facts += 1
            moves_seen = len(known)
            item_known, abil_known = item_agree, abil_agree  # proxy: every world holds the same value
            suffix = " (proxy: worlds agree)"
        on_field = len(bd.onfield[(game, sid)])
        dims = {
            "turn": bucket(turn, TURN_BUCKETS),
            "moves seen": str(min(moves_seen, 4)),
            "item known" + suffix: "yes" if item_known else "no",
            "ability known" + suffix: "yes" if abil_known else "no",
            "turns on field": bucket(on_field, FIELD_BUCKETS),
        }
        for k in range(n_worlds):
            item_bad = items[k] != t_item
            abil_bad = abils[k] != t_abil
            overlap = len(set(w_moves[k]) & set(t_moves))
            moves_ex = overlap == len(t_moves) and len(w_moves[k]) == len(t_moves)
            tera_bad = w_tera[k] != t_tera and t_tera != "typeless"
            bd.add(dims, {
                "n": 1, "item": item_bad, "item_class": item_class(items[k]) != item_class(t_item),
                "ability": abil_bad, "tera_n": t_tera_done, "tera_bad": tera_bad and t_tera_done,
                "slots_wrong": len(t_moves) - overlap, "moves_exact": moves_ex,
                "full_exact": (not item_bad) and (not abil_bad) and moves_ex,
                "distinct": len(sig) / n_worlds,
            })
        item_conf_bad = item_agree and items[0] != t_item
        full_conf = item_agree and abil_agree and moves_agree
        full_conf_bad = full_conf and (
            items[0] != t_item or abils[0] != t_abil or frozenset(w_moves[0]) != frozenset(t_moves)
        )
        obs = {"dec_n": 1, "conf_item_n": item_agree, "conf_item_bad": item_conf_bad,
               "conf_full_n": full_conf, "conf_full_bad": full_conf_bad}
        if fact is not None:
            obs.update(facts_n=1, n_sets=fact.get("n_sets", 0), entropy=fact.get("entropy", 0.0),
                       n_sets_nt=fact.get("n_sets_nt", fact.get("n_sets", 0)),
                       entropy_nt=fact.get("entropy_nt", fact.get("entropy", 0.0)))
        if fact is not None and fact.get("groups"):
            norm = lambda x: "".join(ch for ch in str(x).lower() if ch.isalnum())  # noqa: E731
            t_key = (norm(t_item), norm(t_abil), tuple(sorted(norm(m) for m in t_moves)))
            p_truth, collision, total_p = 0.0, 0.0, 0.0
            for g_item, g_abil, g_moves, prob in fact["groups"]:
                key = (norm(g_item), norm(g_abil), tuple(sorted(norm(m) for m in g_moves.split("/"))))
                total_p += prob
                collision += prob * prob
                if (key[0] == t_key[0] or t_key[0] in ("none", "")) and (key[1] == t_key[1] or t_key[1] in ("none", "")) and key[2] == t_key[2]:  # truth item none = consumed/knocked off
                    p_truth += prob
            if total_p > 0 and p_truth == 0:
                best = None
                for g_item, g_abil, g_moves, prob in fact["groups"]:
                    gm = tuple(sorted(norm(m) for m in g_moves.split("/")))
                    comps = (norm(g_item) == t_key[0] or t_key[0] in ("none", ""), norm(g_abil) == t_key[1] or t_key[1] in ("none", ""), gm == t_key[2])
                    score = (sum(comps), prob)
                    if best is None or score > best[0]:
                        best = (score, comps, norm(g_item), norm(g_abil), gm)
                _, comps, g_item, g_abil, gm = best
                bd.unsup_total += 1
                bd.unsup_comp[tuple(n for n, ok in zip(("item", "ability", "moves"), comps) if not ok)] += 1
                if not comps[0]:
                    bd.unsup_item[(sid, t_key[0], g_item)] += 1
                if not comps[1]:
                    bd.unsup_abil[(sid, t_key[1], g_abil)] += 1
                if not comps[2]:
                    bd.unsup_moves[(sid, "truth-only " + ",".join(sorted(set(t_key[2]) - set(gm))),
                                    "cand-only " + ",".join(sorted(set(gm) - set(t_key[2]))))] += 1
            if total_p > 0:
                obs.update(grp_n=1, grp_support=p_truth > 0, grp_ptruth=p_truth / total_p,
                           grp_ideal_ok=collision / (total_p * total_p))
        bd.add(dims, obs)
        if fact is not None and "vary" in fact and moves_seen >= 2 and turn >= 7:
            bd.amb_n += 1
            bd.amb_species_n[sid] += 1
            bd.amb_species_h[sid] += fact.get("entropy_nt", 0.0)
            for dim, count in fact["vary"].items():
                if count > 1:
                    bd.amb_vary[dim] += 1
                    bd.amb_species_vary[sid][dim] += 1
        if moves_seen >= 3:  # well-revealed mons
            bd.gap_species_total[sid] += n_worlds
            bd.gap_species_bad[sid] += sum(x != t_item for x in items)
            bd.gap_item_total[t_item] += n_worlds
            bd.gap_item[t_item] += sum(x != t_item for x in items)
            for x in items:
                if x != t_item:
                    bd.gap_conf_pair[(sid, t_item, x)] += 1
            if item_conf_bad:
                bd.gap_conf[(sid, t_item, items[0])] += 1
            for x in abils:
                if x != t_abil:
                    bd.gap_ability_pair[(sid, t_abil, x)] += 1
        for m in known - set(w_moves[0]):
            bd.gap_moves[(sid, m)] += 1


def print_breakdown(bd: Breakdown, dims: list[str]) -> None:
    src = (f"tracker facts on {bd.have_facts} mon-decisions" if bd.have_facts
           else "NO tracker facts (older recording): reveal facts are proxies")
    print(f"\n#### breakdowns ({src}; {bd.no_facts} mon-decisions without facts)")
    order = {"turn": [b[2] for b in TURN_BUCKETS], "turns on field": [b[2] for b in FIELD_BUCKETS]}
    for dim, rows in bd.tables.items():
        if not any(d in dim for d in dims):
            continue
        print(f"\n== by {dim} ==")
        print(f"{'bucket':14s} {'mon-dec':>8s} {'item%':>6s} {'iclass%':>7s} {'abil%':>6s} {'teraPost%':>9s} {'mvSlotErr':>9s} "
              f"{'mvExact%':>8s} {'FULLerr%':>8s} {'confItem%':>9s} {'confFull%':>9s} {'nSets':>7s} {'H':>6s} {'nSetsNT':>7s} {'H_NT':>6s} {'distinct/w':>10s} {'support%':>8s} {'P(truth)':>8s} {'idealErr%':>9s}")
        for key in order.get(dim) or sorted(rows):
            a = rows.get(key)
            if a is None or not a["n"]:
                continue
            n = a["n"]

            def pct(x: float, y: float) -> float:
                return 100 * x / y if y else float("nan")

            nsets = a["n_sets"] / a["facts_n"] if a["facts_n"] else float("nan")
            ent = a["entropy"] / a["facts_n"] if a["facts_n"] else float("nan")
            nsnt = a["n_sets_nt"] / a["facts_n"] if a["facts_n"] else float("nan")
            entnt = a["entropy_nt"] / a["facts_n"] if a["facts_n"] else float("nan")
            print(f"{key:14s} {int(a['dec_n']):8d} {pct(a['item'], n):6.1f} {pct(a['item_class'], n):7.1f} {pct(a['ability'], n):6.1f} "
                  f"{pct(a['tera_bad'], a['tera_n']):9.1f} {a['slots_wrong'] / n:9.2f} {pct(a['moves_exact'], n):8.1f} "
                  f"{100 - pct(a['full_exact'], n):8.1f} {pct(a['conf_item_bad'], a['conf_item_n']):9.1f} "
                  f"{pct(a['conf_full_bad'], a['conf_full_n']):9.1f} {nsets:7.1f} {ent:6.2f} {nsnt:7.1f} {entnt:6.2f} {a['distinct'] / n:10.2f} "
                  f"{pct(a['grp_support'], a['grp_n']):8.1f} {a['grp_ptruth'] / a['grp_n'] if a['grp_n'] else float('nan'):8.3f} "
                  f"{100 - pct(a['grp_ideal_ok'], a['grp_n']):9.1f}")
    print("\nsupport% = share of mon-decisions where the truth's exact (item, ability, moves) set is among the tracker's "
          "remaining candidates (100 - support% = wrongly excluded: a bug/missing-rule signal). P(truth) = probability "
          "the tracker's posterior gives the true set. idealErr% = expected FULLerr% of a perfectly calibrated sampler "
          "from the tracker's posterior (compare with FULLerr%: equal = information-limited).")
    print("\nFULLerr% = 100 - P(item, ability and move set all exactly right). teraPost% = tera error only where the "
          "truth already terastallized (tera before that is unknowable). confItem%/confFull% = share of decisions "
          "where all worlds agree but differ from truth (denominator: all-agree decisions).")


def print_gaps(bd: Breakdown, top: int) -> None:
    if bd.unsup_total:
        print(f"\n#### truth set NOT among the tracker's candidates: {bd.unsup_total} mon-decisions; nearest candidate differs in:")
        for comp, c in bd.unsup_comp.most_common():
            print(f"   {'+'.join(comp) or '(none?)':24s} {c:6d}  ({100 * c / bd.unsup_total:.0f}%)")
        print("-- item (species, truth -> nearest candidate):")
        for (sid, t, g), c in bd.unsup_item.most_common(top):
            print(f"   {sid:16s} {t:16s} -> {g:16s} {c}")
        print("-- ability (species, truth -> nearest candidate):")
        for (sid, t, g), c in bd.unsup_abil.most_common(top):
            print(f"   {sid:16s} {t:16s} -> {g:16s} {c}")
        print("-- moves (species, truth-only moves, candidate-only moves):")
        for (sid, a, b), c in bd.unsup_moves.most_common(top):
            print(f"   {sid:16s} {a:40s} {b:40s} {c}")
    if bd.amb_n:
        print(f"\n#### residual ambiguity (tracker's remaining consistent sets), mons with >=2 moves seen at turn>=7: n={bd.amb_n}")
        print("   share of mon-decisions where the remaining sets still differ in: "
              + ", ".join(f"{d} {100 * bd.amb_vary[d] / bd.amb_n:.0f}%" for d in ("item", "ability", "tera", "moves")))
        print("-- species with the most residual entropy (bits, tera ignored), n>=15:")
        rows = [(bd.amb_species_h[sp] / n, n, sp) for sp, n in bd.amb_species_n.items() if n >= 15]
        for h, n, sp in sorted(rows, reverse=True)[:top]:
            vary = ", ".join(f"{d} {100 * c / n:.0f}%" for d, c in bd.amb_species_vary[sp].most_common())
            print(f"   {sp:16s} n={n:5d} H={h:5.2f}  differs in: {vary}")
    print("\n#### gaps among well-revealed mons (>=3 moves seen), item errors counted per world")
    print("-- by truth item (absolute error counts):")
    for item, bad in bd.gap_item.most_common(top):
        print(f"   {item:18s} err {bad:7d} / {bd.gap_item_total[item]:7d} = {100 * bad / bd.gap_item_total[item]:5.1f}%")
    print("-- by species (absolute error counts):")
    for sid, bad in bd.gap_species_bad.most_common(top):
        print(f"   {sid:18s} err {bad:7d} / {bd.gap_species_total[sid]:7d} = {100 * bad / bd.gap_species_total[sid]:5.1f}%")
    print("-- top (species, truth item -> world item):")
    for (sid, t, w), c in bd.gap_conf_pair.most_common(top * 2):
        print(f"   {sid:16s} {t:16s} -> {w:16s} {c}")
    print("-- CONFIDENTLY wrong item (all worlds agree, wrong), (species, truth -> believed):")
    for (sid, t, w), c in bd.gap_conf.most_common(top * 2):
        print(f"   {sid:16s} {t:16s} -> {w:16s} {c}")
    print("-- ability errors (species, truth -> world):")
    for (sid, t, w), c in bd.gap_ability_pair.most_common(top):
        print(f"   {sid:16s} {t:16s} -> {w:16s} {c}")
    print("-- PP-used moves absent from world 0 (species, move):")
    for (sid, m), c in bd.gap_moves.most_common(top):
        print(f"   {sid:16s} {m:16s} {c}")


_PP_CACHE: dict[str, int] = {}


def _max_pp(move_id: str) -> int:
    if move_id not in _PP_CACHE:
        try:
            from showdown_bot.engine_state import _max_pp as engine_max_pp

            _PP_CACHE[move_id] = engine_max_pp(move_id)
        except Exception:
            _PP_CACHE[move_id] = 16
    return _PP_CACHE[move_id]


def iter_decisions(files, rng: random.Random, per_file_cap: int | None):
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        except Exception as exc:  # truncated / unreadable file
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        decisions = [r for r in rows if r.get("type") == "decision" and r.get("truth")]
        for r in decisions:
            r["_game"] = str(path)
        if per_file_cap and len(decisions) > per_file_cap:
            decisions = rng.sample(decisions, per_file_cap)
        yield from decisions


def report(counts: dict[str, Counts], stats: Counter, top: int) -> None:
    print(f"decisions with truth: {stats['decisions']}   worlds: {stats['worlds']}   "
          f"avg revealed opp slots: {stats['revealed_slots'] / max(1, stats['decisions']):.2f}   "
          f"active species wrong (illusion/forme): {stats['active_species_wrong']}")
    order = ["opp_active", "opp_bench", "opp_unrevealed", "our_mons", "side:our", "side:opp", "field"]
    for name in order:
        c = counts.get(name)
        if not c or not c.total:
            continue
        print(f"\n== {name} ==")
        print(f"{'field':44s} {'n':>8s} {'mismatch':>9s} {'mean err':>10s}")
        for key in sorted(c.total):
            n, bad = c.total[key], c.bad[key]
            err = c.err_sum.get(key)
            flag = ""
            if name in ("our_mons", "side:our", "field", "side:opp") and bad:
                flag = "   <-- BUG? (public/known field differs)"
            print(f"{key:44s} {n:8d} {100 * bad / n:8.2f}% "
                  f"{(err / n if err is not None else float('nan')):10.4f}{flag}")
        if c.confident_total:
            print("  confident-wrong (all worlds agree, differs from truth):")
            for key in sorted(c.confident_total):
                n, bad = c.confident_total[key], c.confident_wrong[key]
                print(f"    {key:40s} {n:8d} {100 * bad / n:8.2f}%")
        for key, pairs in c.pairs.items():
            print(f"  most common {key} errors (truth -> world):")
            for (tv, wv), count in pairs.most_common(top):
                print(f"    {tv:>20s} -> {wv:<20s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", help="directory of *.jsonl.gz recordings (or one file)")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--per-file", type=int, default=None, help="max decisions sampled per file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--by", nargs="*", default=None, choices=["turn", "reveals", "onfield", "all"],
                        help="breakdowns: turn / reveals (moves seen, item known, ability known) / "
                             "onfield (turns on field); bare --by = all")
    parser.add_argument("--gaps", action="store_true", help="worst species/items/confident-wrong lists (needs --by)")
    args = parser.parse_args()
    root = Path(args.root)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl.gz"))
    rng = random.Random(args.seed)
    if args.max_files:
        files = rng.sample(files, min(args.max_files, len(files)))
    counts: dict[str, Counts] = defaultdict(Counts)
    stats: Counter = Counter()
    bd = Breakdown() if args.by is not None else None
    for decision in iter_decisions(files, rng, args.per_file):
        try:
            analyse_decision(decision, counts, stats, bd)
        except Exception as exc:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                print(f"decision failed: {exc!r}", file=sys.stderr)
    if bd is None:
        report(counts, stats, args.top)
    else:
        names = {"turn": ["turn"], "reveals": ["moves seen", "item known", "ability known"],
                 "onfield": ["turns on field"],
                 "all": ["turn", "moves seen", "item known", "ability known", "turns on field"]}
        dims = [d for k in (args.by or ["all"]) for d in names[k]]
        print(f"decisions with truth: {stats['decisions']}  worlds: {stats['worlds']}")
        print_breakdown(bd, dims)
        if args.gaps:
            print_gaps(bd, args.top)
    if stats["errors"]:
        print(f"\n{stats['errors']} decisions failed to analyse")


if __name__ == "__main__":
    main()
