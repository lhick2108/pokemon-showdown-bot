"""How close are our sampled worlds to the exact "truth" state? (bead 589.27)

Reads self-play recordings (docs/selfplay-data.md): every searched decision stores
the sampled worlds that were searched and, when both players ran in one process, the
exact opposing team ("truth"). For each decision this script lines up the revealed
opposing Pokemon in each world with the truth and reports per-field mismatch rates.

    python scripts/world_accuracy.py data/selfplay/gen0 [--max-files N] [--top 12]

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


def analyse_decision(decision: dict, counts: dict[str, Counts], stats: Counter) -> None:
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
    args = parser.parse_args()
    root = Path(args.root)
    files = [root] if root.is_file() else sorted(root.glob("*.jsonl.gz"))
    rng = random.Random(args.seed)
    if args.max_files:
        files = rng.sample(files, min(args.max_files, len(files)))
    counts: dict[str, Counts] = defaultdict(Counts)
    stats: Counter = Counter()
    for decision in iter_decisions(files, rng, args.per_file):
        try:
            analyse_decision(decision, counts, stats)
        except Exception as exc:
            stats["errors"] += 1
            if stats["errors"] <= 3:
                print(f"decision failed: {exc!r}", file=sys.stderr)
    report(counts, stats, args.top)
    if stats["errors"]:
        print(f"\n{stats['errors']} decisions failed to analyse")


if __name__ == "__main__":
    main()
