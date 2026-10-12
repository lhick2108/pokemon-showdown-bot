"""Foul Play's world accuracy on OUR team, from its own logs (bead 589.27).

    python scripts/fp_world_accuracy.py <fp-log> [<fp-log> ...] --records data/league/l1/records

Foul Play logs the opposing sets it sampled for every world ("Sampling battle k" followed
by one line per revealed Pokemon, active first, then "Sampling n unrevealed pokemon" and
the sampled rest). When Foul Play plays one of our recorded players, our real team is
known exactly: it is side one of that player's first recorded decision. This script
scores Foul Play's revealed-slot samples against that team with the same metrics as
``world_accuracy.py`` (item, item class, ability, move set), so the two bots' opponent
models can be compared on the same footing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import poke_engine as pe

CHOICE = {"choiceband", "choicespecs", "choicescarf"}
ITEM_CLASS = {
    "choiceband": "choice-atk", "choicespecs": "choice-atk", "choicescarf": "choice-scarf",
    "lifeorb": "lifeorb", "assaultvest": "bulk", "eviolite": "bulk", "leftovers": "recovery",
    "blacksludge": "recovery", "sitrusberry": "recovery", "heavydutyboots": "boots",
    "rockyhelmet": "rockyhelmet", "none": "none",
}
LINE = re.compile(r"^INFO\s+\t\s*(\S+)\s+(\S+)\s+(\S+)\s+\[(.*?)\](?:\s+ttype=(\S+))?")
TAG = re.compile(r"Initialized (battle-\S+) against: (\S+)")


def cls(item: str) -> str:
    return ITEM_CLASS.get(item, "other")


def true_team(records: Path, tag: str, user: str):
    path = records / f"{tag}__{user}.jsonl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "decision" and row.get("worlds"):
                state = pe.State.from_string(row["worlds"][0]["state"])
                team = {}
                for mon in state.side_one.pokemon:
                    mid = str(mon.id).lower()
                    if mid in ("none", ""):
                        continue
                    team[mid] = {
                        "item": str(mon.item).lower(),
                        "ability": str(mon.ability).lower(),
                        "tera": str(mon.tera_type).lower(),
                        "moves": {str(m.id).lower() for m in mon.moves if str(m.id).lower() != "none"},
                    }
                return team
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    records = Path(args.records)
    bad: Counter = Counter()
    total: Counter = Counter()
    err: Counter = Counter()
    pairs: Counter = Counter()
    missing = 0
    battles = 0

    def add(group: str, key: str, mismatch: bool, e: float = 0.0) -> None:
        total[(group, key)] += 1
        bad[(group, key)] += int(mismatch)
        err[(group, key)] += e

    for log in args.logs:
        team = None
        in_world = False
        revealed_phase = False
        slot = 0
        with open(log, "r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                m = TAG.search(raw)
                if m:
                    tag, user = m.groups()
                    team = true_team(records, tag, user)
                    battles += 1
                    if team is None:
                        missing += 1
                    in_world = False
                    continue
                if team is None:
                    continue
                if "Sampling battle" in raw:
                    in_world, revealed_phase, slot = True, True, 0
                    continue
                if "unrevealed pokemon" in raw:
                    revealed_phase = False
                    continue
                if not in_world or not revealed_phase:
                    continue
                m = LINE.match(raw)
                if not m:
                    continue
                species, ability, item, moves, ttype = m.groups()
                moves = {x.strip() for x in moves.split(",") if x.strip()}
                real = team.get(species)
                group = "fp_active" if slot == 0 else "fp_bench"
                slot += 1
                if real is None:
                    add(group, "species (not on real team)", True)
                    continue
                add(group, "species (not on real team)", False)
                # "None" = Foul Play saw the item consumed/knocked off; our reference is the
                # ORIGINAL item (first decision), so those are not model errors.
                if item.lower() != "none":
                    add(group, "item", item != real["item"])
                    if item != real["item"]:
                        pairs[(real["item"], item)] += 1
                    add(group, "item class", cls(item) != cls(real["item"]))
                    add(group, "choice item vs not", (item in CHOICE) != (real["item"] in CHOICE))
                    add(group, "scarf vs not", (item == "choicescarf") != (real["item"] == "choicescarf"))
                add(group, "ability", ability != real["ability"])
                add(group, "moveset exact (4/4)", moves != real["moves"])
                add(group, "moves recall (of 4)", False, len(moves & real["moves"]) / max(1, len(real["moves"])))
                if ttype is not None:
                    add(group, "tera_type", ttype != real["tera"])
    print(f"battles in logs: {battles}  (no record found: {missing})")
    for group in ("fp_active", "fp_bench"):
        print(f"\n== {group} ==")
        for (g, key) in sorted(k for k in total if k[0] == group):
            n = total[(g, key)]
            extra = f"  mean {err[(g, key)] / n:.4f}" if err[(g, key)] else ""
            print(f"{key:40s} {n:8d} {100 * bad[(g, key)] / n:7.2f}%{extra}")
    print("\nmost common item errors (real -> sampled):")
    for (real, sampled), count in pairs.most_common(12):
        print(f"  {real:>16s} -> {sampled:<16s} {count}")


if __name__ == "__main__":
    sys.exit(main())
