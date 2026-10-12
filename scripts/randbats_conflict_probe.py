"""Replay per-side protocol logs (scripts/gen_protocol.js) through OpponentModel and
report every "observations conflict with all randbats roles" case with the evidence.

Usage: python scripts/randbats_conflict_probe.py <proto.jsonl> [cache_dir [line-regex]]
"""
import json
import re
import logging
import sys
from collections import Counter

from poke_env.battle.battle import Battle

from showdown_bot.sets import OpponentModel, RandbatsData

path = sys.argv[1]
only = sys.argv[3] if len(sys.argv) > 3 else "."
data = RandbatsData.load(cache_dir=sys.argv[2] if len(sys.argv) > 2 else ".cache/randbats")
logging.disable(logging.CRITICAL)
by_species: Counter = Counter()
errors: Counter = Counter()
species_seen: Counter = Counter()
examples = []
for line in open(path):
    if not re.search(only, line):
        continue
    rec = json.loads(line)
    for role in ("p1", "p2"):
        battle = Battle(f"{rec['tag']}-{role}", username=role, logger=None, gen=9)
        battle.player_role = role
        model = OpponentModel(data)
        seen = set()
        # Mirror OpponentModelPlayerMixin: observe a whole message batch (one
        # turn of protocol) first, then let poke-env parse it, then update().
        batches, cur = [], []
        for raw in rec[role]:
            if not raw.startswith("|") or raw.startswith(("|request", "|t:", "|upkeep", "|win|", "|tie")) or raw == "|":
                continue
            cur.append(raw)
            if raw.startswith("|turn|") or raw.startswith("|win|") or raw == "|upkeep" and False:
                batches.append(cur)
                cur = []
        if cur:
            batches.append(cur)
        for batch in batches:
            events = [raw.split("|") for raw in batch]
            try:
                for ev in events:
                    model.observe(battle.battle_tag, ev)
                for ev in events:
                    battle.parse_message(ev)
                trackers = model.update(battle)
            except Exception as exc:  # noqa: BLE001
                errors[type(exc).__name__ + str(exc)[:80]] += 1
                continue
            for ident, tr in trackers.items():
                species_seen[ident] = 1
                if tr.inconsistent and ident not in seen:
                    seen.add(ident)
                    by_species[tr.species] += 1
                    examples.append((rec["tag"], role, ident, tr.species, list(tr._observations)))
print(by_species, errors.most_common(5), sum(species_seen.values()))
for e in examples[:40]:
    print(e)
