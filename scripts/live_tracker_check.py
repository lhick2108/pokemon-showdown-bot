"""Smoke test: run OpponentModel inside live local battles and sample every turn."""

import asyncio
import logging
import random
import sys

from poke_env.player import RandomPlayer
from poke_env.ps_client import LocalhostServerConfiguration

from collections import Counter

from showdown_bot.sets import OpponentModel, OpponentModelPlayerMixin, RandbatsData, to_id

conflicts: Counter = Counter()
provisional: Counter = Counter()

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30

warnings: list[str] = []


class Capture(logging.Handler):
    def emit(self, record):
        if record.name.startswith("showdown_bot") and record.levelno >= logging.WARNING:
            warnings.append(record.getMessage())


logging.getLogger().addHandler(Capture())
data = RandbatsData.load()
rng = random.Random(0)
stats = {"turns": 0, "samples": 0, "errors": 0, "trackers": 0, "with_moves": 0, "poke_env_moves": 0, "role_fallbacks": 0}


class TrackedRandom(OpponentModelPlayerMixin, RandomPlayer):
    def choose_move(self, battle):
        stats["turns"] += 1
        try:
            trackers = self.opponent_model.update(battle)
            for tracker in trackers.values():
                tracker.sample(rng)
                stats["samples"] += 1
                if not tracker._matching_full_sets():
                    stats["role_fallbacks"] += 1
            for tracker in trackers.values():
                for kind, value in tracker.provisional_observations:
                    provisional[(tracker.species, kind, to_id(value))] += 1
                if tracker.inconsistent:
                    for kind, option in tracker._observations:
                        if (kind, option) in tracker._provisional_ids:
                            continue
                        legal = any(
                            option in {to_id(o) for o in role.options(kind)}
                            for role in tracker._roles.values()
                        )
                        if not legal:
                            conflicts[(tracker.species, kind, option)] += 1
            if battle.turn >= 15:
                stats["trackers"] += len(trackers)
                stats["with_moves"] += sum(1 for t in trackers.values() if t.revealed_moves)
                stats["poke_env_moves"] += sum(1 for m in battle.opponent_team.values() if m.moves)
        except Exception as exc:  # noqa: BLE001 - smoke test reports everything
            stats["errors"] += 1
            print("SAMPLE ERROR:", repr(exc))
        return super().choose_move(battle)


async def main():
    opts = dict(battle_format="gen9randombattle", server_configuration=LocalhostServerConfiguration,
                max_concurrent_battles=4, log_level=logging.ERROR)
    p1 = TrackedRandom(opponent_model=OpponentModel(data), **opts)
    p2 = RandomPlayer(**opts)
    await asyncio.wait_for(p1.battle_against(p2, n_battles=N), timeout=900)
    print(f"battles={p1.n_finished_battles} {stats} warnings={len(warnings)}")
    for w in sorted(set(warnings))[:15]:
        print("WARN:", w)
    rate = stats["role_fallbacks"] / max(stats["samples"], 1)
    print(f"role-sampler fallbacks: {stats['role_fallbacks']}/{stats['samples']} ({rate:.2%})")
    print(f"conflicting observations (species, kind, option) -> turns seen: {len(conflicts)} distinct")
    for (species, kind, option), count in conflicts.most_common(40):
        print(f"CONFLICT: {species} {kind}={option} x{count}")
    print(f"provisional (Zoroark-explainable) observations: {len(provisional)} distinct")
    for (species, kind, option), count in provisional.most_common(40):
        print(f"PROVISIONAL: {species} {kind}={option} x{count}")


asyncio.run(main())
