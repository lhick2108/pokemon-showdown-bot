"""Stream recorded self-play games as (features, value_target, policy_target).

States are stored as engine strings (``recorder`` schema), so features are
recomputed here with the *current* ``poke_engine.state_features``; a
``FEATURE_VERSION`` change needs no re-recording.

``value_target`` is the game outcome from the viewpoint of the side the features
describe: 1.0 win, 0.5 tie, 0.0 loss. Our player is engine side one in every
stored state; ``sides=(1, 2)`` additionally yields side two's view of the same
state (features from side 2, value flipped, no policy as it was not searched).
``policy_target`` is the decision's aggregated ``{choice: probability}`` (``None``
for side two). CLI: ``python -m showdown_bot.dataset <run-dir-or-file>``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from showdown_bot import recorder

VALUE_BY_RESULT = {recorder.RESULT_WIN: 1.0, recorder.RESULT_TIE: 0.5, recorder.RESULT_LOSS: 0.0}

KIND_WORLD = "world"
KIND_TRUTH = "truth"


@dataclass
class Sample:
    features: list[float]
    value: float
    policy: dict[str, float] | None
    kind: str  # "world" (sampled opponent) or "truth" (exact state)
    side: int  # 1 or 2: whose perspective the features/value use
    battle_tag: str
    turn: int | None
    state: str | None = None  # the engine State string the features were computed from

    def as_tuple(self) -> tuple[list[float], float, dict[str, float] | None]:
        return self.features, self.value, self.policy


@dataclass
class LoadStats:
    games: int = 0
    skipped_games: int = 0  # no result record (unfinished) or unreadable
    bad_states: int = 0  # State strings the current engine could not parse
    results: Counter = field(default_factory=Counter)


def value_target(result: str, side: int = 1) -> float:
    """Outcome for ``side`` given the result from our (side one's) perspective."""
    value = VALUE_BY_RESULT[result]
    return value if side == 1 else 1.0 - value


def iter_samples(
    root: str | Path,
    *,
    sides: Sequence[int] = (1,),
    worlds: bool = True,
    truth: bool = True,
    stats: LoadStats | None = None,
) -> Iterator[Sample]:
    """Yield one ``Sample`` per stored state (every searched world, plus truth)."""
    import poke_engine

    stats = stats if stats is not None else LoadStats()
    for path in recorder.iter_game_files(root):
        try:
            game = recorder.read_game(path)
        except (OSError, ValueError, EOFError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            stats.skipped_games += 1
            continue
        result = (game["result"] or {}).get("result")
        if result not in VALUE_BY_RESULT:
            stats.skipped_games += 1
            continue
        stats.games += 1
        stats.results[result] += 1
        tag = game["header"].get("battle_tag", "")
        for decision in game["decisions"]:
            policy = decision.get("policy") or None
            states: list[tuple[str, str]] = []
            if worlds:
                states += [(KIND_WORLD, w["state"]) for w in decision.get("worlds", ())]
            if truth and decision.get("truth"):
                states.append((KIND_TRUTH, decision["truth"]["state"]))
            for kind, state_string in states:
                try:
                    state = poke_engine.State.from_string(state_string)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:  # includes pyo3 PanicException (not an Exception)
                    stats.bad_states += 1
                    continue
                for side in sides:
                    yield Sample(
                        features=poke_engine.state_features(state, side),
                        value=value_target(result, side),
                        policy=policy if side == 1 else None,
                        kind=kind,
                        side=side,
                        battle_tag=tag,
                        turn=decision.get("turn"),
                        state=state_string,
                    )


def iter_triples(
    root: str | Path, **kwargs: Any
) -> Iterator[tuple[list[float], float, dict[str, float] | None]]:
    """``iter_samples`` reduced to plain ``(features, value, policy)`` tuples."""
    for sample in iter_samples(root, **kwargs):
        yield sample.as_tuple()


def _entropy(policy: dict[str, float]) -> float:
    return -sum(p * math.log(p) for p in policy.values() if p > 0)


def collect_statistics(root: str | Path) -> dict[str, Any]:
    """Counts over a dataset directory (reads records only, no engine needed)."""
    out: dict[str, Any] = {
        "games": 0, "unfinished_or_unreadable": 0, "results": Counter(),
        "decisions": 0, "worlds": 0, "truth_states": 0, "bytes": 0,
        "turns_total": 0, "policy_entropy_sum": 0.0, "policy_top_sum": 0.0,
        "feature_versions": Counter(), "schema_versions": Counter(),
        "decisions_per_game_max": 0, "battle_tags": set(),
    }
    for path in recorder.iter_game_files(root):
        out["bytes"] += path.stat().st_size
        try:
            game = recorder.read_game(path)
        except (OSError, ValueError, EOFError):
            out["unfinished_or_unreadable"] += 1
            continue
        header = game["header"]
        out["feature_versions"][header.get("feature_version")] += 1
        out["schema_versions"][header.get("schema_version")] += 1
        result = (game["result"] or {}).get("result")
        if result is None:
            out["unfinished_or_unreadable"] += 1
            continue
        out["games"] += 1
        out["battle_tags"].add(header.get("battle_tag"))
        out["results"][result] += 1
        out["turns_total"] += (game["result"] or {}).get("turns") or 0
        out["decisions_per_game_max"] = max(out["decisions_per_game_max"], len(game["decisions"]))
        for decision in game["decisions"]:
            out["decisions"] += 1
            out["worlds"] += len(decision.get("worlds", ()))
            out["truth_states"] += 1 if decision.get("truth") else 0
            policy = decision.get("policy") or {}
            if policy:
                out["policy_entropy_sum"] += _entropy(policy)
                out["policy_top_sum"] += max(policy.values())
    return out


def format_statistics(stats: dict[str, Any]) -> str:
    games, decisions = stats["games"], stats["decisions"]
    results = stats["results"]
    lines = [
        f"battles:             {len(stats['battle_tags'])}",
        f"game files (finished): {games}  (unfinished/unreadable: {stats['unfinished_or_unreadable']})",
        "results (our side):   "
        + ", ".join(f"{k}={results.get(k, 0)}" for k in recorder.RESULTS),
        f"decisions:            {decisions}",
        f"worlds (sampled):     {stats['worlds']}",
        f"truth states:         {stats['truth_states']}",
        f"positions (worlds+truth): {stats['worlds'] + stats['truth_states']}",
        f"disk:                 {stats['bytes'] / 1e6:.2f} MB"
        + (
            f" ({stats['bytes'] / 1e6 / len(stats['battle_tags']) * 1000:.1f} MB / 1000 battles)"
            if stats["battle_tags"] else ""
        ),
        f"schema versions:      {dict(stats['schema_versions'])}",
        f"feature versions:     {dict(stats['feature_versions'])}",
    ]
    if games:
        lines.append(
            f"mean decisions/game:  {decisions / games:.1f}  "
            f"(max {stats['decisions_per_game_max']}); mean turns {stats['turns_total'] / games:.1f}"
        )
    if decisions:
        lines.append(
            f"mean policy entropy:  {stats['policy_entropy_sum'] / decisions:.3f} nats; "
            f"mean top-move mass {stats['policy_top_sum'] / decisions:.3f}"
        )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="run directory (or a single .jsonl.gz game file)")
    parser.add_argument(
        "--check-features",
        action="store_true",
        help="also rebuild every State and compute features (reports unparsable states)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = Path(args.path)
    if not root.exists():
        print(f"no such path: {root}", file=sys.stderr)
        return 2
    print(format_statistics(collect_statistics(root)))
    if args.check_features:
        load = LoadStats()
        total, truth_n, value_sum, length = 0, 0, 0.0, 0
        for sample in iter_samples(root, stats=load):
            total += 1
            truth_n += sample.kind == KIND_TRUTH
            value_sum += sample.value
            length = len(sample.features)
        print(
            f"feature check: {total} samples ({truth_n} truth), vector length {length}, "
            f"mean value {value_sum / total if total else float('nan'):.3f}, "
            f"unparsable states {load.bad_states}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
