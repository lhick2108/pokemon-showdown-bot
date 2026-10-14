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
import random
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from showdown_bot import recorder

VALUE_BY_RESULT = {recorder.RESULT_WIN: 1.0, recorder.RESULT_TIE: 0.5, recorder.RESULT_LOSS: 0.0}

KIND_WORLD = "world"
KIND_TRUTH = "truth"
KIND_NODE = "node"  # a sampled interior search node (bead rnm); see ``nodes`` below

NODE_MODES = ("off", "on", "only")

Q_SOURCES = ("decision", "world")
Q_KINDS = ("relative", "absolute")
Q_EVAL_SCALE = 0.0125  # the stock MCTS score is sigmoid(0.0125 * (e_leaf - e_root))


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
    # Search-based value estimate in [0, 1] from this sample's side's view (MCTS root
    # avg_score; side two gets 1 - q). None when no searched world had visits.
    search_value: float | None = None
    # Absolute version: the recorded avg_score is relative to the root evaluation, so
    # q_abs = sigmoid(scale * evaluate(state) + logit(avg_score)) (see absolute_search_value).
    search_value_abs: float | None = None
    stock_eval: float | None = None  # poke_engine.evaluate(state), side one's view, not flipped for side two
    # Position in the recording: ``game_id`` identifies one recording file (one side's
    # view of one battle), ``decision_idx`` the decision's order in it (0-based, counting
    # every stored decision). ``decision_q_abs`` / ``decision_q`` are the decision-level
    # search values (side one's view, never flipped; truth-state absolute q when the
    # decision has a truth state, else the mean over worlds) shared by all samples of the
    # decision; used for bootstrapped (TD / n-step) targets.
    game_id: str = ""
    decision_idx: int = 0
    decision_q_abs: float | None = None
    decision_q: float | None = None
    # kind "node" only: visits of the interior node, joint turns below the searched root, and
    # the evaluator's own absolute value there (``search_value_abs`` is the search's)
    node_visits: int = 0
    node_depth: int = 0
    node_static: float | None = None

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


def _valid_score(world: dict[str, Any]) -> float | None:
    """A world's ``avg_score`` unless it had no visits or no score recorded."""
    score = world.get("avg_score")
    if score is None or not world.get("total_visits"):
        return None
    return float(score)


def absolute_search_value(
    stock_eval: float, avg_score: float, scale: float = Q_EVAL_SCALE
) -> float:
    """Absolute win probability from a root-relative search score.

    ``avg_score`` is ``sigmoid(scale * (e_leaf - e_root))`` averaged, so adding back
    the root's own evaluation gives ``sigmoid(scale * e_root + logit(avg_score))``.
    ``stock_eval`` is ``poke_engine.evaluate`` (side one's view).
    """
    p = min(max(avg_score, 1e-4), 1 - 1e-4)
    z = scale * stock_eval + math.log(p / (1 - p))
    return 1.0 / (1.0 + math.exp(-z))


def decision_search_value(decision: dict[str, Any]) -> float | None:
    """Mean ``avg_score`` over the decision's usable worlds (side one's view), or None."""
    scores = [q for q in map(_valid_score, decision.get("worlds", ())) if q is not None]
    return sum(scores) / len(scores) if scores else None


def iter_samples(
    root: str | Path,
    *,
    sides: Sequence[int] = (1,),
    worlds: bool | str = True,
    truth: bool = True,
    stats: LoadStats | None = None,
    q_source: str = "decision",
    q_eval_scale: float = Q_EVAL_SCALE,
    with_stock_eval: bool = False,
    feature_version: int | None = None,
    max_per_game: int = 0,
    seed: int = 0,
    files: Sequence[Path] | None = None,
    nodes: str = "off",
    node_min_visits: int = 0,
    nodes_per_world: int = 0,
) -> Iterator[Sample]:
    """Yield one ``Sample`` per stored state (every searched world, plus truth).

    ``max_per_game=K > 0`` keeps at most K decisions per game record, chosen uniformly
    without replacement among the decisions that have a state under the ``worlds`` /
    ``truth`` filter, by an RNG seeded with ``seed`` and the record's file name (so it is
    reproducible and independent of the other records). It is applied before any
    parsing or feature extraction, so memory and time drop with it. ``decision_idx``
    stays the decision's index in the full record.

    ``worlds="fallback"`` includes a decision's worlds only when it has no truth
    state (with ``truth=True`` this means "prefer truth").

    ``Sample.search_value`` is the search's value estimate q: the decision's mean
    world ``avg_score`` (``q_source="decision"``, also used for truth states), or
    for world samples that world's own ``avg_score`` (``q_source="world"``, falling
    back to None if that world has no usable score; truth stays the decision mean).

    ``nodes`` (bead rnm): ``off`` ignores the interior nodes some recordings carry
    (``worlds[i]["nodes"]``, written by searches run with ``export_nodes``); ``on`` also
    yields one ``kind="node"`` sample per exported node with at least ``node_min_visits``
    visits; ``only`` yields just those. A node sample's ``search_value_abs`` is the exported
    absolute search value (already anchored to the evaluator the search used, so
    ``q_eval_scale`` does not apply), ``search_value`` equals it, and ``node_visits`` /
    ``node_depth`` / ``node_static`` carry the weighting information for the trainer.
    ``nodes_per_world`` > 0 keeps only that many of a world's eligible nodes (a uniform
    random subset, reproducible from ``seed``, the file name and the decision index).
    """
    if q_source not in Q_SOURCES:
        raise ValueError(f"q_source must be one of {Q_SOURCES}")
    if nodes not in NODE_MODES:
        raise ValueError(f"nodes must be one of {NODE_MODES}")
    import poke_engine

    # None = the engine's default feature version (old engines have no ``version`` argument)
    feat_kwargs = {} if feature_version in (None, 1) else {"version": feature_version}
    stats = stats if stats is not None else LoadStats()
    # ``files`` restricts the walk to those game files (used by the parallel loader)
    for path in (recorder.iter_game_files(root) if files is None else files):
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
        keep: set[int] | None = None
        if max_per_game > 0:
            eligible = [
                i for i, d in enumerate(game["decisions"])
                if (worlds and (worlds != "fallback" or not d.get("truth")) and d.get("worlds"))
                or (truth and d.get("truth"))
            ]
            if len(eligible) > max_per_game:
                keep = set(random.Random(f"{seed}:{path.name}").sample(eligible, max_per_game))
        for decision_idx, decision in enumerate(game["decisions"]):
            if keep is not None and decision_idx not in keep:
                continue
            policy = decision.get("policy") or None
            q_mean = decision_search_value(decision)
            parsed: dict[str, Any] = {}

            def parse(string: str):
                if string not in parsed:
                    try:
                        parsed[string] = poke_engine.State.from_string(string)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except BaseException:  # includes pyo3 PanicException
                        parsed[string] = None
                return parsed[string]

            # absolute q per usable world (needs the parsed world state)
            world_abs: list[float | None] = []
            for w in decision.get("worlds", ()):
                score, st = _valid_score(w), parse(w["state"])
                exported = w.get("search_value")  # absolute root value, exported with the nodes
                world_abs.append(
                    None if score is None or st is None
                    else float(exported) if exported is not None
                    else absolute_search_value(poke_engine.evaluate(st), score, q_eval_scale)
                )
            valid_abs = [v for v in world_abs if v is not None]
            abs_mean = sum(valid_abs) / len(valid_abs) if valid_abs else None
            states: list[tuple[str, str, float | None, float | None]] = []
            if worlds and (worlds != "fallback" or not decision.get("truth")):
                for w, wa in zip(decision.get("worlds", ()), world_abs):
                    own = q_source == "world"
                    states.append((
                        KIND_WORLD, w["state"], _valid_score(w) if own else q_mean,
                        wa if own else abs_mean,
                    ))
            if truth and decision.get("truth"):
                tstring = decision["truth"]["state"]
                tstate = parse(tstring)
                t_abs = (
                    None if tstate is None or q_mean is None
                    else abs_mean if all(w.get("search_value") is not None for w in decision["worlds"])
                    else absolute_search_value(poke_engine.evaluate(tstate), q_mean, q_eval_scale)
                )
                states.append((KIND_TRUTH, tstring, q_mean, t_abs))
            if nodes == "only":
                states = []
            # decision-level absolute q: the truth state's if there is one, else the world mean
            dec_abs = abs_mean
            if truth and decision.get("truth") and states and states[-1][0] == KIND_TRUTH                     and states[-1][3] is not None:
                dec_abs = states[-1][3]
            for kind, state_string, q, q_abs in states:
                state = parse(state_string)
                if state is None:
                    stats.bad_states += 1
                    continue
                stock_eval = float(poke_engine.evaluate(state))  # always filled (absolute q needs it)
                for side in sides:
                    yield Sample(
                        features=poke_engine.state_features(state, side, **feat_kwargs),
                        value=value_target(result, side),
                        policy=policy if side == 1 else None,
                        kind=kind,
                        side=side,
                        battle_tag=tag,
                        turn=decision.get("turn"),
                        state=state_string,
                        search_value=None if q is None else (q if side == 1 else 1.0 - q),
                        stock_eval=stock_eval,
                        game_id=str(path),
                        decision_idx=decision_idx,
                        decision_q_abs=dec_abs,
                        decision_q=q_mean,
                        search_value_abs=(
                            None if q_abs is None else (q_abs if side == 1 else 1.0 - q_abs)
                        ),
                    )
            if nodes != "off":
                for wi, w in enumerate(decision.get("worlds", ())):
                    eligible = [nd for nd in (w.get("nodes") or ()) if nd[1] >= node_min_visits]
                    if nodes_per_world and len(eligible) > nodes_per_world:
                        eligible = random.Random(f"{seed}:{path.name}:{decision_idx}:{wi}").sample(
                            eligible, nodes_per_world)
                    for node in eligible:
                        n_state, n_visits, n_value, n_static, n_depth = node
                        n_parsed = parse(n_state)
                        if n_parsed is None:
                            stats.bad_states += 1
                            continue
                        for side in sides:
                            yield Sample(
                                features=poke_engine.state_features(n_parsed, side, **feat_kwargs),
                                value=value_target(result, side),
                                policy=None,
                                kind=KIND_NODE,
                                side=side,
                                battle_tag=tag,
                                turn=decision.get("turn"),
                                state=n_state,
                                search_value=n_value if side == 1 else 1.0 - n_value,
                                stock_eval=float(poke_engine.evaluate(n_parsed)),
                                game_id=str(path),
                                decision_idx=decision_idx,
                                search_value_abs=n_value if side == 1 else 1.0 - n_value,
                                node_visits=int(n_visits),
                                node_depth=int(n_depth),
                                node_static=(None if n_static is None else
                                             (n_static if side == 1 else 1.0 - n_static)),
                            )


def _fill_gaps(q: np.ndarray, z: float) -> tuple[np.ndarray, int]:
    """Replace NaN entries of a decision chain by the next available q, or ``z`` if none."""
    out = q.astype(np.float64).copy()
    nxt = z
    n_gap = 0
    for t in range(len(out) - 1, -1, -1):
        if np.isnan(out[t]):
            out[t] = nxt
            n_gap += 1
        else:
            nxt = out[t]
    return out, n_gap


def td_lambda_returns(q: Sequence[float], z: float, lam: float) -> np.ndarray:
    """lambda-returns of one recording's decision chain (backward recursion).

    ``q[t]`` is decision t's search value (NaN = missing), t = 0..T-1, and the step
    after the last decision is the outcome ``z``: ``G[T-1] = z``,
    ``G[t] = (1-lam) * q[t+1] + lam * G[t+1]``, i.e. the usual
    ``(1-lam) * sum_n lam^(n-1) q[t+n] + lam^(T-t-1) z``. ``lam=1`` gives ``z``,
    ``lam=0`` gives ``q[t+1]`` (``z`` for the last decision). Missing q is a gap:
    it is replaced by the next available q (or z) before the recursion.
    """
    qf, _ = _fill_gaps(np.asarray(q, dtype=np.float64), z)
    g = np.empty_like(qf)
    if len(qf) == 0:
        return g
    g[-1] = z
    for t in range(len(qf) - 2, -1, -1):
        g[t] = (1.0 - lam) * qf[t + 1] + lam * g[t + 1]
    return g


def nstep_returns(q: Sequence[float], z: float, n: int) -> np.ndarray:
    """``G[t] = q[t+n]`` if ``t+n < T`` else ``z`` (gaps filled as in ``td_lambda_returns``)."""
    qf, _ = _fill_gaps(np.asarray(q, dtype=np.float64), z)
    T = len(qf)
    idx = np.arange(T) + n
    return np.where(idx < T, qf[np.minimum(idx, max(T - 1, 0))] if T else qf, z)


def bootstrapped_targets(
    game_ids: Sequence[str],
    decision_idx: Sequence[int],
    decision_q: Sequence[float],
    z: Sequence[float],
    mode: str,
    lam: float = 0.7,
    n: int = 1,
    mu: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Per-sample TD / n-step targets looking ahead within each recording.

    Samples are grouped by ``game_id`` and ordered by ``decision_idx``; all samples of
    a decision (world and truth) share its ``decision_q`` (NaN = missing) and get the
    same target. ``mode`` is ``td`` (lambda-return, ``lam``) or ``nstep`` (``n``);
    ``mu`` blends the decision's own q: ``(1-mu) * G + mu * q_t`` (own q missing -> G).
    ``z`` is the outcome per sample (constant within a recording). Returns
    ``(targets float32, stats)`` with ``stats['gaps']`` missing-q decisions and
    ``stats['decisions']`` total decisions.
    """
    if mode not in ("td", "nstep"):
        raise ValueError(f"unknown bootstrap mode {mode!r}")
    gid = np.asarray(game_ids)
    didx = np.asarray(decision_idx, dtype=np.int64)
    q = np.asarray(decision_q, dtype=np.float64)
    zz = np.asarray(z, dtype=np.float64)
    out = np.empty(len(gid), dtype=np.float32)
    stats = {"gaps": 0, "decisions": 0}
    order = np.lexsort((didx, gid))
    bounds = np.flatnonzero(np.r_[True, gid[order][1:] != gid[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = order[a:b]
        decs, first, inv = np.unique(didx[rows], return_index=True, return_inverse=True)
        qd, zg = q[rows][first], float(zz[rows][0])
        stats["decisions"] += len(decs)
        stats["gaps"] += int(np.isnan(qd).sum())
        g = td_lambda_returns(qd, zg, lam) if mode == "td" else nstep_returns(qd, zg, n)
        if mu:
            g = np.where(np.isnan(qd), g, (1 - mu) * g + mu * np.nan_to_num(qd))
        out[rows] = g[inv.ravel()]
    return out, stats


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

