"""Endgame mode: exact iterative-deepening expectiminimax instead of MCTS for small endgames.

When few Pokemon are left alive the tree is narrow enough that a depth-first search with
alpha-beta (``poke_engine.iterative_deepening_expectiminimax``) reaches depth 6-8 within a
couple of seconds, which beats sampling-based MCTS on forced-line tactics. This module holds
the pure parts (trigger, time plan, aggregation of per-world matrices); the search call itself
is ``players._worker.endgame_world`` and the wiring is in ``players.search``.

Engine semantics (engine/src/search.rs), which the aggregation relies on:

* The root matrix is ``len(side_one) x len(side_two)``, row-major, from side one's view
  (our options are the rows), in the STOCK hand-written evaluation units (about +-300 for a
  decided position). A terminal win/loss at ``d`` plies left scores ``+-100*d + evaluate``,
  so faster wins/slower losses are preferred.
* The search is stock-eval only: it calls ``evaluate(state)`` and has NO learned-evaluator
  hook (``Evaluator`` / leaf scale apply to MCTS only). The ``evaluator`` setting is therefore
  ignored in endgame decisions; the stock eval is absolute (no root offset).
* The root is alpha-beta pruned with a pure-strategy (maximin) bound. After the first row,
  once a cell of a row is ``<= alpha`` (the best row worst-case so far) the rest of that
  row is skipped and left ``NaN``. So a row with ``NaN`` is known to be no better than the
  best row; its min over the non-NaN cells is an UPPER bound of its true worst case, while
  the rows that were fully evaluated (including the best one) are exact. The rows are
  re-ordered between iterations (best first), so the returned option lists are the engine's
  order, not the legal-move order: always key by name.
* Result is the last COMPLETED depth (``depth_searched``), never a half-finished one.

Eval units are mapped to win probabilities with the stock MCTS leaf shape,
``p = sigmoid(0.0125 * v)`` (the factor used by the engine's relative leaf score), so worlds
are averaged on a common bounded scale.

Aggregation
-----------
``safest``: per world, each option's worst case over the opponent's evaluated columns
(= the engine's ``get_safest_move`` value, exact for the best row, an upper bound for pruned
rows); converted to win probability and averaged over the worlds that offer the option; the
argmax is played. Because the bounds can only overstate a pruned row, a pruned row never
wrongly beats the exact best row of any one world.

``nash``: per world the matrix game is solved with ``nash.solve_matrix_game`` on the
win-probability matrix, then the strategies are averaged over worlds with equal weight
(Nash is invariant to a per-matrix affine shift, but not across worlds, hence the common
sigmoid scale) and the move is sampled. Caveat: the engine's root pruning is maximin-based,
so rows that are only useful inside a MIXED strategy can be pruned; their NaN cells are
filled with the row's smallest evaluated cell (conservative for us). Nash therefore
approximates the true equilibrium and favours rows with solid worst cases; prefer ``safest``
unless the position is a pure guessing game that was fully evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from showdown_bot import nash as nash_module

OFF = "off"
SAFEST = "safest"
NASH = "nash"
MODES = (OFF, SAFEST, NASH)

EVAL_SCALE = 0.0125  # stock MCTS leaf shape: win prob = sigmoid(EVAL_SCALE * evaluation)
TEAM_SIZE = 6
MIN_USEFUL_MS = 50  # below this per-world budget the endgame search is not worth running
DEFAULT_MARGIN_MS = 1000  # reserve for state building, IPC and the first (uninterruptible) depths


def win_prob(value: float) -> float:
    """Eval units -> win probability (stable for the +-1000s a terminal score reaches)."""
    z = max(-60.0, min(60.0, EVAL_SCALE * float(value)))
    return 1.0 / (1.0 + math.exp(-z))


# ---------------------------------------------------------------------------------------------
# Trigger and time plan
# ---------------------------------------------------------------------------------------------


def alive_counts(battle: Any) -> tuple[int, int]:
    """``(ours alive, opponent alive)``; unseen opposing Pokemon count as alive."""
    ours = sum(1 for mon in battle.team.values() if not mon.fainted)
    seen = list(battle.opponent_team.values())
    size = getattr(battle, "opponent_team_size", None) or TEAM_SIZE
    unseen = max(0, int(size) - len(seen))
    return ours, sum(1 for mon in seen if not mon.fainted) + unseen


def triggered(cfg: Any, ours: int, theirs: int) -> bool:
    """Is this an endgame: mode on, total alive <= ``endgame_max_alive`` (and each side <=
    ``endgame_max_side`` when that is set)."""
    if cfg.endgame == OFF:
        return False
    if ours + theirs > cfg.endgame_max_alive:
        return False
    side_cap = getattr(cfg, "endgame_max_side", 0)
    if side_cap and (ours > side_cap or theirs > side_cap):
        return False
    return True


def plan_time(
    cfg: Any, pool_size: int, cap_ms: float | None = None, margin_ms: int = DEFAULT_MARGIN_MS
) -> tuple[int, int] | None:
    """``(worlds, per-world ms)`` for an endgame decision, or None if no budget is left.

    Worlds run concurrently on ``pool_size`` processes in ``waves = ceil(worlds/pool_size)``
    waves, each ``ms`` long, so the decision takes about ``waves * ms + margin``; ``ms`` is
    clamped so that stays under ``cap_ms`` (default ``cfg.endgame_cap_ms``). Running more
    worlds than the pool can run at once is allowed but clamped the same way; worlds beyond
    the pool are dropped first when even the minimum slice would not fit.
    """
    cap = float(cap_ms if cap_ms is not None else cfg.endgame_cap_ms)
    pool = max(1, int(pool_size))
    worlds = max(1, int(cfg.endgame_worlds))
    budget = cap - margin_ms
    while worlds > 1 and budget / math.ceil(worlds / pool) < MIN_USEFUL_MS:
        worlds -= 1
    waves = math.ceil(worlds / pool)
    ms = int(min(cfg.endgame_time_ms, budget / waves))
    if ms < MIN_USEFUL_MS:
        return None
    return worlds, ms


# ---------------------------------------------------------------------------------------------
# Per-world matrices
# ---------------------------------------------------------------------------------------------


@dataclass
class WorldMatrix:
    """One world's expectiminimax result."""

    ours: list[str]  # our options (rows), engine choice strings
    theirs: list[str]  # opponent options (columns)
    matrix: np.ndarray  # (len(ours), len(theirs)) eval units, NaN = pruned / unexplored
    depth: int

    def row_worst(self) -> dict[str, float]:
        """Each evaluated option's worst case (eval units); all-NaN rows are missing."""
        out: dict[str, float] = {}
        for name, row in zip(self.ours, self.matrix):
            known = row[~np.isnan(row)]
            if known.size:
                value = float(known.min())
                out[name] = min(value, out.get(name, value))
        return out


def parse_world(result: Sequence[Any] | None) -> WorldMatrix | None:
    """``(s1, s2, flat matrix, depth)`` from the worker -> ``WorldMatrix`` (None if degenerate)."""
    if not result or len(result) < 4:
        return None
    ours, theirs, flat, depth = list(result[0]), list(result[1]), list(result[2]), int(result[3])
    if not ours or not theirs or len(flat) != len(ours) * len(theirs):
        return None
    matrix = np.asarray(flat, dtype=float).reshape(len(ours), len(theirs))
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    if np.isnan(matrix).all():
        return None
    return WorldMatrix(ours, theirs, matrix, depth)


def safest_values(worlds: Sequence[WorldMatrix]) -> dict[str, float]:
    """Mean win probability of each option's worst case over the worlds that evaluated it."""
    total: dict[str, float] = {}
    count: dict[str, int] = {}
    for world in worlds:
        for name, value in world.row_worst().items():
            total[name] = total.get(name, 0.0) + win_prob(value)
            count[name] = count.get(name, 0) + 1
    return {name: total[name] / count[name] for name in total}


def aggregate_safest(worlds: Sequence[WorldMatrix]) -> tuple[dict[str, float], dict[str, Any]]:
    """``(policy, info)``: policy is the mean worst-case win probability per option (argmax it)."""
    values = safest_values(worlds)
    return values, {"values": values}


def filled_matrix(world: WorldMatrix) -> np.ndarray | None:
    """Win-probability matrix with pruned cells set to the row's smallest evaluated cell;
    all-NaN rows/columns are dropped. Returns None when nothing is left."""
    probs = np.vectorize(win_prob)(np.nan_to_num(world.matrix, nan=0.0))
    known = ~np.isnan(world.matrix)
    probs = np.where(known, probs, np.nan)
    rows = known.any(axis=1)
    if not rows.any():
        return None
    probs = probs[rows]
    row_min = np.nanmin(probs, axis=1, keepdims=True)
    probs = np.where(np.isnan(probs), row_min, probs)
    return probs


def nash_world(world: WorldMatrix) -> tuple[dict[str, float], float, float] | None:
    """Our equilibrium strategy (by option) on one world's win-probability matrix."""
    matrix = filled_matrix(world)
    if matrix is None:
        return None
    names = [n for n, row in zip(world.ours, world.matrix) if not np.isnan(row).all()]
    solution = nash_module.solve_matrix_game(matrix)
    strategy: dict[str, float] = {}
    for name, p in zip(names, solution.row):
        strategy[name] = strategy.get(name, 0.0) + float(p)
    return strategy, solution.value, solution.exploitability


def aggregate_nash(
    worlds: Sequence[WorldMatrix], floor: float = 0.0
) -> tuple[dict[str, float], dict[str, Any]]:
    """Equal-weight average of the worlds' equilibrium strategies (+ per-world records)."""
    weighted: list[tuple[float, dict[str, float]]] = []
    records: list[dict[str, Any] | None] = []
    for world in worlds:
        solved = nash_world(world)
        if solved is None:
            records.append(None)
            continue
        strategy, value, exploitability = solved
        weighted.append((1.0, strategy))
        records.append({"strategy": strategy, "value": value, "exploitability": exploitability})
    policy = nash_module.aggregate_strategies(weighted, floor)
    return policy, {"nash": records}


def aggregate(
    mode: str, worlds: Sequence[WorldMatrix], floor: float = 0.0
) -> tuple[dict[str, float], dict[str, Any]]:
    """Dispatch on the endgame mode; an empty policy means "fall back to MCTS"."""
    if not worlds:
        return {}, {}
    if mode == NASH:
        return aggregate_nash(worlds, floor)
    return aggregate_safest(worlds)


def world_record(state: str, world: WorldMatrix, nash: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decision-record entry for one endgame world (no MCTS statistics: ``total_visits`` is 0
    and ``moves`` empty, so dataset readers skip it as a value/policy source)."""
    record: dict[str, Any] = {
        "state": state,
        "total_visits": 0,
        "avg_score": None,
        "moves": [],
        "endgame": {
            "depth_searched": world.depth,
            "options": world.row_worst(),  # worst case per option, eval units
            "ours": world.ours,
            "theirs": world.theirs,
            "matrix": [None if math.isnan(x) else round(float(x), 2) for x in world.matrix.ravel()],
        },
    }
    if nash is not None:
        record["nash"] = nash
    return record
