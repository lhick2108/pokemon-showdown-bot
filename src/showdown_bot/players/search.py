"""SearchPlayer: determinized poke-engine MCTS for Gen 9 random battles.

Each turn the player

1. updates the opponent model and samples K "worlds" (one full opposing team
   per world: revealed Pokemon get full sets from ``SetTracker.sample``,
   unrevealed slots get species by frequency under the randbats team rules);
2. converts each world to a ``poke_engine.State`` (``engine_state.build_state``);
3. runs MCTS on every world in a process pool;
4. sums each of our moves' visit share over the worlds (every sampled world has
   weight 1/K: worlds are i.i.d. draws from the opponent-set distribution, so
   the plain average is the probability-weighted one); and
5. selects a move: ``mode="mixed"`` (default, Foul Play's rule) samples
   proportionally among moves whose total is at least 75% of the best;
   ``mode="argmax"`` takes the best move; ``mode="nash"`` solves each world's root
   joint-action matrix game and samples from the visit-weighted average of our
   equilibrium strategies (see docs/selection.md).

Defaults equal Foul Play's: 100 ms per world, ``parallelism`` pool processes,
K = 2 x parallelism worlds (4 x parallelism worlds at half the time when at most
three opposing Pokemon have been seen and the active one has shown no move),
1 engine thread per world.

Configuration: constructor ``config=SearchConfig(...)`` or, for players built
without arguments (``bench`` instantiates ``PLAYER_REGISTRY`` classes with
only poke-env options), these environment variables:

    SHOWDOWN_SEARCH_TIME_MS      per-world search time (default 100)
    SHOWDOWN_SEARCH_PARALLELISM  pool processes (default 1)
    SHOWDOWN_SEARCH_WORLDS       fixed world count (default 2 x parallelism, 4x early)
    SHOWDOWN_SEARCH_THREADS      engine threads per world (default 1)
    SHOWDOWN_SEARCH_MODE         "mixed" (default), "argmax" or "nash" (needs threads == 1)
    SHOWDOWN_SEARCH_NASH_PSEUDOCOUNT
                                 shrinkage pseudo-count c >= 0 for the joint matrix (default 5)
    SHOWDOWN_SEARCH_NASH_FLOOR   drop Nash actions below this share, in [0, 1) (default 0.03)
    SHOWDOWN_SEARCH_TENDENCY     "on" turns the opponent-tendency model on (default off, 589.40):
                                 solves the root matrix game like mode "nash" and exploits the
                                 opponent's observed action classes within a safety margin
    SHOWDOWN_SEARCH_TENDENCY_KAPPA
                                 prior pseudo-observations of the Nash class mix (default 4)
    SHOWDOWN_SEARCH_TENDENCY_DELTA
                                 max worst-case win-probability loss vs Nash (default 0.05)
    SHOWDOWN_SEARCH_SELECTION    engine tree-policy: "ucb1" (default), "exp3" or "rm";
                                 exp3/rm need threads == 1 (see docs/selection.md)
    SHOWDOWN_SEARCH_SELECTION_GAMMA
                                 exploration rate in (0, 1] for exp3/rm (default 0.1)
    SHOWDOWN_SEARCH_EVALUATOR    path to a learned .pev leaf evaluator (default: the
                                 stock hand-written evaluation); see docs/training.md.
                                 Each pool worker loads it once and reuses it.
    SHOWDOWN_SEARCH_EVALUATOR_SCALE
                                 leaf scale k > 0 for the learned evaluator: leaf score
                                 sigmoid(k * (z_leaf - z_root)) (default 1.0; see the
                                 "Leaf scale" section of docs/training.md)
    SHOWDOWN_BELIEF_REWEIGHT     "1" turns on soft Bayesian belief reweighting of the opponent's
                                 candidate sets (default off; see showdown_bot.belief)
    SHOWDOWN_BELIEF_TAU          tempering exponent for action evidence (default 0.5)
    SHOWDOWN_BELIEF_FLOOR        no set drops below this share of the likeliest (default 0.05)
    SHOWDOWN_RECORD_DIR          record every searched decision here (default: off;
                                 see showdown_bot.recorder / docs/selfplay-data.md)

The player never crashes a battle: any internal error falls back to a random
legal order and increments ``SearchStats.fallbacks`` (reasons are counted too).
A summary is printed to stderr at interpreter exit.
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import functools
import logging
import math
import os
import random
import secrets
import sys
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown_bot import belief as belief_module
from showdown_bot import decision_quality as dq
from showdown_bot import engine_state, nash as nash_module, recorder as recorder_module
from showdown_bot import opponent_tendency as tendency_module
from showdown_bot.players import _worker
from showdown_bot.recorder import GameRecorder
from showdown_bot.sets import OpponentModel, OpponentModelPlayerMixin, RandbatsData

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "randbats"

MIXED = "mixed"
ARGMAX = "argmax"
NASH = "nash"
MODES = (MIXED, ARGMAX, NASH)
NASH_PSEUDOCOUNT_ENV = "SHOWDOWN_SEARCH_NASH_PSEUDOCOUNT"
NASH_FLOOR_ENV = "SHOWDOWN_SEARCH_NASH_FLOOR"
DEFAULT_NASH_FLOOR = 0.03
MIXED_THRESHOLD = 0.75
EVALUATOR_ENV = "SHOWDOWN_SEARCH_EVALUATOR"
EVALUATOR_SCALE_ENV = "SHOWDOWN_SEARCH_EVALUATOR_SCALE"
SELECTION_ENV = "SHOWDOWN_SEARCH_SELECTION"
SELECTION_GAMMA_ENV = "SHOWDOWN_SEARCH_SELECTION_GAMMA"
SELECTIONS = ("ucb1", "exp3", "rm")


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, raw)
        return default
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", name, raw)
        return default
    return value


def _env_scale(name: str, default: float = 1.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning("Ignoring non-positive or non-finite %s=%r", name, raw)
        return default
    return value


def _env_gamma(name: str, default: float = 0.1) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    if not (math.isfinite(value) and 0 < value <= 1):
        logger.warning("Ignoring %s=%r (need 0 < gamma <= 1)", name, raw)
        return default
    return value


def _env_float(name: str, default: float, low: float, high: float) -> float:
    """A float in ``[low, high)`` from the environment (``default`` if unset or bad)."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r", name, raw)
        return default
    if not (math.isfinite(value) and low <= value < high):
        logger.warning("Ignoring %s=%r (need %g <= value < %g)", name, raw, low, high)
        return default
    return value


@dataclass(frozen=True)
class SearchConfig:
    """Compute and selection settings (defaults equal Foul Play's)."""

    search_time_ms: int = 100
    parallelism: int = 1
    worlds: int | None = None  # None: 2 x parallelism (4 x early, at half the time)
    threads: int = 1
    mode: str = MIXED
    mixed_threshold: float = MIXED_THRESHOLD
    evaluator: str | None = None  # path of a .pev file; None: stock evaluation
    evaluator_scale: float = 1.0  # learned leaf scale k (score = sigmoid(k * dz))
    selection: str = "ucb1"  # engine tree policy: ucb1 | exp3 | rm
    selection_gamma: float = 0.1  # exploration rate for exp3 / rm
    nash_pseudocount: float = nash_module.DEFAULT_PSEUDOCOUNT  # mode="nash": cell shrinkage c
    nash_floor: float = DEFAULT_NASH_FLOOR  # mode="nash": drop actions below this share
    tendency: bool = False  # 589.40 opponent tendency model (implies nash-style root solve)
    tendency_kappa: float = tendency_module.DEFAULT_KAPPA  # prior pseudo-observations
    tendency_delta: float = tendency_module.DEFAULT_DELTA  # safe-exploitation loss bound
    inference: bool = True  # 589.28 opponent set inference (False: pre-589.28 sampling)
    damage_inference: bool = True  # ...of which: damage-roll inference (needs inference)
    eval_lowhp_knee: float = 0.0  # 589.32 experiment: >0 discounts alive/item/boost value of low-HP mons
    eval_boost_scale: float = 1.0  # 589.32 experiment: multiplier on all stat-boost eval terms
    # Soft Bayesian belief reweighting of opponent sets (bead showdown-bot-1wu; default off)
    belief_reweight: bool = False
    belief_tau: float = belief_module.DEFAULT_TAU  # tempering exponent of the action evidence
    belief_floor: float = belief_module.DEFAULT_FLOOR  # min share of the likeliest set
    # 589.47 decision quality (all default off)
    escalate: bool = False  # flat-root escalation: re-search longer when the top two are close
    escalate_gap: float = 0.02  # ...flat when the top two options' mean scores differ by less
    escalate_factor: float = 4.0  # ...final per-world budget as a multiple of the normal one
    escalate_steps: int = 1  # ...budget grows in this many equal steps (each re-searches)
    escalate_cap_ms: int = 3000  # ...hard wall-clock cap on the whole decision's search
    prune_noop: bool = False  # drop moves the engine proves have no effect (unless all do)
    noop_vs_switches: bool = True  # ...no-op must also hold against every opponent switch (strict)
    tera_margin: float = 0.0  # >0: X-tera needs a mean this much above X, else folds into X
    # Playout cap randomization (KataGo): with prob full_frac a decision gets the full search and
    # is recorded; otherwise a cheap fast_time_ms search that is NOT recorded. 1.0 = off.
    full_frac: float = 1.0
    fast_time_ms: int = 50
    # Self-play bookkeeping only (written to the record header; no effect on play).
    label: str = ""
    opponent: str = ""

    @classmethod
    def from_env(cls) -> SearchConfig:
        mode = os.environ.get("SHOWDOWN_SEARCH_MODE", MIXED).strip().lower() or MIXED
        if mode not in MODES:
            logger.warning("Unknown SHOWDOWN_SEARCH_MODE=%r; using %s", mode, MIXED)
            mode = MIXED
        selection = os.environ.get(SELECTION_ENV, "ucb1").strip().lower() or "ucb1"
        if selection not in SELECTIONS:
            logger.warning("Unknown %s=%r; using ucb1", SELECTION_ENV, selection)
            selection = "ucb1"
        return cls(
            search_time_ms=_env_int("SHOWDOWN_SEARCH_TIME_MS", 100) or 100,
            parallelism=_env_int("SHOWDOWN_SEARCH_PARALLELISM", 1) or 1,
            worlds=_env_int("SHOWDOWN_SEARCH_WORLDS", None),
            threads=_env_int("SHOWDOWN_SEARCH_THREADS", 1) or 1,
            mode=mode,
            evaluator=(os.environ.get(EVALUATOR_ENV, "").strip() or None),
            evaluator_scale=_env_scale(EVALUATOR_SCALE_ENV),
            selection=selection,
            selection_gamma=_env_gamma(SELECTION_GAMMA_ENV),
            nash_pseudocount=_env_float(
                NASH_PSEUDOCOUNT_ENV, nash_module.DEFAULT_PSEUDOCOUNT, 0.0, math.inf
            ),
            nash_floor=_env_float(NASH_FLOOR_ENV, DEFAULT_NASH_FLOOR, 0.0, 1.0),
            tendency=os.environ.get("SHOWDOWN_SEARCH_TENDENCY", "off").strip().lower()
            in ("1", "on", "true", "yes"),
            tendency_kappa=_env_float(
                "SHOWDOWN_SEARCH_TENDENCY_KAPPA", tendency_module.DEFAULT_KAPPA, 0.0, math.inf
            ),
            tendency_delta=_env_float(
                "SHOWDOWN_SEARCH_TENDENCY_DELTA", tendency_module.DEFAULT_DELTA, 0.0, 1.0
            ),
            inference=os.environ.get("SHOWDOWN_INFERENCE", "1").strip().lower()
            not in ("0", "off", "false", "no"),
            damage_inference=os.environ.get("SHOWDOWN_INFERENCE_DAMAGE", "1").strip().lower()
            not in ("0", "off", "false", "no"),
            eval_lowhp_knee=_env_float("SHOWDOWN_EVAL_LOWHP_KNEE", 0.0, 0.0, 1.0),
            eval_boost_scale=_env_float("SHOWDOWN_EVAL_BOOST_SCALE", 1.0, 0.0, 10.0),
            belief_reweight=os.environ.get("SHOWDOWN_BELIEF_REWEIGHT", "0").strip().lower()
            in ("1", "on", "true", "yes"),
            belief_tau=_env_float("SHOWDOWN_BELIEF_TAU", belief_module.DEFAULT_TAU, 0.0, 10.0),
            belief_floor=_env_float("SHOWDOWN_BELIEF_FLOOR", belief_module.DEFAULT_FLOOR, 0.0, 1.0),
        )

    def plan(self, early: bool) -> tuple[int, int]:
        """Return ``(number_of_worlds, time_per_world_ms)`` for this decision."""
        if self.worlds is not None:
            return self.worlds, self.search_time_ms
        if early:
            return self.parallelism * 4, self.search_time_ms // 2
        return self.parallelism * 2, self.search_time_ms


@dataclass
class SearchStats:
    decisions: int = 0
    searched: int = 0
    forced: int = 0
    fallbacks: int = 0
    worlds_run: int = 0
    world_errors: int = 0
    iterations: int = 0
    search_ms: int = 0  # sum over searched worlds of the per-world time budget
    unknown_ids: int = 0
    illusion_worlds: int = 0  # worlds whose active opponent was sampled as Zoroark
    escalations: int = 0  # 589.47: decisions whose search was extended
    noop_pruned: int = 0  # 589.47: decisions where >= 1 no-op option was dropped
    tera_folded: int = 0  # 589.47: decisions where >= 1 tera option was folded away
    fallback_reasons: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in self.fallback_reasons.most_common())
        return (
            f"decisions={self.decisions} searched={self.searched} forced={self.forced} "
            f"fallbacks={self.fallbacks}"
            + (f" ({reasons})" if reasons else "")
            + f" worlds={self.worlds_run} world_errors={self.world_errors} "
            f"iterations={self.iterations} engine_unknown_ids={self.unknown_ids} "
            f"illusion_worlds={self.illusion_worlds}"
            + (
                f" escalations={self.escalations} noop_pruned={self.noop_pruned} "
                f"tera_folded={self.tera_folded}"
                if self.escalations or self.noop_pruned or self.tera_folded
                else ""
            )
        )


_ALL_STATS: list[SearchStats] = []


def total_stats() -> SearchStats:
    """Sum of every SearchPlayer's stats in this process."""
    total = SearchStats()
    for stats in _ALL_STATS:
        for name in (
            "decisions", "searched", "forced", "fallbacks", "worlds_run",
            "world_errors", "iterations", "search_ms", "unknown_ids", "illusion_worlds",
            "escalations", "noop_pruned", "tera_folded",
        ):
            setattr(total, name, getattr(total, name) + getattr(stats, name))
        total.fallback_reasons.update(stats.fallback_reasons)
    return total


def inference_summary() -> str:
    """One line on the 589.28 opponent-set inference cost and effect."""
    from showdown_bot import damage_infer, sets

    return (
        f"event_inference_seconds={sets.INFERENCE_SECONDS[0]:.2f} "
        f"damage[{damage_infer.total_damage_stats().summary()}]"
        + (f" belief[{belief_module.STATS.summary()}]" if belief_module.STATS.seconds else "")
    )


def _print_total_stats() -> None:
    if _ALL_STATS and any(stats.decisions for stats in _ALL_STATS):
        print(f"search player stats: {total_stats().summary()}", file=sys.stderr, flush=True)
        print(f"inference stats: {inference_summary()}", file=sys.stderr, flush=True)


atexit.register(_print_total_stats)


_POOLS: dict[int, ProcessPoolExecutor] = {}


def get_pool(workers: int) -> ProcessPoolExecutor:
    """Shared process pool (one per worker count) for all SearchPlayers."""
    pool = _POOLS.get(workers)
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=workers)
        _POOLS[workers] = pool
        for _ in range(workers):
            pool.submit(_worker.ping)
    return pool


def _kill_workers(pool: ProcessPoolExecutor) -> None:
    """Terminate pool worker processes (a hung task would otherwise live forever)."""
    for process in list(getattr(pool, "_processes", {}).values()):
        try:
            process.terminate()
        except Exception:
            pass


def reset_pool(workers: int, only: Executor | None = None, *, kill: bool = False) -> None:
    """Drop the shared pool so the next ``get_pool`` builds a fresh one.

    ``only`` guards against a thundering herd: when several battles notice the
    same broken pool, only the first drops it, and later callers (whose pool is
    already gone or replaced) leave the new one alone.
    """
    pool = _POOLS.get(workers)
    if pool is None or (only is not None and pool is not only):
        return
    del _POOLS[workers]
    if kill:
        _kill_workers(pool)
    pool.shutdown(wait=False, cancel_futures=True)


def _shutdown_pools() -> None:
    for workers in list(_POOLS):
        pool = _POOLS.pop(workers)
        _kill_workers(pool)
        pool.shutdown(wait=False, cancel_futures=True)


atexit.register(_shutdown_pools)

_DATA_CACHE: dict[str, RandbatsData] = {}


def default_randbats_data() -> RandbatsData:
    key = str(_DEFAULT_CACHE_DIR)
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = RandbatsData.load(cache_dir=_DEFAULT_CACHE_DIR)
    return _DATA_CACHE[key]


# ---------------------------------------------------------------------------
# Pure policy helpers (unit-tested without a server)
# ---------------------------------------------------------------------------

# (total_visits, [(choice, visits, total_score), ...]) for ucb1; exp3 / rm append a
# third element: the average-strategy share of each option (aligned with the list).
WorldResult = tuple


def aggregate_policy(results: list[WorldResult]) -> dict[str, float]:
    """Average each choice's share over the worlds (weight 1/len(results)).

    The share is the visit share, or the average-strategy share when the world
    carries one (selection exp3 / rm: the Nash-converging quantity).
    """
    policy: dict[str, float] = {}
    usable = [r for r in results if r[0] > 0 and r[1]]
    if not usable:
        return policy
    weight = 1.0 / len(usable)
    for result in usable:
        total, options = result[0], result[1]
        if len(result) > 2 and result[2] is not None:
            shares = [float(s) for s in result[2]]
        else:
            visits_sum = sum(visits for _, visits, _ in options) or total
            shares = [visits / visits_sum for _, visits, _ in options]
        for (choice, _visits, _score), share in zip(options, shares):
            policy[choice] = policy.get(choice, 0.0) + weight * share
    return policy


@dataclass
class TendencyQuery:
    """The current context's observed counts plus blending settings (mode ``tendency``)."""

    counts: dict[str, float]
    kappa: float = tendency_module.DEFAULT_KAPPA
    delta: float = tendency_module.DEFAULT_DELTA
    diag: list = field(default_factory=list)  # one dict per world that was exploited


def nash_world(
    result: WorldResult, pseudocount: float, tendency: TendencyQuery | None = None
) -> tuple[dict[str, float], float, float] | None:
    """Solve one world's root matrix game: ``(our strategy by choice, value, exploitability)``.

    ``None`` when the world carries no joint matrix. The matrix is built per world
    (the opponent's option set differs between sampled worlds, so matrices from
    different worlds are never averaged).
    """
    if len(result) < 4 or result[3] is None or not result[1]:
        return None
    k2, cells = result[3]
    options = result[1]
    matrix = nash_module.shrunk_matrix(len(options), int(k2), cells, pseudocount)
    solution = nash_module.solve_matrix_game(matrix)
    row = solution.row
    if tendency is not None and len(result) > 4 and result[4]:
        classes = [tendency_module.classify_choice(c) for c in result[4]]
        if len(classes) == matrix.shape[1]:
            row, info = tendency_module.exploit_world(
                matrix, solution.row, solution.col, solution.value, classes,
                tendency.counts, tendency.kappa, tendency.delta,
            )
            info["br_choice"] = options[info["br"]][0]
            tendency.diag.append(info)
    strategy: dict[str, float] = {}
    for (choice, _visits, _score), p in zip(options, row):
        strategy[choice] = strategy.get(choice, 0.0) + float(p)
    return strategy, solution.value, solution.exploitability


def aggregate_nash(
    results: list[WorldResult], pseudocount: float, floor: float = 0.0,
    tendency: TendencyQuery | None = None,
) -> tuple[dict[str, float], list[dict[str, Any] | None]]:
    """Nash policy over worlds, plus a per-world record (aligned with ``results``).

    Each world's equilibrium strategy for us is averaged with weight = that world's
    total root visits, then choices below ``floor`` are dropped and the rest
    renormalised. Empty policy if no world carried a joint matrix.
    """
    weighted: list[tuple[float, dict[str, float]]] = []
    records: list[dict[str, Any] | None] = []
    for result in results:
        solved = nash_world(result, pseudocount, tendency) if result[0] > 0 else None
        if solved is None:
            records.append(None)
            continue
        strategy, value, exploitability = solved
        weighted.append((float(result[0]), strategy))
        records.append(
            {"strategy": strategy, "value": value, "exploitability": exploitability}
        )
    return nash_module.aggregate_strategies(weighted, floor), records


def select_choice(
    policy: dict[str, float],
    mode: str,
    rng: random.Random,
    threshold: float = MIXED_THRESHOLD,
) -> str | None:
    """Pick a choice from an aggregated policy.

    ``mixed``: sample proportionally among choices whose mass is at least
    ``threshold`` x the best choice's. ``argmax``: the best choice. ``nash``: sample
    proportionally from the whole policy (it is already a mixed strategy).
    """
    if not policy:
        return None
    ranked = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    if mode == ARGMAX:
        return ranked[0][0]
    if mode == NASH:
        if ranked[0][1] <= 0:
            return ranked[0][0]
        return rng.choices([c for c, _ in ranked], weights=[m for _, m in ranked])[0]
    best = ranked[0][1]
    considered = [item for item in ranked if item[1] >= best * threshold]
    return rng.choices(
        [choice for choice, _ in considered], weights=[mass for _, mass in considered]
    )[0]


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------


def _opp_facts(battle: Any, trackers: Any) -> list[dict[str, Any]]:
    """What the set tracker knows about each revealed opponent Pokemon (recording only).

    Active Pokemon first, then the others (the order worlds use). ``n_sets`` is
    the number of full randbats sets still consistent with the evidence and
    ``entropy`` the Shannon entropy (bits) of their occurrence-count weights.
    """
    import math

    out: list[dict[str, Any]] = []
    try:
        active = battle.opponent_active_pokemon
        mons = sorted(battle.opponent_team.items(), key=lambda kv: 0 if kv[1] is active else 1)
        for identifier, mon in mons:
            tracker = trackers.get(str(identifier))
            row: dict[str, Any] = {"species": str(getattr(mon, "species", "")).lower()}
            if tracker is not None:
                matches = tracker._matching_full_sets()
                counts = [max(0.0, float(fs.count)) for _, fs in matches]
                total = sum(counts)
                entropy = -sum(c / total * math.log2(c / total) for c in counts if c > 0) if total else 0.0
                # same, ignoring the tera type (unknowable before it is used)
                groups: dict[tuple, float] = {}
                for sid, fs in matches:
                    key = (sid, str(fs.item), str(fs.ability), tuple(sorted(str(m) for m in fs.moves)))
                    groups[key] = groups.get(key, 0.0) + max(0.0, float(fs.count))
                gt = sum(groups.values())
                entropy_nt = -sum(c / gt * math.log2(c / gt) for c in groups.values() if c > 0) if gt else 0.0
                row.update(
                    moves=[str(m) for m in tracker.revealed_moves],
                    item=tracker._revealed_item,
                    ability=tracker._revealed_ability,
                    tera=tracker._revealed_tera,
                    n_constraints=len(tracker.constraints),
                    n_sets=len(matches),
                    entropy=round(entropy, 3),
                    n_sets_nt=len(groups),
                    vary={
                        "item": len({str(fs.item) for _, fs in matches}),
                        "ability": len({str(fs.ability) for _, fs in matches}),
                        "tera": len({str(fs.tera_type) for _, fs in matches}),
                        "moves": len({tuple(sorted(str(m) for m in fs.moves)) for _, fs in matches}),
                    },
                    entropy_nt=round(entropy_nt, 3),
                    groups=[
                        [k[1], k[2], "/".join(k[3]), round(w / gt, 4)]
                        for k, w in sorted(groups.items(), key=lambda kv: -kv[1])[:40]
                    ],
                    inconsistent=bool(tracker.inconsistent),
                )
                if tracker.belief_enabled:
                    row["belief"] = tracker.belief_summary()
            out.append(row)
    except Exception:
        logger.debug("opp_facts failed", exc_info=True)
    return out


def mode_for_record(config: SearchConfig) -> str:
    """``nash`` whenever the root joint matrix is needed (nash mode or tendency)."""
    return NASH if config.tendency else config.mode


class SearchPlayer(OpponentModelPlayerMixin, Player):
    """MCTS over sampled opponent worlds; see the module docstring."""

    def __init__(
        self,
        *args: Any,
        opponent_model: OpponentModel | None = None,
        data: RandbatsData | None = None,
        config: SearchConfig | None = None,
        executor: Executor | None = None,
        seed: int | None = None,
        recorder: GameRecorder | None = None,
        evaluator: str | os.PathLike[str] | None = None,
        evaluator_scale: float | None = None,
        selection: str | None = None,
        selection_gamma: float | None = None,
        mode: str | None = None,
        nash_pseudocount: float | None = None,
        nash_floor: float | None = None,
        tendency: bool | None = None,
        tendency_kappa: float | None = None,
        tendency_delta: float | None = None,
        belief_reweight: bool | None = None,
        **kwargs: Any,
    ):
        data = data or (opponent_model.data if opponent_model else default_randbats_data())
        if "account_configuration" not in kwargs:
            # poke-env numbers default usernames per process ("SearchPlayer 1"),
            # so concurrent benchmark processes would collide on the server.
            kwargs["account_configuration"] = AccountConfiguration(
                f"Search{secrets.token_hex(4)}", None
            )
        super().__init__(
            *args, opponent_model=opponent_model or OpponentModel(data), **kwargs
        )
        self.data = data
        self.config = config or SearchConfig.from_env()
        if evaluator is not None:
            self.config = dataclasses.replace(self.config, evaluator=str(Path(evaluator).resolve()))
        if evaluator_scale is not None:
            self.config = dataclasses.replace(self.config, evaluator_scale=float(evaluator_scale))
        if selection is not None:
            self.config = dataclasses.replace(self.config, selection=str(selection).lower())
        if selection_gamma is not None:
            self.config = dataclasses.replace(self.config, selection_gamma=float(selection_gamma))
        if mode is not None:
            self.config = dataclasses.replace(self.config, mode=str(mode).lower())
        if nash_pseudocount is not None:
            self.config = dataclasses.replace(self.config, nash_pseudocount=float(nash_pseudocount))
        if nash_floor is not None:
            self.config = dataclasses.replace(self.config, nash_floor=float(nash_floor))
        if tendency is not None:
            self.config = dataclasses.replace(self.config, tendency=bool(tendency))
        if tendency_kappa is not None:
            self.config = dataclasses.replace(self.config, tendency_kappa=float(tendency_kappa))
        if tendency_delta is not None:
            self.config = dataclasses.replace(self.config, tendency_delta=float(tendency_delta))
        if belief_reweight is not None:
            self.config = dataclasses.replace(self.config, belief_reweight=bool(belief_reweight))
        if not (0.0 <= self.config.full_frac <= 1.0):
            raise ValueError(f"full_frac must be in [0, 1], got {self.config.full_frac}")
        if self.config.fast_time_ms < 1:
            raise ValueError(f"fast_time_ms must be >= 1, got {self.config.fast_time_ms}")
        if not self.config.inference:
            self.opponent_model.inference_enabled = False
        if not self.config.damage_inference:
            self.opponent_model.damage_enabled = False
        if not (math.isfinite(self.config.belief_tau) and self.config.belief_tau >= 0):
            raise ValueError(f"belief_tau must be finite and >= 0, got {self.config.belief_tau}")
        if not (math.isfinite(self.config.belief_floor) and 0 <= self.config.belief_floor < 1):
            raise ValueError(f"belief_floor must be in [0, 1), got {self.config.belief_floor}")
        if self.config.belief_reweight:
            self.opponent_model.belief_enabled = True
            self.opponent_model.belief_tau = self.config.belief_tau
            self.opponent_model.belief_floor = self.config.belief_floor
        if self.config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.config.mode!r}")
        if not (math.isfinite(self.config.tendency_kappa) and self.config.tendency_kappa >= 0):
            raise ValueError(f"tendency_kappa must be finite and >= 0, got {self.config.tendency_kappa}")
        if not (math.isfinite(self.config.tendency_delta) and 0 <= self.config.tendency_delta < 1):
            raise ValueError(f"tendency_delta must be in [0, 1), got {self.config.tendency_delta}")
        if self.config.tendency and self.config.threads > 1:
            raise ValueError("tendency needs threads == 1 (it solves the root joint matrix)")
        if self.config.mode == NASH and self.config.threads > 1:
            raise ValueError("mode nash needs threads == 1 (the shared-tree search has no joint matrix)")
        if not (math.isfinite(self.config.nash_pseudocount) and self.config.nash_pseudocount >= 0):
            raise ValueError(f"nash_pseudocount must be finite and >= 0, got {self.config.nash_pseudocount}")
        if not (math.isfinite(self.config.nash_floor) and 0 <= self.config.nash_floor < 1):
            raise ValueError(f"nash_floor must be in [0, 1), got {self.config.nash_floor}")
        if self.config.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {self.config.selection!r}")
        if not (math.isfinite(self.config.selection_gamma) and 0 < self.config.selection_gamma <= 1):
            raise ValueError(f"selection_gamma must be in (0, 1], got {self.config.selection_gamma}")
        if self.config.selection != "ucb1" and self.config.threads > 1:
            raise ValueError("selection exp3/rm needs threads == 1 (the shared-tree search is ucb1 only)")
        if not (math.isfinite(self.config.evaluator_scale) and self.config.evaluator_scale > 0):
            raise ValueError(f"evaluator_scale must be finite and > 0, got {self.config.evaluator_scale}")
        if self.config.evaluator is not None:
            # Fail here rather than as a fallback on every decision.
            if not Path(self.config.evaluator).is_file():
                raise FileNotFoundError(f"evaluator file not found: {self.config.evaluator}")
            self.config = dataclasses.replace(
                self.config, evaluator=str(Path(self.config.evaluator).resolve())
            )
        cfg = self.config
        if not (math.isfinite(cfg.escalate_gap) and cfg.escalate_gap >= 0):
            raise ValueError(f"escalate_gap must be finite and >= 0, got {cfg.escalate_gap}")
        if not (math.isfinite(cfg.escalate_factor) and cfg.escalate_factor >= 1):
            raise ValueError(f"escalate_factor must be >= 1, got {cfg.escalate_factor}")
        if cfg.escalate_steps < 1 or cfg.escalate_cap_ms < 1:
            raise ValueError("escalate_steps and escalate_cap_ms must be >= 1")
        if not (math.isfinite(cfg.tera_margin) and cfg.tera_margin >= 0):
            raise ValueError(f"tera_margin must be finite and >= 0, got {cfg.tera_margin}")
        self.stats = SearchStats()
        _ALL_STATS.append(self.stats)
        self._executor = executor
        self._rng = random.Random(seed)
        # Separate stream for the Illusion roll so worlds are unchanged at P = 0.
        self._illusion_rng = random.Random(None if seed is None else f"illusion-{seed}")
        self._cap_rng = random.Random(None if seed is None else f"cap-{seed}")
        self._sampler = engine_state.TeamSampler(data)
        self.last_policy: dict[str, float] = {}
        self.last_tendency: dict[str, Any] | None = None  # diagnostics of the last exploit
        self.cond_turns = 0  # decisions where we held Sucker Punch-like moves
        self.cond_used = 0  # ...of which we chose one
        self.tendency_decisions = 0  # decisions where evidence moved the strategy
        self.last_choice: str | None = None
        self.last_changes: dict[str, Any] = {}
        # Recording is off unless a recorder is passed or SHOWDOWN_RECORD_DIR is set.
        self._recorder = recorder if recorder is not None else recorder_module.from_env()
        self._paired: list[tuple[str, WorldResult]] = []
        self._result_worlds: list[int] = []  # world index of each result of the last run
        self._belief_pending: dict[str, belief_module.ActionRecord] = {}
        if self._recorder is not None and self._recorder.truth:
            recorder_module.register_player(self)

    # -- poke-env entry point ------------------------------------------------

    async def choose_move(self, battle: Any) -> BattleOrder:
        self.stats.decisions += 1
        reason = "no-order"
        try:
            order = await self._choose(battle)
            if order is not None:
                return order
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never crash a battle
            reason = type(exc).__name__
            logger.exception("Search failed in %s; falling back", battle.battle_tag)
        return self._fallback(battle, reason)

    def _fallback(self, battle: Any, reason: str) -> BattleOrder:
        self.stats.fallbacks += 1
        self.stats.fallback_reasons[reason] += 1
        logger.warning("Search fallback (%s) in %s", reason, battle.battle_tag)
        try:
            return self.choose_random_move(battle)
        except Exception:
            return DefaultBattleOrder()

    # -- search --------------------------------------------------------------

    def _is_early(self, battle: Any) -> bool:
        active = battle.opponent_active_pokemon
        if active is None:
            return True
        return (
            len(battle.opponent_team) <= 3
            and not active.fainted
            and len(active.moves) == 0
        )

    def _draw_full(self) -> bool:
        """Playout cap randomization: True = full (recorded) search for this decision."""
        frac = self.config.full_frac
        if frac >= 1.0:
            return True  # off: no rng draw, so seeded play is unchanged
        return self._cap_rng.random() < frac

    async def _choose(self, battle: Any) -> BattleOrder | None:
        query = None
        self._tendency_context = None
        if not getattr(battle, "force_switch", False):
            try:  # bookkeeping for validation (how often we hold Sucker Punch-like moves)
                self._tendency_context = tendency_module.context_from_battle(battle)
                self.cond_turns += self._tendency_context.cond_priority
            except Exception:
                pass
        if self.config.tendency:
            try:
                query = self._tendency_query(battle)
            except Exception:  # the model must never cost a decision
                logger.exception("Tendency query failed in %s", battle.battle_tag)
        if self.config.belief_reweight:
            self._apply_belief_action(battle)
        legal = battle.valid_orders
        if len(legal) == 1:
            self.stats.forced += 1
            return legal[0]
        if not legal:
            return None
        trackers = self.opponent_model.update(battle)
        full = self._draw_full()
        plan_cfg = self.config if full else dataclasses.replace(
            self.config, search_time_ms=self.config.fast_time_ms
        )
        n_worlds, time_ms = plan_cfg.plan(self._is_early(battle))
        recording = self._recorder is not None and full
        if recording:
            started = time.perf_counter()
            truth_state, truth_status = self._truth_state(battle)
            opp_facts = _opp_facts(battle, trackers)
        built = None
        state_strings: list[str] = []
        try:
            illusion = self.opponent_model.illusion_estimate(battle)
        except Exception:  # a suspicion bug must never cost the decision
            logger.exception("Illusion estimate failed in %s", battle.battle_tag)
            illusion = None
        if illusion is not None and illusion.total <= 0:
            illusion = None
        world_slots: list[list[engine_state.OpponentSlot]] = []
        for _ in range(n_worlds):
            slots = engine_state.sample_opponent_team(
                battle, trackers, self._sampler, self._rng,
                illusion=illusion, illusion_rng=self._illusion_rng,
            )
            world_slots.append(slots)
            if slots and slots[0].disguised:
                self.stats.illusion_worlds += 1
            world = engine_state.build_state(battle, slots)
            built = built or world
            self.stats.unknown_ids += len(world.unknown_ids)
            state_strings.append(world.to_string())
        results = await self._run_worlds(state_strings, time_ms)
        result_worlds = list(self._result_worlds)
        self.stats.searched += 1
        if not results:
            raise RuntimeError("every world failed")
        cfg = self.config
        noop: set[str] = set()
        if cfg.prune_noop:
            choices = sorted({c for r in results for c, _v, _s in r[1]})
            noop = dq.noop_choices(state_strings, choices, cfg.noop_vs_switches)
        final_ms = time_ms
        escalated = 0
        pool_size = getattr(self._executor, "_max_workers", None) or cfg.parallelism
        waves = max(1, -(-len(state_strings) // max(1, pool_size)))
        spent_s = waves * time_ms / 1000.0  # compute time used so far (queue-free)
        policy, nash_records, mode = self._policy_of(results, battle, query)
        adjusted, changes = dq.adjust_policy(
            policy, results, noop=noop, tera_margin=cfg.tera_margin
        )
        if full and cfg.escalate and cfg.escalate_steps > 0:
            for step in range(1, cfg.escalate_steps + 1):
                if not dq.flat_root(adjusted, results, cfg.escalate_gap):
                    break
                target = time_ms * (1.0 + (cfg.escalate_factor - 1.0) * step / cfg.escalate_steps)
                # Budget from per-world compute time, never from wall time: wall time
                # includes process-pool queueing, which under load (many games in
                # flight) made escalation silently stop firing. A run of ``ms`` per
                # world costs ``waves * ms`` of compute; spend what the cap leaves.
                left_s = cfg.escalate_cap_ms / 1000.0 - spent_s
                affordable = left_s * 1000.0 / waves
                new_ms = int(min(target, affordable))
                if new_ms < final_ms * 1.25:
                    break
                more = await self._run_worlds(state_strings, new_ms)
                if not more:
                    break
                spent_s += waves * new_ms / 1000.0
                results, final_ms = more, new_ms
                result_worlds = list(self._result_worlds)
                escalated += 1
                policy, nash_records, mode = self._policy_of(results, battle, query)
                adjusted, changes = dq.adjust_policy(
                    policy, results, noop=noop, tera_margin=cfg.tera_margin
                )
        self.stats.escalations += bool(escalated)
        self.stats.noop_pruned += bool(changes["noop_pruned"])
        self.stats.tera_folded += bool(changes["tera_folded"])
        policy = adjusted
        self.last_policy = policy
        self.last_changes = {**changes, "escalated": escalated, "search_time_ms": final_ms}
        assert built is not None
        order = self._order_from_policy(policy, built, battle, mode)
        if self.config.belief_reweight:
            self._remember_action(battle, trackers, world_slots, results, result_worlds)
        if recording:
            self._record_decision(
                battle, order, policy, n_worlds, final_ms, started, truth_state, truth_status,
                nash_records if mode_for_record(self.config) == NASH else None,
                opp_facts,
            )
        return order

    # -- soft belief (bead showdown-bot-1wu) -----------------------------------

    def _apply_belief_action(self, battle: Any) -> None:
        """Fold the opponent's last observed action into its active Pokemon's belief."""
        record = self._belief_pending.get(battle.battle_tag)
        if record is None:
            return
        try:
            if self.opponent_model.observed_action(battle.battle_tag, record.turn) is None:
                if battle.turn > record.turn:
                    self._belief_pending.pop(battle.battle_tag, None)  # it never showed
                return
            self._belief_pending.pop(battle.battle_tag, None)
            self.opponent_model.update(battle)
            self.opponent_model.apply_action_belief(battle, record)
        except Exception:  # the belief must never cost a decision
            logger.exception("Belief action update failed in %s", battle.battle_tag)

    def _remember_action(
        self, battle: Any, trackers: Any, world_slots: list, results: list, result_worlds: list[int]
    ) -> None:
        """Keep the opponent's root visit shares per searched world for the next update."""
        try:
            if getattr(battle, "force_switch", False):
                return  # no simultaneous opponent choice to learn from
            active = battle.opponent_active_pokemon
            if active is None:
                return
            key = next((str(k) for k, v in battle.opponent_team.items() if v is active), None)
            tracker = trackers.get(key) if key else None
            if tracker is None or not tracker.belief_enabled:
                return
            worlds: list[belief_module.WorldAction] = []
            for result, index in zip(results, result_worlds):
                opp = getattr(result, "opp", None)
                slots = world_slots[index] if index < len(world_slots) else None
                if not opp or not slots:
                    continue
                slot = slots[0]
                set_key = None
                if slot.mon is active and not slot.disguised:
                    set_key = engine_state.sampled_set_key(slot)
                worlds.append(belief_module.WorldAction(set_key, belief_module.visit_shares(opp)))
            if worlds:
                self._belief_pending[battle.battle_tag] = belief_module.ActionRecord(
                    int(battle.turn), key, worlds
                )
        except Exception:
            logger.exception("Belief bookkeeping failed in %s", battle.battle_tag)

    def _policy_of(
        self, results: list[WorldResult], battle: Any, query: TendencyQuery | None = None
    ) -> tuple[dict[str, float], list[dict[str, Any] | None], str]:
        """Aggregate the world results into a root policy; returns ``(policy, nash records, mode)``."""
        mode = NASH if self.config.tendency else self.config.mode
        nash_records: list[dict[str, Any] | None] = []
        policy: dict[str, float] = {}
        if mode == NASH:
            policy, nash_records = aggregate_nash(
                results, self.config.nash_pseudocount, self.config.nash_floor, query
            )
            self._note_tendency(query)
            if not policy:  # no world carried a joint matrix: use the visit-share rule
                logger.warning("No Nash policy in %s; using the mixed rule", battle.battle_tag)
                mode = MIXED
        if not policy:
            policy = aggregate_policy(results)
        return policy, nash_records, mode

    def _tendency_query(self, battle: Any) -> TendencyQuery:
        """Observed counts for the current context; also registers it for the tracker."""
        tracker = tendency_module.for_battle(battle.battle_tag, create=True)
        context = tendency_module.context_from_battle(battle, tracker.last_class)
        if not getattr(battle, "force_switch", False):
            tracker.set_context(battle.turn, context)
        self._tendency_context = context
        # The all-clear bucket pools every ordinary turn, so its counts only restate the
        # Nash mix; exploit only contexts that carry a guessing-game feature.
        counts = tracker.counts_for(context) if context.key != "---" else {}
        return TendencyQuery(counts, self.config.tendency_kappa, self.config.tendency_delta)

    def _note_tendency(self, query: TendencyQuery | None) -> None:
        self.last_tendency = None
        if query is None or not query.diag:
            return
        n = query.diag[0]["n"]
        self.last_tendency = {
            "context": self._tendency_context.key,
            "n": n,
            "lambda": sum(d["lambda"] for d in query.diag) / len(query.diag),
            "post": query.diag[0]["post"],
            "br": [d["br_choice"] for d in query.diag],
        }
        if n > 0:
            self.tendency_decisions += 1

    def _order_from_policy(
        self,
        policy: dict[str, float],
        built: engine_state.BuiltState,
        battle: Any,
        mode: str | None = None,
    ) -> BattleOrder | None:
        legal_policy: dict[str, float] = {}
        orders: dict[str, BattleOrder] = {}
        for choice, mass in policy.items():
            order = built.choice_to_order(choice, battle)
            if order is not None:
                legal_policy[choice] = mass
                orders[choice] = order
        choice = select_choice(
            legal_policy, mode or self.config.mode, self._rng, self.config.mixed_threshold
        )
        self.last_choice = choice
        if choice is not None and getattr(self, "_tendency_context", None) is not None:
            if self._tendency_context.cond_priority and choice.removesuffix("-tera") in tendency_module.PRIORITY_CONDITIONAL:
                self.cond_used += 1
        return orders[choice] if choice is not None else None

    async def _run_worlds(self, state_strings: list[str], time_ms: int) -> list[WorldResult]:
        """Search every world; a broken shared pool is rebuilt and retried once."""
        for attempt in (0, 1):
            executor = self._executor or get_pool(self.config.parallelism)
            try:
                return await self._run_worlds_on(executor, state_strings, time_ms)
            except BrokenProcessPool:
                if self._executor is not None:
                    raise
                reset_pool(self.config.parallelism, only=executor)
                if attempt:
                    raise
                logger.warning("Process pool broke; rebuilt it and retrying")
        raise AssertionError("unreachable")

    async def _run_worlds_on(
        self, executor: Executor, state_strings: list[str], time_ms: int
    ) -> list[WorldResult]:
        loop = asyncio.get_running_loop()
        self._paired = []
        search_fn = functools.partial(
            _worker.search_world,
            opp_names=bool(self.config.tendency),
            opp_visits=bool(self.config.belief_reweight),
        ) if (self.config.tendency or self.config.belief_reweight) else _worker.search_world
        futures = [
            loop.run_in_executor(
                executor, search_fn, state, time_ms, self.config.threads,
                self.config.evaluator, self.config.evaluator_scale,
                self.config.selection, self.config.selection_gamma,
                mode_for_record(self.config) == NASH, self.config.eval_lowhp_knee,
                self.config.eval_boost_scale,
            )
            for state in state_strings
        ]
        waves = -(-len(futures) // max(1, self.config.parallelism))
        timeout = waves * (time_ms / 1000.0 + 1.0) + 10.0
        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(*futures, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            # A hung engine call cannot be interrupted; kill the workers so the
            # next decision gets a healthy pool.
            logger.error(
                "MCTS timed out after %.0fs; world states: %s",
                timeout,
                " || ".join(f"STATE[{i}] {state}" for i, state in enumerate(state_strings)),
            )
            if self._executor is None:
                reset_pool(self.config.parallelism, only=executor, kill=True)
            raise
        results: list[WorldResult] = []
        self._result_worlds = []
        for index, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, BrokenProcessPool):
                    raise outcome
                self.stats.world_errors += 1
                logger.warning(
                    "MCTS world failed: %r STATE[%d] %s", outcome, index, state_strings[index]
                )
                continue
            self.stats.worlds_run += 1
            self.stats.iterations += outcome[0]
            self.stats.search_ms += time_ms
            results.append(outcome)
            self._result_worlds.append(index)
            if self._recorder is not None:
                self._paired.append((state_strings[index], outcome))
        return results

    # -- recording (only reached when a recorder is configured) ----------------

    def _truth_state(self, battle: Any) -> tuple[str | None, str]:
        """The exact combined state when the opposing player is in this process."""
        if not self._recorder.truth:
            return None, "disabled"
        try:
            other = recorder_module.peer_battle(battle)
            if other is None:
                return None, "no-peer"
            state = recorder_module.build_truth_state(battle, other)
            return state, ("ok" if state else "turn-mismatch")
        except Exception:
            logger.exception("Truth state failed in %s", battle.battle_tag)
            return None, "error"

    def _record_decision(
        self,
        battle: Any,
        order: BattleOrder | None,
        policy: dict[str, float],
        n_worlds: int,
        time_ms: int,
        started: float,
        truth_state: str | None,
        truth_status: str,
        nash_records: list[dict[str, Any] | None] | None = None,
        opp_facts: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            recorder = self._recorder
            tag = battle.battle_tag
            if not recorder.has_game(tag, self.username):
                recorder.start_game(
                    tag,
                    self.username,
                    role=getattr(battle, "player_role", None),
                    battle_format=self.format,
                    config=asdict(self.config),
                    teams=getattr(self, "team_info", None),
                )
            worlds = [
                recorder_module.world_record(
                    s, r, nash_records[k] if nash_records and k < len(nash_records) else None
                )
                for k, (s, r) in enumerate(self._paired)
            ]
            recorder.add_decision(
                tag,
                self.username,
                {
                    "turn": battle.turn,
                    "side": "one",  # our side is always engine side one in every state
                    "worlds": worlds,
                    "policy": policy,
                    "nash": policy if nash_records is not None else None,
                    "chosen": self.last_choice,
                    "order": order.message if order is not None else None,
                    "compute": {
                        "search_time_ms": time_ms,
                        "worlds_requested": n_worlds,
                        "worlds_searched": len(worlds),
                        "parallelism": self.config.parallelism,
                        "threads": self.config.threads,
                        "iterations": sum(w["total_visits"] for w in worlds),
                        "wall_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                    "quality": self.last_changes or None,
                    "truth": {"state": truth_state} if truth_state else None,
                    "truth_status": truth_status,
                    "opp_facts": opp_facts,
                },
            )
        except Exception:
            logger.exception("Recording failed in %s", battle.battle_tag)

    def _battle_finished_callback(self, battle: Any) -> None:
        super()._battle_finished_callback(battle)
        if self._recorder is None:
            return
        try:
            self._recorder.end_game(
                battle.battle_tag,
                self.username,
                recorder_module.result_from_battle(battle),
                battle.turn,
            )
        except Exception:
            logger.exception("Could not write game record for %s", battle.battle_tag)


class SearchNetPlayer(SearchPlayer):
    """SearchPlayer whose leaves are scored by a learned ``.pev`` evaluator.

    The path comes from the ``evaluator=`` argument, ``config.evaluator`` or
    ``SHOWDOWN_SEARCH_EVALUATOR``; one of them is required (this is the bench
    ``search-net`` player).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.config.evaluator is None:
            raise ValueError(
                f"search-net needs an evaluator: pass evaluator=<path.pev> or set {EVALUATOR_ENV}"
            )
