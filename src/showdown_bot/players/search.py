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
   ``mode="argmax"`` takes the best move.

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
    SHOWDOWN_SEARCH_MODE         "mixed" (default) or "argmax"
    SHOWDOWN_SEARCH_EVALUATOR    path to a learned .pev leaf evaluator (default: the
                                 stock hand-written evaluation); see docs/training.md.
                                 Each pool worker loads it once and reuses it.
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
import logging
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

from showdown_bot import engine_state, recorder as recorder_module
from showdown_bot.players import _worker
from showdown_bot.recorder import GameRecorder
from showdown_bot.sets import OpponentModel, OpponentModelPlayerMixin, RandbatsData

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "randbats"

MIXED = "mixed"
ARGMAX = "argmax"
MIXED_THRESHOLD = 0.75
EVALUATOR_ENV = "SHOWDOWN_SEARCH_EVALUATOR"


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

    @classmethod
    def from_env(cls) -> SearchConfig:
        mode = os.environ.get("SHOWDOWN_SEARCH_MODE", MIXED).strip().lower() or MIXED
        if mode not in (MIXED, ARGMAX):
            logger.warning("Unknown SHOWDOWN_SEARCH_MODE=%r; using %s", mode, MIXED)
            mode = MIXED
        return cls(
            search_time_ms=_env_int("SHOWDOWN_SEARCH_TIME_MS", 100) or 100,
            parallelism=_env_int("SHOWDOWN_SEARCH_PARALLELISM", 1) or 1,
            worlds=_env_int("SHOWDOWN_SEARCH_WORLDS", None),
            threads=_env_int("SHOWDOWN_SEARCH_THREADS", 1) or 1,
            mode=mode,
            evaluator=(os.environ.get(EVALUATOR_ENV, "").strip() or None),
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
    fallback_reasons: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in self.fallback_reasons.most_common())
        return (
            f"decisions={self.decisions} searched={self.searched} forced={self.forced} "
            f"fallbacks={self.fallbacks}"
            + (f" ({reasons})" if reasons else "")
            + f" worlds={self.worlds_run} world_errors={self.world_errors} "
            f"iterations={self.iterations} engine_unknown_ids={self.unknown_ids}"
        )


_ALL_STATS: list[SearchStats] = []


def total_stats() -> SearchStats:
    """Sum of every SearchPlayer's stats in this process."""
    total = SearchStats()
    for stats in _ALL_STATS:
        for name in (
            "decisions", "searched", "forced", "fallbacks", "worlds_run",
            "world_errors", "iterations", "search_ms", "unknown_ids",
        ):
            setattr(total, name, getattr(total, name) + getattr(stats, name))
        total.fallback_reasons.update(stats.fallback_reasons)
    return total


def _print_total_stats() -> None:
    if _ALL_STATS and any(stats.decisions for stats in _ALL_STATS):
        print(f"search player stats: {total_stats().summary()}", file=sys.stderr, flush=True)


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

WorldResult = tuple[int, list[tuple[str, int, float]]]


def aggregate_policy(results: list[WorldResult]) -> dict[str, float]:
    """Average each choice's visit share over the worlds (weight 1/len(results))."""
    policy: dict[str, float] = {}
    usable = [(total, options) for total, options in results if total > 0 and options]
    if not usable:
        return policy
    weight = 1.0 / len(usable)
    for total, options in usable:
        visits_sum = sum(visits for _, visits, _ in options) or total
        for choice, visits, _score in options:
            policy[choice] = policy.get(choice, 0.0) + weight * (visits / visits_sum)
    return policy


def select_choice(
    policy: dict[str, float],
    mode: str,
    rng: random.Random,
    threshold: float = MIXED_THRESHOLD,
) -> str | None:
    """Pick a choice from an aggregated policy.

    ``mixed``: sample proportionally among choices whose mass is at least
    ``threshold`` x the best choice's. ``argmax``: the best choice.
    """
    if not policy:
        return None
    ranked = sorted(policy.items(), key=lambda item: item[1], reverse=True)
    if mode == ARGMAX:
        return ranked[0][0]
    best = ranked[0][1]
    considered = [item for item in ranked if item[1] >= best * threshold]
    return rng.choices(
        [choice for choice, _ in considered], weights=[mass for _, mass in considered]
    )[0]


# ---------------------------------------------------------------------------
# The player
# ---------------------------------------------------------------------------


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
        if self.config.evaluator is not None:
            # Fail here rather than as a fallback on every decision.
            if not Path(self.config.evaluator).is_file():
                raise FileNotFoundError(f"evaluator file not found: {self.config.evaluator}")
            self.config = dataclasses.replace(
                self.config, evaluator=str(Path(self.config.evaluator).resolve())
            )
        self.stats = SearchStats()
        _ALL_STATS.append(self.stats)
        self._executor = executor
        self._rng = random.Random(seed)
        self._sampler = engine_state.TeamSampler(data)
        self.last_policy: dict[str, float] = {}
        self.last_choice: str | None = None
        # Recording is off unless a recorder is passed or SHOWDOWN_RECORD_DIR is set.
        self._recorder = recorder if recorder is not None else recorder_module.from_env()
        self._paired: list[tuple[str, WorldResult]] = []
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

    async def _choose(self, battle: Any) -> BattleOrder | None:
        legal = battle.valid_orders
        if len(legal) == 1:
            self.stats.forced += 1
            return legal[0]
        if not legal:
            return None
        trackers = self.opponent_model.update(battle)
        n_worlds, time_ms = self.config.plan(self._is_early(battle))
        recording = self._recorder is not None
        if recording:
            started = time.perf_counter()
            truth_state, truth_status = self._truth_state(battle)
        built = None
        state_strings: list[str] = []
        for _ in range(n_worlds):
            slots = engine_state.sample_opponent_team(
                battle, trackers, self._sampler, self._rng
            )
            world = engine_state.build_state(battle, slots)
            built = built or world
            self.stats.unknown_ids += len(world.unknown_ids)
            state_strings.append(world.to_string())
        results = await self._run_worlds(state_strings, time_ms)
        self.stats.searched += 1
        if not results:
            raise RuntimeError("every world failed")
        policy = aggregate_policy(results)
        self.last_policy = policy
        assert built is not None
        order = self._order_from_policy(policy, built, battle)
        if recording:
            self._record_decision(
                battle, order, policy, n_worlds, time_ms, started, truth_state, truth_status
            )
        return order

    def _order_from_policy(
        self, policy: dict[str, float], built: engine_state.BuiltState, battle: Any
    ) -> BattleOrder | None:
        legal_policy: dict[str, float] = {}
        orders: dict[str, BattleOrder] = {}
        for choice, mass in policy.items():
            order = built.choice_to_order(choice, battle)
            if order is not None:
                legal_policy[choice] = mass
                orders[choice] = order
        choice = select_choice(
            legal_policy, self.config.mode, self._rng, self.config.mixed_threshold
        )
        self.last_choice = choice
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
        futures = [
            loop.run_in_executor(
                executor, _worker.search_world, state, time_ms, self.config.threads,
                self.config.evaluator,
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
                )
            worlds = [recorder_module.world_record(s, r) for s, r in self._paired]
            recorder.add_decision(
                tag,
                self.username,
                {
                    "turn": battle.turn,
                    "side": "one",  # our side is always engine side one in every state
                    "worlds": worlds,
                    "policy": policy,
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
                    "truth": {"state": truth_state} if truth_state else None,
                    "truth_status": truth_status,
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
