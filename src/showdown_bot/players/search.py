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

The player never crashes a battle: any internal error falls back to a random
legal order and increments ``SearchStats.fallbacks`` (reasons are counted too).
A summary is printed to stderr at interpreter exit.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import random
import secrets
import sys
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder

from showdown_bot import engine_state
from showdown_bot.players import _worker
from showdown_bot.sets import OpponentModel, OpponentModelPlayerMixin, RandbatsData

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_DIR = _REPO_ROOT / ".cache" / "randbats"

MIXED = "mixed"
ARGMAX = "argmax"
MIXED_THRESHOLD = 0.75


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
            "world_errors", "iterations", "unknown_ids",
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
        self.stats = SearchStats()
        _ALL_STATS.append(self.stats)
        self._executor = executor
        self._rng = random.Random(seed)
        self._sampler = engine_state.TeamSampler(data)
        self.last_policy: dict[str, float] = {}

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
        return self._order_from_policy(policy, built, battle)

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
        futures = [
            loop.run_in_executor(
                executor, _worker.search_world, state, time_ms, self.config.threads
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
            results.append(outcome)
        return results
