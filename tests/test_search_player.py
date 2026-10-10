"""SearchPlayer: policy aggregation, move selection, fallback (no server needed)."""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import engine_state  # noqa: E402
from showdown_bot.players import search as search_module  # noqa: E402
from showdown_bot.players.search import (  # noqa: E402
    ARGMAX,
    MIXED,
    SearchConfig,
    SearchPlayer,
    aggregate_policy,
    select_choice,
)
from showdown_bot.sets import RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture
def player(data):
    pool = ThreadPoolExecutor(max_workers=2)
    p = SearchPlayer(
        battle_format="gen9randombattle",
        start_listening=False,
        log_level=50,
        data=data,
        config=SearchConfig(search_time_ms=20),
        executor=pool,
        seed=1,
    )
    yield p
    pool.shutdown(wait=False, cancel_futures=True)


def _battle(**kwargs):
    battle = make_battle(request_for(**kwargs))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


def _legal(battle):
    return {order.message for order in battle.valid_orders}


# --- pure policy helpers -------------------------------------------------------


def test_aggregate_policy_averages_visit_shares_per_world():
    world_a = (100, [("a", 75, 0.0), ("b", 25, 0.0)])
    world_b = (400, [("a", 100, 0.0), ("b", 300, 0.0)])
    policy = aggregate_policy([world_a, world_b])
    assert policy["a"] == pytest.approx((0.75 + 0.25) / 2)
    assert policy["b"] == pytest.approx((0.25 + 0.75) / 2)
    assert sum(policy.values()) == pytest.approx(1.0)
    assert aggregate_policy([]) == {}
    assert aggregate_policy([(0, [])]) == {}


def test_argmax_selection_is_deterministic():
    policy = {"a": 0.5, "b": 0.3, "c": 0.2}
    rng = random.Random(0)
    assert {select_choice(policy, ARGMAX, rng) for _ in range(50)} == {"a"}


def test_mixed_selection_only_considers_moves_within_75_percent_of_best():
    policy = {"best": 0.40, "close": 0.31, "far": 0.29 / 2 * 1.0, "tiny": 0.05}
    # thresholds: 0.75 * 0.40 = 0.30 -> best and close qualify only
    rng = random.Random(0)
    draws = Counter(select_choice(policy, MIXED, rng) for _ in range(20000))
    assert set(draws) == {"best", "close"}
    assert draws["best"] / 20000 == pytest.approx(0.40 / 0.71, abs=0.02)  # proportional
    assert select_choice({}, MIXED, rng) is None


def test_config_defaults_equal_foul_play_and_env_overrides(monkeypatch):
    config = SearchConfig()
    assert config.plan(early=False) == (2, 100)
    assert config.plan(early=True) == (4, 50)
    assert SearchConfig(parallelism=3).plan(early=False) == (6, 100)
    assert SearchConfig(worlds=5, parallelism=3).plan(early=True) == (5, 100)
    for name in list(__import__("os").environ):
        if name.startswith("SHOWDOWN_SEARCH_"):
            monkeypatch.delenv(name)
    assert SearchConfig.from_env() == SearchConfig()
    monkeypatch.setenv("SHOWDOWN_SEARCH_TIME_MS", "250")
    monkeypatch.setenv("SHOWDOWN_SEARCH_WORLDS", "6")
    monkeypatch.setenv("SHOWDOWN_SEARCH_THREADS", "2")
    monkeypatch.setenv("SHOWDOWN_SEARCH_MODE", "ARGMAX")
    monkeypatch.setenv("SHOWDOWN_SEARCH_PARALLELISM", "junk")
    env = SearchConfig.from_env()
    assert (env.search_time_ms, env.worlds, env.threads, env.mode, env.parallelism) == (
        250, 6, 2, ARGMAX, 1,
    )


# --- decisions -----------------------------------------------------------------


def test_choose_move_returns_a_legal_order_after_searching(player):
    battle = _battle()
    order = asyncio.run(player.choose_move(battle))
    assert order.message in _legal(battle)
    assert player.stats.searched == 1 and player.stats.fallbacks == 0
    assert player.stats.worlds_run == 4  # early game: 4 x parallelism worlds
    assert player.last_policy and sum(player.last_policy.values()) == pytest.approx(1.0)


def test_argmax_mode_picks_the_top_legal_choice(data):
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20, mode=ARGMAX), seed=1,
    )
    battle = _battle()
    trackers = p.opponent_model.update(battle)
    slots = engine_state.sample_opponent_team(battle, trackers, p._sampler, random.Random(0))
    built = engine_state.build_state(battle, slots)
    policy = {"notamove": 0.6, "closecombat": 0.1, "headlongrush-tera": 0.3, "switch kingambit": 0.0}
    # The 0.6 choice is illegal and is skipped; the best legal one is played.
    order = p._order_from_policy(policy, built, battle)
    assert order.message == "/choose move headlongrush terastallize"


def test_forced_switch_chooses_a_switch(player):
    battle = _battle(force_switch=True)
    order = asyncio.run(player.choose_move(battle))
    assert order.message.startswith("/choose switch ")
    assert order.message in _legal(battle)
    assert player.stats.fallbacks == 0


def test_single_legal_order_skips_search(player):
    hp = {i: (0, "") for i in range(1, 6)}
    battle = _battle(force_switch=True, hp={**hp, 0: (0, ""), 4: (100, "")})
    legal = _legal(battle)
    assert len(legal) == 1
    order = asyncio.run(player.choose_move(battle))
    assert order.message in legal
    assert player.stats.forced == 1 and player.stats.searched == 0


def test_start_of_battle_with_no_opponent_revealed(player):
    battle = make_battle()
    order = asyncio.run(player.choose_move(battle))
    assert order.message in _legal(battle)
    assert player.stats.fallbacks == 0


def test_fallback_on_injected_exception_returns_legal_order_and_counts(player, monkeypatch):
    battle = _battle()

    async def boom(_battle):
        raise RuntimeError("injected")

    monkeypatch.setattr(player, "_choose", boom)
    order = asyncio.run(player.choose_move(battle))
    assert order.message in _legal(battle)
    assert player.stats.fallbacks == 1
    assert player.stats.fallback_reasons["RuntimeError"] == 1
    assert search_module.total_stats().fallbacks >= 1


def test_fallback_when_every_world_fails(data):
    class BrokenExecutor(ThreadPoolExecutor):
        def submit(self, fn, *args, **kwargs):
            raise OSError("no workers")

    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20), executor=BrokenExecutor(max_workers=1), seed=1,
    )
    battle = _battle()
    order = asyncio.run(p.choose_move(battle))
    assert order.message in _legal(battle)
    assert p.stats.fallbacks == 1


def test_fallback_when_policy_has_no_legal_choice(player, monkeypatch):
    battle = _battle()
    monkeypatch.setattr(
        search_module, "aggregate_policy", lambda results: {"notalegalmove": 1.0}
    )
    order = asyncio.run(player.choose_move(battle))
    assert order.message in _legal(battle)
    assert player.stats.fallbacks == 1
    assert player.stats.fallback_reasons["no-order"] == 1


# -- learned evaluator wiring -------------------------------------------------


@pytest.fixture(scope="module")
def pev_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("pev") / "lin.pev"
    pe = pytest.importorskip("poke_engine")
    pe.write_linear(str(path), [0.0] * pe.FEATURE_LENGTH, 0.0)
    return path


def _spy_player(data, **kwargs):
    calls = []

    class Spy(ThreadPoolExecutor):
        def submit(self, fn, *args, **kw):
            calls.append(args)
            return super().submit(fn, *args, **kw)

    pool = Spy(max_workers=2)
    cls = kwargs.pop("cls", SearchPlayer)
    p = cls(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        executor=pool, seed=1, **kwargs,
    )
    return p, calls, pool


def test_no_evaluator_passes_none_to_worker(data):
    p, calls, pool = _spy_player(data, config=SearchConfig(search_time_ms=20))
    asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert calls and all(args[-1] is None for args in calls)
    assert p.config.evaluator is None


def test_evaluator_path_reaches_worker(data, pev_path):
    p, calls, pool = _spy_player(
        data, config=SearchConfig(search_time_ms=20), evaluator=pev_path
    )
    order = asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert order.message in _legal(_battle())
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0
    assert calls and all(args[-1] == str(pev_path.resolve()) for args in calls)


def test_evaluator_env_var_and_search_net_requirement(data, pev_path, monkeypatch):
    monkeypatch.setenv("SHOWDOWN_SEARCH_EVALUATOR", str(pev_path))
    p, _, pool = _spy_player(data, cls=search_module.SearchNetPlayer)
    pool.shutdown()
    assert p.config.evaluator == str(pev_path.resolve())
    monkeypatch.delenv("SHOWDOWN_SEARCH_EVALUATOR")
    with pytest.raises(ValueError):
        _spy_player(data, cls=search_module.SearchNetPlayer)
    with pytest.raises(FileNotFoundError):
        _spy_player(data, evaluator=pev_path.parent / "missing.pev")


def test_bench_registers_search_net():
    from showdown_bot import bench

    assert bench.get_player_class("search-net") is search_module.SearchNetPlayer


def test_worker_loads_evaluator_once_per_process(tmp_path):
    from concurrent.futures import ProcessPoolExecutor

    from showdown_bot.players import _worker

    pev_path = tmp_path / "fresh.pev"  # a path no earlier test has cached
    pe.write_linear(str(pev_path), [0.0] * pe.FEATURE_LENGTH, 0.0)

    state = pe.State().to_string()
    # in this process
    before = _worker.evaluator_loads()
    for _ in range(3):
        _worker.search_world(state, 10, 1, str(pev_path))
    assert _worker.evaluator_loads() - before == 1
    # in a real pool worker
    with ProcessPoolExecutor(max_workers=1) as pool:
        for _ in range(4):
            pool.submit(_worker.search_world, state, 10, 1, str(pev_path)).result()
        assert pool.submit(_worker.evaluator_loads).result() == 1
