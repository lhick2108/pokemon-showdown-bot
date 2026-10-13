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


def test_aggregate_policy_uses_average_strategy_when_present():
    world_a = (100, [("a", 75, 0.0), ("b", 25, 0.0)], [0.2, 0.8])
    world_b = (400, [("a", 100, 0.0), ("b", 300, 0.0)], [0.4, 0.6])
    policy = aggregate_policy([world_a, world_b])
    assert policy["a"] == pytest.approx(0.3) and policy["b"] == pytest.approx(0.7)


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
    assert calls and all(args[3] is None for args in calls)
    assert p.config.evaluator is None


def test_evaluator_path_reaches_worker(data, pev_path):
    p, calls, pool = _spy_player(
        data, config=SearchConfig(search_time_ms=20), evaluator=pev_path
    )
    order = asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert order.message in _legal(_battle())
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0
    assert calls and all(args[3] == str(pev_path.resolve()) for args in calls)


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


def test_evaluator_scale_config_env_and_worker(data, pev_path, monkeypatch):
    assert SearchConfig().evaluator_scale == 1.0
    monkeypatch.setenv("SHOWDOWN_SEARCH_EVALUATOR_SCALE", "2.5")
    assert SearchConfig.from_env().evaluator_scale == 2.5
    for bad in ("0", "-1", "nan", "inf", "abc"):
        monkeypatch.setenv("SHOWDOWN_SEARCH_EVALUATOR_SCALE", bad)
        assert SearchConfig.from_env().evaluator_scale == 1.0
    monkeypatch.delenv("SHOWDOWN_SEARCH_EVALUATOR_SCALE")
    p, calls, pool = _spy_player(
        data, config=SearchConfig(search_time_ms=20), evaluator=pev_path, evaluator_scale=3.0
    )
    asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0
    assert calls and all(args[4] == 3.0 for args in calls)
    with pytest.raises(ValueError):
        _spy_player(data, evaluator=pev_path, evaluator_scale=0.0)


def test_worker_evaluator_cache_keys_on_scale(tmp_path):
    from showdown_bot.players import _worker

    path = tmp_path / "scaled.pev"
    pe.write_linear(str(path), [0.0] * pe.FEATURE_LENGTH, 0.0)
    a = _worker.get_evaluator(str(path), 1.0)
    assert _worker.get_evaluator(str(path), 1.0) is a
    b = _worker.get_evaluator(str(path), 2.0)
    assert b is not a and b.leaf_scale == 2.0 and a.leaf_scale == 1.0


def test_selection_config_env_and_validation(data, monkeypatch):
    assert SearchConfig().selection == "ucb1" and SearchConfig().selection_gamma == 0.1
    monkeypatch.setenv("SHOWDOWN_SEARCH_SELECTION", "RM")
    monkeypatch.setenv("SHOWDOWN_SEARCH_SELECTION_GAMMA", "0.25")
    cfg = SearchConfig.from_env()
    assert (cfg.selection, cfg.selection_gamma) == ("rm", 0.25)
    monkeypatch.setenv("SHOWDOWN_SEARCH_SELECTION", "bogus")
    monkeypatch.setenv("SHOWDOWN_SEARCH_SELECTION_GAMMA", "2")
    cfg = SearchConfig.from_env()
    assert (cfg.selection, cfg.selection_gamma) == ("ucb1", 0.1)
    with pytest.raises(ValueError):
        _spy_player(data, selection="bogus")
    with pytest.raises(ValueError):
        _spy_player(data, selection="exp3", selection_gamma=0.0)
    with pytest.raises(ValueError):
        _spy_player(data, config=SearchConfig(threads=2), selection="rm")


@pytest.mark.parametrize("mode", ["exp3", "rm"])
def test_selection_reaches_worker_and_policy_uses_strategy(data, mode):
    p, calls, pool = _spy_player(
        data, config=SearchConfig(search_time_ms=30), selection=mode, selection_gamma=0.2
    )
    order = asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert order.message in _legal(_battle())
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0
    assert calls and all(args[5:7] == (mode, 0.2) for args in calls)
    assert sum(p.last_policy.values()) == pytest.approx(1.0, abs=1e-3)


def test_worker_ucb1_result_shape_unchanged_and_strategy_appended():
    from showdown_bot.players import _worker
    from test_engine import make_state

    s = str(make_state().to_string())
    base = _worker.search_world(s, 20, 1)
    assert len(base) == 2
    ex = _worker.search_world(s, 20, 1, None, 1.0, "exp3", 0.1)
    assert len(ex) == 3 and len(ex[2]) == len(ex[1])
    assert sum(ex[2]) == pytest.approx(1.0, abs=1e-3)


# --- Nash mode (589.24) ----------------------------------------------------------

NASH = search_module.NASH


def _world(total, options, k2, cells):
    return (total, [(c, 100, 50.0) for c in options], None, (k2, cells))


def _full(rows, visits=500):
    return [(i, j, visits, v) for i, row in enumerate(rows) for j, v in enumerate(row)]


def test_nash_config_env_and_validation(data, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("SHOWDOWN_SEARCH_"):
            monkeypatch.delenv(name)
    cfg = SearchConfig()
    assert (cfg.nash_pseudocount, cfg.nash_floor) == (5.0, 0.03)
    monkeypatch.setenv("SHOWDOWN_SEARCH_MODE", "Nash")
    monkeypatch.setenv("SHOWDOWN_SEARCH_NASH_PSEUDOCOUNT", "2.5")
    monkeypatch.setenv("SHOWDOWN_SEARCH_NASH_FLOOR", "0.1")
    cfg = SearchConfig.from_env()
    assert (cfg.mode, cfg.nash_pseudocount, cfg.nash_floor) == (NASH, 2.5, 0.1)
    monkeypatch.setenv("SHOWDOWN_SEARCH_NASH_PSEUDOCOUNT", "-1")
    monkeypatch.setenv("SHOWDOWN_SEARCH_NASH_FLOOR", "1.5")
    cfg = SearchConfig.from_env()
    assert (cfg.nash_pseudocount, cfg.nash_floor) == (5.0, 0.03)
    with pytest.raises(ValueError):
        _spy_player(data, mode="bogus")
    with pytest.raises(ValueError):
        _spy_player(data, mode="nash", config=SearchConfig(threads=2))
    with pytest.raises(ValueError):
        _spy_player(data, mode="nash", nash_floor=1.0)
    with pytest.raises(ValueError):
        _spy_player(data, mode="nash", nash_pseudocount=-1.0)
    assert _spy_player(data, mode="nash")[0].config.mode == NASH


def test_select_choice_nash_samples_the_whole_policy():
    rng = random.Random(0)
    policy = {"a": 0.5, "b": 0.3, "c": 0.2}
    counts = Counter(select_choice(policy, NASH, rng) for _ in range(3000))
    assert set(counts) == {"a", "b", "c"}  # unlike mixed, c (< 75% of best) is played
    assert counts["a"] / 3000 == pytest.approx(0.5, abs=0.05)


def test_aggregate_nash_uses_each_worlds_own_option_set_and_visit_weights():
    # World A (1000 visits, 2 opponent options): closecombat strictly dominates.
    a = _world(1000, ["closecombat", "headlongrush"], 2, _full([[0.9, 0.8], [0.2, 0.1]]))
    # World B (3000 visits, 3 opponent options): headlongrush strictly dominates.
    b = _world(
        3000, ["closecombat", "headlongrush"], 3, _full([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    )
    policy, records = search_module.aggregate_nash([a, b], pseudocount=0.0, floor=0.0)
    assert policy == pytest.approx({"closecombat": 0.25, "headlongrush": 0.75})
    assert [r["value"] for r in records] == pytest.approx([0.8, 0.7])
    assert all(r["exploitability"] < 1e-9 for r in records)
    floored, _ = search_module.aggregate_nash([a, b], pseudocount=0.0, floor=0.3)
    assert floored == pytest.approx({"headlongrush": 1.0})
    # worlds without a joint matrix are skipped; none at all -> empty policy
    plain = (500, [("closecombat", 500, 250.0)])
    assert search_module.aggregate_nash([plain], 5.0)[0] == {}
    mixed_in, recs = search_module.aggregate_nash([plain, a], 0.0)
    assert mixed_in == pytest.approx({"closecombat": 1.0, "headlongrush": 0.0}) and recs[0] is None


def test_nash_mode_player_plays_the_aggregated_strategy(data):
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20, mode=NASH, nash_pseudocount=0.0, nash_floor=0.0),
        seed=3,
    )
    game_b = [[0.9, 0.2, 0.5], [0.1, 0.8, 0.5]]

    async def fake_run(state_strings, time_ms):
        # matching pennies in world A, a different 3-column game in world B
        return [
            _world(2000, ["closecombat", "headlongrush"], 2, _full([[1.0, 0.0], [0.0, 1.0]])),
            _world(2000, ["closecombat", "headlongrush"], 3, _full(game_b)),
        ]

    p._run_worlds = fake_run
    battle = _battle()
    counts = Counter()
    for _ in range(200):
        counts[asyncio.run(p.choose_move(battle)).message] += 1
    assert p.stats.fallbacks == 0
    assert all(m in _legal(battle) for m in counts)
    assert set(p.last_policy) == {"closecombat", "headlongrush"}
    assert sum(p.last_policy.values()) == pytest.approx(1.0)
    from showdown_bot.nash import solve_matrix_game

    b = solve_matrix_game(game_b).row[0]
    assert p.last_policy["closecombat"] == pytest.approx((0.5 + b) / 2)
    assert len(counts) >= 2  # genuinely mixed play, not argmax


def test_nash_mode_reaches_worker_and_returns_a_joint_matrix(data):
    p, calls, pool = _spy_player(
        data, config=SearchConfig(search_time_ms=30), mode="nash", nash_pseudocount=3.0
    )
    order = asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert order.message in _legal(_battle())
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0
    assert calls and all(args[7] is True for args in calls)
    assert sum(p.last_policy.values()) == pytest.approx(1.0, abs=1e-6)


def test_worker_joint_result_shape():
    from showdown_bot.players import _worker
    from test_engine import make_state

    s = str(make_state().to_string())
    for selection in ("ucb1", "rm"):
        res = _worker.search_world(s, 20, 1, None, 1.0, selection, 0.1, True)
        assert len(res) == 4
        total, options, strategies, (k2, cells) = res
        assert (strategies is None) == (selection == "ucb1")
        assert k2 >= 1 and cells and all(len(c) == 4 for c in cells)
        assert max(i for i, *_ in cells) < len(options)
    assert len(_worker.search_world(s, 20, 1)) == 2  # default path unchanged


def test_world_record_stores_nash_strategy_and_value():
    from showdown_bot import recorder as recorder_module

    outcome = _world(2000, ["closecombat", "headlongrush"], 2, _full([[1.0, 0.0], [0.0, 1.0]]))
    solved = search_module.aggregate_nash([outcome], 0.0)[1][0]
    record = recorder_module.world_record("S", outcome, solved)
    assert record["nash"]["value"] == pytest.approx(0.5)
    assert record["nash"]["strategy"] == pytest.approx({"closecombat": 0.5, "headlongrush": 0.5})
    assert "strategy" not in record["moves"][0]  # ucb1 world: no average strategy
    assert "nash" not in recorder_module.world_record("S", outcome)
