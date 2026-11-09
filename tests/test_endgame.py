"""Endgame mode (expectiminimax instead of MCTS in small endgames); no server needed."""

from __future__ import annotations

import asyncio
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from showdown_bot import endgame as eg
from showdown_bot.players.search import SearchConfig, SearchPlayer, select_choice

NAN = float("nan")


def result(ours, theirs, flat, depth=4):
    return (list(ours), list(theirs), list(flat), depth)


def world(ours, theirs, flat, depth=4):
    return eg.parse_world(result(ours, theirs, flat, depth))


# --- config: env, validation, spec keys -------------------------------------------------


def test_defaults_off():
    cfg = SearchConfig()
    assert cfg.endgame == "off" and cfg.endgame_max_alive == 5
    assert cfg.endgame_time_ms == 1500 and cfg.endgame_worlds == 2 and cfg.endgame_max_side == 0


def test_env(monkeypatch):
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME", "Nash")
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME_ALIVE", "4")
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME_SIDE", "3")
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME_MS", "900")
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME_WORLDS", "3")
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME_CAP_MS", "3000")
    cfg = SearchConfig.from_env()
    assert (cfg.endgame, cfg.endgame_max_alive, cfg.endgame_max_side) == ("nash", 4, 3)
    assert (cfg.endgame_time_ms, cfg.endgame_worlds, cfg.endgame_cap_ms) == (900, 3, 3000)
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME", "bogus")
    assert SearchConfig.from_env().endgame == "off"
    monkeypatch.setenv("SHOWDOWN_SEARCH_ENDGAME", "on")
    assert SearchConfig.from_env().endgame == "safest"


def test_spec_keys():
    from showdown_bot import league

    spec = league.parse_spec(
        "n=search:scale=1.5:endgame=SAFEST:eg_alive=4:eg_side=3:eg_ms=900:eg_worlds=3:eg_cap=3500"
    )
    fields = league.search_config_fields(spec)
    assert fields["endgame"] == "safest"
    assert fields["endgame_max_alive"] == 4 and fields["endgame_max_side"] == 3
    assert fields["endgame_time_ms"] == 900 and fields["endgame_worlds"] == 3
    assert fields["endgame_cap_ms"] == 3500
    assert "endgame" not in league.search_config_fields(league.parse_spec("n=search"))
    for bad in ("n=search:endgame=maybe", "n=search:eg_alive=0", "n=search:eg_ms=-5",
                "n=search:eg_side=-1"):
        with pytest.raises(ValueError):
            league.parse_spec(bad)
    from showdown_bot import ladder_ab

    cfg = ladder_ab.arm_search_config(spec)
    assert cfg.endgame == "safest" and cfg.endgame_time_ms == 900


def test_player_validates(tmp_path):
    kw = dict(battle_format="gen9randombattle", start_listening=False, log_level=50)
    with pytest.raises(ValueError):
        SearchPlayer(config=SearchConfig(endgame="weird"), **kw)
    with pytest.raises(ValueError):
        SearchPlayer(config=SearchConfig(endgame="safest", endgame_max_alive=0), **kw)
    p = SearchPlayer(endgame="safest", **kw)
    assert p.config.endgame == "safest"


# --- trigger and time plan --------------------------------------------------------------


def mon(fainted=False):
    return SimpleNamespace(fainted=fainted)


def fake_battle(ours_alive, ours_fainted, seen_alive, seen_fainted, size=6):
    team = {f"p1: {i}": mon(i >= ours_alive) for i in range(ours_alive + ours_fainted)}
    opp = {f"p2: {i}": mon(i >= seen_alive) for i in range(seen_alive + seen_fainted)}
    return SimpleNamespace(team=team, opponent_team=opp, opponent_team_size=size)


def test_alive_counts_unseen_count_as_alive():
    assert eg.alive_counts(fake_battle(2, 4, 2, 4)) == (2, 2)
    assert eg.alive_counts(fake_battle(2, 4, 1, 1)) == (2, 1 + 4)  # 4 unseen
    assert eg.alive_counts(fake_battle(2, 0, 1, 1, size=2)) == (2, 1)


def test_trigger():
    cfg = SearchConfig(endgame="safest", endgame_max_alive=5)
    assert eg.triggered(cfg, 3, 2) and eg.triggered(cfg, 1, 1)
    assert not eg.triggered(cfg, 3, 3)
    assert not eg.triggered(SearchConfig(), 1, 1)  # off
    side = SearchConfig(endgame="nash", endgame_max_alive=6, endgame_max_side=2)
    assert eg.triggered(side, 2, 2) and not eg.triggered(side, 3, 2) and not eg.triggered(side, 2, 3)


def test_plan_time_clamps_to_cap():
    cfg = SearchConfig(endgame="safest", endgame_time_ms=1500, endgame_worlds=2, endgame_cap_ms=4000)
    assert eg.plan_time(cfg, pool_size=2) == (2, 1500)  # one wave
    assert eg.plan_time(cfg, pool_size=1) == (2, 1500)  # two waves: 3000 + 1000 margin <= 4000
    assert eg.plan_time(cfg, pool_size=1, cap_ms=3000) == (2, 1000)  # clamped
    assert eg.plan_time(cfg, pool_size=4) == (2, 1500)
    assert eg.plan_time(cfg, 1, cap_ms=1040) is None  # nothing useful left
    one = SearchConfig(endgame="safest", endgame_worlds=1, endgame_time_ms=800)
    assert eg.plan_time(one, 1) == (1, 800)


def test_plan_time_never_exceeds_cap():
    for pool in (1, 2, 3):
        for worlds in (1, 2, 5):
            for cap in (1100, 2000, 5000):
                cfg = SearchConfig(endgame="safest", endgame_worlds=worlds, endgame_time_ms=3000)
                plan = eg.plan_time(cfg, pool, cap_ms=cap)
                if plan is not None:
                    n, ms = plan
                    assert math.ceil(n / pool) * ms + eg.DEFAULT_MARGIN_MS <= cap


# --- aggregation math -------------------------------------------------------------------


def test_parse_world_degenerate():
    assert eg.parse_world(None) is None
    assert eg.parse_world(result([], [], [])) is None
    assert eg.parse_world(result(["a"], ["x", "y"], [1.0])) is None  # wrong size
    assert eg.parse_world(result(["a"], ["x"], [NAN])) is None  # nothing evaluated


def test_row_worst_ignores_nan_and_drops_empty_rows():
    w = world(["a", "b", "c"], ["x", "y"], [10.0, -5.0, 3.0, NAN, NAN, NAN])
    assert w.row_worst() == {"a": -5.0, "b": 3.0}  # c all-NaN: missing


def test_safest_single_world_picks_best_worst_case():
    w = world(["a", "b"], ["x", "y"], [100.0, -200.0, 20.0, 10.0])
    policy, _ = eg.aggregate(eg.SAFEST, [w])
    assert max(policy, key=policy.get) == "b"
    assert policy["b"] == pytest.approx(eg.win_prob(10.0))


def test_safest_multi_world_average_and_missing_options():
    w1 = world(["a", "b"], ["x"], [300.0, 0.0])
    w2 = world(["a", "b"], ["x"], [-100.0, 50.0])
    values = eg.safest_values([w1, w2])
    assert values["a"] == pytest.approx((eg.win_prob(300) + eg.win_prob(-100)) / 2)
    assert max(values, key=values.get) == "a"  # mean win prob, not mean eval
    # option "c" only exists in one world: averaged over the worlds that have it
    w3 = world(["a", "c"], ["x"], [0.0, 400.0])
    values = eg.safest_values([w1, w3])
    assert values["c"] == pytest.approx(eg.win_prob(400))
    assert values["a"] == pytest.approx((eg.win_prob(300) + eg.win_prob(0)) / 2)


def test_safest_pruned_row_cannot_beat_exact_best():
    # row b was pruned after one cell (<= best row's worst case): NaN rest is ignored
    w = world(["a", "b"], ["x", "y"], [50.0, 40.0, 30.0, NAN])
    policy, _ = eg.aggregate(eg.SAFEST, [w])
    assert max(policy, key=policy.get) == "a"


def test_nash_matching_pennies_is_mixed_and_averaged():
    w = world(["h", "t"], ["h", "t"], [200.0, -200.0, -200.0, 200.0])
    policy, info = eg.aggregate(eg.NASH, [w, w])
    assert policy["h"] == pytest.approx(0.5, abs=1e-6) and policy["t"] == pytest.approx(0.5, abs=1e-6)
    assert len(info["nash"]) == 2 and info["nash"][0]["exploitability"] < 1e-6


def test_nash_dominant_row_and_multi_world():
    w1 = world(["a", "b"], ["x", "y"], [100.0, 90.0, 0.0, -50.0])  # a dominates
    w2 = world(["a", "b"], ["x", "y"], [10.0, 20.0, 300.0, 250.0])  # b dominates
    policy, _ = eg.aggregate(eg.NASH, [w1, w2])
    assert policy["a"] == pytest.approx(0.5, abs=1e-6) and policy["b"] == pytest.approx(0.5, abs=1e-6)
    only1, _ = eg.aggregate(eg.NASH, [w1])
    assert only1["a"] == pytest.approx(1.0, abs=1e-6)


def test_nash_handles_nan_cells():
    # b's second cell was pruned: filled with its smallest evaluated cell (-300)
    w = world(["a", "b"], ["x", "y"], [50.0, 60.0, -300.0, NAN])
    policy, info = eg.aggregate(eg.NASH, [w])
    assert policy["a"] == pytest.approx(1.0, abs=1e-6)
    # an all-NaN row is dropped, not solved with garbage
    w2 = world(["a", "b"], ["x"], [50.0, NAN])
    policy2, _ = eg.aggregate(eg.NASH, [w2])
    assert set(policy2) == {"a"}


def test_aggregate_empty_means_fallback():
    assert eg.aggregate(eg.SAFEST, []) == ({}, {})
    assert eg.aggregate(eg.NASH, []) == ({}, {})


def test_world_record_shape():
    w = world(["a", "b"], ["x"], [1.0, NAN], depth=7)
    rec = eg.world_record("STATE", w)
    assert rec["total_visits"] == 0 and rec["avg_score"] is None and rec["moves"] == []
    assert rec["endgame"]["depth_searched"] == 7 and rec["endgame"]["matrix"] == [1.0, None]
    assert rec["endgame"]["options"] == {"a": 1.0}


# --- choice names map to legal orders (tera / switch) ----------------------------------


def test_policy_names_map_to_orders_incl_tera_and_switch():
    """The expectiminimax option strings are the MCTS ones; the argmax over legal ones wins."""
    from battle_helpers import event, make_battle, opp_switch, request_for
    from showdown_bot import engine_state
    from showdown_bot.sets import RandbatsData

    data = RandbatsData.load(cache_dir=Path(__file__).parent / "fixtures")
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    p = SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                     data=data, endgame="safest", seed=1)
    trackers = p.opponent_model.update(battle)
    _slots, strings, built = p._sample_worlds(battle, trackers, 1)
    from showdown_bot.players import _worker

    raw = _worker.endgame_world(strings[0], 60)
    w = eg.parse_world(raw)
    assert w is not None and w.depth >= 1
    assert any(n.startswith("switch ") for n in w.ours)
    for name in w.ours:
        order = built.choice_to_order(name, battle)
        if name.endswith("-tera"):
            assert order is None or order.terastallize
        else:
            assert order is not None, name
    # an illegal best option falls through to the next best legal one
    names = [n for n in w.ours if built.choice_to_order(n, battle) is not None]
    policy = {"not-a-move": 9.0, names[0]: 1.0}
    order = p._order_from_policy(policy, built, battle, "argmax")
    assert p.last_choice == names[0] and order is not None


# --- player integration -----------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    from showdown_bot.sets import RandbatsData

    return RandbatsData.load(cache_dir=Path(__file__).parent / "fixtures")


def _battle():
    from battle_helpers import event, make_battle, opp_switch, request_for

    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


def _player(data, pool, **cfg):
    base = dict(search_time_ms=30, worlds=2, endgame="safest", endgame_max_alive=12,
                endgame_time_ms=150, endgame_worlds=2)
    base.update(cfg)
    return SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                        data=data, config=SearchConfig(**base), executor=pool, seed=3)


@pytest.mark.parametrize("mode", ["safest", "nash"])
def test_endgame_decision_end_to_end_smoke(data, mode):
    pool = ThreadPoolExecutor(max_workers=2)
    p = _player(data, pool, endgame=mode)
    order = asyncio.run(p.choose_move(_battle()))
    assert order is not None
    assert p.stats.endgame_decisions == 1 and p.stats.endgame_fallbacks == 0
    assert p.last_changes["endgame"] and len(p.last_changes["depth_searched"]) == 2
    assert p.last_changes["depth_searched"][0] >= 1
    pool.shutdown(wait=False, cancel_futures=True)


def test_not_triggered_uses_mcts(data):
    pool = ThreadPoolExecutor(max_workers=2)
    p = _player(data, pool, endgame_max_alive=5)  # 12 alive: no endgame
    asyncio.run(p.choose_move(_battle()))
    assert p.stats.endgame_decisions == 0 and p.stats.endgame_fallbacks == 0
    assert not p.last_changes.get("endgame")
    pool.shutdown(wait=False, cancel_futures=True)


def test_endgame_error_falls_back_to_mcts(data, monkeypatch):
    from showdown_bot.players import _worker

    def boom(state, ms):
        raise RuntimeError("engine panic: test")

    monkeypatch.setattr(_worker, "endgame_world", boom)
    pool = ThreadPoolExecutor(max_workers=2)
    p = _player(data, pool)
    order = asyncio.run(p.choose_move(_battle()))
    assert order is not None
    assert p.stats.endgame_decisions == 0 and p.stats.endgame_fallbacks == 1
    assert p.stats.fallbacks == 0  # MCTS answered, not the random fallback
    assert p.stats.worlds_run >= 1
    pool.shutdown(wait=False, cancel_futures=True)


def test_degenerate_result_falls_back_to_mcts(data, monkeypatch):
    from showdown_bot.players import _worker

    monkeypatch.setattr(_worker, "endgame_world", lambda s, ms: ([], [], [], 0))
    pool = ThreadPoolExecutor(max_workers=2)
    p = _player(data, pool)
    assert asyncio.run(p.choose_move(_battle())) is not None
    assert p.stats.endgame_fallbacks == 1 and p.stats.endgame_decisions == 0
    pool.shutdown(wait=False, cancel_futures=True)


def test_endgame_decision_is_recorded(data, tmp_path):
    from showdown_bot import recorder as recorder_module

    pool = ThreadPoolExecutor(max_workers=2)
    rec = recorder_module.GameRecorder(tmp_path, truth=False)
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=30, worlds=2, endgame="safest", endgame_max_alive=12,
                            endgame_time_ms=150),
        executor=pool, seed=3, recorder=rec,
    )
    battle = _battle()
    asyncio.run(p.choose_move(battle))
    lines = rec._games[(battle.battle_tag, p.username)]
    decision = next(line for line in lines if line.get("type") == "decision")
    assert decision["endgame"] is True and decision["endgame_mode"] == "safest"
    assert decision["endgame_alive"] == [6, 6] or sum(decision["endgame_alive"]) >= 2
    assert len(decision["endgame_depth"]) == 2 and decision["endgame_values"]
    assert decision["worlds"][0]["endgame"]["depth_searched"] >= 1
    assert decision["worlds"][0]["total_visits"] == 0
    assert decision["chosen"] in decision["endgame_values"]
    pool.shutdown(wait=False, cancel_futures=True)


# --- engine-only A/B harness --------------------------------------------------------------


def test_swap_sides_roundtrip_and_alive_counts():
    pytest.importorskip("poke_engine")
    from showdown_bot import endgame_eval as ev

    assert ev.swap_sides("a/b/c/d") == "b/a/c/d"
    assert ev.swap_sides(ev.swap_sides("a/b/c/d")) == "a/b/c/d"


def test_ab_summary_and_pairing():
    from showdown_bot import endgame_eval as ev

    results = [
        {"ok": True, "index": 0, "score": 1.0}, {"ok": True, "index": 0, "score": 0.5},
        {"ok": True, "index": 1, "score": 0.0}, {"ok": True, "index": 1, "score": 0.0},
        {"ok": True, "index": 2, "score": 1.0},  # unpaired: dropped
        {"ok": False, "index": 3},
    ]
    pairs = ev.paired_scores(results)
    assert pairs == {0: 0.75, 1: 0.0}
    s = ev.summarize(list(pairs.values()))
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.375)
    assert s["lo"] <= s["mean"] <= s["hi"] and ev.summarize([])["n"] == 0
    assert ev._score("win", 1) == 1.0 and ev._score("win", 2) == 0.0
    assert ev._score("loss", 2) == 1.0 and ev._score("tie", 1) == 0.5


def test_ab_smoke_real_engine():
    pytest.importorskip("poke_engine")
    import glob

    from showdown_bot import endgame_eval as ev

    sources = glob.glob(str(Path(__file__).parents[1] / "data" / "selfplay" / "gen*"))
    positions = ev.extract_positions(sources, 1, 5, random.Random(0)) if sources else []
    if not positions:
        pytest.skip("no self-play records with truth states")
    a = SearchConfig(search_time_ms=20, endgame="safest", endgame_time_ms=60)
    b = SearchConfig(search_time_ms=20)
    g = ev.play_game(positions[0]["state"], 1, a, b, random.Random(0), max_turns=6)
    assert g.score in (0.0, 0.5, 1.0) and g.a_endgame_moves >= 1
    assert select_choice({"x": 1.0}, "argmax", random.Random(0)) == "x"
    assert np.isfinite(g.seconds)
