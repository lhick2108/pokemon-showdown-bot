"""Stall resilience: hard cap on a decision's pool search, pool recycling, stall dumps (offline)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from showdown_bot import league
from showdown_bot.players import _worker
from showdown_bot.players import search as search_module

from test_endgame import _battle, _player, data  # noqa: F401  (fixture re-export)


def _fake_search(state, ms, threads, *rest):
    if state == "HANG":
        time.sleep(5)
    return (100, [("move a", 60, 30.0), ("move b", 40, 10.0)])


@pytest.fixture
def fast_cap(monkeypatch):
    monkeypatch.setattr(search_module, "_CAP_SLACK_S", 0.0)  # cap = planned wave time + 0
    monkeypatch.setattr(_worker, "search_world", _fake_search)


def test_partial_results_used_when_one_world_hangs(data, fast_cap):  # noqa: F811
    pool = ThreadPoolExecutor(max_workers=4)
    p = _player(data, pool)
    started = time.time()
    results = asyncio.run(p._run_worlds_on(pool, ["ok1", "HANG", "ok2"], 100))
    assert time.time() - started < 4.5
    assert len(results) == 2 and p._result_worlds == [0, 2]
    assert p.stats.decision_timeouts == 1 and p.stats.world_errors == 0
    assert "decision_timeouts=1" in p.stats.summary()
    pool.shutdown(wait=False, cancel_futures=True)


def test_all_worlds_hung_raises_and_decision_falls_back(data, fast_cap):  # noqa: F811
    pool = ThreadPoolExecutor(max_workers=4)
    p = _player(data, pool, endgame="off")
    battle = _battle()
    p._sample_worlds = lambda b, t, n: ([], ["HANG", "HANG"], object())
    started = time.time()
    order = asyncio.run(p.choose_move(battle))
    assert time.time() - started < 4.5
    assert order is not None  # a legal fallback move, not a hang
    assert p.stats.fallbacks == 1 and p.stats.decision_timeouts == 1
    assert p.stats.fallback_reasons["TimeoutError"] == 1
    pool.shutdown(wait=False, cancel_futures=True)


def test_shared_pool_is_recycled_and_player_follows_the_new_one(data, fast_cap, monkeypatch):  # noqa: F811
    old = ThreadPoolExecutor(max_workers=4)
    new = ThreadPoolExecutor(max_workers=4)
    monkeypatch.setattr(search_module, "_POOLS", {7: old})
    calls = []

    def fake_reset(workers, only=None, kill=False):
        calls.append((workers, only is old, kill))
        search_module._POOLS[workers] = new

    monkeypatch.setattr(search_module, "reset_pool", fake_reset)
    p = _player(data, old)
    assert p._shared_workers == 7
    asyncio.run(p._run_worlds_on(old, ["ok1", "HANG"], 100))
    assert calls == [(7, True, True)]
    assert p._pick_executor() is new  # later decisions go to the rebuilt pool
    old.shutdown(wait=False, cancel_futures=True)
    new.shutdown(wait=False, cancel_futures=True)


def test_inflight_registry_and_pool_diagnostics(data, fast_cap):  # noqa: F811
    pool = ThreadPoolExecutor(max_workers=2)
    p = _player(data, pool)
    seen = []
    real = _worker.search_world

    def spy(*args):
        time.sleep(0.2)  # let the awaiting side register the search first
        seen.append(search_module.pool_diagnostics())
        return real(*args)

    _worker.search_world = spy
    try:
        asyncio.run(p._run_worlds_on(pool, ["ok1"], 100))
    finally:
        _worker.search_world = real
    assert "inflight mcts" in seen[0] and "worlds=1" in seen[0]
    assert not search_module._INFLIGHT  # cleaned up afterwards
    assert "no pool searches in flight" in search_module.pool_diagnostics()
    pool.shutdown(wait=False, cancel_futures=True)


def test_stall_dump_contains_stacks_and_pool_state(caplog):
    text = league.dump_stall(200)
    assert "STALL: no game finished for 200s" in text
    assert "--- thread" in text and "test_stall_dump_contains" in text
    assert "pool" in text


def test_progress_ticker_dumps_when_no_game_finishes(monkeypatch):
    dumps = []
    monkeypatch.setattr(league, "dump_stall", lambda quiet: dumps.append(quiet))

    async def run():
        ticker = asyncio.ensure_future(
            league._progress(league.Tracker(), 10, interval=0.05, heartbeat=100.0, stall_s=0.2)
        )
        await asyncio.sleep(0.7)
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)

    asyncio.run(run())
    assert 2 <= len(dumps) <= 4  # repeats once per stall period, not every tick


def test_sample_moves_terminates_on_unaddable_move(data):  # noqa: F811
    """A role move with an empty id can never be added; the loop must still end."""
    from types import SimpleNamespace

    from showdown_bot.sets import SetTracker

    tracker = SetTracker.__new__(SetTracker)
    tracker._binding_revealed_moves = []
    role = SimpleNamespace(moves=["", "Tackle"], weights={"moves": {"tackle": 0.5}})
    import random

    assert SetTracker._sample_moves(tracker, role, random.Random(1)) == ("Tackle",)
