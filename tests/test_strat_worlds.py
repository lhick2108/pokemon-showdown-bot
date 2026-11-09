"""Stratified world sampling (589.54) and the randbats cache max-age refresh (589.60)."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("poke_engine")

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot import league, sets  # noqa: E402
from showdown_bot.players.search import SearchConfig  # noqa: E402
from showdown_bot.sets import RandbatsData, SetTracker  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _two_set_tracker(data, monkeypatch, w_a, w_b):
    tracker = SetTracker(data, "gardevoir")
    base = data.full_sets("gardevoir")[0]
    a = dataclasses.replace(base, count=w_a, moves=("a1", "a2", "a3", "a4"))
    b = dataclasses.replace(base, count=w_b, moves=("b1", "b2", "b3", "b4"))
    monkeypatch.setattr(tracker, "_matching_full_sets", lambda: [("gardevoir", a), ("gardevoir", b)])
    monkeypatch.setattr(tracker, "_best_matching_role_name", lambda sid, fs: fs.moves[0][0])
    return tracker


def _a_counts(tracker, seed, k=4):
    quantiles = es.stratified_quantiles(["x"], k, random.Random(seed))
    picks = Counter(tracker.sample(random.Random(0), q["x"]).moves[0] for q in quantiles)
    assert picks["a1"] + picks["b1"] == k
    return picks["a1"]


def test_allocation_counts(data, monkeypatch):
    t = _two_set_tracker(data, monkeypatch, 53, 47)
    assert {_a_counts(t, s) for s in range(60)} == {2, 3}  # never 0, 1 or 4
    assert Counter(_a_counts(t, s) for s in range(300))[2] > 250  # 3 only if offset < 0.03
    t = _two_set_tracker(data, monkeypatch, 75, 25)
    assert {_a_counts(t, s) for s in range(60)} == {3}
    t = _two_set_tracker(data, monkeypatch, 50, 50)
    assert {_a_counts(t, s) for s in range(60)} == {2}


def test_unbiased_on_average(data, monkeypatch):
    t = _two_set_tracker(data, monkeypatch, 53, 47)
    total = sum(_a_counts(t, s) for s in range(400))
    assert total / (400 * 4) == pytest.approx(0.53, abs=0.03)


def test_quantiles_deterministic_and_cover_grid():
    a = es.stratified_quantiles(["x", "y"], 4, random.Random(7))
    b = es.stratified_quantiles(["x", "y"], 4, random.Random(7))
    assert a == b
    xs = sorted(q["x"] for q in a)
    assert [int(u * 4) for u in xs] == [0, 1, 2, 3]  # one point per stratum


def test_off_leaves_sampling_unchanged(data):
    t1, t2 = SetTracker(data, "gardevoir"), SetTracker(data, "gardevoir")
    r1, r2 = random.Random(3), random.Random(3)
    assert [t1.sample(r1) for _ in range(6)] == [t2.sample(r2, None) for _ in range(6)]


def test_sample_opponent_team_quantiles(data):
    from battle_helpers import event, make_battle, opp_switch, request_for
    from showdown_bot.sets import OpponentModel

    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    trackers = OpponentModel(data).update(battle)
    sampler = es.TeamSampler(data)
    a = es.sample_opponent_team(battle, trackers, sampler, random.Random(5))
    b = es.sample_opponent_team(battle, trackers, sampler, random.Random(5), quantiles=None)
    assert [s.sampled for s in a] == [s.sampled for s in b]
    key = str(next(iter(battle.opponent_team)))
    c = es.sample_opponent_team(battle, trackers, sampler, random.Random(5), quantiles={key: 0.1})
    d = es.sample_opponent_team(battle, trackers, sampler, random.Random(5), quantiles={key: 0.1})
    assert c[0].sampled == d[0].sampled


def test_spec_key_and_default_off():
    kw = league._quality_kwargs(league.parse_spec("search:strat=1:belief=1"))
    assert kw["strat_worlds"] is True and kw["belief_reweight"] is True
    assert "strat_worlds" not in league._quality_kwargs(league.parse_spec("search"))
    assert SearchConfig().strat_worlds is False
    with pytest.raises(ValueError):
        league.parse_spec("search:strat=2")


# --- 589.60 -----------------------------------------------------------------


class _Resp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _old(path, hours):
    t = time.time() - hours * 3600
    os.utime(path, (t, t))


def test_stale_cache_is_refreshed(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    path.write_text('{"old": 1}')
    _old(path, 48)
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: _Resp(b'{"new": 2}'))
    assert sets._read_cached_json(path, "http://u", False, 24 * 3600) == {"new": 2}


def test_fresh_cache_not_refreshed(tmp_path, monkeypatch):
    path = tmp_path / "x.json"
    path.write_text('{"old": 1}')

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert sets._read_cached_json(path, "http://u", False, 24 * 3600) == {"old": 1}
    _old(path, 480)
    assert sets._read_cached_json(path, "http://u", False, None) == {"old": 1}


@pytest.mark.parametrize("failure", ["oserror", "badjson"])
def test_stale_cache_survives_network_failure(tmp_path, monkeypatch, failure):
    path = tmp_path / "x.json"
    path.write_text('{"old": 1}')
    _old(path, 48)

    def fake(req, timeout=0):
        if failure == "oserror":
            raise OSError("offline")
        return _Resp(b"<html>oops")

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert sets._read_cached_json(path, "http://u", False, 24 * 3600) == {"old": 1}
    assert json.loads(path.read_text()) == {"old": 1}


def test_missing_cache_still_downloads(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "x.json"
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=0: _Resp(b"[1]"))
    assert sets._read_cached_json(path, "http://u", False, None) == [1]


def test_load_uses_module_default_max_age(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sets, "MAX_CACHE_AGE_SECONDS", 3600.0)
    monkeypatch.setattr(sets, "_read_cached_json", lambda p, u, r, m=None: calls.append(m) or {})
    try:
        RandbatsData.load(cache_dir=tmp_path)
    except Exception:
        pass
    assert calls and set(calls) == {3600.0}
