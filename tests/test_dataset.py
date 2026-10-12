"""Dataset loader and State-string round trip (no server needed)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402
from test_recorder import fake_world, write_fake_game  # noqa: E402

from showdown_bot import dataset, engine_state, recorder  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def state_string():
    data = RandbatsData.load(cache_dir=FIXTURES)
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    trackers = OpponentModel(data).update(battle)
    slots = engine_state.sample_opponent_team(
        battle, trackers, engine_state.TeamSampler(data), random.Random(0)
    )
    return engine_state.build_state(battle, slots).to_string()


def test_state_string_round_trip_gives_identical_features(state_string):
    state = pe.State.from_string(state_string)
    again = pe.State.from_string(state.to_string())
    for side in (1, 2):
        assert pe.state_features(state, side) == pe.state_features(again, side)
    assert len(pe.state_features(state, 1)) == pe.FEATURE_LENGTH


def _write(tmp_path, state, result, tag):
    rec = recorder.GameRecorder(tmp_path)
    rec.start_game(tag, "u")
    rec.add_decision(tag, "u", {
        "turn": 1, "worlds": [fake_world(state)], "policy": {"a": 0.6, "b": 0.4},
        "chosen": "a", "truth": {"state": state}, "truth_status": "ok",
    })
    rec.end_game(tag, "u", result, turns=2)


def test_loader_targets_are_signed_for_both_sides(tmp_path, state_string):
    _write(tmp_path, state_string, recorder.RESULT_WIN, "w")
    _write(tmp_path, state_string, recorder.RESULT_LOSS, "l")
    _write(tmp_path, state_string, recorder.RESULT_TIE, "t")
    samples = list(dataset.iter_samples(tmp_path, sides=(1, 2)))
    assert len(samples) == 3 * 2 * 2  # games x (world + truth) x sides
    by = {(s.battle_tag, s.side, s.kind): s for s in samples}
    assert by["w", 1, "world"].value == 1.0 and by["w", 2, "world"].value == 0.0
    assert by["l", 1, "truth"].value == 0.0 and by["l", 2, "truth"].value == 1.0
    assert by["t", 1, "world"].value == 0.5 and by["t", 2, "world"].value == 0.5
    state = pe.State.from_string(state_string)
    assert by["w", 1, "world"].features == pe.state_features(state, 1)
    assert by["w", 2, "world"].features == pe.state_features(state, 2)
    assert by["w", 1, "world"].policy == {"a": 0.6, "b": 0.4}
    assert by["w", 2, "world"].policy is None


def test_default_triples_are_side_one_and_bad_states_are_counted(tmp_path, state_string):
    _write(tmp_path, state_string, recorder.RESULT_WIN, "w")
    write_fake_game(recorder.GameRecorder(tmp_path), tag="bad")  # "S1": engine cannot parse
    stats = dataset.LoadStats()
    triples = list(dataset.iter_triples(tmp_path, stats=stats))
    assert len(triples) == 2  # the good game's world + truth
    features, value, policy = triples[0]
    assert len(features) == pe.FEATURE_LENGTH and value == 1.0 and policy
    assert stats.bad_states == 3 and stats.games == 2
    assert dataset.collect_statistics(tmp_path)["decisions"] == 2


def _write_q(tmp_path, state, worlds, result=recorder.RESULT_WIN, tag="q"):
    rec = recorder.GameRecorder(tmp_path)
    rec.start_game(tag, "u")
    rec.add_decision(tag, "u", {
        "turn": 1, "worlds": [recorder.world_record(state, w) for w in worlds],
        "policy": {"a": 1.0}, "chosen": "a", "truth": {"state": state}, "truth_status": "ok",
    })
    rec.end_game(tag, "u", result, turns=2)


def test_search_value_decision_mean_world_and_side_flip(tmp_path, state_string):
    worlds = [
        (100, [("a", 100, 70.0)]),  # q = 0.7
        (50, [("a", 50, 10.0)]),    # q = 0.2
        (0, []),                    # zero visits: ignored
    ]
    _write_q(tmp_path, state_string, worlds)
    dec = list(dataset.iter_samples(tmp_path, sides=(1, 2)))
    assert len(dec) == (3 + 1) * 2
    one = [s for s in dec if s.side == 1]
    two = [s for s in dec if s.side == 2]
    assert all(abs(s.search_value - 0.45) < 1e-9 for s in one)  # mean(0.7, 0.2)
    assert all(abs(s.search_value - 0.55) < 1e-9 for s in two)
    assert [s.kind for s in one] == ["world"] * 3 + ["truth"]
    wor = [s for s in dataset.iter_samples(tmp_path, q_source="world")]
    assert [None if s.search_value is None else round(s.search_value, 6) for s in wor] == [
        0.7, 0.2, None, 0.45]  # zero-visit world has no q of its own; truth keeps the mean
    with pytest.raises(ValueError):
        list(dataset.iter_samples(tmp_path, q_source="bogus"))


def test_search_value_none_when_no_world_has_visits(tmp_path, state_string):
    _write_q(tmp_path, state_string, [(0, [])])
    assert all(s.search_value is None for s in dataset.iter_samples(tmp_path))
    assert dataset.decision_search_value({"worlds": [{"avg_score": None, "total_visits": 5}]}) is None
    assert dataset.decision_search_value({}) is None


def test_absolute_search_value_pure_function():
    import math
    # relative score 0.5 (no search improvement): q_abs is sigmoid(scale * eval)
    assert abs(dataset.absolute_search_value(0.0, 0.5) - 0.5) < 1e-12
    assert abs(dataset.absolute_search_value(80.0, 0.5) - 1 / (1 + math.exp(-1.0))) < 1e-9
    # eval and search improvement add in logit space
    q = dataset.absolute_search_value(-80.0, 1 / (1 + math.exp(-1.0)))
    assert abs(q - 0.5) < 1e-9
    # clipping keeps 0/1 scores finite
    assert 0.0 < dataset.absolute_search_value(0.0, 0.0) < dataset.absolute_search_value(0.0, 1.0) < 1.0


def test_search_value_abs_in_samples(tmp_path, state_string):
    _write_q(tmp_path, state_string, [(100, [("a", 100, 70.0)])])
    ev = pe.evaluate(pe.State.from_string(state_string))
    expect = dataset.absolute_search_value(ev, 0.7)
    s1, s2 = [list(dataset.iter_samples(tmp_path, sides=(1, 2), truth=False))[i] for i in (0, 1)]
    assert abs(s1.search_value_abs - expect) < 1e-9 and abs(s2.search_value_abs - (1 - expect)) < 1e-9
    assert s1.stock_eval == ev
    truth = [s for s in dataset.iter_samples(tmp_path) if s.kind == "truth"][0]
    assert abs(truth.search_value_abs - expect) < 1e-9
