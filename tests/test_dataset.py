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
