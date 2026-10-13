"""Ditto / Imposter: the engine copies on switch-in; transformed Dittos carry the copy."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("poke_engine")

import poke_engine as pe  # noqa: E402
from battle_helpers import (  # noqa: E402
    OUR_TEAM,
    event,
    make_battle,
    opp_switch,
    request_for,
)

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402
from test_engine_state import FIXTURES  # noqa: E402

needs_imposter = pytest.mark.skipif(
    not es._ENGINE_TRANSFORMS, reason="installed poke_engine predates Imposter (sdb.3)"
)


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


def _build(battle, data, sampler, seed=0):
    trackers = OpponentModel(data).update(battle)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(seed))
    return es.build_state(battle, slots)


def _ids(pokemon):
    return [m.id.lower() for m in pokemon.moves]


def _ditto_battle():
    """Opp Gardevoir active, Ditto revealed in the back; Great Tusk is +2 Atk."""
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Ditto", "Ditto, L100")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    event(battle, "-boost", "p1a: Great Tusk", "atk", "2")
    return battle


# --- an already-Transformed Ditto carries the copied kit ----------------------


def test_opponent_ditto_that_transformed_carries_our_kit(data, sampler):
    battle = make_battle(request_for())
    event(battle, "start")
    event(battle, "-boost", "p1a: Great Tusk", "atk", "2")
    opp_switch(battle, "Ditto", "Ditto, L100")
    event(battle, "-transform", "p2a: Ditto", "p1a: Great Tusk", "[from] ability: Imposter")
    built = _build(battle, data, sampler)
    ditto = built.state.side_two.pokemon[0]
    tusk = built.state.side_one.pokemon[0]
    assert ditto.id.lower() == "ditto"
    assert set(_ids(ditto)) == {"closecombat", "headlongrush", "knockoff", "rapidspin"}
    assert all(m.pp <= 5 for m in ditto.moves)
    assert (ditto.attack, ditto.defense, ditto.special_attack) == (
        tusk.attack, tusk.defense, tusk.special_attack,
    )
    assert (ditto.special_defense, ditto.speed) == (tusk.special_defense, tusk.speed)
    assert {t.lower() for t in ditto.types} >= {"ground", "fighting"}
    assert built.state.side_two.attack_boost == 2
    assert ditto.ability.lower() == "protosynthesis"
    assert ditto.weight_kg == pytest.approx(tusk.weight_kg)
    if es._ENGINE_TRANSFORMS:
        assert "transformed" in {v.lower() for v in built.state.side_two.volatile_statuses}


def test_our_ditto_that_transformed_carries_the_targets_stats(data, sampler):
    team = [("Ditto", 100, "Choice Scarf", "Imposter", ["Transform"], "Normal")] + OUR_TEAM[1:]
    battle = make_battle(request_for(team))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    event(battle, "-transform", "p1a: Ditto", "p2a: Gardevoir", "[from] ability: Imposter")
    built = _build(battle, data, sampler)
    ditto = built.state.side_one.pokemon[0]
    foe = built.state.side_two.pokemon[0]
    assert ditto.id.lower() == "ditto"
    # The request still reports Ditto's own stats; the engine must use Gardevoir's.
    assert (ditto.attack, ditto.defense, ditto.special_attack) == (
        foe.attack, foe.defense, foe.special_attack,
    )
    assert (ditto.special_defense, ditto.speed) == (foe.special_defense, foe.speed)
    assert {t.lower() for t in ditto.types} >= {"psychic", "fairy"}
    assert ditto.hp == battle.team["p1: Ditto"].current_hp  # HP is not copied
    assert ditto.maxhp != foe.maxhp or ditto.maxhp == battle.team["p1: Ditto"].max_hp


def test_untransformed_ditto_is_left_alone(data, sampler):
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Ditto", "Ditto, L100")
    built = _build(battle, data, sampler)
    ditto = built.state.side_two.pokemon[0]
    assert _ids(ditto)[:1] == ["transform"]
    assert "transformed" not in {v.lower() for v in built.state.side_two.volatile_statuses}


# --- the search position: a revealed Ditto in the back copies our boosts --------


@needs_imposter
def test_forced_switch_to_ditto_copies_our_boosts_and_moves(data, sampler):
    battle = _ditto_battle()
    built = _build(battle, data, sampler)
    state = built.state
    assert state.side_one.attack_boost == 2
    assert "ditto" in [p.id.lower() for p in state.side_two.pokemon]
    outcomes = pe.generate_instructions(state, "knockoff", "ditto")
    assert outcomes
    for outcome in outcomes:
        after = state.apply_instructions(outcome)
        ditto = after.side_two.pokemon[int(after.side_two.active_index)]
        assert ditto.id.lower() == "ditto"
        assert after.side_two.attack_boost == 2, "Imposter must copy our +2 Atk"
        assert set(_ids(ditto)) == {"closecombat", "headlongrush", "knockoff", "rapidspin"}
        assert ditto.attack == state.side_one.pokemon[0].attack
        assert "transformed" in {v.lower() for v in after.side_two.volatile_statuses}
        # reversal restores the untransformed Ditto
        back = after.reverse_instructions(outcome)
        assert back.to_string() == state.to_string()
