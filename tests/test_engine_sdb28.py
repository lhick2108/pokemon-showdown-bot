"""Engine sdb.28 bindings and state conversion: Harvest's last_item, Disable durations, formes."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData, SampledSet  # noqa: E402

pytestmark = pytest.mark.skipif(es._engine_sdb_level() < 28, reason="engine < sdb.28")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


def _mon(**kw):
    base = dict(id="tropius", level=88, hp=300, maxhp=300, ability="harvest", base_ability="harvest",
                moves=[pe.Move(id="splash", pp=16)])
    base.update(kw)
    return pe.Pokemon(**base)


def test_last_item_round_trips_through_the_binding_and_serialization():
    mon = _mon(last_item="sitrusberry")
    assert mon.last_item.lower() == "sitrusberry"
    assert _mon().last_item.lower() == "none"
    state = pe.State(side_one=pe.Side(pokemon=[mon]), side_two=pe.Side(pokemon=[_mon(id="snorlax")]))
    back = pe.State.from_string(state.to_string())
    assert back.side_one.pokemon[0].last_item.lower() == "sitrusberry"
    # unset last_item keeps the pre-sdb.28 serialization
    plain = pe.State(side_one=pe.Side(pokemon=[_mon()]), side_two=pe.Side(pokemon=[_mon(id="snorlax")]))
    assert ",sitrusberry" not in plain.to_string()


def test_rebuild_pokemon_keeps_last_item():
    rebuilt = es._rebuild_pokemon(_mon(last_item="sitrusberry"), hp=10)
    assert rebuilt.last_item.lower() == "sitrusberry" and rebuilt.hp == 10


def test_harvest_regrows_the_berry_in_half_the_branches():
    mon = _mon(hp=100, last_item="sitrusberry", speed=200)
    foe = _mon(id="snorlax", ability="none", base_ability="none", speed=50)
    state = pe.State(side_one=pe.Side(pokemon=[mon]), side_two=pe.Side(pokemon=[foe]))
    outcomes = pe.generate_instructions(state, "splash", "splash")
    items = sorted(str(state.apply_instructions(o).side_one.pokemon[0].item).lower() for o in outcomes)
    assert len(outcomes) == 2
    # regrown Sitrus is eaten again at once (100/300 <= 1/2): both branches end itemless, one healed
    hps = sorted(state.apply_instructions(o).side_one.pokemon[0].hp for o in outcomes)
    assert items == ["none", "none"] and hps == [100, 175]


def test_disable_duration_binding():
    d = pe.VolatileStatusDurations(disable=13)
    assert d.disable == 13
    assert pe.VolatileStatusDurations().disable == 0


def test_opponent_harvest_holder_with_an_eaten_berry_gets_last_item(data, sampler):
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Tropius", "Tropius, L88")
    event(battle, "-enditem", "p2a: Tropius", "Sitrus Berry", "[eat]")
    trackers = OpponentModel(data).update(battle)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(1))
    slots[0].sampled = SampledSet("tropius", 88, "", ["dragondance", "leafblade", "earthquake", "synthesis"],
                                  "harvest", "Sitrus Berry", "Steel")
    state = es.build_state(battle, slots).state
    mon = state.side_two.pokemon[0]
    assert str(mon.item).lower() == "none"
    assert mon.last_item.lower() == "sitrusberry"


def test_terapagos_switch_in_takes_its_terastal_forme():
    terapagos = pe.Pokemon(id="terapagos", level=77, hp=235, maxhp=265, ability="terashift", base_ability="terashift",
                           evs=(85, 85, 85, 85, 85, 85), attack=153, defense=184, special_attack=153,
                           special_defense=184, speed=138, moves=[pe.Move(id="splash", pp=16)])
    lead = _mon(id="snorlax", ability="none", base_ability="none")
    foe = _mon(id="snorlax", ability="none", base_ability="none")
    state = pe.State(side_one=pe.Side(pokemon=[lead, terapagos]), side_two=pe.Side(pokemon=[foe]))
    outcomes = pe.generate_instructions(state, "terapagos", "splash")
    after = state.apply_instructions(outcomes[0]).side_one.pokemon[1]
    assert after.id.lower() == "terapagosterastal"
    assert after.ability.lower() == "terashell"
    assert after.maxhp == 273 and after.hp == 243
