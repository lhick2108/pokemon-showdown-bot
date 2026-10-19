"""Instant charge moves (Solar Beam in sun, Power Herb) and Choice-lock release on item loss."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData, SampledSet  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
_GARDEVOIR_MOVES = ["moonblast", "psyshock", "calmmind", "healingwish"]


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


def _opp_battle():
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


def _charge_battle(*, sun: bool, our_moves: int | None):
    """Our Great Tusk used Solar Beam (|-prepare|); the request offers 1 or all moves."""
    battle = _opp_battle()
    if sun:
        event(battle, "-weather", "SunnyDay")
    event(battle, "move", "p1a: Great Tusk", "Solar Beam", "", "[still]")
    event(battle, "-prepare", "p1a: Great Tusk", "Solar Beam", "p2a: Gardevoir")
    request = request_for()
    if our_moves == 1:
        request["active"][0]["moves"] = [{"move": "Solar Beam", "id": "solarbeam", "pp": 15, "maxpp": 16}]
    battle.parse_request(request)
    return battle


def _volatiles(battle, who):
    mon = battle.active_pokemon if who == "us" else battle.opponent_active_pokemon
    return es._volatiles_for(mon, battle, is_ours=who == "us")[0]


def test_real_solar_beam_charge_sets_volatile_for_our_side():
    battle = _charge_battle(sun=False, our_moves=1)
    assert "solarbeam" in _volatiles(battle, "us")


def test_instant_solar_beam_in_sun_does_not_force_our_side():
    # Regression: battles 2700048217 T19 / 2700022584 T12 (forced into a 2nd Solar Beam).
    battle = _charge_battle(sun=True, our_moves=None)  # full 4-move request
    assert battle.active_pokemon.preparing_move is not None
    assert "solarbeam" not in _volatiles(battle, "us")


def test_request_wins_over_weather_for_our_side():
    battle = _charge_battle(sun=True, our_moves=1)
    assert "solarbeam" in _volatiles(battle, "us")
    battle = _charge_battle(sun=False, our_moves=None)
    assert "solarbeam" not in _volatiles(battle, "us")


def _opp_charge(move, weather=None, item=None):
    battle = _opp_battle()
    if weather:
        event(battle, "-weather", weather)
    event(battle, "move", "p2a: Gardevoir", move, "", "[still]")
    event(battle, "-prepare", "p2a: Gardevoir", move, "p1a: Great Tusk")
    if item:
        battle.opponent_active_pokemon._item = item
    return battle


def test_opponent_instant_charge_in_matching_weather():
    assert "solarbeam" not in _volatiles(_opp_charge("Solar Beam", "SunnyDay"), "opp")
    assert "solarblade" not in _volatiles(_opp_charge("Solar Blade", "DesolateLand"), "opp")
    assert "electroshot" not in _volatiles(_opp_charge("Electro Shot", "RainDance"), "opp")


def test_opponent_real_charge_without_weather_or_in_wrong_weather():
    assert "solarbeam" in _volatiles(_opp_charge("Solar Beam"), "opp")
    assert "solarbeam" in _volatiles(_opp_charge("Solar Beam", "RainDance"), "opp")
    assert "electroshot" in _volatiles(_opp_charge("Electro Shot", "SunnyDay"), "opp")
    assert "fly" in _volatiles(_opp_charge("Fly", "SunnyDay"), "opp")


def test_utility_umbrella_keeps_opponent_solar_beam_charging():
    battle = _opp_charge("Solar Beam", "SunnyDay", item="utilityumbrella")
    assert "solarbeam" in _volatiles(battle, "opp")


def test_power_herb_consumption_clears_charge():
    battle = _opp_charge("Solar Beam")
    event(battle, "-enditem", "p2a: Gardevoir", "Power Herb")
    assert "solarbeam" not in _volatiles(battle, "opp")


@pytest.mark.parametrize(
    "removal",
    [
        ("-enditem", "p2a: Gardevoir", "Choice Specs", "[from] move: Knock Off", "[of] p1a: Great Tusk"),
        ("-enditem", "p2a: Gardevoir", "Choice Specs", "[from] move: Corrosive Gas", "[of] p1a: Great Tusk"),
        ("-item", "p2a: Gardevoir", "Leftovers", "[from] move: Trick"),
        ("-item", "p1a: Great Tusk", "Choice Specs", "[from] move: Thief", "[of] p2a: Gardevoir"),
    ],
    ids=["knockoff", "corrosivegas", "trick", "thief-by-us"],
)
def test_choice_lock_released_when_item_is_removed(data, sampler, removal):
    battle = _opp_battle()
    event(battle, "move", "p2a: Gardevoir", "Moonblast", "p1a: Great Tusk")
    trackers = OpponentModel(data).update(battle)
    event(battle, *removal)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(1))
    slots[0].sampled = SampledSet("gardevoir", 83, "", _GARDEVOIR_MOVES, "trace", "Choice Specs", "Fairy")
    built = es.build_state(battle, slots)
    assert not any(m.disabled for m in built.state.side_two.pokemon[0].moves)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "engine: Choice lock is DisableMove instructions emitted on move use "
        "(engine/src/genx/items.rs get_choice_move_disable_instructions); Knock Off / Trick / Thief / "
        "Covet / Corrosive Gas (engine/src/genx/choice_effects.rs ChangeItem sites) never re-enable "
        "the old holder's moves, so later plies of the search still treat the mon as locked."
    ),
)
def test_engine_releases_choice_lock_when_knock_off_removes_the_item(data, sampler):
    import poke_engine as pe

    battle = _opp_battle()
    event(battle, "move", "p2a: Gardevoir", "Moonblast", "p1a: Great Tusk")
    trackers = OpponentModel(data).update(battle)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(1))
    slots[0].sampled = SampledSet("gardevoir", 83, "", _GARDEVOIR_MOVES, "trace", "Choice Specs", "Fairy")
    state = es.build_state(battle, slots).state
    assert sum(m.disabled for m in state.side_two.pokemon[0].moves) == 3  # locked on entry
    outcomes = pe.generate_instructions(state, "knockoff", "moonblast")
    after = [state.apply_instructions(o) for o in outcomes]
    after = [s for s in after if s.side_two.pokemon[0].hp > 0 and str(s.side_two.pokemon[0].item).lower() == "none"]
    assert after, "Knock Off did not remove the item in any outcome"
    for s in after:
        assert not any(m.disabled for m in s.side_two.pokemon[0].moves)
