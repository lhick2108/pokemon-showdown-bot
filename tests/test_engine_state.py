"""State conversion and opponent-team sampling (no server needed)."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("poke_engine")

from battle_helpers import (  # noqa: E402
    OUR_TEAM,
    event,
    make_battle,
    opp_switch,
    request_for,
)

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData, SampledSet, SetTracker, to_id  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


def _build(battle, data, sampler, seed=0):
    model = OpponentModel(data)
    trackers = model.update(battle)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(seed))
    return es.build_state(battle, slots), slots


def _opp_battle(**kwargs):
    battle = make_battle(request_for(**kwargs))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


# --- stat formula -----------------------------------------------------------


def test_stat_formulas_match_showdown_by_hand():
    # Gardevoir (68/65/65/125/115/80) at L83, 31 IV / 85 EV (floor(85/4) = 21):
    # HP    = floor((136 + 31 + 21) * 83 / 100) + 83 + 10 = 156 + 93 = 249
    # Atk   = floor((130 + 31 + 21) * 83 / 100) + 5       = 151 + 5   = 156
    # SpA   = floor((250 + 31 + 21) * 83 / 100) + 5       = 250 + 5   = 255
    # Spe   = floor((160 + 31 + 21) * 83 / 100) + 5       = 175 + 5   = 180
    base = {"hp": 68, "atk": 65, "def": 65, "spa": 125, "spd": 115, "spe": 80}
    stats = es.compute_stats(base, 83)
    assert stats["hp"] == 249
    assert stats["atk"] == 156
    assert stats["spa"] == 255
    assert stats["spe"] == 180
    assert es.calc_hp(1, 50) == 1  # Shedinja


def test_zero_ev_iv_stat_formula():
    # Atk with 0 EV / 0 IV at L83, base 65: floor(130 * 83 / 100) + 5 = 107 + 5
    assert es.calc_stat(65, 83, ev=0, iv=0) == 112


def test_randbats_rules_zero_attack_for_special_only_sets():
    evs, ivs = es.randbats_evs_ivs(
        "gardevoir", 83, ["calmmind", "focusblast", "moonblast", "psyshock"], "trace", "lifeorb"
    )
    assert evs[1] == 0 and ivs[1] == 0
    assert evs[0] == 85  # Life Orb: no Stealth Rock HP tuning
    physical, _ = es.randbats_evs_ivs(
        "greattusk", 80, ["closecombat", "headlongrush", "knockoff", "rapidspin"], "protosynthesis", "leftovers"
    )
    assert physical[1] == 85


def test_randbats_rules_zero_speed_for_trick_room_and_hp_tuning():
    evs, ivs = es.randbats_evs_ivs(
        "slowbrogalar", 88, ["trickroom", "sludgebomb", "psychic", "slackoff"], "quickdraw", "heavydutyboots"
    )
    assert evs[5] == 0 and ivs[5] == 0
    # Volcarona is 4x weak to Rock and holds nothing that heals: HP EVs are
    # lowered until HP % (4 / 2) > 0, i.e. an odd HP, so it survives the most
    # Stealth Rock switch-ins (teams.ts "Maximize number of Stealth Rock").
    evs, _ = es.randbats_evs_ivs(
        "volcarona", 80, ["quiverdance", "fierydance", "bugbuzz", "morningsun"], "flamebody", "heavydutyboots"
    )
    assert evs[0] == 85  # Boots: immune to Rocks, no tuning
    evs, _ = es.randbats_evs_ivs(
        "volcarona", 80, ["quiverdance", "fierydance", "bugbuzz", "roost"], "flamebody", "lumberry"
    )
    hp = (2 * 85 + 31 + evs[0] // 4 + 100) * 80 // 100 + 10
    assert hp % 2 == 1 and evs[0] <= 85


def test_opponent_stats_in_state_follow_formula(data, sampler):
    battle = _opp_battle()
    built, slots = _build(battle, data, sampler)
    mon = built.state.side_two.pokemon[0]
    assert mon.id.lower() == "gardevoir"
    # Gardevoir sets here are all special attackers: Atk uses 0 EV / 0 IV.
    assert mon.attack == 112
    assert mon.level == 83
    assert mon.maxhp == 249 and mon.hp == 249
    assert mon.special_attack == 255 and mon.speed == 180
    assert mon.defense == 156


# --- carrying battle state over ----------------------------------------------


def test_our_side_is_exact_from_the_request(data, sampler):
    battle = _opp_battle()
    built, _ = _build(battle, data, sampler)
    side = built.state.side_one
    tusk = side.pokemon[0]
    request_pokemon = battle.last_request["side"]["pokemon"][0]
    assert tusk.id.lower() == "greattusk"
    assert tusk.attack == request_pokemon["stats"]["atk"]
    assert tusk.speed == request_pokemon["stats"]["spe"]
    assert tusk.maxhp == battle.team["p1: Great Tusk"].max_hp
    assert tusk.item.lower() == "leftovers" and tusk.ability.lower() == "protosynthesis"
    assert tusk.tera_type.lower() == "ghost" and not tusk.terastallized
    assert {m.id.lower() for m in tusk.moves} == {"closecombat", "headlongrush", "knockoff", "rapidspin"}
    assert [p.id.lower() for p in side.pokemon][0] == "greattusk"
    assert len({p.id for p in side.pokemon}) == 6


def test_boosts_status_hp_hazards_weather_carry_over(data, sampler):
    request = request_for(hp={2: (100, "par"), 3: (0, "")})
    battle = make_battle(request)
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    event(battle, "-boost", "p1a: Great Tusk", "atk", "2")
    event(battle, "-unboost", "p2a: Gardevoir", "spa", "1")
    event(battle, "-status", "p2a: Gardevoir", "brn")
    event(battle, "-damage", "p2a: Gardevoir", "50/100 brn")
    event(battle, "-sidestart", "p1: bot", "Spikes")
    event(battle, "-sidestart", "p1: bot", "Spikes")
    event(battle, "-sidestart", "p1: bot", "move: Stealth Rock")
    event(battle, "-sidestart", "p2: Foe", "move: Reflect")
    event(battle, "-weather", "Sandstorm")
    event(battle, "-fieldstart", "move: Trick Room")
    event(battle, "-fieldstart", "move: Grassy Terrain")
    event(battle, "-start", "p1a: Great Tusk", "Substitute")
    event(battle, "-start", "p1a: Great Tusk", "Leech Seed")
    event(battle, "-start", "p2a: Gardevoir", "confusion")
    built, _ = _build(battle, data, sampler)
    state = built.state
    ours, theirs = state.side_one, state.side_two
    assert ours.attack_boost == 2
    assert theirs.special_attack_boost == -1
    assert theirs.pokemon[0].status.lower() == "burn"
    assert theirs.pokemon[0].hp == round(theirs.pokemon[0].maxhp * 0.5)
    assert ours.side_conditions.spikes == 2
    assert ours.side_conditions.stealth_rock == 1
    assert theirs.side_conditions.reflect > 0 and ours.side_conditions.reflect == 0
    assert state.weather.lower() == "sand" and state.weather_turns_remaining > 0
    assert state.trick_room is True and state.trick_room_turns_remaining > 0
    assert state.terrain.lower() == "grassyterrain"
    assert {"substitute", "leechseed"} <= {v.lower() for v in ours.volatile_statuses}
    assert ours.substitute_health > 0
    assert "confusion" in {v.lower() for v in theirs.volatile_statuses}
    by_id = {p.id.lower(): p for p in ours.pokemon}
    assert by_id["gholdengo"].status.lower() == "paralyze"
    assert by_id["dragapult"].hp == 0  # fainted


def test_tera_state_and_type_carry_over(data, sampler):
    request = request_for(terastallized={0: "Ghost"}, can_tera=False)
    battle = make_battle(request)
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    event(battle, "-terastallize", "p2a: Gardevoir", "Fairy")
    built, _ = _build(battle, data, sampler)
    tusk = built.state.side_one.pokemon[0]
    assert tusk.terastallized and tusk.tera_type.lower() == "ghost"
    assert tuple(t.lower() for t in tusk.types) == ("ground", "fighting")  # original types
    gardevoir = built.state.side_two.pokemon[0]
    assert gardevoir.terastallized and gardevoir.tera_type.lower() == "fairy"


def test_pp_and_disabled_moves_from_request(data, sampler):
    battle = make_battle(request_for(disabled={"knockoff"}))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    event(battle, "move", "p2a: Gardevoir", "Moonblast", "p1a: Great Tusk")
    built, _ = _build(battle, data, sampler)
    tusk = built.state.side_one.pokemon[0]
    knock = next(m for m in tusk.moves if m.id.lower() == "knockoff")
    assert knock.disabled and knock.pp == 16
    moonblast = [m for m in built.state.side_two.pokemon[0].moves if m.id.lower() == "moonblast"]
    assert moonblast and moonblast[0].pp == 23  # 24 max PP minus the observed use


def test_forced_switch_marks_our_side(data, sampler):
    battle = make_battle(request_for(force_switch=True))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    built, _ = _build(battle, data, sampler)
    assert built.state.side_one.force_switch is True


def test_first_request_with_no_opponent_revealed_still_builds(data, sampler):
    """The team-preview-less start: nothing about the opponent is known yet."""
    battle = make_battle()
    built, slots = _build(battle, data, sampler)
    assert len(slots) == 6
    assert len({p.id for p in built.state.side_two.pokemon}) == 6
    assert all(p.hp > 0 for p in built.state.side_two.pokemon)


def test_unknown_ids_are_reported_not_fatal(data, sampler):
    battle = _opp_battle()
    model = OpponentModel(data)
    trackers = model.update(battle)
    slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(3))
    slots[1].sampled = SampledSet(
        slots[1].sampled.species, 80, "", ("definitelynotamove",), "notanability", "notanitem", "Water"
    )
    built = es.build_state(battle, slots)
    assert built.unknown_ids  # surfaced for the stats counter
    built.to_string()


# --- unrevealed-slot sampling ------------------------------------------------


def _info(species: str, level: int = 80) -> es.SpeciesInfo:
    info = es.species_info(species, level)
    assert info is not None
    return info


def test_species_clause_and_type_rules():
    assert es.violates_team_rules(_info("greninja"), [_info("greninjabond")])  # same base species
    # at most two of any type
    assert es.violates_team_rules(_info("charizard"), [_info("arcanine"), _info("heatran")])
    assert not es.violates_team_rules(_info("charizard"), [_info("arcanine")])
    # at most one 4x weakness to a type (Ice: Landorus-T is 2x... Garchomp 4x)
    assert es.violates_team_rules(_info("dragonite"), [_info("garchomp")])
    # three weak to Ground is the limit
    team = [_info("heatran"), _info("magnezone"), _info("jolteon")]
    assert es.violates_team_rules(_info("raikou"), team)
    # only one level-100 Pokemon
    assert es.violates_team_rules(_info("blissey", 100), [_info("ditto", 100)])
    # Zoroark never takes the last slot
    four = [_info(n) for n in ("garchomp", "corviknight", "azumarill", "dragapult")]
    assert not es.violates_team_rules(_info("zoroark"), four)
    assert es.violates_team_rules(_info("zoroark"), [*four, _info("heatran")])


def test_sampled_team_respects_uniqueness_and_rules(data, sampler):
    for seed in range(60):
        battle = _opp_battle()
        trackers = OpponentModel(data).update(battle)
        slots = es.sample_opponent_team(battle, trackers, sampler, random.Random(seed))
        assert len(slots) == 6
        bases = [es.species_info(s.species_id, 80).base_species for s in slots]
        assert len(set(bases)) == 6, bases
        assert slots[0].species_id == "gardevoir" and slots[0].mon is not None
        assert all(s.mon is None for s in slots[1:])
        for slot in slots[1:]:
            assert slot.sampled.moves, slot


def test_unrevealed_species_are_frequency_weighted_not_uniform(data, sampler):
    rng = random.Random(7)
    draws = Counter(sampler.sample_species([], rng) for _ in range(30000))
    totals = {
        species: sum(full.count for full in data.full_sets(species))
        for species in sampler.species_ids
    }
    grand = sum(totals.values())
    for species in ("greninja", "ninetales", "gardevoir"):
        expected = totals[species] / grand
        assert draws[species] / 30000 == pytest.approx(expected, rel=0.15)
    # Genuinely non-uniform: the rarest and commonest species differ a lot.
    assert totals["ninetales"] < totals["azumarill"] / 1.5
    assert draws["ninetales"] < draws["azumarill"]


def test_sampled_sets_are_whole_real_sets(data, sampler):
    rng = random.Random(2)
    legal = {
        tuple(sorted(to_id(m) for m in full.moves))
        for full in data.full_sets("greattusk")
    }
    for _ in range(50):
        sampled = sampler.sample_set("greattusk", rng)
        assert tuple(sorted(to_id(m) for m in sampled.moves)) in legal


# --- engine choice <-> poke-env order mapping --------------------------------


def test_move_mapping_round_trips_moves_switches_and_tera(data, sampler):
    battle = _opp_battle()
    built, _ = _build(battle, data, sampler)
    legal = {order.message for order in battle.valid_orders}
    mapped = set()
    for move in battle.available_moves:
        for suffix in ("", "-tera"):
            order = built.choice_to_order(move.id + suffix, battle)
            assert order is not None
            assert order.message in legal
            mapped.add(order.message)
            assert ("terastallize" in order.message) == (suffix == "-tera")
    for mon in battle.available_switches:
        order = built.choice_to_order(f"switch {to_id(mon.species)}", battle)
        assert order is not None and order.message in legal
        mapped.add(order.message)
    assert "/choose move closecombat terastallize" in mapped
    assert "/choose switch Kingambit" in mapped
    assert built.choice_to_order("switch greattusk", battle) is None  # the active one
    assert built.choice_to_order("none", battle) is None
    assert built.choice_to_order("splash", battle) is None
    assert built.choice_to_order("closecombat-tera", make_battle(request_for(can_tera=False))) is None


def test_every_engine_option_maps_to_a_legal_order(data, sampler):
    """Everything MCTS can ask us to play must be an order Showdown accepts."""
    import poke_engine as pe

    battle = _opp_battle()
    built, _ = _build(battle, data, sampler)
    result = pe.monte_carlo_tree_search(built.state, 20)
    legal = {order.message for order in battle.valid_orders}
    for option in result.side_one:
        order = built.choice_to_order(option.move_choice, battle)
        assert order is not None, option.move_choice
        assert order.message in legal


# --- regressions found by stress-testing MCTS on converted states -------------


def test_sleep_counter_is_clamped_to_the_engines_maximum(data, sampler):
    """A sleep counter above 3 makes the engine's wake chance 1/0 (NaN panic)."""
    import poke_engine as pe

    battle = _opp_battle()
    for mon in (battle.active_pokemon, battle.opponent_active_pokemon):
        mon._status = es.Status.SLP
        mon._status_counter = 7
    built, _ = _build(battle, data, sampler)
    assert built.state.side_one.pokemon[0].sleep_turns == 3
    assert built.state.side_two.pokemon[0].sleep_turns == 3
    assert pe.monte_carlo_tree_search(built.state, 20).total_visits > 0


def test_encore_without_a_known_last_move_is_dropped(data, sampler):
    """The engine panics on an Encore volatile whose last-used move is unknown."""
    import poke_engine as pe

    battle = _opp_battle()
    battle.opponent_active_pokemon._effects[es.Effect.ENCORE] = 1
    built, _ = _build(battle, data, sampler)
    assert "encore" not in {v.lower() for v in built.state.side_two.volatile_statuses}
    assert pe.monte_carlo_tree_search(built.state, 20).total_visits > 0
