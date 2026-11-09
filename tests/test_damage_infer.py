"""Damage-roll inference against the engine's calculator (bead 589.28)."""

from __future__ import annotations

from pathlib import Path

import poke_engine as pe
import pytest

from battle_helpers import event, make_battle, request_for
from showdown_bot import damage_infer, inference
from showdown_bot.inference import InferenceTracker
from showdown_bot.sets import OpponentModel, RandbatsData

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _setup(data, active=0):
    battle = make_battle(request_for(active=active))
    model = OpponentModel(data)
    model.update(battle)
    return battle, model


def _play(model, battle, *events):
    """Feed events the way the player mixin does: observe first, then parse."""
    model.observe_events(battle.battle_tag, [["", *parts] for parts in events], battle)
    for parts in events:
        event(battle, *parts)
    return model.update(battle)


def _our_hp(battle):
    mon = battle.active_pokemon
    return int(mon.current_hp), int(mon.max_hp)


def _items(tracker):
    return {fs.item for _, fs in tracker._matching_full_sets()}


def _engine_range(model, battle, tracker, item, *, opp_attacks, move):
    """[min, max] the engine gives for the first full set with ``item``."""
    from showdown_bot import engine_state as es

    inferer = damage_infer.DamageInferer(model.data)
    species_id, full_set = next(
        (sid, fs) for sid, fs in tracker._matching_full_sets() if fs.item == item
    )
    opp = battle.opponent_active_pokemon
    ours = battle.active_pokemon
    pokemon, moves = inferer._candidate(battle, opp, species_id, full_set)
    our_poke, our_moves = es._own_pokemon(battle, ours, True)
    state = pe.State(
        side_one=es._make_side([(our_poke, our_moves, ours)], ours, battle, battle.side_conditions, is_ours=True),
        side_two=es._make_side([(pokemon, moves, opp)], opp, battle, battle.opponent_side_conditions, is_ours=False),
        **es.field_kwargs(battle),
    )
    if opp_attacks:
        high = pe.calculate_damage(state, "splash", move, True)[1][0]
        return [high * damage_infer.MIN_ROLL, high]
    high = pe.calculate_damage(state, move, "splash", True)[0][0]
    return [x * 100.0 / pokemon.maxhp for x in (high * damage_infer.MIN_ROLL, high)]


def _start(model, battle, ours, theirs):
    hp, max_hp = _our_hp(battle)
    _play(
        model, battle,
        ("switch", f"p1a: {ours[0]}", ours[1], f"{hp}/{max_hp}"),
        ("switch", f"p2a: {theirs[0]}", theirs[1], "100/100"),
        ("turn", "1"),
    )
    return hp, max_hp


def test_opponent_hit_on_us_identifies_choice_specs(data):
    battle, model = _setup(data, active=5)
    hp, max_hp = _start(model, battle, ("Heatran", "Heatran, L80"), ("Gardevoir", "Gardevoir, L83"))
    tracker = model.update(battle)["p2: Gardevoir"]
    low, high = _engine_range(model, battle, tracker, "Choice Specs", opp_attacks=True, move="moonblast")
    damage = int(high * 0.97)
    assert damage < hp
    _play(
        model, battle,
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"),
        ("-supereffective", "p1a: Heatran"),
        ("-damage", "p1a: Heatran", f"{hp - damage}/{max_hp}"),
        ("upkeep",), ("turn", "2"),
    )
    items = _items(model.update(battle)["p2: Gardevoir"])
    assert items == {"Choice Specs"}, items
    assert model.damage_stats.narrowed == 1 and model.damage_stats.all_rejected == 0


def test_a_low_roll_keeps_the_revealed_life_orb_and_the_pool_never_empties(data):
    battle, model = _setup(data, active=5)
    hp, max_hp = _start(model, battle, ("Heatran", "Heatran, L80"), ("Gardevoir", "Gardevoir, L83"))
    tracker = model.update(battle)["p2: Gardevoir"]
    low, high = _engine_range(model, battle, tracker, "Life Orb", opp_attacks=True, move="moonblast")
    damage = int(low * 1.01)
    _play(
        model, battle,
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"),
        ("-damage", "p1a: Heatran", f"{hp - damage}/{max_hp}"),
        ("-damage", "p2a: Gardevoir", "90/100", "[from] item: Life Orb"),
        ("upkeep",), ("turn", "2"),
    )
    items = _items(model.update(battle)["p2: Gardevoir"])
    assert items == {"Life Orb"}
    assert model.damage_stats.all_rejected == 0


def test_our_hit_on_a_bulky_opponent_rejects_assault_vest(data):
    battle, model = _setup(data, active=2)  # Gholdengo
    _start(model, battle, ("Gholdengo", "Gholdengo, L80"), ("Great Tusk", "Great Tusk, L80"))
    tracker = model.update(battle)["p2: Great Tusk"]
    assert {"Assault Vest", "Leftovers"} <= _items(tracker)
    low, high = _engine_range(model, battle, tracker, "Leftovers", opp_attacks=False, move="shadowball")
    after = max(2, round(100 - (low + high) / 2))
    _play(
        model, battle,
        ("move", "p1a: Gholdengo", "Shadow Ball", "p2a: Great Tusk"),
        ("-damage", "p2a: Great Tusk", f"{after}/100"),
        ("upkeep",), ("turn", "2"),
    )
    items = _items(model.update(battle)["p2: Great Tusk"])
    assert "Assault Vest" not in items and "Leftovers" in items
    assert model.damage_stats.narrowed == 1


@pytest.mark.parametrize(
    "extra",
    [
        [("-crit", "p1a: Heatran")],
        [("-activate", "p1a: Heatran", "Substitute", "[damage]")],
        [("-activate", "p1a: Heatran", "move: Protect")],
    ],
)
def test_crits_substitutes_and_protect_are_not_read(extra):
    tracker = InferenceTracker()
    for e in (
        ["", "switch", "p1a: Heatran", "Great Tusk, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
    ):
        tracker.feed(e)
    out = []
    for parts in extra:
        out += tracker.feed(["", *parts])
    out += tracker.feed(["", "-damage", "p1a: Heatran", "50/100"])
    out += tracker.feed(["", "upkeep"])
    assert not [e for e in out if isinstance(e[1], inference.DamageObs)]


def test_second_hit_marks_multihit_unreadable():
    tracker = InferenceTracker()
    emitted = []
    for e in (
        ["", "switch", "p1a: Heatran", "Great Tusk, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "-damage", "p1a: Heatran", "60/100"],
        ["", "-damage", "p1a: Heatran", "30/100"],
        ["", "upkeep"],
    ):
        emitted += tracker.feed(e)
    assert not [x for x in emitted if isinstance(x[1], inference.DamageObs)]


def test_plain_hit_emits_an_observation_and_dirty_turns_do_not():
    tracker = InferenceTracker()
    emitted = []
    for e in (
        ["", "switch", "p1a: Heatran", "Great Tusk, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "-damage", "p1a: Heatran", "50/100"],
        ["", "upkeep"],
    ):
        emitted += tracker.feed(e)
    obs = [x[1] for x in emitted if isinstance(x[1], inference.DamageObs)]
    assert len(obs) == 1 and obs[0].move_id == "moonblast"
    assert obs[0].hp_before == 1.0 and obs[0].hp_after == 0.5

    dirty = InferenceTracker()
    emitted = []
    for e in (
        ["", "switch", "p1a: Heatran", "Great Tusk, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p1a: Heatran", "Swords Dance", "p1a: Heatran"],
        ["", "-boost", "p1a: Heatran", "atk", "2"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "-damage", "p1a: Heatran", "50/100"],
        ["", "upkeep"],
    ):
        emitted += dirty.feed(e)
    assert not [x for x in emitted if isinstance(x[1], inference.DamageObs)]


def test_excluded_moves_are_not_read():
    for move in ("knockoff", "bulletseed", "gyroball", "terablast", "foulplay", "earthquake"):
        assert inference.damage_move_ok(move) is (move == "earthquake"), move


def test_inference_off_skips_damage(data):
    battle = make_battle()
    model = OpponentModel(data, inference=False)
    hp, max_hp = _our_hp(battle)
    for parts in (
        ("switch", "p1a: Heatran", "Heatran, L80", f"{hp}/{max_hp}"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("turn", "1"),
    ):
        event(battle, *parts)
    _play(
        model, battle,
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"),
        ("-damage", "p1a: Heatran", f"{hp // 2}/{max_hp}"),
    )
    assert model.damage_stats is None


def test_roost_and_attacker_hp_are_tracked_for_the_hit():
    base = [
        ["", "switch", "p1a: Heatran", "Great Tusk, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
    ]
    # an earlier hit this turn lowered the attacker's HP: the observation carries it
    tracker = InferenceTracker()
    emitted = []
    for e in base + [
        ["", "move", "p1a: Heatran", "Earthquake", "p2a: Gardevoir"],
        ["", "-damage", "p2a: Gardevoir", "40/100"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "-damage", "p1a: Heatran", "50/100"],
        ["", "upkeep"],
    ]:
        emitted += tracker.feed(e)
    moonblast = [x[1] for x in emitted if isinstance(x[1], inference.DamageObs)
                 and x[1].move_id == "moonblast"]
    assert moonblast and moonblast[0].attacker_hp == pytest.approx(0.4)

    # Roost (-singleturn) removes Flying for the turn: later hits are not plain
    tracker = InferenceTracker()
    emitted = []
    for e in base + [
        ["", "move", "p1a: Heatran", "Roost", "p1a: Heatran"],
        ["", "-singleturn", "p1a: Heatran", "move: Roost"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "-damage", "p1a: Heatran", "50/100"],
        ["", "upkeep"],
    ]:
        emitted += tracker.feed(e)
    assert not [x for x in emitted if isinstance(x[1], inference.DamageObs)]


def test_roll_consistent_window_and_survival_effects():
    # exact HP: rolls span [0.85, 1.0] of 100, widened by the margins
    assert damage_infer.roll_consistent(100, 93, True, 500)
    assert damage_infer.roll_consistent(100, 85, True, 500)
    assert not damage_infer.roll_consistent(100, 60, True, 500)
    assert not damage_infer.roll_consistent(100, 130, True, 500)
    # percent HP
    assert damage_infer.roll_consistent(40.0, 36.0, False, 100.0)
    assert not damage_infer.roll_consistent(40.0, 20.0, False, 100.0)
    # a hit that must have KOed (min roll >= defender HP) but did not is a
    # Focus Sash / Sturdy / Endure style cut: it says nothing about the roll
    assert damage_infer.roll_consistent(300, 97, True, 100)
    assert damage_infer.roll_consistent(173.0, 97.0, False, 100.0)


def test_unmodelled_forms_and_retyping_abilities_are_not_judged():
    class Mon:
        def __init__(self, species):
            self.species = species

    assert damage_infer._unmodelled_forme(Mon("Minior-Meteor"))
    assert damage_infer._unmodelled_forme(Mon("ditto"))
    assert not damage_infer._unmodelled_forme(Mon("Gardevoir"))
    assert {"protean", "libero", "analytic"} <= damage_infer._UNRELIABLE_ABILITIES
