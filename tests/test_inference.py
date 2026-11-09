"""Negative / indirect opponent-set evidence from synthetic protocol (bead 589.28)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from poke_env.battle.battle import Battle

from showdown_bot import inference
from showdown_bot.inference import Constraint, InferenceTracker, constraint_allows
from showdown_bot.sets import FullSet, OpponentModel, RandbatsData

FIXTURES = Path(__file__).parent / "fixtures"
OURS = ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100")


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _replay(data, *events, model=None, battle=None):
    battle = battle or Battle("battle-gen9randombattle-inf", username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    model = model or OpponentModel(data)
    model.update(battle)
    for parts in events:
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        model.update(battle)
    return model, battle


def _items(model, battle, species_key):
    tracker = model.update(battle)[species_key]
    return {full_set.item for _, full_set in tracker._matching_full_sets()}


def _gardevoir(hp="100/100"):
    return ("switch", "p2a: Gardevoir", "Gardevoir, L83", hp)


def _chandelure(hp="100/100"):
    return ("switch", "p2a: Chandelure", "Chandelure, L85", hp)


def _hit(attacker, move, target="p1a: Arcanine"):
    return [("move", attacker, move, target), ("-damage", target, "70/100")]


# --- [from] item / ability reveals ------------------------------------------------


def test_life_orb_recoil_reveals_the_item(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("-damage", "p1a: Arcanine", "60/100"),
        ("-damage", "p2a: Gardevoir", "90/100", "[from] item: Life Orb"),
    )
    assert _items(model, battle, "p2: Gardevoir") == {"Life Orb"}


def test_rocky_helmet_reveals_the_holder_not_the_attacker(data):
    model, battle = _replay(
        data, ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Slowbro", "Slowbro-Galar, L88", "100/100"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Slowbro"),
        ("-damage", "p2a: Slowbro", "70/100"),
        ("-damage", "p1a: Arcanine", "80/100", "[from] item: Rocky Helmet", "[of] p2a: Slowbro"),
    )
    assert model._battles[battle.battle_tag].pokemon[("p2", "slowbro")].current.observations == [
        ("items", "Rocky Helmet")
    ]
    arcanine = model._battles[battle.battle_tag].pokemon[("p1", "arcanine")].current
    assert not [o for o in arcanine.observations if o[0] == "items"]


def test_leftovers_heal_reveals_and_inference_can_be_disabled(data):
    events = [
        OURS, _chandelure(),
        ("-heal", "p2a: Chandelure", "100/100", "[from] item: Leftovers"),
    ]
    model, battle = _replay(data, *events)
    assert _items(model, battle, "p2: Chandelure") == {"Leftovers"}
    off, battle = _replay(data, *events, model=OpponentModel(data, inference=False))
    assert len(_items(off, battle, "p2: Chandelure")) > 1


def test_received_item_is_not_original_evidence(data):
    model, battle = _replay(
        data, OURS, _chandelure(),
        ("-item", "p2a: Chandelure", "Leftovers", "[from] move: Trick"),
        ("-heal", "p2a: Chandelure", "100/100", "[from] item: Leftovers"),
    )
    assert len(_items(model, battle, "p2: Chandelure")) > 1


# --- Life Orb -------------------------------------------------------------------


def test_no_life_orb_recoil_after_a_landed_special_move(data):
    model, battle = _replay(
        data, OURS, _gardevoir(), *_hit("p2a: Gardevoir", "Moonblast"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    )
    items = _items(model, battle, "p2: Gardevoir")
    assert "Life Orb" not in items and items


def test_life_orb_stays_when_recoil_appears(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("-damage", "p1a: Arcanine", "70/100"),
        ("-damage", "p2a: Gardevoir", "90/100", "[from] item: Life Orb"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    )
    assert "Life Orb" in _items(model, battle, "p2: Gardevoir")


@pytest.mark.parametrize(
    "outcome",
    [
        ("-miss", "p2a: Gardevoir", "p1a: Arcanine"),
        ("-immune", "p1a: Arcanine"),
        ("-activate", "p1a: Arcanine", "move: Protect"),
        ("-activate", "p1a: Arcanine", "Substitute", "[damage]"),
        ("-fail", "p2a: Gardevoir"),
    ],
)
def test_life_orb_not_excluded_when_the_move_did_not_land(data, outcome):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        outcome,
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    )
    assert "Life Orb" in _items(model, battle, "p2: Gardevoir")


def test_life_orb_not_excluded_when_the_user_fainted_to_a_helmet(data):
    model, battle = _replay(
        data, OURS, _gardevoir("5/100"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("-damage", "p1a: Arcanine", "70/100"),
        ("-damage", "p2a: Gardevoir", "0 fnt", "[from] item: Rocky Helmet", "[of] p1a: Arcanine"),
        ("faint", "p2a: Gardevoir"),
        ("upkeep",),
    )
    constraints = model._battles[battle.battle_tag].pokemon[("p2", "gardevoir")].current.constraints
    assert [c for c in constraints if c.kind != "ability_not"] == []


# --- Choice / Assault Vest ----------------------------------------------------------


def test_two_distinct_moves_exclude_choice_items(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("move", "p2a: Gardevoir", "Psyshock", "p1a: Arcanine"),
    )
    items = _items(model, battle, "p2: Gardevoir")
    assert items and not items & {"Choice Scarf", "Choice Specs", "Choice Band"}


def test_one_move_repeated_or_called_moves_do_not_exclude_choice(data):
    tracker = InferenceTracker()
    out = tracker.feed(["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"])
    out += tracker.feed(["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"])
    out += tracker.feed(["", "turn", "2"])
    out += tracker.feed(["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"])
    out += tracker.feed(["", "move", "p2a: Gardevoir", "Psyshock", "p1a: Arcanine", "[from] move: Metronome"])
    out += tracker.feed(["", "move", "p2a: Gardevoir", "Psyshock", "p1a: Arcanine", "[from]ability: Dancer"])
    assert not [e for e in out if e[1].data == inference.CHOICE_ITEMS]


def test_choice_lock_resets_on_switch(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("switch", "p2a: Chandelure", "Chandelure, L85", "100/100"),
        _gardevoir()[:2] + ("Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", "Psyshock", "p1a: Arcanine"),
    )
    assert {"Choice Scarf", "Choice Specs"} <= _items(model, battle, "p2: Gardevoir")


def test_item_changes_disable_the_choice_inference(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Trick", "p1a: Arcanine"),
        ("-item", "p2a: Gardevoir", "Leftovers", "[from] move: Trick"),
        ("move", "p2a: Gardevoir", "Psyshock", "p1a: Arcanine"),
    )
    state = model._battles[battle.battle_tag]
    constraints = state.pokemon[("p2", "gardevoir")].current.constraints
    assert all(c.data != inference.CHOICE_ITEMS for c in constraints)  # Trick is a status move: only AV goes


def test_status_move_excludes_assault_vest_only():
    tracker = InferenceTracker()
    tracker.feed(["", "switch", "p2a: Slowbro", "Slowbro-Galar, L88", "100/100"])
    out = tracker.feed(["", "move", "p2a: Slowbro", "Slack Off", "p2a: Slowbro"])
    assert ((("p2", "slowbro")), Constraint("item_not", ("assaultvest",))) in out
    assert not any(c.data == inference.CHOICE_ITEMS for _, c in out)


def test_struggle_and_transform_are_ignored():
    tracker = InferenceTracker()
    tracker.feed(["", "switch", "p2a: Ditto", "Ditto, L90", "100/100"])
    tracker.feed(["", "-transform", "p2a: Ditto", "p1a: Arcanine"])
    out = tracker.feed(["", "move", "p2a: Ditto", "Flare Blitz", "p1a: Arcanine"])
    out += tracker.feed(["", "move", "p2a: Ditto", "Extreme Speed", "p1a: Arcanine"])
    assert out == []


# --- Leftovers / orbs ----------------------------------------------------------------


def _turn_with_damage(*residual, hp="60/100"):
    return [
        OURS,
        _chandelure(),
        ("turn", "1"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Chandelure"),
        ("-damage", "p2a: Chandelure", hp),
        ("", ),
        *residual,
        ("upkeep",),
    ]


def test_missing_leftovers_heal_excludes_leftovers(data):
    model, battle = _replay(data, *_turn_with_damage())
    items = _items(model, battle, "p2: Chandelure")
    assert "Leftovers" not in items and items


def test_leftovers_heal_in_residual_keeps_it(data):
    model, battle = _replay(
        data, *_turn_with_damage(("-heal", "p2a: Chandelure", "66/100", "[from] item: Leftovers"))
    )
    assert _items(model, battle, "p2: Chandelure") == {"Leftovers"}


def test_other_residual_healing_or_heal_block_skips_the_inference(data):
    model, battle = _replay(
        data, *_turn_with_damage(("-heal", "p2a: Chandelure", "66/100", "[from] move: Wish"))
    )
    assert "Leftovers" in _items(model, battle, "p2: Chandelure")
    blocked = _turn_with_damage()
    blocked.insert(5, ("-start", "p2a: Chandelure", "move: Heal Block"))
    model, battle = _replay(data, *blocked)
    assert "Leftovers" in _items(model, battle, "p2: Chandelure")


def test_full_hp_at_residual_start_teaches_nothing(data):
    events = [OURS, _chandelure(), ("turn", "1"), ("", ), ("upkeep",)]
    model, battle = _replay(data, *events)
    assert "Leftovers" in _items(model, battle, "p2: Chandelure")


def test_damage_after_the_heal_phase_is_not_blamed_on_the_item(data):
    # Full HP at residual start, then poison damage: no Leftovers conclusion.
    events = [
        OURS, _chandelure(), ("turn", "1"), ("", ),
        ("-damage", "p2a: Chandelure", "88/100", "[from] psn"),
        ("upkeep",),
    ]
    model, battle = _replay(data, *events)
    assert "Leftovers" in _items(model, battle, "p2: Chandelure")


def _fs(item, ability="Levitate", moves=("tackle",), level=80):
    return FullSet(level=level, item=item, ability=ability, moves=moves, tera_type="Fire", count=1)


def test_orb_constraint_predicates():
    flame = Constraint("no_orb", ("flameorb",))
    assert not constraint_allows(flame, "garchomp", _fs("Flame Orb", "Rough Skin"))
    assert constraint_allows(flame, "garchomp", _fs("Leftovers", "Rough Skin"))
    assert constraint_allows(flame, "arcanine", _fs("Flame Orb", "Flash Fire"))  # Fire type
    assert constraint_allows(flame, "garchomp", _fs("Flame Orb", "Water Veil"))
    assert constraint_allows(flame, "garchomp", _fs("Flame Orb", "Klutz"))
    toxic = Constraint("no_orb", ("toxicorb",))
    assert not constraint_allows(toxic, "garchomp", _fs("Toxic Orb", "Guts"))
    assert constraint_allows(toxic, "skarmory", _fs("Toxic Orb", "Sturdy"))  # Steel type


def test_orb_emitted_for_unstatused_mon_at_end_of_turn_and_not_after_status():
    def run(*residual):
        tracker = InferenceTracker()
        for event in (
            ["", "switch", "p2a: Garchomp", "Garchomp, L80", "100/100"],
            ["", "turn", "1"],
            ["", ""],
            *residual,
            ["", "upkeep"],
        ):
            out = tracker.feed(event)
        return out

    assert [c.data for _, c in run() if c.kind == "no_orb"] == [("flameorb",), ("toxicorb",)]
    assert not run(["", "-status", "p2a: Garchomp", "brn", "[from] item: Flame Orb"])


def test_orb_not_emitted_under_safeguard_or_misty_terrain():
    tracker = InferenceTracker()
    tracker.feed(["", "switch", "p2a: Garchomp", "Garchomp, L80", "100/100"])
    tracker.feed(["", "-sidestart", "p2: Bot", "Safeguard"])
    tracker.feed(["", ""])
    assert not [e for e in tracker.feed(["", "upkeep"]) if e[1].kind == "no_orb"]


# --- Heavy-Duty Boots ------------------------------------------------------------------


def _rocks_then(*after):
    return [
        OURS,
        ("-sidestart", "p2: Opp", "move: Stealth Rock"),
        _chandelure(),
        *after,
        ("turn", "2"),
    ]


def test_no_hazard_damage_means_boots(data):
    model, battle = _replay(data, *_rocks_then())
    assert _items(model, battle, "p2: Chandelure") == {"Heavy-Duty Boots"}


def test_hazard_damage_excludes_boots(data):
    model, battle = _replay(
        data, *_rocks_then(("-damage", "p2a: Chandelure", "88/100", "[from] Stealth Rock"))
    )
    items = _items(model, battle, "p2: Chandelure")
    assert "Heavy-Duty Boots" not in items and items


def test_hazard_predicates_respect_typing_and_abilities():
    spikes = Constraint("no_hazard", ("spikes",))
    rock = Constraint("no_hazard", ("stealthrock",))
    web = Constraint("no_hazard", ("stickyweb",))
    assert constraint_allows(spikes, "corviknight", _fs("Leftovers", "Pressure"))  # Flying
    assert not constraint_allows(spikes, "garchomp", _fs("Leftovers", "Rough Skin"))
    assert constraint_allows(spikes, "garchomp", _fs("Heavy-Duty Boots", "Rough Skin"))
    assert constraint_allows(spikes, "garchomp", _fs("Air Balloon", "Rough Skin"))
    assert constraint_allows(spikes, "blissey", _fs("Leftovers", "Magic Guard"))
    assert not constraint_allows(rock, "corviknight", _fs("Leftovers", "Pressure"))  # SR hits fliers
    assert constraint_allows(rock, "clefable", _fs("Leftovers", "Magic Guard"))
    assert constraint_allows(web, "gengar", _fs("Leftovers", "Levitate"))
    assert not constraint_allows(web, "clefable", _fs("Leftovers", "Magic Guard"))


def test_boots_inference_skipped_when_terastallized_or_gravity():
    tracker = InferenceTracker()
    tracker.feed(["", "-sidestart", "p2: Opp", "move: Stealth Rock"])
    tracker.feed(["", "switch", "p2a: Garchomp", "Garchomp, L80, tera:Fire", "100/100"])
    assert tracker.feed(["", "turn", "2"]) == []
    tracker.feed(["", "-fieldstart", "move: Gravity"])
    tracker.feed(["", "switch", "p2a: Garchomp", "Garchomp, L80", "100/100"])
    assert not [e for e in tracker.feed(["", "turn", "3"]) if e[1].kind == "no_hazard"]


# --- constraints are never allowed to empty the pool -------------------------------------


def test_contradictory_constraint_is_skipped_not_fatal(data):
    model, battle = _replay(data, OURS, _gardevoir())
    tracker = model.update(battle)["p2: Gardevoir"]
    everything = Constraint("item_not", tuple({inference.to_id(f.item) for _, f in tracker._matching_full_sets()}))
    tracker._replace_observations([("constraint", everything, False)])
    assert tracker._matching_full_sets()  # skipped: still has sets


# --- Illusion: constraints from suspect switch-ins are provisional ---------------------------


def test_constraints_from_a_suspect_switch_in_do_not_bind(data):
    model, battle = _replay(
        data, OURS, _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),  # Zoroark-only tell
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
    )
    tracker = model.update(battle)["p2: Gardevoir"]
    assert tracker.provisional_observations
    assert tracker.constraints == ()
    assert {"Choice Scarf", "Choice Specs"} <= {f.item for _, f in tracker._matching_full_sets()}


# --- speed ----------------------------------------------------------------------------------


def _speed_battle(data, our_speed, *events, ours_item="Leftovers"):
    model, real = _replay(data, *events)
    ours = SimpleNamespace(stats={"spe": our_speed}, ability="Intimidate", item=ours_item)
    fake = SimpleNamespace(
        battle_tag=real.battle_tag,
        opponent_team=real.opponent_team,
        player_role="p1",
        team={"p1: Arcanine": ours},
    )
    return model, fake


def _gardevoir_speed(scarf):
    stat = inference._set_speed_stat("gardevoir", 83, ("moonblast",), "trace", "lifeorb")
    return int(stat * 1.5) if scarf else stat


def test_outspeeding_a_faster_mon_implies_choice_scarf(data):
    plain, scarfed = _gardevoir_speed(False), _gardevoir_speed(True)
    ours = plain + 5  # faster than plain Gardevoir, slower than a scarfed one
    assert plain < ours < scarfed
    model, battle = _speed_battle(
        data, ours,
        OURS, _gardevoir(), ("turn", "1"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    )
    items = {f.item for _, f in model.update(battle)["p2: Gardevoir"]._matching_full_sets()}
    assert items == {"Choice Scarf"}


def test_moving_second_to_a_faster_mon_excludes_scarf(data):
    plain, scarfed = _gardevoir_speed(False), _gardevoir_speed(True)
    model, battle = _speed_battle(
        data, plain + 5,
        OURS, _gardevoir(), ("turn", "1"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
    )
    items = {f.item for _, f in model.update(battle)["p2: Gardevoir"]._matching_full_sets()}
    assert "Choice Scarf" not in items and items


def test_trick_room_flips_the_speed_comparison(data):
    plain = _gardevoir_speed(False)
    model, battle = _speed_battle(
        data, plain + 5,
        OURS, _gardevoir(), ("-fieldstart", "move: Trick Room"), ("turn", "1"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    )
    items = {f.item for _, f in model.update(battle)["p2: Gardevoir"]._matching_full_sets()}
    assert "Choice Scarf" not in items and items  # slower than us in TR: no scarf


@pytest.mark.parametrize(
    "middle",
    [
        [("move", "p1a: Arcanine", "Quick Attack", "p2a: Gardevoir")],  # priority differs
        [("cant", "p1a: Arcanine", "par")],
        [("switch", "p1a: Arcanine", "Arcanine, L80", "100/100")],
    ],
)
def test_speed_skipped_for_unequal_priority_cant_or_switch(data, middle):
    plain = _gardevoir_speed(False)
    model, battle = _speed_battle(
        data, plain + 5,
        OURS, _gardevoir(), ("turn", "1"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        *middle,
    )
    state = model._battles[battle.battle_tag]
    assert state.pokemon[("p2", "gardevoir")].current.speed_obs == []


def test_speed_ignored_when_our_mon_has_a_priority_ability_or_the_turn_changed_speed(data):
    plain = _gardevoir_speed(False)
    events = [
        OURS, _gardevoir(), ("turn", "1"),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
    ]
    model, real = _replay(data, *events)
    ours = SimpleNamespace(stats={"spe": plain + 5}, ability="Prankster", item="Leftovers")
    fake = SimpleNamespace(
        battle_tag=real.battle_tag, opponent_team=real.opponent_team, player_role="p1",
        team={"p1: Arcanine": ours},
    )
    assert "Choice Scarf" in {f.item for _, f in model.update(fake)["p2: Gardevoir"]._matching_full_sets()}
    # A speed boost landing before the second mover changes the order basis.
    tracker = InferenceTracker()
    for event in (
        ["", "switch", "p1a: Arcanine", "Arcanine, L80", "100/100"],
        ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
        ["", "turn", "1"],
        ["", "move", "p2a: Gardevoir", "Calm Mind", "p2a: Gardevoir"],
        ["", "-boost", "p2a: Gardevoir", "spe", "1"],
    ):
        tracker.feed(event)
    assert tracker.feed(["", "move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"]) == []


def test_speed_candidates_with_speed_abilities_are_kept():
    obs = Constraint("speed", (200, False, False, 0, False, False, False))  # opp acted second
    assert not constraint_allows(obs, "gardevoir", _fs("Choice Scarf", "Trace", ("moonblast",), 83))
    assert constraint_allows(obs, "gardevoir", _fs("Life Orb", "Trace", ("moonblast",), 83))
    # Swift Swim (could be rain-boosted) is never excluded; a plain scarfed set is.
    fast = Constraint("speed", (10, True, False, 0, False, False, False))  # opp moved first, we're slow
    assert constraint_allows(fast, "gardevoir", _fs("Choice Scarf", "Trace", ("moonblast",), 83))
    slow = Constraint("speed", (10000, True, False, 0, False, False, False))
    assert not constraint_allows(slow, "gardevoir", _fs("Choice Scarf", "Trace", ("moonblast",), 83))
    assert constraint_allows(slow, "gardevoir", _fs("Choice Scarf", "Swift Swim", ("moonblast",), 83))
