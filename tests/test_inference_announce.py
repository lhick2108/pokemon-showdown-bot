"""Silent switch-in announcements rule out abilities (bead 589.34)."""

from __future__ import annotations

from pathlib import Path

import pytest
from poke_env.battle.battle import Battle

from showdown_bot.inference import Constraint, InferenceTracker, constraint_allows
from showdown_bot.sets import FullSet, OpponentModel, RandbatsData

FIXTURES = Path(__file__).parent / "fixtures"
OURS = ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100")
SALAMENCE = ("switch", "p2a: Salamence", "Salamence, L77", "100/100")


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _fs(item, ability):
    return FullSet(level=80, item=item, ability=ability, moves=("tackle",), tera_type="Fire", count=1)


def _nots(emissions, identity=("p2", "salamence")):
    return [e[1] for e in emissions if e[1].kind == "ability_not" and e[0] == identity]


def _feed(tracker, *events):
    out = []
    for parts in events:
        out.extend(tracker.feed(["", *parts]))
    return out


def test_silent_switch_in_excludes_announced_abilities():
    tracker = InferenceTracker()
    out = _feed(tracker, OURS, SALAMENCE, ("turn", "1"))
    nots = _nots(out)
    assert len(nots) == 1 and {"intimidate", "pressure"} <= set(nots[0].data)


def test_announced_ability_is_not_excluded():
    tracker = InferenceTracker()
    out = _feed(tracker, OURS, SALAMENCE, ("-ability", "p2a: Salamence", "Intimidate", "boost"), ("turn", "1"))
    nots = _nots(out)
    assert len(nots) == 1 and "intimidate" not in nots[0].data and "pressure" in nots[0].data


def test_switch_after_a_faint_is_not_evidence():
    tracker = InferenceTracker()
    _feed(tracker, OURS, SALAMENCE, ("-ability", "p2a: Salamence", "Intimidate", "boost"), ("turn", "1"))
    out = _feed(
        tracker,
        ("faint", "p1a: Arcanine"),
        ("", ),
        ("upkeep",),
        ("switch", "p2a: Skarmory", "Skarmory, L80", "100/100"),  # foe slot is a fainted mon
        ("turn", "2"),
    )
    assert _nots(out, ("p2", "skarmory")) == []


def test_neutralizing_gas_disables_the_rule():
    tracker = InferenceTracker()
    out = _feed(
        tracker,
        ("-ability", "p1a: Weezing", "Neutralizing Gas"),
        OURS,
        SALAMENCE,
        ("turn", "1"),
    )
    assert _nots(out) == []


def test_ability_not_constraint_predicate():
    c = Constraint("ability_not", ("intimidate", "pressure"))
    assert not constraint_allows(c, "salamence", _fs("Heavy-Duty Boots", "Intimidate"))
    assert constraint_allows(c, "salamence", _fs("Heavy-Duty Boots", "Moxie"))


def test_silent_switch_in_resolves_salamence_to_moxie():
    try:
        data = RandbatsData.load()  # real data: the small fixture has no Intimidate/Moxie species
    except Exception:
        pytest.skip("randbats data not cached")
    if not data.has_species("salamence"):
        pytest.skip("randbats data lacks Salamence")
    battle = Battle("battle-gen9randombattle-ann", username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    model = OpponentModel(data)
    model.update(battle)
    for parts in (OURS, SALAMENCE, ("turn", "1")):
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        model.update(battle)
    tracker = model.update(battle)["p2: Salamence"]
    assert {fs.ability for _, fs in tracker._matching_full_sets()} == {"Moxie"}


def test_opp_facts_reports_tracker_state(data):
    from showdown_bot.players.search import _opp_facts

    battle = Battle("battle-gen9randombattle-facts", username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    model = OpponentModel(data)
    model.update(battle)
    for parts in (OURS, ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"), ("turn", "1")):
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        model.update(battle)
    facts = _opp_facts(battle, model.update(battle))
    assert len(facts) == 1 and facts[0]["species"] == "gardevoir"
    row = facts[0]
    assert row["n_sets"] >= 1 and row["n_sets_nt"] <= row["n_sets"] and row["entropy_nt"] <= row["entropy"] + 1e-9
    assert abs(sum(g[3] for g in row["groups"]) - 1.0) < 0.01
    assert row["vary"]["moves"] >= 1 and row["moves"] == []
