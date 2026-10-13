"""Sleep / Rest / Sleep Talk / Encore fidelity (bead 589.35).

The protocol excerpts replay data/ladder/s3/logs/battle-gen9randombattle-2697417199.log
(our Mesprit vs a Rest + Sleep Talk Lapras), here with Great Tusk standing in for Mesprit.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("poke_engine")

import poke_engine as pe  # noqa: E402
from battle_helpers import OUR_TEAM, event, make_battle, request_for  # noqa: E402
from test_engine_state import FIXTURES  # noqa: E402

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot import sleep as sleep_module  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402

needs_sdb4 = pytest.mark.skipif(
    not es._ENGINE_SLEEP_FIXES, reason="installed poke_engine predates the sleep fixes (sdb.4)"
)

LAPRAS = "p2a: Lapras"
TUSK = "p1a: Great Tusk"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


class Game:
    """A hand-driven battle that also feeds the raw events to ``showdown_bot.sleep``."""

    def __init__(self, tag: str, team=OUR_TEAM):
        self.battle = make_battle(request_for(team), tag=tag)
        self.feed("start")
        self.feed("switch", LAPRAS, "Lapras, L87, M", "100/100")
        self.feed("turn", "1")  # Lapras is no longer on its switch-in turn

    def feed(self, *parts: str) -> None:
        event(self.battle, *parts)
        sleep_module.observe_events(self.battle.battle_tag, [["", *parts]])

    def lapras_moves(self, *names: str) -> None:
        for name in names:
            self.feed("move", LAPRAS, name, TUSK)

    def state(self, data, sampler):
        model = OpponentModel(data)
        trackers = model.update(self.battle)
        slots = es.sample_opponent_team(self.battle, trackers, sampler, random.Random(0))
        return es.build_state(self.battle, slots)

    @property
    def lapras(self):
        return self.battle.opponent_active_pokemon


def rest_then_two_sleep_talk_turns(game: Game) -> None:
    """Turns 6-8 of the log: Rest, then two turns of Sleep Talk (3 poke-env events each)."""
    game.lapras_moves("Freeze-Dry", "Sparkling Aria")
    game.feed("move", LAPRAS, "Rest", LAPRAS)
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("-heal", LAPRAS, "100/100 slp", "[silent]")
    game.feed("turn", "7")
    for aria in ("Freeze-Dry", "Sparkling Aria"):
        game.feed("cant", LAPRAS, "slp")
        game.feed("move", LAPRAS, "Sleep Talk", LAPRAS)
        game.feed("move", LAPRAS, aria, TUSK, "[from] move: Sleep Talk")
        game.feed("turn", "8")


def test_poke_env_counter_overcounts_sleep_talk_turns_but_the_tracker_does_not():
    game = Game("battle-gen9randombattle-sleep-1")
    rest_then_two_sleep_talk_turns(game)
    assert game.lapras.status == es.Status.SLP
    assert game.lapras.status_counter >= 6  # poke-env: cant + Sleep Talk + called move, twice
    record = sleep_module.for_battle(game.battle.battle_tag).record("p2: Lapras")
    assert (record.kind, record.turns) == ("rest", 2)
    assert record.engine_fields() == (0, 1)  # Rest: 3 - 2 turns lost; it wakes on its next attempt


def test_rest_sleep_reaches_the_engine_as_rest_turns(data, sampler):
    game = Game("battle-gen9randombattle-sleep-2")
    game.lapras_moves("Freeze-Dry", "Sparkling Aria")
    game.feed("move", LAPRAS, "Rest", LAPRAS)
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("turn", "7")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.status, lapras.rest_turns, lapras.sleep_turns) == ("SLEEP", 3, 0)
    game.feed("cant", LAPRAS, "slp")
    game.feed("turn", "8")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.rest_turns, lapras.sleep_turns) == (2, 0)
    game.feed("cant", LAPRAS, "slp")
    game.feed("turn", "9")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.rest_turns, lapras.sleep_turns) == (1, 0)  # wakes (and acts) next attempt


def test_ordinary_sleep_counts_cant_events_only_and_survives_a_switch(data, sampler):
    game = Game("battle-gen9randombattle-sleep-3")
    game.feed("-status", LAPRAS, "slp", "[from] move: Sleep Powder")
    game.feed("cant", LAPRAS, "slp")
    game.feed("move", LAPRAS, "Sleep Talk", LAPRAS)  # extra poke-env ticks must not count
    game.feed("move", LAPRAS, "Freeze-Dry", TUSK, "[from] move: Sleep Talk")
    game.feed("turn", "2")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.status, lapras.sleep_turns, lapras.rest_turns) == ("SLEEP", 1, 0)
    game.feed("switch", "p2a: Ampharos", "Ampharos, L88, M", "100/100")
    game.feed("switch", LAPRAS, "Lapras, L87, M", "100/100 slp")
    game.feed("cant", LAPRAS, "slp")
    game.feed("turn", "3")
    assert sleep_module.for_battle(game.battle.battle_tag).record("p2: Lapras").turns == 2


def test_waking_curing_and_fainting_clear_the_record():
    tracker = sleep_module.SleepTracker()
    for ending in (["", "-curestatus", LAPRAS, "slp", "[msg]"], ["", "faint", LAPRAS],
                   ["", "-status", LAPRAS, "par"]):
        tracker.observe(["", "-status", LAPRAS, "slp", "[from] move: Rest"])
        assert tracker.record(LAPRAS) is not None
        tracker.observe(ending)
        assert tracker.record(LAPRAS) is None


def test_status_sequence_wake_then_paralysis_then_rest_over_paralysis(data, sampler):
    """T9-T11 of the log: woken by Heal Block, Thunder Wave lands, Rest cures it into sleep."""
    game = Game("battle-gen9randombattle-sleep-4")
    game.lapras_moves("Freeze-Dry", "Sparkling Aria", "Sleep Talk")
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("cant", LAPRAS, "slp")
    game.feed("cant", LAPRAS, "slp")
    game.feed("-curestatus", LAPRAS, "slp", "[msg]")
    game.feed("cant", LAPRAS, "move: Heal Block", "Rest")
    game.feed("turn", "10")
    assert game.lapras.status is None
    assert game.state(data, sampler).state.side_two.pokemon[0].status == "NONE"
    game.feed("-status", LAPRAS, "par")
    game.feed("turn", "11")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.status, lapras.sleep_turns, lapras.rest_turns) == ("PARALYZE", 0, 0)
    game.feed("move", LAPRAS, "Rest", LAPRAS)
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("turn", "12")
    lapras = game.state(data, sampler).state.side_two.pokemon[0]
    assert (lapras.status, lapras.sleep_turns, lapras.rest_turns) == ("SLEEP", 0, 3)


def test_called_move_is_revealed_but_sleep_talk_stays_the_last_move(data, sampler):
    game = Game("battle-gen9randombattle-sleep-5")
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("cant", LAPRAS, "slp")
    game.feed("move", LAPRAS, "Sleep Talk", LAPRAS)
    game.feed("move", LAPRAS, "Sparkling Aria", TUSK, "[from] move: Sleep Talk")
    game.feed("turn", "3")
    assert "sparklingaria" in game.lapras.moves
    last = [k for k, m in game.lapras.moves.items() if m.is_last_used]
    assert last == ["sleeptalk"]
    built = game.state(data, sampler)
    opp = built.state.side_two
    index = int(opp.last_used_move.split(":")[1])
    assert opp.pokemon[0].moves[index].id.lower() == "sleeptalk"


def _joint(built, mine: str, theirs: str):
    return pe.generate_instructions(built.state, mine, theirs)


def _outcome_texts(outcomes) -> list[str]:
    return [str(i) for o in outcomes for i in o.instruction_list]


ENCORE_TEAM = [
    ("Great Tusk", 80, "Leftovers", "Protosynthesis",
     ["Close Combat", "Headlong Rush", "Knock Off", "Encore"], "Ghost"),
    *OUR_TEAM[1:],
]


def _encore_outcomes(game: Game, data, sampler, theirs: str) -> list[str]:
    state = game.state(data, sampler).state
    return _outcome_texts(pe.generate_instructions(state, "encore", theirs))


@needs_sdb4
def test_encore_into_a_sleep_talk_user_fails_in_the_engine(data, sampler):
    """T14 of the log: Lapras' last move was Sleep Talk, so our Encore fails."""
    game = Game("battle-gen9randombattle-sleep-6", ENCORE_TEAM)
    game.lapras_moves("Freeze-Dry", "Sparkling Aria", "Rest")
    game.feed("-status", LAPRAS, "slp", "[from] move: Rest")
    game.feed("cant", LAPRAS, "slp")
    game.feed("move", LAPRAS, "Sleep Talk", LAPRAS)
    game.feed("move", LAPRAS, "Freeze-Dry", TUSK, "[from] move: Sleep Talk")
    game.feed("turn", "3")
    assert not any("ENCORE" in t for t in _encore_outcomes(game, data, sampler, "freezedry"))


@needs_sdb4
def test_encore_into_a_normal_last_move_still_works(data, sampler):
    game = Game("battle-gen9randombattle-sleep-8", ENCORE_TEAM)
    game.lapras_moves("Freeze-Dry", "Sparkling Aria")
    game.feed("turn", "3")
    assert any("ENCORE" in t for t in _encore_outcomes(game, data, sampler, "freezedry"))


@needs_sdb4
def test_search_on_a_rest_sleep_state_stays_valid(data, sampler):
    game = Game("battle-gen9randombattle-sleep-7")
    rest_then_two_sleep_talk_turns(game)
    built = game.state(data, sampler)
    assert pe.monte_carlo_tree_search(built.state, 30).total_visits > 0
