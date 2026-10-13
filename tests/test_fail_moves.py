"""Guaranteed-fail moves the engine must see coming (bead 589.39).

Wish is not tracked by poke-env at all, so the protocol tracker in ``showdown_bot.sleep``
supplies the engine's pending-Wish slot (ladder log 2697360721: Umbreon used Wish on
turns 8-13 and three of five uses failed because one was already pending).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("poke_engine")

from battle_helpers import OUR_TEAM, event, make_battle, request_for  # noqa: E402
from test_engine_state import FIXTURES  # noqa: E402

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot import sleep as sleep_module  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402

LAPRAS = "p2a: Lapras"
TUSK = "p1a: Great Tusk"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


@pytest.fixture(scope="module")
def sampler(data):
    return es.TeamSampler(data)


class Game:
    def __init__(self, tag: str):
        self.battle = make_battle(request_for(OUR_TEAM), tag=tag)
        self.feed("start")
        self.feed("switch", LAPRAS, "Lapras, L87, M", "100/100")
        self.feed("turn", "1")

    def feed(self, *parts: str) -> None:
        event(self.battle, *parts)
        sleep_module.observe_events(self.battle.battle_tag, [["", *parts]])

    def state(self, data, sampler):
        model = OpponentModel(data)
        trackers = model.update(self.battle)
        slots = es.sample_opponent_team(self.battle, trackers, sampler, random.Random(0))
        return es.build_state(self.battle, slots).state


def test_pending_wish_reaches_the_engine_and_a_second_wish_does_not_rearm_it(data, sampler):
    game = Game("battle-gen9randombattle-fail-1")
    assert game.state(data, sampler).side_two.wish == (0, 0)
    game.feed("move", LAPRAS, "Wish", LAPRAS)  # turn 1: succeeds
    game.feed("turn", "2")
    assert game.state(data, sampler).side_two.wish == (1, game.state(data, sampler).side_two.pokemon[0].maxhp // 2)
    game.feed("move", LAPRAS, "Wish", "", "[still]")  # turn 2: fails, one is pending
    game.feed("-fail", LAPRAS)
    game.feed("-heal", LAPRAS, "100/100", "[from] move: Wish", "[wisher] Lapras")
    game.feed("turn", "3")
    assert game.state(data, sampler).side_two.wish == (0, 0)
    game.feed("move", LAPRAS, "Wish", LAPRAS)  # turn 3: succeeds again
    game.feed("turn", "4")
    assert game.state(data, sampler).side_two.wish[0] == 1
    assert game.state(data, sampler).side_one.wish == (0, 0)  # our side never used it


def test_wish_pending_is_per_side():
    tracker = sleep_module.SleepTracker()
    tracker.observe(["", "turn", "5"])
    tracker.observe(["", "move", "p1a: Umbreon", "Wish", "p1a: Umbreon"])
    tracker.observe(["", "turn", "6"])
    assert tracker.wish_pending("p1", 6) == "p1: Umbreon"
    assert tracker.wish_pending("p2", 6) is None
    assert tracker.wish_pending("p1", 7) is None


def test_fake_out_state_after_a_move_is_not_a_first_turn(data, sampler):
    game = Game("battle-gen9randombattle-fail-2")
    game.feed("move", LAPRAS, "Freeze-Dry", TUSK)
    game.feed("turn", "2")
    assert game.state(data, sampler).side_two.last_used_move.startswith("move:")
