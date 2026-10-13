"""Ladder request capture (saved into the battle log) and replay with our real sets."""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from battle_helpers import OUR_TEAM, request_for  # noqa: E402

import replay_log  # noqa: E402
from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot import ladder  # noqa: E402

REQ0 = request_for(active=0)
REQ1 = copy.deepcopy(REQ0)
REQ1["rqid"] = 7
REQ1["side"]["pokemon"][0]["item"] = "choicescarf"  # a real item the placeholder never had

TAG = "battle-gen9randombattle-1"


def raw(req) -> str:
    return json.dumps(req, separators=(",", ":"))


class _Battle:
    def __init__(self):
        self._replay_data = [["", "player"], ["", "start"]]


class _Base:
    battles: dict = {}

    async def _handle_battle_message(self, split_messages):
        return "handled"

    def _battle_finished_callback(self, battle):
        pass


def chunk(*lines: str) -> list[list[str]]:
    return [l.split("|") if l.startswith("|") else [l] for l in lines]


def test_capture_hook_stores_requests_in_order_and_passes_through():
    player = ladder.with_ladder_hooks(_Base)()
    player.battles = {TAG: _Battle()}
    out = asyncio.run(player._handle_battle_message(chunk(">" + TAG, "|request|" + raw(REQ0))))
    assert out == "handled"
    player.battles[TAG]._replay_data.append(["", "turn", "1"])
    asyncio.run(player._handle_battle_message(chunk(">" + TAG, "|request|" + raw(REQ1))))
    got = player.battle_requests[TAG]
    assert [pos for pos, _ in got] == [2, 3]
    assert json.loads(got[0][1]) == REQ0 and json.loads(got[1][1]) == REQ1


def test_capture_ignores_everything_but_requests():
    store: dict = {}
    ladder.capture_requests(store, chunk("|challstr|4|abcdef", "|updateuser| me|1|x"), {})
    ladder.capture_requests(store, chunk(">battle-x", "|move|p1a: A|Tackle|p2a: B",
                                         "|request|", "|win|me"), {})
    assert store == {}


def test_log_with_requests_format_and_order_and_no_secret():
    events = [">battle-x", "|start", "|turn|1", "|move|a", "|turn|2"]
    text = ladder.log_with_requests(events, [(3, raw(REQ0)), (5, raw(REQ1))])
    lines = text.split("\n")
    assert lines[:3] == [">battle-x", "|start", "|turn|1"]
    assert lines[3].startswith("|request|{") and json.loads(lines[3][len("|request|"):]) == REQ0
    assert lines[4] == "|move|a" and lines[5] == "|turn|2"
    assert lines[6].startswith("|request|{") and len(lines) == 7  # trailing request kept
    assert "password" not in text.lower() and "token" not in text.lower()


LOG_HEAD = [
    "|player|p1|bot|1|", "|player|p2|opp|1|", "|gametype|singles", "|gen|9",
    "|tier|[Gen 9] Random Battle", "|rule|x", "|start",
    "|switch|p1a: Great Tusk|Great Tusk, L80|100/100",
    "|switch|p2a: Kingambit|Kingambit, L78|100/100",
]


def make_log(with_requests: bool) -> list[str]:
    def req(r):
        return ["|request|" + raw(r)] if with_requests else []

    return (req(REQ0) + LOG_HEAD + ["|turn|1"] + req(REQ0)
            + ["|move|p1a: Great Tusk|Close Combat|p2a: Kingambit", "|-damage|p2a: Kingambit|60/100",
               "|turn|2"] + req(REQ1) + ["|move|p1a: Great Tusk|Close Combat|p2a: Kingambit"])


def test_replay_uses_real_request_at_the_right_turn():
    lines = make_log(True)
    assert replay_log.has_requests(lines)
    battle = replay_log.replay(lines, 2)
    mon = battle.active_pokemon
    assert battle.turn == 2
    assert mon.item == "choicescarf"  # from the turn-2 request, not a placeholder
    assert mon.ability == "protosynthesis"
    assert battle.opponent_active_pokemon.current_hp_fraction == 0.6
    earlier = replay_log.replay(lines, 1)
    assert earlier.active_pokemon.item == "leftovers"


def test_replay_without_requests_falls_back_to_synthesised_team():
    lines = make_log(False)
    assert not replay_log.has_requests(lines)
    team = replay_log.build_team(replay_log.our_species(lines), {}, {"Great Tusk": "Choice Band"})
    battle = replay_log.replay(lines, 2, team)
    assert battle.active_pokemon.item == "choiceband"


def test_real_request_gives_exact_team():
    battle = replay_log.replay(make_log(True), 2)
    assert sorted(p.species for p in battle.team.values()) == sorted(
        es.to_id(t[0]) if hasattr(es, "to_id") else t[0].lower().replace(" ", "") for t in OUR_TEAM
    )
