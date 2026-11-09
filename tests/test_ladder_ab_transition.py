"""Second-game transition of the A/B ladder mode (bug: game 2 stalled mid-game)."""

import asyncio

import pytest

from showdown_bot import ladder, ladder_ab


@pytest.fixture
def pev(tmp_path):
    a, b = tmp_path / "a.pev", tmp_path / "b.pev"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    return a, b


class _Mon:
    base_power, type = 50, None


class _Btl:
    turn = 4
    opponent_active_pokemon = None
    available_moves = [_Mon()]

    def __init__(self, tag):
        self.battle_tag = tag


def _stack(pev, base):
    """The exact wrapper order scripts/ladder.py uses: arm switch under the ladder hooks."""
    arms = ladder_ab.parse_arms([f"a={pev[0]}:scale=1.5", f"b={pev[1]}:scale=2"])
    ctl = ladder_ab.ArmController(arms, seed=1)
    cls = ladder.with_ladder_hooks(ladder_ab.with_arm_switch(base, ctl), move_cap_s=0.3)
    return cls(), ctl


def test_second_game_with_other_arm_keeps_moving(pev):
    """Game 2 (a different arm) must decide every turn, also after a cap hit or a crash."""
    seen = []

    class Base:
        config = None
        mode = "ok"

        async def choose_move(self, battle):
            seen.append((battle.battle_tag, self.config.evaluator_scale))
            if self.mode == "slow":
                await asyncio.sleep(30)
            if self.mode == "boom":
                raise RuntimeError("search exploded")
            return "searched"

        def create_order(self, move):
            return "heuristic"

        def choose_random_move(self, battle):
            return "random"

        def _battle_finished_callback(self, battle):
            pass

    p, ctl = _stack(pev, Base)

    async def play():
        out = []
        ctl.prepare_next()
        out += [await p.choose_move(_Btl("g1")) for _ in range(3)]
        p._battle_finished_callback(_Btl("g1"))
        ctl.prepare_next()
        out.append(await p.choose_move(_Btl("g2")))
        p.mode = "slow"  # mid-game stall of the new arm's search -> capped fallback
        out.append(await p.choose_move(_Btl("g2")))
        p.mode = "boom"  # mid-game exception -> fallback, never a silent stall
        out.append(await p.choose_move(_Btl("g2")))
        return out

    out = asyncio.run(asyncio.wait_for(play(), 10))
    assert out == ["searched"] * 3 + ["searched", "heuristic", "heuristic"]
    assert ctl.get("g1") is not ctl.get("g2")
    assert {s for t, s in seen if t == "g2"} == {ctl.get("g2").scale}


def test_request_handler_crash_still_answers_with_default_choice(pev):
    sent = []

    class Client:
        async def send_message(self, message, room=None):
            sent.append((message, room))

    class Base:
        config = None
        ps_client = Client()

        async def _handle_battle_message(self, split_messages):
            raise KeyError("unknown move")  # e.g. poke-env cannot parse the request

        def _battle_finished_callback(self, battle):
            pass

    p, _ = _stack(pev, Base)
    chunk = [[">battle-gen9randombattle-7"], ["", "request", '{"active": []}']]
    asyncio.run(p._handle_battle_message(chunk))
    assert sent == [("/choose default", "battle-gen9randombattle-7")]
    with pytest.raises(KeyError):  # a non-request chunk still surfaces the error
        asyncio.run(p._handle_battle_message([[">battle-1"], ["", "move", "x"]]))


def test_rejected_choice_is_retried_with_random_order(pev):
    class Base:
        config = None
        calls = 0

        async def choose_move(self, battle):
            Base.calls += 1
            return "same-invalid-order"

        def choose_random_move(self, battle):
            return "random"

        def _battle_finished_callback(self, battle):
            pass

        async def _handle_battle_message(self, split_messages):
            pass

    p, ctl = _stack(pev, Base)
    asyncio.run(p._handle_battle_message(
        [[">battle-1"], ["", "error", "[Invalid choice] Can't move: nope"]]))
    assert asyncio.run(p.choose_move(_Btl("battle-1"))) == "random"
    assert asyncio.run(p.choose_move(_Btl("battle-1"))) == "same-invalid-order"
    assert Base.calls == 1
