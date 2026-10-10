import asyncio
import socket

import pytest
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

import showdown_bot.bench as bench
from showdown_bot.bench import get_player_class, main, wilson_interval


def test_wilson_interval_for_half_wins_is_about_40_to_60_percent():
    lower, upper = wilson_interval(50, 100)

    assert lower == pytest.approx(0.4038, abs=0.0002)
    assert upper == pytest.approx(0.5962, abs=0.0002)


@pytest.mark.parametrize(("wins", "n"), [(0, 1), (1, 1), (0, 10), (10, 10)])
def test_wilson_interval_bounds_and_contains_proportion(wins, n):
    lower, upper = wilson_interval(wins, n)

    assert 0 <= lower <= wins / n <= upper <= 1


@pytest.mark.parametrize(("wins", "n"), [(0, 0), (-1, 10), (11, 10)])
def test_wilson_interval_rejects_invalid_counts(wins, n):
    with pytest.raises(ValueError):
        wilson_interval(wins, n)


def test_player_registry_resolves_baseline_players():
    assert get_player_class("random") is RandomPlayer
    assert get_player_class("maxpower") is MaxBasePowerPlayer
    assert get_player_class("heuristics") is SimpleHeuristicsPlayer


def test_player_registry_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown player"):
        get_player_class("unknown")


def test_main_returns_nonzero_when_server_connection_times_out(monkeypatch, capsys):
    with socket.socket() as unused_socket:
        unused_socket.bind(("127.0.0.1", 0))
        unused_port = unused_socket.getsockname()[1]

    monkeypatch.setattr(
        bench,
        "LocalhostServerConfiguration",
        ServerConfiguration(
            websocket_url=f"ws://127.0.0.1:{unused_port}/showdown/websocket",
            authentication_url="http://127.0.0.1/",
        ),
    )

    result = main(
        [
            "--p1",
            "random",
            "--p2",
            "random",
            "--n",
            "1",
            "--timeout",
            "0.2",
            "--connect-timeout",
            "1",
        ]
    )

    assert result != 0
    assert "timed out" in capsys.readouterr().err.lower()


def test_main_times_out_during_disconnect_without_printing_success(
    monkeypatch, capsys
):
    class SlowDisconnectClient:
        websocket = object()

        async def stop_listening(self):
            await asyncio.sleep(1)

    class InstantPlayer:
        def __init__(self, **_kwargs):
            self.ps_client = SlowDisconnectClient()
            self.n_finished_battles = 1
            self.n_won_battles = 0

        async def battle_against(self, _opponent, n_battles):
            assert n_battles == 1

    async def skip_login_wait(_players, _timeout):
        pass

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "random", InstantPlayer)
    monkeypatch.setattr(bench, "_wait_for_player_logins", skip_login_wait)

    result = main(
        [
            "--p1",
            "random",
            "--p2",
            "random",
            "--n",
            "1",
            "--timeout",
            "0.05",
        ]
    )

    captured = capsys.readouterr()
    assert result != 0
    assert " wins: " not in captured.out


def test_main_stops_pending_listener_when_connection_never_completes(monkeypatch):
    with socket.socket() as unused_socket:
        unused_socket.bind(("127.0.0.1", 0))
        unused_port = unused_socket.getsockname()[1]

    class RecordingRandomPlayer(RandomPlayer):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ps_client.logger.disabled = True
            self.instances.append(self)

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "random", RecordingRandomPlayer)
    monkeypatch.setattr(
        bench,
        "LocalhostServerConfiguration",
        ServerConfiguration(
            websocket_url=f"ws://127.0.0.1:{unused_port}/showdown/websocket",
            authentication_url="http://127.0.0.1/",
        ),
    )

    result = main(
        [
            "--p1",
            "random",
            "--p2",
            "random",
            "--n",
            "1",
            "--timeout",
            "2",
            "--connect-timeout",
            "0.05",
        ]
    )

    assert result != 0
    assert RecordingRandomPlayer.instances
    assert all(
        player.ps_client._listening_coroutine.done()
        for player in RecordingRandomPlayer.instances
    )
    for player in RecordingRandomPlayer.instances:
        ps_client = player.ps_client

        async def has_pending_listener():
            return any(
                task.get_coro().cr_code.co_name == "listen"
                and task.get_coro().cr_frame is not None
                and task.get_coro().cr_frame.f_locals.get("self") is ps_client
                for task in asyncio.all_tasks()
            )

        pending = asyncio.run_coroutine_threadsafe(
            has_pending_listener(), ps_client.loop
        )
        assert not pending.result(timeout=1)
