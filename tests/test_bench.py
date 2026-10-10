import asyncio
import socket
import subprocess

import pytest
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import ServerConfiguration

import showdown_bot.bench as bench
from showdown_bot.bench import (
    build_foulplay_command,
    get_player_class,
    main,
    wilson_interval,
)


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


# --- Foul Play opponent: CLI/option parsing ---------------------------------


def test_player_name_accepts_foulplay():
    parser = bench._build_parser()
    args = parser.parse_args(["--p1", "maxpower", "--p2", "foulplay", "--n", "1"])
    assert args.p1 == "maxpower"
    assert args.p2 == "foulplay"


def test_fp_compute_options_default_values():
    parser = bench._build_parser()
    args = parser.parse_args(["--p1", "maxpower", "--p2", "foulplay", "--n", "1"])
    assert args.fp_search_time_ms == 100
    assert args.fp_parallelism == 1
    assert args.fp_threads == 1


def test_fp_compute_options_parse_custom_values():
    parser = bench._build_parser()
    args = parser.parse_args(
        [
            "--p1",
            "maxpower",
            "--p2",
            "foulplay",
            "--n",
            "1",
            "--fp-search-time-ms",
            "250",
            "--fp-parallelism",
            "4",
            "--fp-threads",
            "2",
        ]
    )
    assert args.fp_search_time_ms == 250
    assert args.fp_parallelism == 4
    assert args.fp_threads == 2


@pytest.mark.parametrize("flag", ["--fp-search-time-ms", "--fp-parallelism", "--fp-threads"])
def test_fp_compute_options_reject_non_positive_values(flag):
    parser = bench._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--p1", "maxpower", "--p2", "foulplay", "--n", "1", flag, "0"]
        )


def test_compute_description_for_foulplay_reports_search_settings():
    description = bench._compute_description("foulplay", 250, 4, 2)
    assert description == "search_time_ms=250, parallelism=4, threads=2"


def test_compute_description_for_heuristic_player_has_no_search_settings():
    assert bench._compute_description("maxpower", 100, 1, 1) == (
        "heuristic policy (no search)"
    )


# --- Foul Play opponent: subprocess command construction --------------------


def test_build_foulplay_command_includes_compute_settings_and_identity():
    command = build_foulplay_command(
        r"C:\fake\.venv\Scripts\python.exe",
        "fpabcd1234",
        20,
        "gen9randombattle",
        search_time_ms=250,
        parallelism=4,
        threads=2,
        log_level="INFO",
    )

    assert command[0] == r"C:\fake\.venv\Scripts\python.exe"
    assert command[1] == "run.py"

    def value_after(flag):
        return command[command.index(flag) + 1]

    assert value_after("--websocket-uri") == "local"
    assert value_after("--ps-username") == "fpabcd1234"
    assert value_after("--bot-mode") == "accept_challenge"
    assert value_after("--pokemon-format") == "gen9randombattle"
    assert value_after("--run-count") == "20"
    assert value_after("--search-time-ms") == "250"
    assert value_after("--search-parallelism") == "4"
    assert value_after("--search-threads") == "2"
    assert value_after("--log-level") == "INFO"


def test_build_foulplay_command_uses_default_log_level():
    command = build_foulplay_command(
        "python.exe", "fpuser", 1, "gen9randombattle", 100, 1, 1
    )
    assert command[command.index("--log-level") + 1] == "INFO"


def test_foulplay_username_is_short_and_unique():
    first = bench._foulplay_username()
    second = bench._foulplay_username()
    assert first != second
    assert len(first) <= 18
    assert first.isalnum()


# --- Foul Play opponent: subprocess cleanup guarantees -----------------------


class _FakeProcess:
    """A minimal stand-in for subprocess.Popen used to test cleanup logic."""

    def __init__(self, exits_on_terminate=True):
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False
        self._exits_on_terminate = exits_on_terminate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        if self._exits_on_terminate:
            self.returncode = -15

    def kill(self):
        self.kill_called = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self.returncode


class _FakeLogFile:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_terminate_foulplay_process_terminates_a_running_process():
    process = _FakeProcess(exits_on_terminate=True)
    log_file = _FakeLogFile()

    bench._terminate_foulplay_process(process, log_file)

    assert process.terminate_called
    assert not process.kill_called
    assert log_file.closed


def test_terminate_foulplay_process_kills_if_terminate_does_not_stop_it():
    process = _FakeProcess(exits_on_terminate=False)
    log_file = _FakeLogFile()

    bench._terminate_foulplay_process(process, log_file)

    assert process.terminate_called
    assert process.kill_called
    assert log_file.closed


def test_terminate_foulplay_process_is_a_noop_on_an_already_exited_process():
    process = _FakeProcess()
    process.returncode = 0
    log_file = _FakeLogFile()

    bench._terminate_foulplay_process(process, log_file)

    assert not process.terminate_called
    assert not process.kill_called
    assert log_file.closed


class _InstantBot:
    """A fake poke-env player that never needs a network connection."""

    instances = []

    def __init__(self, **_kwargs):
        self.n_finished_battles = 0
        self.n_won_battles = 0
        _InstantBot.instances.append(self)

    async def send_challenges(self, _opponent, n_challenges):
        raise AssertionError("send_challenges should not be reached in this test")


@pytest.fixture
def no_real_subprocess(monkeypatch):
    """Patch Foul Play process launch/cleanup so no real process is spawned."""
    process = _FakeProcess(exits_on_terminate=True)
    log_file = _FakeLogFile()
    terminate_calls = []

    monkeypatch.setattr(
        bench, "_launch_foulplay_process", lambda *_a, **_k: (process, log_file)
    )
    monkeypatch.setattr(
        bench,
        "_terminate_foulplay_process",
        lambda p, lf: terminate_calls.append((p, lf)),
    )
    return process, log_file, terminate_calls


def test_play_against_foulplay_cleans_up_subprocess_on_login_error(
    monkeypatch, no_real_subprocess
):
    process, log_file, terminate_calls = no_real_subprocess
    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _InstantBot)

    async def failing_login_wait(_players, _timeout):
        raise RuntimeError("login boom")

    monkeypatch.setattr(bench, "_wait_for_player_logins", failing_login_wait)

    async def run():
        with pytest.raises(RuntimeError, match="login boom"):
            await bench._play_against_foulplay(
                "maxpower", "foulplay", 1, "gen9randombattle", 5, 100, 1, 1
            )

    asyncio.run(run())

    assert terminate_calls == [(process, log_file)]


def test_play_against_foulplay_cleans_up_subprocess_on_ready_timeout(
    monkeypatch, no_real_subprocess
):
    process, log_file, terminate_calls = no_real_subprocess
    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _InstantBot)

    async def instant_login_wait(_players, _timeout):
        pass

    async def timing_out_ready_wait(_process, _log_path, _timeout, min_markers=1):
        raise RuntimeError("Foul Play did not become ready to accept a challenge")

    monkeypatch.setattr(bench, "_wait_for_player_logins", instant_login_wait)
    monkeypatch.setattr(bench, "_wait_for_foulplay_ready", timing_out_ready_wait)

    async def run():
        with pytest.raises(RuntimeError, match="did not become ready"):
            await bench._play_against_foulplay(
                "maxpower", "foulplay", 1, "gen9randombattle", 5, 100, 1, 1
            )

    asyncio.run(run())

    assert terminate_calls == [(process, log_file)]


def test_play_against_foulplay_cleans_up_subprocess_on_battle_failure(
    monkeypatch, no_real_subprocess
):
    process, log_file, terminate_calls = no_real_subprocess

    class _BotThatFailsMidBattle(_InstantBot):
        async def send_challenges(self, _opponent, n_challenges):
            self.n_finished_battles = 0  # fewer than requested -> run_benchmark error

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _BotThatFailsMidBattle)

    async def instant_login_wait(_players, _timeout):
        pass

    async def instant_ready_wait(_process, _log_path, _timeout, min_markers=1):
        pass

    monkeypatch.setattr(bench, "_wait_for_player_logins", instant_login_wait)
    monkeypatch.setattr(bench, "_wait_for_foulplay_ready", instant_ready_wait)

    async def run():
        with pytest.raises(RuntimeError, match="Expected 1 completed battles"):
            await bench._play_against_foulplay(
                "maxpower", "foulplay", 1, "gen9randombattle", 5, 100, 1, 1
            )

    asyncio.run(run())

    assert terminate_calls == [(process, log_file)]


def test_play_battles_rejects_both_sides_being_foulplay():
    async def run():
        with pytest.raises(ValueError, match="Only one"):
            await bench._play_battles(
                "foulplay", "foulplay", 1, "gen9randombattle", 4, 5
            )

    asyncio.run(run())


def test_terminate_foulplay_process_kills_process_tree_on_windows(monkeypatch):
    process = _FakeProcess(exits_on_terminate=True)
    log_file = _FakeLogFile()
    killed = []
    monkeypatch.setattr(bench.os, "name", "nt")
    monkeypatch.setattr(bench, "_taskkill_tree", lambda p: killed.append(p))

    bench._terminate_foulplay_process(process, log_file)

    assert killed == [process]
    assert log_file.closed


def test_terminate_foulplay_process_skips_tree_kill_for_exited_process(monkeypatch):
    process = _FakeProcess()
    process.returncode = 0
    log_file = _FakeLogFile()
    killed = []
    monkeypatch.setattr(bench.os, "name", "nt")
    monkeypatch.setattr(bench, "_taskkill_tree", lambda p: killed.append(p))

    bench._terminate_foulplay_process(process, log_file)

    assert killed == []


def test_terminate_foulplay_process_closes_log_even_if_terminate_raises():
    class _Exploding(_FakeProcess):
        def terminate(self):
            raise OSError("boom")

    process = _Exploding(exits_on_terminate=False)
    log_file = _FakeLogFile()

    bench._terminate_foulplay_process(process, log_file)

    assert process.kill_called
    assert log_file.closed


def test_main_reports_keyboard_interrupt_and_returns_130(monkeypatch, capsys):
    def interrupted(coro):
        raise KeyboardInterrupt

    monkeypatch.setattr(bench.asyncio, "run", interrupted)
    monkeypatch.setattr(bench, "run_benchmark", lambda *_a, **_k: object())
    monkeypatch.setattr(bench.asyncio, "wait_for", lambda aw, timeout: aw)

    exit_code = bench.main(["--p1", "maxpower", "--p2", "foulplay", "--n", "1"])

    assert exit_code == 130
    assert "interrupted" in capsys.readouterr().err


def test_main_prints_compute_for_both_sides(monkeypatch, capsys):
    async def fake_run_benchmark(*_a, **_k):
        return 3

    monkeypatch.setattr(bench, "run_benchmark", fake_run_benchmark)

    exit_code = bench.main(
        [
            "--p1", "maxpower", "--p2", "foulplay", "--n", "4",
            "--fp-search-time-ms", "250", "--fp-parallelism", "2", "--fp-threads", "3",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "maxpower wins: 3/4" in out
    assert "Wilson 95% CI" in out
    assert "maxpower compute: heuristic policy (no search)" in out
    assert "foulplay compute: search_time_ms=250, parallelism=2, threads=3" in out


def test_play_against_foulplay_challenges_sequentially_after_each_ready_marker(
    monkeypatch, no_real_subprocess
):
    events = []

    class _CountingBot(_InstantBot):
        async def send_challenges(self, opponent, n_challenges):
            events.append(("challenge", n_challenges))
            self.n_finished_battles += n_challenges
            self.n_won_battles += 1

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _CountingBot)

    async def instant_login_wait(_players, _timeout):
        pass

    async def recording_ready_wait(_process, _log_path, _timeout, min_markers=1):
        events.append(("ready", min_markers))

    monkeypatch.setattr(bench, "_wait_for_player_logins", instant_login_wait)
    monkeypatch.setattr(bench, "_wait_for_foulplay_ready", recording_ready_wait)

    wins = asyncio.run(
        bench._play_against_foulplay(
            "foulplay", "maxpower", 3, "gen9randombattle", 5, 100, 1, 1
        )
    )

    assert events == [
        ("ready", 1), ("challenge", 1),
        ("ready", 2), ("challenge", 1),
        ("ready", 3), ("challenge", 1),
    ]
    # p1 is foulplay, so its wins are the bot's losses.
    assert wins == 0


def test_wait_for_foulplay_ready_requires_the_requested_marker_count(tmp_path):
    log = tmp_path / "fp.log"
    log.write_text("INFO Waiting for a gen9randombattle challenge\n", encoding="utf-8")

    class _Running:
        returncode = None

        def poll(self):
            return None

    asyncio.run(bench._wait_for_foulplay_ready(_Running(), log, 1, min_markers=1))
    with pytest.raises(RuntimeError, match="did not become ready"):
        asyncio.run(bench._wait_for_foulplay_ready(_Running(), log, 0.3, min_markers=2))


def test_timeout_flags_parse_and_default():
    args = bench._build_parser().parse_args(
        ["--p1", "maxpower", "--p2", "foulplay", "--n", "1"]
    )
    assert args.timeout is None
    assert args.connect_timeout == 30
    assert args.game_timeout == 900

    args = bench._build_parser().parse_args(
        ["--p1", "maxpower", "--p2", "foulplay", "--n", "1",
         "--timeout", "5000", "--game-timeout", "120"]
    )
    assert args.timeout == 5000
    assert args.game_timeout == 120


def test_game_timeout_rejects_non_positive():
    with pytest.raises(SystemExit):
        bench._build_parser().parse_args(
            ["--p1", "maxpower", "--p2", "foulplay", "--n", "1", "--game-timeout", "0"]
        )


def test_slow_game_longer_than_connect_timeout_still_succeeds(
    monkeypatch, no_real_subprocess
):
    class _SlowBot(_InstantBot):
        async def send_challenges(self, _opponent, n_challenges):
            self.n_finished_battles += n_challenges
            self.n_won_battles += 1

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _SlowBot)
    timeouts = []

    async def instant_login_wait(_players, _timeout):
        pass

    async def slow_ready_wait(_process, _log_path, timeout, min_markers=1):
        timeouts.append(timeout)
        if min_markers > 1:
            # Marker arrives after connect_timeout (0.05) but within game_timeout.
            await asyncio.sleep(0.15)

    monkeypatch.setattr(bench, "_wait_for_player_logins", instant_login_wait)
    monkeypatch.setattr(bench, "_wait_for_foulplay_ready", slow_ready_wait)

    wins = asyncio.run(
        bench._play_against_foulplay(
            "maxpower", "foulplay", 2, "gen9randombattle", 0.05, 100, 1, 1,
            game_timeout=5,
        )
    )

    assert wins == 2
    assert timeouts == [0.05, 5]


def test_game_exceeding_game_timeout_fails_clearly_and_cleans_up(
    monkeypatch, no_real_subprocess
):
    process, log_file, terminate_calls = no_real_subprocess

    class _HungBot(_InstantBot):
        async def send_challenges(self, _opponent, n_challenges):
            await asyncio.sleep(30)

    monkeypatch.setitem(bench.PLAYER_REGISTRY, "maxpower", _HungBot)

    async def instant_login_wait(_players, _timeout):
        pass

    async def instant_ready_wait(_process, _log_path, _timeout, min_markers=1):
        pass

    monkeypatch.setattr(bench, "_wait_for_player_logins", instant_login_wait)
    monkeypatch.setattr(bench, "_wait_for_foulplay_ready", instant_ready_wait)

    async def run():
        with pytest.raises(RuntimeError, match="did not finish within 0.1 seconds"):
            await bench._play_against_foulplay(
                "maxpower", "foulplay", 1, "gen9randombattle", 5, 100, 1, 1,
                game_timeout=0.1,
            )

    asyncio.run(run())

    assert terminate_calls == [(process, log_file)]
