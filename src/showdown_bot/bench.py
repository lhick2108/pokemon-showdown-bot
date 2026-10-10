"""Run head-to-head benchmarks against a local Pokémon Showdown server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from poke_env import to_id_str
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import LocalhostServerConfiguration

from showdown_bot.players.search import SearchPlayer


# Add new player classes here as additional bot implementations become available.
PLAYER_REGISTRY: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristics": SimpleHeuristicsPlayer,
    "search": SearchPlayer,
}

# "foulplay" is a special opponent name: it is not a poke-env Player subclass.
# When either --p1 or --p2 is "foulplay", bench launches Foul Play
# (vendor/foul-play, a separate GPL-3.0 process) as a subprocess in
# accept_challenge mode, and a real poke-env player challenges it.
FOULPLAY_NAME = "foulplay"

REPO_ROOT = Path(__file__).resolve().parents[2]
FOULPLAY_DIR = REPO_ROOT / "vendor" / "foul-play"
FOULPLAY_LOG_DIR = REPO_ROOT / ".cache" / "bench-logs"


def get_player_class(name: str) -> type[Player]:
    """Return a registered player class by its CLI name."""
    try:
        return PLAYER_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(PLAYER_REGISTRY))
        raise ValueError(
            f"Unknown player {name!r}. Available players: {available}"
        ) from None


def wilson_interval(wins: int, n: int) -> tuple[float, float]:
    """Return the two-sided 95% Wilson score interval for a win proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= wins <= n:
        raise ValueError("wins must be between 0 and n")

    z = 1.96
    proportion = wins / n
    z_squared = z * z
    denominator = 1 + z_squared / n
    center = (proportion + z_squared / (2 * n)) / denominator
    margin = (
        z
        * (proportion * (1 - proportion) / n + z_squared / (4 * n * n)) ** 0.5
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return parsed


def _player_name(value: str) -> str:
    if value == FOULPLAY_NAME:
        return value
    try:
        get_player_class(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", required=True, type=_player_name)
    parser.add_argument("--p2", required=True, type=_player_name)
    parser.add_argument("--n", required=True, type=_positive_int)
    parser.add_argument("--format", default="gen9randombattle")
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument(
        "--timeout",
        type=_positive_seconds,
        default=None,
        help="Optional limit on the whole run in seconds (default: no limit)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=_positive_seconds,
        default=30,
        help="Login and Foul Play's first readiness, in seconds",
    )
    parser.add_argument(
        "--game-timeout",
        type=_positive_seconds,
        default=900,
        help="Limit on each game in seconds (default: 900)",
    )
    parser.add_argument(
        "--fp-search-time-ms",
        type=_positive_int,
        default=100,
        help="Foul Play --search-time-ms (time to search per state, in ms)",
    )
    parser.add_argument(
        "--fp-parallelism",
        type=_positive_int,
        default=1,
        help="Foul Play --search-parallelism (states searched in parallel)",
    )
    parser.add_argument(
        "--fp-threads",
        type=_positive_int,
        default=1,
        help="Foul Play --search-threads (threads used per state)",
    )
    return parser


def _compute_description(
    player_name: str,
    fp_search_time_ms: int,
    fp_parallelism: int,
    fp_threads: int,
) -> str:
    """Return a one-line description of a side's compute budget."""
    if player_name == FOULPLAY_NAME:
        return (
            f"search_time_ms={fp_search_time_ms}, "
            f"parallelism={fp_parallelism}, threads={fp_threads}"
        )
    return "heuristic policy (no search)"


async def _wait_for_player_logins(players: Sequence[Player], timeout: float) -> None:
    async def wait_for_login(player: Player) -> None:
        async def wait_on_client_loop() -> None:
            await player.ps_client.logged_in.wait()

        future = asyncio.run_coroutine_threadsafe(
            wait_on_client_loop(), player.ps_client.loop
        )
        await asyncio.wrap_future(future)

    try:
        await asyncio.wait_for(
            asyncio.gather(*(wait_for_login(player) for player in players)),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Connection/login timed out after {timeout:g} seconds"
        ) from exc


async def _disconnect_player(player: Player) -> None:
    try:
        ps_client = player.ps_client
        if hasattr(ps_client, "websocket"):
            await asyncio.wait_for(ps_client.stop_listening(), timeout=2)
        else:
            listener = ps_client._listening_coroutine
            listener.cancel()
            await asyncio.wait_for(
                asyncio.gather(
                    asyncio.wrap_future(listener), return_exceptions=True
                ),
                timeout=0.5,
            )
    except Exception:
        # Cleanup errors must not replace a battle failure or timeout.
        pass


def _foulplay_python_executable() -> Path:
    venv_dir = FOULPLAY_DIR / ".venv"
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _foulplay_username() -> str:
    """A short, unique Showdown username for a Foul Play process instance."""
    return "fp" + secrets.token_hex(4)


def build_foulplay_command(
    python_executable: Path | str,
    fp_username: str,
    n: int,
    battle_format: str,
    search_time_ms: int,
    parallelism: int,
    threads: int,
    log_level: str = "INFO",
) -> list[str]:
    """Build the argv for launching Foul Play as a local accept_challenge bot."""
    return [
        str(python_executable),
        "run.py",
        "--websocket-uri",
        "local",
        "--ps-username",
        fp_username,
        "--bot-mode",
        "accept_challenge",
        "--pokemon-format",
        battle_format,
        "--run-count",
        str(n),
        "--search-time-ms",
        str(search_time_ms),
        "--search-parallelism",
        str(parallelism),
        "--search-threads",
        str(threads),
        "--log-level",
        log_level,
    ]


_JOB_HANDLE = None


def _create_kill_on_close_job():
    """Create (once) a Windows job object that kills its processes when bench dies.

    Foul Play's venv python.exe is a launcher that spawns the real interpreter,
    and a hard kill of bench would otherwise orphan both. Processes assigned to
    this job are terminated by the OS when the last handle to the job closes,
    which includes bench being killed. Returns None off Windows or on failure.
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return None
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    try:
        import ctypes
        from ctypes import wintypes

        class _BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in (
                "ReadOps", "WriteOps", "OtherOps",
                "ReadBytes", "WriteBytes", "OtherBytes",
            )]

        class _ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
        ]
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)  # ExtendedLimitInformation
        )
        if not ok:
            return None
        _JOB_HANDLE = job
        return job
    except Exception:
        return None


def _assign_to_kill_on_close_job(process) -> None:
    """Best-effort: tie a child process (and its descendants) to bench's lifetime."""
    job = _create_kill_on_close_job()
    if job is None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(int(process._handle)))
    except Exception:
        pass


def _taskkill_tree(process) -> None:
    """Forcefully kill a running process and all its descendants (Windows only)."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except Exception:
        pass


def _launch_foulplay_process(command: Sequence[str], log_path: Path):
    """Start Foul Play, capturing stdout/stderr to log_path. Returns (process, log_file)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(FOULPLAY_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        log_file.close()
        raise
    _assign_to_kill_on_close_job(process)
    return process, log_file


def _terminate_foulplay_process(process, log_file) -> None:
    """Ensure the Foul Play subprocess tree and its log file are never left behind.

    Safe to call whether the process is still running, already finished, or
    never fully started. Must be called from a finally block on every exit
    path (success, failure, timeout and interruption). The whole process tree
    is killed, since on Windows the venv launcher has a child interpreter.
    """
    try:
        if process.poll() is None:
            # Kill descendants first (Windows venv launcher stub -> real python),
            # while the launcher is still alive so its children can be found.
            if os.name == "nt":
                _taskkill_tree(process)
            try:
                process.terminate()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                pass
            if process.poll() is None:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
    finally:
        try:
            log_file.close()
        except Exception:
            pass


FOULPLAY_READY_MARKER = "waiting for a"


def _count_ready_markers(log_path: Path) -> int:
    try:
        contents = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return contents.lower().count(FOULPLAY_READY_MARKER)


async def _wait_for_foulplay_ready(
    process, log_path: Path, timeout: float, min_markers: int = 1
) -> None:
    """Wait until Foul Play is waiting to accept a challenge.

    Foul Play only reads challenge messages while it is idle, and it logs a
    "Waiting for a <format> challenge" line each time it becomes idle. A
    challenge sent before that line is lost, so each game waits for the
    ``min_markers``-th marker (1 for the first game, 2 for the second, ...).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        if process.poll() is not None:
            tail = ""
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            except OSError:
                pass
            raise RuntimeError(
                f"Foul Play exited early with code {process.returncode}. "
                f"See {log_path} for details.\n{tail}"
            )
        if _count_ready_markers(log_path) >= min_markers:
            return
        if loop.time() >= deadline:
            raise RuntimeError(
                f"Foul Play did not become ready to accept a challenge within "
                f"{timeout:g} seconds. See {log_path} for details."
            )
        await asyncio.sleep(0.2)


async def _play_against_foulplay(
    p1_name: str,
    p2_name: str,
    n: int,
    battle_format: str,
    connect_timeout: float,
    fp_search_time_ms: int,
    fp_parallelism: int,
    fp_threads: int,
    game_timeout: float = 900.0,
) -> int:
    """Play n battles between a poke-env bot and a Foul Play subprocess.

    Foul Play runs in accept_challenge mode and plays one battle at a time,
    so the poke-env bot challenges it sequentially. Returns p1's win count.
    """
    p1_is_fp = p1_name == FOULPLAY_NAME
    bot_name = p2_name if p1_is_fp else p1_name

    fp_username = _foulplay_username()
    command = build_foulplay_command(
        _foulplay_python_executable(),
        fp_username,
        n,
        battle_format,
        fp_search_time_ms,
        fp_parallelism,
        fp_threads,
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = FOULPLAY_LOG_DIR / f"foulplay-{fp_username}-{timestamp}.log"

    process, log_file = _launch_foulplay_process(command, log_path)
    bot: Player | None = None
    try:
        player_options = {
            "battle_format": battle_format,
            # Foul Play plays one battle at a time in accept_challenge mode.
            "max_concurrent_battles": 1,
            "server_configuration": LocalhostServerConfiguration,
            "log_level": logging.ERROR,
            "open_timeout": connect_timeout,
        }
        bot = get_player_class(bot_name)(**player_options)

        await _wait_for_player_logins([bot], connect_timeout)

        # Sequential challenges: one battle at a time, each only once Foul Play
        # has gone back to waiting (otherwise the challenge is silently lost).
        # Readiness for game k includes playing game k-1, so only the first
        # wait is bounded by the connect timeout; later ones by the game timeout.
        for game in range(1, n + 1):
            await _wait_for_foulplay_ready(
                process,
                log_path,
                connect_timeout if game == 1 else game_timeout,
                min_markers=game,
            )
            # send_challenges returns once the battle has finished, so this
            # also bounds the last game's completion before counting it.
            try:
                await asyncio.wait_for(
                    bot.send_challenges(to_id_str(fp_username), 1),
                    timeout=game_timeout,
                )
            except TimeoutError:
                raise RuntimeError(
                    f"Game {game}/{n} did not finish within {game_timeout:g} "
                    f"seconds (--game-timeout)"
                ) from None

        bot_finished = bot.n_finished_battles
        if bot_finished != n:
            raise RuntimeError(
                f"Expected {n} completed battles, but {bot_name} completed "
                f"{bot_finished}"
            )
        bot_wins = bot.n_won_battles
    finally:
        _terminate_foulplay_process(process, log_file)
        if bot is not None:
            try:
                await _disconnect_player(bot)
            except Exception:
                pass

    return n - bot_wins if p1_is_fp else bot_wins


async def _play_battles(
    p1_name: str,
    p2_name: str,
    n: int,
    battle_format: str,
    concurrency: int,
    connect_timeout: float,
    fp_search_time_ms: int = 100,
    fp_parallelism: int = 1,
    fp_threads: int = 1,
    game_timeout: float = 900.0,
) -> int:
    p1_is_fp = p1_name == FOULPLAY_NAME
    p2_is_fp = p2_name == FOULPLAY_NAME
    if p1_is_fp and p2_is_fp:
        raise ValueError("Only one of --p1/--p2 may be 'foulplay'")
    if p1_is_fp or p2_is_fp:
        return await _play_against_foulplay(
            p1_name,
            p2_name,
            n,
            battle_format,
            connect_timeout,
            fp_search_time_ms,
            fp_parallelism,
            fp_threads,
            game_timeout,
        )

    player_options = {
        "battle_format": battle_format,
        "max_concurrent_battles": concurrency,
        "server_configuration": LocalhostServerConfiguration,
        "log_level": logging.ERROR,
        "open_timeout": connect_timeout,
    }
    players: list[Player] = []
    try:
        p1 = get_player_class(p1_name)(**player_options)
        players.append(p1)
        p2 = get_player_class(p2_name)(**player_options)
        players.append(p2)

        await _wait_for_player_logins(players, connect_timeout)
        # Games run `concurrency` at a time, so allow one game timeout per wave.
        waves = -(-n // concurrency)
        try:
            await asyncio.wait_for(
                p1.battle_against(p2, n_battles=n), timeout=game_timeout * waves
            )
        except TimeoutError:
            raise RuntimeError(
                f"{n} games did not finish within {game_timeout * waves:g} seconds "
                f"({game_timeout:g} per wave of {concurrency}; --game-timeout)"
            ) from None

        p1_finished = p1.n_finished_battles
        p2_finished = p2.n_finished_battles
        if p1_finished != n or p2_finished != n:
            raise RuntimeError(
                f"Expected {n} completed battles, but p1 completed {p1_finished} "
                f"and p2 completed {p2_finished}"
            )
        return p1.n_won_battles
    finally:
        try:
            await asyncio.gather(*(_disconnect_player(player) for player in players))
        except Exception:
            # Preserve the original run result if websocket cleanup is interrupted.
            pass


async def run_benchmark(
    p1_name: str,
    p2_name: str,
    n: int,
    battle_format: str = "gen9randombattle",
    concurrency: int = 4,
    connect_timeout: float = 30,
    fp_search_time_ms: int = 100,
    fp_parallelism: int = 1,
    fp_threads: int = 1,
    game_timeout: float = 900.0,
) -> int:
    """Play N local battles and return player one's win count."""
    try:
        return await _play_battles(
            p1_name,
            p2_name,
            n,
            battle_format,
            concurrency,
            connect_timeout,
            fp_search_time_ms,
            fp_parallelism,
            fp_threads,
            game_timeout,
        )
    except Exception as exc:
        raise RuntimeError(f"Battle run failed: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        wins = asyncio.run(
            asyncio.wait_for(
                run_benchmark(
                    args.p1,
                    args.p2,
                    args.n,
                    battle_format=args.format,
                    concurrency=args.concurrency,
                    connect_timeout=args.connect_timeout,
                    fp_search_time_ms=args.fp_search_time_ms,
                    fp_parallelism=args.fp_parallelism,
                    fp_threads=args.fp_threads,
                    game_timeout=args.game_timeout,
                ),
                timeout=args.timeout,
            )
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted", file=sys.stderr)
        return 130
    except TimeoutError:
        print(
            f"Benchmark timed out after {args.timeout:g} seconds", file=sys.stderr
        )
        return 1
    except Exception as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1

    win_rate = wins / args.n
    lower, upper = wilson_interval(wins, args.n)
    print(
        f"{args.p1} wins: {wins}/{args.n} ({win_rate:.1%}); "
        f"Wilson 95% CI: [{lower:.1%}, {upper:.1%}]"
    )
    print(
        f"{args.p1} compute: "
        f"{_compute_description(args.p1, args.fp_search_time_ms, args.fp_parallelism, args.fp_threads)}"
    )
    print(
        f"{args.p2} compute: "
        f"{_compute_description(args.p2, args.fp_search_time_ms, args.fp_parallelism, args.fp_threads)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
