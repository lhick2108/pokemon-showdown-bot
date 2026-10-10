"""Run head-to-head benchmarks against a local Pokémon Showdown server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from collections.abc import Sequence

from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.ps_client import LocalhostServerConfiguration


# Add new player classes here as additional bot implementations become available.
PLAYER_REGISTRY: dict[str, type[Player]] = {
    "random": RandomPlayer,
    "maxpower": MaxBasePowerPlayer,
    "heuristics": SimpleHeuristicsPlayer,
}


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
    parser.add_argument("--timeout", type=_positive_seconds, default=600)
    parser.add_argument("--connect-timeout", type=_positive_seconds, default=30)
    return parser


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


async def _play_battles(
    p1_name: str,
    p2_name: str,
    n: int,
    battle_format: str,
    concurrency: int,
    connect_timeout: float,
) -> int:
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
        await p1.battle_against(p2, n_battles=n)

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
                ),
                timeout=args.timeout,
            )
        )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
