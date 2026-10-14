"""Fair-compute preflight and progress/throughput reporting shared by duel and league.

Searches are time based (``time_ms`` per world in a process-pool worker), so every
world job needs its own free core for the whole budget. Too many workers for the
machine, or more world jobs queued than workers, starves the search: fewer iterations
per world (distorted strengths) and queue-inflated wall time per decision.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

RESERVED_CORES = 2  # main process + the node Showdown server


def preflight(
    workers: int, demand: int, cores: int | None = None, strict: bool = False
) -> tuple[list[str], list[str], bool]:
    """Check pool ``workers`` and peak concurrent world ``demand`` against the cores.

    Returns ``(info lines, warnings, refuse)``. ``refuse`` is set only with ``strict``
    when ``workers > cores - 2``; world demand above workers is always only a warning.
    """
    cores = cores if cores is not None else os.cpu_count() or 1
    info = [f"compute preflight: {workers} pool workers, {cores} cores "
            f"(reserve {RESERVED_CORES} for main process + server), "
            f"peak concurrent world jobs {demand}"]
    warnings: list[str] = []
    budget = max(cores - RESERVED_CORES, 1)
    if workers > cores:
        warnings.append(f"WARNING: {workers} workers > {cores} cores: searches share cores, "
                        "iterations per world drop and results are distorted")
    elif workers > budget:
        warnings.append(f"WARNING: {workers} workers leaves fewer than {RESERVED_CORES} of "
                        f"{cores} cores for the main process and the server")
    if demand > workers:
        warnings.append(
            f"WARNING: peak demand {demand} world jobs > {workers} workers: jobs queue, "
            "wall time per decision inflates (about "
            f"{-(-demand // max(workers, 1))}x the per-world budget)")
    refuse = strict and workers > cores - RESERVED_CORES
    if refuse:
        warnings.append(f"--strict-compute: refusing to start, workers {workers} > cores "
                        f"{cores} - {RESERVED_CORES} = {cores - RESERVED_CORES}")
    return info, warnings, refuse


def iterations_line(label: str, iterations: int, search_ms: int, **extra: int) -> str:
    """One throughput line: MCTS iterations per second of per-world budget."""
    rate = iterations / (search_ms / 1000) if search_ms else 0
    tail = "".join(f" {k}={v}" for k, v in extra.items())
    return (f"{label}: {rate:,.0f} iterations per second of per-world budget "
            f"({iterations:,} iterations over {search_ms / 1000:.0f} s){tail}")


def progress_line(
    done: int, total: int, label: str, wins: int, losses: int, ties: int = 0,
    paired: str | None = None, now: float | None = None,
) -> str:
    """``[12:03:41] 37/300 games  net 19-18 (51.4%)  paired 15 pairs ...``."""
    stamp = time.strftime("%H:%M:%S", time.localtime(now))
    played = wins + losses + ties
    score = (wins + 0.5 * ties) / played if played else 0.0
    tie_text = f"-{ties}" if ties else ""
    line = f"[{stamp}] {done}/{total} games  {label} {wins}-{losses}{tie_text} ({score:.1%})"
    return line + (f"  {paired}" if paired else "")


class ProgressPrinter:
    """Prints a flushed progress line whenever the finished count changes."""

    def __init__(self, total: int, out=None) -> None:
        self.total = total
        self.last = -1
        self.out = out
        self.changed = False  # whether the last ``update`` printed

    def update(self, done: int, label: str, wins: int, losses: int, ties: int = 0,
               paired: str | None = None) -> str | None:
        self.changed = False
        if done == self.last or done <= 0:
            return None
        self.last = done
        self.changed = True
        line = progress_line(done, self.total, label, wins, losses, ties, paired)
        print(line, file=self.out, flush=True)
        return line


def paired_text(rows: Sequence[tuple[int, float]]) -> str | None:
    """Short ``paired N pairs mean% [lo, hi]`` text from ``(pair id, score)`` rows."""
    from showdown_bot import teams as teams_module

    scores = teams_module.pair_scores(rows)
    if not scores:
        return None
    s = teams_module.paired_summary(scores)
    if s["pairs"] < 2:
        return f"paired {s['pairs']} pair {s['mean']:.1%}"
    return (f"paired {s['pairs']} pairs {s['mean']:.1%} "
            f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}]")
