"""Process-pool worker for MCTS (imports nothing heavy besides poke_engine)."""

from __future__ import annotations


def ping() -> int:
    """Warm a worker process (imports poke_engine); returns its pid."""
    import os

    import poke_engine  # noqa: F401

    return os.getpid()


def search_world(
    state_string: str, search_time_ms: int, threads: int
) -> tuple[int, list[tuple[str, int, float]]]:
    """Run MCTS on one serialised world.

    Returns ``(total_visits, [(our_choice, visits, total_score), ...])``.
    Plain tuples keep the result cheap to pickle. A Rust panic surfaces in
    Python as ``pyo3_runtime.PanicException`` (a ``BaseException``), so it is
    converted to an ordinary ``RuntimeError`` here.
    """
    from poke_engine import State, monte_carlo_tree_search

    try:
        state = State.from_string(state_string)
        result = monte_carlo_tree_search(state, search_time_ms, threads=threads)
        return (
            int(result.total_visits),
            [
                (option.move_choice, int(option.visits), float(option.total_score))
                for option in result.side_one
            ],
        )
    except Exception:
        raise
    except BaseException as exc:  # pyo3 PanicException
        raise RuntimeError(f"engine panic: {exc!r}") from None
