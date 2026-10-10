"""Process-pool worker for MCTS (imports nothing heavy besides poke_engine)."""

from __future__ import annotations


def ping() -> int:
    """Warm a worker process (imports poke_engine); returns its pid."""
    import os

    import poke_engine  # noqa: F401

    return os.getpid()


# Evaluators loaded in this worker process, keyed by path; ``EVALUATOR_LOADS``
# counts actual file loads (the cache must make it 1 per path per process).
_EVALUATORS: dict[str, object] = {}
EVALUATOR_LOADS = 0


def get_evaluator(path: str):
    """The ``poke_engine.Evaluator`` for ``path``, loaded once per process."""
    global EVALUATOR_LOADS
    evaluator = _EVALUATORS.get(path)
    if evaluator is None:
        from poke_engine import Evaluator

        evaluator = Evaluator.load(path)
        EVALUATOR_LOADS += 1
        _EVALUATORS[path] = evaluator
    return evaluator


def search_world(
    state_string: str, search_time_ms: int, threads: int, evaluator_path: str | None = None
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
        if evaluator_path is None:
            result = monte_carlo_tree_search(state, search_time_ms, threads=threads)
        else:
            result = monte_carlo_tree_search(
                state, search_time_ms, threads=threads, evaluator=get_evaluator(evaluator_path)
            )
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


def evaluator_loads() -> int:
    """How many evaluator files this process has loaded (tests; picklable)."""
    return EVALUATOR_LOADS
