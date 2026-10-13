"""Process-pool worker for MCTS (imports nothing heavy besides poke_engine)."""

from __future__ import annotations


def ping() -> int:
    """Warm a worker process (imports poke_engine); returns its pid."""
    import os

    import poke_engine  # noqa: F401

    return os.getpid()


# Evaluators loaded in this worker process, keyed by (path, leaf scale); ``EVALUATOR_LOADS``
# counts actual file loads (the cache must make it 1 per path per process).
_EVALUATORS: dict[tuple[str, float], object] = {}
EVALUATOR_LOADS = 0


def get_evaluator(path: str, scale: float = 1.0):
    """The ``poke_engine.Evaluator`` for ``(path, scale)``, loaded once per process."""
    global EVALUATOR_LOADS
    key = (path, float(scale))
    evaluator = _EVALUATORS.get(key)
    if evaluator is None:
        from poke_engine import Evaluator

        evaluator = Evaluator.load(path, leaf_scale=float(scale))
        EVALUATOR_LOADS += 1
        _EVALUATORS[key] = evaluator
    return evaluator


def search_world(
    state_string: str,
    search_time_ms: int,
    threads: int,
    evaluator_path: str | None = None,
    evaluator_scale: float = 1.0,
    selection: str = "ucb1",
    selection_gamma: float = 0.1,
    joint: bool = False,
    eval_knee: float = 0.0,
    eval_boost_scale: float = 1.0,
    opp_names: bool = False,
) -> tuple:
    """Run MCTS on one serialised world.

    Returns ``(total_visits, [(our_choice, visits, total_score), ...])``; for
    ``selection`` other than ucb1 a third element is appended: our average-strategy
    share per option (aligned with the list).
    With ``joint`` (Nash mode) the result is always a 4-tuple: the third element is the
    strategy list or ``None`` (ucb1), the fourth ``(k2, [(i, j, visits, mean), ...])``,
    the root joint-action matrix (``k2`` = the opponent's option count).
    ``eval_boost_scale`` multiplies every stat-boost term (1.0 = stock).
    ``eval_knee`` > 0 enables the experimental low-HP evaluation discount (bead 589.32; needs
    an engine build that has ``set_eval_low_hp_knee``); 0.0 is the stock evaluation. The
    setting is process-wide in the engine, so it is applied before every search.
    With ``opp_names`` (tendency model, needs ``joint``) a fifth element lists the
    opponent's option choice strings, aligned with the joint matrix columns.
    Plain tuples keep the result cheap to pickle. A Rust panic surfaces in
    Python as ``pyo3_runtime.PanicException`` (a ``BaseException``), so it is
    converted to an ordinary ``RuntimeError`` here.
    """
    import poke_engine
    from poke_engine import State, monte_carlo_tree_search

    try:
        setter = getattr(poke_engine, "set_eval_low_hp_knee", None)
        if setter is not None:
            setter(float(eval_knee))
            poke_engine.set_eval_boost_scale(float(eval_boost_scale))
        elif eval_knee or eval_boost_scale != 1.0:
            raise RuntimeError("this poke_engine build has no 589.32 eval switches (wheel needed)")
        state = State.from_string(state_string)
        kwargs: dict = {}
        if evaluator_path is not None:
            kwargs["evaluator"] = get_evaluator(evaluator_path, evaluator_scale)
        if selection != "ucb1":
            kwargs["selection"] = selection
            kwargs["gamma"] = float(selection_gamma)
        if joint:
            kwargs["return_joint"] = True
        result = monte_carlo_tree_search(state, search_time_ms, threads=threads, **kwargs)
        options = [
            (option.move_choice, int(option.visits), float(option.total_score))
            for option in result.side_one
        ]
        if joint:
            cells = [(int(i), int(j), int(v), float(m)) for i, j, v, m in result.joint]
            strategies = (
                [float(option.avg_strategy) for option in result.side_one]
                if selection != "ucb1"
                else None
            )
            out = (int(result.total_visits), options, strategies, (len(result.side_two), cells))
            if opp_names:
                out += ([str(option.move_choice) for option in result.side_two],)
            return out
        if selection != "ucb1":
            return (
                int(result.total_visits),
                options,
                [float(option.avg_strategy) for option in result.side_one],
            )
        return (int(result.total_visits), options)
    except Exception:
        raise
    except BaseException as exc:  # pyo3 PanicException
        raise RuntimeError(f"engine panic: {exc!r}") from None


def evaluator_loads() -> int:
    """How many evaluator files this process has loaded (tests; picklable)."""
    return EVALUATOR_LOADS
