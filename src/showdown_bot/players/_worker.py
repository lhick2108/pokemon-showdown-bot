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


def get_evaluator(path: str, scale: float = 1.0, material: float = 0.0):
    """The ``poke_engine.Evaluator`` for ``(path, scale, material)``, loaded once per process."""
    global EVALUATOR_LOADS
    key = (path, float(scale)) if not material else (path, float(scale), float(material))
    evaluator = _EVALUATORS.get(key)
    if evaluator is None:
        from poke_engine import Evaluator

        evaluator = (Evaluator.load(path, leaf_scale=float(scale), material=float(material))
                     if material else Evaluator.load(path, leaf_scale=float(scale)))
        EVALUATOR_LOADS += 1
        _EVALUATORS[key] = evaluator
    return evaluator


# Policy priors (``.ppol``) and human models loaded in this worker process, by path.
_PRIORS: dict[str, object] = {}
_HUMAN: dict[str, object] = {}


def get_prior(path: str):
    """The ``poke_engine.PolicyPrior`` for ``path``, loaded once per process."""
    prior = _PRIORS.get(path)
    if prior is None:
        from poke_engine import PolicyPrior

        prior = _PRIORS[path] = PolicyPrior.load(path)
    return prior


def _human_root_prior(state, state_string: str, path: str, elo) -> dict[str, float]:
    """The human action model's distribution over side two's root options (root-only prior)."""
    import poke_engine

    model = _HUMAN.get(path)
    if model is None:
        from showdown_bot.human_policy import HumanPolicy

        model = _HUMAN[path] = HumanPolicy.load(path)
    choices = [c for c, _slot in poke_engine.policy_actions(state, 2)]
    return model.predict(state_string, choices, elo, side=2)


def prior_kwargs(
    state,
    state_string: str,
    prior_path: str | None,
    prior_opp: str,
    c_puct: float,
    prior_eps: float,
    prior_fpu: float,
    human_path: str | None,
    human_elo,
) -> dict:
    """``monte_carlo_tree_search`` keywords for the PUCT prior settings.

    ``prior_opp``: ``self`` (our net mirrored for the opponent), ``none`` (the opponent keeps
    UCB1), ``human`` (mirrored net below the root, the human model at the root) or a ``.ppol``
    path (a different net for the opponent).
    """
    out: dict = {"c_puct": float(c_puct), "prior_eps": float(prior_eps), "fpu_reduction": float(prior_fpu)}
    net = get_prior(prior_path) if prior_path else None
    out["prior"] = net
    if prior_opp in ("self", "human"):
        out["prior_opp"] = net
    elif prior_opp != "none":
        out["prior_opp"] = get_prior(prior_opp)
    if prior_opp == "human":
        if not human_path:
            raise RuntimeError("prior_opp=human needs a human-policy model path")
        out["root_prior_s2"] = _human_root_prior(state, state_string, human_path, human_elo)
    return out


class OppTuple(tuple):
    """A search result tuple that also carries ``opp``: the opponent's ``(choice, visits)``
    root options (soft belief, bead showdown-bot-1wu).  Equal to the plain tuple; the
    attribute survives pickling to the parent process."""

    def __new__(cls, items=(), opp=None, nodes=None, search_value=None):
        obj = super().__new__(cls, items)
        obj.opp = opp
        obj.nodes = nodes  # sampled interior nodes (bead rnm), see search_world(export_nodes=)
        obj.search_value = search_value  # the search's absolute root value, set with ``nodes``
        return obj

    def __getnewargs__(self):
        return (tuple(self), self.opp, self.nodes, self.search_value)


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
    opp_visits: bool = False,
    prior_path: str | None = None,
    prior_opp: str = "self",
    c_puct: float = 1.0,
    prior_eps: float = 0.03,
    prior_fpu: float = 0.0,
    human_path: str | None = None,
    human_elo=None,
    export_nodes: int = 0,
    export_min_visits: int = 16,
    material: float = 0.0,
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
    With ``opp_visits`` the result is an ``OppTuple`` whose ``.opp`` lists the opponent's
    (side_two) root ``(choice, visits)`` options.
    With ``prior_path`` (a ``.ppol`` policy net) side one's selection becomes PUCT; see
    ``prior_kwargs`` for ``prior_opp``. Single-threaded only; the result shape is unchanged.
    With ``export_nodes`` > 0 (single-threaded only) the result is an ``OppTuple`` whose
    ``.nodes`` lists up to that many sampled interior search nodes as
    ``(state, visits, value, static_value, depth)`` (values are absolute P(side one wins); see
    ``poke_engine.MctsResult.nodes``) and whose ``.search_value`` is the root's absolute search
    value; an engine without the export raises.
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
            kwargs["evaluator"] = get_evaluator(evaluator_path, evaluator_scale, material)
        if selection != "ucb1":
            kwargs["selection"] = selection
            kwargs["gamma"] = float(selection_gamma)
        if joint:
            kwargs["return_joint"] = True
        if prior_path:
            if threads > 1:
                raise RuntimeError("policy priors need threads == 1")
            kwargs.update(prior_kwargs(
                state, state_string, prior_path, prior_opp, c_puct, prior_eps, prior_fpu,
                human_path, human_elo,
            ))
        if export_nodes:
            kwargs["export_nodes"] = int(export_nodes)
            kwargs["export_min_visits"] = int(export_min_visits)
        result = monte_carlo_tree_search(state, search_time_ms, threads=threads, **kwargs)
        out = _format(result, selection, joint, opp_names)
        opp = [(str(o.move_choice), int(o.visits)) for o in result.side_two] if opp_visits else None
        if export_nodes:
            nodes = [(str(s), int(v), float(val), float(st), int(d)) for s, v, val, st, d in result.nodes or ()]
            return OppTuple(out, opp, nodes, float(result.root_search_value))
        if opp_visits:
            return OppTuple(out, opp)
        return out
    except Exception:
        raise
    except BaseException as exc:  # pyo3 PanicException
        raise RuntimeError(f"engine panic: {exc!r}") from None


def _format(result, selection: str, joint: bool, opp_names: bool) -> tuple:
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


def evaluator_loads() -> int:
    """How many evaluator files this process has loaded (tests; picklable)."""
    return EVALUATOR_LOADS


def endgame_world(state_string: str, time_ms: int) -> tuple:
    """Iterative-deepening expectiminimax on one serialised world (endgame mode).

    Returns ``(our options, their options, flat row-major matrix, depth searched)``; pruned
    cells are NaN (see ``showdown_bot.endgame``). The search uses the engine's stock
    evaluation only (no learned evaluator hook); a Rust panic becomes ``RuntimeError``.
    """
    from poke_engine import State, iterative_deepening_expectiminimax

    try:
        state = State.from_string(state_string)
        result = iterative_deepening_expectiminimax(state, int(time_ms))
        return (
            [str(c) for c in result.side_one],
            [str(c) for c in result.side_two],
            [float(x) for x in result.matrix],
            int(result.depth_searched),
        )
    except Exception:
        raise
    except BaseException as exc:  # pyo3 PanicException
        raise RuntimeError(f"engine panic: {exc!r}") from None
