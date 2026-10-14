import struct
from dataclasses import dataclass
from enum import StrEnum

from .poke_engine import *


class Weather(StrEnum):
    NONE = "none"
    SUN = "sun"
    RAIN = "rain"
    SAND = "sand"
    HAIL = "hail"
    SNOW = "snow"
    HARSH_SUN = "harshsun"
    HEAVY_RAIN = "heavyrain"


class Terrain(StrEnum):
    NONE = "none"
    GRASSY = "grassyterrain"
    ELECTRIC = "electricterrain"
    MISTY = "mistyterrain"
    PSYCHIC = "psychicterrain"


class PokemonIndex(StrEnum):
    P0 = "0"
    P1 = "1"
    P2 = "2"
    P3 = "3"
    P4 = "4"
    P5 = "5"


@dataclass
class IterativeDeepeningResult:
    """
    Result of an Iterative Deepening Expectiminimax Search

    :param side_one: The moves for side_one
    :type side_one: list[str]
    :param side_two: The moves for side_two
    :type side_two: list[str]
    :param matrix: A vector representing the payoff matrix of the search.
        Pruned branches are represented by None
    :type matrix: int
    :param depth_searched: The depth that was searched to
    :type depth_searched: int
    """

    side_one: list[str]
    side_two: list[str]
    matrix: list[float]
    depth_searched: int

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=rust_result.s1,
            side_two=rust_result.s2,
            matrix=rust_result.matrix,
            depth_searched=rust_result.depth_searched,
        )

    def get_safest_move(self) -> str:
        """
        Get the safest move for side_one
        The safest move is the move that minimizes the loss for the turn

        :return: The safest move
        :rtype: str
        """
        safest_value = float("-inf")
        safest_s1_index = 0
        vec_index = 0
        for i in range(len(self.side_one)):
            worst_case_this_row = float("inf")
            for _ in range(len(self.side_two)):
                score = self.matrix[vec_index]
                if score < worst_case_this_row:
                    worst_case_this_row = score

            if worst_case_this_row > safest_value:
                safest_s1_index = i
                safest_value = worst_case_this_row

        return self.side_one[safest_s1_index]


@dataclass
class MctsSideResult:
    """
    Result of a Monte Carlo Tree Search for a single side

    :param move_choice: The move that was chosen
    :type move_choice: str
    :param total_score: The total score of the chosen move
    :type total_score: float
    :param visits: The number of times the move was chosen
    :type visits: int
    :param avg_strategy: share of the root's average strategy (the Nash-converging
        quantity under selection="exp3"/"rm"; equals the visit share under "ucb1")
    :type avg_strategy: float
    """

    move_choice: str
    total_score: float
    visits: int
    avg_strategy: float = 0.0


@dataclass
class MctsResult:
    """
    Result of a Monte Carlo Tree Search

    :param side_one: Result for side one
    :type side_one: list[MctsSideResult]
    :param side_two: Result for side two
    :type side_two: list[MctsSideResult]
    :param total_visits: Total number of monte carlo iterations
    :type total_visits: int
    :param joint: root joint-action matrix, only with ``return_joint=True``: a list of
        ``(i, j, visits, mean)`` for every expanded cell, i indexing ``side_one`` and j
        ``side_two``; ``mean`` is side one's mean score in [0, 1]
    :type joint: list[tuple[int, int, int, float]] | None
    :param nodes: sampled interior search nodes, only with ``export_nodes > 0``: a list of
        ``(state_string, visits, value, static_value, depth)``. ``value`` is the search's
        absolute P(side one wins) for the node (mean backed-up leaf score of its visits, converted
        from the root-relative score); ``static_value`` is the evaluator's own absolute P at the
        node's position; ``depth`` counts joint turns below the root
    :type nodes: list[tuple[str, int, float, float, int]] | None
    :param root_value: the evaluator's absolute P(side one wins) at the root (NaN if threads > 1)
    :type root_value: float
    :param root_search_value: the search's absolute P(side one wins) at the root, comparable with
        the ``value`` of the exported nodes (NaN if threads > 1)
    :type root_search_value: float
    """

    side_one: list[MctsSideResult]
    side_two: list[MctsSideResult]
    total_visits: int
    joint: "list[tuple[int, int, int, float]] | None" = None
    depth_max: int = 0
    depth_mean: float = 0.0
    nodes: "list[tuple[str, int, float, float, int]] | None" = None
    root_value: float = float("nan")
    root_search_value: float = float("nan")

    @classmethod
    def _from_rust(cls, rust_result):
        return cls(
            side_one=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                    avg_strategy=i.avg_strategy,
                )
                for i in rust_result.s1
            ],
            side_two=[
                MctsSideResult(
                    move_choice=i.move_choice,
                    total_score=i.total_score,
                    visits=i.visits,
                    avg_strategy=i.avg_strategy,
                )
                for i in rust_result.s2
            ],
            total_visits=rust_result.iteration_count,
            joint=(None if rust_result.joint is None else list(rust_result.joint)),
            depth_max=rust_result.depth_max,
            depth_mean=rust_result.depth_mean,
            nodes=(None if rust_result.nodes is None else list(rust_result.nodes)),
            root_value=rust_result.root_value,
            root_search_value=rust_result.root_search_value,
        )


def monte_carlo_tree_search(
    state: State,
    duration_ms: int = 1000,
    iterations: int = 0,
    threads: int = 1,
    evaluator: "Evaluator | None" = None,
    selection: str = "ucb1",
    gamma: float = 0.1,
    return_joint: bool = False,
    prior: "PolicyPrior | None" = None,
    prior_opp: "PolicyPrior | None" = None,
    c_puct: float = 1.0,
    prior_eps: float = 0.03,
    fpu_reduction: float = 0.0,
    root_prior_s1: "dict[str, float] | None" = None,
    root_prior_s2: "dict[str, float] | None" = None,
    export_nodes: int = 0,
    export_min_visits: int = 16,
) -> MctsResult:
    """
    Perform monte-carlo-tree-search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search. ignored if iterations > 0
    :type duration_ms: int
    :param iterations: exact number of monte-carlo iterations to run
    :type iterations: int
    :param threads: number of threads to use for the search
    :type threads: int
    :param evaluator: None (stock hand-written leaf evaluation, unchanged) or a
        loaded ``Evaluator`` (see ``Evaluator.load``) used to score MCTS leaves
    :type evaluator: Evaluator | None
    :param selection: "ucb1" (default, decoupled UCB1), "exp3" or "rm" (regret matching
        with exploration). exp3/rm are single-threaded only (threads > 1 raises ValueError)
    :type selection: str
    :param gamma: exploration rate in (0, 1] for exp3/rm; ignored by ucb1
    :type gamma: float
    :param return_joint: also return the root joint-action matrix (``MctsResult.joint``);
        single-threaded only (threads > 1 raises ValueError)
    :type return_joint: bool
    :param prior: a ``PolicyPrior`` for side one: switches side one's selection from UCB1 to PUCT
        (``Q + c_puct * P * sqrt(N) / (1 + n)``). None keeps UCB1 for that side
    :param prior_opp: same for side two (evaluated from side two's perspective). Pass the same
        object as ``prior`` to mirror it
    :param c_puct: PUCT exploration constant; ``prior_eps`` mixes the prior with uniform;
        ``fpu_reduction`` is subtracted from the parent mean to value unvisited options
    :param root_prior_s1: / ``root_prior_s2``: ``{choice: probability}`` that replaces the net's
        prior at the root only (e.g. a human action model for the opponent)
    :return: the result of the search (``depth_max`` / ``depth_mean`` are tree depth statistics)
    :param export_nodes: also return up to this many sampled interior nodes of the search tree
        (``MctsResult.nodes``; sampled without replacement, probability proportional to visits,
        among nodes with at least ``export_min_visits`` visits); 0 = off. Single-threaded only
    :type export_nodes: int
    :param export_min_visits: minimum visits of an exported node
    :type export_min_visits: int
    :rtype: MctsResult
    """
    return MctsResult._from_rust(
        mcts(
            state, duration_ms, iterations, threads, evaluator, selection, gamma, return_joint,
            prior, prior_opp, c_puct, prior_eps, fpu_reduction, root_prior_s1, root_prior_s2,
            export_nodes, export_min_visits,
        )
    )


def iterative_deepening_expectiminimax(
    state: State, duration_ms: int = 1000
) -> IterativeDeepeningResult:
    """
    Perform an iterative-deepening expectiminimax search on the given state and for the given duration

    :param state: the state to search through
    :type state: State
    :param duration_ms: time in milliseconds to run the search
    :type duration_ms: int
    :return: the result of the search
    :rtype: IterativeDeepeningResult
    """
    return IterativeDeepeningResult._from_rust(id(state, duration_ms))


# --- learned-evaluator weight files (sdb fork) -------------------------------
# Format (little endian): "PEVN", u32 format=1, u32 feature_version, u32 kind
# (1 linear, 2 mlp), u32 n_in, u32 n_hidden, then f32 data. See engine/README.md.
# Residual evaluators use format=2: u32 flags (bit0 = residual) and f32 stock_coef
# follow n_hidden. Plain evaluators are still written as format 1.

_MAGIC = b"PEVN"
_FORMAT_VERSION = 1
_FORMAT_VERSION_RESIDUAL = 2


def _flat(x) -> list:
    """Flatten nested lists / numpy arrays / torch tensors into a list of floats."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if hasattr(x, "ravel"):
        return [float(v) for v in x.ravel().tolist()]
    out = []
    for item in x:
        if isinstance(item, (list, tuple)) or hasattr(item, "ravel"):
            out.extend(_flat(item))
        else:
            out.append(float(item))
    return out


def _pack(
    kind: int, n_hidden: int, *arrays, stock_coef: float = 0.0, feature_version: int | None = None
) -> bytes:
    version = FEATURE_VERSION if feature_version is None else feature_version
    n_in = feature_length(version)
    if stock_coef != 0.0:
        header = _MAGIC + struct.pack(
            "<5IIf", _FORMAT_VERSION_RESIDUAL, version, kind, n_in,
            n_hidden, 1, float(stock_coef),
        )
    else:
        header = _MAGIC + struct.pack("<5I", _FORMAT_VERSION, version, kind, n_in, n_hidden)
    body = b"".join(struct.pack(f"<{len(a)}f", *a) for a in arrays)
    return header + body


def write_linear(
    path, weights, bias: float = 0.0, stock_coef: float = 0.0, feature_version: int | None = None
) -> None:
    """Write a linear evaluator: value = weights . state_features + bias.

    ``stock_coef != 0`` makes it residual (format 2): value additionally gets
    ``stock_coef * evaluate(state)``. 0 writes the plain format 1.
    ``feature_version`` (default ``FEATURE_VERSION``) selects the feature layout (1 or 2).
    """
    w = _flat(weights)
    n_in = feature_length(FEATURE_VERSION if feature_version is None else feature_version)
    if len(w) != n_in:
        raise ValueError(f"expected {n_in} weights, got {len(w)}")
    with open(path, "wb") as f:
        f.write(_pack(1, 0, w, [float(bias)], stock_coef=stock_coef, feature_version=feature_version))


def write_mlp(
    path, w1, b1, w2, b2: float = 0.0, stock_coef: float = 0.0, feature_version: int | None = None
) -> None:
    """Write a 2-layer MLP evaluator: value = w2 . relu(W1 x + b1) + b2.

    ``w1`` has shape [hidden][n_in] (the layout of torch's ``nn.Linear(n_in, hidden).weight``;
    ``n_in = feature_length(feature_version)``), ``b1`` and ``w2`` have
    length hidden. ``stock_coef != 0`` makes it residual (see ``write_linear``).
    """
    w1f, b1f, w2f = _flat(w1), _flat(b1), _flat(w2)
    hidden = len(b1f)
    n_in = feature_length(FEATURE_VERSION if feature_version is None else feature_version)
    if hidden == 0 or len(w1f) != hidden * n_in or len(w2f) != hidden:
        raise ValueError(
            f"inconsistent shapes: w1={len(w1f)} (want {hidden}*{n_in}), "
            f"b1={hidden}, w2={len(w2f)}"
        )
    with open(path, "wb") as f:
        f.write(
            _pack(2, hidden, w1f, b1f, w2f, [float(b2)], stock_coef=stock_coef,
                  feature_version=feature_version)
        )
