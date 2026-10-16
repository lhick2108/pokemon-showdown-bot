"""Zero-sum matrix-game solver and root joint-matrix helpers for Nash move selection.

Pure numpy, no other dependencies. The game: the row player (us) picks row ``i``,
the column player (the opponent) picks column ``j``, and the row player receives
``A[i, j]`` (a win probability in [0, 1], but any real matrix works).

``solve_matrix_game`` is an exact simplex solve of the standard LP (Bland's rule, so
it terminates on degenerate games); ``solve_regret_matching`` is an iterative
fallback (regret matching+, alternating updates, linear averaging) used only if the
simplex ever fails numerically. Both return a ``NashResult`` whose ``exploitability``
is measured on the returned strategies, so callers can check it.

``shrunk_matrix`` turns the engine's root statistics (visit count and mean score per
expanded joint cell) into a payoff matrix, see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

DEFAULT_PSEUDOCOUNT = 5.0
_EPS = 1e-12


@dataclass(frozen=True)
class NashResult:
    row: np.ndarray  # row player's (our) mixed strategy
    col: np.ndarray  # column player's mixed strategy
    value: float  # game value for the row player
    exploitability: float  # (best response to col) - (worst case of row), >= 0


def exploitability(A: np.ndarray, row: np.ndarray, col: np.ndarray) -> float:
    """How far the pair is from equilibrium: max_i (A col)_i - min_j (row A)_j."""
    return float(max((A @ col).max() - (row @ A).min(), 0.0))


def _result(A: np.ndarray, row: np.ndarray, col: np.ndarray) -> NashResult:
    row = np.maximum(row, 0.0)
    col = np.maximum(col, 0.0)
    row = row / row.sum()
    col = col / col.sum()
    value = float(row @ A @ col)
    return NashResult(row, col, value, exploitability(A, row, col))


def solve_simplex(A: np.ndarray) -> NashResult | None:
    """Exact solve by the simplex method; ``None`` if it fails to converge.

    Shift ``A`` to ``B = A - min(A) + 1 > 0``. Then ``max sum(u)`` subject to
    ``B u <= 1, u >= 0`` has optimum ``1 / v`` (v = value of B); the column strategy
    is ``u * v`` and the row strategy is the dual solution scaled to sum 1.
    """
    m, n = A.shape
    B = A - A.min() + 1.0
    # tableau rows: m constraints [B | I | 1] and the objective row [-1 | 0 | 0]
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = B
    T[:m, n : n + m] = np.eye(m)
    T[:m, -1] = 1.0
    T[m, :n] = -1.0
    basis = list(range(n, n + m))
    for _ in range(50 * (m + n) + 100):
        negative = np.flatnonzero(T[m, :-1] < -1e-10)
        if negative.size == 0:
            break
        col = int(negative[0])  # Bland: smallest index
        column = T[:m, col]
        positive = column > 1e-10
        if not positive.any():
            return None  # unbounded: cannot happen for B > 0
        ratios = np.full(m, np.inf)
        ratios[positive] = T[:m, -1][positive] / column[positive]
        best = ratios.min()
        ties = np.flatnonzero(ratios <= best + 1e-12)
        r = int(min(ties, key=lambda k: basis[k]))
        T[r] /= T[r, col]
        for k in range(m + 1):
            if k != r and T[k, col] != 0.0:
                T[k] -= T[k, col] * T[r]
        basis[r] = col
    else:
        return None
    u = np.zeros(n + m)
    for k, b in enumerate(basis):
        u[b] = T[k, -1]
    col_strategy = u[:n]
    row_strategy = T[m, n : n + m]
    if col_strategy.sum() <= _EPS or row_strategy.sum() <= _EPS:
        return None
    return _result(A, row_strategy.copy(), col_strategy.copy())


def solve_regret_matching(A: np.ndarray, iterations: int = 2000) -> NashResult:
    """Regret matching+ with alternating updates and linear averaging (fallback)."""
    m, n = A.shape
    rr = np.zeros(m)
    rc = np.zeros(n)
    x = np.full(m, 1.0 / m)
    y = np.full(n, 1.0 / n)
    sx = np.zeros(m)
    sy = np.zeros(n)
    for t in range(1, iterations + 1):
        # row player moves against the current y
        u = A @ y
        rr = np.maximum(rr + u - x @ u, 0.0)
        total = rr.sum()
        x = rr / total if total > 0 else np.full(m, 1.0 / m)
        # column player (minimiser) responds to the new x
        w = x @ A
        rc = np.maximum(rc - w + y @ w, 0.0)
        total = rc.sum()
        y = rc / total if total > 0 else np.full(n, 1.0 / n)
        sx += t * x
        sy += t * y
    return _result(A, sx, sy)


def solve_matrix_game(A: np.ndarray | Sequence[Sequence[float]]) -> NashResult:
    """Nash equilibrium of the zero-sum game with row payoff ``A`` (shape m x n)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.size == 0:
        raise ValueError(f"payoff matrix must be non-empty 2-D, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("payoff matrix has non-finite entries")
    if A.shape[0] == 1:
        col = np.zeros(A.shape[1])
        col[int(np.argmin(A[0]))] = 1.0
        return _result(A, np.ones(1), col)
    if A.shape[1] == 1:
        row = np.zeros(A.shape[0])
        row[int(np.argmax(A[:, 0]))] = 1.0
        return _result(A, row, np.ones(1))
    result = solve_simplex(A)
    if result is None or result.exploitability > 1e-6:
        fallback = solve_regret_matching(A)
        if result is None or fallback.exploitability < result.exploitability:
            result = fallback
    return result


def shrunk_matrix(
    k1: int,
    k2: int,
    cells: Iterable[tuple[int, int, int, float]],
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    default: float = 0.5,
) -> np.ndarray:
    """Payoff matrix (k1 x k2) from root joint statistics, shrunk toward a row prior.

    ``cells`` are ``(i, j, visits, mean)`` for the expanded joint actions (missing
    cells have zero visits). Each entry is

        A_ij = (visits_ij * mean_ij + c * prior_i) / (visits_ij + c)

    with ``c`` the pseudo-count and ``prior_i`` row i's visit-weighted mean (the
    overall root mean when the row has no visits, ``default`` when nothing was
    visited). The search deliberately leaves bad joint actions under-visited, so a
    cell's raw mean can rest on one or two rollouts; an unshrunk matrix lets that
    noise (or a zero-visit placeholder) create fake equilibria. Shrinking makes a
    cell with few visits say "about as good as the rest of its row".
    """
    if pseudocount < 0:
        raise ValueError("pseudocount must be >= 0")
    visits = np.zeros((k1, k2))
    totals = np.zeros((k1, k2))
    for i, j, v, mean in cells:
        if v > 0:
            visits[i, j] += v
            totals[i, j] += v * mean
    row_visits = visits.sum(axis=1)
    all_visits = row_visits.sum()
    root_mean = totals.sum() / all_visits if all_visits > 0 else default
    prior = np.where(
        row_visits > 0, totals.sum(axis=1) / np.maximum(row_visits, 1.0), root_mean
    )[:, None]
    if pseudocount == 0:
        return np.where(visits > 0, totals / np.maximum(visits, 1.0), prior)
    return (totals + pseudocount * prior) / (visits + pseudocount)


def aggregate_strategies(
    strategies: Sequence[tuple[float, dict[str, float]]], floor: float = 0.0
) -> dict[str, float]:
    """Weighted average of per-world strategies keyed by choice string.

    ``strategies`` is ``[(weight, {choice: probability})]``. Choices a world lacks
    get probability 0 there. ``floor`` drops choices below that share (never the best
    one) and renormalises.
    """
    mass: dict[str, float] = {}
    total = 0.0
    for weight, strategy in strategies:
        if weight <= 0:
            continue
        total += weight
        for choice, p in strategy.items():
            mass[choice] = mass.get(choice, 0.0) + weight * p
    if total <= 0 or not mass:
        return {}
    policy = {c: m / total for c, m in mass.items()}
    if floor > 0:
        best = max(policy.values())
        policy = {c: p for c, p in policy.items() if p >= floor or p == best}
        norm = sum(policy.values())
        policy = {c: p / norm for c, p in policy.items()}
    return policy


# ---------------------------------------------------------------------------
# Bayesian (cross-world) root solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesResult:
    x: np.ndarray  # our single mixed strategy over the shared rows
    value: float  # sum_w p_w min_j (x^T M_w)_j : guaranteed expected win prob
    world_values: tuple  # per-world min_j (x^T M_w)_j
    ys: tuple  # per-world opponent best-response (mixed) strategies from the LP dual / BR


def bayes_world_values(x: np.ndarray, matrices: Sequence[np.ndarray]) -> list[float]:
    """Each world's value ``min_j (x^T M_w)_j`` of our strategy ``x`` (opponent best-responds)."""
    return [float((np.asarray(x) @ M).min()) for M in matrices]


def bayes_value(
    x: np.ndarray, matrices: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> float:
    """Bayesian objective ``sum_w p_w min_j (x^T M_w)_j`` (weights normalised)."""
    vals = np.array(bayes_world_values(x, matrices))
    p = _norm_weights(weights, len(matrices))
    return float(p @ vals)


def _norm_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    p = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if p.shape != (n,) or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("weights must be non-negative, one per world, with a positive sum")
    return p / p.sum()


def _check_worlds(matrices: Sequence[np.ndarray]) -> list[np.ndarray]:
    mats = [np.asarray(M, dtype=float) for M in matrices]
    if not mats:
        raise ValueError("need at least one world")
    m = mats[0].shape[0]
    for M in mats:
        if M.ndim != 2 or M.size == 0 or M.shape[0] != m:
            raise ValueError("every world needs a non-empty matrix with the same row count")
        if not np.all(np.isfinite(M)):
            raise ValueError("payoff matrix has non-finite entries")
    return mats


def solve_bayes_lp(
    matrices: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> BayesResult | None:
    """Exact solve of ``max_x sum_w p_w min_j (x^T M_w)_j`` by the simplex method.

    We commit to one strategy ``x`` over the shared rows; each world's opponent knows
    its own set and best-responds separately. LP over ``z = [x, v_1..v_W]``: maximise
    ``sum p_w v_w`` s.t. ``v_w - (x^T B_w)_j <= 0`` for every world column and
    ``sum x <= 1`` (``B_w = M_w - min + 1 > 0``, so the sum constraint is tight and all
    right-hand sides are >= 0: the slack basis is feasible). Bland's rule. ``None`` if the
    simplex fails to converge.
    """
    mats = _check_worlds(matrices)
    p = _norm_weights(weights, len(mats))
    m, W = mats[0].shape[0], len(mats)
    lo = min(M.min() for M in mats)
    Bs = [M - lo + 1.0 for M in mats]
    rows = sum(M.shape[1] for M in Bs) + 1
    nv = m + W
    T = np.zeros((rows + 1, nv + rows + 1))
    r = 0
    owner = []
    for w, B in enumerate(Bs):
        for j in range(B.shape[1]):
            T[r, :m] = -B[:, j]
            T[r, m + w] = 1.0
            owner.append((w, j))
            r += 1
    T[r, :m] = 1.0
    T[:rows, nv : nv + rows] = np.eye(rows)
    T[:rows, -1] = 0.0
    T[r, -1] = 1.0
    T[rows, m:nv] = -p  # objective row
    basis = list(range(nv, nv + rows))
    for _ in range(200 * (rows + nv) + 1000):
        negative = np.flatnonzero(T[rows, :-1] < -1e-10)
        if negative.size == 0:
            break
        col = int(negative[0])
        column = T[:rows, col]
        positive = column > 1e-10
        if not positive.any():
            return None
        ratios = np.full(rows, np.inf)
        ratios[positive] = T[:rows, -1][positive] / column[positive]
        best = ratios.min()
        ties = np.flatnonzero(ratios <= best + 1e-12)
        pr = int(min(ties, key=lambda k: basis[k]))
        T[pr] /= T[pr, col]
        for k in range(rows + 1):
            if k != pr and T[k, col] != 0.0:
                T[k] -= T[k, col] * T[pr]
        basis[pr] = col
    else:
        return None
    z = np.zeros(nv + rows)
    for k, b in enumerate(basis):
        z[b] = T[k, -1]
    x = np.maximum(z[:m], 0.0)
    if x.sum() <= _EPS:
        return None
    x = x / x.sum()
    duals = np.maximum(T[rows, nv : nv + rows - 1], 0.0)  # one per world column row
    ys, off = [], 0
    for B in Bs:
        d = duals[off : off + B.shape[1]]
        off += B.shape[1]
        ys.append(d / d.sum() if d.sum() > _EPS else np.full(B.shape[1], 1.0 / B.shape[1]))
    wv = bayes_world_values(x, mats)
    return BayesResult(x, float(p @ np.array(wv)), tuple(wv), tuple(ys))


def solve_bayes_rm(
    matrices: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    iterations: int = 20000,
) -> BayesResult:
    """Regret matching+ on ``x`` against per-world best responses (linear averaging)."""
    mats = _check_worlds(matrices)
    p = _norm_weights(weights, len(mats))
    m = mats[0].shape[0]
    reg = np.zeros(m)
    x = np.full(m, 1.0 / m)
    sx = np.zeros(m)
    for t in range(1, iterations + 1):
        u = np.zeros(m)
        for pw, M in zip(p, mats):
            u += pw * M[:, int(np.argmin(x @ M))]
        reg = np.maximum(reg + u - x @ u, 0.0)
        s = reg.sum()
        x = reg / s if s > 0 else np.full(m, 1.0 / m)
        sx += t * x
    x = sx / sx.sum()
    wv = bayes_world_values(x, mats)
    ys = tuple(np.eye(M.shape[1])[int(np.argmin(x @ M))] for M in mats)
    return BayesResult(x, float(p @ np.array(wv)), tuple(wv), ys)


def solve_bayes(
    matrices: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> BayesResult:
    """Bayesian cross-world solve: exact LP, regret matching if the simplex fails."""
    mats = _check_worlds(matrices)
    result = solve_bayes_lp(mats, weights)
    if result is None:
        result = solve_bayes_rm(mats, weights)
    return result
