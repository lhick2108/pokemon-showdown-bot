"""Tests for showdown_bot.nash (zero-sum matrix-game solver, shrinkage, aggregation)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from showdown_bot.nash import (
    aggregate_strategies, shrunk_matrix, solve_matrix_game, solve_regret_matching,
)

RPS = np.array([[0.5, 0.0, 1.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])


def test_rps_is_uniform_with_value_half():
    r = solve_matrix_game(RPS)
    assert np.allclose(r.row, 1 / 3, atol=1e-9)
    assert np.allclose(r.col, 1 / 3, atol=1e-9)
    assert r.value == pytest.approx(0.5)
    assert r.exploitability < 1e-9


def test_matching_pennies_variants():
    r = solve_matrix_game([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(r.row, [0.5, 0.5]) and r.value == pytest.approx(0.5)
    # row 0 dominates (0.9 > 0.4, 0.3 > 0.2): pure strategy
    r = solve_matrix_game(np.array([[0.9, 0.3], [0.4, 0.2]]))
    assert np.allclose(r.row, [1.0, 0.0]) and r.value == pytest.approx(0.3)
    r = solve_matrix_game(np.array([[0.8, 0.2], [0.3, 0.7]]))
    p = (0.7 - 0.3) / (0.8 - 0.2 - 0.3 + 0.7)  # mixed 2x2 formula
    assert r.row[0] == pytest.approx(p)
    assert r.exploitability < 1e-9


def test_dominated_rows_get_zero_mass():
    A = np.vstack([RPS, [0.1, 0.1, 0.1], [0.0, 0.2, 0.0]])
    r = solve_matrix_game(A)
    assert r.row[3:].sum() < 1e-9
    assert r.value == pytest.approx(0.5)


def test_pure_saddle_point():
    A = np.array([[0.6, 0.7, 0.9], [0.4, 0.5, 0.3], [0.2, 0.6, 0.1]])
    r = solve_matrix_game(A)
    assert np.allclose(r.row, [1, 0, 0]) and np.allclose(r.col, [1, 0, 0])
    assert r.value == pytest.approx(0.6)


def test_single_row_and_column():
    r = solve_matrix_game([[0.7, 0.2, 0.9]])
    assert r.row.tolist() == [1.0] and r.value == pytest.approx(0.2)
    r = solve_matrix_game([[0.7], [0.9], [0.1]])
    assert r.row.tolist() == [0.0, 1.0, 0.0] and r.value == pytest.approx(0.9)


def test_constant_and_degenerate_matrices():
    for A in (np.full((4, 5), 0.5), np.zeros((3, 3)), np.ones((2, 6)),
              np.array([[0.5, 0.5], [0.5, 0.5]])):
        r = solve_matrix_game(A)
        assert r.exploitability < 1e-9
        assert r.row.sum() == pytest.approx(1.0) and r.col.sum() == pytest.approx(1.0)


def test_random_games_have_tiny_exploitability():
    rng = np.random.default_rng(0)
    for _ in range(300):
        m, n = rng.integers(1, 14, size=2)
        A = rng.random((m, n))
        if rng.random() < 0.3:  # quantised -> many ties / degeneracy
            A = np.round(A * 4) / 4
        r = solve_matrix_game(A)
        assert r.exploitability < 1e-6, (A, r)
        assert r.row.min() >= 0 and r.col.min() >= 0
        assert r.value == pytest.approx(r.row @ A @ r.col)


def test_regret_matching_fallback_converges():
    rng = np.random.default_rng(1)
    for A in (RPS, rng.random((6, 6)), rng.random((13, 13))):
        assert solve_regret_matching(A, iterations=4000).exploitability < 2e-2
    exact = solve_matrix_game(RPS)
    assert solve_regret_matching(RPS).value == pytest.approx(exact.value, abs=1e-2)


def test_non_finite_and_empty_rejected():
    with pytest.raises(ValueError):
        solve_matrix_game(np.array([[0.5, np.nan], [0.1, 0.2]]))
    with pytest.raises(ValueError):
        solve_matrix_game(np.zeros((0, 3)))


def test_solver_is_fast_at_13x13():
    rng = np.random.default_rng(2)
    games = [rng.random((13, 13)) for _ in range(50)]
    solve_matrix_game(games[0])
    start = time.perf_counter()
    for A in games:
        solve_matrix_game(A)
    per_solve_ms = (time.perf_counter() - start) / len(games) * 1000
    print(f"13x13 solve: {per_solve_ms:.2f} ms")
    assert per_solve_ms < 10  # generous bound for a loaded box; typical is far below


def test_shrunk_matrix_pulls_low_visit_cells_to_row_prior():
    cells = [(0, 0, 100, 0.8), (0, 1, 1, 0.0), (1, 0, 50, 0.4)]
    A = shrunk_matrix(2, 2, cells, pseudocount=5.0)
    prior0 = (100 * 0.8 + 0.0) / 101
    assert A[0, 0] == pytest.approx((80 + 5 * prior0) / 105)
    assert A[0, 1] == pytest.approx((0 + 5 * prior0) / 6)  # mostly the prior, not 0
    assert A[0, 1] > 0.6
    assert A[1, 1] == pytest.approx(0.4)  # never visited: exactly the row prior


def test_shrunk_matrix_empty_row_uses_root_mean_and_nothing_uses_default():
    A = shrunk_matrix(2, 2, [(0, 0, 10, 0.9)], pseudocount=5.0)
    assert A[1, 0] == pytest.approx(0.9) and A[1, 1] == pytest.approx(0.9)
    assert np.allclose(shrunk_matrix(2, 2, [], 5.0), 0.5)
    raw = shrunk_matrix(2, 2, [(0, 0, 10, 0.9)], pseudocount=0.0)
    assert raw[0, 0] == pytest.approx(0.9) and raw[0, 1] == pytest.approx(0.9)
    with pytest.raises(ValueError):
        shrunk_matrix(1, 1, [], -1.0)


def test_unvisited_cell_does_not_create_a_fake_equilibrium():
    # A noisy 1-visit 1.0 in row 1 must not make row 1 attractive.
    cells = [(0, 0, 200, 0.6), (0, 1, 200, 0.6), (1, 0, 200, 0.3), (1, 1, 1, 1.0)]
    r = solve_matrix_game(shrunk_matrix(2, 2, cells, 5.0))
    assert r.row[0] > 0.9


def test_aggregate_strategies_weights_and_floor():
    mix = [(3.0, {"a": 1.0}), (1.0, {"b": 0.5, "c": 0.5})]
    assert aggregate_strategies(mix) == pytest.approx({"a": 0.75, "b": 0.125, "c": 0.125})
    assert aggregate_strategies(mix, floor=0.2) == pytest.approx({"a": 1.0})
    assert aggregate_strategies([(0.0, {"a": 1.0})]) == {}
    even = aggregate_strategies([(1.0, {"a": 0.5, "b": 0.5})], floor=0.9)
    assert even == pytest.approx({"a": 0.5, "b": 0.5})
