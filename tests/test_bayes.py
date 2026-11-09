"""mode=bayes: the cross-world Bayesian root solve (nash.solve_bayes*, search.aggregate_bayes)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from showdown_bot import nash
from showdown_bot.players.search import (
    BAYES, MODES, SearchConfig, aggregate_bayes, aggregate_nash, aggregate_policy,
    needs_joint, select_choice,
)


def _world(rows, visits, mat, v=10000):
    """A WorldResult: options ``rows`` with root ``visits``, full joint grid from ``mat``."""
    mat = np.asarray(mat, dtype=float)
    cells = [(i, j, v, float(mat[i, j])) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
    options = [(c, n, n * float(mat[i].mean())) for i, (c, n) in enumerate(zip(rows, visits))]
    return (sum(visits), options, None, (mat.shape[1], cells))


def test_one_world_reduces_to_the_plain_matrix_game():
    rng = np.random.default_rng(3)
    for _ in range(10):
        M = rng.random((5, 4))
        ref = nash.solve_matrix_game(M)
        got = nash.solve_bayes_lp([M])
        assert got is not None
        assert got.value == pytest.approx(ref.value, abs=1e-8)
        assert nash.bayes_value(got.x, [M]) == pytest.approx(ref.value, abs=1e-8)


def test_matching_pennies_one_world():
    got = nash.solve_bayes([np.array([[1.0, 0.0], [0.0, 1.0]])])
    assert got.value == pytest.approx(0.5)
    assert got.x == pytest.approx([0.5, 0.5])


def test_lp_value_dominates_any_other_strategy():
    rng = np.random.default_rng(7)
    for _ in range(10):
        mats = [rng.random((6, rng.integers(2, 6))) for _ in range(4)]
        got = nash.solve_bayes_lp(mats)
        assert got is not None
        for _ in range(200):
            x = rng.dirichlet(np.ones(6))
            assert nash.bayes_value(x, mats) <= got.value + 1e-9
        assert got.x.sum() == pytest.approx(1.0)


def test_lp_and_regret_matching_agree():
    rng = np.random.default_rng(11)
    for _ in range(5):
        mats = [rng.random((5, int(rng.integers(2, 6)))) for _ in range(3)]
        lp = nash.solve_bayes_lp(mats)
        rm = nash.solve_bayes_rm(mats, iterations=30000)
        assert lp is not None
        assert rm.value == pytest.approx(lp.value, abs=5e-3)
        assert rm.value <= lp.value + 1e-9


def test_worlds_may_have_different_column_counts():
    A = np.array([[0.2, 0.9, 0.5], [0.8, 0.1, 0.5]])
    B = np.array([[0.6], [0.4]])
    got = nash.solve_bayes([A, B])
    assert got.value == pytest.approx(nash.bayes_value(got.x, [A, B]))
    assert len(got.ys) == 2 and len(got.ys[0]) == 3 and len(got.ys[1]) == 1


def test_bayes_picks_the_hedge_that_per_world_nash_averaging_misses():
    # Two worlds, one opponent reply each. World 1 rewards a, world 2 rewards c, b is
    # the safe middle. Solving each world alone (we "know" the set) plays a or c; the
    # move that is best when we cannot tell the worlds apart is b.
    rows = ["a", "b", "c"]
    m1 = [[0.9], [0.5], [0.0]]
    m2 = [[0.0], [0.5], [0.9]]
    results = [_world(rows, [70000, 20000, 10000], m1), _world(rows, [30000, 20000, 50000], m2)]
    nash_policy, _ = aggregate_nash(results, 5.0, 0.0)
    assert max(nash_policy, key=nash_policy.get) in ("a", "c")
    assert nash_policy.get("b", 0.0) == pytest.approx(0.0, abs=1e-9)
    policy, info = aggregate_bayes(results, 5.0, 0.0)
    assert policy == pytest.approx({"b": 1.0})
    assert info["value"] == pytest.approx(0.5, abs=0.02)
    # argmax (visit shares: a 0.5, b 0.2, c 0.3) also takes a
    arg = aggregate_policy(results)
    assert max(arg, key=arg.get) == "a"


def test_opponent_best_responds_per_world_in_the_bayes_game():
    # world 1: opponent column 0 punishes a; world 2 has a single neutral column
    rows = ["a", "b"]
    m1 = [[0.0, 1.0], [0.4, 0.4]]
    m2 = [[0.6], [0.6]]
    results = [_world(rows, [10, 10], m1), _world(rows, [10, 10], m2)]
    policy, info = aggregate_bayes(results, 0.0, 0.0)
    assert info["world_values"][0] == pytest.approx(0.4, abs=1e-6)
    assert policy["b"] > 0.99


def test_rows_are_aligned_by_choice_name():
    r1 = _world(["a", "b"], [10, 10], [[0.9], [0.1]])
    r2 = _world(["b", "a"], [10, 10], [[0.1], [0.9]])
    policy, info = aggregate_bayes([r1, r2], 0.0, 0.0)
    assert policy == pytest.approx({"a": 1.0})
    assert not info["row_mismatch"]


def test_rows_missing_from_a_world_are_dropped_and_flagged():
    r1 = _world(["a", "b", "c"], [10, 10, 10], [[0.9], [0.5], [0.1]])
    r2 = _world(["a", "b"], [10, 10], [[0.8], [0.4]])
    policy, info = aggregate_bayes([r1, r2], 0.0, 0.0)
    assert set(policy) <= {"a", "b"} and info["row_mismatch"]


def test_floor_and_argmax_option():
    rows = ["a", "b"]
    results = [_world(rows, [10, 10], [[0.0, 1.0], [1.0, 0.0]])]  # matching pennies
    mixed, _ = aggregate_bayes(results, 0.0, 0.03)
    assert mixed == pytest.approx({"a": 0.5, "b": 0.5}, abs=1e-6)
    one_hot, _ = aggregate_bayes(results, 0.0, 0.03, argmax=True)
    assert list(one_hot.values()) == [1.0]


def test_no_matrix_gives_empty_policy_for_fallback():
    assert aggregate_bayes([(100, [("a", 100, 50.0)])], 5.0) == ({}, None)
    assert aggregate_bayes([], 5.0) == ({}, None)


def test_mode_is_registered_and_samples_like_nash():
    assert BAYES == "bayes" and BAYES in MODES
    assert needs_joint(SearchConfig(mode=BAYES))
    rng = random.Random(0)
    picks = {select_choice({"a": 0.5, "b": 0.5}, BAYES, rng) for _ in range(50)}
    assert picks == {"a", "b"}
    assert select_choice({"a": 1.0}, BAYES, rng) == "a"
