"""Tests for our poke-engine fork (engine/): features, evaluators, MCTS hook.

Requires the fork to be installed in the project venv (scripts/build_engine.ps1).
"""

from __future__ import annotations

import math

import pytest

poke_engine = pytest.importorskip("poke_engine")

from poke_engine import (  # noqa: E402
    FEATURE_LENGTH,
    FEATURE_VERSION,
    Evaluator,
    Move,
    Pokemon,
    Side,
    State,
    monte_carlo_tree_search,
    state_feature_names,
    state_features,
    write_linear,
    write_mlp,
)

# Documented in engine/FEATURES.md.
DOCUMENTED_FEATURE_LENGTH = 738


def _pokemon(name, types, stats, moves, tera="normal"):
    hp, atk, dfn, spa, spd, spe = stats
    return Pokemon(
        id=name,
        level=100,
        types=types,
        base_types=types,
        hp=hp,
        maxhp=hp,
        ability="none",
        base_ability="none",
        item="none",
        attack=atk,
        defense=dfn,
        special_attack=spa,
        special_defense=spd,
        speed=spe,
        status="none",
        tera_type=tera,
        moves=[Move(id=m, pp=16) for m in moves],
    )


def make_state() -> State:
    """1v1: side one can hit (closecombat) or do nothing (splash); side two can only splash."""
    one = _pokemon(
        "greattusk",
        ("ground", "fighting"),
        (434, 361, 354, 144, 142, 300),
        ("closecombat", "splash"),
    )
    two = _pokemon(
        "kingambit",
        ("dark", "steel"),
        (1500, 405, 600, 140, 226, 136),
        ("splash",),
    )
    return State(side_one=Side(pokemon=[one]), side_two=Side(pokemon=[two]))


def _closecombat_share(result) -> float:
    """Share of side-one visits on Close Combat (with or without Terastallizing)."""
    total = sum(m.visits for m in result.side_one)
    hits = sum(
        m.visits for m in result.side_one if m.move_choice.startswith("closecombat")
    )
    return hits / total


def _opp_hp_weights(value: float) -> list[float]:
    names = state_feature_names()
    w = [0.0] * FEATURE_LENGTH
    w[names.index("opp.p0.hp")] = value
    return w


def test_import_and_constants():
    assert FEATURE_VERSION == 1
    assert FEATURE_LENGTH == DOCUMENTED_FEATURE_LENGTH
    assert len(state_feature_names()) == FEATURE_LENGTH


def test_state_features_length_and_values():
    state = make_state()
    feats = state_features(state)
    assert len(feats) == DOCUMENTED_FEATURE_LENGTH
    names = state_feature_names()
    assert feats[names.index("own.p0.hp")] == 1.0
    assert feats[names.index("own.p0.alive")] == 1.0
    assert feats[names.index("own.p0.type.ground")] == 1.0
    assert feats[names.index("opp.p0.type.dark")] == 1.0
    # slot 1 is an empty (fainted) placeholder
    assert feats[names.index("own.p1.alive")] == 0.0
    # side two's view swaps own and opp
    flipped = state_features(state, side=2)
    assert flipped[names.index("own.p0.type.dark")] == 1.0
    with pytest.raises(ValueError):
        state_features(state, side=3)


def test_mcts_stock_runs_and_returns_visits():
    result = monte_carlo_tree_search(make_state(), iterations=2000, evaluator=None)
    assert result.total_visits >= 2000
    assert sum(m.visits for m in result.side_one) > 0
    assert {m.move_choice for m in result.side_one} >= {"closecombat", "splash"}


def test_linear_evaluator_shifts_visits_toward_damage():
    state = make_state()
    # reward lower opponent HP (value falls as opp hp rises) vs the flipped weights
    good = Evaluator.linear(_opp_hp_weights(-3.0))
    bad = Evaluator.linear(_opp_hp_weights(+3.0))
    share_good = _closecombat_share(
        monte_carlo_tree_search(state, iterations=5000, evaluator=good)
    )
    share_bad = _closecombat_share(
        monte_carlo_tree_search(state, iterations=5000, evaluator=bad)
    )
    assert share_good > 0.7, share_good
    assert share_bad < 0.45, share_bad
    assert share_good - share_bad > 0.3


def test_evaluator_load_roundtrip_and_mlp(tmp_path):
    state = make_state()
    path = tmp_path / "lin.pev"
    write_linear(path, _opp_hp_weights(-3.0), bias=0.5)
    loaded = Evaluator.load(str(path))
    direct = Evaluator.linear(_opp_hp_weights(-3.0), bias=0.5)
    assert loaded.kind == "linear"
    assert loaded.evaluate(state) == pytest.approx(direct.evaluate(state))
    # opp hp fraction is 1.0 in the root state: -3 + 0.5
    assert loaded.evaluate(state) == pytest.approx(-2.5)

    # 2-hidden-unit MLP: h0 = relu(-opp.hp + 1), h1 = relu(opp.hp - 5); z = 10*h0 + h1
    names = state_feature_names()
    w1 = [[0.0] * FEATURE_LENGTH for _ in range(2)]
    w1[0][names.index("opp.p0.hp")] = -1.0
    w1[1][names.index("opp.p0.hp")] = 1.0
    mlp_path = tmp_path / "mlp.pev"
    write_mlp(mlp_path, w1, [1.0, -5.0], [10.0, 1.0], b2=0.25)
    mlp = Evaluator.load(str(mlp_path))
    assert mlp.kind == "mlp" and mlp.hidden_size == 2
    # opp.hp = 1 -> h0 = 0, h1 = 0 (relu(-4)) -> z = 0.25
    assert mlp.evaluate(state) == pytest.approx(0.25)
    result = monte_carlo_tree_search(state, iterations=500, evaluator=mlp)
    assert result.total_visits >= 500


def test_bad_weights_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        Evaluator.linear([0.0, 1.0])
    bad = tmp_path / "bad.pev"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        Evaluator.load(str(bad))


def test_multithreaded_search_accepts_evaluator():
    ev = Evaluator.linear(_opp_hp_weights(-3.0))
    result = monte_carlo_tree_search(
        make_state(), iterations=3000, threads=2, evaluator=ev
    )
    assert result.total_visits >= 3000
    assert _closecombat_share(result) > 0.5


def test_leaf_scale_default_getter_and_validation(tmp_path):
    assert Evaluator.linear(_opp_hp_weights(-3.0)).leaf_scale == 1.0
    assert Evaluator.linear(_opp_hp_weights(-3.0), leaf_scale=2.5).leaf_scale == 2.5
    path = tmp_path / "s.pev"
    poke_engine.write_linear(str(path), _opp_hp_weights(-3.0), 0.0)
    assert Evaluator.load(str(path), leaf_scale=4.0).leaf_scale == 4.0
    mlp = Evaluator.mlp([0.0] * FEATURE_LENGTH, [0.0], [0.0], leaf_scale=0.5)
    assert mlp.leaf_scale == 0.5
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Evaluator.linear(_opp_hp_weights(-3.0), leaf_scale=bad)
        with pytest.raises(ValueError):
            Evaluator.load(str(path), leaf_scale=bad)
        with pytest.raises(ValueError):
            Evaluator.mlp([0.0] * FEATURE_LENGTH, [0.0], [0.0], leaf_scale=bad)


def test_logit_and_stock_evaluate_bindings():
    ev = Evaluator.linear(_opp_hp_weights(-3.0), bias=0.5)
    state = make_state()
    assert ev.logit(state) == ev.evaluate(state)
    assert isinstance(poke_engine.evaluate(state), float)
    assert math.isfinite(poke_engine.evaluate(state))
    assert poke_engine.evaluate(state) == poke_engine.evaluate(make_state())


def test_leaf_scale_does_not_change_logit():
    state = make_state()
    a = Evaluator.linear(_opp_hp_weights(-3.0))
    b = Evaluator.linear(_opp_hp_weights(-3.0), leaf_scale=3.0)
    assert a.logit(state) == b.logit(state)
    assert monte_carlo_tree_search(state, iterations=200, evaluator=b).total_visits > 0


def test_selection_modes_return_a_strategy():
    ucb = monte_carlo_tree_search(make_state(), iterations=3000, selection="ucb1")
    total = sum(m.visits for m in ucb.side_one)
    for m in ucb.side_one:  # ucb1 reports its visit share as the strategy
        assert m.avg_strategy == pytest.approx(m.visits / total, abs=1e-5)
    for mode in ("exp3", "rm"):
        result = monte_carlo_tree_search(make_state(), iterations=5000, selection=mode, gamma=0.1)
        assert result.total_visits >= 5000
        for side in (result.side_one, result.side_two):
            assert sum(m.avg_strategy for m in side) == pytest.approx(1.0, abs=1e-3)
            assert all(m.avg_strategy >= 0.1 / len(side) - 1e-3 for m in side)


def test_selection_validation_and_threads():
    state = make_state()
    with pytest.raises(ValueError):
        monte_carlo_tree_search(state, iterations=10, selection="nope")
    with pytest.raises(ValueError):
        monte_carlo_tree_search(state, iterations=10, selection="rm", gamma=0.0)
    with pytest.raises(ValueError):
        monte_carlo_tree_search(state, iterations=10, selection="exp3", threads=2)
    assert monte_carlo_tree_search(state, iterations=500, threads=2).total_visits > 0


def test_return_joint_matrix_is_consistent():
    plain = monte_carlo_tree_search(make_state(), iterations=500)
    assert plain.joint is None
    for mode in ("ucb1", "exp3", "rm"):
        result = monte_carlo_tree_search(
            make_state(), iterations=4000, selection=mode, return_joint=True
        )
        k1, k2 = len(result.side_one), len(result.side_two)
        assert result.joint
        visited = sum(v for _, _, v, _ in result.joint)
        assert result.total_visits - 1 <= visited <= result.total_visits
        for i, j, visits, mean in result.joint:
            assert 0 <= i < k1 and 0 <= j < k2 and visits > 0
            assert 0.0 <= mean <= 1.0
        for i, option in enumerate(result.side_one):  # rows sum to the option visits
            assert sum(v for a, _, v, _ in result.joint if a == i) == option.visits
        for j, option in enumerate(result.side_two):  # columns too
            assert sum(v for _, b, v, _ in result.joint if b == j) == option.visits
        # the visit-weighted mean of the cells is side one's root score
        total = sum(v * m for _, _, v, m in result.joint)
        by_option = sum(o.total_score for o in result.side_one)
        assert total == pytest.approx(by_option, rel=1e-3, abs=1e-2)


def test_return_joint_rejects_threads():
    with pytest.raises(ValueError):
        monte_carlo_tree_search(make_state(), iterations=10, threads=2, return_joint=True)


# ---- feature version 2 (589.42) ---------------------------------------------


def _v2_state() -> State:
    one = _pokemon(
        "corviknight", ("flying", "steel"), (400, 250, 300, 150, 200, 200),
        ("roost", "swordsdance", "bravebird", "defog"),
    )
    one = Pokemon(**{**{k: getattr(one, k) for k in (
        "id", "level", "types", "base_types", "hp", "maxhp", "ability", "base_ability",
        "attack", "defense", "special_attack", "special_defense", "speed", "status",
        "tera_type", "moves")}, "item": "lifeorb"})
    two = _pokemon(
        "ferrothorn", ("grass", "steel"), (400, 200, 300, 100, 250, 50),
        ("stealthrock", "leechseed", "powerwhip", "protect"),
    )
    return State(side_one=Side(pokemon=[one]), side_two=Side(pokemon=[two]))


def test_feature_versions_and_lengths():
    assert poke_engine.FEATURE_VERSIONS == [1, 2, 3]
    assert poke_engine.feature_length() == FEATURE_LENGTH == DOCUMENTED_FEATURE_LENGTH
    assert poke_engine.feature_length(2) == poke_engine.FEATURE_LENGTH_V2 > FEATURE_LENGTH
    assert 1500 < poke_engine.FEATURE_LENGTH_V2 < 3000
    assert len(state_feature_names(2)) == poke_engine.FEATURE_LENGTH_V2
    assert state_feature_names(2)[:FEATURE_LENGTH] == state_feature_names(1)
    assert poke_engine.feature_length(3) == poke_engine.FEATURE_LENGTH_V3 == FEATURE_LENGTH + 5 * 36 + 8
    assert len(state_feature_names(3)) == poke_engine.FEATURE_LENGTH_V3
    assert state_feature_names(3)[:FEATURE_LENGTH] == state_feature_names(1)
    with pytest.raises(ValueError):
        state_features(make_state(), 1, 7)


def test_v2_features_extend_v1_with_identity_features():
    state = _v2_state()
    v1, v2 = state_features(state, 1), state_features(state, 1, 2)
    assert v2[:FEATURE_LENGTH] == v1 and len(v2) == poke_engine.FEATURE_LENGTH_V2
    names = state_feature_names(2)

    def at(n):
        return v2[names.index(n)]

    assert at("own.p0.item.lifeorb") == 1.0 and at("own.p0.item.leftovers") == 0.0
    assert at("own.p0.mv.recovery") == 1.0 and at("own.p0.mv.hazard_removal") == 1.0
    assert at("own.p0.mv.setup.attack") == 1.0
    assert at("opp.p0.mv.hazard_setter") == 1.0 and at("opp.p0.mv.protect") == 1.0
    assert at("opp.p0.mv.leechseed") == 1.0
    assert at("own.active.moves_first") == 1.0
    flipped = state_features(state, 2, 2)
    assert flipped[names.index("opp.p0.item.lifeorb")] == 1.0


def test_v3_matchup_features_extend_v1():
    state = _v2_state()  # corviknight (fast, Bravebird vs grass) against ferrothorn
    v1, v3 = state_features(state, 1), state_features(state, 1, 3)
    assert v3[:FEATURE_LENGTH] == v1 and len(v3) == poke_engine.FEATURE_LENGTH_V3
    names = state_feature_names(3)

    def at(vec, n):
        return vec[names.index(n)]

    assert at(v3, "match.speed.own0.opp0") == 1.0
    assert at(v3, "match.dmg_own.own0.opp0") > at(v3, "match.dmg_opp.own0.opp0") > 0.0
    assert at(v3, "match.speed.own1.opp0") == 0.0  # empty slots
    flipped = state_features(state, 2, 3)
    assert at(flipped, "match.speed.own0.opp0") == -1.0
    assert at(flipped, "match.dmg_opp.own0.opp0") == at(v3, "match.dmg_own.own0.opp0")


def test_v3_evaluator_roundtrip(tmp_path):
    n = poke_engine.FEATURE_LENGTH_V3
    names = state_feature_names(3)
    w1 = [0.0] * (2 * n)
    w1[names.index("match.speed.own0.opp0")] = 2.0  # hidden 0
    path = tmp_path / "v3.pev"
    write_mlp(str(path), w1, [0.0, 0.0], [1.0, 0.0], 0.0, feature_version=3)
    ev = Evaluator.load(str(path))
    assert ev.feature_version == 3 and ev.hidden_size == 2
    assert ev.evaluate(_v2_state()) == pytest.approx(2.0)


def test_v2_evaluator_roundtrip(tmp_path):
    n = poke_engine.FEATURE_LENGTH_V2
    names = state_feature_names(2)
    w1 = [0.0] * (2 * n)
    w1[names.index("own.p0.mv.recovery")] = 1.5  # hidden 0
    w1[n + names.index("opp.p0.mv.hazard_setter")] = -2.0  # hidden 1
    path = tmp_path / "v2.pev"
    write_mlp(str(path), w1, [0.0, 1.0], [1.0, 1.0], 0.1, feature_version=2)
    raw = path.read_bytes()
    assert int.from_bytes(raw[8:12], "little") == 2 and int.from_bytes(raw[16:20], "little") == n
    ev = Evaluator.load(str(path))
    assert ev.feature_version == 2 and ev.hidden_size == 2
    # hidden0 = relu(1.5) = 1.5; hidden1 = relu(-2 + 1) = 0
    assert ev.evaluate(_v2_state()) == pytest.approx(1.5 + 0.1)
    # a v2 file with v1-sized weights is rejected up front
    with pytest.raises(ValueError):
        write_mlp(str(path), [0.0] * (2 * FEATURE_LENGTH), [0.0, 0.0], [1.0, 1.0], feature_version=2)
    plain = tmp_path / "v1.pev"
    write_linear(str(plain), [0.0] * FEATURE_LENGTH, 0.5)
    assert Evaluator.load(str(plain)).feature_version == 1
