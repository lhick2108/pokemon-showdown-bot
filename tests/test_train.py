"""Value-net training: split, loss reduction, .pev export round trip (no server)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import augment, engine_state, train  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def state_strings():
    data = RandbatsData.load(cache_dir=FIXTURES)
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    trackers = OpponentModel(data).update(battle)
    sampler = engine_state.TeamSampler(data)
    return [
        engine_state.build_state(
            battle, engine_state.sample_opponent_team(battle, trackers, sampler, random.Random(seed))
        ).to_string()
        for seed in range(8)
    ]


@pytest.mark.parametrize("kind,hidden", [("mlp", 16), ("mlp", 64), ("linear", 0)])
def test_export_round_trip_matches_torch(tmp_path, state_strings, kind, hidden):
    torch.manual_seed(3)
    model = train.build_model(kind, pe.FEATURE_LENGTH, hidden)
    with torch.no_grad():  # make the biases and weights non-trivial
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    out = tmp_path / "m.pev"
    train.export_pev(model, out)
    assert pe.Evaluator.load(str(out)).kind == kind
    worst = train.verify_export(model, out, state_strings, tol=1e-3)
    assert worst < 1e-3


def test_split_never_shares_a_battle_tag():
    tags = [f"battle-{i % 23}" for i in range(500)]
    for seed in range(5):
        tr, va = train.split_by_battle(tags, 0.2, seed)
        assert len(tr) + len(va) == len(tags)
        assert not {tags[i] for i in tr} & {tags[i] for i in va}
        assert 1 <= len({tags[i] for i in va}) <= 23 - 1
    # deterministic for a fixed seed
    assert (train.split_by_battle(tags, 0.2, 7)[1] == train.split_by_battle(tags, 0.2, 7)[1]).all()


def test_weights_downweight_long_games():
    tags = ["a"] * 100 + ["b"] * 4
    for mode in ("sqrt", "game"):
        w = train.sample_weights(tags, mode)
        assert abs(w.mean() - 1) < 1e-5 and w[0] < w[-1]
    assert train.sample_weights(tags, "none").tolist() == [1.0] * 104
    w = train.sample_weights(tags, "game")
    assert abs(w[:100].sum() - w[100:].sum()) < 1e-3


def _synthetic(n: int, seed: int) -> train.Data:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 20)).astype(np.float32)
    y = (x @ np.linspace(-1, 1, 20) + 0.3 * rng.normal(size=n) > 0).astype(np.float32)
    tags = [f"t{i // 10}" for i in range(n)]
    return train.Data(x, y, np.ones(n, dtype=np.float32), tags, [None] * n)


def test_training_reduces_loss_on_tiny_dataset():
    data = _synthetic(600, 0)
    tr_idx, va_idx = train.split_by_battle(data.tags, 0.25, 0)
    tr, va = data.subset(tr_idx), data.subset(va_idx)
    base = train.constant_baseline_bce(va.y, va.w, tr.y, tr.w)
    model = train.build_model("mlp", 20, 8)
    result = train.fit(
        model, tr, va, device=torch.device("cpu"), epochs=40, patience=8, lr=1e-2, log=lambda *_: None
    )
    first, best = result.history[0][2], min(h[2] for h in result.history)
    assert best < first
    assert best < base - 0.1  # clearly beats the constant predictor
    acc, n = train.decisive_accuracy(
        train.logits_of(result.model, torch.from_numpy(va.x)).numpy(), va.y
    )
    assert n == len(va.y) and acc > 0.8


def test_metrics_helpers():
    logits = np.array([-3.0, 3.0, 0.5, -0.5])
    y = np.array([0.0, 1.0, 0.5, 1.0], dtype=np.float32)
    acc, n = train.decisive_accuracy(logits, y)
    assert n == 3 and abs(acc - 2 / 3) < 1e-9
    assert sum(b[2] for b in train.calibration_buckets(logits, y)) == 4


def test_make_targets_modes():
    y = np.array([1.0, 0.0, 0.5, 1.0], dtype=np.float32)
    q = np.array([0.8, 0.4, np.nan, 0.2], dtype=np.float32)
    out, n = train.make_targets(y, q, "outcome")
    assert out is y and n == 0  # exactly the old target
    out, n = train.make_targets(y, None, "blend")
    assert out is y and n == 0
    out, n = train.make_targets(y, q, "search")
    assert n == 1 and np.allclose(out, [0.8, 0.4, 0.5, 0.2])  # NaN falls back to z
    out, n = train.make_targets(y, q, "blend", 0.25)
    assert n == 1 and np.allclose(out, [0.95, 0.1, 0.5, 0.8])
    out, _ = train.make_targets(y, q, "blend", 0.0)
    assert np.allclose(out, y)
    with pytest.raises(ValueError):
        train.make_targets(y, q, "nope")


def test_data_targets_follow_subset_and_fit_uses_them():
    data = _synthetic(60, 1)
    data.q = np.where(np.arange(60) % 5 == 0, np.nan, 0.5).astype(np.float32)
    assert data.y_fit is data.y  # outcome mode: unchanged labels
    data.target, _ = train.make_targets(data.y, data.q, "search")
    sub = data.subset(np.array([0, 1, 2]))
    assert sub.y_fit.tolist() == [data.y[0], 0.5, 0.5] and sub.y.tolist() == data.y[:3].tolist()
    corr, n = train.q_z_correlation(data)
    assert n == 48 and np.isnan(corr)  # q is constant, so the correlation is undefined


# --- residual evaluator (stock + learned correction) -------------------------


def _perturb(model):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)


@pytest.mark.parametrize("kind,hidden", [("mlp", 16), ("linear", 0)])
def test_residual_export_round_trip_matches_torch(tmp_path, state_strings, kind, hidden):
    torch.manual_seed(5)
    model = train.build_model(kind, pe.FEATURE_LENGTH, hidden, residual=(0.7, -0.1))
    _perturb(model.net)
    out = tmp_path / "res.pev"
    train.export_pev(model, out)
    raw = out.read_bytes()
    assert int.from_bytes(raw[4:8], "little") == 2  # format 2
    ev = pe.Evaluator.load(str(out))
    assert ev.residual
    assert ev.stock_coef == pytest.approx(0.7 * 0.0125, rel=1e-6)
    assert ev.kind == kind
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3
    # stock term really contributes: differs from the bare net on some state
    bare = tmp_path / "bare.pev"
    train.export_pev(model.net, bare)
    bev = pe.Evaluator.load(str(bare))
    assert not bev.residual and bev.stock_coef == 0.0
    states = [pe.State.from_string(s) for s in state_strings]
    for s in states:
        assert ev.evaluate(s) == pytest.approx(
            bev.evaluate(s) + ev.stock_coef * pe.evaluate(s), abs=1e-4
        )


@pytest.mark.parametrize("kind,hidden", [("mlp", 8), ("linear", 0)])
def test_zero_init_residual_is_the_stock_logistic(state_strings, kind, hidden):
    a, b = 0.5, -0.04
    model = train.build_model(kind, pe.FEATURE_LENGTH, hidden, residual=(a, b))
    states = [pe.State.from_string(s) for s in state_strings]
    rows = [pe.state_features(s, 1) + [pe.evaluate(s)] for s in states]
    z = train.logits_of(model, torch.tensor(rows, dtype=torch.float32)).numpy()
    expected = np.array([a * 0.0125 * pe.evaluate(s) + b for s in states])
    np.testing.assert_allclose(z, expected, atol=1e-5)


def test_fit_stock_logistic_recovers_parameters():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 150, 60000).astype(np.float32)
    p = 1 / (1 + np.exp(-(0.5 * 0.0125 * e - 0.04)))
    y = (rng.random(len(e)) < p).astype(np.float32)
    a, b = train.fit_stock_logistic(e, y, np.ones_like(y))
    assert a == pytest.approx(0.5, abs=0.03) and b == pytest.approx(-0.04, abs=0.03)
    a2, b2 = train.fit_stock_logistic(e, y, np.ones_like(y), fixed_a=0.5)
    assert a2 == 0.5 and b2 == pytest.approx(-0.04, abs=0.03)


def test_v1_files_still_load_and_plain_writes_stay_v1(tmp_path):
    path = tmp_path / "plain.pev"
    pe.write_mlp(str(path), [0.0] * (4 * pe.FEATURE_LENGTH), [0.0] * 4, [1.0] * 4, 0.25)
    raw = path.read_bytes()
    assert int.from_bytes(raw[4:8], "little") == 1
    assert len(raw) == 24 + 4 * (4 * pe.FEATURE_LENGTH + 4 + 4 + 1)
    ev = pe.Evaluator.load(str(path))
    assert not ev.residual and ev.stock_coef == 0.0 and ev.hidden_size == 4
    state = pe.State.from_string(pe.State().to_string())
    assert ev.evaluate(state) == pytest.approx(0.25)
    # an explicit stock_coef=0 is also plain v1
    path0 = tmp_path / "plain0.pev"
    pe.write_linear(str(path0), [0.0] * pe.FEATURE_LENGTH, 0.5, stock_coef=0.0)
    assert int.from_bytes(path0.read_bytes()[4:8], "little") == 1


def test_residual_training_starts_at_stock_and_does_not_get_worse(state_strings):
    states = [pe.State.from_string(s) for s in state_strings]
    x = np.array([pe.state_features(s, 1) + [pe.evaluate(s)] for s in states], dtype=np.float32)
    rng = np.random.default_rng(1)
    y = (rng.random(len(x)) < 0.5).astype(np.float32)
    w = np.ones(len(x), dtype=np.float32)
    data = train.Data(x, y, w, [f"b{i}" for i in range(len(x))], list(state_strings), residual=True)
    a0, b0 = train.fit_stock_logistic(data.e, y, w)
    model = train.build_model("mlp", pe.FEATURE_LENGTH, 4, (a0, b0), freeze_a=True)
    res = train.fit(model, data, data, device=torch.device("cpu"), epochs=3, seed=0, log=lambda *_: None)
    assert float(res.model.a) == pytest.approx(a0)  # frozen
    assert res.history[0][2] < 1.0


# ---- augmentation (589.20) -------------------------------------------------


def _batch(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, pe.FEATURE_LENGTH, generator=g)


def test_layout_matches_engine_names():
    augment.check_layout()
    bad = list(pe.state_feature_names())
    bad[augment.SLOT] = "own.p1.attack"
    with pytest.raises(AssertionError):
        augment.check_layout(bad)


def test_bench_permutation_moves_whole_blocks_only():
    x = _batch()
    out = augment.permute_bench(x, torch.Generator().manual_seed(1))
    S, B = augment.SLOT, augment.SIDE
    for s in range(2):
        o = s * B
        assert torch.equal(out[:, o:o + S], x[:, o:o + S])  # active fixed
        assert torch.equal(out[:, o + 6 * S:o + B], x[:, o + 6 * S:o + B])  # extras fixed
        for r in range(len(x)):  # each output bench block is exactly one input bench block, bijectively
            src = []
            for k in range(1, 6):
                blk = out[r, o + k * S:o + (k + 1) * S]
                m = [j for j in range(1, 6) if torch.equal(blk, x[r, o + j * S:o + (j + 1) * S])]
                assert len(m) == 1
                src.append(m[0])
            assert sorted(src) == [1, 2, 3, 4, 5]
    assert torch.equal(out[:, 2 * B:], x[:, 2 * B:])  # field fixed
    assert not torch.equal(out, x)


def test_bench_permutation_leaves_residual_column_alone():
    x = _batch(8, 2)
    out = augment.permute_bench(x)
    assert torch.equal(out[:, -1], x[:, -1])


def test_side_swap_equals_side_two_features(state_strings):
    rows1 = torch.tensor([pe.state_features(pe.State.from_string(s), 1) for s in state_strings])
    rows2 = torch.tensor([pe.state_features(pe.State.from_string(s), 2) for s in state_strings])
    flip = torch.ones(len(rows1), dtype=torch.bool)
    assert torch.equal(augment.swap_sides(rows1, flip), rows2)
    assert torch.equal(augment.swap_sides(rows1, ~flip), rows1)


def test_augmenter_flips_labels_and_residual_eval():
    x = torch.cat([_batch(32, 3), torch.randn(32, 1)], dim=1)
    y = torch.rand(32)
    aug = augment.Augmenter(bench=False, sides=True, residual=True, seed=4)
    xo, yo = aug(x, y)
    flipped = ~torch.isclose(yo, y)
    assert flipped.any() and (~flipped).any()
    assert torch.allclose(yo[flipped], 1 - y[flipped])
    assert torch.equal(xo[flipped, -1], -x[flipped, -1])
    assert torch.equal(xo[~flipped], x[~flipped])
    assert torch.equal(xo[flipped, :augment.SIDE], x[flipped, augment.SIDE:2 * augment.SIDE])


def test_dropout_model_exports_exactly(tmp_path, state_strings):
    torch.manual_seed(5)
    model = train.build_model("mlp", pe.FEATURE_LENGTH, 16, dropout=0.5)
    out = tmp_path / "d.pev"
    train.export_pev(model, out)
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3
    xs = _batch(4)
    model.train()
    assert not torch.equal(model(xs), model(xs))  # dropout active in training
    res = train.build_model("mlp", pe.FEATURE_LENGTH, 16, residual=(0.7, -0.1), dropout=0.5)
    train.export_pev(res, tmp_path / "r.pev")
    assert train.verify_export(res, tmp_path / "r.pev", state_strings, tol=1e-3) < 1e-3


def test_td_modes_in_make_targets_leave_outcome_untouched():
    y = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    target, n = train.make_targets(y, None, "outcome")
    assert target is y and n == 0
    assert {"td", "nstep"} <= set(train.VALUE_TARGETS)


# ---- feature version 2 (589.42) ---------------------------------------------


@pytest.mark.parametrize("residual", [False, True])
def test_v2_export_round_trip(tmp_path, state_strings, residual):
    n = pe.feature_length(2)
    torch.manual_seed(4)
    model = train.build_model("mlp", n, 16, (0.8, 0.1) if residual else None)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    out = tmp_path / "v2.pev"
    train.export_pev(model, out, feature_version=2)
    assert pe.Evaluator.load(str(out)).feature_version == 2
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3


@pytest.mark.parametrize("residual", [False, True])
def test_v3_export_round_trip(tmp_path, state_strings, residual):
    n = pe.feature_length(3)
    torch.manual_seed(5)
    model = train.build_model("mlp", n, 16, (0.8, 0.1) if residual else None)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)
    out = tmp_path / "v3.pev"
    train.export_pev(model, out, feature_version=3)
    assert pe.Evaluator.load(str(out)).feature_version == 3
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3


def test_float16_data_keeps_the_stock_eval_separate(state_strings):
    states = [pe.State.from_string(s) for s in state_strings]
    x = np.array([pe.state_features(s, 1, 2) for s in states], dtype=np.float16)
    e = np.array([pe.evaluate(s) for s in states], dtype=np.float32)
    n = len(x)
    y = (np.arange(n) % 2).astype(np.float32)
    data = train.Data(
        x, y, np.ones(n, dtype=np.float32), [f"b{i}" for i in range(n)], list(state_strings),
        stock_eval=e, residual=True, e_sep=e, feature_version=2,
    )
    assert np.array_equal(data.e, e) and data.x.shape[1] == pe.feature_length(2)
    sub = data.subset(np.arange(0, n, 2))
    assert sub.e_sep.shape == (len(sub),) and sub.x.dtype == np.float16
    a0, b0 = train.fit_stock_logistic(data.e, y, data.w)
    model = train.build_model("mlp", pe.feature_length(2), 4, (a0, b0), freeze_a=True)
    res = train.fit(model, data, data, device=torch.device("cpu"), epochs=2, log=lambda *_: None)
    assert res.history[0][2] < 1.5


def test_stock_logistic_baseline_is_train_fit_val_scored_apples_to_apples():
    rng = np.random.default_rng(3)

    def make(n, tag):
        e = rng.normal(0, 150, n).astype(np.float32)
        y = (rng.random(n) < 1 / (1 + np.exp(-0.6 * 0.0125 * e))).astype(np.float32)
        w = rng.uniform(0.5, 1.5, n).astype(np.float32)
        w /= w.mean()
        x = np.zeros((n, 2), dtype=np.float32)
        return train.Data(x, y, w, [f"{tag}{i}" for i in range(n)], [None] * n, stock_eval=e,
                          residual=True, e_sep=e)

    tr, va = make(4000, "t"), make(1500, "v")
    bce, a, b = train.stock_eval_logistic_bce(tr, va)
    a0, b0 = train.fit_stock_logistic(tr.e, tr.y, tr.w)  # the residual-mode "stock-only" fit
    assert (a, b) == pytest.approx((a0, b0), abs=1e-6)
    logits = torch.from_numpy((0.0125 * a0 * va.e + b0).astype(np.float32))
    expect = train.weighted_bce(logits, torch.from_numpy(va.y), torch.from_numpy(va.w))
    assert bce == pytest.approx(expect, abs=1e-5)  # val weights, train fit
    # it is a train fit: refitting on val is at least as good in-sample
    av, bv = train.fit_stock_logistic(va.e, va.y, va.w)
    own = torch.from_numpy((0.0125 * av * va.e + bv).astype(np.float32))
    assert train.weighted_bce(own, torch.from_numpy(va.y), torch.from_numpy(va.w)) <= bce + 1e-6
    assert a == pytest.approx(0.6, abs=0.15)
