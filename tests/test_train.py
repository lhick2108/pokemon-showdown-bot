"""Value-net training: split, loss reduction, .pev export round trip (no server)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import engine_state, train  # noqa: E402
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
