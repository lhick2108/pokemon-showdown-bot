"""Feature version 5 (v4 without the v3 matchup block, bead showdown-bot-589.66): layout, export
round trip, partial warm start from a v4 net and the CLI."""

from __future__ import annotations

import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from test_recorder import fake_world  # noqa: E402,F401
from test_train import state_strings  # noqa: E402,F401
from test_train_subsample import _write_games  # noqa: E402

from showdown_bot import dataset, train  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(pe, "FEATURE_LENGTH_V5"), reason="engine wheel without feature version 5")

DIMS = (4, 3, 2, 5)


def _net(version: int, hidden: int = 8, residual: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    n_dense = pe.FEATURE_LENGTH_V4 if version == 4 else pe.FEATURE_LENGTH_V5
    model = train.build_model("mlp", n_dense + 84, hidden, residual=(0.9, -0.1) if residual else None,
                              embed_dims=DIMS)
    net = model.net if residual else model
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


def test_layout(state_strings):  # noqa: F811
    assert pe.FEATURE_LENGTH_V5 == pe.FEATURE_LENGTH_V2 == 2191 == train.V5_DENSE
    assert pe.FEATURE_LENGTH_V4 == train.V4_DENSE
    assert 5 in pe.FEATURE_VERSIONS
    s = pe.State.from_string(state_strings[0])
    assert pe.state_features(s, 1, 5) == pe.state_features(s, 1, 2)
    assert pe.state_features(s, 1, 4)[:2191] == pe.state_features(s, 1, 5)
    assert pe.state_feature_names(5) == pe.state_feature_names(2)
    row = dataset.state_row(pe, s, 1, 5)
    assert len(row) == 2191 + 84
    assert row[2191:] == [float(i) for i in pe.state_identity(s, 1)]


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("hidden", [8, 128])
def test_export_round_trip(tmp_path, state_strings, residual, hidden):  # noqa: F811
    model = _net(5, hidden=hidden, residual=residual)
    out = tmp_path / "v5.pev"
    train.export_pev(model, out, 5)
    ev = pe.Evaluator.load(str(out))
    assert ev.feature_version == 5 and ev.hidden_size == hidden and list(ev.embed_dims) == list(DIMS)
    raw = out.read_bytes()
    assert raw[4:8] == (3).to_bytes(4, "little") and raw[8:12] == (5).to_bytes(4, "little")
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3
    parsed = train.read_pev(out)
    assert parsed["feature_version"] == 5 and parsed["n_in"] == 2191
    assert '"feature_version": 5' in (tmp_path / "v5.pev.vocab.json").read_text()


@pytest.mark.parametrize("residual", [False, True])
def test_partial_warm_start_from_v4(tmp_path, state_strings, residual):  # noqa: F811
    """v5 seeded from a v4 net = that v4 net with the matchup input columns removed: its logits
    equal the v4 net's on states whose matchup block is multiplied by zero weights."""
    src = _net(4, residual=residual, seed=1)
    # zero the matchup columns of the source so both nets compute the same function
    net4 = src.net if residual else src
    with torch.no_grad():
        net4.l1.weight[:, train.V5_DENSE:train.V4_DENSE] = 0.0
    p4 = tmp_path / "v4.pev"
    train.export_pev(src, p4, 4)
    n_feat = pe.FEATURE_LENGTH_V5 + 84
    fresh = train.build_model("mlp", n_feat, 8, residual=(0.3, 0.0) if residual else None, embed_dims=DIMS)
    train.load_init(fresh, p4, model_kind="mlp", n_feat=n_feat, hidden=8, feature_version=5, residual=residual)
    p5 = tmp_path / "v5.pev"
    train.export_pev(fresh, p5, 5)
    ev4, ev5 = pe.Evaluator.load(str(p4)), pe.Evaluator.load(str(p5))
    for ss in state_strings:
        s = pe.State.from_string(ss)
        assert ev5.evaluate(s) == pytest.approx(ev4.evaluate(s), abs=1e-4)
    # tables and the other layers are carried over
    a, b = train.read_pev(p4), train.read_pev(p5)
    assert (a["b1"] == b["b1"]).all() and (a["w2"] == b["w2"]).all()
    assert all((x == y).all() for x, y in zip(a["tables"], b["tables"]))
    # the reverse direction and plain v5 -> v4 are refused
    with pytest.raises(ValueError, match="feature version"):
        train.load_init(_net(4, residual=residual), p5, model_kind="mlp",
                        n_feat=pe.FEATURE_LENGTH_V4 + 84, hidden=8, feature_version=4, residual=residual)


def test_cli_smoke_v5_with_v4_init(tmp_path, state_strings, capsys):  # noqa: F811
    root = tmp_path / "games"
    _write_games(root, state_strings, n_games=12, n_dec=4, searched=True, prefix="s")
    common = ["--data", str(root), "--embed-dim", "4", "--hidden", "8", "--epochs", "1", "--device", "cpu",
              "--load-workers", "1", "--val-fraction", "0.25", "--residual", "--value-target", "search"]
    v4 = tmp_path / "v4.pev"
    assert train.main(common + ["--features", "v4", "--out", str(v4)]) == 0
    v5 = tmp_path / "v5.pev"
    assert train.main(common + ["--features", "v5", "--init", str(v4), "--lr", "1e-4", "--out", str(v5)]) == 0
    cap = capsys.readouterr()
    assert "warm start" in cap.out + cap.err
    ev = pe.Evaluator.load(str(v5))
    assert ev.feature_version == 5 and ev.residual and list(ev.embed_dims) == [4, 4, 4, 4]
