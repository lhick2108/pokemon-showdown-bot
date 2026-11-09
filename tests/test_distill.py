"""train.py --distill: teacher targets, truncation init, CLI smoke (docs/eval-speed.md)."""
from __future__ import annotations

import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from test_recorder import fake_world  # noqa: E402,F401
from test_train import state_strings  # noqa: E402,F401
from test_train_subsample import _write_games  # noqa: E402

from showdown_bot import train  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(pe, "FEATURE_LENGTH_V5"), reason="engine wheel without feature version 5")


def _teacher(path, hidden=16, seed=3):
    torch.manual_seed(seed)
    n_feat = pe.FEATURE_LENGTH_V5 + 84
    model = train.build_model("mlp", n_feat, hidden, residual=(0.9, -0.1), embed_dims=(4, 4, 4, 4))
    with torch.no_grad():
        for p in model.net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    train.export_pev(model, path, 5)
    return model


def test_load_teacher_matches_export(tmp_path, state_strings):  # noqa: F811
    p = tmp_path / "t.pev"
    _teacher(p)
    t = train.load_teacher(p, n_feat=pe.FEATURE_LENGTH_V5 + 84, feature_version=5, residual=True)
    assert train.verify_export(t, p, state_strings, tol=1e-4) < 1e-4


def test_cli_distill_trunc_and_targets(tmp_path, state_strings, capsys):  # noqa: F811
    root = tmp_path / "games"
    _write_games(root, state_strings, n_games=12, n_dec=4, searched=True, prefix="s")
    teacher = tmp_path / "t.pev"
    _teacher(teacher, hidden=16)
    common = ["--data", str(root), "--embed-dim", "4", "--epochs", "1", "--device", "cpu",
              "--load-workers", "1", "--val-fraction", "0.25", "--residual", "--value-target", "search",
              "--features", "v5", "--distill", str(teacher)]
    # same hidden size, lr 0: the truncation init reproduces the teacher exactly
    same = tmp_path / "same.pev"
    assert train.main(common + ["--hidden", "16", "--lr", "0", "--out", str(same)]) == 0
    out = capsys.readouterr().out
    assert "distill: teacher" in out and "|student - teacher| logit mean 0.0000" in out
    ev_t, ev_s = pe.Evaluator.load(str(teacher)), pe.Evaluator.load(str(same))
    for ss in state_strings:
        s = pe.State.from_string(ss)
        assert ev_s.evaluate(s) == pytest.approx(ev_t.evaluate(s), abs=1e-5)
    # smaller student: trains and exports; larger than the teacher is refused for trunc init
    small = tmp_path / "small.pev"
    assert train.main(common + ["--hidden", "8", "--out", str(small)]) == 0
    assert pe.Evaluator.load(str(small)).hidden_size == 8
    assert train.main(common + ["--hidden", "32", "--out", str(tmp_path / "big.pev")]) == 2
    assert train.main(common + ["--hidden", "32", "--distill-init", "emb",
                                "--out", str(tmp_path / "big.pev")]) == 0
