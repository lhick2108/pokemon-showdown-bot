"""train.py --init (warm start from a .pev) and scripts/human_eval.py."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

pe = pytest.importorskip("poke_engine")

from test_train import state_strings  # noqa: E402,F401
from test_train_subsample import _write_games  # noqa: E402

from showdown_bot import train  # noqa: E402


def _perturbed(kind, hidden, residual, features=1, n_in=None):
    torch.manual_seed(11)
    n_in = n_in or pe.FEATURE_LENGTH
    model = train.build_model(kind, n_in, hidden, residual=(0.8, -0.2) if residual else None)
    net = model.net if residual else model
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


@pytest.mark.parametrize("kind,hidden,residual", [("mlp", 16, True), ("mlp", 8, False),
                                                  ("linear", 0, True)])
def test_init_round_trip(tmp_path, state_strings, kind, hidden, residual):  # noqa: F811
    src = _perturbed(kind, hidden, residual)
    path = tmp_path / "src.pev"
    train.export_pev(src, path)
    fresh = train.build_model(kind, pe.FEATURE_LENGTH, hidden,
                              residual=(0.3, 0.0) if residual else None)
    train.load_init(fresh, path, model_kind=kind, n_feat=pe.FEATURE_LENGTH, hidden=hidden,
                    feature_version=1, residual=residual)
    again = tmp_path / "again.pev"
    train.export_pev(fresh, again)
    assert again.read_bytes() == path.read_bytes()  # bit-identical weights, a and bias
    assert train.verify_export(fresh, path, state_strings, tol=1e-3) < 1e-3
    if residual:
        assert fresh.stock_coef == pytest.approx(src.stock_coef, rel=1e-6)


def test_init_mismatch_errors_name_the_problem(tmp_path):
    path = tmp_path / "m.pev"
    train.export_pev(_perturbed("mlp", 16, True), path)

    def load(**over):
        kw = dict(model_kind="mlp", n_feat=pe.FEATURE_LENGTH, hidden=16, feature_version=1,
                  residual=True)
        kw.update(over)
        model = train.build_model(kw["model_kind"], kw["n_feat"], kw["hidden"],
                                  residual=(0.5, 0.0) if kw["residual"] else None)
        train.load_init(model, path, **kw)

    load()  # baseline matches
    with pytest.raises(ValueError, match="hidden 16 .training 32"):
        load(hidden=32)
    with pytest.raises(ValueError, match="not residual|without --residual"):
        load(residual=False)
    with pytest.raises(ValueError, match="feature version 1 .training 2"):
        load(feature_version=2)
    with pytest.raises(ValueError, match="model kind mlp .training linear"):
        load(model_kind="linear", hidden=0)
    bad = tmp_path / "bad.pev"
    bad.write_bytes(b"nope" * 10)
    with pytest.raises(ValueError, match="bad magic"):
        train.read_pev(bad)


def test_cli_init_mismatch_exits_2(tmp_path, state_strings, capsys):  # noqa: F811
    games = tmp_path / "g"
    _write_games(games, state_strings, n_games=4, n_dec=3, searched=False, prefix="b")
    path = tmp_path / "m.pev"
    train.export_pev(_perturbed("mlp", 16, True), path)
    rc = train.main(["--data", str(games), "--residual", "--hidden", "8", "--model", "mlp",
                     "--init", str(path), "--out", str(tmp_path / "o.pev"), "--device", "cpu",
                     "--load-workers", "1", "--epochs", "1", "--states", "truth"])
    assert rc == 2
    assert "hidden 16" in capsys.readouterr().err


def _human_eval():
    spec = importlib.util.spec_from_file_location(
        "human_eval", Path(__file__).resolve().parent.parent / "scripts" / "human_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_human_eval_matches_engine_logit_and_is_reproducible(tmp_path, state_strings):  # noqa: F811
    he = _human_eval()
    games = tmp_path / "games"
    _write_games(games, state_strings, n_games=5, n_dec=6, searched=False, prefix="w", result="win")
    _write_games(games, state_strings, n_games=3, n_dec=6, searched=False, prefix="l", result="loss")
    m1, m2 = tmp_path / "a.pev", tmp_path / "b.pev"
    train.export_pev(_perturbed("mlp", 8, True), m1)
    train.export_pev(_perturbed("mlp", 8, False), m2)

    res = he.evaluate_models([str(m1), str(m2)], games, per_game=2, seed=0)
    assert res["n"] == 8 * 2
    assert res == {**he.evaluate_models([str(m1), str(m2)], games, per_game=2, seed=0),
                   "seconds": res["seconds"]}
    # reference: same positions scored directly with Evaluator.logit
    pos = list(he.sample_positions(games, per_game=2, seed=0))
    assert len(pos) == 16
    ev = pe.Evaluator.load(str(m1))
    ref = sum(max(z, 0) - z * y + math.log1p(math.exp(-abs(z)))
              for z, y in ((float(ev.logit(pe.State.from_string(s))), y) for s, y, _ in pos)) / len(pos)
    assert res["models"][str(m1)]["all"]["bce"] == pytest.approx(ref, rel=1e-9)
    assert res["models"][str(m1)]["all"]["bce"] != res["models"][str(m2)]["all"]["bce"]
    # 5 wins + 3 losses, 2 each: constant predictor p = 10/16
    p = 10 / 16
    assert res["constant"]["all"]["bce"] == pytest.approx(-(p * math.log(p) + (1 - p) * math.log(1 - p)))
    # turns 1..6 fall in the first two buckets only
    assert set(res["models"][str(m1)]) <= {"all", "1-5", "6-10"}
    assert len(list(he.sample_positions(games, max_games=2, per_game=0, seed=0))) == 12
