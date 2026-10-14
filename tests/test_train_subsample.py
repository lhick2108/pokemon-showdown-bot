"""Per-source subsampling, source weights and --val-data in showdown_bot.train."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

pe = pytest.importorskip("poke_engine")

from test_recorder import fake_world  # noqa: E402
from test_train import _synthetic, state_strings  # noqa: E402,F401

from showdown_bot import dataset, recorder, train  # noqa: E402


def _write_games(root, states, n_games, n_dec, *, searched, prefix, result="win"):
    """``n_games`` records of ``n_dec`` decisions: truth + one searched world, or truth only."""
    rec = recorder.GameRecorder(root)
    for g in range(n_games):
        tag = f"{prefix}-{g}"
        rec.start_game(tag, "u")
        for d in range(n_dec):
            st = states[(g + d) % len(states)]
            dec = {"turn": d + 1, "policy": {"a": 1.0}, "chosen": "a",
                   "truth": {"state": st}, "truth_status": "ok"}
            if searched:
                dec["worlds"] = [fake_world(st)]
            rec.add_decision(tag, "u", dec)
        rec.end_game(tag, "u", getattr(recorder, f"RESULT_{result.upper()}"), turns=n_dec)


def test_parse_source_syntax():
    p = train.parse_source
    assert p("data/a") == train.Source(Path("data/a"), None, 1.0)
    assert p("data/a@3") == train.Source(Path("data/a"), 3, 1.0)
    assert p("data/a@3:w0.5") == train.Source(Path("data/a"), 3, 0.5)
    assert p("data/a:w2") == train.Source(Path("data/a"), None, 2.0)
    assert p(r"C:\data\human@4:w.25") == train.Source(Path(r"C:\data\human"), 4, 0.25)
    assert p("dir@x") == train.Source(Path("dir@x"), None, 1.0)  # not a spec suffix
    with pytest.raises(argparse.ArgumentTypeError):
        p("data/a:w0")
    args = train._parser().parse_args(
        ["--data", "a@2:w3", "--data", "b", "--val-data", "c@1", "--max-per-game", "5",
         "--early-stop-on", "val-data", "--out", "o.pev"])
    assert [(s.path.name, s.max_per_game, s.weight) for s in args.data] == [
        ("a", 2, 3.0), ("b", None, 1.0)]
    assert [s.path.name for s in args.val_data] == ["c"] and args.max_per_game == 5
    assert args.early_stop_on == "val-data"
    d = train._parser().parse_args(["--data", "a", "--out", "o.pev"])
    assert d.max_per_game == 0 and d.val_data == [] and d.early_stop_on == "val"


def test_subsample_count_and_reproducibility(tmp_path, state_strings):  # noqa: F811
    _write_games(tmp_path, state_strings, n_games=6, n_dec=8, searched=False, prefix="h")

    def picks(k, seed):
        return sorted((s.game_id, s.decision_idx) for s in dataset.iter_samples(
            tmp_path, truth=True, worlds=False, max_per_game=k, seed=seed))

    full = picks(0, 0)
    assert len(full) == 48
    a = picks(3, 0)
    assert len(a) == 18 and len(set(a)) == 18  # 3 distinct decisions per record
    assert {sum(1 for g, _ in a if g == gid) for gid, _ in a} == {3}
    assert a == picks(3, 0) and set(a) <= set(full)  # reproducible
    assert a != picks(3, 1)  # the seed matters
    assert len(picks(20, 0)) == 48  # K larger than the record keeps all
    loaded = train.load_data([train.Source(tmp_path, 2)], states="truth", seed=4)
    assert len(loaded) == 12  # 6 records x 2


def test_load_data_per_source_counts_weights_and_fallback(
        tmp_path, state_strings, capsys):  # noqa: F811
    human, selfp = tmp_path / "human", tmp_path / "selfplay"
    _write_games(human, state_strings, 5, 6, searched=False, prefix="h", result="loss")
    _write_games(selfp, state_strings, 3, 6, searched=True, prefix="s", result="win")
    data = train.load_data(
        [train.Source(human, 2, 0.5), train.Source(selfp, 4, 2.0)], states="truth",
        weighting="none", value_target="search", seed=1)
    out = capsys.readouterr().out
    assert (f"source {human}: max-per-game 2, weight x0.5: 5 records, 10 decisions kept, "
            f"10 samples") in out
    assert (f"source {selfp}: max-per-game 4, weight x2: 3 records, 12 decisions kept, "
            f"12 samples") in out
    assert f"source {human}: 10 of 10 samples fell back" in out
    assert f"source {selfp}: 0 of 12 samples fell back" in out
    assert len(data) == 22 and abs(float(data.w.mean()) - 1) < 1e-5
    is_h = np.array([t.startswith("h-") for t in data.tags])
    assert np.allclose(data.w[is_h], data.w[is_h][0])
    assert np.allclose(data.w[~is_h], data.w[~is_h][0])
    assert data.w[~is_h][0] / data.w[is_h][0] == pytest.approx(4.0)  # 2.0 / 0.5
    # human records have no search value: the search target falls back to the outcome z
    assert np.isnan(data.q[is_h]).all() and (data.target[is_h] == 0.0).all()
    assert (data.y[is_h] == 0.0).all() and (data.y[~is_h] == 1.0).all()
    # the multiplier composes with the sqrt weighting before normalisation
    sq = train.load_data([train.Source(human, 0, 3.0), train.Source(selfp, 0, 1.0)],
                         states="truth", weighting="sqrt")
    mult = np.where([t.startswith("h-") for t in sq.tags], 3.0, 1.0)
    assert np.allclose(sq.w, train.sample_weights(sq.tags, "sqrt", mult))


def test_val_data_reported_each_epoch_and_early_stopping_choice():
    train_d, val_d, extra = _synthetic(200, 0), _synthetic(60, 1), _synthetic(80, 2)
    lines = []
    kw = dict(device=train.select_device("cpu"), epochs=4, patience=100, seed=0, log=lines.append)
    torch.manual_seed(0)
    model = train.build_model("mlp", 20, 8)
    res = train.fit(model, train_d, val_d, extra_val=[("held", extra)], **kw)
    assert len(res.extra_history) == 4 and all(len(r) == 1 for r in res.extra_history)
    assert all("val-data[held]" in line for line in lines)
    final = train.evaluate_extra(model, extra, torch.device("cpu"))
    expect = train.weighted_bce(train.logits_of(model, torch.from_numpy(extra.x)),
                                torch.from_numpy(extra.y), torch.from_numpy(extra.w))
    assert final["bce_w"] == pytest.approx(expect, abs=1e-6)
    # default: val-data is only reported, the run is identical to one without it
    torch.manual_seed(0)
    plain = train.fit(train.build_model("mlp", 20, 8), train_d, val_d, **kw)
    assert plain.best_epoch == res.best_epoch and plain.history == res.history
    torch.manual_seed(0)
    byx = train.fit(train.build_model("mlp", 20, 8), train_d, val_d, extra_val=[("held", extra)],
                    early_stop_on="val-data", **kw)
    curve = [r[0] for r in byx.extra_history]
    assert byx.best_epoch == int(np.argmin(curve)) + 1
    with pytest.raises(ValueError):
        train.fit(train.build_model("mlp", 20, 8), train_d, val_d, early_stop_on="val-data", **kw)


def test_cli_end_to_end_with_val_data(tmp_path, state_strings, capsys):  # noqa: F811
    tr, va = tmp_path / "tr", tmp_path / "va"
    _write_games(tr, state_strings, 8, 5, searched=False, prefix="t")
    _write_games(va, state_strings, 3, 5, searched=False, prefix="v", result="loss")
    rc = train.main([
        "--data", f"{tr}@2:w0.5", "--val-data", f"{va}@1", "--states", "truth",
        "--epochs", "2", "--hidden", "4", "--device", "cpu", "--out", str(tmp_path / "m.pev"),
        "--value-target", "search"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "8 records, 16 decisions kept, 16 samples" in out
    assert "3 records, 3 decisions kept, 3 samples" in out
    assert out.count("val-data[") == 2 and f"val-data {va}: BCE vs outcome" in out
    assert (tmp_path / "m.pev").exists()
    assert train.main(["--data", str(tr), "--early-stop-on", "val-data", "--out", "x.pev"]) == 2
