"""Absolute search values of recordings made with a learned evaluator (bead: q-anchor)."""

from __future__ import annotations

import math
import random

import pytest

pe = pytest.importorskip("poke_engine")

from test_train import state_strings  # noqa: E402,F401

from showdown_bot import dataset, recorder  # noqa: E402
from showdown_bot.players import _worker  # noqa: E402

HAS_VALUE = hasattr(pe.MctsResult, "root_search_value") if hasattr(pe, "MctsResult") else True


def _mlp(stock_coef=0.0125, leaf_scale=1.5, relative=True, seed=0, hidden=4):
    rng = random.Random(seed)
    n = pe.FEATURE_LENGTH
    return pe.Evaluator.mlp(
        [rng.uniform(-0.05, 0.05) for _ in range(hidden * n)],
        [rng.uniform(-0.1, 0.1) for _ in range(hidden)],
        [rng.uniform(-0.5, 0.5) for _ in range(hidden)],
        0.1, relative_to_root=relative, leaf_scale=leaf_scale, stock_coef=stock_coef,
    )


def _reconstruct(res, ev, state, scale):
    avg = sum(o.total_score for o in res.side_one) / res.total_visits
    return dataset.learned_absolute_search_value(ev, state, avg, scale)


@pytest.mark.parametrize("stock_coef,leaf_scale", [(0.0125, 1.5), (0.0, 2.0), (0.02, 1.0)])
def test_reconstruction_matches_engine_root_search_value(state_strings, stock_coef, leaf_scale):  # noqa: F811
    ev = _mlp(stock_coef=stock_coef, leaf_scale=leaf_scale)
    st = pe.State.from_string(state_strings[0])
    res = pe.monte_carlo_tree_search(st, 50, iterations=1500, evaluator=ev)
    assert math.isfinite(res.root_search_value)
    assert _reconstruct(res, ev, st, leaf_scale) == pytest.approx(res.root_search_value, abs=2e-4)


def test_absolute_evaluator_ignores_the_root_logit(state_strings):  # noqa: F811
    ev = _mlp(relative=False, leaf_scale=2.0)
    st = pe.State.from_string(state_strings[0])
    res = pe.monte_carlo_tree_search(st, 50, iterations=1000, evaluator=ev)
    assert _reconstruct(res, ev, st, 2.0) == pytest.approx(res.root_search_value, abs=2e-4)


def _write_game(root, state, header_config, tag="g", exported=None):
    rec = recorder.GameRecorder(root)
    rec.start_game(tag, "u", config=header_config)
    world = recorder.world_record(state, (100, [("a", 100, 70.0)]))
    if exported is not None:
        world["search_value"] = exported
    rec.add_decision(tag, "u", {"turn": 1, "worlds": [world], "policy": {"a": 1.0}, "chosen": "a",
                                "truth": {"state": state}, "truth_status": "ok"})
    rec.end_game(tag, "u", recorder.RESULT_WIN, turns=2)


def _abs(root, **kw):
    return {s.kind: s.search_value_abs for s in dataset.iter_samples(root, **kw)}


def test_loader_uses_header_evaluator_and_leaf_scale(tmp_path, state_strings):  # noqa: F811
    ev_path = tmp_path / "m.pev"
    ev = _mlp()
    ev.save(str(ev_path))
    _write_game(tmp_path, state_strings[0], {"evaluator": str(ev_path), "evaluator_scale": 1.5})
    st = pe.State.from_string(state_strings[0])
    expect = dataset.learned_absolute_search_value(ev, st, 0.7, 1.5)
    got = _abs(tmp_path)
    assert got["world"] == pytest.approx(expect, abs=1e-6) and got["truth"] == pytest.approx(expect, abs=1e-6)
    stock = dataset.absolute_search_value(pe.evaluate(st), 0.7)
    assert abs(expect - stock) > 1e-3  # the stock anchor really differs
    # overrides: scale, then forcing the stock anchor
    sc = _abs(tmp_path, q_leaf_scale=3.0)["world"]
    assert sc == pytest.approx(dataset.learned_absolute_search_value(ev, st, 0.7, 3.0), abs=1e-6)
    assert _abs(tmp_path, q_evaluator="stock")["world"] == pytest.approx(stock, abs=1e-9)


def test_exported_value_wins_unless_disabled(tmp_path, state_strings):  # noqa: F811
    ev_path = tmp_path / "m.pev"
    _mlp().save(str(ev_path))
    _write_game(tmp_path, state_strings[0], {"evaluator": str(ev_path), "evaluator_scale": 1.5},
                exported=0.123)
    assert _abs(tmp_path)["world"] == pytest.approx(0.123)
    assert _abs(tmp_path, use_exported_q=False)["world"] != pytest.approx(0.123, abs=1e-3)


def test_stock_header_keeps_stock_fallback(tmp_path, state_strings):  # noqa: F811
    _write_game(tmp_path, state_strings[0], {"evaluator": None})
    st = pe.State.from_string(state_strings[0])
    assert _abs(tmp_path)["world"] == pytest.approx(
        dataset.absolute_search_value(pe.evaluate(st), 0.7), abs=1e-9)


def test_unloadable_or_ambiguous_evaluator_fails_loudly(tmp_path, state_strings):  # noqa: F811
    _write_game(tmp_path, state_strings[0], {"evaluator": str(tmp_path / "missing.pev")})
    with pytest.raises(RuntimeError, match="--q-evaluator"):
        list(dataset.iter_samples(tmp_path))
    other = tmp_path / "ens"
    _write_game(other, state_strings[0], {"evaluator": "a.pev", "evaluator2": "b.pev"})
    with pytest.raises(ValueError, match="ensemble"):
        list(dataset.iter_samples(other))


def test_recording_anchor():
    assert dataset.recording_anchor({}) == (None, 1.0)
    assert dataset.recording_anchor({"config": {"evaluator": "x.pev", "evaluator_scale": 1.5}}) == ("x.pev", 1.5)
    assert dataset.recording_anchor({"config": {"evaluator": "x.pev"}}, q_leaf_scale=2.0) == ("x.pev", 2.0)
    assert dataset.recording_anchor({"config": {"evaluator": "x.pev"}}, "stock") == (None, 1.0)
    assert dataset.recording_anchor({}, "y.pev", 2.5) == ("y.pev", 2.5)


def test_worker_records_root_value_without_node_export(state_strings):  # noqa: F811
    out = _worker.search_world(state_strings[0], 30, 1, record_value=True)
    assert isinstance(out, _worker.OppTuple) and out.nodes is None
    assert 0.0 < out.search_value < 1.0
    rec = recorder.world_record(state_strings[0], out)
    assert rec["search_value"] == pytest.approx(out.search_value, abs=1e-5) and "nodes" not in rec
    plain = _worker.search_world(state_strings[0], 30, 1)
    assert not isinstance(plain, _worker.OppTuple)
    assert "search_value" not in recorder.world_record(state_strings[0], plain)
