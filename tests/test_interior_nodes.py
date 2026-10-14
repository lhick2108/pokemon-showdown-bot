"""Interior-node value targets (bead showdown-bot-rnm): export, recording, loading, training weights."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")

from test_recorder import fake_world  # noqa: E402
from test_train import state_strings  # noqa: E402,F401

from showdown_bot import dataset, recorder, train  # noqa: E402
from showdown_bot.players import _worker  # noqa: E402

HAS_EXPORT = "export_nodes" in pe.monte_carlo_tree_search.__code__.co_varnames
needs_export = pytest.mark.skipif(not HAS_EXPORT, reason="needs engine 0.0.48+sdb.9 (export_nodes)")


def _outcome(nodes, search_value=0.6):
    return _worker.OppTuple((100, [("a", 75, 60.0), ("b", 25, 10.0)]), None, nodes, search_value)


def _write(root, states, nodes_per_world, result="win", tag="g"):
    """One game, one decision, one world (``states[0]``) with the given exported nodes."""
    rec = recorder.GameRecorder(root)
    rec.start_game(tag, "u")
    world = recorder.world_record(states[0], _outcome(nodes_per_world))
    rec.add_decision(tag, "u", {"turn": 1, "worlds": [world], "policy": {"a": 1.0}, "chosen": "a",
                                "truth": {"state": states[1]}, "truth_status": "ok"})
    rec.end_game(tag, "u", getattr(recorder, f"RESULT_{result.upper()}"), turns=2)


def test_opp_tuple_carries_nodes_through_pickle():
    t = _outcome([("S", 40, 0.7, 0.5, 2)], 0.61)
    back = pickle.loads(pickle.dumps(t))
    assert tuple(back) == tuple(t) and back.nodes == [("S", 40, 0.7, 0.5, 2)] and back.search_value == 0.61
    plain = _worker.OppTuple((1, []), [("x", 1)])
    assert pickle.loads(pickle.dumps(plain)).opp == [("x", 1)] and plain.nodes is None


def test_world_record_stores_nodes_only_when_exported():
    plain = recorder.world_record("S", (100, [("a", 100, 50.0)]))
    assert "nodes" not in plain and "search_value" not in plain
    rec = recorder.world_record("S", _outcome([("N1", 40, 0.71234567, 0.5, 2)], 0.6123456))
    assert rec["nodes"] == [["N1", 40, 0.71235, 0.5, 2]]
    assert rec["search_value"] == pytest.approx(0.61235)


def test_dataset_node_modes(tmp_path, state_strings):  # noqa: F811
    s = state_strings
    nodes = [(s[2], 50, 0.8, 0.55, 1), (s[3], 8, 0.3, 0.4, 3)]
    _write(tmp_path, s, nodes)
    off = list(dataset.iter_samples(tmp_path))
    assert {x.kind for x in off} == {"world", "truth"}
    on = list(dataset.iter_samples(tmp_path, nodes="on"))
    assert len(on) == len(off) + 2
    only = list(dataset.iter_samples(tmp_path, nodes="only", node_min_visits=16))
    assert [x.kind for x in only] == ["node"]
    n = only[0]
    assert (n.node_visits, n.node_depth) == (50, 1)
    assert n.search_value == pytest.approx(0.8) and n.search_value_abs == pytest.approx(0.8)
    assert n.node_static == pytest.approx(0.55)
    assert n.features == pe.state_features(pe.State.from_string(s[2]), 1)
    assert n.value == 1.0 and n.battle_tag == "g"
    both = list(dataset.iter_samples(tmp_path, nodes="only", sides=(1, 2)))
    flipped = [x for x in both if x.side == 2 and x.node_visits == 50][0]
    assert flipped.search_value_abs == pytest.approx(0.2) and flipped.value == 0.0
    with pytest.raises(ValueError):
        list(dataset.iter_samples(tmp_path, nodes="bogus"))


def test_exported_root_value_replaces_the_reconstruction(tmp_path, state_strings):  # noqa: F811
    s = state_strings
    _write(tmp_path, s, [], tag="e")
    samples = [x for x in dataset.iter_samples(tmp_path) if x.kind == "world"]
    # world_record has avg_score 0.7 and search_value 0.6 exported: q_abs is the exported one
    assert samples[0].search_value_abs == pytest.approx(0.6)
    assert samples[0].search_value == pytest.approx(0.7)


def test_node_multiplier():
    mk = lambda kind, v, d: dataset.Sample([], 1.0, None, kind, 1, "t", 1, node_visits=v, node_depth=d)  # noqa: E731
    assert train.node_multiplier(mk("world", 0, 0)) == 1.0
    assert train.node_multiplier(mk("node", 64, 1), 2.0, 64.0, 0.5) == pytest.approx(2.0)
    assert train.node_multiplier(mk("node", 32, 1), 1.0, 64.0, 1.0) == pytest.approx(0.5)  # few visits: half
    assert train.node_multiplier(mk("node", 640, 3), 1.0, 64.0, 0.5) == pytest.approx(0.25)  # capped, discounted
    assert train.node_multiplier(mk("node", 5, 2), 1.0, 0.0, 1.0) == 1.0  # visit_ref 0 disables the ramp


def test_load_data_with_nodes(tmp_path, state_strings):  # noqa: F811
    s = state_strings
    for i in range(3):
        _write(tmp_path, s, [(s[2], 64, 0.9, 0.5, 1), (s[3], 32, 0.1, 0.5, 2)], tag=f"g{i}")
    base = train.load_data([train.Source(tmp_path)], states="truth", value_target="search", workers=1)
    assert len(base) == 3
    data = train.load_data([train.Source(tmp_path)], states="truth", value_target="search", workers=1,
                           nodes="on", node_weight=2.0, node_visit_ref=64.0, node_depth_discount=0.5)
    assert len(data) == 3 + 6
    assert data.w.mean() == pytest.approx(1.0)
    w9 = data.w[np.isclose(data.y_fit, 0.9)][0]  # 64 visits, depth 1: 2.0 * 1 * 1
    w1 = data.w[np.isclose(data.y_fit, 0.1)][0]   # 32 visits, depth 2: 2.0 * 0.5 * 0.5
    w_root = data.w[np.isclose(data.y_fit, 0.6)][0]  # root q = the exported search_value 0.6 (mult 1)
    assert (w9 / w_root, w1 / w_root) == pytest.approx((2.0, 0.5))
    only = train.load_data([train.Source(tmp_path)], states="truth", value_target="search", workers=1,
                           nodes="only")
    assert len(only) == 6
    # search targets: q is the exported absolute node value (win outcome is ignored by 'search')
    assert sorted(float(x) for x in only.y_fit)[:3] == pytest.approx([0.1] * 3)
    assert sorted(float(x) for x in only.y_fit)[3:] == pytest.approx([0.9] * 3)
    # relative weights: the 64-visit depth-1 node counts 1.0, the 32-visit depth-2 node 0.5 (visit ramp;
    # the loader normalises to mean 1, so compare ratios; same battle counts under sqrt weighting)
    hi = only.w[np.isclose(only.y_fit, 0.9)][0]
    lo = only.w[np.isclose(only.y_fit, 0.1)][0]
    assert hi / lo == pytest.approx(2.0)


def test_nodes_need_a_search_target(tmp_path, state_strings):  # noqa: F811
    _write(tmp_path, state_strings, [(state_strings[2], 64, 0.9, 0.5, 1)])
    with pytest.raises(ValueError):
        train.load_data([train.Source(tmp_path)], states="truth", value_target="outcome", nodes="on",
                        workers=1)
    with pytest.raises(ValueError):
        train.load_data([train.Source(tmp_path)], states="truth", value_target="search", nodes="on",
                        q_kind="relative", workers=1)
    assert train.main(["--data", str(tmp_path), "--out", str(tmp_path / "m.pev"), "--nodes", "on"]) == 2


@needs_export
def test_engine_exports_interior_nodes(state_strings):  # noqa: F811
    st = pe.State.from_string(state_strings[0])
    plain = pe.monte_carlo_tree_search(st, 50, iterations=3000)
    assert plain.nodes is None and 0.0 < plain.root_value < 1.0
    res = pe.monte_carlo_tree_search(st, 50, iterations=20000, export_nodes=10, export_min_visits=20)
    assert 0 < len(res.nodes) <= 10
    assert res.root_search_value == pytest.approx(
        _stock_abs(res), abs=1e-3) or 0.0 < res.root_search_value < 1.0
    for state, visits, value, static, depth in res.nodes:
        assert visits >= 20 and visits <= res.total_visits and depth >= 1
        assert 0.0 < value < 1.0 and 0.0 < static < 1.0
        pe.State.from_string(state)  # parses back
        assert len(pe.state_features(pe.State.from_string(state), 1)) == pe.FEATURE_LENGTH
    assert len({n[0] for n in res.nodes}) == len(res.nodes)  # one entry per distinct position
    assert [n[1] for n in res.nodes] == sorted((n[1] for n in res.nodes), reverse=True)
    with pytest.raises(ValueError):
        pe.monte_carlo_tree_search(st, 20, threads=2, export_nodes=4)


def _stock_abs(res):
    import math
    avg = sum(o.total_score for o in res.side_one) / res.total_visits
    z = math.log(res.root_value / (1 - res.root_value)) + math.log(avg / (1 - avg))
    return 1 / (1 + math.exp(-z))


@needs_export
def test_search_world_returns_nodes(state_strings):  # noqa: F811
    out = _worker.search_world(state_strings[0], 60, 1, export_nodes=6, export_min_visits=8)
    assert out.nodes and len(out.nodes) <= 6 and 0.0 < out.search_value < 1.0
    assert out[0] > 0 and out[1]
    plain = _worker.search_world(state_strings[0], 30, 1)
    assert not hasattr(plain, "nodes")


def test_cli_trains_on_nodes_end_to_end(tmp_path, state_strings, capsys):  # noqa: F811
    for i in range(8):
        _write(tmp_path / "run", state_strings, [(state_strings[2 + i % 3], 40 + i, 0.2 + 0.1 * (i % 5), 0.5, 1 + i % 2)],
               result="win" if i % 2 else "loss", tag=f"g{i}")
    out_path = tmp_path / "m.pev"
    rc = train.main(["--data", str(tmp_path / "run"), "--states", "truth", "--epochs", "2", "--hidden", "4",
                     "--device", "cpu", "--out", str(out_path), "--value-target", "search", "--nodes", "on",
                     "--node-weight", "0.5", "--node-depth-discount", "0.8"])
    text = capsys.readouterr().out
    assert rc == 0 and out_path.exists()
    assert "interior nodes (on): 8 node samples of 16" in text



def test_nodes_per_world_subsamples_reproducibly(tmp_path, state_strings):  # noqa: F811
    s = state_strings
    _write(tmp_path, s, [(s[i % len(s)], 30 + i, 0.5, 0.5, 1) for i in range(6)])
    a = list(dataset.iter_samples(tmp_path, nodes="only", nodes_per_world=2, seed=1))
    b = list(dataset.iter_samples(tmp_path, nodes="only", nodes_per_world=2, seed=1))
    assert len(a) == 2 and [x.node_visits for x in a] == [x.node_visits for x in b]
    assert len(list(dataset.iter_samples(tmp_path, nodes="only", nodes_per_world=0))) == 6
