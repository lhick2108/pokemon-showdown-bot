"""Policy priors (PUCT): engine binding, worker/search-config/league plumbing, trainer helpers."""

from __future__ import annotations

import asyncio
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pe = pytest.importorskip("poke_engine")
if not hasattr(pe, "PolicyPrior"):  # engine older than sdb.10
    pytest.skip("needs poke_engine >= 0.0.48+sdb.10 (policy priors)", allow_module_level=True)

from test_engine import make_state  # noqa: E402
from test_search_player import FIXTURES, _battle, _legal, _spy_player  # noqa: E402

from showdown_bot import league  # noqa: E402
from showdown_bot.players import _worker  # noqa: E402
from showdown_bot.players.search import SearchConfig  # noqa: E402
from showdown_bot.sets import RandbatsData  # noqa: E402


def _random_weights(hidden: int = 16, seed: int = 1, bias_slot: int | None = None, bias: float = 0.0):
    rng = random.Random(seed)
    n_in, E, D = pe.FEATURE_LENGTH_V2, pe.N_POLICY_EMB, pe.N_POLICY_AFEAT
    w = dict(
        w1=[rng.uniform(-0.05, 0.05) for _ in range(hidden * n_in)],
        b1=[rng.uniform(0.0, 0.1) for _ in range(hidden)],
        w2=[rng.uniform(-0.3, 0.3) for _ in range(pe.N_POLICY_ACTIONS * hidden)],
        b2=[0.0] * pe.N_POLICY_ACTIONS,
        p=[rng.uniform(-0.2, 0.2) for _ in range(E * hidden)],
        c=[rng.uniform(-0.1, 0.1) for _ in range(E)],
        a=[rng.uniform(-0.5, 0.5) for _ in range(E * D)],
        v=[rng.uniform(-0.5, 0.5) for _ in range(E)],
    )
    if bias_slot is not None:
        w["b2"][bias_slot] = bias
    return w


def _random_net(hidden: int = 16, seed: int = 1, bias_slot: int | None = None, bias: float = 0.0):
    w = _random_weights(hidden, seed, bias_slot, bias)
    return pe.PolicyPrior.mlp(w["w1"], w["b1"], w["w2"], w["b2"], w["p"], w["c"], w["a"], w["v"], 2), w


@pytest.fixture(scope="module")
def ppol(tmp_path_factory):
    path = tmp_path_factory.mktemp("ppol") / "net.ppol"
    net, _ = _random_net()
    net.save(str(path))
    return path


def test_policy_actions_cover_root_options_with_distinct_slots():
    st = make_state()
    acts = pe.policy_actions(st, 1)
    assert {c for c, _ in acts} == {m.move_choice for m in pe.monte_carlo_tree_search(st, 10).side_one}
    slots = [s for _, s in acts]
    assert all(0 <= s < pe.N_POLICY_ACTIONS for s in slots) and len(set(slots)) == len(slots)
    # moves are slots 0-3, their tera variants 4-7
    by_choice = dict(acts)
    assert by_choice["closecombat"] == 0 and by_choice["splash"] == 1
    assert all(by_choice[c] == by_choice[c.replace("-tera", "")] + 4 for c in by_choice if c.endswith("-tera"))


def test_action_features_describe_each_option():
    st = make_state()
    rows = {c: (slot, f) for c, slot, f in pe.policy_action_features(st, 1)}
    assert set(rows) == {c for c, _ in pe.policy_actions(st, 1)}
    assert all(len(f) == pe.N_POLICY_AFEAT for _, f in rows.values())
    cc, splash = rows["closecombat"][1], rows["splash"][1]
    assert cc[0] == 1.0 and cc[3] == 1.0 and splash[3] == 0.0  # is_move, damaging
    assert cc[13] > splash[13] == 0.0  # estimated damage fraction
    assert rows["closecombat-tera"][1][1] == 1.0 and cc[1] == 0.0  # tera flag


def test_prior_file_roundtrip_logits_and_numpy_reference(tmp_path):
    import numpy as np

    hidden = 16
    net, w = _random_net(hidden=hidden, seed=3)
    path = tmp_path / "a.ppol"
    net.save(str(path))
    back = pe.PolicyPrior.load(str(path))
    st = make_state()
    E, D = pe.N_POLICY_EMB, pe.N_POLICY_AFEAT
    for side in (1, 2):
        got = back.option_logits(st, side)
        assert [g[:2] for g in got] == [g[:2] for g in net.option_logits(st, side)]
        assert np.allclose([g[2] for g in got], [g[2] for g in net.option_logits(st, side)])
        x = np.asarray(pe.state_features(st, side, 2), np.float32)
        hr = np.maximum(np.asarray(w["w1"], np.float32).reshape(hidden, -1) @ x + np.asarray(w["b1"]), 0)
        feats = {c: f for c, _s, f in pe.policy_action_features(st, side)}
        for choice, slot, z in got:
            ph = np.asarray(w["p"], np.float32).reshape(E, hidden) @ hr + np.asarray(w["c"])
            e = np.maximum(ph + np.asarray(w["a"], np.float32).reshape(E, D) @ np.asarray(feats[choice]), 0)
            ref = (w["b2"][slot] + np.asarray(w["w2"], np.float32).reshape(-1, hidden)[slot] @ hr
                   + np.asarray(w["v"]) @ e)
            assert z == pytest.approx(float(ref), abs=1e-4), (side, choice)
    assert back.hidden_size == hidden and back.feature_version == 2


def test_prior_rejects_bad_files_and_shapes(tmp_path):
    bad = tmp_path / "bad.ppol"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        pe.PolicyPrior.load(str(bad))
    w = _random_weights(16)
    with pytest.raises(ValueError):
        pe.PolicyPrior.mlp([0.0] * 10, w["b1"], w["w2"], w["b2"], w["p"], w["c"], w["a"], w["v"], 2)
    w12 = _random_weights(16)
    with pytest.raises(ValueError):  # hidden size not supported by the fast path
        pe.PolicyPrior.mlp(
            [0.0] * (12 * pe.FEATURE_LENGTH_V2), [0.0] * 12, [0.0] * 13 * 12, w12["b2"], [0.0] * 16 * 12,
            w12["c"], w12["a"], w12["v"], 2,
        )


def test_priors_is_a_distribution_over_the_legal_options():
    net, _ = _random_net()
    st = make_state()
    pri = net.priors(st, 1)
    assert [c for c, _ in pri] == [c for c, _ in pe.policy_actions(st, 1)]
    assert sum(p for _, p in pri) == pytest.approx(1.0, abs=1e-5)


def test_default_search_unchanged_and_reports_depth():
    st = make_state()
    res = pe.monte_carlo_tree_search(st, 50, iterations=2000)
    assert all(m.visits > 0 for m in res.side_one + res.side_two)  # UCB1 tries every option
    assert res.total_visits == 2000
    assert res.depth_max >= 1 and 1.0 <= res.depth_mean <= res.depth_max


def test_prior_shifts_root_visits_toward_the_prior():
    st = make_state()
    base = pe.monte_carlo_tree_search(st, 50, iterations=3000)
    options = [m.move_choice for m in base.side_one]
    target = "splash"
    net, _ = _random_net(hidden=16, seed=5, bias_slot=dict(pe.policy_actions(st, 1))[target], bias=10.0)
    res = pe.monte_carlo_tree_search(st, 50, iterations=3000, prior=net, prior_opp=net, c_puct=2.0)
    share = {m.move_choice: m.visits / res.total_visits for m in res.side_one}
    base_share = {m.move_choice: m.visits / base.total_visits for m in base.side_one}
    assert max(share, key=share.get) == target or share[target] > base_share[target]
    assert share[target] > base_share[target] + 0.05, (share, base_share)
    assert set(share) == set(options)


def test_root_prior_override_and_validation():
    st = make_state()
    opts = [m.move_choice for m in pe.monte_carlo_tree_search(st, 10).side_one]
    res = pe.monte_carlo_tree_search(
        st, 50, iterations=1500, root_prior_s1={opts[-1]: 1.0}, c_puct=3.0
    )
    visits = {m.move_choice: m.visits for m in res.side_one}
    assert max(visits, key=visits.get) == opts[-1]
    with pytest.raises(ValueError):
        pe.monte_carlo_tree_search(st, 50, iterations=100, threads=2, root_prior_s1={opts[0]: 1.0})
    with pytest.raises(ValueError):
        pe.monte_carlo_tree_search(st, 50, iterations=100, selection="exp3", root_prior_s1={opts[0]: 1.0})
    with pytest.raises(ValueError):
        pe.monte_carlo_tree_search(st, 50, iterations=100, root_prior_s1={opts[0]: 1.0}, c_puct=0.0)


def test_worker_prior_path_keeps_result_shape(ppol):
    s = str(make_state().to_string())
    plain = _worker.search_world(s, 20, 1)
    withp = _worker.search_world(s, 20, 1, prior_path=str(ppol))
    assert len(plain) == len(withp) == 2
    assert [o[0] for o in plain[1]] == [o[0] for o in withp[1]]
    none_opp = _worker.search_world(s, 20, 1, prior_path=str(ppol), prior_opp="none")
    assert len(none_opp) == 2
    other = _worker.search_world(s, 20, 1, prior_path=str(ppol), prior_opp=str(ppol), c_puct=0.5)
    assert len(other) == 2
    with pytest.raises(RuntimeError):
        _worker.search_world(s, 20, 2, prior_path=str(ppol))  # shared-tree search has no priors
    # the net is cached per process
    assert _worker.get_prior(str(ppol)) is _worker.get_prior(str(ppol))


def test_worker_human_root_prior(ppol):
    human = Path(__file__).resolve().parent.parent / "models" / "human-policy.npz"
    if not human.is_file():
        main = Path(__file__).resolve()
        for parent in main.parents:
            if (parent / "models" / "human-policy.npz").is_file():
                human = parent / "models" / "human-policy.npz"
                break
    if not human.is_file():
        pytest.skip("models/human-policy.npz not available")
    s = str(make_state().to_string())
    out = _worker.search_world(s, 20, 1, prior_path=str(ppol), prior_opp="human", human_path=str(human))
    assert len(out) == 2 and out[0] > 0
    with pytest.raises(RuntimeError):
        _worker.search_world(s, 20, 1, prior_path=str(ppol), prior_opp="human")


# -- SearchConfig / SearchPlayer ----------------------------------------------------------


def test_config_defaults_are_off_and_env_overrides(monkeypatch, ppol):
    cfg = SearchConfig()
    assert cfg.prior is None and cfg.prior_opp == "self" and cfg.c_puct == 1.0
    monkeypatch.setenv("SHOWDOWN_SEARCH_PRIOR", str(ppol))
    monkeypatch.setenv("SHOWDOWN_SEARCH_PRIOR_OPP", "none")
    monkeypatch.setenv("SHOWDOWN_SEARCH_C_PUCT", "0.7")
    cfg = SearchConfig.from_env()
    assert (cfg.prior, cfg.prior_opp, cfg.c_puct) == (str(ppol), "none", 0.7)


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def test_search_player_validates_prior_settings(data, ppol, tmp_path):
    def build(**cfg):
        p, _calls, pool = _spy_player(data, config=SearchConfig(**cfg))
        pool.shutdown()
        return p

    p = build(prior=str(ppol), search_time_ms=20)
    assert p.config.prior == str(ppol.resolve()) and p.config.prior_opp == "self"
    with pytest.raises(FileNotFoundError):
        build(prior=str(tmp_path / "missing.ppol"))
    with pytest.raises(ValueError):
        build(prior=str(ppol), selection="exp3")
    with pytest.raises(ValueError):
        build(prior=str(ppol), threads=2)
    with pytest.raises(ValueError):
        build(prior=str(ppol), c_puct=0.0)
    with pytest.raises(ValueError):
        build(prior=str(ppol), prior_opp="human")  # no human model configured
    with pytest.raises(FileNotFoundError):
        build(prior=str(ppol), prior_opp=str(tmp_path / "nope.ppol"))


def test_prior_reaches_worker_only_when_enabled(data, ppol):
    p, calls, pool = _spy_player(data, config=SearchConfig(search_time_ms=20))
    asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert calls and p._prior_kwargs() == {}  # off: the worker is called exactly as before

    p, calls, pool = _spy_player(data, config=SearchConfig(search_time_ms=20, prior=str(ppol), c_puct=0.8))
    order = asyncio.run(p.choose_move(_battle()))
    pool.shutdown()
    assert order.message in _legal(_battle())
    assert p.stats.fallbacks == 0 and p.stats.world_errors == 0 and p.stats.worlds_run > 0
    assert p._prior_kwargs()["c_puct"] == 0.8 and p._prior_kwargs()["prior_path"] == str(ppol.resolve())


def test_node_export_and_prior_in_one_search(ppol):
    st = make_state()
    net, _ = _random_net()
    both = pe.monte_carlo_tree_search(
        st, 50, iterations=4000, prior=net, prior_opp=net, c_puct=1.0, export_nodes=20, export_min_visits=8
    )
    assert both.nodes and len(both.nodes) <= 20
    assert all(len(n) == 5 and n[1] >= 8 for n in both.nodes)
    assert math.isfinite(both.root_search_value) and both.depth_max >= 1
    assert both.total_visits == 4000
    only_export = pe.monte_carlo_tree_search(st, 50, iterations=2000, export_nodes=5, export_min_visits=8)
    assert only_export.nodes and only_export.depth_max >= 1
    only_prior = pe.monte_carlo_tree_search(st, 50, iterations=1000, prior=net)
    assert only_prior.nodes is None
    out = _worker.search_world(
        str(st.to_string()), 20, 1, prior_path=str(ppol), export_nodes=10, export_min_visits=4
    )
    assert out.nodes and len(out) == 2


# -- league spec keys -----------------------------------------------------------------------


def test_league_spec_prior_keys(ppol):
    spec = league.parse_spec(f"pr=search-net:evaluator=m.pev:prior={ppol}:prior_opp=NONE:c_puct=0.5:fpu=0.05:prior_eps=0.1")
    assert spec.param("prior") == str(ppol) and spec.param("prior_opp") == "none"
    fields = league.search_config_fields(spec)
    assert fields["prior"] == str(ppol) and fields["prior_opp"] == "none"
    assert (fields["c_puct"], fields["prior_fpu"], fields["prior_eps"]) == (0.5, 0.05, 0.1)
    assert "prior" not in league.search_config_fields(league.parse_spec("search:time_ms=20"))
    for bad in ("search:c_puct=1", "search:prior=a.ppol:c_puct=0", "search:prior=a.ppol:prior_eps=1",
                "search:prior_opp=self"):
        with pytest.raises(ValueError):
            league.parse_spec(bad)


# -- trainer helpers ----------------------------------------------------------------------


def test_trainer_metrics_and_split():
    import numpy as np

    from showdown_bot import prior_train as pt

    logits = np.zeros((2, pt.N_ACTIONS), np.float32)
    logits[0, 1] = 5.0
    v = np.zeros_like(logits)
    v[0, 1], v[0, 2] = 90, 10
    v[1, 0], v[1, 3] = 50, 50
    m = np.zeros_like(logits, dtype=bool)
    m[:, :4] = True
    out = pt.metrics(logits, v, m)
    assert out["n"] == 2 and out["top1"] == pytest.approx(1.0)  # row 1: the tie goes to slot 0 on both sides
    assert out["ce_uniform"] == pytest.approx(math.log(4), abs=1e-6)
    assert out["ce"] < out["ce_uniform"]
    tags = [f"battle-{i}" for i in range(2000)]
    val = [t for t in tags if pt.is_val(t, 0.1, 0)]
    assert 120 < len(val) < 280 and val == [t for t in tags if pt.is_val(t, 0.1, 0)]
