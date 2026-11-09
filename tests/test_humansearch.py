"""HumanSearchPlayer: policy blending, spec parsing and the final-move choice."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")

from showdown_bot import human_policy as hp  # noqa: E402
from showdown_bot import league  # noqa: E402
from showdown_bot.players.humansearch import HumanSearchPlayer, blend_policies  # noqa: E402
from showdown_bot.players.search import SearchConfig  # noqa: E402
from showdown_bot.sets import RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _model_file(tmp_path):
    path = tmp_path / "human.npz"
    rng = np.random.default_rng(0)
    h = 8
    hp.HumanPolicy({
        "w1": rng.normal(0, 0.01, (hp.N_FEATURES, h)), "b1": np.zeros(h), "wm": np.zeros((hp.N_SLOTS, h)),
        "elo": np.zeros((hp.N_ELO + 1, h)), "w2": rng.normal(0, 0.1, (h, h)), "b2": np.zeros(h),
        "w3": rng.normal(0, 0.1, (h, hp.N_SLOTS)), "b3": np.zeros(hp.N_SLOTS),
    }).save(path)
    return path


def test_blend_endpoints_and_normalisation():
    s = {"a": 0.7, "b": 0.2, "c": 0.1}
    h = {"a": 0.1, "b": 0.1, "c": 0.8}
    for blend in ("geo", "mix"):
        p0 = blend_policies(s, h, 0.0, blend=blend)
        p1 = blend_policies(s, h, 1.0, blend=blend)
        assert p0 == pytest.approx(s) and p1 == pytest.approx(h)
        mid = blend_policies(s, h, 0.5, blend=blend)
        assert sum(mid.values()) == pytest.approx(1.0)
    geo = blend_policies(s, h, 0.5)
    expect = np.sqrt([0.07, 0.02, 0.08])
    assert list(geo.values()) == pytest.approx(list(expect / expect.sum()))
    mix = blend_policies(s, h, 0.25, blend="mix")
    assert mix["a"] == pytest.approx(0.75 * 0.7 + 0.25 * 0.1)


def test_blend_moves_the_top_choice_with_weight():
    s = {"a": 0.8, "b": 0.2}
    h = {"a": 0.1, "b": 0.9}
    tops = [max((p := blend_policies(s, h, w)), key=p.get) for w in (0.0, 0.3, 0.7, 1.0)]
    assert tops[0] == "a" and tops[-1] == "b"


def test_blend_floors_zeros_and_missing_choices():
    s = {"a": 1.0, "b": 0.0}
    p = blend_policies(s, {"a": 0.5}, 0.5)  # b is zero for search, missing for the human
    assert p["b"] > 0 and sum(p.values()) == pytest.approx(1.0)
    assert blend_policies({}, {"a": 1.0}, 0.5) == {}
    assert blend_policies({"a": 0.0, "b": 0.0}, {}, 0.5, blend="mix") == pytest.approx({"a": 0.5, "b": 0.5})


def test_temp_sharpens_the_human_side():
    s = {"a": 0.5, "b": 0.5}
    h = {"a": 0.6, "b": 0.4}
    flat = blend_policies(s, h, 1.0)["a"]
    sharp = blend_policies(s, h, 1.0, temp=0.5)["a"]
    assert sharp > flat


def test_spec_parsing():
    spec = league.parse_spec(
        "hs=humansearch:evaluator=a.pev:scale=2:time_ms=1000:worlds=4:model=C:\\m\\h.npz"
        ":w=0.4:pick=ARGMAX:blend=mix:temp=0.8:elo=1500"
    )
    assert spec.type == "humansearch" and spec.type in league.SEARCH_BOTS
    assert spec.param("w") == 0.4 and spec.param("pick") == "argmax" and spec.param("blend") == "mix"
    assert spec.param("model") == "C:\\m\\h.npz" and spec.param("worlds") == 4
    for bad in ("hs=humansearch:evaluator=a.pev",  # model required
                "hs=humansearch:model=m.npz:w=1.5", "hs=humansearch:model=m.npz:w=-0.1",
                "hs=humansearch:model=m.npz:blend=avg", "hs=humansearch:model=m.npz:pick=roll",
                "hs=search:model=m.npz", "hs=search:w=0.5"):
        with pytest.raises(ValueError):
            league.parse_spec(bad)


def _player(data, path, **kw):
    return HumanSearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20, mode="argmax"), model=path, seed=5, **kw,
    )


class _Built:
    def to_string(self):
        return "state"

    def choice_to_order(self, choice, battle):
        return SimpleNamespace(message=choice)


def test_final_move_follows_blend_and_logs_changes(data, tmp_path):
    path = _model_file(tmp_path)
    battle = SimpleNamespace(battle_tag="b")
    policy = {"a": 0.8, "b": 0.2}
    stub = SimpleNamespace(predict=lambda state, choices, elo: {"a": 0.05, "b": 0.95})
    p = _player(data, path, human_weight=1.0, pick="argmax")
    p.human = stub
    assert p._order_from_policy(policy, _Built(), battle).message == "b"
    assert p.blend_decisions == 1 and p.blend_changed == 1 and p.blend_change_rate == 1.0
    p0 = _player(data, path, human_weight=0.0, pick="argmax")
    p0.human = stub
    assert p0._order_from_policy(policy, _Built(), battle).message == "a"
    assert p0.blend_decisions == 0  # w=0 is plain search
    ps = _player(data, path, human_weight=0.5, pick="sample")
    ps.human = stub
    seen = {ps._order_from_policy(policy, _Built(), battle).message for _ in range(60)}
    assert seen == {"a", "b"} and ps.blend_decisions == 60


def test_model_failure_falls_back_to_search(data, tmp_path):
    p = _player(data, _model_file(tmp_path), human_weight=1.0, pick="argmax")

    def boom(*a):
        raise RuntimeError("x")

    p.human = SimpleNamespace(predict=boom)
    assert p._order_from_policy({"a": 0.9, "b": 0.1}, _Built(), SimpleNamespace(battle_tag="b")).message == "a"


def test_validation(data, tmp_path):
    path = _model_file(tmp_path)
    for kw in ({"human_weight": 1.5}, {"pick": "roll"}, {"blend": "x"}, {"temp": 0}):
        with pytest.raises(ValueError):
            _player(data, path, **kw)
    with pytest.raises(ValueError):
        HumanSearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50, data=data)
