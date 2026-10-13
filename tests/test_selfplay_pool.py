"""Opponent pool parsing/sampling, playout cap randomization, skipped-decision datasets."""

from __future__ import annotations

import asyncio
import importlib.util
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import dataset, engine_state, recorder  # noqa: E402
from showdown_bot.players.search import SearchConfig, SearchPlayer  # noqa: E402
from showdown_bot.sets import OpponentModel, RandbatsData  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
_SPEC = importlib.util.spec_from_file_location("selfplay_script", ROOT / "scripts" / "selfplay.py")
selfplay = importlib.util.module_from_spec(_SPEC)
sys.modules["selfplay_script"] = selfplay
_SPEC.loader.exec_module(selfplay)


def test_parse_pool_specs():
    entries = selfplay.parse_pool("stock, v1=C:\\nets\\a.pev:scale=1.5,v2=b.pev")
    assert [e.label for e in entries] == ["stock", "v1", "v2"]
    assert entries[0].evaluator is None
    assert (entries[1].evaluator, entries[1].scale) == ("C:\\nets\\a.pev", 1.5)
    assert (entries[2].evaluator, entries[2].scale) == ("b.pev", 1.0)


@pytest.mark.parametrize(
    "bad", ["", "foo", "a=", "=x.pev", "a=x.pev:scale=zz", "a=x.pev:scale=0", "stock,stock"]
)
def test_parse_pool_rejects_bad_specs(bad):
    with pytest.raises(ValueError):
        selfplay.parse_pool(bad)


def test_pool_sampling_follows_weights_and_seed():
    entries = selfplay.parse_pool("stock,a=a.pev,b=b.pev")
    uniform = selfplay.parse_weights(None, 3)
    rng = random.Random(0)
    draws = [selfplay.sample_pool(entries, uniform, rng).label for _ in range(3000)]
    for label in ("stock", "a", "b"):
        assert abs(draws.count(label) / 3000 - 1 / 3) < 0.05
    weights = selfplay.parse_weights("1,0,3", 3)
    rng = random.Random(1)
    draws = [selfplay.sample_pool(entries, weights, rng).label for _ in range(2000)]
    assert "a" not in draws and abs(draws.count("b") / 2000 - 0.75) < 0.05

    def seq():
        r = random.Random(5)
        return [selfplay.sample_pool(entries, weights, r).label for _ in range(8)]

    assert seq() == seq()
    with pytest.raises(ValueError):
        selfplay.parse_weights("1,2", 3)


def _player(pool, rec, **cfg):
    data = RandbatsData.load(cache_dir=FIXTURES)
    return SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20, **cfg), executor=pool, seed=3, recorder=rec,
    )


def test_full_frac_rate_seed_and_default_off():
    with ThreadPoolExecutor(max_workers=1) as pool:
        p = _player(pool, None, full_frac=0.3)
        hits = sum(p._draw_full() for _ in range(4000))
        assert abs(hits / 4000 - 0.3) < 0.03
        a = [_player(pool, None, full_frac=0.3)._draw_full() for _ in range(1)]
        b = [_player(pool, None, full_frac=0.3)._draw_full() for _ in range(1)]
        assert a == b
        assert all(_player(pool, None)._draw_full() for _ in range(10))
        with pytest.raises(ValueError):
            _player(pool, None, full_frac=1.5)


def _battle():
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


@pytest.mark.parametrize("frac,recorded", [(0.0, 0), (1.0, 1)])
def test_fast_decisions_are_not_recorded(tmp_path, frac, recorded):
    rec = recorder.GameRecorder(tmp_path / "run")
    with ThreadPoolExecutor(max_workers=2) as pool:
        p = _player(pool, rec, full_frac=frac, fast_time_ms=5, label="x", opponent="y")
        battle = _battle()
        order = asyncio.run(p.choose_move(battle))
        battle.won_by("bot")
        p._battle_finished_callback(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert p.stats.searched == 1  # fast decisions still search
    files = list(recorder.iter_game_files(tmp_path / "run"))
    assert len(files) == recorded
    if recorded:
        header = recorder.read_game(files[0])["header"]
        assert header["config"]["label"] == "x" and header["config"]["opponent"] == "y"


def _state():
    data = RandbatsData.load(cache_dir=FIXTURES)
    battle = _battle()
    trackers = OpponentModel(data).update(battle)
    slots = engine_state.sample_opponent_team(
        battle, trackers, engine_state.TeamSampler(data), random.Random(0)
    )
    return engine_state.build_state(battle, slots).to_string()


def test_dataset_loads_record_with_skipped_decisions(tmp_path):
    state = _state()
    rec = recorder.GameRecorder(tmp_path)
    rec.start_game("g", "u")
    for turn, q in ((2, 0.4), (7, 0.7), (11, 0.9)):  # turns in between were fast/unrecorded
        rec.add_decision("g", "u", {
            "turn": turn,
            "worlds": [recorder.world_record(state, (100, [("a", 100, q * 100)]))],
            "policy": {"a": 1.0}, "chosen": "a", "truth": {"state": state}, "truth_status": "ok",
        })
    rec.end_game("g", "u", recorder.RESULT_WIN, turns=12)
    samples = list(dataset.iter_samples(tmp_path))
    truth = [s for s in samples if s.kind == "truth"]
    assert [s.turn for s in truth] == [2, 7, 11]
    assert [s.decision_idx for s in truth] == [0, 1, 2]
    assert all(s.value == 1.0 for s in samples)
    out, stats = dataset.bootstrapped_targets(
        [s.game_id for s in samples], [s.decision_idx for s in samples],
        [s.decision_q if s.decision_q is not None else float("nan") for s in samples],
        [s.value for s in samples], "nstep", n=1,
    )
    assert stats["decisions"] == 3 and stats["gaps"] == 0
    # n-step looks at the next *recorded* decision, whatever its turn gap
    first = [o for o, s in zip(out, samples) if s.decision_idx == 0][0]
    assert first == pytest.approx(truth[1].decision_q)
