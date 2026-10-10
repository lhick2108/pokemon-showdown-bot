"""Recorder schema and SearchPlayer recording hook (no server needed)."""

from __future__ import annotations

import asyncio
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import recorder  # noqa: E402
from showdown_bot.players.search import SearchConfig, SearchPlayer  # noqa: E402
from showdown_bot.sets import RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def fake_world(state: str = "STATE"):
    return recorder.world_record(state, (100, [("a", 75, 60.0), ("b", 25, 10.0)]))


def write_fake_game(rec, result="win", tag="battle-x-1", user="alice"):
    rec.start_game(tag, user, role="p1", battle_format="gen9randombattle", config={"k": 1})
    rec.add_decision(tag, user, {
        "turn": 3, "side": "one", "worlds": [fake_world("S1"), fake_world("S2")],
        "policy": {"a": 0.75, "b": 0.25}, "chosen": "a", "order": "/choose move a",
        "compute": {"search_time_ms": 100}, "truth": {"state": "T"}, "truth_status": "ok",
    })
    return rec.end_game(tag, user, result, turns=4)


def test_schema_round_trip(tmp_path):
    rec = recorder.GameRecorder(tmp_path)
    path = write_fake_game(rec)
    assert path is not None and path.name == "battle-x-1__alice.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        kinds = [json.loads(line)["type"] for line in handle]
    assert kinds == ["game", "decision", "result"]
    game = recorder.read_game(path)
    header = game["header"]
    assert header["schema_version"] == recorder.SCHEMA_VERSION
    assert header["feature_version"] == recorder.feature_version()
    assert (header["battle_tag"], header["role"], header["config"]) == ("battle-x-1", "p1", {"k": 1})
    (decision,) = game["decisions"]
    assert decision["turn"] == 3 and decision["chosen"] == "a" and decision["truth"] == {"state": "T"}
    first = decision["worlds"][0]
    assert first["state"] == "S1" and first["total_visits"] == 100
    assert first["avg_score"] == pytest.approx(0.7)
    assert first["moves"][0] == {"choice": "a", "visits": 75, "score": 60.0}
    assert game["result"] == {"type": "result", "result": "win", "turns": 4}
    assert rec.games_written == 1 and rec.decisions_written == 1 and rec.worlds_written == 2
    assert list(recorder.iter_game_files(tmp_path)) == [path]


def test_game_without_decisions_writes_nothing(tmp_path):
    rec = recorder.GameRecorder(tmp_path)
    rec.start_game("t", "u")
    assert rec.end_game("t", "u", "loss") is None
    assert list(tmp_path.iterdir()) == []


def test_read_rejects_unknown_schema(tmp_path):
    path = tmp_path / "g.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps({"type": "game", "schema_version": 999}) + "\n")
    with pytest.raises(ValueError, match="schema_version"):
        recorder.read_game(path)


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.delenv(recorder.RECORD_DIR_ENV, raising=False)
    assert recorder.from_env() is None
    monkeypatch.setenv(recorder.RECORD_DIR_ENV, str(tmp_path / "run"))
    assert recorder.from_env().run_dir == tmp_path / "run"


# --- SearchPlayer hook ---------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _player(data, pool, **kwargs):
    return SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20), executor=pool, seed=1, **kwargs,
    )


def _battle():
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


def test_recording_off_writes_nothing_and_calls_no_recorder(data, tmp_path, monkeypatch):
    monkeypatch.delenv(recorder.RECORD_DIR_ENV, raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("recorder used while recording is off")

    for name in ("world_record", "register_player", "peer_battle", "build_truth_state"):
        monkeypatch.setattr(recorder, name, boom)
    for name in ("start_game", "add_decision", "end_game"):
        monkeypatch.setattr(recorder.GameRecorder, name, boom)
    monkeypatch.chdir(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        player = _player(data, pool)
        assert player._recorder is None
        battle = _battle()
        order = asyncio.run(player.choose_move(battle))
        battle.won_by("bot")
        player._battle_finished_callback(battle)
    assert order.message in {o.message for o in battle.valid_orders}
    assert player.stats.searched == 1 and player.stats.fallbacks == 0
    assert list(tmp_path.rglob("*")) == []


def test_recording_on_records_decision_and_result(data, tmp_path):
    rec = recorder.GameRecorder(tmp_path / "run")
    with ThreadPoolExecutor(max_workers=2) as pool:
        player = _player(data, pool, recorder=rec)
        battle = _battle()
        order = asyncio.run(player.choose_move(battle))
        battle.won_by("bot")
        player._battle_finished_callback(battle)
    (path,) = recorder.iter_game_files(tmp_path / "run")
    game = recorder.read_game(path)
    (decision,) = game["decisions"]
    assert game["result"]["result"] == "win"
    assert len(decision["worlds"]) == 4 and decision["compute"]["worlds_searched"] == 4
    assert decision["order"] == order.message
    assert decision["chosen"] in decision["policy"]
    assert decision["truth"] is None and decision["truth_status"] == "no-peer"
    assert all(w["state"] and w["moves"] for w in decision["worlds"])
