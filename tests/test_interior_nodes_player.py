"""SearchPlayer plumbing of the interior-node export (bead showdown-bot-rnm)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import recorder  # noqa: E402
from showdown_bot.players.search import SearchConfig, SearchPlayer  # noqa: E402
from showdown_bot.sets import RandbatsData  # noqa: E402

HAS_EXPORT = "export_nodes" in pe.monte_carlo_tree_search.__code__.co_varnames
needs_export = pytest.mark.skipif(not HAS_EXPORT, reason="needs engine 0.0.48+sdb.9 (export_nodes)")


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=Path(__file__).parent / "fixtures")


def _player(data, pool, **kwargs):
    return SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                        data=data, executor=pool, seed=1, **kwargs)


def _recorded_decision(data, tmp_path, **cfg):
    rec = recorder.GameRecorder(tmp_path / "run")
    with ThreadPoolExecutor(max_workers=2) as pool:
        player = _player(data, pool, config=SearchConfig(search_time_ms=40, **cfg), recorder=rec)
        battle = make_battle(request_for())
        event(battle, "start")
        opp_switch(battle, "Gardevoir", "Gardevoir, L83")
        asyncio.run(player.choose_move(battle))
        battle.won_by("bot")
        player._battle_finished_callback(battle)
    (path,) = recorder.iter_game_files(tmp_path / "run")
    return recorder.read_game(path)["decisions"][0]


@needs_export
def test_search_player_records_exported_nodes(data, tmp_path):
    decision = _recorded_decision(data, tmp_path, export_nodes=5, export_min_visits=8)
    assert decision["worlds"]
    for world in decision["worlds"]:
        assert 0.0 < world["search_value"] < 1.0
        assert 0 < len(world["nodes"]) <= 5
        state, visits, value, static, depth = world["nodes"][0]
        pe.State.from_string(state)
        assert visits >= 8 and depth >= 1


def test_export_is_off_by_default_and_validated(data, tmp_path):
    assert SearchConfig().export_nodes == 0
    decision = _recorded_decision(data, tmp_path)
    assert all("nodes" not in w and "search_value" not in w for w in decision["worlds"])
    with ThreadPoolExecutor(max_workers=1) as pool:
        for bad in ({"export_nodes": 4, "threads": 2}, {"export_nodes": -1}, {"export_min_visits": 0}):
            with pytest.raises(ValueError):
                _player(data, pool, config=SearchConfig(**bad))
