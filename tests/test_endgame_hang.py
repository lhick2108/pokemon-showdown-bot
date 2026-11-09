"""Hang / crash resilience of endgame search and the A/B harness (offline, mocked workers)."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from showdown_bot.players import _worker
from showdown_bot.players import search as search_module
from showdown_bot.players.search import SearchConfig

from test_endgame import _battle, _player, data  # noqa: F401  (fixture re-export)


def test_hanging_endgame_worker_times_out_and_mcts_answers(data, monkeypatch):  # noqa: F811
    monkeypatch.setattr(_worker, "endgame_world", lambda s, ms: time.sleep(6))
    pool = ThreadPoolExecutor(max_workers=4)  # 2 stay blocked by the hung worker; MCTS uses the rest
    p = _player(data, pool, endgame_cap_ms=1300, endgame_time_ms=200)
    started = time.time()
    order = asyncio.run(p.choose_move(_battle()))
    elapsed = time.time() - started
    assert order is not None and p.stats.fallbacks == 0  # MCTS answered, not the random fallback
    assert p.stats.endgame_fallbacks == 1 and p.stats.endgame_decisions == 0
    assert elapsed < 4.5, elapsed  # hard limit ~1.2 s, not the worker's 6 s
    pool.shutdown(wait=False, cancel_futures=True)


def test_owned_pool_is_killed_after_endgame_timeout(data, monkeypatch):  # noqa: F811
    killed = []
    monkeypatch.setattr(
        search_module, "reset_pool", lambda workers, only=None, kill=False: killed.append(kill)
    )
    monkeypatch.setattr(_worker, "endgame_world", lambda s, ms: time.sleep(5))
    p = _player(data, None, endgame_cap_ms=1200, endgame_time_ms=100)
    p._executor = None  # the player does not own a caller-supplied executor
    hung = ThreadPoolExecutor(max_workers=2)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(p._run_endgame_on(hung, ["x", "y"], 100))
    assert killed == [True]  # the pool is recycled so later decisions do not queue behind it
    hung.shutdown(wait=False, cancel_futures=True)


def test_supplied_executor_is_not_killed(data, monkeypatch):  # noqa: F811
    killed = []
    monkeypatch.setattr(
        search_module, "reset_pool", lambda workers, only=None, kill=False: killed.append(kill)
    )
    monkeypatch.setattr(_worker, "endgame_world", lambda s, ms: time.sleep(5))
    hung = ThreadPoolExecutor(max_workers=2)
    p = _player(data, hung, endgame_cap_ms=1200, endgame_time_ms=100)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(p._run_endgame_on(hung, ["x", "y"], 100))
    assert killed == []
    hung.shutdown(wait=False, cancel_futures=True)


def test_run_games_survives_dead_and_hung_children():
    from showdown_bot import endgame_eval as ev

    tasks = [
        {"index": 0, "a_side": 1, "state": "S0", "mode": "ok"},
        {"index": 1, "a_side": 1, "state": "S1", "mode": "die"},
        {"index": 2, "a_side": 1, "state": "S2", "mode": "hang"},
        {"index": 3, "a_side": 2, "state": "S3", "mode": "ok"},
    ]
    got: list[dict] = []
    ev.run_games(tasks, workers=3, timeout_s=8, on_result=got.append, target=ev._selftest_child)
    by = {r["index"]: r for r in got}
    assert len(got) == 4
    assert by[0]["ok"] and by[3]["ok"]
    assert not by[1]["ok"] and "died" in by[1]["error"] and by[1]["state"] == "S1"
    assert not by[2]["ok"] and "timeout" in by[2]["error"] and by[2]["state"] == "S2"


def test_long_mcts_on_a_degenerate_chain_does_not_overflow_the_stack():
    """Engine sdb.8: MCTS selection/backpropagate are iterative. This position makes the tree a
    chain tens of thousands deep within ~1.5 s; the recursive version died with
    STATUS_STACK_OVERFLOW, which hung a multiprocessing.Pool. Run in a subprocess so a regression
    is a failure here, not a dead test runner."""
    import subprocess
    import sys
    from importlib import metadata
    from pathlib import Path

    try:
        version = metadata.version("poke_engine")
    except metadata.PackageNotFoundError:
        pytest.skip("poke_engine not installed")
    if "sdb." not in version or int(version.rsplit("sdb.", 1)[1].split(".")[0]) < 8:
        pytest.skip(f"needs engine sdb.8 or newer (have {version})")
    state = Path(__file__).parent / "fixtures" / "mcts_deep_chain.state"
    code = (
        "import sys; from poke_engine import State, monte_carlo_tree_search;"
        "s = State.from_string(open(sys.argv[1]).read().strip());"
        "r = monte_carlo_tree_search(s, 2500, threads=1); print(r.total_visits)"
    )
    done = subprocess.run([sys.executable, "-c", code, str(state)], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.returncode
    assert int(done.stdout.strip()) > 500_000


def test_incremental_results_resume_and_torn_line(tmp_path):
    from showdown_bot import endgame_eval as ev

    path = tmp_path / "r.jsonl"
    ev.append_result(path, {"index": 0, "a_side": 1, "ok": True, "score": 1.0})
    ev.append_result(path, {"index": 0, "a_side": 2, "ok": False, "error": "timeout"})
    with open(path, "a") as fh:
        fh.write('{"index": 1, "a_si')  # torn last line from a killed run
    assert len(ev.load_results(path)) == 2
    ev.append_result(path, {"index": 0, "a_side": 2, "ok": True, "score": 0.0})  # retried ok
    latest = ev.latest_results(ev.load_results(path))
    assert {(r["index"], r["a_side"]): r["ok"] for r in latest} == {(0, 1): True, (0, 2): True}
    assert ev.load_results(tmp_path / "missing.jsonl") == []


def test_script_writes_jsonl_incrementally(tmp_path, monkeypatch):
    """The script appends each game to --out as it finishes (stubbed runner and positions)."""
    import importlib.util
    from pathlib import Path

    from showdown_bot import endgame_eval as ev

    spec = importlib.util.spec_from_file_location(
        "endgame_ab", Path(__file__).parents[1] / "scripts" / "endgame_ab.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(ev, "extract_positions", lambda *a, **k: [
        {"state": f"S{i}", "alive": [1, 1]} for i in range(3)])
    seen_after_first: list[int] = []
    out = tmp_path / "o.jsonl"

    def fake_run(tasks, workers, timeout_s, on_result, target=None):
        for t in tasks:
            on_result({"index": t["index"], "a_side": t["a_side"], "ok": True, "score": 1.0,
                       "steps": 3, "capped": False, "a_endgame_moves": 1, "a_fallbacks": 0,
                       "a_depths": [4], "seconds": 0.1})
            seen_after_first.append(len(ev.load_results(out)))

    monkeypatch.setattr(ev, "run_games", fake_run)
    argv = ["--source", str(tmp_path), "--positions", "3", "--out", str(out)]
    assert script.main(argv) == 0
    assert seen_after_first == [1, 2, 3, 4, 5, 6]  # flushed game by game
    assert script.main(argv + ["--resume"]) == 0  # everything done: nothing to play
    assert len(ev.load_results(out)) == 6
