"""Per-game @K capping in the sharded human-replay layout, and the load-memory reductions."""

from __future__ import annotations

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")

from test_train import state_strings  # noqa: E402,F401
from test_train_subsample import _write_games  # noqa: E402

from showdown_bot import recorder, train  # noqa: E402


def _shard_layout(root, states):
    """Human-replay layout: ``<root>/shard-NNNNN/<battle>__p1|p2.jsonl.gz``, one file per game side."""
    for shard in range(3):
        rec = recorder.GameRecorder(root / f"shard-{shard:05d}")
        for g in range(4):
            for user in ("p1", "p2"):
                tag = f"b{shard}{g}"
                rec.start_game(tag, user)
                for d in range(7 + g):
                    st = states[(shard + g + d) % len(states)]
                    rec.add_decision(tag, user, {
                        "turn": d + 1, "policy": {"a": 1.0}, "chosen": "a",
                        "truth": {"state": st}, "truth_status": "ok"})
                rec.end_game(tag, user, recorder.RESULT_WIN, turns=7 + g)


def test_cap_is_per_game_record_in_sharded_layout(tmp_path, state_strings, monkeypatch):  # noqa: F811
    _shard_layout(tmp_path, state_strings)
    monkeypatch.setattr(train, "LOAD_CHUNK_FILES", 5)  # chunks straddle shard directories
    monkeypatch.setattr(train, "STATE_KEEP_EVERY", 1)
    n_files = len(list(recorder.iter_game_files(tmp_path)))
    assert n_files == 24
    a = train.load_data([train.Source(tmp_path, 3)], states="truth", seed=2, workers=1)
    b = train.load_data([train.Source(tmp_path, 3)], states="truth", seed=2, workers=2)
    assert len(a) == 3 * n_files  # K per game record, not per shard
    assert a.tags == b.tags and a.states == b.states and np.array_equal(a.x, b.x)
    full = train.load_data([train.Source(tmp_path, 0)], states="truth", workers=1)
    assert len(full) == 3 * 2 * sum(7 + g for g in range(4))
    assert all(a.states)  # keep-every 1 keeps every string


def test_state_strings_kept_for_a_small_deterministic_subset(tmp_path, state_strings):  # noqa: F811
    _write_games(tmp_path, state_strings, n_games=40, n_dec=30, searched=False, prefix="h")
    a = train.load_data([train.Source(tmp_path)], states="truth", workers=1)
    kept = [s for s in a.states if s]
    assert len(a) == 1200 and 0 < len(kept) < len(a) // 10  # ~1/50 of the samples
    assert set(kept) <= set(state_strings)
    b = train.load_data([train.Source(tmp_path)], states="truth", workers=1)
    assert a.states == b.states  # deterministic
    sub = a.subset(np.arange(0, len(a), 2))  # subset() carries the Nones along
    assert len(sub.states) == len(sub) and sum(s is not None for s in sub.states) <= len(kept)


def test_load_reports_memory_and_cleans_spill(tmp_path, state_strings, capsys):  # noqa: F811
    _write_games(tmp_path, state_strings, n_games=3, n_dec=4, searched=False, prefix="h")
    d = train.load_data([train.Source(tmp_path)], states="truth", workers=1, x_dtype="float16")
    out = capsys.readouterr().out
    assert "12 samples, x " in out and "GB" in out and f"loaded {len(d)} samples" in out
    assert d.x.dtype == np.float16
