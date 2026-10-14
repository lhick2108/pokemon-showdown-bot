import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "gauntlet", Path(__file__).resolve().parents[1] / "scripts" / "gauntlet.py")
gauntlet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gauntlet)


def test_wilson_and_summary():
    lo, hi = gauntlet.wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    s = gauntlet.summarize({"a": [1.0, 0.0, 0.5, 1.0], "b": [0.0, 0.0]}, ["a", "b", "c"])
    assert set(s["opponents"]) == {"a", "b"}
    assert abs(s["mean"] - (0.625 + 0.0) / 2) < 1e-9
    assert s["mean_ci"][0] <= s["mean"] <= s["mean_ci"][1]


def test_resume_plays_only_missing_games(tmp_path, monkeypatch):
    calls = []

    def fake_chunk(opp, spec, cand, n, idx, args, out_dir):
        calls.append((opp, n, idx))
        if len(calls) == 2:  # simulated stall: only part of the chunk finished
            n = n // 2
        return [{"opponent": opp, "chunk": idx, "battle_tag": f"{opp}-{idx}-{i}", "score": 1.0, "turns": 5}
                for i in range(n)]

    monkeypatch.setattr(gauntlet, "run_chunk", fake_chunk)
    out = tmp_path / "g.jsonl"
    argv = ["--candidate", "search", "--out", str(out), "--games", "10", "--chunk", "4", "--only", "search"]
    assert gauntlet.main(argv) == 0
    header, scores, chunks = gauntlet.read_log(out)
    assert len(scores["search"]) == 10 and header["candidate"] == "search"
    n_calls = len(calls)
    assert gauntlet.main(argv) == 0  # complete: nothing replayed
    assert len(calls) == n_calls
    out.write_text("\n".join(out.read_text().splitlines()[:-3]) + "\n")  # drop 3 games: resume
    gauntlet.main(argv)
    assert len(gauntlet.read_log(out)[1]["search"]) == 10
    json.loads(out.read_text().splitlines()[0])
