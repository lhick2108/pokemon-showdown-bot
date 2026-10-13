"""scripts/debug_ko_value.py helpers (bead 589.43)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import debug_ko_value as dkv  # noqa: E402
from poke_engine import State  # noqa: E402


def test_maximin_pure_picks_best_worst_case():
    matrix = {"a": {"x": 1.0, "y": 5.0}, "b": {"x": 3.0, "y": 4.0}}
    pure, mixed, _ = dkv.maximin(matrix, ["a", "b"], ["x", "y"])
    assert pure == (3.0, "b")
    if mixed is not None:  # scipy available: row b dominates, same value
        assert abs(mixed - 3.0) < 1e-6


def test_drop_mon_faints_only_that_opponent_mon():
    puzzle = json.loads((Path(__file__).parent / "puzzles" / "setup.json").read_text())[1]
    state = State.from_string(puzzle["state"])
    alt = dkv.drop_mon(state, "snorlax")
    before = {p.id: p.hp for p in state.side_two.pokemon}
    after = {p.id: p.hp for p in alt.side_two.pokemon}
    assert after["SNORLAX"] == 0 and before["SNORLAX"] > 0
    assert all(after[k] == before[k] for k in before if k != "SNORLAX")
