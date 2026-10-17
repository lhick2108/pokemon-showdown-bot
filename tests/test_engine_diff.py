"""scripts/engine_diff.py: differential engine-vs-Showdown harness (bead 589.68.2).

The turn tests need ``node`` and a built local Showdown checkout (``vendor/pokemon-showdown/dist``);
they are skipped when either is missing.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import engine_diff as ed  # noqa: E402


def _sim_available() -> bool:
    if shutil.which("node") is None:
        return False
    try:
        ed.find_showdown_dir()
    except SystemExit:
        return False
    return True


needs_sim = pytest.mark.skipif(not _sim_available(), reason="node or local Showdown checkout (dist/) missing")


def test_hp_tolerance_accepts_roll_spread_and_rejects_big_errors():
    # engine uses 0.925 x max; the simulator's rolls span 0.85..1.0 x max: about +-8% of the damage
    assert ed.hp_ok(300, 300 - 100, 300 - 92, 300)
    assert ed.hp_ok(300, 300 - 100, 300 - 108, 300)
    assert not ed.hp_ok(300, 300 - 100, 300 - 25, 300)  # 4x too high (Tera Shell-style error)
    assert not ed.hp_ok(300, 300 - 100, 300 - 150, 300)
    assert ed.hp_ok(300, 300, 298, 300)  # tiny absolute differences are noise


def test_to_id_and_engine_volatile_probe():
    assert ed.to_id("Tera Blast") == "terablast"
    assert ed.engine_supports_volatile("substitute")
    assert not ed.engine_supports_volatile("definitelynotavolatile")


@needs_sim
def test_seeded_turns_run_and_are_reproducible():
    recs = list(ed.node_records(1, 1000, 3))
    assert [r["turn"] for r in recs] == [1, 2, 3]
    again = list(ed.node_records(1, 1000, 3))
    assert json.dumps(recs, sort_keys=True) == json.dumps(again, sort_keys=True)  # the simulator side is deterministic

    rows = [ed.analyse_turn(r) for r in recs]
    for row in rows:
        assert row["status"] in ("match", "pp_only", "diverge", "skipped"), row
        assert row["conv_errors"] == [], "state builder lost values on the way in"
        assert row["unknown_ids"] == []
        assert row["branches"] >= 1
        assert row["tags"]
    # seed 1000 turn 1: a Giga Impact KO. The engine (<= sdb.19) sets no mustrecharge volatile, so the harness
    # reports "volatile.missing_in_engine"; once that engine bug is fixed the turn simply matches.
    first = rows[0]
    if first["status"] == "diverge":
        assert any(m["field"] == "volatile.missing_in_engine" for m in first["mismatches"])


@needs_sim
def test_report_renders_from_rows():
    rows = [ed.analyse_turn(r) for r in ed.node_records(1, 1001, 4)]
    text = ed.render_report({"games": 1, "seed": 1001, "rows": rows, "elapsed_s": 0.0}, top=5)
    assert "overall divergence" in text
    assert "Known suspects" in text
    assert "PP tracking" in text


@needs_sim
def test_pp_is_not_decremented_by_the_engine_above_nine():
    """Documents the known engine behaviour: only low PP is tracked (engine/README.md)."""
    high = []
    for rec in ed.node_records(1, 1000, 3):
        high += [o for o in ed.analyse_turn(rec)["pp_obs"] if o[0] >= 10]
    assert high
    assert all(engine_delta == 0 for _pre, engine_delta, _sim in high)
    assert any(sim_delta != 0 for _pre, _e, sim_delta in high)
