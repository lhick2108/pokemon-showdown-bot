"""The decision-quality puzzle suite (bead 589.32): structure and helper checks (no long searches)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
_SPEC = importlib.util.spec_from_file_location("puzzles", ROOT / "scripts" / "puzzles.py")
P = importlib.util.module_from_spec(_SPEC)
sys.modules["puzzles"] = P
_SPEC.loader.exec_module(P)

pe = pytest.importorskip("poke_engine")


def test_policy_helpers():
    pol = P.policy_of([(10, [("recover", 8, 1.0), ("switch x", 2, 0.1), ("recover-tera", 0, 0.0)])])
    assert pol == {"recover": pytest.approx(0.8), "x": pytest.approx(0.2)}
    # mixed rule: only choices within 75% of the best are kept
    assert P.mixed_probs({"a": 0.5, "b": 0.4, "c": 0.1}) == {"a": pytest.approx(5 / 9), "b": pytest.approx(4 / 9)}
    assert P.norm("switch toxapex") == "toxapex" and P.norm("earthquake-tera") == "earthquake"


def test_pass_rate_is_closed_form_over_world_pairs():
    w1 = (10, [("a", 10, 1.0), ("b", 0, 0.0)])
    w2 = (10, [("a", 0, 0.0), ("b", 10, 1.0)])
    rate, dist = P.pass_rate([w1, w1, w2], {"a"}, set())
    # pairs: (w1,w1) -> a ; (w1,w2) -> a:0.5 b:0.5 (twice)
    assert rate == pytest.approx((1 + 0.5 + 0.5) / 3)
    assert dist["a"] == pytest.approx(rate)


def test_builder_makes_parsable_states():
    ours = P.side([P.mon("TOXAPEX", hp=0.4, moves=["recover", "liquidation"], item="LEFTOVERS")])
    theirs = P.side([P.mon("MEW", moves=["knockoff"], item="LIFEORB")], rocks=1)
    st = pe.State.from_string(P.state(ours, theirs))
    assert st.side_one.pokemon[0].id == "TOXAPEX"
    assert st.side_one.pokemon[0].hp == round(0.4 * st.side_one.pokemon[0].maxhp)
    assert st.side_two.pokemon[0].item == "LIFEORB"
    assert st.side_two.side_conditions.stealth_rock == 1
    assert sum(1 for m in st.side_one.pokemon if m.hp > 0) == 1


PUZZLES = P.load_puzzles()


def test_suite_has_enough_puzzles_per_theme():
    counts = {}
    for p in PUZZLES:
        counts[(p["theme"], p["kind"])] = counts.get((p["theme"], p["kind"]), 0) + 1
    for theme in ("stall", "sacrifice"):
        for kind in ("positive", "control"):
            assert counts.get((theme, kind), 0) >= 10, counts
    assert counts.get(("setup", "positive"), 0) >= 10, counts


@pytest.mark.parametrize("pz", PUZZLES, ids=[p["id"] for p in PUZZLES])
def test_puzzle_is_well_formed(pz):
    st = pe.State.from_string(pz["state"])
    assert pz["accept"], "every puzzle names its acceptable moves"
    legal = {P.norm(c) for c in P._legal(st, 1)}
    for a in pz["accept"] + pz.get("reject", []):
        assert a in legal, (a, legal)
    assert not set(pz["accept"]) & set(pz.get("reject", []))
    assert pz["desc"] and pz["rationale"]
    if pz["theme"] == "setup":
        # deterministic evidence: every accepted attack knocks the foe out on its minimum damage roll
        foe = st.side_two.pokemon[int(st.side_two.active_index)]
        for a in pz["accept"]:
            lo, _ = pe.calculate_damage(st, a, foe.moves[0].id.lower(), True)[0]
            assert lo >= foe.hp, (a, lo, foe.hp)
        me = st.side_one.pokemon[int(st.side_one.active_index)]
        assert me.speed > foe.speed
        return
    # the independent evidence is stored and agrees with the labels
    values = pz["verify_search"]["values"]
    good = max(values[a] for a in pz["accept"])
    bad = max(values[a] for a in pz.get("reject", []))
    assert good - bad >= 0.15
