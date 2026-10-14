"""The misplay regression suite (scripts/misplay_suite.py): helpers and suite structure (no long searches)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("misplay_suite", ROOT / "scripts" / "misplay_suite.py")
M = importlib.util.module_from_spec(_SPEC)
sys.path.insert(0, str(ROOT / "scripts"))  # spawned search workers import the module by name
sys.modules["misplay_suite"] = M
_SPEC.loader.exec_module(M)

pe = pytest.importorskip("poke_engine")

CATEGORIES = {"ko_value", "heal_loop", "sub_loop", "toxic", "taunt", "encore", "tera", "sleep_clause",
              "role", "pivot"}


def test_norm_folds_tera_and_switch():
    assert M.norm("earthquake-tera") == "earthquake"
    assert M.norm("earthquake-tera", fold_tera=False) == "earthquake-tera"
    assert M.norm("switch toxapex") == "toxapex"
    assert M.norm("switch toxapex-tera") == "toxapex"


def test_policy_sums_visit_shares_over_worlds():
    w1 = (10, [("a", 8, 4.0), ("b-tera", 2, 1.0)])
    w2 = (10, [("a", 2, 1.0), ("b", 8, 4.0)])
    assert M.policy([w1, w2]) == pytest.approx({"a": 0.5, "b": 0.5})
    assert M.policy([w1], fold_tera=False) == pytest.approx({"a": 0.8, "b-tera": 0.2})
    assert M.policy([(0, [])]) == {}


def test_passes_accept_reject_and_empty():
    assert M.passes({"a": 0.6, "b": 0.4}, {"accept": ["a"]})
    assert not M.passes({"a": 0.4, "b": 0.6}, {"accept": ["a"]})
    assert M.passes({"a": 0.4, "b": 0.6}, {"reject": ["a"]})
    assert not M.passes({"a": 0.6, "b": 0.4}, {"reject": ["a"]})
    assert not M.passes({}, {"accept": ["a"]})
    # a tera puzzle keeps the suffix: rejecting "x-tera" passes the plain "x"
    assert M.passes({"x": 0.5, "x-tera": 0.4}, {"reject": ["x-tera"], "fold_tera": False})


def test_table_reports_pass_rate_per_category():
    puzzles = [{"id": "a", "cat": "toxic"}, {"id": "b", "cat": "toxic"}, {"id": "c", "cat": "tera"}]
    out = M.table(puzzles, {100: {"a": 1.0, "b": 0.0, "c": 0.5}, 1000: {"a": 1.0, "b": 1.0, "c": 0.5}})
    rows = {line.split()[0]: line.split() for line in out.splitlines()[1:]}
    assert rows["toxic"][1:] == ["2", "0.50", "1.00"]
    assert rows["tera"][1:] == ["1", "0.50", "0.50"]
    assert rows["ALL"][1:] == ["3", "0.50", "0.83"]


SUITE = M.load_suite(tier="all") if M.SUITE.exists() else []
VALIDATED = [p for p in SUITE if p["tier"] == "validated"]


def test_table_separates_controls():
    puzzles = [{"id": "a", "cat": "tera"}, {"id": "b", "cat": "tera", "kind": "control"}]
    out = M.table(puzzles, {100: {"a": 0.0, "b": 1.0}})
    assert "tera (ctl)" in out and "tera " in out


@pytest.mark.skipif(not SUITE, reason="no suite file")
def test_suite_structure():
    ids = [p["id"] for p in SUITE]
    assert len(ids) == len(set(ids))
    assert 30 <= len(VALIDATED) <= 70
    for p in SUITE:
        assert p["cat"] in CATEGORIES, p["id"]
        assert p["tier"] in ("validated", "review"), p["id"]
        assert p["states"] and p["desc"] and p["why"] and p["evidence"], p["id"]
        assert p.get("accept") or p.get("reject"), p["id"]
        if p["tier"] == "validated":
            assert set(p["evidence"]) & {"playout", "long_search"}, p["id"]
        else:
            assert "review" in p["evidence"], p["id"]
        legal = _legal_choices(p["states"][0], p.get("fold_tera", True))
        for choice in list(p.get("accept", [])) + list(p.get("reject", [])):
            assert choice in legal, (p["id"], choice, sorted(legal))
        for s in p["states"]:
            st = pe.State.from_string(s)
            assert st.side_one.pokemon[int(st.side_one.active_index)].hp > 0


@pytest.mark.skipif(not SUITE, reason="no suite file")
def test_validated_evidence_thresholds():
    for p in VALIDATED:
        ev = p["evidence"]
        if "playout" in ev:
            assert ev["playout"]["gap"] >= 0.25, p["id"]
            if ev["playout"]["confirm"] is not None:
                assert ev["playout"]["confirm"]["gap"] >= 0.15, p["id"]
        if "long_search" in ev:
            assert ev["long_search"]["seconds"] * ev["long_search"]["worlds"] >= 8, p["id"]


@pytest.mark.skipif(not SUITE, reason="no suite file")
def test_every_category_is_represented():
    cats = {p["cat"] for p in VALIDATED}
    assert cats >= CATEGORIES - {"sub_loop"}, CATEGORIES - cats
    assert {p["cat"] for p in SUITE} >= CATEGORIES


def _legal_choices(state_string: str, fold_tera: bool) -> set[str]:
    res = pe.monte_carlo_tree_search(pe.State.from_string(state_string), 20, iterations=300)
    out = {M.norm(o.move_choice, fold_tera) for o in res.side_one}
    if not fold_tera:
        out |= {M.norm(o.move_choice) for o in res.side_one}
    return out


@pytest.mark.skipif(not SUITE, reason="no suite file")
def test_run_reports_a_pass_rate_per_category(tmp_path):
    puzzles = [p for p in VALIDATED if p["cat"] in ("toxic", "encore")][:3]
    res = M.run_suite(puzzles, None, 1.0, [20], 2, 1, 1)
    assert set(res) == {20} and set(res[20]) == {p["id"] for p in puzzles}
    assert all(0.0 <= v <= 1.0 for v in res[20].values())
    assert "ALL" in M.table(puzzles, res)


def test_suite_file_is_json_list():
    if M.SUITE.exists():
        assert isinstance(json.loads(M.SUITE.read_text()), list)
