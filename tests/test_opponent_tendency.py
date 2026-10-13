"""Tests for showdown_bot.opponent_tendency (589.40): tracker, blending and safe exploitation."""

from __future__ import annotations

import numpy as np
import pytest

from showdown_bot import opponent_tendency as ot
from showdown_bot.nash import solve_matrix_game
from showdown_bot.players import search as search_module
from showdown_bot.players.biased import BiasedSearchPlayer


def test_classify_choices():
    assert ot.classify_choice("earthquake") == ot.ATTACK
    assert ot.classify_choice("earthquake-tera") == ot.ATTACK
    assert ot.classify_choice("suckerpunch") == ot.PRIORITY
    assert ot.classify_choice("iceshard") == ot.PRIORITY
    assert ot.classify_choice("swordsdance") == ot.STATUS
    assert ot.classify_choice("roost") == ot.STATUS
    assert ot.classify_choice("protect") == ot.PROTECT
    assert ot.classify_choice("kingsshield") == ot.PROTECT
    assert ot.classify_choice("switch gholdengo") == ot.SWITCH
    assert ot.classify_choice("none") == ot.STATUS


def _events(*lines):
    return [l.split("|") for l in lines]


def test_tracker_records_first_action_per_turn_under_context():
    tr = ot.TendencyTracker()
    cond = ot.Context(cond_priority=True)
    plain = ot.Context()
    tr.observe_events(_events("|turn|1"), "p2")
    tr.set_context(1, cond)
    tr.observe_events(_events(
        "|move|p2a: Foo|Swords Dance|p2a: Foo", "|move|p1a: Us|Sucker Punch|p2a: Foo",
        "|-fail|p1a: Us", "|upkeep|", "|switch|p2a: Bar|Bar, L80|100/100", "|turn|2"), "p2")
    tr.set_context(2, cond)
    tr.observe_events(_events("|move|p1a: Us|Tackle|p2a: Bar", "|move|p2a: Bar|Protect|p2a: Bar",
                              "|move|p2a: Bar|Tackle|p1a: Us", "|turn|3"), "p2")
    tr.set_context(3, plain)
    tr.observe_events(_events("|switch|p2a: Baz|Baz, L80|100/100", "|turn|4"), "p2")
    # replacement switch after |upkeep| is not a chosen action; a turn without our context is skipped
    tr.observe_events(_events("|move|p2a: Baz|Tackle|p1a: Us", "|upkeep|", "|switch|p2a: Foo|Foo, L80|50/100", "|turn|5"), "p2")
    assert tr.counts == {"P--": {ot.STATUS: 1.0, ot.PROTECT: 1.0}, "---": {ot.SWITCH: 1.0}}
    assert [h[2] for h in tr.history] == [ot.STATUS, ot.PROTECT, ot.SWITCH]
    assert tr.counts_for(cond) == {ot.STATUS: 1.0, ot.PROTECT: 1.0}


def test_tracker_ignores_called_moves_and_own_side():
    tr = ot.TendencyTracker()
    tr.observe_events(_events("|turn|1"), "p2")
    tr.set_context(1, ot.Context())
    tr.observe_events(_events("|move|p1a: Us|Tackle|p2a: Foo",
                              "|move|p2a: Foo|Earthquake|p1a: Us|[from]move: Sleep Talk"), "p2")
    assert tr.counts == {}
    tr.observe_events(_events("|move|p2a: Foo|Earthquake|p1a: Us"), "p2")
    assert tr.counts == {"---": {ot.ATTACK: 1.0}}
    # events without a known role are dropped
    other = ot.TendencyTracker()
    other.observe_events(_events("|turn|1"), None)
    assert other.turn == 0


def test_posterior_is_prior_with_no_data_and_moves_gradually():
    nash = {ot.ATTACK: 0.7, ot.STATUS: 0.3}
    assert ot.posterior_classes(nash, {}, 4) == pytest.approx(nash)
    post = ot.posterior_classes(nash, {ot.STATUS: 3}, 4)
    assert post[ot.STATUS] == pytest.approx((3 + 4 * 0.3) / 7)
    assert post[ot.ATTACK] == pytest.approx(4 * 0.7 / 7)
    assert sum(post.values()) == pytest.approx(1)
    # "attack drops substantially" after 3/3 baits, but not to zero
    assert 0.7 > post[ot.ATTACK] > 0.3
    # more data -> more movement; counts of unavailable classes are ignored
    assert ot.posterior_classes(nash, {ot.STATUS: 30}, 4)[ot.ATTACK] < post[ot.ATTACK]
    assert ot.posterior_classes(nash, {ot.PROTECT: 50}, 4) == pytest.approx(nash)
    # kappa=0 with data is the empirical frequency
    assert ot.posterior_classes(nash, {ot.STATUS: 3}, 0)[ot.STATUS] == pytest.approx(1.0)


def test_blend_belief_keeps_within_class_nash_weights():
    col = np.array([0.5, 0.2, 0.3, 0.0])
    classes = [ot.ATTACK, ot.ATTACK, ot.STATUS, ot.PROTECT]
    belief, post, n = ot.blend_belief(col, classes, {ot.STATUS: 3}, 4)
    assert belief.sum() == pytest.approx(1)
    assert n == 3
    assert belief[0] / belief[1] == pytest.approx(0.5 / 0.2)
    assert belief[3] == pytest.approx(0.0)  # Nash never protects and no evidence
    assert belief[2] == pytest.approx(post[ot.STATUS])
    # a class with zero Nash mass but evidence gets spread uniformly over its columns
    belief, _, _ = ot.blend_belief(col, classes, {ot.PROTECT: 6}, 4)
    assert belief[3] == pytest.approx(6 / 10)


# Our rows: [attack-into-both, setup]; opp columns: [attack, status]. Sucker-Punch-like game:
#   row0 = Sucker Punch (great vs attack, fails vs status), row1 = setup, row2 = plain attack.
BAIT_GAME = np.array([
    [0.80, 0.20],  # Sucker Punch
    [0.30, 0.70],  # set up
    [0.55, 0.55],  # plain attack
])


def test_safe_exploit_shifts_against_baiter_but_stays_within_delta():
    sol = solve_matrix_game(BAIT_GAME)
    classes = [ot.ATTACK, ot.STATUS]
    # no evidence: identical to Nash
    strat, info = ot.exploit_world(BAIT_GAME, sol.row, sol.col, sol.value, classes, {})
    assert strat == pytest.approx(sol.row)
    assert info["lambda"] == 0
    # 6/6 baits (status): best response is to set up, but the loss vs Nash is bounded
    strat, info = ot.exploit_world(BAIT_GAME, sol.row, sol.col, sol.value, classes,
                                   {ot.STATUS: 6}, kappa=4, delta=0.05)
    assert strat.sum() == pytest.approx(1)
    assert info["br"] == 1
    assert strat[1] > sol.row[1] + 0.05
    worst = (strat @ BAIT_GAME).min()
    assert worst >= sol.value - 0.05 - 1e-9
    # against a baiter the exploit beats Nash; against the opposite behaviour it loses <= delta
    baiter = np.array([0.0, 1.0])
    attacker = np.array([1.0, 0.0])
    assert strat @ BAIT_GAME @ baiter > sol.row @ BAIT_GAME @ baiter
    assert strat @ BAIT_GAME @ attacker >= sol.row @ BAIT_GAME @ attacker - 0.05 - 1e-9


def test_safe_exploit_delta_zero_is_exactly_nash():
    sol = solve_matrix_game(BAIT_GAME)
    strat, info = ot.exploit_world(BAIT_GAME, sol.row, sol.col, sol.value,
                                   [ot.ATTACK, ot.STATUS], {ot.STATUS: 50}, delta=0.0)
    assert strat == pytest.approx(sol.row, abs=1e-6)


def test_safe_exploit_never_exceeds_confidence_weight():
    A = np.array([[1.0, 1.0], [0.0, 0.0]])  # dominant row: exploiting costs nothing but it is the BR anyway
    row = np.array([0.5, 0.5])
    out, lam, br = ot.safe_exploit(A, row, 0.5, np.array([0.5, 0.5]), weight=0.25, delta=1.0)
    assert lam == pytest.approx(0.25) and br == 0
    assert out == pytest.approx([0.5 * 0.75 + 0.25, 0.5 * 0.75])


def test_nash_world_with_tendency_uses_opp_choice_classes():
    options = ["suckerpunch", "swordsdance", "tackle"]
    cells = [(i, j, 500, float(BAIT_GAME[i, j])) for i in range(3) for j in range(2)]
    result = (3000, [(c, 100, 50.0) for c in options], None, (2, cells), ["tackle", "swordsdance"])
    plain, _, _ = search_module.nash_world(result, 0.0)
    query = search_module.TendencyQuery({ot.STATUS: 6})
    blended, _, _ = search_module.nash_world(result, 0.0, query)
    assert blended["swordsdance"] > plain["swordsdance"]
    assert blended["suckerpunch"] < plain["suckerpunch"]
    assert sum(blended.values()) == pytest.approx(1)
    assert query.diag and query.diag[0]["br_choice"] == "swordsdance"
    # no evidence -> identical; worlds without opponent names are plain Nash
    same, _, _ = search_module.nash_world(result, 0.0, search_module.TendencyQuery({}))
    assert same == pytest.approx(plain)
    nameless = result[:4]
    assert search_module.nash_world(nameless, 0.0, search_module.TendencyQuery({ot.STATUS: 6}))[0] == pytest.approx(plain)


def test_config_env_and_validation(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("SHOWDOWN_SEARCH_"):
            monkeypatch.delenv(name)
    cfg = search_module.SearchConfig()
    assert cfg.tendency is False
    monkeypatch.setenv("SHOWDOWN_SEARCH_TENDENCY", "on")
    monkeypatch.setenv("SHOWDOWN_SEARCH_TENDENCY_KAPPA", "2")
    monkeypatch.setenv("SHOWDOWN_SEARCH_TENDENCY_DELTA", "0.1")
    cfg = search_module.SearchConfig.from_env()
    assert (cfg.tendency, cfg.tendency_kappa, cfg.tendency_delta) == (True, 2.0, 0.1)
    with pytest.raises(ValueError):
        search_module.SearchPlayer(tendency=True, config=search_module.SearchConfig(threads=2),
                                   start_listening=False)
    with pytest.raises(ValueError):
        BiasedSearchPlayer(bias="nope", start_listening=False)


def test_league_spec_accepts_tendency():
    from showdown_bot import league

    spec = league.parse_spec("t=search:tendency=on:tkappa=3:tdelta=0.02")
    assert spec.param("tendency") == "on" and spec.param("tkappa") == 3
    with pytest.raises(ValueError):
        league.parse_spec("t=search:tendency=maybe")


def test_worker_returns_opponent_choice_names():
    from showdown_bot.players import _worker
    from test_engine import make_state

    s = str(make_state().to_string())
    res = _worker.search_world(s, 20, 1, None, 1.0, "ucb1", 0.1, True, opp_names=True)
    assert len(res) == 5 and len(res[4]) == res[3][0]
    assert all(isinstance(c, str) for c in res[4])
    assert len(_worker.search_world(s, 20, 1, None, 1.0, "ucb1", 0.1, True)) == 4
