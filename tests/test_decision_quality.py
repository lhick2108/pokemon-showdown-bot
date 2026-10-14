"""Bead 589.47: flat-root escalation, no-op pruning and the tera edge (no server needed)."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

pe = pytest.importorskip("poke_engine")

from battle_helpers import event, make_battle, opp_switch, request_for  # noqa: E402

from showdown_bot import decision_quality as dq  # noqa: E402
from showdown_bot import engine_state  # noqa: E402
from showdown_bot.players.search import SearchConfig, SearchPlayer  # noqa: E402
from showdown_bot.sets import RandbatsData  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def world(*options):
    """A synthetic world result: options are (choice, visits, mean_score)."""
    total = sum(v for _, v, _ in options)
    return (total, [(c, v, v * m) for c, v, m in options])


# --- tera edge -----------------------------------------------------------------


def test_tera_folds_into_plain_when_edge_is_small():
    results = [world(("hypervoice", 100, 0.60), ("hypervoice-tera", 100, 0.605), ("protect", 50, 0.3))]
    policy = {"hypervoice": 0.4, "hypervoice-tera": 0.4, "protect": 0.2}
    out, folded = dq.fold_tera(policy, dq.option_means(results), 0.02)
    assert folded == ["hypervoice-tera"]
    assert out == {"hypervoice": pytest.approx(0.8), "protect": 0.2}


def test_tera_kept_when_edge_exceeds_margin_and_margin_zero_is_off():
    results = [world(("a", 100, 0.50), ("a-tera", 100, 0.55))]
    policy = {"a": 0.5, "a-tera": 0.5}
    means = dq.option_means(results)
    out, folded = dq.fold_tera(policy, means, 0.02)
    assert folded == [] and out == policy
    worse = {"a": 0.5, "a-tera": 0.0}
    out, folded = dq.fold_tera(worse, {"a": 0.5, "a-tera": 0.3}, 0.0)
    assert folded == [] and out == worse  # off


def test_tera_without_plain_sibling_is_left_alone():
    out, folded = dq.fold_tera({"a-tera": 1.0}, {"a-tera": 0.1}, 0.05)
    assert folded == [] and out == {"a-tera": 1.0}


def test_tera_means_are_visit_weighted_over_worlds():
    results = [
        world(("a", 300, 0.5), ("a-tera", 100, 0.9)),
        world(("a", 100, 0.5), ("a-tera", 300, 0.5)),
    ]
    means = dq.option_means(results)
    assert means["a"] == pytest.approx(0.5)
    assert means["a-tera"] == pytest.approx((100 * 0.9 + 300 * 0.5) / 400)


# --- no-op pruning (policy side) --------------------------------------------------


def test_prune_policy_drops_noops_but_never_everything():
    policy = {"rest": 0.5, "thunderbolt": 0.3, "switch x": 0.2}
    out, dropped = dq.prune_policy(policy, {"rest"})
    assert dropped == ["rest"] and out == {"thunderbolt": 0.3, "switch x": 0.2}
    out, dropped = dq.prune_policy({"rest": 0.6, "wish": 0.4}, {"rest", "wish"})
    assert dropped == [] and out == {"rest": 0.6, "wish": 0.4}


def test_adjust_policy_folds_tera_then_prunes():
    results = [world(("x", 50, 0.5), ("x-tera", 50, 0.5), ("y", 100, 0.4))]
    policy = {"x": 0.25, "x-tera": 0.25, "y": 0.5}
    out, changes = dq.adjust_policy(policy, results, noop={"x"}, tera_margin=0.02)
    assert out == {"y": 0.5}
    assert changes == {"tera_folded": ["x-tera"], "noop_pruned": ["x"]}


# --- flat-root detection -----------------------------------------------------------


def test_flat_root_detects_close_top_two_and_merges_tera_siblings():
    results = [world(("a", 100, 0.50), ("b", 90, 0.505), ("c", 5, 0.1))]
    policy = {"a": 0.5, "b": 0.45, "c": 0.05}
    assert dq.flat_root(policy, results, 0.02)
    clear = [world(("a", 100, 0.80), ("b", 90, 0.40))]
    assert not dq.flat_root({"a": 0.5, "b": 0.5}, clear, 0.02)
    # a and a-tera are the same decision: second group is b
    sib = [world(("a", 100, 0.50), ("a-tera", 100, 0.50), ("b", 50, 0.2))]
    assert not dq.flat_root({"a": 0.4, "a-tera": 0.4, "b": 0.2}, sib, 0.02)
    # a single group can't be flat
    assert not dq.flat_root({"a": 0.5, "a-tera": 0.5}, [world(("a", 5, 0.5), ("a-tera", 5, 0.5))], 0.02)


# --- no-op detection (engine side) ----------------------------------------------------

TEAM = [
    ("Corviknight", 82, "Rocky Helmet", "Pressure",
     ["Toxic", "Protect", "Substitute", "Brave Bird"], "Dragon"),
    ("Great Tusk", 80, "Leftovers", "Protosynthesis",
     ["Close Combat", "Headlong Rush", "Knock Off", "Rapid Spin"], "Ghost"),
] + [("Heatran", 80, "Leftovers", "Flash Fire",
      ["Magma Storm", "Earth Power", "Stealth Rock", "Taunt"], "Grass")] * 4


def _state(player, opp_status=None):
    battle = make_battle(request_for(TEAM))
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    if opp_status:
        event(battle, "-status", "p2a: Gardevoir", opp_status)
    trackers = player.opponent_model.update(battle)
    slots = engine_state.sample_opponent_team(battle, trackers, player._sampler, random.Random(0))
    return engine_state.build_state(battle, slots).to_string()


@pytest.fixture
def player(data):
    return SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=20), seed=1,
    )


CHOICES = ["toxic", "protect", "substitute", "bravebird", "switch greattusk"]


def test_status_into_statused_target_is_a_noop_unless_it_can_switch(player):
    state = _state(player, "brn")
    strict = dq.noop_choices([state], CHOICES)  # the switch-in would take the status
    assert "toxic" not in strict
    noop = dq.noop_choices([state], CHOICES, switches=False)
    assert "toxic" in noop
    for real in ("protect", "substitute", "bravebird", "switch greattusk"):
        assert real not in noop, real  # protect / substitute have effects


def test_status_move_on_healthy_target_is_not_a_noop(player):
    noop = dq.noop_choices([_state(player)], CHOICES, switches=False)
    assert "toxic" not in noop and "protect" not in noop and "substitute" not in noop


def test_noop_must_hold_in_every_world(player):
    healthy, burned = _state(player), _state(player, "brn")
    assert "toxic" not in dq.noop_choices([healthy, burned], CHOICES, switches=False)
    assert dq.noop_choices([], CHOICES) == set()


# --- player integration ---------------------------------------------------------------


def _battle():
    battle = make_battle(request_for())
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    return battle


def test_flags_default_off_and_validated(data):
    cfg = SearchConfig()
    assert not cfg.escalate and not cfg.prune_noop and cfg.tera_margin == 0.0
    with pytest.raises(ValueError):
        SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                     data=data, config=SearchConfig(tera_margin=-1.0))
    with pytest.raises(ValueError):
        SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                     data=data, config=SearchConfig(escalate_factor=0.5))


def test_escalation_reruns_longer_when_flat_and_respects_cap(data, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    calls = []
    pool = ThreadPoolExecutor(max_workers=2)
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=40, worlds=2, escalate=True, escalate_gap=2.0,
                            escalate_factor=3.0, escalate_cap_ms=10_000),
        executor=pool, seed=1,
    )
    real = p._run_worlds

    async def spy(states, time_ms):
        calls.append(time_ms)
        return await real(states, time_ms)

    monkeypatch.setattr(p, "_run_worlds", spy)
    order = asyncio.run(p.choose_move(_battle()))
    assert order is not None
    assert calls == [40, 120]  # gap 2.0 makes every root "flat"; one step of x3
    assert p.stats.escalations == 1 and p.last_changes["escalated"] == 1

    # a tiny wall-clock cap forbids the second search
    calls.clear()
    p.config = SearchConfig(search_time_ms=40, worlds=2, escalate=True, escalate_gap=2.0,
                            escalate_cap_ms=1)
    asyncio.run(p.choose_move(_battle()))
    assert calls == [40]
    pool.shutdown(wait=False, cancel_futures=True)


def test_escalation_off_never_reruns(data, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max_workers=2)
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=30, worlds=2), executor=pool, seed=1,
    )
    calls = []
    real = p._run_worlds

    async def spy(states, time_ms):
        calls.append(time_ms)
        return await real(states, time_ms)

    monkeypatch.setattr(p, "_run_worlds", spy)
    asyncio.run(p.choose_move(_battle()))
    assert calls == [30] and p.stats.escalations == 0
    pool.shutdown(wait=False, cancel_futures=True)


def test_league_spec_keys_reach_search_config():
    from showdown_bot import league

    spec = league.parse_spec("q=search:escalate=1:escalate_cap_ms=2500:prune_noop=1:tera_margin=0.03")
    assert league._quality_kwargs(spec) == {
        "escalate": True, "prune_noop": True, "escalate_cap_ms": 2500, "tera_margin": 0.03,
    }
    with pytest.raises(ValueError):
        league.parse_spec("q=search:escalate=2")
    with pytest.raises(ValueError):
        league.parse_spec("q=search:tera_margin=-1")


def test_opponent_replies_include_switches_only_in_strict_mode(player):
    state = pe.State.from_string(_state(player))
    strict = dq._opponent_replies(state)
    loose = dq._opponent_replies(state, switches=False)
    assert len(strict) > len(loose) >= 2 and "none" in loose
    assert set(loose) < set(strict)


def test_escalation_ignores_pool_queueing_delay(data, monkeypatch):
    """Queue wait inflates wall time, not compute; it must not suppress escalation."""
    from concurrent.futures import ThreadPoolExecutor

    calls = []
    pool = ThreadPoolExecutor(max_workers=2)
    p = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
        config=SearchConfig(search_time_ms=40, worlds=2, escalate=True, escalate_gap=2.0,
                            escalate_factor=3.0, escalate_cap_ms=1000),
        executor=pool, seed=1,
    )
    real = p._run_worlds

    async def queued(states, time_ms):
        calls.append(time_ms)
        if len(calls) == 1:
            await asyncio.sleep(0.9)  # simulated wait behind other games' jobs
        return await real(states, time_ms)

    monkeypatch.setattr(p, "_run_worlds", queued)
    asyncio.run(p.choose_move(_battle()))
    assert calls == [40, 120] and p.stats.escalations == 1
    pool.shutdown(wait=False, cancel_futures=True)
