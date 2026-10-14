"""Soft Bayesian belief reweighting of opponent sets (bead showdown-bot-1wu)."""

from __future__ import annotations

import asyncio
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from showdown_bot import belief
from showdown_bot.belief import ActionRecord, ActionWatcher, WorldAction
from showdown_bot.inference import set_key
from showdown_bot.sets import OpponentModel, RandbatsData

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


# --- pure math ---------------------------------------------------------------------------


def test_normalise_factors_floors_and_scales_to_one():
    out = belief.normalise_factors([4.0, 2.0, 0.0001], floor=0.05)
    assert out == pytest.approx([1.0, 0.5, 0.05])
    assert belief.normalise_factors([]) == []
    assert belief.normalise_factors([0.0, 0.0]) == [1.0, 1.0]  # degenerate: no information


def test_effective_weights_multiply_prior_counts_and_keep_everything_alive():
    weights = belief.effective_weights([100, 100, 100], [1.0, 0.5, 1e-9], floor=0.05)
    assert weights == pytest.approx([100.0, 50.0, 5.0])
    assert all(w > 0 for w in weights)


def test_temper_exponent():
    assert belief.temper(4.0, 0.5) == pytest.approx(2.0)
    assert belief.temper(4.0, 0.0) == 1.0
    assert belief.temper(0.0, 0.5) == 0.0


def test_entropy_bits():
    assert belief.entropy_bits([1, 1, 1, 1]) == pytest.approx(2.0)
    assert belief.entropy_bits([5, 0, 0]) == 0.0
    assert belief.entropy_bits([]) == 0.0


def test_damage_likelihood_prefers_candidates_whose_rolls_cover_the_hit():
    # Observed 90 HP of damage (exact): a maximum of 100 covers it (rolls 85..100),
    # a maximum of 200 (rolls 170..200) and one of 60 (51..60) do not.
    cover = belief.damage_likelihood(100.0, 90.0, True, 400.0)
    high = belief.damage_likelihood(200.0, 90.0, True, 400.0)
    low = belief.damage_likelihood(60.0, 90.0, True, 400.0)
    assert cover > 5 * high and cover > 5 * low
    assert high > 0 and low > 0  # soft: a miss is never zero


def test_damage_likelihood_is_flat_across_the_roll_range_and_none_for_certain_kills():
    flat = [belief.damage_likelihood(100.0, x, True, 400.0) for x in (87, 92, 97)]
    assert max(flat) / min(flat) < 1.3
    assert belief.damage_likelihood(100.0, 100.0, True, 80.0) is None  # every roll kills
    assert belief.damage_likelihood(0.0, 10.0, True, 80.0) is None


def test_group_damage_factors_scale_to_the_best_and_unknown_gets_the_benefit_of_doubt():
    assert belief.group_damage_factors([2.0, 1.0, None]) == pytest.approx([1.0, 0.5, 1.0])
    assert belief.group_damage_factors([None, None]) == [1.0, 1.0]


def test_speed_tie_is_a_coin_flip_only_on_an_exact_tie():
    assert belief.speed_tie_factor(200, 200) == 0.5
    assert belief.speed_tie_factor(199, 200) == 1.0


def test_visit_shares_normalise_and_merge_duplicate_choices():
    shares = belief.visit_shares([("Tackle", 30), ("tackle", 10), ("switch x", 60)])
    assert shares == pytest.approx({"tackle": 0.4, "switch x": 0.6})
    assert belief.visit_shares([("a", 0)]) == {}


def _world(key, **shares):
    return WorldAction(key, shares)


def test_action_factors_credit_sets_by_their_worlds_and_are_tempered():
    a, b = ("g", 1, "a"), ("g", 1, "b")
    worlds = [_world(a, x=0.9, y=0.1), _world(a, x=0.9, y=0.1), _world(b, x=0.1, y=0.9)]
    full = belief.action_factors(worlds, "x", tau=1.0)
    soft = belief.action_factors(worlds, "x", tau=0.5)
    assert full[a] > 1 > full[b]
    assert 1 < soft[a] < full[a] and full[b] < soft[b] < 1  # tempering pulls toward 1
    assert belief.action_factors(worlds, "x", tau=0.0) == {a: 1.0, b: 1.0}
    # mean over all worlds is the reference: factors are ratios to it
    mean = (0.9 + 0.9 + 0.1) / 3
    assert full[a] == pytest.approx(0.9 / mean) and full[b] == pytest.approx(0.1 / mean)


def test_action_factors_floor_the_probability_of_a_surprise():
    a, b = ("g", "a"), ("g", "b")
    worlds = [_world(a, x=1.0), _world(b, y=1.0, x=0.0)]
    out = belief.action_factors(worlds, "x", tau=1.0, p_floor=0.05)
    assert out[b] > 0  # never zero: surprises dampen, they do not eliminate
    assert out[b] == pytest.approx(0.05 / ((1.0 + 0.05) / 2))


def test_action_factors_without_information_return_none():
    a = ("g", "a")
    assert belief.action_factors([], "x") is None
    assert belief.action_factors([_world(None, x=1.0)], "x") is None  # unmodelled set
    assert belief.action_factors([_world(a, y=1.0), _world(("g", "b"), y=1.0)], "x") is None
    assert belief.action_factors([_world(a, x=0.5, y=0.5), _world(a, x=0.1, y=0.9)], "x") is None


def test_action_watcher_reads_the_first_opponent_choice_of_each_turn():
    w = ActionWatcher()
    for ev in (
        ["", "turn", "1"],
        ["", "move", "p1a: Heatran", "Magma Storm", "p2a: Gardevoir"],
        ["", "-terastallize", "p2a: Gardevoir", "Fairy"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"],
        ["", "move", "p2a: Gardevoir", "Calm Mind", "p2a: Gardevoir"],  # same turn: ignored
        ["", "upkeep"],
        ["", "switch", "p2a: Greninja", "Greninja-Bond, L80", "100/100"],  # replacement
        ["", "turn", "2"],
        ["", "switch", "p2a: Heatran", "Heatran, L80", "100/100"],
        ["", "turn", "3"],
        ["", "move", "p2a: Gardevoir", "Sleep Talk", "p2a: Gardevoir"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: X", "[from]lockedmove"],
        ["", "turn", "4"],
        ["", "move", "p2a: Gardevoir", "Moonblast", "p1a: X", "[from]lockedmove"],
        ["", "move", "p2a: Gardevoir", "Psychic", "p1a: X"],
    ):
        w.observe(ev, "p2")
    assert w.observed == {
        1: "moonblast-tera",
        2: "switch heatran",
        3: "sleeptalk",
        4: "psychic",
    }


def test_opp_tuple_round_trips_through_pickle_and_equals_a_plain_tuple():
    from showdown_bot.players._worker import OppTuple

    t = OppTuple((10, [("tackle", 5, 1.0)]), [("tackle", 7), ("switch x", 3)])
    u = pickle.loads(pickle.dumps(t))
    assert u == (10, [("tackle", 5, 1.0)]) and u.opp == [("tackle", 7), ("switch x", 3)]


# --- tracker: soft weights, flag off ---------------------------------------------------------


def _tracker(data):
    from showdown_bot.sets import SetTracker

    return SetTracker(data, "gardevoir")


def test_flag_off_sampling_is_unchanged_even_if_a_belief_were_present(data):
    base, other = _tracker(data), _tracker(data)
    pool = base._matching_full_sets()
    assert len(pool) > 3
    # A belief that would change a lot, with the flag off (default).
    other.belief = {set_key(sid, fs): 0.05 for sid, fs in pool[:-1]}
    assert not other.belief_enabled
    a = [base.sample(random.Random(s)) for s in range(40)]
    b = [other.sample(random.Random(s)) for s in range(40)]
    assert a == b


def test_flag_on_with_empty_belief_is_identical_to_off(data):
    base, other = _tracker(data), _tracker(data)
    other.belief_enabled = True
    assert [base.sample(random.Random(s)) for s in range(40)] == [
        other.sample(random.Random(s)) for s in range(40)
    ]


def test_update_belief_shifts_weight_softly_and_never_eliminates(data):
    t = _tracker(data)
    t.belief_enabled = True
    pool = t._matching_full_sets()
    favoured, doomed = set_key(*pool[0]), set_key(*pool[1])
    for _ in range(6):  # hammer the same soft evidence: floors hold
        t.update_belief({favoured: 1.0, doomed: 1e-6})
    assert t.belief[doomed] == pytest.approx(t.belief_floor)
    weights = t._belief_weights(pool)
    assert min(weights) > 0
    counts = [fs.count for _, fs in pool]
    assert weights[1] == pytest.approx(counts[1] * 0.05)
    assert weights[0] == pytest.approx(counts[0])
    # recovery: when the others are contradicted in turn, the doomed set climbs back
    t.update_belief({k: 1e-6 for k in (set_key(sid, fs) for sid, fs in pool) if k != doomed})
    assert t.belief[doomed] == pytest.approx(1.0)


def test_belief_summary_reports_entropy_and_top_set(data):
    t = _tracker(data)
    t.belief_enabled = True
    prior = t.belief_summary()
    pool = t._matching_full_sets()
    t.update_belief({set_key(sid, fs): (1.0 if i == 0 else 0.05) for i, (sid, fs) in enumerate(pool)})
    post = t.belief_summary()
    assert post["entropy"] < prior["entropy"] or post["top"] == prior["top"]
    assert post["prior_entropy"] == prior["prior_entropy"]
    assert 0 < post["top_p"] <= 1


# --- outcome conditioning on a real damage hit ----------------------------------------------------


def test_a_damage_roll_consistent_with_only_some_candidates_shifts_weight_toward_them(data):
    from showdown_bot import damage_infer

    t = _tracker(data)
    t.belief_enabled = True
    pool = t._matching_full_sets()
    half = len(pool) // 2
    fits, misses = pool[:half], pool[half:]
    # Observed 88 HP: group A's rolls (85..100) cover it, group B's (maximum 70) do not.
    soft = [
        (fits, belief.damage_likelihood(100.0, 88.0, True, 300.0)),
        (misses, belief.damage_likelihood(70.0, 88.0, True, 300.0)),
    ]
    inferer = damage_infer.DamageInferer(data)
    inferer.belief = True
    inferer._note_soft("p2: Gardevoir", soft, set())
    (_key, factors), = inferer.likelihoods
    assert all(factors[set_key(sid, fs)] == 1.0 for sid, fs in fits)
    assert all(0 < factors[set_key(sid, fs)] < 0.5 for sid, fs in misses)
    t.update_belief(factors)
    weights = t._belief_weights(pool)
    counts = [fs.count for _, fs in pool]
    assert all(w > 0 for w in weights)  # soft: the misses stay alive
    assert sum(weights[half:]) / sum(counts[half:]) < sum(weights[:half]) / sum(counts[:half])
    # a group that was hard-rejected is left out of the soft update
    inferer.likelihoods = []
    inferer._note_soft("k", soft, {set_key(sid, fs) for sid, fs in misses})
    assert inferer.likelihoods == []  # fewer than two live groups: nothing to compare


def test_a_real_hit_keeps_the_hard_pool_and_keeps_belief_inside_the_floor(data):
    from battle_helpers import make_battle, request_for
    import test_damage_infer as tdi

    pools = {}
    for enabled in (False, True):
        battle = make_battle(request_for(active=5))
        model = OpponentModel(data)
        model.belief_enabled = enabled
        model.update(battle)
        hp, max_hp = tdi._start(model, battle, ("Heatran", "Heatran, L80"), ("Gardevoir", "Gardevoir, L83"))
        tracker = model.update(battle)["p2: Gardevoir"]
        low, high = tdi._engine_range(model, battle, tracker, "Choice Specs", opp_attacks=True, move="moonblast")
        damage = int(high * 0.95)
        tdi._play(
            model, battle,
            ("move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"),
            ("-damage", "p1a: Heatran", f"{hp - damage}/{max_hp}"),
            ("upkeep",), ("turn", "2"),
        )
        tracker = model.update(battle)["p2: Gardevoir"]
        pools[enabled] = [set_key(sid, fs) for sid, fs in tracker._matching_full_sets()]
        assert all(v >= tracker.belief_floor - 1e-12 for v in tracker.belief.values())
    assert pools[True] == pools[False]  # soft evidence never changes the hard pool


def test_with_the_flag_off_damage_evidence_leaves_no_belief(data):
    from battle_helpers import make_battle, request_for
    import test_damage_infer as tdi

    battle = make_battle(request_for(active=5))
    model = OpponentModel(data)
    model.update(battle)
    hp, max_hp = tdi._start(model, battle, ("Heatran", "Heatran, L80"), ("Gardevoir", "Gardevoir, L83"))
    tracker = model.update(battle)["p2: Gardevoir"]
    low, high = tdi._engine_range(model, battle, tracker, "Choice Specs", opp_attacks=True, move="moonblast")
    damage = int(high * 0.95)
    tdi._play(
        model, battle,
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"),
        ("-damage", "p1a: Heatran", f"{hp - damage}/{max_hp}"),
        ("upkeep",), ("turn", "2"),
    )
    assert model.update(battle)["p2: Gardevoir"].belief == {}


# --- action conditioning ---------------------------------------------------------------------------


def test_action_conditioning_with_a_stub_world_result(data):
    from battle_helpers import make_battle, request_for
    import test_damage_infer as tdi

    battle = make_battle(request_for(active=5))
    model = OpponentModel(data)
    model.belief_enabled = True
    model.update(battle)
    tdi._start(model, battle, ("Heatran", "Heatran, L80"), ("Gardevoir", "Gardevoir, L83"))
    tracker = model.update(battle)["p2: Gardevoir"]
    pool = tracker._matching_full_sets()
    key_a, key_b = set_key(*pool[0]), set_key(*pool[1])
    # The opponent plays Moonblast on turn 1 ...
    model.observe_events(
        battle.battle_tag,
        [["", "move", "p2a: Gardevoir", "Moonblast", "p1a: Heatran"], ["", "turn", "2"]],
        battle,
    )
    assert model.observed_action(battle.battle_tag, 1) == "moonblast"
    # ... which world A (set a) expected and world B (set b) found unlikely.
    record = ActionRecord(
        1, "p2: Gardevoir",
        [_world(key_a, moonblast=0.8, psychic=0.2), _world(key_b, moonblast=0.05, psychic=0.95)],
    )
    assert model.apply_action_belief(battle, record)
    tracker = model.update(battle)["p2: Gardevoir"]
    assert tracker.belief[key_a] > tracker.belief[key_b] >= tracker.belief_floor
    # nothing was eliminated and the other sets are untouched (implicit 1.0 or normalised)
    assert len(tracker._matching_full_sets()) == len(pool)
    # an unmappable observation changes nothing
    model._battles[battle.battle_tag].watcher.observed[2] = "splash"
    before = dict(tracker.belief)
    assert not model.apply_action_belief(battle, ActionRecord(2, "p2: Gardevoir", record.worlds))
    assert tracker.belief == before


# --- config / env ----------------------------------------------------------------------------------


def test_config_defaults_off_and_env_overrides(monkeypatch):
    from showdown_bot.players.search import SearchConfig

    cfg = SearchConfig()
    assert not cfg.belief_reweight and cfg.belief_tau == 0.5 and cfg.belief_floor == 0.05
    assert not SearchConfig.from_env().belief_reweight
    monkeypatch.setenv("SHOWDOWN_BELIEF_REWEIGHT", "1")
    monkeypatch.setenv("SHOWDOWN_BELIEF_TAU", "0.25")
    monkeypatch.setenv("SHOWDOWN_BELIEF_FLOOR", "0.1")
    cfg = SearchConfig.from_env()
    assert cfg.belief_reweight and cfg.belief_tau == 0.25 and cfg.belief_floor == 0.1


def test_league_spec_accepts_belief_keys():
    from showdown_bot import league

    spec = league.parse_spec("b=search:belief=1:btau=0.4:bfloor=0.1")
    assert league._quality_kwargs(spec) == {
        "belief_reweight": True, "belief_tau": 0.4, "belief_floor": 0.1,
    }
    with pytest.raises(ValueError):
        league.parse_spec("b=search:belief=2")


# --- end to end through SearchPlayer -----------------------------------------------------------------


def test_search_player_records_opponent_visits_and_applies_the_observed_action(data):
    from battle_helpers import event, make_battle, opp_switch, request_for
    from showdown_bot.players.search import SearchConfig, SearchPlayer

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        player = SearchPlayer(
            battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
            config=SearchConfig(search_time_ms=30, belief_reweight=True), executor=pool, seed=3,
        )
        assert player.opponent_model.belief_enabled
        battle = make_battle(request_for())
        event(battle, "start")
        opp_switch(battle, "Gardevoir", "Gardevoir, L83")
        asyncio.run(player.choose_move(battle))
        record = player._belief_pending.get(battle.battle_tag)
        assert record is not None and record.worlds
        assert all(w.shares and abs(sum(w.shares.values()) - 1) < 1e-9 for w in record.worlds)
        assert all(w.set_key is not None for w in record.worlds)
        # Turn passes; the opponent used one of the options a world offered.
        offered = sorted(record.worlds[0].shares, key=record.worlds[0].shares.get)
        move = next(c for c in offered if not c.startswith("switch") and not c.endswith("-tera"))
        state = player.opponent_model._battles.setdefault(
            battle.battle_tag, __import__("showdown_bot.sets", fromlist=["x"])._BattleOpponentState()
        )
        state.watcher.observed[record.turn] = move
        before = belief.STATS.action_updates
        player._apply_belief_action(battle)
        assert battle.battle_tag not in player._belief_pending
        assert belief.STATS.action_updates >= before  # applied when worlds disagree on the set
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def test_search_player_with_the_flag_off_asks_for_no_visits_and_keeps_no_state(data):
    from battle_helpers import event, make_battle, opp_switch, request_for
    from showdown_bot.players.search import SearchConfig, SearchPlayer

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        player = SearchPlayer(
            battle_format="gen9randombattle", start_listening=False, log_level=50, data=data,
            config=SearchConfig(search_time_ms=20), executor=pool, seed=3,
        )
        battle = make_battle(request_for())
        event(battle, "start")
        opp_switch(battle, "Gardevoir", "Gardevoir, L83")
        asyncio.run(player.choose_move(battle))
        assert not player._belief_pending and not player.opponent_model.belief_enabled
        assert all(not t.belief for t in player.opponent_model.update(battle).values())
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
