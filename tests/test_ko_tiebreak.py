"""KO tie-break and material spec keys (bead showdown-bot-589.55)."""

from __future__ import annotations

import re

import pytest

pe = pytest.importorskip("poke_engine")

from test_decision_quality import _state, data, player  # noqa: E402,F401  (fixtures)

from showdown_bot import decision_quality as dq  # noqa: E402
from showdown_bot.players.search import SearchConfig, SearchPlayer  # noqa: E402


def _weak_target_state(player, hp=1):
    # Pokemon.hp is read-only: set the target's hp in the serialised state
    text = re.sub(r"(GARDEVOIR,83(?:,[A-Z]+){4},)\d+(?=,)", lambda m: m.group(1) + str(hp), _state(player), count=1)
    assert text != _state(player)
    return pe.State.from_string(text)


def _healthy(player):
    return pe.State.from_string(_state(player))


def _res(*opts):
    return (sum(o[1] for o in opts), list(opts))


CLOSE = [_res(("protect", 100, 20.8), ("bravebird", 100, 20.6))]  # means 0.208 vs 0.206
POLICY = {"protect": 0.6, "bravebird": 0.4}


def test_ko_option_detection(player):
    weak = _weak_target_state(player)
    assert dq.is_ko_option(weak, "bravebird")
    assert not dq.is_ko_option(weak, "protect")
    assert not dq.is_ko_option(weak, "switch greattusk")
    assert not dq.is_ko_option(_healthy(player), "bravebird")


def test_picks_ko_within_eps(player):
    new, pick = dq.ko_tiebreak(POLICY, CLOSE, [_weak_target_state(player)], 0.01)
    assert pick == "bravebird" and new == {"bravebird": 1.0}


def test_off_far_or_already_ko_or_no_ko(player):
    weak = _weak_target_state(player)
    assert dq.ko_tiebreak(POLICY, CLOSE, [weak], 0.0) == (POLICY, None)
    far = [_res(("protect", 100, 30.0), ("bravebird", 100, 20.6))]
    assert dq.ko_tiebreak(POLICY, far, [weak], 0.01) == (POLICY, None)
    ko_first = {"protect": 0.4, "bravebird": 0.6}
    assert dq.ko_tiebreak(ko_first, CLOSE, [weak], 0.01) == (ko_first, None)
    assert dq.ko_tiebreak(POLICY, CLOSE, [_healthy(player)], 0.01) == (POLICY, None)


def test_needs_a_majority_of_worlds(player):
    weak, healthy = _weak_target_state(player), _healthy(player)
    results = CLOSE * 3
    assert dq.ko_tiebreak(POLICY, results, [weak, healthy, healthy], 0.01)[1] is None
    assert dq.ko_tiebreak(POLICY, results, [weak, weak, healthy], 0.01)[1] == "bravebird"


def test_spec_keys_and_validation(data):
    from showdown_bot import league

    spec = league.parse_spec("n=search-net:evaluator=m.pev:material=0.05:ko_tiebreak=0.01")
    kw = league._quality_kwargs(spec)
    assert kw["material"] == 0.05 and kw["ko_tiebreak"] == 0.01
    plain = league._quality_kwargs(league.parse_spec("n=search-net:evaluator=m.pev"))
    assert "material" not in plain and "ko_tiebreak" not in plain
    for bad in ("material=1", "material=-0.1", "ko_tiebreak=-1"):
        with pytest.raises(ValueError):
            league.parse_spec(f"n=search-net:evaluator=m.pev:{bad}")
    cfg = SearchConfig()
    assert cfg.material == 0.0 and cfg.ko_tiebreak == 0.0
    with pytest.raises(ValueError):  # material needs a learned evaluator
        SearchPlayer(battle_format="gen9randombattle", start_listening=False, log_level=50,
                     data=data, config=SearchConfig(material=0.05))
