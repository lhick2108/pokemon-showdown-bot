"""Illusion (Zoroark) suspicion: protocol-replay tests plus world sampling (589.9)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from poke_env.battle.battle import Battle

from showdown_bot import illusion
from showdown_bot.sets import IllusionEstimate, OpponentModel, RandbatsData

FIXTURES = Path(__file__).parent / "fixtures"
ARCANINE = ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100")


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _battle(tag="battle-gen9randombattle-illusion"):
    battle = Battle(tag, username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    return battle


def _replay(data, *events, model=None, battle=None):
    battle = battle or _battle()
    model = model or OpponentModel(data)
    model.update(battle)
    for parts in events:
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        model.update(battle)
    return model, battle


def _gardevoir(hp="100/100"):
    return ("switch", "p2a: Gardevoir", "Gardevoir, L83", hp)


def _total(model, battle):
    return sum(model.illusion_probability(battle).values())


# --- the four acceptance cases ---------------------------------------------------


def test_dark_pulse_from_gardevoir_is_high_suspicion_for_zoroark(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
    )
    probs = model.illusion_probability(battle)
    assert probs["zoroark"] == pytest.approx(illusion.KIT_SUSPECT_P)
    assert probs["zoroarkhisui"] == 0.0  # Zoroark-Hisui has no Dark Pulse


def test_psychic_move_failing_against_gardevoir_is_certainly_zoroark(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p1a: Arcanine", "Psychic", "p2a: Gardevoir"),
        ("-immune", "p2a: Gardevoir"),
    )
    assert model.illusion_probability(battle) == {"zoroark": 1.0, "zoroarkhisui": 0.0}


def test_species_clause_duplicate_with_a_proven_real_twin_is_certain(data):
    """Gardevoir showed Moonblast (impossible for Zoroark), left at 60%, and a
    Gardevoir later comes back at 30%: nothing heals on the bench, so these
    are two Pokémon and the second must be the Zoroark."""
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Moonblast", "p1a: Arcanine"),
        ("-damage", "p2a: Gardevoir", "60/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        _gardevoir("30/100"),
    )
    probs = model.illusion_probability(battle)
    assert sum(probs.values()) == pytest.approx(illusion.DUPLICATE_EARLIER_REAL_P)


def test_species_clause_duplicate_without_a_tell_is_a_coin_flip(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("-damage", "p2a: Gardevoir", "60/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        _gardevoir("30/100"),
    )
    assert _total(model, battle) == pytest.approx(illusion.DUPLICATE_AMBIGUOUS_P)


def test_a_gardevoir_returning_after_fainting_is_the_real_one(data):
    """Illusion ends silently when its user faints, so the fainted 'Gardevoir'
    was the Zoroark."""
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("faint", "p2a: Gardevoir"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        _gardevoir("50/100"),
    )
    assert model.illusion_probability(battle) == {}


def test_returning_at_the_same_hp_is_not_a_duplicate(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("-damage", "p2a: Gardevoir", "60/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        _gardevoir("60/100"),
    )
    assert _total(model, battle) < illusion.DUPLICATE_AMBIGUOUS_P


def test_no_zoroark_possible_gives_zero(data):
    full_team = [
        ("switch", "p2a: Abomasnow", "Abomasnow, L85", "100/100"),
        ("switch", "p2a: Azumarill", "Azumarill, L84", "100/100"),
        ("switch", "p2a: Espeon", "Espeon, L85", "100/100"),
        ("switch", "p2a: Great Tusk", "Great Tusk, L78", "100/100"),
        ("switch", "p2a: Iron Hands", "Iron Hands, L82", "100/100"),
        _gardevoir(),
    ]
    model, battle = _replay(
        data,
        ARCANINE,
        *full_team,
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
    )
    assert len(battle.opponent_team) == 6
    assert model.illusion_probability(battle) == {}


def test_revealed_and_fainted_zoroark_gives_zero(data):
    model, battle = _replay(
        data,
        ARCANINE,
        ("switch", "p2a: Zoroark", "Zoroark, L83", "100/100"),
        ("faint", "p2a: Zoroark"),
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
    )
    assert model.illusion_probability(battle) == {}


def test_the_active_zoroark_itself_gives_zero(data):
    model, battle = _replay(
        data, ARCANINE, ("switch", "p2a: Zoroark", "Zoroark, L83", "100/100")
    )
    assert model.illusion_probability(battle) == {}


# --- immunity tells ------------------------------------------------------------


@pytest.mark.parametrize("move", ["Shadow Ball", "Hyper Voice", "Close Combat"])
def test_zoroark_hisui_is_immune_to_ghost_normal_and_fighting(data, move):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p1a: Arcanine", move, "p2a: Gardevoir"),
        ("-immune", "p2a: Gardevoir"),
    )
    # Pure Dark Zoroark takes neutral/super effective damage from all three.
    assert model.illusion_probability(battle) == {"zoroark": 0.0, "zoroarkhisui": 1.0}


def test_an_immunity_the_shown_species_really_has_teaches_nothing(data):
    """Fairy is immune to Dragon, so Gardevoir shrugging off Dragon Claw is
    ordinary and must not look like a tell."""
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p1a: Arcanine", "Dragon Claw", "p2a: Gardevoir"),
        ("-immune", "p2a: Gardevoir"),
    )
    assert _total(model, battle) < 0.5


def test_ability_immunity_is_not_a_type_tell(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p1a: Arcanine", "Psychic", "p2a: Gardevoir"),
        ("-immune", "p2a: Gardevoir", "[from] ability: Levitate"),
    )
    assert _total(model, battle) < 0.5


# --- negative tells, resets, prior -----------------------------------------------


def test_a_direct_hit_that_does_not_break_illusion_means_real(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
        ("-damage", "p2a: Gardevoir", "40/100"),
    )
    assert model.illusion_probability(battle) == {}


def test_killing_a_substitute_does_not_prove_real(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
        ("-activate", "p2a: Gardevoir", "Substitute", "[damage]"),
    )
    assert _total(model, battle) == pytest.approx(illusion.KIT_SUSPECT_P)


def test_suspicion_resets_after_replace_and_on_each_new_switch_in(data):
    model, battle = _replay(
        data,
        ARCANINE,
        _gardevoir(),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
    )
    assert _total(model, battle) == pytest.approx(illusion.KIT_SUSPECT_P)

    for parts in (
        ("move", "p1a: Arcanine", "Flare Blitz", "p2a: Gardevoir"),
        ("-damage", "p2a: Gardevoir", "40/100"),
        ("replace", "p2a: Zoroark", "Zoroark, L83", "40/100"),
        ("-end", "p2a: Zoroark", "Illusion"),
    ):
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        model.update(battle)
    assert model.illusion_probability(battle) == {}  # identity known

    _replay(
        data,
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        model=model,
        battle=battle,
    )
    probs = model.illusion_probability(battle)
    # The living, revealed Zoroark can re-disguise: only the base rate remains.
    assert 0 < sum(probs.values()) < 0.5
    assert set(probs) == {"zoroark"}  # Species Clause: no Zoroark-Hisui too


def test_no_evidence_gives_only_the_small_prior(data):
    model, battle = _replay(data, ARCANINE, _gardevoir())
    probs = model.illusion_probability(battle)
    assert 0 < sum(probs.values()) < 0.1
    assert set(probs) == {"zoroark", "zoroarkhisui"}


def test_base_rate_formula():
    # q = 5%, 5 unrevealed slots, 6 alive candidates.
    assert illusion.base_rate({"zoroark": 5.0}, 100.0, 5, 6) == pytest.approx(
        (1 - 0.95**5) / 6
    )
    assert illusion.base_rate({"zoroark": 5.0}, 100.0, 0, 6) == 0.0
    assert illusion.base_rate({"zoroark": 5.0}, 100.0, 0, 4, "zoroark") == 0.25


def test_real_kit_tell_and_disguised_tell_together_fall_back_to_the_prior():
    evidence = illusion.IllusionEvidence(
        shown_species="gardevoir",
        form_weights={"zoroark": 1.0},
        prior=0.03,
        kit_suspect=True,
        kit_forms={"zoroark"},
        kit_real=True,
    )
    assert illusion.estimate(evidence) == {"zoroark": 0.03}
    real_only = illusion.IllusionEvidence(
        shown_species="gardevoir", form_weights={"zoroark": 1.0}, prior=0.03, kit_real=True
    )
    assert illusion.estimate(real_only) == {"zoroark": 0.0}


# --- world sampling ------------------------------------------------------------


def _sampling_setup(data):
    from showdown_bot import engine_state
    from battle_helpers import event, make_battle, opp_switch

    battle = make_battle()
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    model = OpponentModel(data)
    trackers = model.update(battle)
    return engine_state, battle, model, trackers


def test_world_sampling_uses_zoroark_at_the_estimated_rate(data):
    pytest.importorskip("poke_engine")
    engine_state, battle, model, trackers = _sampling_setup(data)
    sampler = engine_state.TeamSampler(data)
    estimate = IllusionEstimate({"zoroark": 0.3, "zoroarkhisui": 0.1}, {})
    illusion_rng = random.Random(1234)
    counts = {"zoroark": 0, "zoroarkhisui": 0, "gardevoir": 0}
    n = 4000
    for i in range(n):
        slots = engine_state.sample_opponent_team(
            battle, trackers, sampler, random.Random(i),
            illusion=estimate, illusion_rng=illusion_rng,
        )
        counts[slots[0].species_id] += 1
        assert len(slots) == 6
        if slots[0].disguised:
            assert slots[0].mon is battle.opponent_active_pokemon
            # The copied Pokémon is on the bench, not revealed on the field.
            assert any(s.species_id == "gardevoir" and s.mon is None for s in slots[1:])
            assert sum(s.species_id.startswith("zoroark") for s in slots) == 1
    assert counts["zoroark"] / n == pytest.approx(0.3, abs=0.03)
    assert counts["zoroarkhisui"] / n == pytest.approx(0.1, abs=0.02)
    assert counts["gardevoir"] / n == pytest.approx(0.6, abs=0.03)


def test_world_sampling_is_unchanged_when_probability_is_zero(data):
    pytest.importorskip("poke_engine")
    engine_state, battle, model, trackers = _sampling_setup(data)
    sampler = engine_state.TeamSampler(data)

    def species(slots):
        return [(s.species_id, s.sampled.moves, s.sampled.item) for s in slots]

    plain = engine_state.sample_opponent_team(battle, trackers, sampler, random.Random(7))
    zero = engine_state.sample_opponent_team(
        battle, trackers, sampler, random.Random(7),
        illusion=IllusionEstimate({"zoroark": 0.0}, {}), illusion_rng=random.Random(0),
    )
    assert species(plain) == species(zero)


def test_disguised_world_builds_an_engine_state_with_zoroark_types(data):
    pytest.importorskip("poke_engine")
    engine_state, battle, model, trackers = _sampling_setup(data)
    sampler = engine_state.TeamSampler(data)
    estimate = IllusionEstimate({"zoroark": 1.0}, {})
    slots = engine_state.sample_opponent_team(
        battle, trackers, sampler, random.Random(0),
        illusion=estimate, illusion_rng=random.Random(0),
    )
    built = engine_state.build_state(battle, slots)
    active = built.state.side_two.pokemon[0]
    assert active.id.lower() == "zoroark"
    assert "dark" in tuple(str(t).lower() for t in active.types)
    assert "psychic" not in tuple(str(t).lower() for t in active.types)


def test_search_player_samples_zoroark_worlds_at_the_computed_rate(data):
    pytest.importorskip("poke_engine")
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from battle_helpers import event, make_battle, opp_switch
    from showdown_bot.players.search import SearchConfig, SearchPlayer

    pool = ThreadPoolExecutor(max_workers=2)
    player = SearchPlayer(
        battle_format="gen9randombattle", start_listening=False, log_level=50,
        data=data, config=SearchConfig(search_time_ms=10, worlds=40), executor=pool, seed=3,
    )
    battle = make_battle()
    event(battle, "start")
    opp_switch(battle, "Gardevoir", "Gardevoir, L83")
    for parts in (
        ("move", "p1a: Great Tusk", "Psychic", "p2a: Gardevoir"),
        ("-immune", "p2a: Gardevoir"),
    ):
        player.opponent_model.observe(battle.battle_tag, ["", *parts])
        battle.parse_message(["", *parts])
    player.opponent_model.update(battle)
    assert player.opponent_model.illusion_probability(battle)["zoroark"] == 1.0
    asyncio.run(player.choose_move(battle))
    pool.shutdown(wait=False, cancel_futures=True)
    assert player.stats.illusion_worlds == 40

    # No evidence at all: the stat counts only the rare prior-rate worlds.
    clean = make_battle(tag="battle-gen9randombattle-clean")
    event(clean, "start")
    opp_switch(clean, "Gardevoir", "Gardevoir, L83")
    player.opponent_model.update(clean)
    probs = player.opponent_model.illusion_probability(clean)
    assert 0 < sum(probs.values()) < 0.1
