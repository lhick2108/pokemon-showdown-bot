import asyncio
import random
import urllib.request
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle

from showdown_bot.sets import (
    OpponentModel,
    OpponentModelPlayerMixin,
    RandbatsData,
    SetTracker,
    _details_tera,
    _hidden_forme_species,
    _resolve_randbats_species,
    to_id,
)


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def data():
    return RandbatsData.load(cache_dir=FIXTURES)


def _battle(tag="battle-gen9randombattle-sets"):
    battle = Battle(tag, username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    return battle


def _event(battle, *parts):
    battle.parse_message(["", *parts])


def _switch(battle, pokemon, details):
    _event(battle, "switch", pokemon, details, "100/100")


def _replay(battle, model, *events):
    """Feed actual split protocol records to poke-env and the event model."""
    trackers = model.update(battle)
    for parts in events:
        event = ["", *parts]
        model.observe(battle.battle_tag, event)
        battle.parse_message(event)
        trackers = model.update(battle)
    return trackers


def _double_battle(tag):
    battle = DoubleBattle(tag, username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    return battle


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Great Tusk", "greattusk"),
        ("Close Combat", "closecombat"),
        ("Heavy-Duty Boots", "heavydutyboots"),
        ("Urshifu-Rapid-Strike", "urshifurapidstrike"),
        (" Mr. Mime! ", "mrmime"),
        ("", ""),
    ],
)
def test_to_id_uses_showdown_normalization(name, expected):
    assert to_id(name) == expected


def test_fixture_contains_real_randbats_species_and_role_weights(data):
    assert data.species_ids() == (
        "abomasnow",
        "azumarill",
        "calyrexice",
        "calyrexshadow",
        "carbink",
        "chandelure",
        "ditto",
        "espeon",
        "gardevoir",
        "greattusk",
        "greninja",
        "greninjabond",
        "ironhands",
        "ninetales",
        "ninetalesalola",
        "ogerpon",
        "ogerponcornerstone",
        "ogerponhearthflame",
        "ogerponwellspring",
        "palafin",
        "slowbrogalar",
        "terapagos",
        "urshifu",
        "urshifurapidstrike",
        "zoroark",
        "zoroarkhisui",
    )
    assert data.level("Urshifu-Rapid-Strike") == 75

    role = data.roles("Abomasnow")["Bulky Support"]
    assert role.moves[:2] == ("Aurora Veil", "Blizzard")
    assert role.option_weight("moves", "Aurora Veil") == pytest.approx(1.0)
    assert role.option_weight("moves", "Earthquake") == pytest.approx(0.6828)


@pytest.mark.parametrize(
    ("species", "kind", "observation", "expected_role"),
    [
        ("Palafin", "move", "Flip Turn", "Wallbreaker"),
        ("Palafin", "item", "Choice Band", "Wallbreaker"),
        ("Ogerpon", "move", "Swords Dance", "Setup Sweeper"),
        ("Great Tusk", "move", "Stealth Rock", "Bulky Support"),
        ("Great Tusk", "item", "Booster Energy", "Fast Bulky Setup"),
        ("Chandelure", "move", "Calm Mind", "Fast Bulky Setup"),
    ],
)
def test_observations_filter_using_real_role_options(
    data, species, kind, observation, expected_role
):
    tracker = SetTracker(data, species)
    getattr(tracker, f"observe_{kind}")(observation)

    assert tracker.candidate_roles() == {expected_role: 1.0}
    assert not tracker.inconsistent


def test_unknown_item_sentinel_is_not_set_evidence(data):
    tracker = SetTracker(data, "Great Tusk")
    original_candidates = tracker.candidate_roles()

    tracker.observe_item("unknown_item")

    assert tracker._revealed_item is None
    assert tracker.candidate_roles() == original_candidates
    assert not tracker.inconsistent


def test_inconsistent_observations_keep_least_conflicting_roles_and_warn(data, caplog):
    tracker = SetTracker(data, "Great Tusk")

    tracker.observe_move("Not a Great Tusk move")

    assert tracker.inconsistent
    assert tracker.candidate_roles()
    assert "conflict with all randbats roles" in caplog.text


def test_role_sampling_uses_real_stats_and_produces_four_legal_moves(data):
    """Exercises ``_sample_role_based`` directly: the real per-role marginal
    fallback used when no full set matches binding evidence. ``sample()``
    itself now prefers full joint-set sampling (589.11) whenever real full
    data is available, which this fixture's Great Tusk has, so calling
    ``sample()`` here would test the full-set path instead.
    """
    tracker = SetTracker(data, "Great Tusk")
    expected = {
        role_name: role.weight
        for role_name, role in data.roles("Great Tusk").items()
    }
    counts = dict.fromkeys(expected, 0)
    rng = random.Random(8128)

    for _ in range(5000):
        sampled = tracker._sample_role_based(rng)
        counts[sampled.role] += 1
        role = data.roles("Great Tusk")[sampled.role]
        move_ids = [to_id(move) for move in sampled.moves]
        assert len(sampled.moves) == min(4, len(role.moves))
        assert len(move_ids) == len(set(move_ids))
        assert set(move_ids) <= {to_id(move) for move in role.moves}

    for role_name, weight in expected.items():
        assert counts[role_name] / 5000 == pytest.approx(weight, abs=0.03)


@pytest.mark.parametrize(
    "revealed_moves", [(), ("Earthquake",), ("Earthquake", "Ice Shard")]
)
def test_move_samples_have_exact_count_with_unrevealed_and_partial_evidence(
    data, revealed_moves
):
    """Exercises the ``_sample_role_based`` fallback directly; see the
    docstring on ``test_role_sampling_uses_real_stats_and_produces_four_legal_moves``.
    """
    tracker = SetTracker(data, "Abomasnow")
    role = data.roles("Abomasnow")["Bulky Support"]
    for move in revealed_moves:
        tracker.observe_move(move)
    rng = random.Random(773)
    mandatory = {"auroraveil", "blizzard"}

    for _ in range(100):
        sampled = tracker._sample_role_based(rng)
        sampled_ids = {to_id(move) for move in sampled.moves}
        assert len(sampled.moves) == min(4, len(role.moves))
        assert len(sampled_ids) == len(sampled.moves)
        assert sampled_ids <= {to_id(move) for move in role.moves}
        assert mandatory <= sampled_ids
        assert {to_id(move) for move in revealed_moves} <= sampled_ids


def test_abomasnow_move_inclusion_rates_match_stats_and_keep_mandatory_moves(data):
    """Exercises the ``_sample_role_based`` fallback directly; see the
    docstring on ``test_role_sampling_uses_real_stats_and_produces_four_legal_moves``.
    """
    tracker = SetTracker(data, "Abomasnow")
    role = data.roles("Abomasnow")["Bulky Support"]
    counts = dict.fromkeys((to_id(move) for move in role.moves), 0)
    rng = random.Random(20261010)
    mandatory = {"auroraveil", "blizzard"}

    for _ in range(5000):
        sampled = tracker._sample_role_based(rng)
        sampled_ids = {to_id(move) for move in sampled.moves}
        assert len(sampled.moves) == 4
        assert mandatory <= sampled_ids
        for move_id in sampled_ids:
            counts[move_id] += 1

    for move in role.moves:
        move_id = to_id(move)
        assert counts[move_id] / 5000 == pytest.approx(
            role.option_weight("moves", move), abs=0.05
        )


def test_revealed_only_sampler_mutation_breaks_the_move_count_contract(
    data, monkeypatch
):
    tracker = SetTracker(data, "Abomasnow")
    role = data.roles("Abomasnow")["Bulky Support"]
    monkeypatch.setattr(
        SetTracker,
        "_sample_moves",
        lambda self, role, rng: tuple(self._revealed_moves),
    )

    # This is the prior broken implementation expressed as a mutation. The
    # same exact-count assertion used for real samples rejects it.
    # (Directly exercises the role-based fallback: see the docstring on
    # test_role_sampling_uses_real_stats_and_produces_four_legal_moves.)
    with pytest.raises(AssertionError):
        assert len(
            tracker._sample_role_based(random.Random(19)).moves
        ) == min(4, len(role.moves))


def test_nonuniform_ability_item_and_tera_sampling_matches_stats(data):
    """Exercises the ``_sample_role_based`` fallback directly; see the
    docstring on ``test_role_sampling_uses_real_stats_and_produces_four_legal_moves``.
    """
    tracker = SetTracker(data, "Chandelure")
    tracker.observe_move("Calm Mind")
    role = data.roles("Chandelure")["Fast Bulky Setup"]
    options = {
        "ability": ("abilities", role.abilities),
        "item": ("items", role.items),
        "tera_type": ("teraTypes", role.tera_types),
    }
    counts = {
        result_field: dict.fromkeys((to_id(option) for option in role_options), 0)
        for result_field, (_, role_options) in options.items()
    }
    rng = random.Random(50117)

    for _ in range(5000):
        sampled = tracker._sample_role_based(rng)
        assert len(sampled.moves) == min(4, len(role.moves))
        for result_field in options:
            counts[result_field][to_id(getattr(sampled, result_field))] += 1

    for result_field, (kind, role_options) in options.items():
        assert len(role_options) > 1
        for option in role_options:
            assert counts[result_field][to_id(option)] / 5000 == pytest.approx(
                role.option_weight(kind, option), abs=0.03
            )


def _full_set_identity(full_set):
    """A hashable identity for a FullSet/SampledSet, ignoring display casing."""
    return (
        full_set.level,
        to_id(full_set.item) if full_set.item else None,
        to_id(full_set.ability),
        frozenset(to_id(move) for move in full_set.moves),
        to_id(full_set.tera_type),
    )


def test_full_set_sampling_returns_only_real_full_sets_with_binding_evidence(data):
    """``sample()`` now draws whole sets jointly from pkmn/randbats' real
    "full" data (589.11) instead of drawing each option independently.
    Every sampled set must appear verbatim (as a multiset of options) in the
    real full data for the species, and must contain all binding evidence.
    """
    tracker = SetTracker(data, "Great Tusk")
    tracker.observe_move("Close Combat")
    legal_identities = {
        _full_set_identity(full_set)
        for full_set in data.full_sets("Great Tusk")
        if "closecombat" in {to_id(move) for move in full_set.moves}
    }
    assert legal_identities

    rng = random.Random(42)
    for _ in range(300):
        sampled = tracker.sample(rng)
        assert sampled.species == "Great Tusk"
        assert "closecombat" in {to_id(move) for move in sampled.moves}
        assert _full_set_identity(sampled) in legal_identities
        # The full set determines a real role whenever one could have
        # generated it (see _best_matching_role_name); it always can here.
        assert sampled.role in data.roles("Great Tusk")


def test_full_set_sampling_matches_real_set_frequencies_with_no_evidence(data):
    """With no evidence, the top sets for a species must appear with their
    real relative frequency (to within 3 percentage points over 5,000
    samples), not the per-move independent rates the old sampler used.
    """
    tracker = SetTracker(data, "Abomasnow")
    full_sets = data.full_sets("Abomasnow")
    total = sum(full_set.count for full_set in full_sets)
    top_three = sorted(full_sets, key=lambda fs: fs.count, reverse=True)[:3]
    expected = {_full_set_identity(fs): fs.count / total for fs in top_three}

    rng = random.Random(20261010)
    counts = Counter(
        _full_set_identity(tracker.sample(rng)) for _ in range(5000)
    )

    for identity, expected_fraction in expected.items():
        assert counts[identity] / 5000 == pytest.approx(expected_fraction, abs=0.03)


def test_full_set_sampling_with_one_revealed_move_draws_only_matching_sets(data):
    """Evidence must restrict sampling to full sets containing it, in the
    correct (real, renormalized) proportions -- not just any role-legal set.
    """
    tracker = SetTracker(data, "Abomasnow")
    tracker.observe_move("Earthquake")
    matching = [
        full_set
        for full_set in data.full_sets("Abomasnow")
        if "earthquake" in {to_id(move) for move in full_set.moves}
    ]
    assert len(matching) >= 2
    total = sum(full_set.count for full_set in matching)
    expected = {_full_set_identity(fs): fs.count / total for fs in matching}

    rng = random.Random(773)
    counts = Counter()
    for _ in range(5000):
        sampled = tracker.sample(rng)
        assert "earthquake" in {to_id(move) for move in sampled.moves}
        identity = _full_set_identity(sampled)
        assert identity in expected
        counts[identity] += 1

    for identity, expected_fraction in expected.items():
        assert counts[identity] / 5000 == pytest.approx(expected_fraction, abs=0.03)


def test_full_set_sampling_never_produces_a_move_combo_absent_from_full_data(data):
    """Mutation check: Abomasnow's real full data always carries exactly two
    of {Earthquake, Ice Shard, Wood Hammer}, never one and never all three.
    A per-move independent sampler can (and did) produce both of those
    forbidden combinations; joint full-set sampling never can, because every
    sample is copied verbatim from a real set.
    """
    tracker = SetTracker(data, "Abomasnow")
    correlated = {"earthquake", "iceshard", "woodhammer"}
    rng = random.Random(99)

    for _ in range(2000):
        sampled = tracker.sample(rng)
        move_ids = {to_id(move) for move in sampled.moves}
        assert len(move_ids & correlated) == 2


def test_full_set_sampling_mixes_hidden_forme_siblings_in_proportion(data):
    """An unrevealed Greninja must sample from both the Greninja and
    Greninja-Bond full sets (see ``_hidden_forme_species``), weighted by
    their real relative counts; once Battle Bond is revealed, only
    Greninja-Bond sets may appear.
    """
    tracker = SetTracker(
        data, "Greninja", sibling_species=_hidden_forme_species(data, "greninja")
    )
    greninja_total = sum(fs.count for fs in data.full_sets("Greninja"))
    bond_total = sum(fs.count for fs in data.full_sets("Greninja-Bond"))
    expected = {
        "Greninja": greninja_total / (greninja_total + bond_total),
        "Greninja-Bond": bond_total / (greninja_total + bond_total),
    }

    rng = random.Random(2024)
    species_counts = Counter(tracker.sample(rng).species for _ in range(5000))
    for species, expected_fraction in expected.items():
        assert species_counts[species] / 5000 == pytest.approx(
            expected_fraction, abs=0.03
        )

    tracker.observe_ability("Battle Bond")
    for _ in range(200):
        sampled = tracker.sample(rng)
        assert sampled.species == "Greninja-Bond"
        assert to_id(sampled.ability) == "battlebond"


def test_full_set_sampling_falls_back_to_role_based_when_no_exact_set_matches(data):
    """Choice Scarf and Stealth Rock are each legal for Great Tusk's "Bulky
    Support" role (so role elimination stays consistent), but no single real
    full set pairs them -- a case 589.11 says must fall back gracefully to
    the role-based sampler rather than crash or return illegal evidence.
    """
    tracker = SetTracker(data, "Great Tusk")
    tracker.observe_item("Choice Scarf")
    tracker.observe_move("Stealth Rock")
    assert not tracker.inconsistent
    assert tracker._matching_full_sets() == []

    rng = random.Random(0)
    for _ in range(50):
        sampled = tracker.sample(rng)
        assert sampled.role in data.roles("Great Tusk")
        assert to_id(sampled.item) == "choicescarf"
        assert "stealthrock" in {to_id(move) for move in sampled.moves}
        assert len(sampled.moves) == 4


def test_pokedex_resolves_temporary_and_preserves_selectable_forms(data):
    assert _resolve_randbats_species(data, "Palafin-Hero") == "palafin"
    assert (
        _resolve_randbats_species(data, "Ogerpon-Wellspring-Tera")
        == "ogerponwellspring"
    )
    assert (
        _resolve_randbats_species(data, "Urshifu-Rapid-Strike")
        == "urshifurapidstrike"
    )


def test_fresh_real_great_tusk_unknown_item_does_not_eliminate_roles(data):
    battle = _battle("battle-unknown-item")
    _switch(battle, "p2a: Great Tusk", "Great Tusk, L77")
    pokemon = battle.opponent_team["p2: Great Tusk"]
    assert pokemon.item == "unknown_item"

    trackers = OpponentModel(data).update(battle)

    tracker = trackers["p2: Great Tusk"]
    assert tracker._revealed_item is None
    assert len(tracker.candidate_roles()) == 3
    assert not tracker.inconsistent


def test_palafin_hero_switch_reuses_original_palafin_tracker(data):
    battle = _battle("battle-palafin-hero")
    model = OpponentModel(data)
    first = _replay(
        battle,
        model,
        ("switch", "p2a: Palafin", "Palafin, L77", "100/100"),
    )["p2: Palafin"]

    _replay(
        battle,
        model,
        ("switch", "p2a: Great Tusk", "Great Tusk, L77", "100/100"),
        ("switch", "p2a: Palafin", "Palafin-Hero, L77", "100/100"),
    )
    second = model.update(battle)["p2: Palafin"]

    assert battle.opponent_team["p2: Palafin"].species == "palafinhero"
    assert second is first
    assert second.species == "Palafin"


def test_gardevoir_trace_does_not_replace_original_ability(data):
    battle = _battle("battle-trace")
    model = OpponentModel(data)
    _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        (
            "-ability",
            "p2a: Gardevoir",
            "Intimidate",
            "Trace",
            "[from] ability: Trace",
            "[of] p1a: Arcanine",
        ),
    )

    pokemon = battle.opponent_team["p2: Gardevoir"]
    assert pokemon.ability == "intimidate"
    assert pokemon._ability == "trace"
    tracker = model.update(battle)["p2: Gardevoir"]
    assert tracker._revealed_ability == "trace"


def test_transformed_ditto_records_transform_but_not_copied_moves(data):
    battle = _battle("battle-transform")
    model = OpponentModel(data)
    _replay(
        battle,
        model,
        ("switch", "p1a: Great Tusk", "Great Tusk, L77", "100/100"),
        ("switch", "p2a: Ditto", "Ditto, L87", "100/100"),
        ("move", "p2a: Ditto", "Transform", "p1a: Great Tusk"),
        (
            "-transform",
            "p2a: Ditto",
            "p1a: Great Tusk",
            "[from] move: Transform",
        ),
        ("move", "p2a: Ditto", "Earthquake", "p1a: Great Tusk"),
    )

    pokemon = battle.opponent_team["p2: Ditto"]
    assert pokemon.transformed
    assert "earthquake" in pokemon.moves
    tracker = model.update(battle)["p2: Ditto"]

    assert {to_id(move) for move in tracker.revealed_moves} == {"transform"}
    assert tracker._revealed_ability is None
    assert not tracker.inconsistent


def test_imposter_transform_event_reveals_its_original_ability(data):
    battle = _battle("battle-imposter-transform")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Great Tusk", "Great Tusk, L77", "100/100"),
        ("switch", "p2a: Ditto", "Ditto, L87", "100/100"),
        (
            "-transform",
            "p2a: Ditto",
            "p1a: Great Tusk",
            "[from] ability: Imposter",
        ),
    )

    assert trackers["p2: Ditto"]._revealed_ability == "imposter"


def test_mimic_and_called_moves_do_not_add_borrowed_move_evidence(data):
    battle = _battle("battle-mimic")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", "Mimic", "p1a: Arcanine"),
        ("-start", "p2a: Gardevoir", "Mimic", "Earthquake"),
        ("move", "p2a: Gardevoir", "Earthquake", "p1a: Arcanine"),
    )

    assert {to_id(move) for move in trackers["p2: Gardevoir"].revealed_moves} == {"mimic"}


def test_struggle_is_never_recorded_as_set_evidence(data):
    """Struggle (data/moves.ts) is forced on any Pokemon with no usable move
    (no PP, fully disabled, etc.) regardless of its actual set, and carries
    no distinguishing ``[from]`` tag -- it must always be excluded.
    """
    battle = _battle("battle-struggle")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", "Struggle", "p1a: Arcanine"),
    )

    assert trackers["p2: Gardevoir"].revealed_moves == ()


@pytest.mark.parametrize(
    ("caller", "origin"),
    [
        ("Metronome", "move: Metronome"),
        ("Copycat", "move: Copycat"),
        ("Mirror Move", "Mirror Move"),
        ("Assist", "move: Assist"),
        ("Me First", "move: Me First"),
        ("Nature Power", "move: Nature Power"),
    ],
)
def test_move_calls_are_not_recorded_as_original_moves(data, caller, origin):
    """data/moves.ts ``callsMove: true`` entries call a move the Pokemon did
    not choose from its own set (a random move, the battle's last move, the
    target's last move, a teammate's move, the opponent's move, or a
    terrain-dependent move) -- except Sleep Talk, covered separately below.
    """
    battle = _battle(f"battle-called-{to_id(caller)}")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", caller, "p1a: Arcanine"),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine", f"[from] {origin}"),
    )

    assert {to_id(move) for move in trackers["p2: Gardevoir"].revealed_moves} == {
        to_id(caller)
    }


def test_sleep_talks_called_move_is_original_set_evidence(data):
    """Unlike the other callers above, Sleep Talk's onHit (data/moves.ts)
    samples only from ``pokemon.moveSlots`` -- the user's own moveset -- so
    the move it calls out IS legitimate original-set evidence and must stay
    recorded, not be treated like Metronome/Copycat/Mirror Move/etc.
    """
    battle = _battle("battle-called-sleeptalk")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", "Sleep Talk", "p1a: Arcanine"),
        ("move", "p2a: Gardevoir", "Psychic", "p1a: Arcanine", "[from] move: Sleep Talk"),
    )

    assert {to_id(move) for move in trackers["p2: Gardevoir"].revealed_moves} == {
        "sleeptalk",
        "psychic",
    }


@pytest.mark.parametrize(
    ("reflector_move", "origin"),
    [
        ("Magic Bounce", "[from] ability: Magic Bounce"),
        ("Magic Coat", "[from] Magic Coat"),
    ],
)
def test_bounced_moves_are_not_the_reflectors_own_evidence(data, reflector_move, origin):
    """Magic Bounce/Magic Coat re-use battle-actions.ts#useMove() to replay
    the ORIGINAL attacker's move back at them (sim/battle.ts#boost and
    data/abilities.ts magicbounce; data/moves.ts magiccoat), printing
    ``|move|<reflector>|<move>|<attacker>|[from] ...``. That reflected move is
    evidence for the original attacker, not the reflecting Pokemon.
    """
    battle = _battle(f"battle-bounced-{to_id(reflector_move)}")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p1a: Arcanine", "Taunt", "p2a: Gardevoir"),
        ("move", "p2a: Gardevoir", "Taunt", "p1a: Arcanine", origin),
    )

    assert trackers["p2: Gardevoir"].revealed_moves == ()


def test_snatched_move_is_not_the_snatchers_own_evidence(data):
    """Snatch (data/moves.ts) steals a status move meant for someone else and
    runs it through the snatcher via the same sourceEffect-less useMove()
    path; the snatcher's own |move| event for it is not their set evidence.

    Arcanine (the snatcher) is on p1, the side ``_battle()`` marks as the
    bot's own side, so it is checked via the raw protocol-provenance state
    (like ``test_skill_swap_records_both_originals_not_received_intimidate``)
    rather than via ``OpponentModel.update``'s opponent-only trackers.
    """
    battle = _battle("battle-snatch")
    model = OpponentModel(data)
    _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p1a: Arcanine", "Snatch", "p1a: Arcanine"),
        ("move", "p2a: Gardevoir", "Cotton Spore", "p1a: Arcanine"),
        ("-activate", "p1a: Arcanine", "move: Snatch", "[of] p2a: Gardevoir"),
        ("move", "p1a: Arcanine", "Cotton Spore", "p1a: Arcanine", "[from] Snatch"),
    )

    state = model._battles[battle.battle_tag]
    arcanine = state.pokemon[("p1", "arcanine")]
    observed_moves = {
        to_id(value) for kind, value in arcanine.current.observations if kind == "moves"
    }
    assert observed_moves == {"snatch"}


def test_trick_received_item_is_not_original_set_evidence(data):
    battle = _battle("battle-trick")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Great Tusk", "Great Tusk, L77", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("-item", "p2a: Gardevoir", "Choice Scarf"),
        ("-item", "p1a: Great Tusk", "Leftovers"),
    )
    tracker = trackers["p2: Gardevoir"]
    assert tracker._revealed_item == "Choice Scarf"

    tracker = _replay(
        battle,
        model,
        ("move", "p2a: Gardevoir", "Trick", "p1a: Great Tusk"),
        ("-activate", "p2a: Gardevoir", "move: Trick", "[of] p1a: Great Tusk"),
        ("-item", "p1a: Great Tusk", "Choice Scarf", "[from] move: Trick"),
        ("-item", "p2a: Gardevoir", "Leftovers", "[from] move: Trick"),
    )["p2: Gardevoir"]

    assert tracker._revealed_item == "Choice Scarf"
    assert ("items", "leftovers") not in tracker._observation_ids


def test_consumed_sitrus_berry_remains_evidence_after_next_update(data):
    battle = _battle("battle-consumed-item")
    model = OpponentModel(data)
    tracker = _replay(
        battle,
        model,
        ("switch", "p2a: Azumarill", "Azumarill, L80", "100/100"),
    )["p2: Azumarill"]
    assert tracker._revealed_item is None

    tracker = _replay(
        battle, model, ("-enditem", "p2a: Azumarill", "Sitrus Berry", "[eat]")
    )["p2: Azumarill"]
    assert battle.opponent_team["p2: Azumarill"].item is None

    assert tracker._revealed_item == "Sitrus Berry"
    assert "Bulky Attacker" in tracker.candidate_roles()


def test_illusion_evidence_moves_from_disguise_to_zoroark_on_replace(data):
    battle = _battle("battle-illusion")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Gardevoir", "Gardevoir, L80", "100/100"),
        ("move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"),
    )
    assert "darkpulse" in {to_id(move) for move in trackers["p2: Gardevoir"].revealed_moves}

    trackers = _replay(
        battle,
        model,
        ("replace", "p2a: Zoroark", "Zoroark, L80", "100/100"),
    )
    assert trackers["p2: Gardevoir"].revealed_moves == ()
    assert "darkpulse" in {to_id(move) for move in trackers["p2: Zoroark"].revealed_moves}


def test_skill_swap_records_both_originals_not_received_intimidate(data):
    battle = _battle("battle-skill-swap")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p1a: Arcanine", "Skill Swap", "p2a: Gardevoir"),
        (
            "-activate",
            "p1a: Arcanine",
            "Skill Swap",
            "Synchronize",
            "Intimidate",
            "[of] p2a: Gardevoir",
        ),
        ("-activate", "p2a: Gardevoir", "ability: Intimidate"),
    )
    gardevoir = trackers["p2: Gardevoir"]
    assert gardevoir._revealed_ability == "synchronize"
    assert ("abilities", "intimidate") not in gardevoir._observation_ids

    state = model._battles[battle.battle_tag]
    arcanine = state.pokemon[("p1", "arcanine")]
    assert ("abilities", "intimidate") in arcanine.current.observations


def test_simple_beam_received_ability_is_not_original_evidence(data):
    battle = _battle("battle-simple-beam")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p1a: Arcanine", "Simple Beam", "p2a: Gardevoir"),
        (
            "-ability",
            "p2a: Gardevoir",
            "Simple",
            "Synchronize",
            "[from] move: Simple Beam",
            "[of] p1a: Arcanine",
        ),
    )

    assert trackers["p2: Gardevoir"]._revealed_ability == "synchronize"
    assert ("abilities", "simple") not in trackers["p2: Gardevoir"]._observation_ids


def test_frisk_item_reveal_with_nontransfer_origin_is_original(data):
    battle = _battle("battle-frisk-item")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        (
            "-item",
            "p2a: Gardevoir",
            "Choice Scarf",
            "[from] ability: Frisk",
            "[of] p1a: Arcanine",
        ),
    )

    assert trackers["p2: Gardevoir"]._revealed_item == "Choice Scarf"


def test_failed_trick_does_not_taint_original_item_revealed_by_knock_off(data):
    battle = _battle("battle-failed-trick")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Great Tusk", "Great Tusk, L77", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("move", "p2a: Gardevoir", "Trick", "p1a: Great Tusk", "[miss]"),
        ("move", "p1a: Great Tusk", "Knock Off", "p2a: Gardevoir"),
        (
            "-enditem",
            "p2a: Gardevoir",
            "Choice Scarf",
            "[from] move: Knock Off",
            "[of] p1a: Great Tusk",
        ),
    )

    assert trackers["p2: Gardevoir"]._revealed_item == "Choice Scarf"


def test_covet_transfer_records_donor_item_not_recipient_item(data):
    battle = _double_battle("battle-covet")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("switch", "p2b: Great Tusk", "Great Tusk, L77", "100/100"),
        ("move", "p2b: Great Tusk", "Covet", "p2a: Gardevoir"),
        (
            "-item",
            "p2b: Great Tusk",
            "Choice Scarf",
            "[from] move: Covet",
            "[of] p2a: Gardevoir",
        ),
    )

    assert trackers["p2: Gardevoir"]._revealed_item == "Choice Scarf"
    assert trackers["p2: Great Tusk"]._revealed_item is None


def test_player_mixin_observes_protocol_before_poke_env_handles_request(data):
    """A bare battle tag (direct/synthetic callers) still works end to end."""
    battle = _battle("battle-player-hook")
    model = OpponentModel(data)

    class PlayerBase:
        def __init__(self, battle):
            self._battles = {battle.battle_tag: battle}
            self.request_trackers = None

        async def _handle_battle_message(self, split_messages):
            for event in split_messages[1:]:
                if event[0] == "" and event[1] != "request":
                    self._battles[split_messages[0][0]].parse_message(event)
                elif event[1] == "request":
                    await self._handle_battle_request(self._battles[split_messages[0][0]])

        async def _handle_battle_request(self, battle):
            self.request_trackers = self.opponent_model.update(battle)

    class ModelledPlayer(OpponentModelPlayerMixin, PlayerBase):
        pass

    player = ModelledPlayer(battle=battle, opponent_model=model)
    asyncio.run(
        player._handle_battle_message(
            [
                [battle.battle_tag],
                ["", "switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"],
                ["", "move", "p2a: Gardevoir", "Dark Pulse", "p1a: Arcanine"],
                ["", "request", "{}"],
            ]
        )
    )

    assert "darkpulse" in {
        to_id(move)
        for move in player.request_trackers["p2: Gardevoir"].revealed_moves
    }


@pytest.mark.parametrize(
    ("first_line", "expected_tag"),
    [
        # Real poke-env traffic: PSClient routes on a leading ">" that
        # player.py's _create_battle/_get_battle both strip (player.py
        # lines 182-262); battle.battle_tag never has it.
        (">battle-gen9randombattle-1234", "battle-gen9randombattle-1234"),
        # Best-of wrapper room: not a single battle, carries no mon events.
        (">game-gen9randombattlebo3-1234", None),
        # Bare tag: the convenience path synthetic/unit-test callers use.
        ("battle-gen9randombattle-1234", "battle-gen9randombattle-1234"),
    ],
)
def test_real_battle_tag_matches_poke_env_room_routing(first_line, expected_tag):
    from showdown_bot.sets import _real_battle_tag

    assert _real_battle_tag([[first_line], ["", "switch"]]) == expected_tag


def test_real_battle_tag_handles_empty_input():
    from showdown_bot.sets import _real_battle_tag

    assert _real_battle_tag([]) is None
    assert _real_battle_tag([[]]) is None


def _raw_showdown_lines(text: str) -> list[list[str]]:
    """Split a raw multi-line websocket frame exactly as PSClient does.

    See ps_client.py's ``_handle_message``: ``[m.split("|") for m in
    message.split("\\n")]``.
    """
    return [line.split("|") for line in text.split("\n")]


def test_player_mixin_observes_through_real_poke_env_message_path(data):
    """Drive the mixin through the real, installed poke-env ``Player``.

    Real Showdown traffic prefixes the battle-room tag with ">" (for example
    ">battle-gen9randombattle-1234"); ``battle.battle_tag`` never has that
    prefix. This reproduces the exact frames ``PSClient._handle_message``
    hands to ``Player._handle_battle_message`` and fails if the mixin ever
    again fails to recognize that shape.
    """
    from poke_env.player import RandomPlayer

    class TrackedRandom(OpponentModelPlayerMixin, RandomPlayer):
        pass

    model = OpponentModel(data)
    player = TrackedRandom(
        opponent_model=model,
        battle_format="gen9randombattle",
        start_listening=False,
        log_level=50,
    )

    init_frame = _raw_showdown_lines(
        ">battle-gen9randombattle-1234\n|init|battle\n|title|p1 vs. p2"
    )
    asyncio.run(player._handle_battle_message(init_frame))

    battle_tag = "battle-gen9randombattle-1234"
    assert battle_tag in player._battles
    battle = player._battles[battle_tag]
    battle.player_role = "p1"

    # A realistic frame interleaves lines the model must ignore without
    # crashing: a |t:| timestamp, a |split|p1 marker (poke-env itself relays
    # both the secret and public variants as separate lines), and a trailing
    # |request| line.
    battle_frame = _raw_showdown_lines(
        f">{battle_tag}\n"
        "|t:|1700000000\n"
        "|switch|p2a: Gardevoir|Gardevoir, L83|100/100\n"
        "|split|p2\n"
        "|move|p2a: Gardevoir|Dark Pulse|p1a: Arcanine\n"
        "|move|p2a: Gardevoir|Dark Pulse|p1a: Arcanine\n"
        "|request|"
    )
    asyncio.run(player._handle_battle_message(battle_frame))

    trackers = model.update(battle)
    assert "darkpulse" in {
        to_id(move) for move in trackers["p2: Gardevoir"].revealed_moves
    }


def test_unresolved_species_is_logged_and_skipped_without_crashing(data, caplog):
    battle = SimpleNamespace(
        battle_tag="battle-unresolved",
        opponent_team={"p2: Mysterymon": SimpleNamespace(species="mysterymon", gen=9)},
        _replay_data=[],
        player_role="p1",
    )

    assert OpponentModel(data).update(battle) == {}
    assert "no randbats species mapping" in caplog.text


def test_magic_bounce_reflected_move_is_not_original_evidence(data):
    """Magic Bounce's onTryHit (data/abilities.ts) reruns the ORIGINAL
    attacker's move through battle-actions.ts#useMove() without an explicit
    sourceEffect, so sim/battle.ts falls back to ``this.battle.effect``
    (the ability) and prints ``|move|<bouncer>|<move>|<attacker>|[from]
    ability: Magic Bounce``. That reflected move must not count as Espeon's
    own randbats-set evidence.
    """
    battle = _battle("battle-magic-bounce")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Espeon", "Espeon, L84", "100/100"),
        ("move", "p1a: Arcanine", "Stealth Rock", "p2a: Espeon"),
        (
            "move",
            "p2a: Espeon",
            "Stealth Rock",
            "p1a: Arcanine",
            "[from] ability: Magic Bounce",
        ),
    )

    tracker = trackers["p2: Espeon"]
    assert tracker.revealed_moves == ()
    assert not tracker.inconsistent


def test_transformed_abilities_are_not_recorded_until_switch_out(data):
    """sim/pokemon.ts#transformInto copies the target's whole kit (ability
    included); after ``-transform`` (Imposter or the move Transform), every
    ability event for that Pokemon until it switches out is a copied
    ability, not original. Imposter itself is the one exception, recorded
    directly by the ``-transform`` handler.
    """
    battle = _battle("battle-imposter-copied-ability")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Ditto", "Ditto, L87", "100/100"),
        (
            "-transform",
            "p2a: Ditto",
            "p1a: Arcanine",
            "[from] ability: Imposter",
        ),
        ("-ability", "p2a: Ditto", "Intimidate"),
    )

    tracker = trackers["p2: Ditto"]
    assert tracker._revealed_ability == "imposter"
    assert ("abilities", "intimidate") not in tracker._observation_ids
    assert not tracker.inconsistent


@pytest.mark.parametrize(
    ("species", "switch_detail", "companion_ability", "expected_id"),
    [
        ("Calyrex-Ice", "Calyrex-Ice, L72", "Chilling Neigh", "asoneglastrier"),
        ("Calyrex-Shadow", "Calyrex-Shadow, L64", "Grim Neigh", "asonespectrier"),
    ],
)
def test_as_one_component_abilities_fold_into_the_species_variant(
    data, species, switch_detail, companion_ability, expected_id
):
    """As One's onStart (data/abilities.ts asoneglastrier/asonespectrier)
    prints two plain ``-ability`` events back to back, "As One" then
    "Unnerve" -- As One's literal protocol name has no parenthetical, so
    ``to_id`` gives "asone", not randbats' "asoneglastrier"/"asonespectrier"
    role ability id. Its post-KO stat boost (data/abilities.ts
    chillingneigh/grimneigh via sim/battle.ts#boost) separately prints the
    gen 8 component ability name as its own ``-ability|mon|Name|boost``
    event. None of the three may be recorded as their own, separate,
    conflicting ability.
    """
    battle = _battle(f"battle-as-one-{to_id(species)}")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", f"p2a: {species}", switch_detail, "100/100"),
        ("-ability", f"p2a: {species}", "As One"),
        ("-ability", f"p2a: {species}", "Unnerve"),
        ("-ability", f"p2a: {species}", companion_ability, "boost"),
    )

    tracker = trackers[f"p2: {species}"]
    assert tracker._observation_ids == {("abilities", expected_id)}
    assert not tracker.inconsistent


def test_embody_aspect_is_never_original_evidence(data):
    """Embody Aspect only activates when Ogerpon Terastallizes into its
    battle-only "-Tera" forme (data/abilities.ts embodyaspect*; data/
    pokedex.ts ogerpon*tera entries), printing a plain
    ``-ability|mon|Embody Aspect (Wellspring)|boost`` event (sim/battle.ts
    #boost). It must never be recorded as the base randbats entry's ability.
    """
    battle = _battle("battle-embody-aspect")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Ogerpon-Wellspring", "Ogerpon-Wellspring, L76", "100/100"),
        ("-terastallize", "p2a: Ogerpon-Wellspring", "Water"),
        ("-ability", "p2a: Ogerpon-Wellspring", "Embody Aspect (Wellspring)", "boost"),
    )

    tracker = trackers["p2: Ogerpon-Wellspring"]
    assert tracker._revealed_ability is None
    assert not tracker.inconsistent


def test_terapagos_tera_shell_is_discarded_but_tera_shift_is_kept(data):
    """Tera Shift (data/abilities.ts terashift onSwitchIn) is Terapagos's own
    randbats ability and must stay recorded. Tera Shell only ever surfaces
    via Terapagos-Terastal's passive type-effectiveness activation
    (sim/pokemon.ts#runEffectiveness, printed as a plain
    ``-activate|mon|ability: Tera Shell``) after Tera Shift silently
    forme-changes it, and is never original randbats evidence.
    """
    battle = _battle("battle-terapagos")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Terapagos", "Terapagos, L77", "100/100"),
        ("-activate", "p2a: Terapagos", "ability: Tera Shift"),
        ("-activate", "p2a: Terapagos", "ability: Tera Shell"),
    )

    tracker = trackers["p2: Terapagos"]
    assert tracker._revealed_ability == "terashift"
    assert not tracker.inconsistent


def test_greninja_bond_set_is_tracked_until_battle_bond_narrows_it(data):
    """pkmn/randbats buckets a Battle-Bond roll under a separate
    "Greninja-Bond" entry, but Showdown's gen9 random team generator still
    shows it as plain "Greninja" at switch-in (data/random-battles/gen9/
    teams.ts: ``name: species.baseSpecies``, identical for every ability
    roll). Until the ability is revealed, both entries' roles must stay
    candidates; once Battle Bond narrows it, sampling must report the
    correct entry.
    """
    battle = _battle("battle-greninja-bond")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p2a: Greninja", "Greninja, L80", "100/100"),
    )
    tracker = trackers["p2: Greninja"]
    assert set(tracker.candidate_roles()) == {
        "Greninja: Wallbreaker",
        "Greninja: Fast Attacker",
        "Greninja-Bond: Fast Attacker",
    }

    trackers = _replay(
        battle,
        model,
        ("-ability", "p2a: Greninja", "Battle Bond"),
    )
    tracker = trackers["p2: Greninja"]
    assert set(tracker.candidate_roles()) == {"Greninja-Bond: Fast Attacker"}
    assert not tracker.inconsistent

    sampled = tracker.sample(random.Random(0))
    assert sampled.species == "Greninja-Bond"
    assert to_id(sampled.ability) == "battlebond"


def test_hyper_voice_on_slowbro_galar_is_held_as_provisional_zoroark_evidence(data):
    """A move illegal for every Slowbro-Galar role, but legal for an
    unrevealed Zoroark-Hisui (almost certainly an undisguised Illusion that
    switched out before breaking -- task 589.9 resolves this), must be held
    provisional rather than recorded as Slowbro-Galar's own move or forcing
    every role into conflict.
    """
    battle = _battle("battle-slowbro-galar-provisional")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        ("move", "p2a: Slowbro-Galar", "Hyper Voice", "p1a: Arcanine"),
    )

    tracker = trackers["p2: Slowbro-Galar"]
    assert "Hyper Voice" in tracker.revealed_moves
    assert tuple(to_id(move) for move in tracker.provisional_moves) == ("hypervoice",)
    assert not tracker.inconsistent
    assert tracker.candidate_roles()

    sampled = tracker.sample(random.Random(0))
    assert "hypervoice" not in {to_id(move) for move in sampled.moves}


def test_provisional_zoroark_evidence_stays_provisional_even_once_zoroark_is_confirmed_elsewhere(
    data,
):
    """Illusion re-disguises its user as the last healthy teammate every time
    it switches in, so a Zoroark-Hisui that later appears as itself is
    consistent with an earlier, never-broken disguise. Singles sequence:
    disguised Slowbro-Galar uses Hyper Voice and switches out, then the
    Zoroark-Hisui later switches in undisguised.
    """
    battle = _battle("battle-slowbro-galar-zoroark-confirmed")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        ("move", "p2a: Slowbro-Galar", "Hyper Voice", "p1a: Arcanine"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("switch", "p2a: Zoroark-Hisui", "Zoroark-Hisui, L80", "100/100"),
    )

    tracker = trackers["p2: Slowbro-Galar"]
    assert tuple(to_id(move) for move in tracker.provisional_moves) == ("hypervoice",)
    assert not tracker.inconsistent


def test_only_formes_showdown_hides_in_details_share_a_tracker(data):
    """sim/pokemon.ts#getUpdatedDetails prints only Greninja-Bond and
    Rockruff-Dusk under their base species. Every other forme (regional
    variants, Urshifu-Rapid-Strike, gender formes) is visible at switch-in
    and must never be merged with its base species.
    """
    merged = {
        species_id: _hidden_forme_species(data, species_id)
        for species_id in data.species_ids()
        if _hidden_forme_species(data, species_id)
    }
    assert merged == {"greninja": ("greninjabond",)}


@pytest.mark.parametrize(
    ("details", "identifier", "expected_species"),
    [
        ("Urshifu-Rapid-Strike, L75", "p2: Urshifu", "Urshifu-Rapid-Strike"),
        ("Urshifu, L76", "p2: Urshifu", "Urshifu"),
        ("Ninetales-Alola, L82", "p2: Ninetales", "Ninetales-Alola"),
        ("Ninetales, L88", "p2: Ninetales", "Ninetales"),
    ],
)
def test_visible_formes_are_never_merged_with_their_base_species(
    data, details, identifier, expected_species
):
    battle = _battle(f"battle-forme-{to_id(details)}")
    model = OpponentModel(data)
    nickname = identifier.split(": ", 1)[1]
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", f"p2a: {nickname}", details, "100/100"),
    )

    tracker = trackers[identifier]
    assert tracker.species == expected_species
    expected_roles = set(data.roles(expected_species))
    assert set(tracker.candidate_roles()) == expected_roles
    rng = random.Random(0)
    assert {tracker.sample(rng).species for _ in range(200)} == {expected_species}


def test_disguised_zoroark_tera_moves_to_zoroark_when_illusion_breaks(data):
    """battle-actions.ts#terastallize logs ``-terastallize`` with the
    Pokemon's toString(), which is the Illusion name. Tera Poison is impossible
    for every Carbink role and legal for Zoroark, so it stays provisional on
    Carbink, and ``replace`` hands it to Zoroark.
    """
    battle = _battle("battle-zoroark-tera")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Arcanine", "Arcanine, L80", "100/100"),
        ("switch", "p2a: Carbink", "Carbink, L90", "100/100"),
        ("-terastallize", "p2a: Carbink", "Poison"),
    )
    carbink = trackers["p2: Carbink"]
    assert carbink._revealed_tera is None
    assert ("teraTypes", "Poison") in carbink.provisional_observations
    assert not carbink.inconsistent
    rng = random.Random(0)
    assert {to_id(carbink.sample(rng).tera_type) for _ in range(100)} == {"fighting"}

    trackers = _replay(
        battle,
        model,
        ("replace", "p2a: Zoroark", "Zoroark, L82", "100/100"),
    )
    assert trackers["p2: Carbink"]._observations == []
    assert trackers["p2: Zoroark"]._revealed_tera == "Poison"


def test_frisk_on_a_disguise_holds_the_whole_switch_in_provisional(data):
    """Frisk reveals an item without breaking Illusion. Choice Specs is
    impossible for Slowbro-Galar and legal for Zoroark, so every observation
    from that switch-in, including Psychic (which both can have), stays
    provisional after the disguise switches out unbroken.
    """
    battle = _battle("battle-frisk-disguise")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Noivern", "Noivern, L84", "100/100"),
        ("switch", "p2a: Slowbro-Galar", "Slowbro-Galar, L87", "100/100"),
        ("-item", "p2a: Slowbro-Galar", "Choice Specs", "[from] ability: Frisk", "[of] p1a: Noivern"),
        ("move", "p2a: Slowbro-Galar", "Psychic", "p1a: Noivern"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
    )

    slowbro = trackers["p2: Slowbro-Galar"]
    assert slowbro._revealed_item is None
    assert set(slowbro.provisional_observations) == {
        ("items", "Choice Specs"),
        ("moves", "Psychic"),
    }
    assert not slowbro.inconsistent
    rng = random.Random(0)
    own_items = {to_id(i) for role in data.roles("Slowbro-Galar").values() for i in role.items}
    assert {to_id(slowbro.sample(rng).item) for _ in range(100)} <= own_items


def test_tera_in_switch_details_keeps_a_returning_disguise_suspect(data):
    """A disguised "Gardevoir" terastallizes to Poison and uses Psychic, then
    switches out unbroken. On its return Showdown's switch details carry
    ``tera:Poison`` (sim/pokemon.ts#getFullDetails), which Gardevoir can't
    have, so the second switch-in is suspect too and its Psychic can't
    confirm the first one's.
    """
    battle = _battle("battle-returning-disguise")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Noivern", "Noivern, L84", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83", "100/100"),
        ("-terastallize", "p2a: Gardevoir", "Poison"),
        ("move", "p2a: Gardevoir", "Psychic", "p1a: Noivern"),
        ("switch", "p2a: Azumarill", "Azumarill, L82", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83, tera:Poison", "100/100"),
        ("move", "p2a: Gardevoir", "Psychic", "p1a: Noivern"),
    )

    gardevoir = trackers["p2: Gardevoir"]
    assert set(gardevoir.provisional_observations) == {
        ("teraTypes", "Poison"),
        ("moves", "Psychic"),
    }
    assert gardevoir._revealed_tera is None
    assert gardevoir._binding_revealed_moves == []


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        ("Gardevoir, L83, F, tera:Fairy", "Fairy"),
        ("Terapagos, L76, tera:Stellar", "Stellar"),
        ("Gardevoir, L83, F, shiny, tera:Fire", "Fire"),
        ("Gardevoir, L83, F", None),
        ("Gardevoir, L83, tera:NotAType", None),
    ],
)
def test_details_tera_accepts_only_real_types(details, expected):
    assert _details_tera(details) == expected


def test_genuine_tera_first_seen_in_switch_details_is_recorded(data):
    battle = _battle("battle-tera-details")
    model = OpponentModel(data)
    trackers = _replay(
        battle,
        model,
        ("switch", "p1a: Noivern", "Noivern, L84", "100/100"),
        ("switch", "p2a: Gardevoir", "Gardevoir, L83, F, tera:Fairy", "100/100"),
    )

    gardevoir = trackers["p2: Gardevoir"]
    assert gardevoir._revealed_tera == "Fairy"
    assert gardevoir.provisional_observations == ()
    rng = random.Random(0)
    assert {to_id(gardevoir.sample(rng).tera_type) for _ in range(50)} == {"fairy"}


def test_conflicting_tera_stays_binding_when_no_zoroark_can_be_on_the_team(data):
    """With all six opponents revealed and none of them Zoroark, nothing can
    explain the conflict away. It stays binding evidence and the tracker
    flags itself inconsistent, rather than hiding a real data mismatch.
    """
    battle = _battle("battle-no-zoroark-possible")
    model = OpponentModel(data)
    team = [
        ("Gardevoir", "Gardevoir, L83"),
        ("Great Tusk", "Great Tusk, L76"),
        ("Espeon", "Espeon, L86"),
        ("Abomasnow", "Abomasnow, L84"),
        ("Azumarill", "Azumarill, L82"),
        ("Carbink", "Carbink, L90"),
    ]
    events = [("switch", "p1a: Arcanine", "Arcanine, L80", "100/100")]
    events += [("switch", f"p2a: {name}", details, "100/100") for name, details in team]
    events.append(("-terastallize", "p2a: Carbink", "Poison"))
    trackers = _replay(battle, model, *events)

    carbink = trackers["p2: Carbink"]
    assert carbink.provisional_observations == ()
    assert carbink._revealed_tera == "Poison"
    assert carbink.inconsistent


@pytest.mark.network
def test_live_randbats_data_fetches_and_parses_all_files(tmp_path, monkeypatch):
    fetched_urls = []
    real_urlopen = urllib.request.urlopen

    def recording_urlopen(request, *args, **kwargs):
        fetched_urls.append(request.full_url)
        return real_urlopen(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", recording_urlopen)
    live_data = RandbatsData.load(refresh=True, cache_dir=tmp_path)

    sets_path = tmp_path / "gen9randombattle.json"
    stats_path = tmp_path / "stats" / "gen9randombattle.json"
    assert sets_path.is_file() and sets_path.stat().st_size > 0
    assert stats_path.is_file() and stats_path.stat().st_size > 0
    full_path = tmp_path / "full" / "gen9randombattle.json"
    assert full_path.is_file() and full_path.stat().st_size > 0
    assert set(fetched_urls) == {
        "https://pkmn.github.io/randbats/data/gen9randombattle.json",
        "https://pkmn.github.io/randbats/data/stats/gen9randombattle.json",
        "https://pkmn.github.io/randbats/data/full/gen9randombattle.json",
    }
    assert len(live_data.species_ids()) >= 300
    abomasnow_sets = live_data.full_sets("Abomasnow")
    assert abomasnow_sets and all(fs.count > 0 for fs in abomasnow_sets)
    assert len(live_data.full_sets("Greninja")) > 0
    assert 0 < live_data.roles("Abomasnow")["Bulky Support"].option_weight(
        "moves", "Earthquake"
    ) < 1

