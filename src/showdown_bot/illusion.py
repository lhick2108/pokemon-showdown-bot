"""Illusion suspicion: how likely is the active opponent a disguised Zoroark?

Gen 9 Random Battle runs the "Illusion Level Mod", so a disguised Zoroark
shows the disguise's level and there is no level tell. This module is the pure
core (no battle, no protocol parsing) that turns the tells collected by
``sets.OpponentModel`` into ``{zoroark_form_id: probability}``:

* ``form_weights`` - the Zoroark forms that can still be on the opponent's
  team (empty: none, so the probability is zero). Weights split probability
  between the forms (their randbats frequency, or 1 for a revealed form).
* ``prior`` - per-switch-in base rate with no evidence (see ``base_rate``).
* hard immunity tell - a damaging move showed ``-immune`` against a species
  that is not immune to its type while a Zoroark form is. Zoroark is pure
  Dark (immune to Psychic); Zoroark-Hisui is Normal/Ghost (immune to Normal,
  Fighting and Ghost). Probability ``IMMUNITY_P`` for the forms that explain
  every contradiction.
* soft kit tell - the switch-in showed a move/item/ability/tera type that
  the shown species can't have but a Zoroark form can (``KIT_SUSPECT_P``,
  restricted to the forms that can have all of it).
* duplicates (Species Clause: no species twice) - see ``DUPLICATE_*``.
* negative tells - a direct hit on the active that did not break Illusion,
  or something the shown species has that Zoroark can't, means "real".

Evidence combines by taking the largest probability among the tells; the
forms allowed are the intersection of the forms each tell points at (a tell
that would leave no form is dropped). Contradictory evidence (a "real" tell
alongside a "disguised" tell) falls back to the prior.

Not modelled: the "Illusion copies the last non-fainted party slot" tell
(the opponent's party order is unknown with no team preview), and the
Terastallized Ogerpon/Terapagos exception (Illusion can't copy them while
Terastallized; skipped because the copied Pokemon's tera state is unknowable).
Damage-range tells are a follow-up task.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache

from poke_env.data import GenData

ZOROARK_FORM_IDS = ("zoroark", "zoroarkhisui")

#: Probability the active is a Zoroark when a damaging move was type-immune on
#: a species that should not be immune but a Zoroark form is.
IMMUNITY_P = 1.0
#: Soft tell: a switch-in showed something illegal for the shown species but
#: legal for a Zoroark form (a Zoroark that hasn't been explained away may
#: also be a rarer real-set variation, hence below 1).
KIT_SUSPECT_P = 0.9
#: Two distinct Pokemon carried this species name and we can't tell which was
#: Zoroark: either the earlier or the current one, so a coin flip.
DUPLICATE_AMBIGUOUS_P = 0.5
#: Duplicate whose earlier appearance showed something Zoroark can't have, so
#: the earlier one was real and the current one must be the Zoroark.
DUPLICATE_EARLIER_REAL_P = 1.0

# Moves whose type depends on state, so the move name doesn't give the type.
_VARIABLE_TYPE_MOVES = frozenset(
    {
        "hiddenpower", "judgment", "weatherball", "terablast", "terastarstorm",
        "ragingbull", "ivycudgel", "revelationdance", "terrainpulse", "aurawheel",
        "technoblast", "multiattack", "naturalgift", "struggle",
    }
)
# Attacker abilities that change a move's type: an ``-immune`` then says
# nothing about the move's printed type.
TYPE_CHANGING_ABILITIES = frozenset(
    {"pixilate", "aerilate", "refrigerate", "galvanize", "normalize", "liquidvoice"}
)


@lru_cache(maxsize=1)
def _gen_data() -> GenData:
    return GenData.from_gen(9)


def type_multiplier(attacking: str, defending: Iterable[str]) -> float:
    """Type-chart multiplier of ``attacking`` against ``defending`` types."""
    chart = _gen_data().type_chart
    multiplier = 1.0
    for defender in defending:
        multiplier *= chart[str(defender).upper()].get(str(attacking).upper(), 1)
    return multiplier


def species_types(species_id: str) -> tuple[str, ...]:
    entry = _gen_data().pokedex.get(species_id)
    return tuple(t.upper() for t in entry["types"]) if entry else ()


def damaging_move_type(move_id: str) -> str | None:
    """Printed type of a damaging move with a fixed type, else None."""
    if move_id in _VARIABLE_TYPE_MOVES:
        return None
    entry = _gen_data().moves.get(move_id)
    if not entry or entry.get("category") == "Status":
        return None
    return str(entry["type"]).upper()


def is_damaging_move(move_id: str) -> bool:
    entry = _gen_data().moves.get(move_id)
    return bool(entry) and entry.get("category") != "Status"


def base_rate(
    form_counts: Mapping[str, float],
    total_count: float,
    unrevealed_slots: int,
    alive_count: int,
    revealed_alive_form: str | None = None,
) -> float:
    """P(this switch-in is a Zoroark) with no other evidence.

    ``q`` = share of randbats teams' slots that are a Zoroark form. With
    ``u`` unrevealed slots, P(a Zoroark is among them) = 1 - (1 - q)^u; each
    switch-in is equally likely to be any of the ``alive_count`` non-fainted
    team members (revealed ones plus unrevealed slots). A revealed, living
    Zoroark is certainly on the team (it re-disguises every switch-in), so
    only the 1 / alive_count factor applies.
    """
    if alive_count <= 0:
        return 0.0
    if revealed_alive_form is not None:
        return min(1.0, 1.0 / alive_count)
    if total_count <= 0 or unrevealed_slots <= 0:
        return 0.0
    q = min(1.0, sum(form_counts.values()) / total_count)
    present = 1.0 - (1.0 - q) ** unrevealed_slots
    return min(1.0, max(0.0, present / alive_count))


@dataclass(frozen=True)
class IllusionEvidence:
    """Everything the pure estimator needs; see the module docstring."""

    shown_species: str
    form_weights: Mapping[str, float]
    prior: float = 0.0
    shown_types: tuple[str, ...] = ()
    immune_types: tuple[str, ...] = ()
    kit_suspect: bool = False
    kit_forms: Collection[str] | None = None
    kit_real: bool = False
    duplicate: str | None = None  # "ambiguous" | "earlier_real" | "current_real" | "fainted"
    direct_hit: bool = False
    form_types: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def immunity_forms(
    shown_types: Sequence[str],
    immune_types: Iterable[str],
    forms: Iterable[str],
    form_types: Mapping[str, Sequence[str]] | None = None,
) -> set[str] | None:
    """Forms that explain the immunity contradictions, or None if there are none.

    A type shown immune is a contradiction when the shown species isn't
    immune to it. Contradictions no form can explain (an ability immunity the
    protocol didn't tag, say) are ignored.
    """
    explained = set(forms)
    contradiction = False
    for attack_type in immune_types:
        if shown_types and type_multiplier(attack_type, shown_types) == 0:
            continue  # the shown species really is immune: nothing learned
        candidates = {
            form
            for form in explained
            if type_multiplier(
                attack_type, (form_types or {}).get(form) or species_types(form)
            )
            == 0
        }
        if candidates:
            explained = candidates
            contradiction = True
    return explained if contradiction else None


def estimate(evidence: IllusionEvidence) -> dict[str, float]:
    """``{form_id: P(active opponent is that Zoroark form)}``.

    ``{}`` means no Zoroark form can be the active one (probability zero):
    none is left on the team, the active IS a Zoroark, it took a direct hit
    without Illusion breaking, or an earlier Pokemon wearing this name
    fainted (so the current one is the real one).
    """
    forms = {f: w for f, w in evidence.form_weights.items() if w > 0}
    if (
        not forms
        or evidence.shown_species in ZOROARK_FORM_IDS
        or evidence.direct_hit
        or evidence.duplicate == "fainted"
    ):
        return {}

    allowed = set(forms)
    totals: list[float] = []
    real = evidence.kit_real or evidence.duplicate == "current_real"

    explained = immunity_forms(
        evidence.shown_types, evidence.immune_types, allowed, evidence.form_types
    )
    if explained:
        allowed &= explained
        totals.append(IMMUNITY_P)
    if evidence.kit_suspect:
        kit = set(evidence.kit_forms) if evidence.kit_forms is not None else set(allowed)
        if allowed & kit:
            allowed &= kit
            totals.append(KIT_SUSPECT_P)
    if evidence.duplicate == "ambiguous":
        totals.append(DUPLICATE_AMBIGUOUS_P)
    elif evidence.duplicate == "earlier_real":
        totals.append(DUPLICATE_EARLIER_REAL_P)

    if real:
        if totals:
            totals = []  # contradictory evidence: trust nothing but the prior
            allowed = set(forms)
        else:
            return {form: 0.0 for form in forms}

    total = min(1.0, max([evidence.prior, *totals]))
    weight_sum = sum(forms[f] for f in allowed)
    return {
        form: (total * forms[form] / weight_sum if form in allowed else 0.0)
        for form in forms
    }
