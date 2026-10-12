"""Damage-roll inference: which candidate opponent sets could have produced a hit?

``inference.InferenceTracker`` reads clean single-hit damaging moves out of the
protocol (``DamageObs``).  This module evaluates one against the *pre-message*
battle state with the engine's own damage calculator (``poke_engine.
calculate_damage``) for every candidate full set of the opposing Pokemon
(item / ability / tera / level / EV spread) and returns the set of candidates
whose damage range cannot contain the observed damage.

* Opponent attacks us: our HP is exact, so the observed damage is an exact HP
  number; compared with each candidate attacker's roll range.
* We attack the opponent: its HP is whole percent, so the observed damage is a
  percent (+-1 for the rounding of both readings); compared with each
  candidate defender's roll range as a share of its own max HP.

``calculate_damage`` only returns the MAXIMUM roll (and the maximum crit roll);
the minimum of Showdown's 16-value table is 85% of the maximum.  Candidates are
only rejected outside that range widened by a relative margin plus an absolute
one (the engine is not byte-exact with Showdown's modifier chain).  Engine
results are cached per damage-relevant set key.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import poke_engine as pe

from showdown_bot import engine_state
from showdown_bot.inference import Constraint, DamageObs, set_key, to_id

logger = logging.getLogger(__name__)

MIN_ROLL = 0.85
REL_MARGIN = 0.04  # of the engine's min/max
ABS_HP_MARGIN = 2  # HP points (our exact HP)
ABS_PCT_MARGIN = 1.6  # percentage points (opponent HP is rounded twice)
MAX_GROUPS = 160
_FILLER_MOVE = "splash"

_POKEMON_FIELDS = (
    "id", "level", "types", "base_types", "hp", "maxhp", "ability", "base_ability",
    "item", "nature", "evs", "attack", "defense", "special_attack", "special_defense",
    "speed", "status", "rest_turns", "sleep_turns", "weight_kg", "moves",
    "mega_evolved", "terastallized", "tera_type",
)


def _clone(pokemon: Any, **overrides: Any) -> Any:
    """A copy of an (immutable) engine Pokemon with some fields replaced."""
    fields = {name: getattr(pokemon, name) for name in _POKEMON_FIELDS}
    fields.update(overrides)
    return pe.Pokemon(**fields)


@dataclass
class DamageStats:
    observations: int = 0
    evaluated: int = 0
    narrowed: int = 0  # evaluations that rejected at least one candidate
    all_rejected: int = 0  # every candidate rejected: engine/model mismatch signal
    candidates: int = 0
    rejected: int = 0
    errors: int = 0
    seconds: float = 0.0
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self) -> str:
        skipped = ", ".join(f"{k}={v}" for k, v in sorted(self.skipped.items()))
        return (
            f"obs={self.observations} evaluated={self.evaluated} narrowed={self.narrowed} "
            f"all_rejected={self.all_rejected} candidates={self.candidates} "
            f"rejected={self.rejected} errors={self.errors} seconds={self.seconds:.2f}"
            + (f" skipped({skipped})" if skipped else "")
        )


ALL_STATS: list[DamageStats] = []


def total_damage_stats() -> DamageStats:
    """Sum of every ``DamageInferer``'s stats in this process."""
    total = DamageStats()
    for stats in ALL_STATS:
        for name in ("observations", "evaluated", "narrowed", "all_rejected", "candidates",
                     "rejected", "errors", "seconds"):
            setattr(total, name, getattr(total, name) + getattr(stats, name))
        for reason, count in stats.skipped.items():
            total.skipped[reason] = total.skipped.get(reason, 0) + count
    return total


class DamageInferer:
    """Evaluates ``DamageObs`` against the pre-message battle state."""

    def __init__(self, data: Any):
        self.data = data
        self.stats = DamageStats()
        ALL_STATS.append(self.stats)
        self._bad_moves: set[str] = set()

    def evaluate_all(
        self, battle: Any, trackers: dict[str, Any], observations: list[DamageObs]
    ) -> list[tuple[DamageObs, str, Constraint]]:
        """Return ``(obs, tracker_key, constraint)`` for every informative observation."""
        results: list[tuple[DamageObs, str, Constraint]] = []
        started = time.perf_counter()
        try:
            for obs in observations:
                self.stats.observations += 1
                try:
                    found = self._evaluate(battle, trackers, obs)
                except Exception as exc:  # never cost the decision
                    self.stats.errors += 1
                    logger.debug("Damage inference failed for %r: %s", obs, exc)
                    continue
                if found is not None:
                    results.append(found)
        finally:
            self.stats.seconds += time.perf_counter() - started
        return results

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _tracker_key(battle: Any, mon: Any) -> str | None:
        for key, value in battle.opponent_team.items():
            if value is mon:
                return str(key)
        return None

    def _evaluate(self, battle, trackers, obs: DamageObs):
        role = getattr(battle, "player_role", None)
        opp_attacks = obs.attacker[0] != role
        opp_identity = obs.attacker if opp_attacks else obs.defender
        our_identity = obs.defender if opp_attacks else obs.attacker
        opp_mon = battle.opponent_active_pokemon
        our_mon = battle.active_pokemon
        if opp_mon is None or our_mon is None:
            self.stats.skip("no-active")
            return None
        key = self._tracker_key(battle, opp_mon)
        tracker = trackers.get(key) if key else None
        if tracker is None:
            self.stats.skip("no-tracker")
            return None
        if _identity_of(key) != opp_identity:
            self.stats.skip("opp-not-active")
            return None
        if _our_name_id(battle, our_mon) != our_identity[1]:
            self.stats.skip("our-not-active")
            return None
        if obs.move_id in self._bad_moves:
            return None
        pool = tracker._matching_full_sets()
        if len(pool) < 2:
            self.stats.skip("pool<2")
            return None
        groups: dict[tuple, list[tuple[str, Any]]] = {}
        for species_id, full_set in pool:
            groups.setdefault(_damage_key(species_id, full_set), []).append((species_id, full_set))
        if len(groups) > MAX_GROUPS:
            self.stats.skip("too-many")
            return None

        fields = engine_state.field_kwargs(battle)
        our_poke, our_moves = engine_state._own_pokemon(battle, our_mon, True)
        if opp_attacks:
            observed = (obs.hp_before - obs.hp_after) * max(1, our_poke.maxhp)
            if round(obs.hp_after * our_poke.maxhp) <= 1:
                self.stats.skip("our-after<=1")
                return None
            our_poke = _clone(our_poke, hp=max(1, round(obs.hp_before * our_poke.maxhp)))
        else:
            observed_pct = (obs.hp_before - obs.hp_after) * 100.0
            if obs.hp_after * 100.0 <= 1.5:
                self.stats.skip("their-after<=1")
                return None
        side_one = engine_state._make_side(
            [(our_poke, our_moves, our_mon)], our_mon, battle, battle.side_conditions, is_ours=True
        )

        rejected: set[tuple] = set()
        for members in groups.values():
            species_id, sample = members[0]
            pokemon, moves = self._candidate(battle, opp_mon, species_id, sample)
            if not opp_attacks:
                pokemon = _clone(pokemon, hp=max(1, round(pokemon.maxhp * obs.hp_before)))
            side_two = engine_state._make_side(
                [(pokemon, moves, opp_mon)], opp_mon, battle, battle.opponent_side_conditions,
                is_ours=False,
            )
            state = pe.State(side_one=side_one, side_two=side_two, **fields)
            if opp_attacks:
                result = self._calc(state, _FILLER_MOVE, obs.move_id)
                if not result:
                    return None
                high = float(result[1][0])
                low = high * MIN_ROLL
                ok = (
                    low * (1 - REL_MARGIN) - ABS_HP_MARGIN
                    <= observed
                    <= high * (1 + REL_MARGIN) + ABS_HP_MARGIN
                )
            else:
                result = self._calc(state, obs.move_id, _FILLER_MOVE)
                if not result:
                    return None
                high = float(result[0][0])
                low = high * MIN_ROLL
                scale = 100.0 / max(1, pokemon.maxhp)
                ok = (
                    low * scale * (1 - REL_MARGIN) - ABS_PCT_MARGIN
                    <= observed_pct
                    <= high * scale * (1 + REL_MARGIN) + ABS_PCT_MARGIN
                )
            if not ok:
                rejected.update(set_key(sid, fs) for sid, fs in members)

        self.stats.evaluated += 1
        self.stats.candidates += len(pool)
        if not rejected:
            return None
        if len(rejected) >= len(pool):
            self.stats.all_rejected += 1
            return None
        self.stats.narrowed += 1
        self.stats.rejected += len(rejected)
        return obs, key, Constraint("reject", (frozenset(rejected),))

    def _candidate(self, battle, opp_mon, species_id: str, full_set):
        sampled = engine_state.SampledSet(
            species=species_id,
            level=full_set.level,
            role="full-set",
            moves=full_set.moves,
            ability=full_set.ability,
            item=full_set.item,
            tera_type=full_set.tera_type,
        )
        slot = engine_state.OpponentSlot(species_id, sampled, opp_mon)
        pokemon, moves, _ = engine_state._opponent_pokemon(slot, battle)
        return pokemon, moves

    def _calc(self, state, our_move: str, their_move: str):
        try:
            return pe.calculate_damage(state, our_move, their_move, True)
        except BaseException as exc:  # pyo3 panics on unknown move ids
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self._bad_moves.add(their_move if our_move == _FILLER_MOVE else our_move)
            return None


def _damage_key(species_id: str, full_set: Any) -> tuple:
    """Everything of a candidate that changes its damage (moves only via EV spreads)."""
    moves = tuple(to_id(m) for m in full_set.moves)
    item = to_id(full_set.item) if full_set.item else ""
    ability = to_id(full_set.ability)
    evs, ivs = engine_state.randbats_evs_ivs(species_id, int(full_set.level), moves, ability, item)
    return (
        species_id, int(full_set.level), item, ability, to_id(full_set.tera_type),
        tuple(evs), tuple(ivs),
    )


def _identity_of(tracker_key: str) -> tuple[str, str] | None:
    side, _, name = str(tracker_key).partition(":")
    name_id = to_id(name)
    return (side.strip().lower()[:2], name_id) if name_id else None


def _our_name_id(battle: Any, mon: Any) -> str | None:
    for key, value in battle.team.items():
        if value is mon:
            return to_id(str(key).partition(":")[2])
    return None
