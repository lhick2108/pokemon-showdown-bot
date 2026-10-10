"""Convert a live poke-env Battle plus one sampled opponent team into a poke_engine State.

Everything here is written from scratch for this project (Foul Play is GPL-3.0
and was read only to understand the idea).

Pipeline
--------
1. ``sample_opponent_team`` builds ONE determinized opponent team ("world"):
   every revealed opposing Pokemon gets a full set drawn by its ``SetTracker``
   (``SetTracker.sample``), and every unrevealed slot gets a species drawn by
   frequency from the randbats full-set data (``TeamSampler``) under the
   randbats team rules, then a full set for that species.
2. ``build_state`` turns the battle (our side, exact) and that opponent team
   into a ``BuiltState`` holding a ``poke_engine.State`` plus the bookkeeping
   needed to map engine move choices back to poke-env orders.
3. ``BuiltState.choice_to_order`` maps an engine choice string ("closecombat",
   "closecombat-tera", "switch greattusk") to a legal poke-env order.

Stats
-----
Our side's stats come from the request (``Pokemon.stats``), so they are exact.
Opponent stats use Showdown's formula with 31 IVs and 85 EVs in every stat and
a neutral nature, adjusted by the same rules ``data/random-battles/gen9/
teams.ts`` applies (see ``randbats_evs_ivs``)::

    HP    = floor((2*B + IV + floor(EV/4)) * L / 100) + L + 10   (Shedinja: 1)
    other = floor((2*B + IV + floor(EV/4)) * L / 100) + 5

Opponent HP is current HP% times the computed max HP (at least 1 if alive).

Fields that are NOT mapped (the engine has no field, or poke-env does not
track the value). Every one of these is silently approximated or dropped:

* Our non-active Pokemon's PP is only what poke-env saw them use; the request
  gives PP for the active Pokemon only.
* Opponent natures/EV spreads beyond the randbats rules above (randbats uses
  neutral natures, so this is exact for randbats).
* Substitute HP (poke-env does not track it): a present substitute is given a
  quarter of max HP.
* Sleep counter is the number of turns observed asleep. Rest's 2-turn timer
  (``rest_turns``) is not distinguished from ordinary sleep.
* Volatile durations: Confusion/Encore/Taunt/Lockedmove use poke-env's observed
  turn counters (clamped to the engine's valid range); Yawn is set from presence
  only; Slow Start is derived as 5 minus observed turns.
* Volatile statuses without an engine equivalent are dropped (Trapped, Illusion,
  Dynamax, Commander, Substitute-like abilities, ...). Turn-scoped statuses
  (Protect-family, Flinch, Helping Hand, Follow Me, Destiny Bond, ...) are
  intentionally NOT carried: they have expired by the time a decision is made.
* Wish, Future Sight / Doom Desire, Healing Wish, Lunar Dance, Mat Block,
  Quick/Wide/Crafty Guard: not tracked by poke-env, left at their defaults.
* Weather/terrain/Trick Room/screen/Tailwind/etc. remaining turns are derived
  from the start turn and the standard 5-turn (Tailwind: 4) duration; item
  extensions (Heat Rock, Light Clay, ...) are unknown and clamp to 1 turn left.
  Strong winds (Delta Stream), Gravity, Magic/Wonder Room, Mud/Water Sport,
  Fairy Lock, Neutralizing Gas and Heal-Block-as-a-field have no engine field.
* Pledge/G-Max side conditions: engine has none.
* Formes: Pokemon are given the id of their current poke-env forme (ours) or of
  the sampled randbats species (opponent). Mid-battle forme changes of the
  opponent (e.g. Mimikyu-Busted) are not reflected.
* Mega/Z/Dynamax: not part of Gen 9 randbats.
* Illusion: the opponent model's own handling decides which set a disguised
  Pokemon gets; this module does not second-guess it.
* Ids the engine does not know are replaced by its "none" id; they are counted
  in ``BuiltState.unknown_ids`` and logged at debug level.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any

import poke_engine as pe
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.move import Move as PEMove
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from showdown_bot.sets import RandbatsData, SampledSet, SetTracker, to_id

logger = logging.getLogger(__name__)

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
DEFAULT_EV = 85
DEFAULT_IV = 31
TEAM_SIZE = 6

_GEN = 9


# ---------------------------------------------------------------------------
# Static data helpers
# ---------------------------------------------------------------------------


def _gen_data() -> GenData:
    return GenData.from_gen(_GEN)


def dex_entry(species_id: str) -> Mapping[str, Any] | None:
    """Return the pokedex entry for a species id, or None if unknown."""
    return _gen_data().pokedex.get(to_id(species_id))


def calc_hp(base: int, level: int, ev: int = DEFAULT_EV, iv: int = DEFAULT_IV) -> int:
    """Showdown's HP formula (Shedinja, base 1, always has 1 HP)."""
    if base == 1:
        return 1
    return (2 * base + iv + ev // 4) * level // 100 + level + 10


def calc_stat(base: int, level: int, ev: int = DEFAULT_EV, iv: int = DEFAULT_IV) -> int:
    """Showdown's non-HP stat formula with a neutral nature."""
    return (2 * base + iv + ev // 4) * level // 100 + 5


def compute_stats(
    base_stats: Mapping[str, int],
    level: int,
    evs: Sequence[int] = (DEFAULT_EV,) * 6,
    ivs: Sequence[int] = (DEFAULT_IV,) * 6,
) -> dict[str, int]:
    """Return {hp, atk, def, spa, spd, spe} for the given base stats."""
    out: dict[str, int] = {}
    for index, key in enumerate(STAT_KEYS):
        base = int(base_stats[key])
        if key == "hp":
            out[key] = calc_hp(base, level, evs[index], ivs[index])
        else:
            out[key] = calc_stat(base, level, evs[index], ivs[index])
    return out


# Damaging moves whose damage does not depend on the user's Attack stat
# (``damage``/``damageCallback`` in Showdown's move data); used by the
# "minimize confusion damage" rule below.
_FIXED_DAMAGE_MOVES = frozenset(
    {
        "seismictoss", "nightshade", "sonicboom", "dragonrage", "psywave",
        "superfang", "naturesmadness", "ruination", "endeavor", "counter",
        "mirrorcoat", "metalburst", "comeuppance", "finalgambit", "bide",
        "guardianofalola",
    }
)
_CRASH_MOVES = frozenset({"axekick", "highjumpkick", "jumpkick", "supercellslam"})


def _type_multiplier(attacking: str, defending: Sequence[str]) -> float:
    chart = _gen_data().type_chart
    multiplier = 1.0
    for defender in defending:
        multiplier *= chart[defender.upper()].get(attacking.upper(), 1)
    return multiplier


def randbats_evs_ivs(
    species_id: str,
    level: int,
    moves: Sequence[str],
    ability: str | None,
    item: str | None,
) -> tuple[list[int], list[int]]:
    """EVs and IVs ``teams.ts`` would give this set (order hp, atk, def, spa, spd, spe).

    Randbats starts at 85 EVs / 31 IVs everywhere and then (1) lowers HP EVs in
    steps of 4 to land on an HP total that maximises Stealth Rock switch-ins (or
    meets Substitute / Belly Drum / Sitrus parity requirements), (2) zeroes Atk
    EVs and IVs for sets with no Attack-stat-using move, and (3) zeroes Speed
    EVs and IVs for Gyro Ball / Trick Room sets.
    """
    evs = [DEFAULT_EV] * 6
    ivs = [DEFAULT_IV] * 6
    entry = dex_entry(species_id)
    if entry is None:
        return evs, ivs
    base = entry["baseStats"]
    types = [t.upper() for t in entry["types"]]
    move_ids = {to_id(m) for m in moves}
    ability_id = to_id(ability)
    item_id = to_id(item)

    # (1) Optimal HP.
    sr_immune = ability_id == "magicguard" or item_id == "heavydutyboots"
    sr_weakness = 0
    if not sr_immune:
        multiplier = _type_multiplier("ROCK", types)
        sr_weakness = 0 if multiplier <= 0 else int(round(math.log2(multiplier)))
    if move_ids & _CRASH_MOVES:
        sr_weakness = 2
    if base["hp"] != 1:
        while evs[0] > 1:
            hp = (2 * base["hp"] + ivs[0] + evs[0] // 4 + 100) * level // 100 + 10
            if ("substitute" in move_ids and item_id == "sitrusberry") or species_id == "minior":
                if hp % 4 == 0:
                    break
            elif (move_ids & {"bellydrum", "filletaway", "shedtail"}) and (
                item_id == "sitrusberry" or ability_id == "gluttony"
            ):
                if hp % 2 == 0:
                    break
            elif "substitute" in move_ids and "endeavor" in move_ids:
                if hp % 4 > 0:
                    break
            else:
                if sr_weakness <= 0 or ability_id == "regenerator" or item_id in (
                    "leftovers",
                    "lifeorb",
                ):
                    break
                if item_id != "sitrusberry" and hp % (4 // sr_weakness) > 0:
                    break
                if item_id == "sitrusberry" and hp % (4 // sr_weakness) == 0:
                    break
            evs[0] -= 4

    # (2) Minimise confusion damage.
    moves_data = _gen_data().moves

    def no_attack_stat(move_id: str) -> bool:
        if move_id in _FIXED_DAMAGE_MOVES:
            return True
        if move_id == "shellsidearm":
            return False
        data = moves_data.get(move_id, {})
        if move_id == "terablast" and (
            species_id == "porygon2"
            or ability_id in ("contrary", "defiant")
            or "shiftgear" in move_ids
            or base["atk"] > base["spa"]
        ):
            return False
        return data.get("category") != "Physical" or move_id in ("bodypress", "foulplay")

    if move_ids and all(no_attack_stat(m) for m in move_ids) and "transform" not in move_ids:
        evs[1] = 0
        ivs[1] = 0

    # (3) Gyro Ball / Trick Room want minimum Speed.
    if "gyroball" in move_ids or "trickroom" in move_ids:
        evs[5] = 0
        ivs[5] = 0
    return evs, ivs


# ---------------------------------------------------------------------------
# Randbats team rules and unrevealed-slot sampling
# ---------------------------------------------------------------------------

_WEB_SETTERS = frozenset(
    {"ariados", "smeargle", "masquerain", "kricketune", "leavanny", "galvantula",
     "vikavolt", "ribombee", "araquanid", "spidops"}
)
_SCREEN_SETTERS = frozenset({"meowstic", "grimmsnarl", "ninetalesalola", "abomasnow"})
_SUN_SETTERS = frozenset({"ninetales", "torkoal", "groudon", "koraidon"})
# Pairs (A, B) that cannot share a team (teams.ts getPokemonCompatibility).
_INCOMPATIBLE: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"blissey"}), frozenset({"chansey"})),
    (frozenset({"illumise"}), frozenset({"volbeat"})),
    (_WEB_SETTERS, _WEB_SETTERS),
    (_SCREEN_SETTERS, _SCREEN_SETTERS),
    (frozenset({"toxicroak"}), _SUN_SETTERS),
)


@dataclass(frozen=True)
class SpeciesInfo:
    """Everything the team rules need to know about one randbats species."""

    species_id: str
    base_species: str
    level: int
    types: tuple[str, ...]
    weak: frozenset[str]
    double_weak: frozenset[str]
    freeze_dry_weak: bool


def species_info(species_id: str, level: int) -> SpeciesInfo | None:
    entry = dex_entry(species_id)
    if entry is None:
        return None
    types = tuple(t.upper() for t in entry["types"])
    weak: set[str] = set()
    double_weak: set[str] = set()
    for attacking in _gen_data().type_chart:
        multiplier = _type_multiplier(attacking, types)
        if multiplier > 1:
            weak.add(attacking)
        if multiplier > 2:
            double_weak.add(attacking)
    abilities = {to_id(a) for a in entry.get("abilities", {}).values()}
    if _type_multiplier("FIRE", types) == 1 and abilities & {"dryskin", "fluffy"}:
        weak.add("FIRE")
    ice = _type_multiplier("ICE", types)
    freeze_dry = ice > 1 or ("WATER" in types and ice > 0.25)
    return SpeciesInfo(
        species_id=to_id(species_id),
        base_species=to_id(entry.get("baseSpecies") or species_id),
        level=level,
        types=types,
        weak=frozenset(weak),
        double_weak=frozenset(double_weak),
        freeze_dry_weak=freeze_dry,
    )


def violates_team_rules(
    candidate: SpeciesInfo, team: Sequence[SpeciesInfo], *, team_size: int = TEAM_SIZE
) -> bool:
    """True if adding ``candidate`` breaks a randbats team-generation rule.

    Rules (``randomTeam`` in teams.ts): one of each base species; at most two
    Pokemon of any type; at most three weak to any type and one 4x-weak to any
    type; at most four weak to Freeze-Dry; at most one level-100 Pokemon; the
    incompatible-pair list; and no Zoroark in the last slot.
    """
    if any(member.base_species == candidate.base_species for member in team):
        return True
    if violates_uniqueness_free_rules(candidate, team, team_size=team_size):
        return True
    return False


def violates_uniqueness_free_rules(
    candidate: SpeciesInfo, team: Sequence[SpeciesInfo], *, team_size: int = TEAM_SIZE
) -> bool:
    """The soft (non-species-clause) randbats rules."""
    if candidate.base_species == "zoroark" and len(team) >= team_size - 1:
        return True
    for type_name in set(candidate.types):
        if sum(type_name in member.types for member in team) >= 2:
            return True
    for attacking in candidate.weak:
        if sum(attacking in member.weak for member in team) >= 3:
            return True
    for attacking in candidate.double_weak:
        if sum(attacking in member.double_weak for member in team) >= 1:
            return True
    if candidate.freeze_dry_weak and sum(m.freeze_dry_weak for m in team) >= 4:
        return True
    if candidate.level == 100 and sum(m.level == 100 for m in team) >= 1:
        return True
    for group_a, group_b in _INCOMPATIBLE:
        if candidate.species_id in group_b and any(m.species_id in group_a for m in team):
            return True
        if candidate.species_id in group_a and any(m.species_id in group_b for m in team):
            return True
    return False


class TeamSampler:
    """Samples species for unrevealed opposing slots.

    Species are drawn with probability proportional to the sum of their full-set
    counts (how often the species appeared in pkmn/randbats' generated teams),
    not uniformly. A draw is rejected when it would break a randbats team rule;
    after ``max_tries`` rejections the soft rules are dropped (the species
    clause is never dropped), mirroring how Showdown itself eventually relaxes.
    """

    def __init__(self, data: RandbatsData, max_tries: int = 40):
        self.data = data
        self.max_tries = max_tries
        ids: list[str] = []
        weights: list[float] = []
        self._info: dict[str, SpeciesInfo] = {}
        for species_id in data.species_ids():
            total = sum(full_set.count for full_set in data.full_sets(species_id))
            if total <= 0:
                continue
            info = species_info(species_id, data.level(species_id))
            if info is None:
                continue
            ids.append(species_id)
            weights.append(float(total))
            self._info[species_id] = info
        self.species_ids: tuple[str, ...] = tuple(ids)
        self.weights: tuple[float, ...] = tuple(weights)
        self._cum_weights = tuple(accumulate(weights))

    def info(self, species_id: str) -> SpeciesInfo | None:
        species_id = to_id(species_id)
        info = self._info.get(species_id)
        if info is None and self.data.has_species(species_id):
            info = species_info(species_id, self.data.level(species_id))
        return info

    def sample_species(
        self, team: Sequence[SpeciesInfo], rng: random.Random, *, team_size: int = TEAM_SIZE
    ) -> str | None:
        """Draw one species id that fits ``team`` (a list of SpeciesInfo)."""
        if not self.species_ids:
            return None
        taken = {member.base_species for member in team}
        fallback: str | None = None
        for _ in range(self.max_tries):
            species_id = rng.choices(self.species_ids, cum_weights=self._cum_weights)[0]
            info = self._info[species_id]
            if info.base_species in taken:
                continue
            fallback = fallback or species_id
            if not violates_uniqueness_free_rules(info, team, team_size=team_size):
                return species_id
        # Relax the soft rules; keep the species clause.
        if fallback is not None:
            return fallback
        for species_id in self.species_ids:
            if self._info[species_id].base_species not in taken:
                return species_id
        return None

    def sample_set(self, species_id: str, rng: random.Random) -> SampledSet:
        """A whole set for an unrevealed species (no evidence to condition on)."""
        return SetTracker(self.data, species_id).sample(rng)


@dataclass
class OpponentSlot:
    """One opposing team member in a sampled world."""

    species_id: str
    sampled: SampledSet
    mon: Any | None = None  # poke-env Pokemon if this slot has been revealed


def sample_opponent_team(
    battle: Any,
    trackers: Mapping[str, SetTracker],
    sampler: TeamSampler,
    rng: random.Random,
    *,
    team_size: int = TEAM_SIZE,
) -> list[OpponentSlot]:
    """Build one determinized opponent team for ``battle``.

    The active Pokemon (if any) comes first, then the other revealed Pokemon in
    battle order, then sampled unrevealed slots up to ``team_size``.
    """
    revealed = list(battle.opponent_team.items())
    active = battle.opponent_active_pokemon
    revealed.sort(key=lambda item: 0 if item[1] is active else 1)
    slots: list[OpponentSlot] = []
    for identifier, mon in revealed:
        tracker = trackers.get(str(identifier))
        if tracker is None:
            slot = _slot_without_tracker(mon, sampler, rng)
            if slot is None:
                continue
        else:
            sampled = tracker.sample(rng)
            slot = OpponentSlot(to_id(sampled.species), sampled, mon)
        slots.append(slot)
    team = [info for info in (sampler.info(s.species_id) for s in slots) if info]
    while len(slots) < team_size:
        species_id = sampler.sample_species(team, rng, team_size=team_size)
        if species_id is None:
            break
        sampled = sampler.sample_set(species_id, rng)
        slots.append(OpponentSlot(species_id, sampled, None))
        info = sampler.info(species_id)
        if info:
            team.append(info)
    return slots


def _slot_without_tracker(mon: Any, sampler: TeamSampler, rng: random.Random) -> OpponentSlot | None:
    """A revealed Pokemon the opponent model could not map to a randbats species."""
    species_id = to_id(getattr(mon, "species", ""))
    if dex_entry(species_id) is None:
        return None
    if sampler.data.has_species(species_id):
        sampled = sampler.sample_set(species_id, rng)
    else:
        revealed = tuple(getattr(mon, "moves", {}) or ())
        sampled = SampledSet(
            species=species_id,
            level=int(getattr(mon, "level", 80) or 80),
            role="",
            moves=revealed,
            ability=getattr(mon, "ability", None),
            item=None,
            tera_type=None,
        )
    return OpponentSlot(species_id, sampled, mon)


# ---------------------------------------------------------------------------
# Battle -> poke_engine.State
# ---------------------------------------------------------------------------

_STATUS_IDS = {
    Status.BRN: "burn",
    Status.FRZ: "freeze",
    Status.PAR: "paralyze",
    Status.PSN: "poison",
    Status.SLP: "sleep",
    Status.TOX: "toxic",
}

_WEATHER_IDS = {
    Weather.SUNNYDAY: "sun",
    Weather.RAINDANCE: "rain",
    Weather.SANDSTORM: "sand",
    Weather.HAIL: "hail",
    Weather.SNOW: "snow",
    Weather.DESOLATELAND: "harshsun",
    Weather.PRIMORDIALSEA: "heavyrain",
}
_PERMANENT_WEATHER = {Weather.DESOLATELAND, Weather.PRIMORDIALSEA}

_TERRAIN_IDS = {
    Field.ELECTRIC_TERRAIN: "electricterrain",
    Field.PSYCHIC_TERRAIN: "psychicterrain",
    Field.MISTY_TERRAIN: "mistyterrain",
    Field.GRASSY_TERRAIN: "grassyterrain",
}

# poke-env Effect name -> engine PokemonVolatileStatus id. Whitelist of statuses
# that persist across turns; turn-scoped ones are deliberately absent.
_EFFECT_VOLATILES = {
    "AQUA_RING": "aquaring", "ATTRACT": "attract", "AUTOTOMIZE": "autotomize",
    "CHARGE": "charge", "CONFUSION": "confusion", "CURSE": "curse",
    "DEFENSE_CURL": "defensecurl", "DISABLE": "disable", "EMBARGO": "embargo",
    "ENCORE": "encore", "FLASH_FIRE": "flashfire", "FOCUS_ENERGY": "focusenergy",
    "FORESIGHT": "foresight", "GASTRO_ACID": "gastroacid", "HEAL_BLOCK": "healblock",
    "IMPRISON": "imprison", "INGRAIN": "ingrain", "LASER_FOCUS": "laserfocus",
    "LEECH_SEED": "leechseed", "LOCKED_MOVE": "lockedmove", "MAGNET_RISE": "magnetrise",
    "MINIMIZE": "minimize", "MIRACLE_EYE": "miracleeye", "MUST_RECHARGE": "mustrecharge",
    "NIGHTMARE": "nightmare", "NO_RETREAT": "noretreat", "OCTOLOCK": "octolock",
    "POWER_TRICK": "powertrick", "SALT_CURE": "saltcure", "SLOW_START": "slowstart",
    "SMACK_DOWN": "smackdown", "SUBSTITUTE": "substitute", "SYRUP_BOMB": "syrupbomb",
    "TAR_SHOT": "tarshot", "TAUNT": "taunt", "TELEKINESIS": "telekinesis",
    "THROAT_CHOP": "throatchop", "TORMENT": "torment", "UPROAR": "uproar",
    "YAWN": "yawn",
    "BIND": "partiallytrapped", "CLAMP": "partiallytrapped", "FIRE_SPIN": "partiallytrapped",
    "INFESTATION": "partiallytrapped", "MAGMA_STORM": "partiallytrapped",
    "SAND_TOMB": "partiallytrapped", "SNAP_TRAP": "partiallytrapped",
    "THUNDER_CAGE": "partiallytrapped", "WHIRLPOOL": "partiallytrapped",
    "WRAP": "partiallytrapped", "PARTIALLY_TRAPPED": "partiallytrapped",
    "PERISH0": "perish4", "PERISH1": "perish1", "PERISH2": "perish2", "PERISH3": "perish3",
    "PROTOSYNTHESISATK": "protosynthesisatk", "PROTOSYNTHESISDEF": "protosynthesisdef",
    "PROTOSYNTHESISSPA": "protosynthesisspa", "PROTOSYNTHESISSPD": "protosynthesisspd",
    "PROTOSYNTHESISSPE": "protosynthesisspe",
    "QUARKDRIVEATK": "quarkdriveatk", "QUARKDRIVEDEF": "quarkdrivedef",
    "QUARKDRIVESPA": "quarkdrivespa", "QUARKDRIVESPD": "quarkdrivespd",
    "QUARKDRIVESPE": "quarkdrivespe",
    "STOCKPILE": "stockpile", "STOCKPILE1": "stockpile", "STOCKPILE2": "stockpile",
    "STOCKPILE3": "stockpile",
}
_EFFECT_VOLATILES = {
    getattr(Effect, name): value
    for name, value in _EFFECT_VOLATILES.items()
    if hasattr(Effect, name)
}

# Two-turn moves in flight (poke-env ``preparing_move``) -> engine volatile.
_CHARGING_VOLATILES = {
    "fly": "fly", "dig": "dig", "dive": "dive", "bounce": "bounce",
    "phantomforce": "phantomforce", "shadowforce": "shadowforce", "skydrop": "skydrop",
    "solarbeam": "solarbeam", "solarblade": "solarblade", "skullbash": "skullbash",
    "razorwind": "razorwind", "skyattack": "skyattack", "meteorbeam": "meteorbeam",
    "electroshot": "electroshot", "geomancy": "geomancy", "freezeshock": "freezeshock",
    "iceburn": "iceburn",
}

_BOOST_FIELDS = {
    "atk": "attack_boost", "def": "defense_boost", "spa": "special_attack_boost",
    "spd": "special_defense_boost", "spe": "speed_boost",
    "accuracy": "accuracy_boost", "evasion": "evasion_boost",
}

_move_pp_cache: dict[str, int] = {}


def _max_pp(move_id: str) -> int:
    cached = _move_pp_cache.get(move_id)
    if cached is not None:
        return cached
    try:
        value = int(PEMove(move_id, gen=_GEN).max_pp)
    except Exception:
        value = 16
    _move_pp_cache[move_id] = value
    return value


def _type_id(value: Any) -> str:
    name = getattr(value, "name", value)
    return str(name).lower() if name else "typeless"


def _types_pair(types: Sequence[Any]) -> tuple[str, str]:
    names = [_type_id(t) for t in types if t is not None]
    first = names[0] if names else "typeless"
    second = names[1] if len(names) > 1 else "typeless"
    return first, second


def _clamp8(value: int) -> int:
    return max(-128, min(127, int(value)))


def _engine_move(move_id: str, pp: int, disabled: bool = False) -> pe.Move:
    return pe.Move(id=to_id(move_id), pp=_clamp8(pp), disabled=disabled)


def _remaining_turns(start_turn: int, current_turn: int, duration: int) -> int:
    """Turns of an effect left when its start turn and total duration are known."""
    return max(1, duration - (current_turn - start_turn))


def _side_conditions(
    conditions: Mapping[SideCondition, int],
    turn: int,
    toxic_count: int,
    protect_counter: int,
) -> pe.SideConditions:
    def timed(condition: SideCondition, duration: int) -> int:
        start = conditions.get(condition)
        return 0 if start is None else _remaining_turns(start, turn, duration)

    def layers(condition: SideCondition, maximum: int) -> int:
        return min(maximum, int(conditions.get(condition, 0)))

    return pe.SideConditions(
        spikes=layers(SideCondition.SPIKES, 3),
        toxic_spikes=layers(SideCondition.TOXIC_SPIKES, 2),
        stealth_rock=1 if SideCondition.STEALTH_ROCK in conditions else 0,
        sticky_web=1 if SideCondition.STICKY_WEB in conditions else 0,
        tailwind=timed(SideCondition.TAILWIND, 4),
        lucky_chant=timed(SideCondition.LUCKY_CHANT, 5),
        reflect=timed(SideCondition.REFLECT, 5),
        light_screen=timed(SideCondition.LIGHT_SCREEN, 5),
        aurora_veil=timed(SideCondition.AURORA_VEIL, 5),
        safeguard=timed(SideCondition.SAFEGUARD, 5),
        mist=timed(SideCondition.MIST, 5),
        protect=min(127, protect_counter),
        toxic_count=min(127, toxic_count),
    )


@dataclass
class BuiltState:
    """A poke_engine State plus what is needed to map choices back to orders."""

    state: Any  # poke_engine.State (normalised through a to_string round trip)
    our_mons: list[Any]  # poke-env Pokemon in engine index order
    our_move_ids: list[str]  # lowercase engine move ids of our active Pokemon
    unknown_ids: list[str] = field(default_factory=list)

    def to_string(self) -> str:
        return self.state.to_string()

    def choice_to_order(self, choice: str, battle: Any) -> BattleOrder | None:
        """Map an engine choice string to a legal poke-env order, else None."""
        choice = choice.strip().lower()
        if not choice or choice in ("none", "no move"):
            return None
        if choice.startswith("switch "):
            target = choice[len("switch "):].strip()
            for index, mon in enumerate(self.our_mons):
                if self._engine_name(index) == target:
                    if mon in battle.available_switches:
                        return SingleBattleOrder(mon)
                    return None
            return None
        tera = choice.endswith("-tera")
        move_id = choice[: -len("-tera")] if tera else choice
        for move in battle.available_moves:
            if self._normalise_move_id(move.id) == move_id:
                if tera and not battle.can_tera:
                    return None
                return SingleBattleOrder(move, terastallize=tera)
        return None

    # engine names are the lowercase ids of the round-tripped state
    def _engine_name(self, index: int) -> str:
        return to_id(self.state.side_one.pokemon[index].id)

    @staticmethod
    def _normalise_move_id(move_id: str) -> str:
        return to_id(move_id)


def _own_moves(battle: Any, mon: Any, is_active: bool) -> list[pe.Move]:
    request_moves: list[Mapping[str, Any]] = []
    if is_active:
        active_request = (battle.last_request or {}).get("active") or [{}]
        request_moves = list(active_request[0].get("moves", ()) or ())
    out: list[pe.Move] = []
    if request_moves:
        for entry in request_moves:
            move_id = to_id(entry.get("id") or entry.get("move"))
            if not move_id or move_id in ("recharge", "struggle"):
                continue
            pp = entry.get("pp")
            if pp is None:
                known = mon.moves.get(move_id)
                pp = known.current_pp if known is not None else _max_pp(move_id)
            out.append(_engine_move(move_id, pp, bool(entry.get("disabled", False))))
        return out[:4]
    for move_id, move in list(mon.moves.items())[:4]:
        out.append(_engine_move(move_id, move.current_pp))
    return out


# The engine's Gen 9 sleep lasts at most 3 turns; a larger counter makes its
# wake-up chance 1/0 (a NaN probability that panics MCTS), so clamp.
_MAX_SLEEP_TURNS = 3


def _sleep_turns(mon: Any, status: str) -> int:
    if status != "sleep":
        return 0
    return max(0, min(_MAX_SLEEP_TURNS, int(mon.status_counter or 0)))


def _status_of(mon: Any) -> str:
    status = mon.status
    return _STATUS_IDS.get(status, "none") if status is not None else "none"


def _volatiles_for(mon: Any) -> tuple[set[str], dict[str, int]]:
    """Engine volatile ids and duration counters for an active Pokemon."""
    volatiles: set[str] = set()
    durations = {"confusion": 0, "encore": 0, "lockedmove": 0, "slowstart": 0, "taunt": 0, "yawn": 0}
    for effect, counter in mon.effects.items():
        volatile = _EFFECT_VOLATILES.get(effect)
        if volatile is None:
            continue
        volatiles.add(volatile)
        counter = int(counter or 0)
        if volatile == "confusion":
            durations["confusion"] = max(0, min(counter, 4))
        elif volatile == "encore":
            durations["encore"] = max(0, min(counter, 2))
        elif volatile == "lockedmove":
            durations["lockedmove"] = max(0, min(counter, 1))
        elif volatile == "taunt":
            durations["taunt"] = max(0, min(counter, 2))
        elif volatile == "slowstart":
            durations["slowstart"] = max(1, 5 - counter)
        elif volatile == "yawn":
            durations["yawn"] = 1 if counter >= 1 else 0
    preparing = getattr(mon, "preparing_move", None)
    if preparing is not None:
        charging = _CHARGING_VOLATILES.get(to_id(getattr(preparing, "id", preparing)))
        if charging:
            volatiles.add(charging)
    return volatiles, durations


def _last_used_move(mon: Any, engine_moves: Sequence[pe.Move], active_index: int) -> str:
    if getattr(mon, "first_turn", False):
        return f"switch:{active_index}"
    for index, engine_move in enumerate(engine_moves):
        poke_env_move = mon.moves.get(engine_move.id)
        if poke_env_move is not None and getattr(poke_env_move, "_is_last_used", False):
            return f"move:{index}"
    return "move:none"


def _own_pokemon(battle: Any, mon: Any, is_active: bool) -> tuple[pe.Pokemon, list[pe.Move]]:
    request = getattr(mon, "_last_request", None) or {}
    level = int(mon.level or 100)
    species_id = to_id(mon.species)
    entry = dex_entry(species_id)
    stats = {key: mon.stats.get(key) for key in STAT_KEYS}
    max_hp = int(mon.max_hp or 0)
    if entry is not None and any(stats[key] is None for key in STAT_KEYS[1:]):
        computed = compute_stats(entry["baseStats"], level)
        stats = {key: stats[key] if stats[key] is not None else computed[key] for key in STAT_KEYS}
        if not max_hp:
            max_hp = computed["hp"]
    hp = 0 if mon.fainted else int(mon.current_hp)
    max_hp = max(max_hp, hp, 1)
    ability = to_id(mon.ability) or to_id(mon.base_ability)
    base_ability = to_id(mon.base_ability) or ability
    item = to_id(mon.item) if mon.item and mon.item != "unknown_item" else "none"
    tera_name = request.get("teraType") or getattr(mon.tera_type, "name", None)
    tera_type = to_id(tera_name) or "typeless"
    status = _status_of(mon)
    moves = _own_moves(battle, mon, is_active)
    weight = float(entry["weightkg"]) if entry else 0.0
    pokemon = pe.Pokemon(
        id=species_id,
        level=level,
        types=_types_pair(mon.base_types),
        base_types=_types_pair(mon.base_types),
        hp=hp,
        maxhp=max_hp,
        ability=ability or "none",
        base_ability=base_ability or "none",
        item=item,
        attack=int(stats["atk"] or 100),
        defense=int(stats["def"] or 100),
        special_attack=int(stats["spa"] or 100),
        special_defense=int(stats["spd"] or 100),
        speed=int(stats["spe"] or 100),
        status=status,
        sleep_turns=_sleep_turns(mon, status),
        weight_kg=weight,
        terastallized=bool(mon.is_terastallized or request.get("terastallized")),
        tera_type=tera_type,
        moves=moves,
    )
    return pokemon, moves


def _opponent_pokemon(slot: OpponentSlot, battle: Any) -> tuple[pe.Pokemon, list[pe.Move], Any | None]:
    sampled = slot.sampled
    mon = slot.mon
    entry = dex_entry(slot.species_id)
    level = int(sampled.level)
    move_ids = [to_id(m) for m in sampled.moves][:4]
    ability = to_id(sampled.ability)
    item = to_id(sampled.item)
    if mon is not None:
        if mon.item is None or mon.item == "":
            item = ""
        if mon.is_terastallized and mon.tera_type is not None:
            tera_type = _type_id(mon.tera_type)
        else:
            tera_type = to_id(sampled.tera_type) or "typeless"
    else:
        tera_type = to_id(sampled.tera_type) or "typeless"
    if entry is not None:
        evs, ivs = randbats_evs_ivs(slot.species_id, level, move_ids, ability, item)
        stats = compute_stats(entry["baseStats"], level, evs, ivs)
        types = _types_pair(mon.base_types) if mon is not None else _types_pair(
            [t.lower() for t in entry["types"]]
        )
        weight = float(entry["weightkg"])
    else:
        evs, ivs = [DEFAULT_EV] * 6, [DEFAULT_IV] * 6
        stats = {key: 100 for key in STAT_KEYS}
        types = ("typeless", "typeless")
        weight = 0.0
    max_hp = max(1, stats["hp"])
    hp = max_hp
    status = "none"
    sleep_turns = 0
    if mon is not None:
        if mon.fainted:
            hp = 0
        else:
            hp = max(1, int(round(max_hp * float(mon.current_hp_fraction)))) if mon.current_hp else 0
        status = _status_of(mon)
        sleep_turns = _sleep_turns(mon, status)
    moves: list[pe.Move] = []
    for move_id in move_ids:
        pp = _max_pp(move_id)
        if mon is not None:
            known = mon.moves.get(move_id)
            if known is not None:
                pp = known.current_pp
        moves.append(_engine_move(move_id, pp))
    pokemon = pe.Pokemon(
        id=slot.species_id,
        level=level,
        types=types,
        base_types=types,
        hp=hp,
        maxhp=max_hp,
        ability=ability or "none",
        base_ability=ability or "none",
        item=item or "none",
        evs=tuple(evs),
        attack=stats["atk"],
        defense=stats["def"],
        special_attack=stats["spa"],
        special_defense=stats["spd"],
        speed=stats["spe"],
        status=status,
        sleep_turns=sleep_turns,
        weight_kg=weight,
        terastallized=bool(mon is not None and mon.is_terastallized),
        tera_type=tera_type,
        moves=moves,
    )
    return pokemon, moves, mon


def _make_side(
    ordered: Sequence[tuple[pe.Pokemon, list[pe.Move], Any | None]],
    active_mon: Any | None,
    battle: Any,
    conditions: Mapping[SideCondition, int],
    *,
    is_ours: bool,
) -> pe.Side:
    """Assemble a Side: active Pokemon at index 0, then the rest."""
    pokemon = [entry[0] for entry in ordered]
    kwargs: dict[str, Any] = {}
    toxic_count = 0
    protect_counter = 0
    if active_mon is not None and not active_mon.fainted:
        volatiles, durations = _volatiles_for(active_mon)
        kwargs["volatile_statuses"] = volatiles
        kwargs["volatile_status_durations"] = pe.VolatileStatusDurations(**durations)
        if "substitute" in volatiles:
            kwargs["substitute_health"] = max(1, pokemon[0].maxhp // 4)
        for stat, boost in active_mon.boosts.items():
            attr = _BOOST_FIELDS.get(stat)
            if attr:
                kwargs[attr] = max(-6, min(6, int(boost)))
        if active_mon.status == Status.TOX:
            toxic_count = min(127, int(active_mon.status_counter or 0))
        protect_counter = int(getattr(active_mon, "protect_counter", 0) or 0)
        last_used = _last_used_move(active_mon, ordered[0][1], 0)
        kwargs["last_used_move"] = last_used
        has_last_move = last_used.startswith("move:") and last_used != "move:none"
        if "encore" in volatiles and not has_last_move:
            # The engine panics on Encore without a known last move.
            volatiles.discard("encore")
            kwargs["volatile_status_durations"] = pe.VolatileStatusDurations(
                **{**durations, "encore": 0}
            )
    if is_ours:
        kwargs["force_switch"] = bool(battle.force_switch) and not (
            active_mon is None or active_mon.fainted
        )
        kwargs["force_trapped"] = bool(battle.trapped)
    side_conditions = _side_conditions(conditions, battle.turn, toxic_count, protect_counter)
    return pe.Side(
        pokemon=pokemon,
        side_conditions=side_conditions,
        active_index="0",
        **kwargs,
    )


def build_state(
    battle: Any,
    opponent_team: Sequence[OpponentSlot],
) -> BuiltState:
    """Convert ``battle`` and one sampled ``opponent_team`` into a ``BuiltState``."""
    # --- our side (exact) ---
    our_active = battle.active_pokemon
    ours_sorted = sorted(
        battle.team.values(), key=lambda mon: 0 if mon is our_active else 1
    )
    our_entries = []
    for mon in ours_sorted[:TEAM_SIZE]:
        pokemon, moves = _own_pokemon(battle, mon, mon is our_active)
        our_entries.append((pokemon, moves, mon))
    side_one = _make_side(
        our_entries, our_active, battle, battle.side_conditions, is_ours=True
    )

    # --- opponent (sampled) ---
    opp_active = battle.opponent_active_pokemon
    opp_entries = [_opponent_pokemon(slot, battle) for slot in opponent_team[:TEAM_SIZE]]
    opp_active_mon = None
    if opp_entries:
        opp_active_mon = opp_entries[0][2] if opp_entries[0][2] is opp_active else None
    side_two = _make_side(
        opp_entries, opp_active_mon, battle, battle.opponent_side_conditions, is_ours=False
    )

    # --- field ---
    weather_id, weather_turns = "none", 0
    for weather, start_turn in battle.weather.items():
        mapped = _WEATHER_IDS.get(weather)
        if mapped:
            weather_id = mapped
            weather_turns = (
                0 if weather in _PERMANENT_WEATHER else _remaining_turns(start_turn, battle.turn, 5)
            )
            break
    terrain_id, terrain_turns = "none", 0
    trick_room, trick_room_turns = False, 0
    for field_, start_turn in battle.fields.items():
        if field_ in _TERRAIN_IDS:
            terrain_id = _TERRAIN_IDS[field_]
            terrain_turns = _remaining_turns(start_turn, battle.turn, 5)
        elif field_ == Field.TRICK_ROOM:
            trick_room = True
            trick_room_turns = _remaining_turns(start_turn, battle.turn, 5)

    raw_state = pe.State(
        side_one=side_one,
        side_two=side_two,
        weather=weather_id,
        weather_turns_remaining=weather_turns,
        terrain=terrain_id,
        terrain_turns_remaining=terrain_turns,
        trick_room=trick_room,
        trick_room_turns_remaining=trick_room_turns,
    )
    # Round trip through the engine's own serialisation: this validates every
    # id (unknown species/moves/abilities/items decay to "none") and gives us
    # the engine-normalised names used by ``choice_to_order``.
    state = pe.State.from_string(raw_state.to_string())
    unknown = _find_unknown_ids(raw_state, state)
    if unknown:
        logger.debug("Engine did not recognise ids: %s", unknown)
    return BuiltState(
        state=state,
        our_mons=[entry[2] for entry in our_entries],
        our_move_ids=[to_id(move.id) for move in our_entries[0][1]] if our_entries else [],
        unknown_ids=unknown,
    )


def _find_unknown_ids(requested: Any, normalised: Any) -> list[str]:
    unknown: list[str] = []
    for side_name in ("side_one", "side_two"):
        sent = getattr(requested, side_name).pokemon
        got = getattr(normalised, side_name).pokemon
        for sent_mon, got_mon in zip(sent, got):
            if to_id(sent_mon.id) != to_id(got_mon.id) and sent_mon.hp > 0:
                unknown.append(f"species:{sent_mon.id}")
            if to_id(sent_mon.ability) != to_id(got_mon.ability):
                unknown.append(f"ability:{sent_mon.ability}")
            if to_id(sent_mon.item) != to_id(got_mon.item):
                unknown.append(f"item:{sent_mon.item}")
            for sent_move, got_move in zip(sent_mon.moves, got_mon.moves):
                if sent_move.id and to_id(sent_move.id) != to_id(got_move.id):
                    unknown.append(f"move:{sent_move.id}")
    return unknown
