"""Random Battle set data and opponent set inference.

The data is published by pkmn/randbats.  It is cached locally so normal loads
do not depend on network access.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


from showdown_bot.inference import (
    Constraint,
    DamageObs,
    InferenceTracker,
    SpeedObs,
    constraint_allows,
    resolve_speed,
)

logger = logging.getLogger(__name__)

#: Wall-clock seconds spent in event-level inference (``OpponentModel.observe``'s
#: negative-evidence part), summed over every model in this process.
INFERENCE_SECONDS = [0.0]

_DATA_URL = "https://pkmn.github.io/randbats/data"
_DEFAULT_CACHE_DIR = Path(".cache") / "randbats"

_TERA_TYPE_IDS = frozenset(
    {
        "normal", "fire", "water", "electric", "grass", "ice", "fighting",
        "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
        "dragon", "dark", "steel", "fairy", "stellar",
    }
)


def to_id(name: Any) -> str:
    """Return the Showdown id for a display name or enum-like value."""
    if name is None:
        return ""
    # poke-env represents Tera types as enum members; their names are the
    # Showdown type names (for example, ``FIRE``).
    value = getattr(name, "name", name)
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _display_name(value: Any) -> str:
    if value is None:
        return ""
    enum_name = getattr(value, "name", None)
    if enum_name is not None:
        return str(enum_name)
    value_id = getattr(value, "id", None)
    if value_id is not None:
        return str(value_id)
    return str(value)


def _weight(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed) or parsed < 0:
        return default
    return parsed


def _options(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Mapping):
        values = tuple(value)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ()

    result: list[str] = []
    seen: set[str] = set()
    for option in values:
        display = _display_name(option)
        option_id = to_id(display)
        if option_id and option_id not in seen:
            result.append(display)
            seen.add(option_id)
    return tuple(result)


def _option_weights(value: Any) -> Mapping[str, float]:
    if isinstance(value, Mapping):
        pairs = value.items()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        pairs = ((option, 1.0) for option in value)
    else:
        pairs = ()
    weights: dict[str, float] = {}
    for option, amount in pairs:
        option_id = to_id(option)
        if option_id:
            weights[option_id] = _weight(amount)
    return MappingProxyType(weights)


def _normalize_kind(kind: str) -> str:
    normalized = kind.replace("_", "").lower()
    aliases = {
        "ability": "abilities",
        "abilities": "abilities",
        "item": "items",
        "items": "items",
        "teratype": "teraTypes",
        "teratypes": "teraTypes",
        "move": "moves",
        "moves": "moves",
    }
    try:
        return aliases[normalized]
    except KeyError:
        raise KeyError(f"Unknown role option kind: {kind}") from None


@dataclass(frozen=True)
class RoleData(Mapping[str, Any]):
    """One legal randbats role, with Showdown options and usage weights."""

    name: str
    abilities: tuple[str, ...]
    items: tuple[str, ...] = field()
    tera_types: tuple[str, ...]
    moves: tuple[str, ...]
    weight: float
    weights: Mapping[str, Mapping[str, float]]

    def options(self, kind: str) -> tuple[str, ...]:
        key = _normalize_kind(kind)
        if key == "abilities":
            return self.abilities
        if key == "items":
            return self.items
        if key == "teraTypes":
            return self.tera_types
        return self.moves

    def option_weight(self, kind: str, option: Any) -> float:
        key = _normalize_kind(kind)
        return self.weights[key].get(to_id(option), 1.0)

    def __getitem__(self, key: str) -> Any:
        if key in {"teraTypes", "tera_types"}:
            return self.tera_types
        if key in {"ability_weights", "item_weights", "tera_type_weights", "move_weights"}:
            weight_key = {
                "ability_weights": "abilities",
                "item_weights": "items",
                "tera_type_weights": "teraTypes",
                "move_weights": "moves",
            }[key]
            return self.weights[weight_key]
        if key in {"abilities", "items", "moves", "weight", "weights"}:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("abilities", "items", "teraTypes", "moves", "weight", "weights"))

    def __len__(self) -> int:
        return 6


@dataclass(frozen=True)
class _SpeciesData:
    name: str
    level: int
    roles: Mapping[str, RoleData]


@dataclass(frozen=True)
class FullSet:
    """One exact randbats set (level, item, ability, all moves, tera type).

    Foul Play samples these jointly from pkmn/randbats' "full" data instead
    of drawing each option independently from per-role marginals, because
    the joint distribution has real correlations (for example Abomasnow's
    "Bulky Support" role never actually uses all of Earthquake, Ice Shard
    and Wood Hammer together, only two of the three) that per-move sampling
    can't reproduce. ``count`` is the number of times this exact set
    appeared across pkmn/randbats' 100,000 generated teams, directly
    comparable across sets and even across sibling species (see
    ``_hidden_forme_species``), so weighting by raw ``count`` mixes them in
    the right proportion without renormalizing per species first.
    """

    level: int
    item: str | None
    ability: str
    moves: tuple[str, ...]
    tera_type: str
    count: float


def _parse_full_set_key(key: str) -> tuple[int, str, str, tuple[str, ...], str] | None:
    """Split one ``full/<format>.json`` key into its component ids.

    Keys look like ``level,item,ability,move1,move2,move3,move4[,tera]``,
    already Showdown ids for item/ability/moves/tera (an empty item field
    means no held item). The move count is normally four but can be fewer
    -- Ditto's only set carries just Transform -- so the split anchors from
    the right: the last field is always a real tera type id (every entry in
    the real file has one), and every field between ability and tera is a
    move. Returns ``None`` for a key this parser can't make sense of, so the
    caller can skip it rather than crash a battle on a malformed entry.
    """
    parts = key.split(",")
    if len(parts) < 5:
        return None
    try:
        level = int(parts[0])
    except ValueError:
        return None
    item_id = to_id(parts[1])
    ability_id = to_id(parts[2])
    tera_id = to_id(parts[-1])
    if tera_id not in _TERA_TYPE_IDS:
        return None
    move_ids = tuple(to_id(move) for move in parts[3:-1] if to_id(move))
    if not move_ids:
        return None
    return level, item_id, ability_id, move_ids, tera_id


def _parse_full_sets(
    full_payload: Any,
    move_display: Mapping[str, str],
    item_display: Mapping[str, str],
    ability_display: Mapping[str, str],
) -> dict[str, tuple[FullSet, ...]]:
    """Parse a ``full/<format>.json`` payload into per-species ``FullSet``s.

    The full file only gives ids; ``*_display`` map those ids back to the
    display names already harvested from the regular sets file's role
    options, so ``FullSet`` stays in the same display-name style as
    ``RoleData``/``SampledSet``. An id with no known display name (full-set
    data for an option no listed role happens to use) falls back to the bare
    id rather than crashing.
    """
    if not isinstance(full_payload, Mapping):
        return {}
    full_sets: dict[str, tuple[FullSet, ...]] = {}
    for species_name, species_sets in full_payload.items():
        species_id = to_id(species_name)
        if not species_id or not isinstance(species_sets, Mapping):
            continue
        parsed_sets: list[FullSet] = []
        for raw_key, raw_count in species_sets.items():
            parsed_key = _parse_full_set_key(str(raw_key))
            if parsed_key is None:
                continue
            level, item_id, ability_id, move_ids, tera_id = parsed_key
            parsed_sets.append(
                FullSet(
                    level=level,
                    item=item_display.get(item_id, item_id) if item_id else None,
                    ability=ability_display.get(ability_id, ability_id),
                    moves=tuple(
                        move_display.get(move_id, move_id) for move_id in move_ids
                    ),
                    tera_type=tera_id.capitalize(),
                    count=_weight(raw_count),
                )
            )
        if parsed_sets:
            full_sets[species_id] = tuple(parsed_sets)
    return full_sets


class RandbatsData:
    """Parsed sets and usage stats indexed by Showdown species id."""

    def __init__(
        self,
        species: Mapping[str, _SpeciesData],
        full_sets: Mapping[str, tuple[FullSet, ...]] = MappingProxyType({}),
    ):
        self._species = MappingProxyType(dict(species))
        self._full_sets = MappingProxyType(dict(full_sets))

    @classmethod
    def load(
        cls,
        format: str = "gen9randombattle",
        refresh: bool = False,
        cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    ) -> RandbatsData:
        """Load a format's sets and stats, fetching either missing cache file.

        The full joint-set file is fetched/cached the same way, under
        ``<cache_dir>/full/<format>.json``. Unlike the sets and stats files,
        a missing or unparsable full file never fails the load: full-set
        sampling simply has nothing to match and ``SetTracker.sample``
        falls back to the role-based sampler, so loading never crashes a
        battle for lack of (or stale/unreachable) full data.
        """
        cache_path = Path(cache_dir)
        sets_path = cache_path / f"{format}.json"
        stats_path = cache_path / "stats" / f"{format}.json"
        full_path = cache_path / "full" / f"{format}.json"
        sets_url = f"{_DATA_URL}/{format}.json"
        stats_url = f"{_DATA_URL}/stats/{format}.json"
        full_url = f"{_DATA_URL}/full/{format}.json"

        sets_payload = _read_cached_json(sets_path, sets_url, refresh)
        stats_payload = _read_cached_json(stats_path, stats_url, refresh)
        try:
            full_payload = _read_cached_json(full_path, full_url, refresh)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not load full randbats sets for %s (%s); full-set "
                "sampling will fall back to role-based sampling",
                format,
                exc,
            )
            full_payload = {}
        return cls._from_payloads(sets_payload, stats_payload, full_payload)

    @classmethod
    def _from_payloads(
        cls,
        sets_payload: Any,
        stats_payload: Any,
        full_payload: Any = MappingProxyType({}),
    ) -> RandbatsData:
        if not isinstance(sets_payload, Mapping) or not isinstance(stats_payload, Mapping):
            raise ValueError("randbats sets and stats must each be JSON objects")

        stats_species_by_id = {
            to_id(name): value
            for name, value in stats_payload.items()
            if to_id(name) and isinstance(value, Mapping)
        }
        # Harvested while walking species/role options below, so the full
        # file's ids (moves/items/abilities only, no display names) can be
        # mapped back to the same display-name style as RoleData/SampledSet.
        move_display: dict[str, str] = {}
        item_display: dict[str, str] = {}
        ability_display: dict[str, str] = {}

        def _record_display_names(target: dict[str, str], raw: Any) -> None:
            for option in _options(raw):
                target.setdefault(to_id(option), option)

        parsed: dict[str, _SpeciesData] = {}
        for species_name, species_raw in sets_payload.items():
            species_id = to_id(species_name)
            if not species_id or not isinstance(species_raw, Mapping):
                continue
            _record_display_names(ability_display, species_raw.get("abilities", ()))
            _record_display_names(item_display, species_raw.get("items", ()))
            stats_raw = stats_species_by_id.get(species_id, {})
            stats_roles_raw = stats_raw.get("roles", {})
            if not isinstance(stats_roles_raw, Mapping):
                stats_roles_raw = {}
            stats_roles_by_id = {
                to_id(name): value
                for name, value in stats_roles_raw.items()
                if to_id(name) and isinstance(value, Mapping)
            }

            role_payload = species_raw.get("roles", {})
            if not isinstance(role_payload, Mapping):
                role_payload = {}
            roles: dict[str, RoleData] = {}
            for role_name, role_raw in role_payload.items():
                if not isinstance(role_raw, Mapping):
                    continue
                _record_display_names(ability_display, role_raw.get("abilities", ()))
                _record_display_names(item_display, role_raw.get("items", ()))
                _record_display_names(move_display, role_raw.get("moves", ()))
                role_stats = stats_roles_by_id.get(to_id(role_name), {})
                weights = MappingProxyType(
                    {
                        "abilities": _option_weights(role_stats.get("abilities")),
                        "items": _option_weights(role_stats.get("items")),
                        "teraTypes": _option_weights(role_stats.get("teraTypes")),
                        "moves": _option_weights(role_stats.get("moves")),
                    }
                )
                roles[str(role_name)] = RoleData(
                    name=str(role_name),
                    abilities=_options(role_raw.get("abilities", ())),
                    items=_options(role_raw.get("items", ())),
                    tera_types=_options(role_raw.get("teraTypes", ())),
                    moves=_options(role_raw.get("moves", ())),
                    weight=_weight(role_stats.get("weight", 1.0), 1.0),
                    weights=weights,
                )

            try:
                level = int(species_raw["level"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Missing or invalid level for {species_name!r}") from exc
            parsed[species_id] = _SpeciesData(
                name=str(species_name),
                level=level,
                roles=MappingProxyType(roles),
            )
        full_sets = _parse_full_sets(
            full_payload, move_display, item_display, ability_display
        )
        return cls(parsed, full_sets)

    def species_ids(self) -> tuple[str, ...]:
        """Return all indexed species ids in stable sorted order."""
        return tuple(sorted(self._species))

    def has_species(self, species: Any) -> bool:
        return to_id(species) in self._species

    def full_sets(self, species: Any) -> tuple[FullSet, ...]:
        """Return the exact randbats sets for a species, or ``()`` if none."""
        return self._full_sets.get(to_id(species), ())

    def species_name(self, species: Any) -> str:
        return self._species[to_id(species)].name

    def level(self, species: Any) -> int:
        return self._species[to_id(species)].level

    def roles(self, species: Any) -> Mapping[str, RoleData]:
        """Return role-specific legal options and stats weights for a species."""
        return self._species[to_id(species)].roles


def _read_cached_json(path: Path, url: str, refresh: bool) -> Any:
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "showdown-bot randbats data loader"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        path.write_bytes(payload)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse randbats data at {path}: {exc}") from exc


def _normalized_weights(names: Sequence[str], raw_weights: Sequence[float]) -> list[float]:
    if not names:
        return []
    total = sum(raw_weights)
    if total <= 0:
        return [1 / len(names)] * len(names)
    return [weight / total for weight in raw_weights]


def _weighted_choice(rng: Any, options: Sequence[str], weights: Sequence[float]) -> str | None:
    if not options:
        return None
    clean_weights = [_weight(weight) for weight in weights]
    if len(clean_weights) != len(options) or sum(clean_weights) <= 0:
        clean_weights = [1.0] * len(options)
    return rng.choices(list(options), weights=clean_weights, k=1)[0]


def _weighted_index(rng: Any, weights: Sequence[float]) -> int:
    """Pick an index into ``weights``, proportional to each entry's weight.

    Used to pick among heterogeneous ``(species_id, FullSet)`` pairs that
    ``_weighted_choice`` (string options only) can't carry directly.
    """
    clean_weights = [_weight(weight) for weight in weights]
    if sum(clean_weights) <= 0:
        clean_weights = [1.0] * len(weights)
    return rng.choices(range(len(weights)), weights=clean_weights, k=1)[0]


def _resolve_randbats_species(data: RandbatsData, species: Any, gen: int = 9) -> str | None:
    """Resolve a current battle forme to its selectable randbats species.

    Exact randbats keys win, so distinct selectable formes such as
    Urshifu-Rapid-Strike and Ogerpon-Wellspring stay distinct. For temporary
    formes, walk poke-env's GenData pokedex links in battleOnly, changesFrom,
    then baseSpecies order. Return None when no randbats entry can be proved.
    """
    species_id = to_id(species)
    if not species_id:
        return None
    if data.has_species(species_id):
        return species_id

    try:
        from poke_env.data.gen_data import GenData

        pokedex = GenData.from_gen(gen).pokedex
    except (ImportError, KeyError, OSError, ValueError):
        return None

    pending = [species_id]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if data.has_species(current):
            return current
        entry = pokedex.get(current)
        if not isinstance(entry, Mapping):
            continue
        linked_ids: list[str] = []
        for field_name in ("battleOnly", "changesFrom", "baseSpecies"):
            for name in _options(entry.get(field_name)):
                linked_id = to_id(name)
                if linked_id and linked_id not in visited:
                    linked_ids.append(linked_id)
        pending.extend(linked_ids)
    return None


@dataclass(frozen=True)
class SampledSet:
    species: str
    level: int
    role: str
    moves: tuple[str, ...]
    ability: str | None
    item: str | None
    tera_type: str | None


class SetTracker:
    """Role candidates for one opposing Pokémon, narrowed by observations."""

    def __init__(
        self, data: RandbatsData, species: Any, *, sibling_species: Sequence[Any] = ()
    ):
        """Track one opposing Pokemon's candidate randbats role(s).

        ``sibling_species`` names other randbats species entries this same
        Pokemon could secretly be (see ``_hidden_forme_species``), for
        example "Greninja-Bond" alongside a resolved "Greninja". When given,
        every candidate species' roles are merged and, because role names can
        collide across species (both Greninja and Greninja-Bond have a "Fast
        Attacker" role), each merged role is qualified with its source
        species' display name so ``candidate_roles()``/``sample()`` can still
        report unambiguously which entry narrowed evidence points to.
        """
        self.data = data
        primary_id = to_id(species)
        sibling_ids = [to_id(sibling) for sibling in sibling_species]
        self._species_ids = (primary_id, *dict.fromkeys(
            sid for sid in sibling_ids if sid and sid != primary_id
        ))
        self.species = data.species_name(primary_id)
        self.level = data.level(primary_id)
        ambiguous = len(self._species_ids) > 1
        self._roles: dict[str, RoleData] = {}
        self._role_species: dict[str, str] = {}
        for species_id in self._species_ids:
            for role_name, role in data.roles(species_id).items():
                key = (
                    f"{data.species_name(species_id)}: {role_name}"
                    if ambiguous
                    else role_name
                )
                self._roles[key] = role
                self._role_species[key] = species_id
        self._observations: list[tuple[str, str]] = []
        self._observation_ids: set[tuple[str, str]] = set()
        self._revealed_moves: list[str] = []
        self._revealed_item: str | None = None
        self._revealed_ability: str | None = None
        self._revealed_tera: str | None = None
        self._candidate_names = list(self._roles)
        self.inconsistent = False
        self._provisional_ids: set[tuple[str, str]] = set()
        self._provisional_move_names: list[str] = []
        self._provisional_observations: list[tuple[str, str]] = []
        self._binding_revealed_moves: list[str] = []
        # Negative / indirect evidence (task 589.28): predicates over full sets.
        self._constraints: list[Constraint] = []
        self._provisional_constraints: list[Constraint] = []
        self._signature: tuple | None = None
        self._match_cache: list[tuple[str, FullSet]] | None = None

    @property
    def constraints(self) -> tuple[Constraint, ...]:
        """Binding negative-evidence predicates applied when sampling."""
        return tuple(self._constraints)

    @property
    def revealed_moves(self) -> tuple[str, ...]:
        return tuple(self._revealed_moves)

    @property
    def provisional_moves(self) -> tuple[str, ...]:
        """Moves held out of role elimination and sampling.

        These still appear in ``revealed_moves`` as raw evidence, but never
        narrow ``candidate_roles()`` and are never force-included by
        ``sample()``. See ``provisional_observations``.
        """
        return tuple(self._provisional_move_names)

    @property
    def provisional_observations(self) -> tuple[tuple[str, str], ...]:
        """Every ``(kind, value)`` held out of role elimination and sampling.

        ``OpponentModel.update`` marks a whole switch-in provisional when it
        contains something this species can't have but a disguised
        Zoroark/Zoroark-Hisui could, because everything seen during that
        switch-in may then belong to the Zoroark. Task 589.9 decides who was
        really responsible.
        """
        return tuple(self._provisional_observations)

    def _replace_observations(
        self, observations: Sequence[tuple[str, Any, bool]]
    ) -> None:
        """Replace model-fed evidence after protocol provenance is reconciled."""
        signature = tuple(observations)
        if signature == self._signature:
            return
        self._signature = signature
        self._match_cache = None
        self._constraints = []
        self._provisional_constraints = []
        self._observations.clear()
        self._observation_ids.clear()
        self._revealed_moves.clear()
        self._provisional_ids.clear()
        self._provisional_move_names.clear()
        self._provisional_observations.clear()
        self._binding_revealed_moves.clear()
        self._revealed_item = None
        self._revealed_ability = None
        self._revealed_tera = None
        self._candidate_names = list(self._roles)
        self.inconsistent = False
        for kind, value, provisional in observations:
            if kind in {"moves", "items", "abilities", "teraTypes"}:
                self._observe(kind, value, provisional=provisional)
            elif kind == "constraint":
                target = self._provisional_constraints if provisional else self._constraints
                if value not in target:
                    target.append(value)
        # _observe() cleared the cache and signature; restore the signature.
        self._signature = signature
        self._match_cache = None

    def observe_move(self, move: Any, *, provisional: bool = False) -> None:
        self._observe("moves", move, provisional=provisional)

    def observe_item(self, item: Any, *, provisional: bool = False) -> None:
        self._observe("items", item, provisional=provisional)

    def observe_ability(self, ability: Any, *, provisional: bool = False) -> None:
        self._observe("abilities", ability, provisional=provisional)

    def observe_tera(self, tera_type: Any, *, provisional: bool = False) -> None:
        self._observe("teraTypes", tera_type, provisional=provisional)

    def _observe(self, kind: str, value: Any, *, provisional: bool = False) -> None:
        display = _display_name(value)
        option_id = to_id(display)
        if not option_id:
            return
        if kind == "items" and option_id == "unknownitem":
            return
        self._signature = None
        self._match_cache = None
        key = (kind, option_id)
        if key in self._observation_ids:
            return
        self._observation_ids.add(key)
        self._observations.append((kind, option_id))
        if provisional:
            self._provisional_ids.add(key)
            self._provisional_observations.append((kind, display))
        if kind == "moves":
            self._revealed_moves.append(display)
            if provisional:
                self._provisional_move_names.append(display)
            else:
                self._binding_revealed_moves.append(display)
        elif provisional:
            # A possibly-borrowed item, ability or tera type must never
            # become the value sample() reports for this Pokémon.
            pass
        elif kind == "items":
            self._revealed_item = display
        elif kind == "abilities":
            self._revealed_ability = display
        else:
            self._revealed_tera = display
        self._recalculate_candidates()

    def _recalculate_candidates(self) -> None:
        conflict_counts: dict[str, int] = {}
        for role_name, role in self._roles.items():
            conflicts = 0
            for kind, option_id in self._observations:
                if (kind, option_id) in self._provisional_ids:
                    # Excluded from sampling legality: it may be Zoroark's
                    # move, not this species', so it must not eliminate roles.
                    continue
                legal_ids = {to_id(option) for option in role.options(kind)}
                if option_id not in legal_ids:
                    conflicts += 1
            conflict_counts[role_name] = conflicts

        compatible = [name for name, count in conflict_counts.items() if count == 0]
        if compatible:
            self._candidate_names = compatible
            self.inconsistent = False
            return

        if conflict_counts:
            least_conflicts = min(conflict_counts.values())
            self._candidate_names = [
                name for name, count in conflict_counts.items() if count == least_conflicts
            ]
            self.inconsistent = True
            logger.warning(
                "Observations for %s conflict with all randbats roles; keeping the "
                "least-conflicting roles: %s",
                self.species,
                ", ".join(self._candidate_names),
            )

    def candidate_roles(self) -> dict[str, float]:
        """Return candidate role probabilities normalized to sum to one."""
        names = list(self._candidate_names)
        raw = [_weight(self._roles[name].weight) for name in names]
        normalized = _normalized_weights(names, raw)
        return dict(zip(names, normalized))

    def sample(self, rng: Any) -> SampledSet:
        """Sample a whole set, preferring real joint data over role marginals.

        Draws from the full randbats sets (level, item, ability, all moves
        and tera together) of every candidate species id -- the primary
        species plus any hidden-forme siblings (see ``_hidden_forme_species``)
        -- restricted to sets consistent with every BINDING observation and
        weighted by real occurrence count. Provisional observations (see
        ``provisional_observations``) never constrain this: they may belong
        to a disguised Zoroark, not this species. If no full set is
        consistent with the evidence (full data missing for this species, or
        the tracker's observations are ``inconsistent`` with the real data),
        falls back to the role-based marginal sampler below.
        """
        matches = self._matching_full_sets()
        if not matches:
            logger.debug(
                "No full randbats set for %s matches binding evidence; "
                "falling back to role-based sampling",
                self.species,
            )
            return self._sample_role_based(rng)
        weights = [full_set.count for _, full_set in matches]
        index = _weighted_index(rng, weights)
        species_id, full_set = matches[index]
        return SampledSet(
            species=self.data.species_name(species_id),
            level=full_set.level,
            role=self._best_matching_role_name(species_id, full_set),
            moves=full_set.moves,
            ability=full_set.ability,
            item=full_set.item,
            tera_type=full_set.tera_type,
        )

    def _matching_full_sets(self) -> list[tuple[str, FullSet]]:
        """Full sets, across every candidate species id, legal given evidence.

        Binding revealed evidence is mandatory. Negative-evidence constraints
        (task 589.28) are applied one at a time and a constraint that would
        leave nothing is skipped, so a wrongly-inferred exclusion can degrade
        sampling to "no inference" but never to the role-based fallback.
        """
        if self._match_cache is not None:
            return self._match_cache
        matches = [
            (species_id, full_set)
            for species_id in self._species_ids
            for full_set in self.data.full_sets(species_id)
            if self._full_set_consistent(full_set)
        ]
        for constraint in self._constraints:
            if not matches:
                break
            kept = [
                (species_id, full_set)
                for species_id, full_set in matches
                if constraint_allows(constraint, species_id, full_set)
            ]
            if kept:
                matches = kept
        self._match_cache = matches
        return matches

    def _full_set_consistent(self, full_set: FullSet) -> bool:
        move_ids = {to_id(move) for move in full_set.moves}
        for move in self._binding_revealed_moves:
            if to_id(move) not in move_ids:
                return False
        if self._revealed_item is not None:
            have_item_id = to_id(full_set.item) if full_set.item else ""
            if to_id(self._revealed_item) != have_item_id:
                return False
        if self._revealed_ability is not None:
            if to_id(self._revealed_ability) != to_id(full_set.ability):
                return False
        if self._revealed_tera is not None:
            if to_id(self._revealed_tera) != to_id(full_set.tera_type):
                return False
        return True

    def _best_matching_role_name(self, species_id: str, full_set: FullSet) -> str:
        """Best-effort label for which marginal role a full set resembles.

        A full set is reported under a real role's name when every one of
        its moves, its ability, its tera type, and (if it carries one) its
        item are all legal options of that role -- i.e. the role's marginal
        data could have generated it. The first such role wins; when none
        of this species' roles could have generated it (full-set data can
        be finer-grained than the role option lists it's rolled up from),
        the set is reported as ``"full-set"`` rather than a misleading role
        name.
        """
        move_ids = {to_id(move) for move in full_set.moves}
        ability_id = to_id(full_set.ability)
        item_id = to_id(full_set.item) if full_set.item else ""
        tera_id = to_id(full_set.tera_type)
        for role_name, role in self._roles.items():
            if self._role_species[role_name] != species_id:
                continue
            if not move_ids <= {to_id(move) for move in role.moves}:
                continue
            if ability_id not in {to_id(ability) for ability in role.abilities}:
                continue
            if item_id and item_id not in {to_id(item) for item in role.items}:
                continue
            if tera_id not in {to_id(tera) for tera in role.tera_types}:
                continue
            return role_name
        return "full-set"

    def _sample_role_based(self, rng: Any) -> SampledSet:
        """Fall back to drawing a role, then each option, from marginals.

        Kept for when no full set is consistent with this tracker's binding
        evidence: full data missing for this species (download failure or a
        species the real file doesn't cover), or the tracker's own
        observations are ``inconsistent`` with the real full data. Each
        option is drawn independently from its per-role marginal frequency,
        so -- unlike ``sample()`` -- the result can combine options that
        never actually co-occur in a single real set.
        """
        candidates = self.candidate_roles()
        if not candidates:
            raise ValueError(f"No randbats roles available for {self.species!r}")
        role_names = list(candidates)
        role_name = _weighted_choice(rng, role_names, list(candidates.values()))
        assert role_name is not None
        role = self._roles[role_name]
        species_id = self._role_species[role_name]
        moves = self._sample_moves(role, rng)
        return SampledSet(
            species=self.data.species_name(species_id),
            level=self.data.level(species_id),
            role=role.name,
            moves=moves,
            ability=self._revealed_ability
            or _weighted_choice(
                rng,
                role.abilities,
                [role.option_weight("abilities", option) for option in role.abilities],
            ),
            item=self._revealed_item
            or _weighted_choice(
                rng,
                role.items,
                [role.option_weight("items", option) for option in role.items],
            ),
            tera_type=self._revealed_tera
            or _weighted_choice(
                rng,
                role.tera_types,
                [role.option_weight("teraTypes", option) for option in role.tera_types],
            ),
        )

    def _sample_moves(self, role: RoleData, rng: Any) -> tuple[str, ...]:
        moves: list[str] = []
        seen: set[str] = set()

        def add(move: str) -> None:
            move_id = to_id(move)
            if move_id and move_id not in seen:
                moves.append(move)
                seen.add(move_id)

        # Reveals are evidence and stay in the sample even if later evidence
        # makes every role inconsistent. Provisional moves (see
        # ``provisional_moves``) are excluded: they may belong to a disguised
        # Zoroark instead of this species, so they must not force their way
        # into a sampled set this species' randbats data never licensed.
        for move in self._binding_revealed_moves:
            add(move)

        target_count = min(4, len(role.moves))
        move_weights = role.weights["moves"]
        mandatory = [
            move
            for move in role.moves
            if move_weights.get(to_id(move), 0.0) >= 0.999
        ]
        for move in mandatory:
            add(move)

        remaining = [move for move in role.moves if to_id(move) not in seen]
        while len(moves) < target_count and remaining:
            weights: list[float] = []
            for move in remaining:
                probability = move_weights.get(to_id(move), 0.0)
                probability = min(1 - 1e-6, max(1e-6, probability))
                weights.append(probability / (1 - probability))
            selected = _weighted_choice(rng, remaining, weights)
            if selected is None:
                break
            add(selected)
            remaining = [move for move in remaining if to_id(move) not in seen]
        return tuple(moves)


_ORIGIN_REVEAL_EVENTS = frozenset(
    {"-damage", "-heal", "-status", "-boost", "-unboost", "-curestatus", "-sethp",
     "-immune", "-activate", "-block"}
)

_PROTOCOL_IDENTIFIER = re.compile(r"^(p[12])([a-z])?:\s*(.+)$", re.IGNORECASE)
_TRANSFER_MOVES = {"trick", "switcheroo", "bestow", "thief", "covet"}
_TRANSFER_ABILITIES = {"magician", "pickpocket", "symbiosis"}
# Moves whose |move| event names a move the Pokemon did not choose from its
# own set: every data/moves.ts ``callsMove: true`` entry (Assist, Copycat, Me
# First, Metronome, Mirror Move, Nature Power) EXCEPT Sleep Talk, whose onHit
# only ever samples ``pokemon.moveSlots`` -- the user's own moveset -- so a
# move it calls IS legitimate original-set evidence and must stay out of this
# set. Dancer (ability), Magic Bounce (ability) and Magic Coat/Snatch (move
# conditions) are the same class of bug from the other direction: they run a
# *different* Pokemon's move through battle-actions.ts#useMove() without an
# explicit sourceEffect, so the reflecting/copying Pokemon's own |move| event
# for it is not their own set evidence either.
_FOREIGN_MOVE_EFFECTS = frozenset(
    {
        "assist",
        "copycat",
        "mefirst",
        "metronome",
        "mirrormove",
        "naturepower",
        "dancer",
        "magicbounce",
        "magiccoat",
        "snatch",
    }
)


@dataclass
class _ProtocolSession:
    observations: list[tuple[str, str]] = field(default_factory=list)
    received_item: bool = False
    received_abilities: set[str] = field(default_factory=set)
    transformed: bool = False
    mimicked: bool = False
    # Illusion tells (task 589.9): damaging moves aimed at this Pokemon that
    # showed ``-immune`` as (move id, attacker identity), moves that dealt
    # direct damage, whether it Terastallized, and a duplicate-species flag.
    immune_events: list[tuple[str, tuple[str, str]]] = field(default_factory=list)
    hit_moves: list[str] = field(default_factory=list)
    terastallized: bool = False
    duplicate: str | None = None  # "hp" | "fainted"
    # Negative / indirect evidence (task 589.28): predicates over full sets and
    # raw speed-order observations. Held per session so they follow Illusion's
    # ``replace`` reattribution and its provisional handling like moves do.
    constraints: list[Constraint] = field(default_factory=list)
    speed_obs: list[SpeedObs] = field(default_factory=list)


@dataclass
class _ProtocolPokemon:
    sessions: list[_ProtocolSession] = field(default_factory=list)
    current: _ProtocolSession | None = None
    hp: float | None = None  # last HP fraction seen under this name
    session_start_hp: float | None = None  # ``hp`` when the current session began
    fainted: bool = False

    def begin_session(self) -> _ProtocolSession:
        # A received item remains non-original after switching out and back in.
        received_item = any(session.received_item for session in self.sessions)
        self.current = _ProtocolSession(received_item=received_item)
        self.sessions.append(self.current)
        return self.current


@dataclass
class _PendingItemTransfer:
    actor: tuple[str, str]
    target: tuple[str, str]
    touched: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class _BattleOpponentState:
    trackers: dict[str, SetTracker] = field(default_factory=dict)
    pokemon: dict[tuple[str, str], _ProtocolPokemon] = field(default_factory=dict)
    active_by_slot: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    pending_item_transfer: _PendingItemTransfer | None = None
    # (target slot, move id, attacker identity) of the last move aimed at the
    # other side, until its outcome (-immune/-damage/...) is seen.
    pending_hit: tuple[tuple[str, str], str, tuple[str, str]] | None = None
    # Dancers whose ``-activate|...|ability: Dancer`` has been seen but whose
    # copied ``|move|`` event has not: that move is the dancer's copy of
    # someone else's move, even when the server leaves off its ``[from]`` tag.
    dancer_pending: set[tuple[str, str]] = field(default_factory=set)
    inference: InferenceTracker = field(default_factory=InferenceTracker)
    # (DamageObs, attacker session, defender session) read from the current
    # message and not yet evaluated against the pre-message battle state.
    pending_damage: list = field(default_factory=list)


def _protocol_name(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.lower().startswith("[of] "):
        value = value[5:].strip()
    match = _PROTOCOL_IDENTIFIER.match(value)
    if match is None:
        return None
    return match.group(1).lower(), match.group(3).strip()


def _protocol_identity(value: Any) -> tuple[str, str] | None:
    parsed = _protocol_name(value)
    if parsed is None:
        return None
    side, name = parsed
    name_id = to_id(name)
    return (side, name_id) if name_id else None


def _protocol_slot(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = _PROTOCOL_IDENTIFIER.match(value.strip())
    if match is None or not match.group(2):
        return None
    return match.group(1).lower(), match.group(2).lower()


def _details_tera(details: Any) -> str | None:
    """Tera type from switch details such as ``Gardevoir, L83, F, tera:Poison``.

    sim/pokemon.ts#getFullDetails appends ``tera:<type>`` once a Pokémon has
    terastallized, using the Illusion details while disguised. Anything that
    isn't a real tera type is ignored.
    """
    for part in str(details).split(","):
        part = part.strip()
        if part.lower().startswith("tera:"):
            tera = part[len("tera:") :].strip()
            return tera if to_id(tera) in _TERA_TYPE_IDS else None
    return None


def _tagged_value(event: Sequence[str], prefix: str) -> str | None:
    prefix_id = prefix.lower()
    for part in event:
        text = str(part).strip()
        if text.lower().startswith(prefix_id):
            return text[len(prefix) :].strip()
    return None


def _effect_id(value: Any) -> str:
    text = str(value).strip()
    for prefix in ("move:", "ability:", "item:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    return to_id(text)


def _transfer_origin(event: Sequence[str]) -> tuple[str, str] | None:
    """Return a recognized successful-transfer cause, if one is tagged."""
    origin = _tagged_value(event[4:], "[from]") if len(event) > 4 else None
    if origin is None:
        return None
    text = origin.strip()
    if text.lower().startswith("move:"):
        effect = _effect_id(text)
        return ("move", effect) if effect in _TRANSFER_MOVES else None
    if text.lower().startswith("ability:"):
        effect = _effect_id(text)
        return ("ability", effect) if effect in _TRANSFER_ABILITIES else None
    return None


def _attributed_identity(
    event: Sequence[str], value: Any, *, label: str
) -> tuple[str, str] | None:
    identity = _protocol_identity(value)
    if identity is None:
            logger.debug(
                "Skipping %s event with unattributable Pokémon %r: %r",
                label,
                value,
                event,
            )
    return identity


def _record_protocol_observation(
    pokemon: _ProtocolPokemon, kind: str, value: Any
) -> None:
    display = _display_name(value)
    if kind == "abilities":
        # poke-env stores abilities by Showdown id (for example ``trace``).
        display = to_id(display)
    option_id = to_id(display)
    if not option_id or (kind == "items" and option_id == "unknownitem"):
        return
    session = pokemon.current or pokemon.begin_session()
    observation = (kind, display)
    if observation not in session.observations:
        session.observations.append(observation)


def _parse_hp_fraction(value: Any) -> float | None:
    """HP fraction from a protocol condition such as ``45/100 par`` or ``0 fnt``."""
    text = str(value).strip().split(" ")[0]
    if not text or text.startswith("["):
        return None
    if "/" in text:
        current, _, maximum = text.partition("/")
        try:
            return max(0.0, min(1.0, float(current) / float(maximum)))
        except (ValueError, ZeroDivisionError):
            return None
    return 0.0 if text == "0" else None


# Moves that stop a move from damaging its target (so no direct hit follows).
_BLOCKING_EFFECTS = frozenset(
    {
        "substitute", "protect", "detect", "banefulbunker", "kingsshield",
        "spikyshield", "silktrap", "burningbulwark", "obstruct", "maxguard",
    }
)
_HP_SLACK = 0.0049  # opponent HP arrives as whole percent


def _current_session(
    state: _BattleOpponentState, identity: tuple[str, str]
) -> _ProtocolSession:
    pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
    return pokemon.current or pokemon.begin_session()


def _track_illusion(
    state: _BattleOpponentState, event_type: str, event: Sequence[str]
) -> None:
    """Record the protocol facts the Illusion tells need (task 589.9).

    Tracks, per Pokémon name: HP fraction and fainting (for the
    Species-Clause duplicate tell), type-immunity outcomes and direct hits
    of the last move aimed at it, and Terastallization. It never changes the
    set-evidence bookkeeping.
    """
    if event_type == "move":
        state.pending_hit = None
        actor = _protocol_slot(event[2]) if len(event) > 2 else None
        target = _protocol_slot(event[4]) if len(event) > 4 else None
        attacker = _protocol_identity(event[2]) if len(event) > 2 else None
        if (
            actor is not None
            and target is not None
            and attacker is not None
            and actor[0] != target[0]
            and not any(
                str(part).lower().startswith(("[miss]", "[notarget]"))
                for part in event[5:]
            )
        ):
            state.pending_hit = (target, to_id(event[3]), attacker)
        return

    if event_type in {"turn", "switch", "drag", "-miss", "-fail", "-notarget", "cant"}:
        state.pending_hit = None

    if event_type == "-activate" and len(event) > 3:
        if _effect_id(event[3]) in _BLOCKING_EFFECTS:
            state.pending_hit = None
        return

    if event_type == "faint" and len(event) > 2:
        identity = _protocol_identity(event[2])
        if identity is not None:
            pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
            pokemon.fainted = True
            pokemon.hp = 0.0
        return

    if event_type == "-terastallize" and len(event) > 2:
        identity = _protocol_identity(event[2])
        if identity is not None:
            _current_session(state, identity).terastallized = True
        return

    if event_type == "-immune" and len(event) > 2:
        pending = state.pending_hit
        slot = _protocol_slot(event[2])
        has_cause = any(str(part).strip().startswith("[from]") for part in event[3:])
        if pending is not None and slot == pending[0] and not has_cause:
            identity = state.active_by_slot.get(slot) or _protocol_identity(event[2])
            if identity is not None:
                session = _current_session(state, identity)
                if not session.terastallized:
                    session.immune_events.append((pending[1], pending[2]))
        if pending is not None and slot == pending[0]:
            state.pending_hit = None
        return

    if event_type in {"-damage", "-heal", "-sethp"} and len(event) > 3:
        identity = _protocol_identity(event[2])
        if identity is None:
            return
        pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
        tags = " ".join(str(part).lower() for part in event[4:])
        if event_type == "-heal" and "revivalblessing" in to_id(tags):
            pokemon.fainted = False
        hp = _parse_hp_fraction(event[3])
        if hp is not None:
            pokemon.hp = hp
        if event_type == "-damage":
            pending = state.pending_hit
            slot = _protocol_slot(event[2])
            if (
                pending is not None
                and slot == pending[0]
                and "[from]" not in tags
            ):
                _current_session(state, identity).hit_moves.append(pending[1])
                state.pending_hit = None


def _record_ability_if_original(
    pokemon: _ProtocolPokemon, session: _ProtocolSession, ability_name: Any
) -> None:
    """Record ability evidence unless a Transform replaced this stay's kit.

    After ``-transform`` (Imposter or the move Transform), every ability the
    Pokemon shows until it switches out belongs to the copied target, not to
    this Pokemon's own randbats set (sim/pokemon.ts#transformInto copies
    ``moveSlots``/ability wholesale). Imposter itself is the one exception,
    and it is recorded directly by the ``-transform`` handler before this
    guard ever runs.
    """
    if session.transformed:
        return
    _record_protocol_observation(pokemon, "abilities", ability_name)


# Ability ids Showdown attributes to a Pokemon only as a side effect of a
# forme change it goes through mid-battle (Terastallizing, or the "As One"
# ability printing its two legacy component names), never as part of the
# original randbats set for that species. Mapping to another id folds the
# observation into the real randbats ability (As One's literal protocol name
# is "As One", with no parenthetical, so ``to_id`` gives "asone" -- not either
# of randbats' "asoneglastrier"/"asonespectrier" ids; Unnerve/Chilling
# Neigh/Grim Neigh are the same ability activating under its gen 8 component
# names, see data/abilities.ts asoneglastrier/asonespectrier onStart and the
# chillingneigh/grimneigh onSourceAfterFaint boost(..., effect) calls, which
# battle.ts#boost() prints as a plain ``-ability|mon|Name|boost`` event).
_FORME_CHANGE_ABILITY_ALIASES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "calyrexice": MappingProxyType(
            {"asone": "asoneglastrier", "unnerve": "asoneglastrier", "chillingneigh": "asoneglastrier"}
        ),
        "calyrexshadow": MappingProxyType(
            {"asone": "asonespectrier", "unnerve": "asonespectrier", "grimneigh": "asonespectrier"}
        ),
    }
)

# Ability ids that only ever appear as a forme-change side effect, for every
# species that can show them, and are never a randbats role's original
# ability: Embody Aspect activates when Ogerpon Terastallizes into its
# battle-only "-Tera" forme (data/abilities.ts embodyaspect*, data/
# pokedex.ts ogerpon*tera baseSpecies entries); Tera Shell/Teraform Zero are
# Terapagos's post-Tera-Shift/Tera-Shift-twice formes (data/pokedex.ts
# terapagosterastal/terapagosstellar), while the base species' own randbats
# ability is "Tera Shift" (data/abilities.ts terashift) and stays unaffected.
_DISCARD_FORME_CHANGE_ABILITIES = frozenset(
    {
        "embodyaspectteal",
        "embodyaspectwellspring",
        "embodyaspecthearthflame",
        "embodyaspectcornerstone",
        "terashell",
        "teraformzero",
    }
)


def _resolve_forme_change_ability(species_ids: Sequence[str], ability_id: str) -> str | None:
    """Fold or drop ability evidence that the protocol ties to a forme change.

    Returns the ability id to record (possibly remapped), or ``None`` when
    the observation must be dropped entirely because it can never be this
    species' original randbats ability.
    """
    for species_id in species_ids:
        aliases = _FORME_CHANGE_ABILITY_ALIASES.get(species_id)
        if aliases and ability_id in aliases:
            return aliases[ability_id]
    if ability_id in _DISCARD_FORME_CHANGE_ABILITIES:
        return None
    return ability_id


# sim/pokemon.ts#getUpdatedDetails prints exactly these formes under their
# base species name, so a switch-in showing the base name may be either one.
# Every other forme (regional variants, Urshifu-Rapid-Strike, gender formes,
# Squawkabilly colours) shows its own name in the switch details.
_HIDDEN_DISPLAY_FORMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "greninja": ("greninjabond",),
        "rockruff": ("rockruffdusk",),
    }
)


def _hidden_forme_species(data: RandbatsData, species_id: str) -> tuple[str, ...]:
    """Return randbats entries that switch in looking exactly like ``species_id``."""
    return tuple(
        forme_id
        for forme_id in _HIDDEN_DISPLAY_FORMES.get(to_id(species_id), ())
        if data.has_species(forme_id)
    )


# Species ids the Illusion ability can disguise as something else. Species
# Clause allows at most one of them per team.
_ZOROARK_SPECIES_IDS = ("zoroark", "zoroarkhisui")


def _zoroark_option_ids(data: RandbatsData, kind: str) -> set[str]:
    """Option ids (moves, items, tera types...) any Zoroark role can have."""
    return {
        to_id(option)
        for species_id in _ZOROARK_SPECIES_IDS
        if data.has_species(species_id)
        for role in data.roles(species_id).values()
        for option in role.options(kind)
    }


def _zoroark_could_be_on_team(team: Mapping[Any, Any]) -> bool:
    """Whether an Illusion user could be among this opponent's Pokémon.

    True when a Zoroark/Zoroark-Hisui has been seen (fainted or not: earlier
    disguised appearances may still have been it) or when fewer than six
    team members are known, so an unseen slot could hold one.
    """
    species_ids = {to_id(getattr(pokemon, "species", "")) for pokemon in team.values()}
    if species_ids & set(_ZOROARK_SPECIES_IDS):
        return True
    return len(team) < 6


@dataclass(frozen=True)
class IllusionEstimate:
    """``OpponentModel.illusion_estimate`` result.

    ``probabilities`` maps a Zoroark form id to P(the active opponent is that
    form, disguised). ``trackers`` holds, per form with non-zero probability,
    a ``SetTracker`` already fed this switch-in's Zoroark-legal observations,
    for sampling the form's set.
    """

    probabilities: Mapping[str, float]
    trackers: Mapping[str, SetTracker]

    @property
    def total(self) -> float:
        return sum(self.probabilities.values())


class OpponentModel:
    """Persistent opponent-set trackers keyed by battle tag and team id.

    Keep one instance for a bot's lifetime and call ``update`` after every
    battle-state update. Unknown or unresolved species are logged and skipped.
    Feed it raw split protocol events with ``observe`` and pass the current
    poke-env Battle to ``update`` before decisions are made. Moves, abilities,
    and items come exclusively from ordered events. If a transfer omits its
    origin and no matching transfer event identifies it, an untagged item
    reveal is indistinguishable from an original item and is treated as
    original. Tagged transfers without an identifiable donor taint the
    recipient and skip donor evidence. Copied abilities with no restoration
    event remain treated as received until switch-out. Ability evidence that
    the protocol only ever attributes to a forme change (Embody Aspect, Tera
    Shell/Teraform Zero, As One's Unnerve/Chilling Neigh/Grim Neigh
    component names) is folded into the real randbats ability or dropped,
    never recorded as its own ability -- see ``_resolve_forme_change_ability``.
    A forme Showdown prints under its base name (Greninja-Bond) is tracked
    against both entries' roles until evidence narrows it -- see
    ``_hidden_forme_species``.

    Illusion: when a switch-in shows anything (move, item, ability or tera
    type) that the shown species can't have but a Zoroark/Zoroark-Hisui
    could, and a Zoroark could be on the team, every observation from that
    switch-in is held on ``SetTracker.provisional_observations`` and never
    narrows roles or feeds ``sample()``. Limitation: a disguised Zoroark
    that only shows things the disguise could also have is indistinguishable
    from the real Pokémon here; ``illusion_probability`` / ``illusion_estimate``
    (task 589.9, see ``showdown_bot.illusion``) add immunity, duplicate-species
    and other tells and give P(active opponent is a disguised Zoroark).
    """

    def __init__(self, data: RandbatsData, *, inference: bool = True, damage: bool = True):
        self.data = data
        self._battles: dict[str, _BattleOpponentState] = {}
        self.damage_enabled = damage
        self._damage_inferer: Any = None
        # False disables the 589.28 inference (item/ability origin reveals,
        # negative evidence, speed): the pre-589.28 behaviour, for A/B duels.
        self.inference_enabled = inference

    def observe(self, battle_tag: str, event: list[str]) -> None:
        """Consume one raw ``parse_message`` event (see ``_observe_core``).

        After the ordered-provenance bookkeeping, item/ability reveals tagged
        ``[from] item:`` / ``[from] ability:`` are recorded and the negative
        evidence tracker (``showdown_bot.inference``) is fed; both attach to
        the same protocol sessions as every other observation.
        """
        self._observe_core(battle_tag, event)
        if (
            not self.inference_enabled
            or not isinstance(event, Sequence)
            or isinstance(event, str)
            or len(event) < 2
        ):
            return
        started = time.perf_counter()
        try:
            state = self._battles.setdefault(str(battle_tag), _BattleOpponentState())
            self._record_origin_reveal(state, event)
            for identity, evidence in state.inference.feed(event):
                if isinstance(evidence, DamageObs):
                    if self.damage_enabled and len(state.pending_damage) < 8:
                        state.pending_damage.append(
                            (
                                evidence,
                                _current_session(state, evidence.attacker),
                                _current_session(state, evidence.defender),
                            )
                        )
                    continue
                session = _current_session(state, identity)
                if isinstance(evidence, SpeedObs):
                    if evidence not in session.speed_obs:
                        session.speed_obs.append(evidence)
                elif evidence not in session.constraints:
                    session.constraints.append(evidence)
        except Exception as exc:  # inference must never break the battle loop
            logger.debug("Inference skipped event for %s (%s): %r", battle_tag, exc, event)
        finally:
            INFERENCE_SECONDS[0] += time.perf_counter() - started

    def observe_events(
        self, battle_tag: str, events: Sequence[Sequence[str]], battle: Any | None = None
    ) -> None:
        """Observe one message's events, then evaluate damage rolls against ``battle``.

        ``battle`` must still be in its state from BEFORE these events were
        applied (poke-env has not parsed them yet): damage-roll inference
        calculates each clean hit in that state. Without a battle the hits are
        simply dropped.
        """
        for event in events:
            if len(event) >= 2 and event[0] == "":
                self.observe(battle_tag, list(event))
        self.resolve_damage(battle_tag, battle)

    def resolve_damage(self, battle_tag: str, battle: Any | None) -> None:
        state = self._battles.get(str(battle_tag))
        if state is None or not state.pending_damage:
            return
        pending, state.pending_damage = state.pending_damage, []
        if battle is None or not (self.inference_enabled and self.damage_enabled):
            return
        try:
            from showdown_bot.damage_infer import DamageInferer

            if self._damage_inferer is None:
                self._damage_inferer = DamageInferer(self.data)
            trackers = self.update(battle)
            try:
                illusion_p = sum(self.illusion_probability(battle).values())
            except Exception:
                illusion_p = 0.0
            results = self._damage_inferer.evaluate_all(
                battle, trackers, [record[0] for record in pending], illusion_p
            )
            role = getattr(battle, "player_role", None)
            for obs, _key, constraint in results:
                for record in pending:
                    if record[0] is obs:
                        session = record[2] if obs.attacker[0] == role else record[1]
                        if constraint not in session.constraints:
                            session.constraints.append(constraint)
            if results:
                self.update(battle)
        except Exception as exc:  # inference must never break the battle loop
            logger.debug("Damage inference failed in %s: %s", battle_tag, exc)

    @property
    def damage_stats(self) -> Any:
        return None if self._damage_inferer is None else self._damage_inferer.stats

    @staticmethod
    def _record_origin_reveal(state: _BattleOpponentState, event: Sequence[str]) -> None:
        """Record items/abilities named by a ``[from] item:|ability:`` tag.

        The tagged effect's owner is the event's Pokemon, except on ``-damage``
        where ``[of]`` names the holder of Rocky Helmet, Rough Skin, Jaboca
        Berry and similar punishers.
        """
        event_type = str(event[1]).lower()
        if len(event) < 4 or event_type not in _ORIGIN_REVEAL_EVENTS:
            return
        owner = event[2]
        origin = _tagged_value(event[3:], "[from]")
        if origin is None and event_type in {"-activate", "-block"}:
            effect = str(event[3]).strip()
            if effect.lower().startswith("item:"):
                origin = effect
        if origin is None:
            return
        kind, _, name = origin.partition(":")
        kind, name = kind.strip().lower(), name.strip()
        if kind not in {"item", "ability"} or not name:
            return
        if event_type == "-damage":
            holder = _tagged_value(event[3:], "[of]")
            if holder:
                owner = holder
        elif kind == "ability" and event_type not in {"-heal", "-immune"}:
            return
        identity = _protocol_identity(owner)
        if identity is None:
            return
        pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
        session = pokemon.current or pokemon.begin_session()
        if kind == "item":
            if not session.received_item:
                _record_protocol_observation(pokemon, "items", name)
        elif to_id(name) not in session.received_abilities:
            _record_ability_if_original(pokemon, session, name)

    def _observe_core(self, battle_tag: str, event: list[str]) -> None:
        """Consume one raw ``parse_message`` event and keep ordered provenance.

        Illusion observations are provisional within a switch-in session and
        move to the revealed Pokémon on ``replace``. If a transfer omits its
        origin and no matching transfer event identifies it, an untagged item
        reveal is indistinguishable from an original item; it follows the
        available protocol tags.
        """
        if (
            not isinstance(event, Sequence)
            or isinstance(event, str)
            or len(event) < 2
        ):
            logger.debug("Skipping malformed battle event for %s: %r", battle_tag, event)
            return
        try:
            event_type = str(event[1]).lower()
            state = self._battles.setdefault(str(battle_tag), _BattleOpponentState())
            _track_illusion(state, event_type, event)
            if event_type in {"turn", "cant", "switch", "drag", "replace", "faint"}:
                # A Dancer that never reached its |move| event (e.g. it was
                # stopped by paralysis) must not tag a later, real move.
                state.dancer_pending.clear()

            if event_type in {"switch", "drag"}:
                if len(event) < 3:
                    logger.debug("Skipping malformed %s event: %r", event_type, event)
                    return
                identity = _attributed_identity(event, event[2], label=event_type)
                slot = _protocol_slot(event[2])
                if identity is None or slot is None:
                    return
                previous = state.active_by_slot.get(slot)
                if previous is not None:
                    previous_pokemon = state.pokemon.get(previous)
                    if previous_pokemon is not None:
                        previous_pokemon.current = None
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                hp = _parse_hp_fraction(event[4]) if len(event) > 4 else None
                # Species Clause: a name that comes back alive after fainting,
                # or with less HP than it left with (nothing damages the
                # bench), is two different Pokémon, one of them a Zoroark.
                duplicate = None
                if pokemon.sessions and hp is not None and hp > 0:
                    if pokemon.fainted:
                        duplicate = "fainted"
                    elif pokemon.hp is not None and hp < pokemon.hp - _HP_SLACK:
                        duplicate = "hp"
                session = pokemon.begin_session()
                session.duplicate = duplicate
                pokemon.session_start_hp = pokemon.hp
                if hp is not None:
                    pokemon.hp = hp
                    pokemon.fainted = hp <= 0
                tera = _details_tera(event[3]) if len(event) > 3 else None
                if tera:
                    session.terastallized = True
                    _record_protocol_observation(pokemon, "teraTypes", tera)
                state.active_by_slot[slot] = identity
                state.pending_item_transfer = None
                return

            if event_type == "replace":
                if len(event) < 4:
                    logger.debug("Skipping malformed replace event: %r", event)
                    return
                new_identity = _attributed_identity(event, event[2], label="replace")
                slot = _protocol_slot(event[2])
                old_identity = state.active_by_slot.get(slot) if slot else None
                if new_identity is None or slot is None:
                    return
                if old_identity is None:
                    logger.debug("Cannot attribute Illusion replace event: %r", event)
                    state.active_by_slot[slot] = new_identity
                    return
                if old_identity != new_identity:
                    old_pokemon = state.pokemon.get(old_identity)
                    new_pokemon = state.pokemon.setdefault(new_identity, _ProtocolPokemon())
                    # The HP seen under the disguise name was the Zoroark's.
                    new_pokemon.hp = _parse_hp_fraction(event[4]) if len(event) > 4 else None
                    if old_pokemon is not None:
                        old_pokemon.hp = old_pokemon.session_start_hp
                    if old_pokemon is not None and old_pokemon.current is not None:
                        session = old_pokemon.current
                        old_pokemon.sessions.remove(session)
                        old_pokemon.current = None
                        new_pokemon.sessions.append(session)
                        new_pokemon.current = session
                    elif new_pokemon.current is None:
                        new_pokemon.begin_session()
                    tera = _details_tera(event[3])
                    if tera:
                        _record_protocol_observation(new_pokemon, "teraTypes", tera)
                state.active_by_slot[slot] = new_identity
                state.pending_item_transfer = None
                return

            if event_type == "move":
                if len(event) < 4:
                    logger.debug("Skipping malformed move event: %r", event)
                    return
                state.pending_item_transfer = None
                identity = _attributed_identity(event, event[2], label="move")
                if identity is None:
                    return
                # battle.ts runMove() only tags a Dancer copy "[from] ability:
                # Dancer" when the move actually runs; one that fails its
                # pre-checks (Choice lock, Disable, Taunt, ...) is logged as a
                # bare "|move|X|Move||[still]" right after the Dancer
                # "-activate", so that activation is the only tell.
                dancer_copy = identity in state.dancer_pending
                state.dancer_pending.discard(identity)
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                session = pokemon.current or pokemon.begin_session()
                move = str(event[3])
                invoked_by = {
                    _effect_id(str(part)[len("[from]") :].strip())
                    for part in event[4:]
                    if str(part).strip().lower().startswith("[from]")
                }
                if (
                    # Struggle (data/moves.ts) is forced on ANY Pokemon with
                    # no usable move (no PP, fully disabled/encored, etc.);
                    # it is never part of a randbats set for any species.
                    to_id(move) != "struggle"
                    and not (invoked_by & _FOREIGN_MOVE_EFFECTS)
                    and not dancer_copy
                    and not session.transformed
                    and not session.mimicked
                ):
                    _record_protocol_observation(pokemon, "moves", move)
                return

            if event_type == "-transform":
                if len(event) < 3:
                    logger.debug("Skipping malformed transform event: %r", event)
                    return
                identity = _attributed_identity(event, event[2], label="transform")
                if identity is None:
                    return
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                session = pokemon.current or pokemon.begin_session()
                session.transformed = True
                origin = _tagged_value(event[3:], "[from]") if len(event) > 3 else None
                if origin and _effect_id(origin) == "imposter":
                    if "imposter" not in session.received_abilities:
                        _record_protocol_observation(pokemon, "abilities", "Imposter")
                return

            if event_type == "-start" and len(event) > 3 and to_id(event[3]) == "mimic":
                identity = _attributed_identity(event, event[2], label="mimic")
                if identity is not None:
                    pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                    (pokemon.current or pokemon.begin_session()).mimicked = True
                return

            if event_type == "-ability":
                if len(event) < 4:
                    logger.debug("Skipping malformed ability event: %r", event)
                    return
                identity = _attributed_identity(event, event[2], label="ability")
                if identity is None:
                    return
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                session = pokemon.current or pokemon.begin_session()
                origin = _tagged_value(event[4:], "[from]") if len(event) > 4 else None
                if origin:
                    # Showdown's setAbility event is:
                    # -ability|pokemon|new ability|old ability|[from] EFFECT.
                    # The new value is borrowed; the old value is original only
                    # if the Pokémon had not already received an ability this stay.
                    old_ability = (
                        event[4]
                        if len(event) > 4 and not str(event[4]).strip().startswith("[")
                        else None
                    )
                    if old_ability and not session.received_abilities:
                        _record_ability_if_original(pokemon, session, old_ability)
                    received_ability = to_id(event[3])
                    if received_ability:
                        session.received_abilities.add(received_ability)
                    return
                if to_id(event[3]) not in session.received_abilities:
                    _record_ability_if_original(pokemon, session, event[3])
                return

            if event_type == "-activate":
                if len(event) < 4:
                    return
                effect = str(event[3]).strip()
                effect_id = _effect_id(effect)
                effect_kind = (
                    effect.split(":", 1)[0].strip().lower() if ":" in effect else ""
                )

                if effect_id == "skillswap":
                    actor_id = _attributed_identity(event, event[2], label="Skill Swap")
                    target_id = None
                    for part in reversed(event[4:]):
                        target_id = _protocol_identity(part)
                        if target_id is not None:
                            break
                    if actor_id is None or target_id is None or len(event) < 6:
                        logger.debug(
                            "Cannot attribute Skill Swap ability evidence: %r", event
                        )
                        return
                    actor = state.pokemon.setdefault(actor_id, _ProtocolPokemon())
                    target = state.pokemon.setdefault(target_id, _ProtocolPokemon())
                    actor_session = actor.current or actor.begin_session()
                    target_session = target.current or target.begin_session()
                    # Showdown logs target's old ability first, then the user's.
                    if not target_session.received_abilities:
                        _record_ability_if_original(target, target_session, event[4])
                    if not actor_session.received_abilities:
                        _record_ability_if_original(actor, actor_session, event[5])
                    actor_session.received_abilities.add(to_id(event[4]))
                    target_session.received_abilities.add(to_id(event[5]))
                    return

                if effect_kind == "ability" and effect_id in {"mummy", "lingeringaroma"}:
                    source_id = _attributed_identity(
                        event, event[2], label="ability"
                    )
                    target_id = _protocol_identity(event[4]) if len(event) > 4 else None
                    old_ability = None
                    for part in event[5:]:
                        if str(part).lower().startswith("[ability]"):
                            old_ability = str(part)[len("[ability]") :].strip()
                            break
                    if source_id is not None:
                        source = state.pokemon.setdefault(source_id, _ProtocolPokemon())
                        source_session = source.current or source.begin_session()
                        if effect_id not in source_session.received_abilities:
                            ability_name = effect.split(":", 1)[1].strip()
                            _record_ability_if_original(source, source_session, ability_name)
                    if target_id is not None:
                        target = state.pokemon.setdefault(target_id, _ProtocolPokemon())
                        target_session = target.current or target.begin_session()
                        if old_ability and not target_session.received_abilities:
                            _record_ability_if_original(target, target_session, old_ability)
                        target_session.received_abilities.add(effect_id)
                    return

                if effect_kind == "ability" and effect_id == "dancer":
                    dancer_id = _protocol_identity(event[2])
                    if dancer_id is not None:
                        state.dancer_pending.add(dancer_id)

                if effect_kind == "ability":
                    identity = _attributed_identity(event, event[2], label="ability activation")
                    ability_name = effect.split(":", 1)[1].strip()
                    if identity is not None and ability_name:
                        pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                        session = pokemon.current or pokemon.begin_session()
                        if to_id(ability_name) not in session.received_abilities:
                            _record_ability_if_original(pokemon, session, ability_name)

                if effect_kind == "ability" and effect_id == "symbiosis":
                    donor_id = _attributed_identity(
                        event, event[2], label="Symbiosis donor"
                    )
                    receiver_id = None
                    for part in event[4:]:
                        receiver_id = _protocol_identity(part)
                        if receiver_id is not None:
                            break
                    item = event[4] if len(event) > 4 else ""
                    if receiver_id is not None:
                        receiver = state.pokemon.setdefault(receiver_id, _ProtocolPokemon())
                        (receiver.current or receiver.begin_session()).received_item = True
                    if donor_id is not None and item:
                        donor = state.pokemon.setdefault(donor_id, _ProtocolPokemon())
                        donor_session = donor.current or donor.begin_session()
                        if not donor_session.received_item:
                            _record_protocol_observation(donor, "items", item)
                    return

                if effect_id in _TRANSFER_MOVES:
                    actor_id = _protocol_identity(event[2])
                    target_id = None
                    for part in event[4:]:
                        target_id = _protocol_identity(part)
                        if target_id is not None:
                            break
                    if actor_id is not None and target_id is not None:
                        state.pending_item_transfer = _PendingItemTransfer(actor_id, target_id)
                return

            if event_type in {"-item", "-enditem"}:
                if len(event) < 4:
                    logger.debug("Skipping malformed item event: %r", event)
                    return
                identity = _attributed_identity(event, event[2], label="item")
                if identity is None:
                    return
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                session = pokemon.current or pokemon.begin_session()
                origin = _transfer_origin(event)
                item = event[3]

                if event_type == "-item":
                    if origin is None:
                        if not session.received_item:
                            _record_protocol_observation(pokemon, "items", item)
                        return
                    session.received_item = True
                    pending = state.pending_item_transfer
                    pending_complete = False
                    if (
                        pending is not None
                        and identity in {pending.actor, pending.target}
                    ):
                        pending.touched.add(identity)
                        pending_complete = pending.touched == {
                            pending.actor,
                            pending.target,
                        }
                    donor_id = None
                    for part in event[4:]:
                        donor_id = _protocol_identity(part)
                        if donor_id is not None:
                            break
                    if donor_id is None and origin[0] == "move":
                        if pending is not None:
                            donor_id = (
                                pending.target
                                if identity == pending.actor
                                else pending.actor
                                if identity == pending.target
                                else None
                            )
                    if donor_id is not None:
                        donor = state.pokemon.setdefault(donor_id, _ProtocolPokemon())
                        donor_session = donor.current or donor.begin_session()
                        if not donor_session.received_item:
                            _record_protocol_observation(donor, "items", item)
                    else:
                        logger.debug("Cannot attribute donor for transferred item: %r", event)
                    if pending_complete:
                        state.pending_item_transfer = None
                    return

                # An enditem with a transfer origin proves the named holder's
                # removed item, while tagged Trick/Switcheroo enditems are donors.
                if origin is not None:
                    if not session.received_item:
                        _record_protocol_observation(pokemon, "items", item)
                    if state.pending_item_transfer is not None:
                        state.pending_item_transfer.touched.add(identity)
                        if state.pending_item_transfer.touched == {
                            state.pending_item_transfer.actor,
                            state.pending_item_transfer.target,
                        }:
                            state.pending_item_transfer = None
                    return
                if not session.received_item:
                    _record_protocol_observation(pokemon, "items", item)
                return

            if event_type == "-terastallize":
                # battle-actions.ts#terastallize names the Pokémon by its
                # toString(), which is the Illusion name while disguised, so
                # record it in the switch-in session that ``replace`` moves.
                if len(event) < 4:
                    logger.debug("Skipping malformed terastallize event: %r", event)
                    return
                identity = _attributed_identity(event, event[2], label="terastallize")
                if identity is None:
                    return
                pokemon = state.pokemon.setdefault(identity, _ProtocolPokemon())
                _record_protocol_observation(pokemon, "teraTypes", event[3])
                return
        except Exception as exc:
            logger.debug("Skipping odd battle event for %s (%s): %r", battle_tag, exc, event)
            return
        logger.debug("Ignoring unhandled protocol event %r", event)

    def update(self, battle: Any) -> dict[str, SetTracker]:
        """Resolve protocol evidence against the battle's opponent species list.

        Snapshot data is used only for species identity. Moves, abilities,
        items and tera types come from ordered protocol events, because
        poke-env's snapshot can hold transformed, copied, transferred or
        disguised values that are not original set evidence.
        """
        battle_tag = str(
            getattr(battle, "battle_tag", None) or f"<battle:{id(battle)}>"
        )
        state = self._battles.setdefault(battle_tag, _BattleOpponentState())
        team = getattr(battle, "opponent_team", {})
        if not isinstance(team, Mapping):
            logger.warning("Skipping malformed opponent team in battle %s", battle_tag)
            return dict(state.trackers)

        for identifier, pokemon in team.items():
            identifier = str(identifier)
            if identifier in state.trackers:
                continue
            species = getattr(pokemon, "species", None)
            try:
                randbats_species = _resolve_randbats_species(
                    self.data, species, getattr(pokemon, "gen", 9)
                )
            except (KeyError, TypeError, ValueError):
                randbats_species = None
            if randbats_species is None:
                logger.warning(
                    "Skipping opponent Pokemon %s (%r) in battle %s: no randbats species mapping",
                    identifier,
                    species,
                    battle_tag,
                )
                continue
            state.trackers[identifier] = SetTracker(
                self.data,
                randbats_species,
                sibling_species=_hidden_forme_species(self.data, randbats_species),
            )

        zoroark_possible = _zoroark_could_be_on_team(team)
        for identifier in team:
            tracker_id = str(identifier)
            tracker = state.trackers.get(tracker_id)
            if tracker is None:
                continue
            identity = _protocol_identity(tracker_id)
            protocol_pokemon = state.pokemon.get(identity) if identity is not None else None
            can_be_disguise = zoroark_possible and not (
                set(tracker._species_ids) & set(_ZOROARK_SPECIES_IDS)
            )
            observations: list[tuple[str, str, bool]] = []
            for session in protocol_pokemon.sessions if protocol_pokemon else ():
                resolved = self._resolved_observations(tracker, session)
                suspect = can_be_disguise and any(
                    to_id(value) not in self._own_option_ids(tracker, kind)
                    and to_id(value) in _zoroark_option_ids(self.data, kind)
                    for kind, value in resolved
                )
                if (
                    can_be_disguise
                    and not suspect
                    and resolved
                    and self._fits_no_role(tracker, resolved)
                    and all(
                        to_id(value) in _zoroark_option_ids(self.data, kind)
                        for kind, value in resolved
                    )
                ):
                    # Every value is legal for the shown species on its own, but
                    # no single role has them all -- one switch-in cannot be
                    # two roles, while a Zoroark can have them all (e.g. Mew
                    # Encore + Dark Pulse).
                    suspect = True
                observations.extend((kind, value, suspect) for kind, value in resolved)
                if self.inference_enabled:
                    for constraint in session.constraints:
                        observations.append(("constraint", constraint, suspect))
                    for obs in session.speed_obs:
                        speed = self._speed_constraint(battle, obs)
                        if speed is not None:
                            observations.append(("constraint", speed, suspect))
            # The tracker keeps the first copy of a repeated observation, so
            # feed binding evidence first: a clean switch-in confirming
            # something seen during a suspect one makes it binding.
            observations.sort(key=lambda observation: observation[2])
            tracker._replace_observations(observations)

        return dict(state.trackers)

    @staticmethod
    def _speed_constraint(battle: Any, obs: SpeedObs) -> Constraint | None:
        """Resolve a raw speed-order observation against our own mon's stats."""
        role = getattr(battle, "player_role", None)
        if role is None or obs.our_identity[0] != role:
            return None
        team = getattr(battle, "team", None)
        if not isinstance(team, Mapping):
            return None
        for identifier, mon in team.items():
            ident = str(identifier)
            if to_id(ident.partition(":")[2]) != obs.our_identity[1]:
                continue
            stats = getattr(mon, "stats", None) or {}
            ability = getattr(mon, "ability", None)
            item = getattr(mon, "item", None)
            return resolve_speed(obs, stats.get("spe"), ability or "", item or "")
        return None

    @staticmethod
    def _resolved_observations(
        tracker: SetTracker, session: _ProtocolSession
    ) -> list[tuple[str, str]]:
        resolved: list[tuple[str, str]] = []
        for kind, value in session.observations:
            if kind == "abilities":
                ability = _resolve_forme_change_ability(
                    tracker._species_ids, to_id(value)
                )
                if ability is None:
                    continue
                value = ability
            resolved.append((kind, value))
        return resolved

    # -- Illusion suspicion (task 589.9) --------------------------------------

    def illusion_probability(self, battle: Any) -> dict[str, float]:
        """``{zoroark_form_id: P(active opponent is that disguised form)}``.

        Empty when no Zoroark form can be the active one. See
        ``showdown_bot.illusion`` for the tells and constants.
        """
        return dict(self.illusion_estimate(battle).probabilities)

    def illusion_estimate(self, battle: Any) -> IllusionEstimate:
        """Probabilities plus per-form trackers conditioned on this switch-in."""
        from showdown_bot import illusion

        empty = IllusionEstimate({}, {})
        battle_tag = str(
            getattr(battle, "battle_tag", None) or f"<battle:{id(battle)}>"
        )
        state = self._battles.get(battle_tag)
        team = getattr(battle, "opponent_team", None)
        active = getattr(battle, "opponent_active_pokemon", None)
        if state is None or not isinstance(team, Mapping) or active is None:
            return empty
        if getattr(active, "fainted", False):
            return empty
        key = next((str(k) for k, v in team.items() if v is active), None)
        identity = _protocol_identity(key) if key is not None else None
        shown_id = to_id(getattr(active, "species", ""))
        if key is None or identity is None or not shown_id:
            return empty
        tracker = state.trackers.get(key)
        protocol_pokemon = state.pokemon.get(identity)
        session = protocol_pokemon.current if protocol_pokemon else None

        data = self.data
        forms_present = [f for f in _ZOROARK_SPECIES_IDS if data.has_species(f)]
        if shown_id in _ZOROARK_SPECIES_IDS or not forms_present:
            return empty
        revealed = {to_id(getattr(mon, "species", "")): mon for mon in team.values()}
        revealed_forms = [f for f in _ZOROARK_SPECIES_IDS if f in revealed]
        revealed_alive = None
        if revealed_forms:
            # Species Clause: one Zoroark per team, so a revealed form is it.
            form = revealed_forms[0]
            if getattr(revealed[form], "fainted", False):
                return empty
            revealed_alive = form
            weights = {form: 1.0}
        else:
            if len(team) >= 6:
                return empty
            weights = {
                f: float(sum(fs.count for fs in data.full_sets(f)))
                for f in forms_present
            }
            if not any(weights.values()):
                weights = {f: 1.0 for f in forms_present}
        total_count = float(
            sum(fs.count for sid in data.species_ids() for fs in data.full_sets(sid))
        )
        form_counts = {
            f: float(sum(fs.count for fs in data.full_sets(f))) for f in weights
        }
        unrevealed = max(0, 6 - len(team))
        alive = unrevealed + sum(
            1 for mon in team.values() if not getattr(mon, "fainted", False)
        )
        prior = illusion.base_rate(
            form_counts, total_count, unrevealed, alive, revealed_alive
        )

        immune_types: list[str] = []
        direct_hit = False
        kit_suspect = False
        kit_forms: set[str] | None = None
        kit_real = False
        duplicate = None
        if session is not None:
            our_team = getattr(battle, "team", None) or {}
            for move_id, attacker in session.immune_events:
                move_type = illusion.damaging_move_type(move_id)
                if move_type is None:
                    continue
                attacker_mon = next(
                    (
                        mon
                        for k, mon in our_team.items()
                        if _protocol_identity(str(k)) == attacker
                    ),
                    None,
                )
                ability = to_id(getattr(attacker_mon, "ability", "") or "")
                if ability in illusion.TYPE_CHANGING_ABILITIES:
                    continue
                immune_types.append(move_type)
            direct_hit = any(illusion.is_damaging_move(m) for m in session.hit_moves)
            if tracker is not None and not (
                set(tracker._species_ids) & set(_ZOROARK_SPECIES_IDS)
            ):
                own_cache: dict[str, set[str]] = {}
                form_opts = {
                    (f, kind): self._form_option_ids(f, kind)
                    for f in weights
                    for kind in ("moves", "items", "abilities", "teraTypes")
                }
                for kind, value in self._resolved_observations(tracker, session):
                    value_id = to_id(value)
                    own = own_cache.setdefault(kind, self._own_option_ids(tracker, kind))
                    in_forms = {f for f in weights if value_id in form_opts[(f, kind)]}
                    has_data = any(form_opts[(f, kind)] for f in weights)
                    if value_id not in own:
                        if in_forms:
                            kit_suspect = True
                            kit_forms = (
                                set(in_forms) if kit_forms is None else kit_forms & in_forms
                            )
                    elif has_data and not in_forms:
                        kit_real = True
                if kit_suspect and not kit_forms:
                    kit_suspect, kit_forms = False, None
            if session.duplicate == "fainted":
                duplicate = "fainted"
            elif session.duplicate == "hp" and protocol_pokemon is not None:
                duplicate = self._classify_duplicate(protocol_pokemon, session, weights)

        evidence = illusion.IllusionEvidence(
            shown_species=shown_id,
            form_weights=weights,
            prior=prior,
            shown_types=illusion.species_types(shown_id),
            immune_types=tuple(immune_types),
            kit_suspect=kit_suspect,
            kit_forms=kit_forms,
            kit_real=kit_real,
            duplicate=duplicate,
            direct_hit=direct_hit,
        )
        probabilities = illusion.estimate(evidence)
        form_trackers: dict[str, SetTracker] = {}
        for form, probability in probabilities.items():
            if probability <= 0:
                continue
            form_tracker = SetTracker(data, form)
            if session is not None and tracker is not None:
                for kind, value in self._resolved_observations(tracker, session):
                    if to_id(value) in self._form_option_ids(form, kind):
                        form_tracker._observe(kind, value)
            form_trackers[form] = form_tracker
        return IllusionEstimate(probabilities, form_trackers)

    def _form_option_ids(self, form: str, kind: str) -> set[str]:
        return {
            to_id(option)
            for role in self.data.roles(form).values()
            for option in role.options(kind)
        }

    def _classify_duplicate(
        self,
        pokemon: _ProtocolPokemon,
        session: _ProtocolSession,
        weights: Mapping[str, float],
    ) -> str:
        """Which of two Pokémon sharing a name is the Zoroark, if decidable."""

        def zoroark_cannot_have(observations: Sequence[tuple[str, str]]) -> bool:
            for kind, value in observations:
                options = [self._form_option_ids(f, kind) for f in weights]
                if any(options) and all(to_id(value) not in o for o in options):
                    return True
            return False

        if zoroark_cannot_have(
            [obs for obs in session.observations if obs[0] != "abilities"]
        ):
            return "current_real"
        earlier = [
            obs
            for other in pokemon.sessions
            if other is not session
            for obs in other.observations
            if obs[0] != "abilities"
        ]
        if zoroark_cannot_have(earlier):
            return "earlier_real"
        return "ambiguous"

    @staticmethod
    def _fits_no_role(tracker: SetTracker, observations: Sequence[tuple[str, str]]) -> bool:
        """True when no single randbats role of ``tracker`` has every observation."""
        for role in tracker._roles.values():
            if all(
                to_id(value) in {to_id(option) for option in role.options(kind)}
                for kind, value in observations
            ):
                return False
        return bool(tracker._roles)

    def _own_option_ids(self, tracker: SetTracker, kind: str) -> set[str]:
        return {
            to_id(option)
            for species_id in tracker._species_ids
            for role in self.data.roles(species_id).values()
            for option in role.options(kind)
        }

    def forget(self, battle_tag: str) -> None:
        """Release one finished battle after its results are no longer needed."""
        self._battles.pop(str(battle_tag), None)


def _real_battle_tag(split_messages: list[list[str]]) -> str | None:
    """Return the battle tag for one raw ``on_battle_message`` payload, if any.

    poke-env's ``PSClient._handle_message`` splits each websocket frame into
    lines, then pipes, and routes it here whenever the first line's first
    token starts with ``">battle"`` or ``">game"`` (see ps_client.py around
    line 168). ``Player._handle_battle_message`` (player.py around line 295)
    and ``_create_battle``/``_get_battle`` (player.py lines 182-262) both key
    battles by that first token with exactly its leading ``">"`` stripped, for
    both the "init" (new battle) and ongoing-battle code paths; that result is
    ``battle.battle_tag``. ``">game-..."`` lines are a best-of-series room
    wrapper around individual sub-battles, not a battle themselves, and carry
    no Pokémon events worth observing, so they are skipped here.

    Callers that already hold a bare battle tag (direct unit tests, replay
    tooling) may pass it as ``split_messages[0][0]`` too; it is returned as-is
    since it can never collide with a real ``">..."`` room tag.
    """
    if not split_messages or not split_messages[0]:
        return None
    raw_tag = str(split_messages[0][0])
    if raw_tag.startswith(">battle"):
        return raw_tag[1:]
    if raw_tag.startswith(">"):
        return None
    return raw_tag


class OpponentModelPlayerMixin:
    """Feed raw Showdown records to an ``OpponentModel`` in poke-env players.

    Use as the first base (``class MyPlayer(OpponentModelPlayerMixin, Player)``)
    and pass ``opponent_model=...`` to the constructor. The model refreshes
    before ``Player`` handles each request, so ``choose_move`` sees current
    evidence.
    """

    def __init__(self, *args: Any, opponent_model: OpponentModel, **kwargs: Any):
        self.opponent_model = opponent_model
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> Any:
        battle_tag = _real_battle_tag(split_messages)
        if battle_tag is not None:
            self.opponent_model.observe_events(
                battle_tag, split_messages[1:], getattr(self, "_battles", {}).get(battle_tag)
            )
        result = await super()._handle_battle_message(split_messages)
        if battle_tag is not None:
            battle = getattr(self, "_battles", {}).get(battle_tag)
            if battle is not None:
                self.opponent_model.update(battle)
        return result

    async def _handle_battle_request(
        self, battle: Any, *args: Any, **kwargs: Any
    ) -> Any:
        self.opponent_model.update(battle)
        return await super()._handle_battle_request(battle, *args, **kwargs)
