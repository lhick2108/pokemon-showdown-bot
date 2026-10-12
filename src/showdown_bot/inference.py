"""Opponent set inference from negative / indirect protocol evidence (bead 589.28).

``sets.OpponentModel`` narrows an opposing Pokemon's randbats set from things it
*reveals* (moves, abilities, items).  This module adds the things it *fails to
do*, which also constrain the set:

* it hit with a damaging move and took no Life Orb recoil (no Life Orb),
* it used two different moves since switching in (no Choice item),
* it used a status move (no Assault Vest),
* it ended a turn below max HP with no Leftovers/Black Sludge heal,
* it ended a turn on the field with no status and no Flame/Toxic Orb activation,
* it switched into hazards and took no damage/effect (Heavy-Duty Boots, or an
  ability/typing that explains it),
* it moved before / after our Pokemon at equal priority (Choice Scarf / speed).

The tracker here is a pure state machine over raw split protocol events (the
same lists ``OpponentModel.observe`` receives).  It never touches sets: it emits
``(identity, Constraint)`` pairs, and ``sets.py`` stores each constraint on the
protocol *session* of that Pokemon, so Illusion's provisional-evidence machinery
(sessions move to the real Pokemon on ``replace`` and are held out of sampling
when suspect) applies to constraints exactly as it does to moves and items.

A ``Constraint`` is a predicate over a candidate full set: ``constraint_allows``.
Everything is deliberately conservative; whenever something uncertain could
explain a missing event (an ability that might block the effect, a field effect,
a held-out turn) the constraint is simply not emitted or the candidate is kept.

The implementation is original and written from the Showdown protocol and the
simulator's documented item/ability behaviour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Sequence

from poke_env.data.gen_data import GenData

_GEN = 9


def to_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower()) if value is not None else ""


Identity = tuple[str, str]  # (side "p1"/"p2", name id)

# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """A predicate on candidate full sets.

    kind:
      ``item_not``    data = excluded item ids.
      ``no_lo``       hit with a damaging move, no Life Orb recoil.
      ``no_heal``     data = item ids whose end-of-turn heal was absent.
      ``no_orb``      data = ("flameorb"|"toxicorb",): no activation at end of turn.
      ``no_hazard``   data = hazards that stayed silent on a switch-in.
      ``speed``       data = (our_speed, opp_first, trick_room, stage, par, tailwind)
    """

    kind: str
    data: tuple = ()


CHOICE_ITEMS = ("choiceband", "choicespecs", "choicescarf")

# Abilities that can stop a hazard / orb / heal effect or change speed in a way
# the checks below don't model; candidates with them stay consistent.
_BURN_IMMUNE_ABILITIES = frozenset(
    {"waterveil", "waterbubble", "thermalexchange", "comatose", "purifyingsalt",
     "shieldsdown", "leafguard", "flowerveil", "klutz"}
)
_POISON_IMMUNE_ABILITIES = frozenset(
    {"immunity", "pastelveil", "comatose", "purifyingsalt", "shieldsdown",
     "leafguard", "flowerveil", "klutz"}
)
_SPEED_ABILITIES = frozenset(
    {"swiftswim", "chlorophyll", "sandrush", "slushrush", "surgesurfer",
     "unburden", "quickfeet", "protosynthesis", "quarkdrive", "slowstart",
     "stall"}
)
_PRIORITY_ABILITIES = frozenset({"prankster", "galewings", "triage", "myceliummight"})

# Moves whose priority is not their data value (or whose order is otherwise not
# plain speed).
_ODD_PRIORITY_MOVES = frozenset(
    {"grassyglide", "pursuit", "counter", "mirrorcoat", "metalburst", "focuspunch",
     "beakblast", "shelltrap", "upperhand", "firstimpression"}
)


@lru_cache(maxsize=None)
def _dex(species_id: str) -> dict:
    return GenData.from_gen(_GEN).pokedex.get(species_id) or {}


@lru_cache(maxsize=None)
def _move(move_id: str) -> dict:
    return GenData.from_gen(_GEN).moves.get(move_id) or {}


def species_types(species_id: str) -> frozenset[str]:
    return frozenset(to_id(t) for t in _dex(species_id).get("types", ()))


@lru_cache(maxsize=None)
def base_speed(species_id: str) -> int | None:
    base = _dex(species_id).get("baseStats")
    return int(base["spe"]) if base else None


@lru_cache(maxsize=65536)
def _set_speed_stat(
    species_id: str, level: int, moves: tuple[str, ...], ability: str, item: str
) -> int | None:
    base = base_speed(species_id)
    if base is None:
        return None
    from showdown_bot import engine_state  # lazy: engine_state imports sets

    evs, ivs = engine_state.randbats_evs_ivs(species_id, level, moves, ability, item)
    return engine_state.calc_stat(base, level, evs[5], ivs[5])


_STAGE = {s: (2 + s) / 2 if s >= 0 else 2 / (2 - s) for s in range(-6, 7)}


def effective_speed(stat: int, stage: int, par: bool, tailwind: bool, scarf: bool) -> float:
    speed = int(stat * _STAGE[max(-6, min(6, stage))])
    if scarf:
        speed = int(speed * 1.5)
    if tailwind:
        speed *= 2
    if par:
        speed = int(speed * 0.5)
    return speed


def _hazard_blocked(hazard: str, types: frozenset[str], ability: str, item: str) -> bool:
    """Whether the set would take no effect from ``hazard`` on switch-in."""
    if hazard == "stealthrock":
        return ability == "magicguard"
    airborne = "flying" in types or ability == "levitate" or item == "airballoon"
    if hazard == "spikes":
        return airborne or ability == "magicguard"
    if hazard == "stickyweb":
        return airborne
    return False


def constraint_allows(constraint: Constraint, species_id: str, full_set: Any) -> bool:
    """Whether ``full_set`` (a ``sets.FullSet``) is consistent with the constraint."""
    item = to_id(full_set.item) if full_set.item else ""
    ability = to_id(full_set.ability)
    kind = constraint.kind
    if kind == "item_not":
        return item not in constraint.data
    if kind == "no_lo":
        return not (item == "lifeorb" and ability not in ("magicguard", "sheerforce"))
    if kind == "no_heal":
        if ability == "klutz" or item not in constraint.data:
            return True
        if item == "blacksludge":
            return "poison" not in species_types(species_id)  # others take damage instead
        return False
    if kind == "no_orb":
        orb = constraint.data[0]
        if item != orb or ability == "klutz":
            return True
        types = species_types(species_id)
        if orb == "flameorb":
            return "fire" in types or ability in _BURN_IMMUNE_ABILITIES
        return bool(types & {"poison", "steel"}) or ability in _POISON_IMMUNE_ABILITIES
    if kind == "no_hazard":
        if item == "heavydutyboots":
            return True
        types = species_types(species_id)
        return all(_hazard_blocked(h, types, ability, item) for h in constraint.data)
    if kind == "speed":
        return _speed_allows(constraint.data, species_id, full_set, item, ability)
    return True


def _speed_allows(data: tuple, species_id: str, full_set: Any, item: str, ability: str) -> bool:
    our_speed, opp_first, trick_room, stage, par, tailwind, status_move = data
    if ability in _SPEED_ABILITIES:
        return True
    if status_move and ability in _PRIORITY_ABILITIES:
        return True
    if ability == "galewings":
        return True
    stat = _set_speed_stat(
        species_id, int(full_set.level), tuple(to_id(m) for m in full_set.moves), ability, item
    )
    if stat is None:
        return True
    speed = effective_speed(stat, stage, par, tailwind, item == "choicescarf")
    if item == "ironball":
        speed = int(speed * 0.5)
    tolerance = 1.0
    faster = opp_first != trick_room  # True: the opponent acted first and TR is off, or vice versa
    if faster:
        return speed + tolerance >= our_speed
    return speed - tolerance <= our_speed


# ---------------------------------------------------------------------------
# Raw speed observation (resolved against our own stats in OpponentModel.update)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpeedObs:
    """The opponent acted before/after ``our`` mon at equal priority."""

    our_identity: Identity
    opp_first: bool
    trick_room: bool
    opp_stage: int
    opp_par: bool
    opp_tailwind: bool
    our_stage: int
    our_par: bool
    our_tailwind: bool
    status_move: bool  # either move was a status move (Prankster possible)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


@dataclass
class _Mon:
    hp: float | None = None
    status: str = ""
    stage_spe: int = 0
    moves_used: list[str] = field(default_factory=list)
    item_changed: bool = False
    transformed: bool = False
    terastallized: bool = False
    heal_blocked: bool = False
    # residual-phase flags (reset at the blank line that starts the residual)
    res_heal: bool = False
    res_status: bool = False
    pre_hp: float | None = None
    pre_status: str = ""
    in_residual: bool = False


@dataclass
class _Pending:
    kind: str  # "lo" | "switch"
    identity: Identity
    saw_hit: bool = False
    spoiled: bool = False
    acted: set = field(default_factory=set)
    hazards: tuple = ()
    skip: bool = False


Emission = tuple  # (identity, Constraint | SpeedObs)


def _parse_hp(text: Any) -> float | None:
    token = str(text).strip().split(" ")[0]
    if not token or token.startswith("["):
        return None
    if "/" in token:
        cur, _, mx = token.partition("/")
        try:
            return max(0.0, min(1.0, float(cur) / float(mx)))
        except (ValueError, ZeroDivisionError):
            return None
    return 0.0 if token == "0" else None


def _status_of(text: Any) -> str:
    parts = str(text).strip().split(" ")
    return parts[1] if len(parts) > 1 and not parts[1].startswith("[") else ""


_IDENT = re.compile(r"^(p[12])([a-z])?:\s*(.+)$", re.IGNORECASE)


def _ident(value: Any) -> Identity | None:
    match = _IDENT.match(str(value).strip())
    if not match:
        return None
    name = to_id(match.group(3))
    return (match.group(1).lower(), name) if name else None


def _slot(value: Any) -> tuple[str, str] | None:
    match = _IDENT.match(str(value).strip())
    if not match or not match.group(2):
        return None
    return match.group(1).lower(), match.group(2).lower()


def _tag(event: Sequence[str], prefix: str) -> str | None:
    for part in event:
        text = str(part)
        if text.lower().startswith(prefix):
            return text[len(prefix):].strip()
    return None


class InferenceTracker:
    """Per-battle state machine; ``feed`` returns the evidence one event completes."""

    def __init__(self) -> None:
        self.active: dict[tuple[str, str], Identity] = {}
        self.mons: dict[Identity, _Mon] = {}
        self.hazards: dict[str, dict[str, int]] = {"p1": {}, "p2": {}}
        self.tailwind: dict[str, bool] = {"p1": False, "p2": False}
        self.trick_room = False
        self.gravity = False
        self.magic_room = False
        self.misty = False
        self.safeguard: dict[str, bool] = {"p1": False, "p2": False}
        self.weather = ""
        self.pending: list[_Pending] = []
        self.actions: list[tuple[str, str, Identity | None]] = []  # (kind, move id, identity)
        self.action_status: list[bool] = []
        self.turn_snapshot: tuple | None = None
        self.speed_dirty = False
        self.turn_evaluated = False

    # -- helpers ---------------------------------------------------------

    def _mon(self, identity: Identity) -> _Mon:
        return self.mons.setdefault(identity, _Mon())

    def _speed_signature(self) -> tuple:
        sig = [self.trick_room, self.weather, self.tailwind["p1"], self.tailwind["p2"]]
        for slot in (("p1", "a"), ("p2", "a")):
            identity = self.active.get(slot)
            mon = self.mons.get(identity) if identity else None
            sig.append((identity, mon.stage_spe, mon.status == "par") if mon else None)
        return tuple(sig)

    def _close_pending(self, out: list[Emission]) -> None:
        pending, self.pending = self.pending, []
        for item in pending:
            if item.skip or item.spoiled:
                continue
            mon = self.mons.get(item.identity)
            if item.kind == "lo":
                if item.saw_hit and mon is not None and (mon.hp or 0) > 0:
                    out.append((item.identity, Constraint("no_lo")))
            elif item.kind == "switch":
                silent = tuple(h for h in item.hazards if h not in item.acted)
                if item.acted:
                    out.append((item.identity, Constraint("item_not", ("heavydutyboots",))))
                silent = tuple(
                    h for h in silent if h in ("stealthrock", "spikes", "stickyweb")
                )
                if silent and not self.gravity:
                    out.append((item.identity, Constraint("no_hazard", tuple(sorted(silent)))))

    # -- main entry ------------------------------------------------------

    def feed(self, event: Sequence[str]) -> list[Emission]:
        out: list[Emission] = []
        if len(event) < 2:
            return out
        kind = str(event[1]).lower()
        handler = getattr(self, "_on_" + re.sub(r"[^a-z]", "", kind), None) if kind else self._on_blank
        if handler is not None:
            handler(event, out)
        return out

    # -- handlers --------------------------------------------------------

    def _on_blank(self, event, out) -> None:
        self._close_pending(out)
        for identity in self.active.values():
            mon = self._mon(identity)
            mon.in_residual = True
            mon.res_heal = False
            mon.res_status = False
            mon.pre_hp = mon.hp
            mon.pre_status = mon.status

    def _on_turn(self, event, out) -> None:
        self._close_pending(out)
        self.actions = []
        self.action_status = []
        self.turn_snapshot = self._speed_signature()
        self.turn_evaluated = False
        for mon in self.mons.values():
            mon.in_residual = False

    def _on_cant(self, event, out) -> None:
        self._close_pending(out)
        identity = _ident(event[2]) if len(event) > 2 else None
        self._record_action("cant", "", identity, False, out)

    def _on_faint(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            mon = self._mon(identity)
            mon.hp = 0.0

    def _on_win(self, event, out) -> None:
        self._close_pending(out)

    def _on_switch(self, event, out) -> None:
        self._close_pending(out)
        if len(event) < 5:
            return
        identity, slot = _ident(event[2]), _slot(event[2])
        if identity is None or slot is None:
            return
        self.active[slot] = identity
        mon = _Mon(hp=_parse_hp(event[4]), status=_status_of(event[4]))
        mon.terastallized = "tera:" in str(event[3]).lower()
        self.mons[identity] = mon
        self._record_action("switch", "", identity, False, out)
        hazards = tuple(sorted(self.hazards[identity[0]]))
        if hazards:
            self.pending.append(
                _Pending("switch", identity, hazards=hazards, skip=mon.terastallized)
            )

    _on_drag = _on_switch

    def _on_replace(self, event, out) -> None:
        # Illusion ends: the HP/state seen under the disguise name is the
        # Zoroark's, but we cannot reattribute pending checks; drop them.
        self.pending = []
        self.actions = []
        slot = _slot(event[2]) if len(event) > 2 else None
        identity = _ident(event[2]) if len(event) > 2 else None
        if slot and identity:
            self.active[slot] = identity
            self.mons[identity] = _Mon(hp=_parse_hp(event[4]) if len(event) > 4 else None)

    def _record_action(self, kind, move_id, identity, status_move, out) -> None:
        self.actions.append((kind, move_id, identity))
        self.action_status.append(status_move)
        if len(self.actions) == 2 and not self.turn_evaluated:
            self.turn_evaluated = True
            self._evaluate_speed(out)

    def _evaluate_speed(self, out) -> None:
        first, second = self.actions
        if first[0] != "move" or second[0] != "move":
            return
        (_, move_a, who_a), (_, move_b, who_b) = first, second
        if who_a is None or who_b is None or who_a[0] == who_b[0]:
            return
        if self.turn_snapshot != self._speed_signature():
            return
        data_a, data_b = _move(move_a), _move(move_b)
        if not data_a or not data_b:
            return
        if int(data_a.get("priority", 0)) != int(data_b.get("priority", 0)):
            return
        if move_a in _ODD_PRIORITY_MOVES or move_b in _ODD_PRIORITY_MOVES:
            return
        for identity in (who_a, who_b):
            if self.active.get((identity[0], "a")) != identity:
                return
        if any(abs(self._mon(i).stage_spe) > 6 for i in (who_a, who_b)):
            return
        status_move = any(self.action_status)
        for opp_is_first, (opp, us) in ((True, (who_a, who_b)), (False, (who_b, who_a))):
            opp_mon, our_mon = self._mon(opp), self._mon(us)
            obs = SpeedObs(
                our_identity=us,
                opp_first=opp_is_first,
                trick_room=self.trick_room,
                opp_stage=opp_mon.stage_spe,
                opp_par=opp_mon.status == "par",
                opp_tailwind=self.tailwind[opp[0]],
                our_stage=our_mon.stage_spe,
                our_par=our_mon.status == "par",
                our_tailwind=self.tailwind[us[0]],
                status_move=status_move,
            )
            out.append((opp, obs))

    def _on_move(self, event, out) -> None:
        self._close_pending(out)
        if len(event) < 4:
            return
        identity = _ident(event[2])
        if identity is None:
            return
        move_id = to_id(event[3])
        foreign = any(str(p).strip().lower().startswith("[from]") for p in event[4:])
        mon = self._mon(identity)
        data = _move(move_id)
        category = data.get("category", "")
        if foreign:
            return
        self._record_action("move", move_id, identity, category == "Status", out)
        if move_id == "struggle" or mon.transformed:
            return
        if move_id not in mon.moves_used:
            mon.moves_used.append(move_id)
        if not mon.item_changed:
            if len(mon.moves_used) >= 2:
                out.append((identity, Constraint("item_not", CHOICE_ITEMS)))
            if category == "Status":
                out.append((identity, Constraint("item_not", ("assaultvest",))))
        if category in ("Physical", "Special") and move_id not in ("dragontail", "circlethrow"):
            if any(str(p).lower().startswith(("[miss]", "[notarget]")) for p in event[4:]):
                return
            target = _slot(event[4]) if len(event) > 4 else None
            actor_slot = _slot(event[2])
            if target and actor_slot and target[0] != actor_slot[0]:
                self.pending.append(_Pending("lo", identity))

    def _pending_of(self, kind, identity) -> _Pending | None:
        for item in self.pending:
            if item.kind == kind and item.identity == identity:
                return item
        return None

    def _spoil_lo_impl(self) -> None:
        for item in self.pending:
            if item.kind == "lo":
                item.spoiled = True

    def _spoil(self, event, out) -> None:
        self._spoil_lo_impl()

    _on_miss = _on_fail = _on_notarget = _on_immune = _on_block = _spoil

    def _on_enditem(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).item_changed = True
        self._spoil_lo_impl()

    def _on_item(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).item_changed = True

    def _on_transform(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).transformed = True

    def _on_terastallize(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).terastallized = True
        for item in self.pending:
            if item.identity == identity:
                item.skip = True

    def _on_activate(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        effect = to_id(str(event[3]).split(":", 1)[-1])
        if effect == "stickyweb" and identity:
            pending = self._pending_of("switch", identity)
            if pending is not None:
                pending.acted.add("stickyweb")
        if effect in {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker",
                      "silktrap", "burningbulwark", "obstruct", "substitute", "endure"}:
            self._spoil_lo_impl()
        if effect == "trick" or effect == "switcheroo":
            for key in (identity, _ident(event[4]) if len(event) > 4 else None):
                if key:
                    self._mon(key).item_changed = True

    def _on_damage(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        if identity is None:
            return
        mon = self._mon(identity)
        hp = _parse_hp(event[3])
        if hp is not None:
            mon.hp = hp
        origin = _tag(event[4:], "[from]") or ""
        origin_id = to_id(origin.split(":", 1)[-1])
        if origin_id in ("stealthrock", "spikes"):
            pending = self._pending_of("switch", identity)
            if pending is not None:
                pending.acted.add(origin_id)
            return
        if origin.lower().startswith("item:"):
            if origin_id == "lifeorb":
                for item in self.pending:
                    if item.kind == "lo" and item.identity == identity:
                        item.spoiled = True
            if mon.in_residual:
                mon.res_heal = mon.res_heal or origin_id == "blacksludge"
            return
        if origin:
            return
        # Direct damage: the attacker's pending Life Orb check saw a hit.
        for item in self.pending:
            if item.kind == "lo" and item.identity[0] != identity[0]:
                item.saw_hit = True

    def _on_heal(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        if identity is None:
            return
        mon = self._mon(identity)
        hp = _parse_hp(event[3])
        if hp is not None:
            mon.hp = hp
        if mon.in_residual:
            mon.res_heal = True

    def _on_sethp(self, event, out) -> None:
        self._on_heal(event, out)

    def _on_status(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        if identity is None:
            return
        mon = self._mon(identity)
        mon.status = str(event[3]).strip().lower()
        if mon.in_residual:
            mon.res_status = True
        pending = self._pending_of("switch", identity)
        if pending is not None and "toxicspikes" in pending.hazards:
            pending.acted.add("toxicspikes")

    def _on_cureteam(self, event, out) -> None:
        side = str(event[2])[:2].lower() if len(event) > 2 else ""
        for identity, mon in self.mons.items():
            if identity[0] == side:
                mon.status = ""

    def _on_curestatus(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).status = ""

    def _on_boost(self, event, out) -> None:
        self._stage(event, +1)

    def _on_unboost(self, event, out) -> None:
        self._stage(event, -1)

    def _stage(self, event, sign) -> None:
        if len(event) < 5 or to_id(event[3]) != "spe":
            return
        identity = _ident(event[2])
        if identity is None:
            return
        try:
            amount = int(event[4])
        except ValueError:
            return
        mon = self._mon(identity)
        mon.stage_spe = max(-6, min(6, mon.stage_spe + sign * amount))

    def _on_setboost(self, event, out) -> None:
        if len(event) >= 5 and to_id(event[3]) == "spe":
            identity = _ident(event[2])
            if identity:
                try:
                    self._mon(identity).stage_spe = int(event[4])
                except ValueError:
                    pass

    def _clear_all_boosts(self) -> None:
        for mon in self.mons.values():
            mon.stage_spe = 0

    def _on_clearallboost(self, event, out) -> None:
        self._clear_all_boosts()

    def _on_clearboost(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).stage_spe = 0

    def _on_clearnegativeboost(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            mon = self._mon(identity)
            mon.stage_spe = max(0, mon.stage_spe)

    def _on_invertboost(self, event, out) -> None:
        identity = _ident(event[2]) if len(event) > 2 else None
        if identity:
            self._mon(identity).stage_spe *= -1

    def _unknown_boosts(self, event, out) -> None:
        # Swap/copy/restore: we cannot follow the speed stage; poison the turn.
        for mon in self.mons.values():
            mon.stage_spe = 99  # outside table: speed signature always differs

    _on_swapboost = _on_copyboost = _on_restoreboost = _unknown_boosts

    def _on_start(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        effect = to_id(str(event[3]).split(":", 1)[-1])
        if identity and effect in ("healblock", "embargo"):
            self._mon(identity).heal_blocked = True
        if identity and effect == "mimic":
            self._mon(identity).transformed = True

    def _on_end(self, event, out) -> None:
        if len(event) < 4:
            return
        identity = _ident(event[2])
        effect = to_id(str(event[3]).split(":", 1)[-1])
        if identity and effect in ("healblock", "embargo"):
            self._mon(identity).heal_blocked = False

    def _on_sidestart(self, event, out) -> None:
        if len(event) < 4:
            return
        side = (_ident(event[2]) or (str(event[2])[:2].lower(), ""))[0]
        if side not in self.hazards:
            return
        effect = to_id(str(event[3]).split(":", 1)[-1])
        if effect in ("stealthrock", "spikes", "toxicspikes", "stickyweb"):
            self.hazards[side][effect] = self.hazards[side].get(effect, 0) + 1
        elif effect == "tailwind":
            self.tailwind[side] = True
        elif effect == "safeguard":
            self.safeguard[side] = True

    def _on_sideend(self, event, out) -> None:
        if len(event) < 4:
            return
        side = str(event[2])[:2].lower()
        if side not in self.hazards:
            return
        effect = to_id(str(event[3]).split(":", 1)[-1])
        self.hazards[side].pop(effect, None)
        if effect == "tailwind":
            self.tailwind[side] = False
        elif effect == "safeguard":
            self.safeguard[side] = False

    def _on_fieldstart(self, event, out) -> None:
        effect = to_id(str(event[2]).split(":", 1)[-1]) if len(event) > 2 else ""
        if effect == "trickroom":
            self.trick_room = True
        elif effect == "gravity":
            self.gravity = True
        elif effect == "magicroom":
            self.magic_room = True
        elif effect == "mistyterrain":
            self.misty = True

    def _on_fieldend(self, event, out) -> None:
        effect = to_id(str(event[2]).split(":", 1)[-1]) if len(event) > 2 else ""
        if effect == "trickroom":
            self.trick_room = False
        elif effect == "gravity":
            self.gravity = False
        elif effect == "magicroom":
            self.magic_room = False
        elif effect == "mistyterrain":
            self.misty = False

    def _on_weather(self, event, out) -> None:
        if len(event) > 2:
            self.weather = to_id(event[2]) if to_id(event[2]) != "none" else ""

    def _on_upkeep(self, event, out) -> None:
        self._close_pending(out)
        for slot, identity in self.active.items():
            mon = self.mons.get(identity)
            if mon is None or not mon.in_residual or mon.terastallized:
                continue
            if mon.pre_hp is None or mon.hp is None or mon.hp <= 0 or mon.pre_hp <= 0:
                continue
            side = identity[0]
            # Leftovers / Black Sludge.
            if (
                mon.pre_hp < 0.995
                and not mon.res_heal
                and not mon.heal_blocked
                and not self.magic_room
                and mon.hp < 1.0
            ):
                # res_heal covers every heal source in the residual phase (a
                # heal that tops the mon up would also hide a missing item).
                out.append((identity, Constraint("no_heal", ("leftovers", "blacksludge"))))
            # Flame / Toxic Orb.
            if (
                mon.pre_status == ""
                and mon.status == ""
                and not mon.res_status
                and not self.magic_room
                and not self.misty
                and not self.safeguard[side]
            ):
                out.append((identity, Constraint("no_orb", ("flameorb",))))
                out.append((identity, Constraint("no_orb", ("toxicorb",))))


_OUR_SKIP_ITEMS = frozenset({"quickclaw", "laggingtail", "fullincense", "custapberry", "quickpowder"})


def resolve_speed(
    obs: SpeedObs, stat: int | None, ability: str, item: str
) -> Constraint | None:
    """Turn a raw ``SpeedObs`` into a constraint using our mon's real speed stat.

    Returns None when anything on our side could change the move order in a way
    this model does not track (priority/speed abilities, order-changing items).
    """
    if stat is None or not (-6 <= obs.our_stage <= 6) or not (-6 <= obs.opp_stage <= 6):
        return None
    ability, item = to_id(ability), to_id(item)
    if ability in _SPEED_ABILITIES or ability in _PRIORITY_ABILITIES or item in _OUR_SKIP_ITEMS:
        return None
    ours = effective_speed(
        int(stat), obs.our_stage, obs.our_par, obs.our_tailwind, item == "choicescarf"
    )
    if item == "ironball":
        ours = int(ours * 0.5)
    return Constraint(
        "speed",
        (
            ours,
            obs.opp_first,
            obs.trick_room,
            obs.opp_stage,
            obs.opp_par,
            obs.opp_tailwind,
            obs.status_move,
        ),
    )
