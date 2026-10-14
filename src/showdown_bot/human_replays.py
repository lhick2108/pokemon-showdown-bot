"""Human Showdown battle logs -> value records (self-play format) + policy rows.

Input: server-side logs of human ``gen9randombattle`` games (PokeChamp dataset): lowercased
names, exact HP for both sides, no ``|request|`` lines. See ``docs/human-replays.md``.

Pipeline for one game (``convert_game``):

1. ``sanitize_log`` keeps only battle events and the winner's *side*. Chat, joins, usernames,
   timestamps and ``|-message|`` lines never leave this function.
2. Hindsight teams (Metamon style). For each POV role the whole log is replayed through
   poke-env with the ``OpponentModel`` (damage inference off, see ``hindsight_team``); every
   reveal about the OTHER side (moves, item, ability, tera type, ...) narrows that side's
   ``SetTracker``. A full randbats set is then sampled for each revealed Pokemon and the
   unrevealed slots are filled by ``TeamSampler`` (``es.sample_opponent_team``). Both sides
   therefore get a complete, consistent six-Pokemon team.
3. For each side the log is replayed again with that side's hindsight team fed to poke-env as
   a synthetic ``|request|`` (exact randbats stats, moves, item, ability, Tera type) and the
   other side's hindsight team as the "opponent" slots. At every ``|turn|N`` the engine state
   is built with ``engine_state.build_state`` (our side = engine side one) -- a truth-like
   state: current HP is exact for both sides, status / boosts / volatiles / side conditions /
   field / Tera state / PP come from the log via poke-env.
4. The action each side took at that turn is read off the turn's events and mapped to engine
   choice strings (``closecombat``, ``closecombat-tera``, ``switch kingambit``).
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poke_env.battle.battle import Battle

from showdown_bot import engine_state as es
from showdown_bot import recorder
from showdown_bot import sleep as sleep_module
from showdown_bot.sets import OpponentModel, RandbatsData, SampledSet, to_id

logger = logging.getLogger(__name__)

FORMAT = "gen9randombattle"

# --- sanitising -----------------------------------------------------------------------

# Event types that survive sanitising. Everything else (chat |c|, |j|, |l|, |n|, |player|,
# |win|, |inactive|, |t:|, |raw|, |-message|, |-hint|, ...) is dropped: those lines are
# where usernames and free text live.
_KEEP = frozenset(
    {"start", "switch", "drag", "replace", "detailschange", "move", "cant", "faint", "turn",
     "upkeep", "swap"}
)
_DROP_DASH = frozenset({"-message", "-hint", "-anim", "-center"})
_SIDE_ONLY = re.compile(r"^p([12]): .*$")
_SIDE_EVENTS = frozenset({"-sidestart", "-sideend", "-swapsideconditions"})


def _fix_details(details: str) -> str:
    """``eternatus, l69, f`` -> ``eternatus, L69, F`` (poke-env wants Showdown's casing)."""
    parts = [p.strip() for p in details.split(",")]
    out = [parts[0]]
    for part in parts[1:]:
        if re.fullmatch(r"l\d+", part):
            part = "L" + part[1:]
        elif part in ("m", "f"):
            part = part.upper()
        elif part.startswith("tera:"):
            part = "tera:" + part[5:].capitalize()
        out.append(part)
    return ", ".join(out)


_DISPLAY: dict[str, dict[str, str]] = {}
_TAGGED = re.compile(r"^(\[from\] ?|\[from\])?(move|ability|item): (.+)$")


def _display_maps() -> dict[str, dict[str, str]]:
    """id -> Showdown display name for moves / abilities / items (poke-env compares them
    case-sensitively, e.g. ``[from] ability: Frisk``, while the human logs are lowercased)."""
    if not _DISPLAY:
        from poke_env.data import GenData

        gen = GenData.from_gen(9)
        _DISPLAY["move"] = {k: v["name"] for k, v in gen.moves.items() if "name" in v}
        abilities: dict[str, str] = {}
        for entry in gen.pokedex.values():
            for name in entry.get("abilities", {}).values():
                abilities[to_id(name)] = name
        _DISPLAY["ability"] = abilities
        items: dict[str, str] = {}
        from showdown_bot.players.search import default_randbats_data

        data = default_randbats_data()
        for species in data.species_ids():
            for role in data.roles(species).values():
                for name in role.options("items"):
                    items[to_id(name)] = name
        _DISPLAY["item"] = items
    return _DISPLAY


def _restore_case(part: str) -> str:
    match = _TAGGED.match(part)
    if match is None:
        return part
    prefix, kind, name = match.groups()
    shown = _display_maps()[kind].get(to_id(name)) or name.title()
    return f"{prefix or ''}{kind}: {shown}"


@dataclass
class Log:
    events: list[list[str]]  # poke-env style: ["", "switch", "p1a: x", ...]
    winner: str | None  # "p1" / "p2" / "tie" / None (no result)
    skip: str | None = None  # reason this game cannot be converted
    turns: int = 0


def sanitize_log(text: str) -> Log:
    """Parse a raw server log into battle events plus the winner's side.

    Privacy: only the side ("p1"/"p2") of the winner is kept; usernames and chat are
    discarded here.
    """
    names: dict[str, str] = {}
    winner_name: str | None = None
    tie = False
    events: list[list[str]] = []
    skip: str | None = None
    turns = 0
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        kind = parts[1]
        if kind == "player" and len(parts) > 3:
            names[parts[3].strip().lower()] = parts[2]
            continue
        if kind == "win" and len(parts) > 2:
            winner_name = parts[2].strip().lower()
            continue
        if kind == "tie":
            tie = True
            continue
        if kind.startswith("-"):
            if kind in _DROP_DASH:
                continue
            if kind == "-transform":
                skip = skip or "transform"
            if kind == "-ability" and any(
                ("[from] ability: " + a) in part.lower()
                for part in parts[4:]
                for a in ("trace", "mummy", "lingering aroma", "wandering spirit", "receiver",
                          "power of alchemy", "neutralizing gas")
            ):
                skip = skip or "ability_swap"
            if kind == "-ability" and any(
                m in part.lower() for part in parts[4:]
                for m in ("[from] move: skill swap", "[from] move: role play",
                          "[from] move: worry seed", "[from] move: simple beam",
                          "[from] move: entrainment", "[from] move: doodle")
            ):
                skip = skip or "ability_swap"
            parts = parts[:3] + [_restore_case(x) for x in parts[3:]]
            if kind in _SIDE_EVENTS and len(parts) > 2 and _SIDE_ONLY.match(parts[2]):
                parts[2] = parts[2][:2] + ": side"
            events.append(parts)
            continue
        if kind not in _KEEP:
            continue
        parts = parts[:4] + [_restore_case(x) for x in parts[4:]]
        if kind == "replace":
            skip = skip or "illusion"
        if kind in ("switch", "drag", "replace", "detailschange") and len(parts) > 3:
            parts[3] = _fix_details(parts[3])
        if kind == "turn" and len(parts) > 2 and parts[2].isdigit():
            turns = int(parts[2])
        events.append(parts)
    winner: str | None = None
    if tie:
        winner = "tie"
    elif winner_name is not None:
        winner = names.get(winner_name)
    if not any(e[1] == "turn" for e in events):
        skip = skip or "no_turns"
    if winner is None:
        skip = skip or "no_result"
    return Log(events, winner, skip, turns)


# --- actions --------------------------------------------------------------------------


@dataclass
class Action:
    kind: str  # "move" | "switch"
    move_id: str = ""
    tera: bool = False
    ident: str = ""  # switch target's ident name (lowercase, as in the log)
    species_id: str = ""  # switch target's species id (from the switch details)


def _side_of(ref: str) -> str:
    return ref[:2]


def _name_of(ref: str) -> str:
    return ref.split(": ", 1)[1] if ": " in ref else ref


def extract_actions(events: list[list[str]]) -> dict[int, dict[str, Action | None]]:
    """The action each side took at each turn start (``None`` when it is not a free choice).

    A turn's events run from ``|turn|N`` to the next ``|turn|``. A side's choice is its first
    plain ``switch`` or its first ``move`` not tagged ``[from]`` (locked moves, called moves,
    Dancer, ...) before ``|upkeep|`` (switches after ``|upkeep|`` are forced replacements
    after a faint). ``|cant|`` (sleep, flinch, recharge, ...) hides the choice: ``None``.
    ``|-terastallize|`` before the move adds the Tera flag.
    """
    out: dict[int, dict[str, Action | None]] = {}
    turn: int | None = None
    acted: set[str] = set()
    tera: set[str] = set()
    for event in events:
        kind = event[1]
        if kind == "turn":
            turn = int(event[2])
            out[turn] = {"p1": None, "p2": None}
            acted, tera = set(), set()
            continue
        if turn is None or "upkeep" in acted:
            continue
        if kind == "upkeep":
            acted.add("upkeep")
            continue
        if kind == "-terastallize" and len(event) > 2:
            tera.add(_side_of(event[2]))
            continue
        if len(event) < 3 or ":" not in event[2]:
            continue
        side = _side_of(event[2])
        if side not in ("p1", "p2") or side in acted:
            continue
        has_from = any(p.startswith("[from]") for p in event[4:])
        if kind == "switch" and not has_from:
            details = event[3].split(",")[0]
            out[turn][side] = Action(
                "switch", ident=_name_of(event[2]), species_id=to_id(details)
            )
            acted.add(side)
        elif kind == "move" and not has_from:
            move_id = to_id(event[3])
            acted.add(side)
            if move_id in ("struggle", "recharge"):
                continue
            out[turn][side] = Action("move", move_id=move_id, tera=side in tera)
        elif kind == "move":
            acted.add(side)  # locked / called move: not a free choice
        elif kind == "cant":
            acted.add(side)
    return out


# --- teams ---------------------------------------------------------------------------


@dataclass
class HMon:
    """One member of a hindsight team."""

    ident: str | None  # the log's ident name, None for an unrevealed Pokemon
    details: str  # "ursaluna, L79, M"-style details (display species + level)
    species_id: str
    sampled: SampledSet
    revealed: bool


class SkipGame(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _display_name(species_id: str) -> str:
    entry = es.dex_entry(species_id)
    return str(entry["name"]) if entry else species_id


def _first_switches(events: list[list[str]], role: str) -> dict[str, tuple[str, str]]:
    """ident name -> (details, hp_status) of each of ``role``'s Pokemon at its first switch-in."""
    seen: dict[str, tuple[str, str]] = {}
    for event in events:
        if event[1] in ("switch", "drag") and event[2].startswith(role):
            seen.setdefault(_name_of(event[2]), (event[3], event[4] if len(event) > 4 else ""))
    return seen


def _details_level(details: str) -> int:
    match = re.search(r", L(\d+)", details)
    return int(match.group(1)) if match else 100


def _request_mon(role: str, hmon: HMon, active: bool = False) -> dict[str, Any]:
    sampled = hmon.sampled
    entry = es.dex_entry(hmon.species_id)
    if entry is None:
        raise SkipGame("unknown_species")
    level = int(sampled.level)
    moves = [to_id(m) for m in sampled.moves][:4]
    ability = to_id(sampled.ability)
    item = to_id(sampled.item) if sampled.item else ""
    evs, ivs = es.randbats_evs_ivs(hmon.species_id, level, moves, ability, item)
    stats = es.compute_stats(entry["baseStats"], level, evs, ivs)
    hp = stats.pop("hp")
    name = hmon.ident or str(entry["name"]).lower()
    tera = str(sampled.tera_type or "").capitalize() or "Normal"
    return {
        "ident": f"{role}: {name}",
        "details": hmon.details,
        "condition": f"{hp}/{hp}",
        "active": active,
        "stats": stats,
        "moves": moves,
        "baseAbility": ability,
        "item": item,
        "pokeball": "pokeball",
        "ability": ability,
        "commanding": False,
        "reviving": False,
        "teraType": tera,
        "terastallized": "",
    }


def synthetic_request(role: str, team: list[HMon]) -> dict[str, Any]:
    return {
        "side": {"name": role, "id": role, "pokemon": [_request_mon(role, m) for m in team]},
        "rqid": 1,
    }


def _team_from_slots(
    slots: list[es.OpponentSlot], mon_ident: dict[int, str], first: dict[str, tuple[str, str]],
) -> list[HMon]:
    team = []
    for slot in slots:
        ident = mon_ident.get(id(slot.mon)) if slot.mon is not None else None
        if ident is not None and ident in first:
            details = first[ident][0]
            level = _details_level(details)
            sampled = dataclasses.replace(slot.sampled, level=level)
        else:
            ident = None
            level = int(slot.sampled.level)
            details = f"{_display_name(slot.species_id)}, L{level}"
            sampled = slot.sampled
        team.append(HMon(ident, details, slot.species_id, sampled, ident is not None))
    return team


def _placeholder_team(role: str, first: dict[str, tuple[str, str]]) -> list[HMon]:
    """A team with the real species and levels but placeholder sets (pass one's own side)."""
    team: list[HMon] = []
    for ident, (details, _hp) in first.items():
        species_id = to_id(details.split(",")[0])
        entry = es.dex_entry(species_id)
        if entry is None:
            raise SkipGame("unknown_species")
        ability = to_id(next(iter(entry["abilities"].values())))
        sampled = SampledSet(
            species_id, _details_level(details), "", ("tackle", "protect", "rest", "substitute"),
            ability, "leftovers", "normal",
        )
        team.append(HMon(ident, details, species_id, sampled, True))
    fillers = iter(_FILLERS)
    while len(team) < es.TEAM_SIZE:
        species_id = next(fillers)
        if any(m.species_id == species_id for m in team):
            continue
        ability = to_id(next(iter(es.dex_entry(species_id)["abilities"].values())))
        sampled = SampledSet(species_id, 80, "", ("tackle", "protect", "rest", "substitute"),
                             ability, "leftovers", "normal")
        team.append(HMon(None, f"{_display_name(species_id)}, L80", species_id, sampled, False))
    return team[: es.TEAM_SIZE]


_FILLERS = ("blissey", "chansey", "snorlax", "toxapex", "corviknight", "clefable", "slowbro",
            "skarmory")
def _new_battle(tag: str, role: str, team: list[HMon]) -> Battle:
    battle = Battle(tag, username="hindsight", logger=None, gen=9)
    battle.player_role = role
    battle.parse_request(synthetic_request(role, team))
    return battle


_seq = [0]


def _tag(prefix: str) -> str:
    _seq[0] += 1
    return f"battle-{FORMAT}-{prefix}{_seq[0]}"


def hindsight_team(
    events: list[list[str]], role: str, other_role: str, data: RandbatsData,
    sampler: es.TeamSampler, rng: random.Random, model: OpponentModel,
) -> list[HMon]:
    """The complete team of ``other_role``, from replaying the whole log as ``role``.

    ``role``'s own side is a placeholder (real species and levels, dummy sets): that is enough
    for the opponent model's event-based reveals and speed evidence but not for damage rolls, so
    damage-roll inference is off (the model is built with ``damage=False``).
    """
    first_own = _first_switches(events, role)
    first_other = _first_switches(events, other_role)
    if not first_other or not first_own:
        raise SkipGame("no_switches")
    battle = _new_battle(_tag("h"), role, _placeholder_team(role, first_own))
    tag = battle.battle_tag
    try:
        for event in events:
            model.observe_events(tag, [event], battle)
            battle.parse_message(event)
        trackers = model.update(battle)
        slots = es.sample_opponent_team(battle, trackers, sampler, rng)
        mon_ident = {id(mon): str(ident)[4:] for ident, mon in battle.opponent_team.items()}
    finally:
        model.forget(tag)
    team = _team_from_slots(slots, mon_ident, first_other)
    if sum(m.revealed for m in team) != len(first_other) or len(team) != es.TEAM_SIZE:
        raise SkipGame("unmapped_species")  # e.g. a CAP / non-randbats species
    return team


# --- states ---------------------------------------------------------------------------


@dataclass
class TurnState:
    turn: int
    state: str
    legal: list[str]
    chosen: str | None
    choice_ok: bool | None  # chosen action is among the engine-derived legal options


def legal_choices(built: es.BuiltState) -> list[str]:
    """Move / Tera-move / switch options of side one, derived from the engine state."""
    side = built.state.side_one
    active = side.pokemon[int(side.active_index)]
    out: list[str] = []
    can_tera = (
        not any(bool(p.terastallized) for p in side.pokemon)
        and str(active.tera_type) not in ("typeless", "none", "")
        and not bool(active.terastallized)
    )
    usable = [m for m in active.moves if int(m.pp) > 0 and not m.disabled and m.id != "none"]
    for move in usable:
        out.append(to_id(move.id))
        if can_tera:
            out.append(to_id(move.id) + "-tera")
    for index, mon in enumerate(side.pokemon):
        if index != int(side.active_index) and int(mon.hp) > 0 and mon.id != "none":
            out.append("switch " + to_id(mon.id))
    return out


def _choice_string(action: Action | None, built: es.BuiltState, battle: Battle, role: str) -> str | None:
    if action is None:
        return None
    if action.kind == "move":
        return action.move_id + ("-tera" if action.tera else "")
    mon = battle.team.get(f"{role}: {action.ident}")
    for index, ours in enumerate(built.our_mons):
        if ours is mon:
            return "switch " + to_id(built.state.side_one.pokemon[index].id)
    return None


@dataclass
class FirstPerson:
    """Live-style opponent sampling for ``states_for_role`` (the hindsight-free mode).

    ``model`` observes the log event by event, so at ``|turn|N`` its set trackers only know what
    was revealed up to then; ``sampler``/``rng`` fill the rest exactly like ``SearchPlayer``.
    """

    model: OpponentModel
    sampler: es.TeamSampler
    rng: random.Random


def states_for_role(
    events: list[list[str]], role: str, own: list[HMon], opp: list[HMon] | None,
    actions: dict[int, dict[str, Action | None]], stats: Counter,
    fp: FirstPerson | None = None,
) -> list[TurnState]:
    """Replay the log as ``role`` (own = hindsight team) and build the state at every turn.

    By default the opponent side is the shared hindsight team ``opp``. With ``fp`` it is instead
    sampled per turn from only what ``role`` had seen by that turn (``opp`` is then unused).
    """
    other = "p2" if role == "p1" else "p1"
    battle = _new_battle(_tag("s"), role, own)
    tag = battle.battle_tag
    sleep_module._TRACKERS.pop(tag, None)
    first_own = _first_switches(events, role)
    out: list[TurnState] = []
    request_by_ident = {m.ident: _request_mon(role, m) for m in own if m.ident}
    for ident, (details, hp_status) in first_own.items():
        req = request_by_ident.get(ident)
        if req is None:
            continue
        denominator = hp_status.split("/")[1].split(" ")[0] if "/" in hp_status else ""
        if denominator.isdigit():
            if denominator == "100" and not req["condition"].endswith("/100"):
                raise SkipGame("percent_hp")
            stats["hp_checked"] += 1
            if req["condition"].split("/")[1] != denominator:
                stats["hp_mismatch"] += 1
    try:
        for event in events:
            sleep_module.observe_events(tag, [event])
            if fp is not None:
                fp.model.observe_events(tag, [event], battle)
            battle.parse_message(event)
            if event[1] != "turn":
                continue
            turn = int(event[2])
            if fp is None:
                assert opp is not None
                built = _build(battle, other, opp)
            else:
                _lock_active_moves(battle)
                trackers = fp.model.update(battle)
                slots = es.sample_opponent_team(battle, trackers, fp.sampler, fp.rng)
                built = es.build_state(battle, slots)
            chosen = _choice_string(actions.get(turn, {}).get(role), built, battle, role)
            legal = legal_choices(built)
            out.append(TurnState(
                turn, built.to_string(), legal, chosen,
                None if chosen is None else chosen in legal,
            ))
    finally:
        sleep_module._TRACKERS.pop(tag, None)
        if fp is not None:
            fp.model.forget(tag)
    return out


def _lock_active_moves(battle: Battle) -> None:
    """Synthesise the request's ``active`` block (PP, Choice lock) for our active Pokemon."""
    mon = battle.active_pokemon
    if mon is None:
        return
    move_ids = list(mon.moves)
    item = to_id(mon.item) if mon.item and mon.item != "unknown_item" else ""
    locked = es._locked_move(mon, item, move_ids)
    battle._last_request = {
        "side": (battle.last_request or {}).get("side", {}),
        "active": [{
            "moves": [
                {"id": mid, "move": mid, "pp": mon.moves[mid].current_pp,
                 "disabled": locked is not None and mid != locked}
                for mid in move_ids
            ]
        }],
    }


def _build(battle: Battle, other: str, opp: list[HMon]) -> es.BuiltState:
    _lock_active_moves(battle)
    active = battle.opponent_active_pokemon
    slots = []
    for hmon in opp:
        mon = battle.opponent_team.get(f"{other}: {hmon.ident}") if hmon.ident else None
        slots.append(es.OpponentSlot(hmon.species_id, hmon.sampled, mon))
    slots.sort(key=lambda s: 0 if (s.mon is not None and s.mon is active) else 1)
    return es.build_state(battle, slots)


# --- one game ------------------------------------------------------------------------


@dataclass
class GameOutput:
    battle_id: str
    meta: dict[str, Any]
    result: dict[str, str]  # role -> win/loss/tie
    turns: int
    states: dict[str, list[TurnState]]


def result_for(winner: str, role: str) -> str:
    if winner == "tie":
        return recorder.RESULT_TIE
    return recorder.RESULT_WIN if winner == role else recorder.RESULT_LOSS


class Converter:
    """Holds the per-process randbats data / samplers; ``convert`` turns one log into states."""

    def __init__(self, data: RandbatsData | None = None, *, first_person: bool = False,
                 fp_damage: bool = False):
        if data is None:
            from showdown_bot.players.search import default_randbats_data

            data = default_randbats_data()
        self.data = data
        self.sampler = es.TeamSampler(data)
        self.model = OpponentModel(data, damage=False)
        self.first_person = first_person
        # first-person pass: its own model (trackers are per battle tag, so no clash); damage-roll
        # inference needs the actor's real stats, which the hindsight own team approximates
        self.fp_model = OpponentModel(data, damage=fp_damage) if first_person else None

    def convert(self, text: str, battle_id: str, seed: int = 0) -> GameOutput:
        log = sanitize_log(text)
        if log.skip:
            raise SkipGame(log.skip)
        digest = hashlib.sha256(f"{seed}:{battle_id}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        events = log.events
        teams: dict[str, list[HMon]] = {}
        for role, other in (("p1", "p2"), ("p2", "p1")):
            # the replay as ``role`` reveals ``other``'s team
            teams[other] = hindsight_team(
                events, role, other, self.data, self.sampler, rng, self.model
            )
        actions = extract_actions(events)
        stats: Counter = Counter()
        states = {}
        for role in ("p1", "p2"):
            fp = None
            if self.first_person:
                assert self.fp_model is not None
                fp = FirstPerson(self.fp_model, self.sampler, random.Random(rng.getrandbits(64)))
            states[role] = states_for_role(
                events, role, teams[role], teams["p2" if role == "p1" else "p1"], actions, stats, fp
            )
        assert log.winner is not None
        return GameOutput(
            battle_id,
            {"hp_checked": stats["hp_checked"], "hp_mismatch": stats["hp_mismatch"]},
            {r: result_for(log.winner, r) for r in ("p1", "p2")},
            log.turns,
            states,
        )


# --- writing --------------------------------------------------------------------------


def write_game(
    rec: recorder.GameRecorder, out: GameOutput, header: dict[str, Any]
) -> list[dict[str, Any]]:
    """Write both sides' value files; return the policy rows."""
    policy_rows: list[dict[str, Any]] = []
    for role, turns in out.states.items():
        if not turns:
            continue
        rec.start_game(
            out.battle_id, role, role=role, battle_format=FORMAT,
            config={}, player="human", extra={"source": "human", **header},
        )
        for ts in turns:
            rec.add_decision(out.battle_id, role, {
                "turn": ts.turn, "side": "one", "worlds": [], "policy": None,
                "chosen": ts.chosen, "order": None, "compute": {},
                "truth": {"state": ts.state}, "truth_status": "ok", "legal": ts.legal,
            })
            if ts.chosen is not None:
                policy_rows.append({
                    "battle_tag": out.battle_id, "role": role, "turn": ts.turn,
                    **header, "result": out.result[role],
                    "state": ts.state, "action": ts.chosen, "legal": ts.legal,
                    "action_legal": ts.choice_ok,
                })
        rec.end_game(out.battle_id, role, out.result[role], out.turns)
    return policy_rows


# --- batch driver (process pool worker) -----------------------------------------------

_CONVERTER: list[Converter] = []


_FP_OPTIONS: dict[str, bool] = {"first_person": False, "fp_damage": False}


def set_worker_options(first_person: bool = False, fp_damage: bool = False) -> None:
    """Worker-pool initializer: the conversion mode of every Converter this process builds."""
    _FP_OPTIONS.update(first_person=first_person, fp_damage=fp_damage)
    _init_worker()


def _init_worker() -> None:
    _CONVERTER.append(Converter(**_FP_OPTIONS))


def process_chunk(job: dict[str, Any]) -> dict[str, Any]:
    """Convert a list of games and write shard ``job['shard_dir']`` + its policy file."""
    if not _CONVERTER:
        _init_worker()
    conv = _CONVERTER[0]
    shard_dir = Path(job["shard_dir"])
    rec = recorder.GameRecorder(shard_dir, truth=True)
    policy_path = Path(job["policy_path"]) if job.get("policy_path") else None
    skips: Counter = Counter()
    examples: dict[str, list[str]] = {}
    ok = decisions = actions = illegal = hp_checked = hp_mismatch = 0
    rows_all: list[dict[str, Any]] = []
    started = time.perf_counter()
    for game in job["games"]:
        header = {"elo": game["elo"], "month_year": game["month_year"], "split": game["split"]}
        try:
            out = conv.convert(game["text"], game["battle_id"], job["seed"])
            rows = write_game(rec, out, header)
        except SkipGame as exc:
            reason = exc.reason
            skips[reason] += 1
            examples.setdefault(reason, []).append(game["battle_id"])
            continue
        except Exception as exc:  # noqa: BLE001 - one bad log must not kill the run
            reason = f"error:{type(exc).__name__}"
            skips[reason] += 1
            examples.setdefault(reason, []).append(f"{game['battle_id']} {str(exc)[:120]}")
            continue
        ok += 1
        decisions += sum(len(t) for t in out.states.values())
        actions += len(rows)
        illegal += sum(1 for r in rows if r["action_legal"] is False)
        hp_checked += out.meta["hp_checked"]
        hp_mismatch += out.meta["hp_mismatch"]
        rows_all.extend(rows)
    if policy_path is not None and rows_all:
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = policy_path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in rows_all:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        tmp.replace(policy_path)
    result = {
        "games": len(job["games"]), "ok": ok, "skips": dict(skips),
        "examples": {k: v[:5] for k, v in examples.items()},
        "decisions": decisions, "actions": actions, "illegal_actions": illegal,
        "hp_checked": hp_checked, "hp_mismatch": hp_mismatch,
        "seconds": time.perf_counter() - started,
    }
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / ".done").write_text(json.dumps(result))  # --resume marker
    return result
