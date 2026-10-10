"""Build real poke-env Battles (no server) for engine-state and search tests."""

from __future__ import annotations

import json
from typing import Any

from poke_env.battle.battle import Battle
from poke_env.data import GenData

from showdown_bot import engine_state as es
from showdown_bot.sets import to_id

# (species display name, level, item, ability, moves, tera type)
OUR_TEAM = [
    ("Great Tusk", 80, "Leftovers", "Protosynthesis",
     ["Close Combat", "Headlong Rush", "Knock Off", "Rapid Spin"], "Ghost"),
    ("Kingambit", 78, "Black Glasses", "Supreme Overlord",
     ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Swords Dance"], "Flying"),
    ("Gholdengo", 80, "Choice Scarf", "Good as Gold",
     ["Make It Rain", "Shadow Ball", "Trick", "Nasty Plot"], "Water"),
    ("Dragapult", 80, "Choice Specs", "Infiltrator",
     ["Draco Meteor", "Shadow Ball", "Flamethrower", "U-turn"], "Ghost"),
    ("Corviknight", 82, "Rocky Helmet", "Pressure",
     ["Brave Bird", "Body Press", "Roost", "U-turn"], "Dragon"),
    ("Heatran", 80, "Leftovers", "Flash Fire",
     ["Magma Storm", "Earth Power", "Stealth Rock", "Taunt"], "Grass"),
]


def own_stats(species: str, level: int) -> dict[str, int]:
    base = GenData.from_gen(9).pokedex[to_id(species)]["baseStats"]
    return es.compute_stats(base, level)


def request_for(
    team=OUR_TEAM,
    *,
    active: int = 0,
    force_switch: bool = False,
    hp: dict[int, tuple[int, str]] | None = None,
    can_tera: bool = True,
    terastallized: dict[int, str] | None = None,
    disabled: set[str] | None = None,
) -> dict[str, Any]:
    hp = hp or {}
    terastallized = terastallized or {}
    disabled = disabled or set()
    side_pokemon = []
    for index, (species, level, item, ability, moves, tera) in enumerate(team):
        stats = own_stats(species, level)
        max_hp = stats.pop("hp")
        current, status = hp.get(index, (max_hp, ""))
        condition = f"{current}/{max_hp}" if current else "0 fnt"
        if status and current:
            condition += f" {status}"
        side_pokemon.append(
            {
                "ident": f"p1: {species}",
                "details": f"{species}, L{level}",
                "condition": condition,
                "active": index == active,
                "stats": stats,
                "moves": [to_id(m) for m in moves],
                "baseAbility": to_id(ability),
                "item": to_id(item),
                "pokeball": "pokeball",
                "ability": to_id(ability),
                "commanding": False,
                "reviving": False,
                "teraType": tera,
                "terastallized": terastallized.get(index, ""),
            }
        )
    request: dict[str, Any] = {"side": {"name": "bot", "id": "p1", "pokemon": side_pokemon}, "rqid": 5}
    if force_switch:
        request["forceSwitch"] = [True]
    else:
        active_moves = [
            {
                "move": m,
                "id": to_id(m),
                "pp": 16,
                "maxpp": 16,
                "target": "normal",
                "disabled": to_id(m) in disabled,
            }
            for m in team[active][4]
        ]
        entry: dict[str, Any] = {"moves": active_moves}
        if can_tera:
            entry["canTerastallize"] = team[active][5]
        request["active"] = [entry]
    return request


def make_battle(request: dict[str, Any] | None = None, tag: str = "battle-gen9randombattle-1") -> Battle:
    battle = Battle(tag, username="bot", logger=None, gen=9)
    battle.player_role = "p1"
    battle.parse_request(request or request_for())
    return battle


def event(battle: Battle, *parts: str) -> None:
    battle.parse_message(["", *parts])


def opp_switch(battle: Battle, nickname: str, details: str, hp: str = "100/100") -> None:
    event(battle, "switch", f"p2a: {nickname}", details, hp)
