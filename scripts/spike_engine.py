"""Small Gen 9 poke-engine state-construction and throughput spike."""

from __future__ import annotations

import time

from poke_engine import (
    Move,
    Pokemon,
    Side,
    State,
    calculate_damage,
    monte_carlo_tree_search,
)


# Level 100 opening state: representative Gen 9 OU-style teams, all at full HP.
# Stats are plausible finished battle stats; exact sets are only for exercising
# the API and are not intended to be a calibrated competitive evaluation.
TEAM_ONE = (
    ("greattusk", ("ground", "fighting"), "protosynthesis", "heavydutyboots", "ghost", (434, 361, 354, 144, 142, 300), ("earthquake", "closecombat", "knockoff", "rapidspin"), "jolly"),
    ("kingambit", ("dark", "steel"), "supremeoverlord", "blackglasses", "flying", (404, 405, 296, 140, 226, 136), ("kowtowcleave", "suckerpunch", "ironhead", "swordsdance"), "adamant"),
    ("gholdengo", ("steel", "ghost"), "goodasgold", "leftovers", "water", (315, 157, 237, 369, 217, 293), ("makeitrain", "shadowball", "recover", "nastyplot"), "timid"),
    ("dragapult", ("dragon", "ghost"), "infiltrator", "choicespecs", "ghost", (317, 212, 186, 327, 186, 421), ("dracometeor", "shadowball", "flamethrower", "uturn"), "timid"),
    ("corviknight", ("flying", "steel"), "pressure", "rockyhelmet", "dragon", (399, 228, 339, 176, 217, 170), ("bravebird", "bodypress", "roost", "uturn"), "impish"),
    ("heatran", ("fire", "steel"), "flashfire", "leftovers", "grass", (385, 204, 248, 359, 248, 253), ("magmastorm", "earthpower", "stealthrock", "taunt"), "timid"),
)

TEAM_TWO = (
    ("gliscor", ("ground", "flying"), "poisonheal", "toxicorb", "water", (354, 226, 314, 128, 207, 317), ("earthquake", "knockoff", "stealthrock", "protect"), "impish"),
    ("roaringmoon", ("dark", "dragon"), "protosynthesis", "boosterenergy", "flying", (351, 405, 216, 140, 197, 343), ("acrobatics", "knockoff", "earthquake", "dragondance"), "jolly"),
    ("ironvaliant", ("fairy", "fighting"), "quarkdrive", "boosterenergy", "fairy", (303, 359, 196, 359, 196, 364), ("moonblast", "closecombat", "thunderbolt", "encore"), "naive"),
    ("primarina", ("water", "fairy"), "torrent", "assaultvest", "steel", (341, 157, 216, 361, 350, 220), ("moonblast", "surf", "psychicnoise", "flipturn"), "modest"),
    ("ragingbolt", ("electric", "dragon"), "protosynthesis", "boosterenergy", "fairy", (434, 176, 226, 405, 246, 199), ("dracometeor", "thunderclap", "thunderbolt", "calmmind"), "modest"),
    ("skeledirge", ("fire", "ghost"), "unaware", "heavydutyboots", "fairy", (388, 158, 264, 303, 217, 149), ("torchsong", "slackoff", "willowisp", "hex"), "bold"),
)


def make_side(roster: tuple[tuple, ...]) -> Side:
    pokemon = []
    for (
        name,
        types,
        ability,
        item,
        tera_type,
        stats,
        moves,
        nature,
    ) in roster:
        hp, attack, defense, special_attack, special_defense, speed = stats
        pokemon.append(
            Pokemon(
                id=name,
                level=100,
                types=types,
                base_types=types,
                hp=hp,
                maxhp=hp,
                ability=ability,
                base_ability=ability,
                item=item,
                nature=nature,
                attack=attack,
                defense=defense,
                special_attack=special_attack,
                special_defense=special_defense,
                speed=speed,
                status="none",
                tera_type=tera_type,
                moves=[Move(id=move, pp=16) for move in moves],
            )
        )
    return Side(pokemon=pokemon)


def make_state() -> State:
    return State(side_one=make_side(TEAM_ONE), side_two=make_side(TEAM_TWO))


def main() -> None:
    state = make_state()
    side_one_rolls, side_two_rolls = calculate_damage(
        state, "closecombat", "earthquake", side_one_moves_first=True
    )
    print(
        "Max damage [normal, crit] from independent state copies "
        "(Great Tusk Close Combat vs Gliscor Earthquake): "
        f"side 1={side_one_rolls}, side 2={side_two_rolls}"
    )

    for seconds in (1, 2, 5):
        started = time.perf_counter()
        result = monte_carlo_tree_search(
            state,
            duration_ms=seconds * 1000,
            iterations=0,
            threads=1,
        )
        elapsed = time.perf_counter() - started
        iterations_per_second = result.total_visits / elapsed if elapsed else 0.0
        top_moves = sorted(
            result.side_one, key=lambda move: move.visits, reverse=True
        )[:3]
        summary = ", ".join(
            f"{move.move_choice} ({move.visits})" for move in top_moves
        )
        print(
            f"requested={seconds}s elapsed={elapsed:.3f}s "
            f"iterations={result.total_visits} "
            f"iterations/sec={iterations_per_second:.1f} "
            f"top3=[{summary}]"
        )


if __name__ == "__main__":
    main()
