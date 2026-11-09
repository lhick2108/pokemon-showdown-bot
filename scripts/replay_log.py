"""Replay a recorded ladder battle log through poke-env and show the engine state / search.

    python scripts/replay_log.py LOG --turn 11 [--time-ms 300] [--worlds 4]
        [--moves "Mesprit=Knock Off,Psychic Noise,Thunder Wave,Encore"] [--item Mesprit=Leftovers]

The log holds only the public protocol, so our own team (the ``|request|`` the bot saw) is
reconstructed from the species that appear on our side, with placeholder sets unless
``--moves`` / ``--item`` fix the ones that matter.

Ladder logs written by scripts/ladder.py also contain the ``|request|{json}`` lines the
server sent us (our real moves/PP/item/ability/tera/stats/HP): each is fed to poke-env at
its original position, so our side is exact and --moves/--item are ignored.

    python scripts/replay_log.py LOG --turn 11 --from-record RECORDS [--research --time-ms 300]

prints what the bot saw and decided live (records/*.jsonl.gz; no re-search unless
``--research``). The state is built the way
``SearchPlayer`` builds it (opponent worlds sampled from the tracker) at the START of the
requested turn (right after ``|turn|N``) and searched with the engine.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from battle_helpers import request_for  # noqa: E402
from poke_env.battle.battle import Battle  # noqa: E402

from showdown_bot import engine_state as es  # noqa: E402
from showdown_bot import sleep as sleep_module  # noqa: E402
from showdown_bot.players import _worker  # noqa: E402
from showdown_bot.players.search import aggregate_policy, default_randbats_data  # noqa: E402
from showdown_bot.sets import OpponentModel  # noqa: E402

DEFAULT_MOVES = ["Tackle", "Protect", "Rest", "Substitute"]
SKIP = {"inactive", "inactiveoff", "request", "title", "j", "l", "c", "raw", ""}


def our_role(lines: list[str], username: str = "porygonzplaya") -> str:
    """Our side ("p1"/"p2") from the ``|player|`` lines; p1 if the log doesn't say."""
    for line in lines:
        parts = line.split("|")
        if len(parts) > 3 and parts[1] == "player" and parts[3].strip().lower() == username:
            return parts[2]
    return "p1"


def side_request(team, active: int, role: str) -> dict:
    request = request_for(team, active=active)
    request["side"]["id"] = role
    for mon in request["side"]["pokemon"]:
        mon["ident"] = role + mon["ident"][2:]
    return request


def our_species(lines: list[str], role: str = "p1") -> list[tuple[str, int]]:
    seen: dict[str, int] = {}
    for line in lines:
        match = re.match(rf"\|switch\|{role}a: [^|]+\|([^,|]+), L(\d+)", line)
        if match:
            seen.setdefault(match.group(1), int(match.group(2)))
    return list(seen.items())


def build_team(species, moves, items):
    team = [
        (name, level, items.get(name, "Leftovers"), "Pressure", moves.get(name, DEFAULT_MOVES), "Normal")
        for name, level in species
    ]
    while len(team) < 6:
        team.append(("Blissey", 80, "Leftovers", "Natural Cure", DEFAULT_MOVES, "Normal"))
    return team


def has_requests(lines: list[str]) -> bool:
    return any(line.startswith("|request|{") for line in lines)


def replay(lines: list[str], turn: int, team=None, model: OpponentModel | None = None) -> Battle:
    """Feed the log up to the start of ``turn``; ``model.update`` runs at every turn start
    (the bot updates its opponent model at each decision, and the trackers are incremental)."""
    battle = Battle("battle-gen9randombattle-replay", username="porygonzplaya", logger=None, gen=9)
    role = our_role(lines)
    battle.player_role = role
    sleep_module._TRACKERS.pop(battle.battle_tag, None)
    real = has_requests(lines)
    if not real:
        names = [t[0] for t in team]
        battle.parse_request(side_request(team, 0, role))
    at_turn = False  # reached |turn|N: the server's request for N follows it in the log
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if parts[1] == "request":
            if real and len(parts) > 2 and parts[2].strip():
                battle.parse_request(json.loads("|".join(parts[2:])))
            if at_turn and real:
                break
            continue
        if at_turn:
            break
        if parts[1] in SKIP:
            continue
        if not real and parts[1] == "switch" and parts[2].startswith(role + "a"):
            battle.parse_request(side_request(team, names.index(parts[3].split(",")[0]), role))
        if model is not None:
            model.observe_events(battle.battle_tag, [parts], battle)
        sleep_module.observe_events(battle.battle_tag, [parts])  # as OpponentModelPlayerMixin does
        battle.parse_message(parts)
        if parts[1] == "turn":
            if model is not None:
                model.update(battle)
            if int(parts[2]) == turn:
                if not real:
                    break
                at_turn = True
    return battle


def describe(built: es.BuiltState, bench: bool = False) -> None:
    describe_state(built.state, bench)


def find_decision(records: str, turn: int, battle_tag: str | None = None):
    """The recorded decision for ``turn`` from a records file or directory."""
    from showdown_bot import recorder

    for path in recorder.iter_game_files(records):
        if battle_tag and battle_tag.lower() not in Path(path).name.lower():
            continue
        game = recorder.read_game(path)
        for decision in game["decisions"]:
            if decision.get("turn") == turn:
                return path, game, decision
    return None


def show_record(records, turn, battle_tag, research, time_ms, evaluator, scale) -> None:
    from poke_engine import State

    found = find_decision(records, turn, battle_tag)
    if found is None:
        raise SystemExit(f"no recorded decision for turn {turn} in {records}")
    path, game, decision = found
    print(f"record {Path(path).name} turn {decision['turn']} "
          f"(worlds searched: {len(decision['worlds'])})")
    for index, world in enumerate(decision["worlds"]):
        print(f"--- world {index}: visits={world['total_visits']} avg_score={world['avg_score']:.3f}")
        describe_state(State.from_string(world["state"]), bench=index == 0)
        options = [(m["choice"], m["visits"], round(m["score"] / max(m["visits"], 1), 3))
                   for m in world["moves"]]
        print("      options (choice, visits, mean score):", options)
    policy = decision["policy"]
    print("policy:", {k: round(v, 3) for k, v in sorted(policy.items(), key=lambda kv: -kv[1])})
    print(f"chosen: {decision['chosen']}  order: {decision.get('order')}")
    if research:
        results = [_worker.search_world(w["state"], time_ms, 1, evaluator, scale)
                   for w in decision["worlds"]]
        new = aggregate_policy(results)
        print(f"re-search ({time_ms} ms/world):",
              {k: round(v, 3) for k, v in sorted(new.items(), key=lambda kv: -kv[1])})


def describe_state(state, bench: bool = False) -> None:
    for tag, side in (("US", state.side_one), ("OPP", state.side_two)):
        mon = side.pokemon[int(side.active_index)]
        print(f"[{tag}] {mon.id} hp={mon.hp}/{mon.maxhp} status={mon.status} "
              f"sleep_turns={mon.sleep_turns} rest_turns={mon.rest_turns} "
              f"item={mon.item} ability={mon.ability} tera={getattr(mon, 'terastallized', '')}"
              f"/{getattr(mon, 'tera_type', '')}")
        print(f"      moves={[(m.id, m.pp, m.disabled) for m in mon.moves]}")
        print(f"      volatiles={sorted(side.volatile_statuses)} last_used={side.last_used_move} "
              f"durations={side.volatile_status_durations}")
        if bench and tag == "US":
            for i, other in enumerate(side.pokemon):
                if i != int(side.active_index):
                    print(f"      bench {other.id} hp={other.hp}/{other.maxhp} item={other.item} "
                          f"ability={other.ability} moves={[(m.id, m.pp) for m in other.moves]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("--turn", type=int, required=True)
    parser.add_argument("--from-record", default=None, metavar="RECORDS",
                        help="records .jsonl.gz file or dir: print what the bot saw/decided live")
    parser.add_argument("--research", action="store_true",
                        help="with --from-record: search the recorded worlds again")
    parser.add_argument("--time-ms", type=int, default=300)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--moves", action="append", default=[])
    parser.add_argument("--item", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluator", default=None, help="learned leaf evaluator (.pev); default stock")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--instr-world",type=int, default=0)
    parser.add_argument("--instr",action="append", default=[], metavar="MINE,THEIRS",
                        help="print the engine's instruction outcomes for this joint choice")
    args = parser.parse_args()
    if args.from_record:
        show_record(args.from_record, args.turn, Path(args.log).stem,
                    args.research, args.time_ms, args.evaluator, args.scale)
        return
    lines = Path(args.log).read_text(encoding="utf-8").splitlines()
    moves = {k: v.split(",") for k, v in (m.split("=", 1) for m in args.moves)}
    items = dict(i.split("=", 1) for i in args.item)
    team = None if has_requests(lines) else build_team(our_species(lines, our_role(lines)), moves, items)
    print("our side:", "real |request| sets" if team is None else "synthesised (no requests in log)")
    data = default_randbats_data()
    model = OpponentModel(data)
    battle = replay(lines, args.turn, team, model)
    opp = battle.opponent_active_pokemon
    print(f"turn {battle.turn}: poke-env opp {opp.species} status={opp.status} "
          f"counter={opp.status_counter} effects={dict(opp.effects)} "
          f"pp={ {k: v.current_pp for k, v in opp.moves.items()} } "
          f"last_used={[k for k, v in opp.moves.items() if v.is_last_used]}")
    sampler = es.TeamSampler(data)
    rng = random.Random(args.seed)
    trackers = model.update(battle)
    results = []
    for index in range(args.worlds):
        slots = es.sample_opponent_team(battle, trackers, sampler, rng)
        built = es.build_state(battle, slots)
        if index == 0:
            describe(built, bench=True)
        if index == args.instr_world:
            first_state = built.to_string()
            if index:
                describe(built)
        results.append(
            _worker.search_world(built.to_string(), args.time_ms, 1, args.evaluator, args.scale)
        )
    for pair in args.instr:
        mine, theirs = pair.split(",")
        from poke_engine import State, generate_instructions

        outcomes = generate_instructions(State.from_string(first_state), mine, theirs)
        print(f"instructions {mine} vs {theirs}:")
        for outcome in outcomes[:8]:
            print(f"  {outcome.percentage:6.2f}%  {[str(i) for i in outcome.instruction_list]}")
    policy = aggregate_policy(results)
    print("policy:", {k: round(v, 3) for k, v in sorted(policy.items(), key=lambda kv: -kv[1])})
    for index, result in enumerate(results):
        print(f"visits (world {index}):", [(c, v, round(s / max(v, 1), 3)) for c, v, s in result[1]])


if __name__ == "__main__":
    main()
