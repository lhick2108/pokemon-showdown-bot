"""Human replay conversion: sanitising, actions, hindsight teams, states, privacy."""

from __future__ import annotations

import gzip
import json

import pytest

pytest.importorskip("poke_engine")

from poke_engine import State  # noqa: E402

from showdown_bot import dataset, human_replays as hr, recorder  # noqa: E402

# A real server log (first five turns of a 2024 ladder game), usernames replaced by fakes and
# chat lines kept to prove they never leak. Lowercased names, exact HP for both sides.
LOG = """\
|j|☆alice_secret
|j|☆bob_secret
|t:|1725608019
|gametype|singles
|player|p1|alice_secret|preschooler|1163
|player|p2|bob_secret|1|1172
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[gen 9] random battle
|rated|
|rule|species clause: limit one of each pokémon
|
|t:|1725608019
|start
|switch|p1a: eternatus|eternatus, l69|308/308
|switch|p2a: mightyena|mightyena, l95, f|287/287
|-ability|p1a: eternatus|pressure
|-ability|p2a: mightyena|intimidate|boost
|-unboost|p1a: eternatus|atk|1
|turn|1
|inactive|battle timer is on: inactive players will automatically lose when time's up.
|
|t:|1725608038
|switch|p2a: farigiraf|farigiraf, l91, f|366/366
|move|p1a: eternatus|toxic spikes|p2a: farigiraf
|-sidestart|p2: bob_secret|move: toxic spikes
|
|upkeep
|turn|2
|c|☆alice_secret|my secret chat line
|
|t:|1725608056
|switch|p1a: ceruledge|ceruledge, l78, f|245/245
|move|p2a: farigiraf|future sight|p1a: ceruledge
|-start|p2a: farigiraf|move: future sight
|
|upkeep
|turn|3
|
|t:|1725608068
|move|p1a: ceruledge|swords dance|p1a: ceruledge
|-boost|p1a: ceruledge|atk|2
|move|p2a: farigiraf|wish|p2a: farigiraf
|
|upkeep
|turn|4
|c|☆bob_secret|bob chat
|
|t:|1725608076
|move|p2a: farigiraf|protect|p2a: farigiraf
|-singleturn|p2a: farigiraf|protect
|move|p1a: ceruledge|poltergeist|p2a: farigiraf
|-activate|p2a: farigiraf|move: protect
|
|-end|p1a: ceruledge|move: future sight
|-damage|p1a: ceruledge|91/245
|upkeep
|turn|5
|
|t:|1725608085
|move|p1a: ceruledge|bitter blade|p2a: farigiraf
|-damage|p2a: farigiraf|144/366
|-heal|p1a: ceruledge|202/245|[from] drain|[of] p2a: farigiraf
|move|p2a: farigiraf|future sight|p1a: ceruledge
|-start|p2a: farigiraf|move: future sight
|
|-heal|p2a: farigiraf|166/366|[from] item: leftovers
|upkeep
|turn|6
|
|t:|1725608095
|switch|p2a: mightyena|mightyena, l95, f|287/287
|-terastallize|p1a: ceruledge|fire
|move|p1a: ceruledge|bitter blade|p2a: mightyena
|-status|p2a: mightyena|brn
|-damage|p2a: mightyena|120/287 brn
|
|upkeep
|turn|7
|inactive|bob_secret has 120 seconds left.
|-message|bob_secret lost due to inactivity.
|
|win|alice_secret
"""
USERNAMES = ("alice_secret", "bob_secret", "secret chat", "bob chat")


@pytest.fixture(scope="module")
def converted():
    conv = hr.Converter()
    return conv.convert(LOG, "test-battle-1", seed=1)


def _active(state, side):
    s = getattr(state, side)
    return s.pokemon[int(s.active_index)], s


def test_sanitize_drops_names_and_chat():
    log = hr.sanitize_log(LOG)
    blob = json.dumps(log.events)
    for secret in USERNAMES:
        assert secret not in blob
    assert log.winner == "p1"
    assert log.skip is None
    assert log.turns == 7
    details = [e[3] for e in log.events if e[1] == "switch"]
    assert "eternatus, L69" in details and "mightyena, L95, F" in details  # level / gender casing
    sidestart = next(e for e in log.events if e[1] == "-sidestart")
    assert sidestart[2] == "p2: side"


def test_winner_and_skips():
    assert hr.sanitize_log(LOG.replace("|win|alice_secret", "|win|bob_secret")).winner == "p2"
    assert hr.sanitize_log(LOG.replace("|win|alice_secret", "|tie")).winner == "tie"
    assert hr.sanitize_log(LOG.replace("|win|alice_secret\n", "")).skip == "no_result"
    zoroark = LOG.replace("|turn|2", "|replace|p2a: zoroark|zoroark, L80\n|turn|2")
    assert hr.sanitize_log(zoroark).skip == "illusion"


def test_extract_actions():
    actions = hr.extract_actions(hr.sanitize_log(LOG).events)
    assert actions[1]["p1"] == hr.Action("move", move_id="toxicspikes")
    assert actions[1]["p2"] == hr.Action("switch", ident="farigiraf", species_id="farigiraf")
    assert actions[2]["p1"] == hr.Action("switch", ident="ceruledge", species_id="ceruledge")
    assert actions[2]["p2"] == hr.Action("move", move_id="futuresight")
    assert actions[3]["p1"] == hr.Action("move", move_id="swordsdance")
    assert actions[5]["p1"] == hr.Action("move", move_id="bitterblade")
    assert actions[6]["p1"] == hr.Action("move", move_id="bitterblade", tera=True)
    assert actions[6]["p2"].kind == "switch"


def test_hindsight_teams_cover_whole_team():
    conv = hr.Converter()
    events = hr.sanitize_log(LOG).events
    import random

    team = hr.hindsight_team(events, "p1", "p2", conv.data, conv.sampler, random.Random(0), conv.model)
    assert len(team) == 6
    assert len({m.species_id for m in team}) == 6
    revealed = [m for m in team if m.revealed]
    assert {m.ident for m in revealed} == {"mightyena", "farigiraf"}
    farigiraf = next(m for m in team if m.ident == "farigiraf")
    assert farigiraf.sampled.level == 91
    assert "futuresight" in [x.replace(" ", "").lower() for x in farigiraf.sampled.moves]
    assert "wish" in [x.replace(" ", "").lower() for x in farigiraf.sampled.moves]
    assert "protect" in [x.replace(" ", "").lower() for x in farigiraf.sampled.moves]
    assert farigiraf.sampled.item.lower() == "leftovers"  # [from] item: leftovers


def test_state_matches_log(converted):
    by_turn = {ts.turn: ts for ts in converted.states["p1"]}
    assert sorted(by_turn) == list(range(1, 8))
    # turn 1: both leads at full HP, Eternatus -1 Atk from Intimidate
    s1 = State.from_string(by_turn[1].state)
    us, us_side = _active(s1, "side_one")
    opp, opp_side = _active(s1, "side_two")
    assert (us.id.lower(), us.hp, us.maxhp) == ("eternatus", 308, 308)
    assert (opp.id.lower(), opp.hp, opp.maxhp) == ("mightyena", 287, 287)
    assert us_side.attack_boost == -1
    assert us.ability.lower() == "pressure" and opp.ability.lower() == "intimidate"
    # turn 5: Ceruledge at 91/245 with +2 Attack, Farigiraf not yet hurt
    s5 = State.from_string(by_turn[5].state)
    us, us_side = _active(s5, "side_one")
    opp, _ = _active(s5, "side_two")
    assert (us.id.lower(), us.hp, us.maxhp) == ("ceruledge", 91, 245)
    assert us_side.attack_boost == 2
    assert (opp.id.lower(), opp.hp, opp.maxhp) == ("farigiraf", 366, 366)
    # turn 6: Farigiraf at 166/366 (leftovers heal applied); Ceruledge back to 245
    s6 = State.from_string(by_turn[6].state)
    us, _ = _active(s6, "side_one")
    opp, _ = _active(s6, "side_two")
    assert opp.hp == 166 and opp.maxhp == 366
    assert us.hp == 202
    # turn 7: Mightyena burned at 120/287; our Ceruledge Terastallized
    s7 = State.from_string(by_turn[7].state)
    us, _ = _active(s7, "side_one")
    opp, _ = _active(s7, "side_two")
    assert (opp.id.lower(), opp.hp, opp.status.lower()) == ("mightyena", 120, "burn")
    assert us.terastallized and us.tera_type.lower() == "fire"
    # toxic spikes on the p2 side (side two from p1's view)
    assert s7.side_two.side_conditions.toxic_spikes == 1


def test_state_from_other_side(converted):
    by_turn = {ts.turn: ts for ts in converted.states["p2"]}
    s5 = State.from_string(by_turn[5].state)
    us, _ = _active(s5, "side_one")  # our = p2 = farigiraf
    opp, opp_side = _active(s5, "side_two")
    assert (us.id.lower(), us.hp) == ("farigiraf", 366)
    assert (opp.id.lower(), opp.hp) == ("ceruledge", 91)
    assert opp_side.attack_boost == 2
    assert converted.result == {"p1": "win", "p2": "loss"}


def test_choice_strings_and_legality(converted):
    p1 = {ts.turn: ts for ts in converted.states["p1"]}
    assert p1[1].chosen == "toxicspikes" and p1[1].choice_ok
    assert p1[2].chosen == "switch ceruledge" and p1[2].choice_ok
    assert p1[6].chosen == "bitterblade-tera" and p1[6].choice_ok
    assert "bitterblade-tera" not in p1[7].legal  # already Terastallized
    p2 = {ts.turn: ts for ts in converted.states["p2"]}
    assert p2[1].chosen == "switch farigiraf"
    assert p2[6].chosen == "switch mightyena"
    assert all(ts.choice_ok for ts in p1.values() if ts.chosen)


def test_write_and_load_round_trip(tmp_path, converted):
    rec = recorder.GameRecorder(tmp_path / "shard")
    header = {"elo": "1000-1199", "month_year": "September2024", "split": "test"}
    rows = hr.write_game(rec, converted, header)
    assert rows and all(r["action"] for r in rows)
    game = recorder.read_game(tmp_path / "shard" / "test-battle-1__p1.jsonl.gz")
    assert game["header"]["source"] == "human" and game["header"]["elo"] == "1000-1199"
    assert game["header"]["split"] == "test" and game["header"]["role"] == "p1"
    assert game["result"]["result"] == "win"
    samples = list(dataset.iter_samples(tmp_path / "shard", sides=(1,), truth=True))
    assert len(samples) == 14  # 7 turns x 2 sides
    assert {s.value for s in samples} == {0.0, 1.0}
    assert all(s.kind == "truth" and s.search_value is None for s in samples)
    # privacy: nothing written contains usernames or chat
    blobs = []
    for path in (tmp_path / "shard").glob("*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            blobs.append(handle.read())
    blobs.append(json.dumps(rows))
    for blob in blobs:
        for secret in USERNAMES:
            assert secret not in blob


def test_process_chunk_counts_skips(tmp_path):
    games = [
        {"text": LOG, "battle_id": "g1", "elo": "1200-1399", "month_year": "x", "split": "train"},
        {"text": "|j|☆x\n|c|☆x|hi", "battle_id": "g2", "elo": "1200-1399",
         "month_year": "x", "split": "train"},
    ]
    result = hr.process_chunk({
        "games": games, "seed": 0, "shard_dir": str(tmp_path / "train" / "shard-00000"),
        "policy_path": str(tmp_path / "policy" / "train" / "shard-00000.jsonl.gz"),
    })
    assert result["ok"] == 1 and result["skips"] == {"no_turns": 1}
    with gzip.open(tmp_path / "policy" / "train" / "shard-00000.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert rows[0]["battle_tag"] == "g1" and rows[0]["split"] == "train"
    assert (tmp_path / "train" / "shard-00000" / ".done").exists()


def test_policy_dir_is_not_a_game_file(tmp_path, converted):
    rec = recorder.GameRecorder(tmp_path / "test" / "shard-00000")
    hr.write_game(rec, converted, {"elo": "x", "month_year": "y", "split": "test"})
    (tmp_path / "policy" / "test").mkdir(parents=True)
    with gzip.open(tmp_path / "policy" / "test" / "shard-00000.jsonl.gz", "wt") as handle:
        handle.write('{"action": "x"}\n')
    files = list(recorder.iter_game_files(tmp_path))
    assert len(files) == 2 and all("policy" not in f.parts for f in files)
