"""Aux targets of the human replay conversion (alive at end, KO credit, slot alignment)."""

from __future__ import annotations

import gzip
import json

import pytest

pytest.importorskip("poke_engine")

from poke_engine import State  # noqa: E402

from showdown_bot import human_replays as hr, recorder  # noqa: E402
from test_human_replays import LOG  # noqa: E402

# LOG with Mightyena KO'd by Ceruledge's Bitter Blade on turn 6 (the faint is in turn 6's events)
FAINT_LOG = LOG.replace(
    "|-status|p2a: mightyena|brn\n|-damage|p2a: mightyena|120/287 brn\n",
    "|-damage|p2a: mightyena|0 fnt\n|faint|p2a: mightyena\n",
)


def _e(*parts):
    return ["", *parts]


def test_game_facts_ko_credit():
    facts = hr.game_facts(hr.sanitize_log(FAINT_LOG).events, 7)
    assert facts.fainted == {"p2:mightyena"}
    assert facts.faint_count == {"p1": 0, "p2": 1}
    assert facts.ko_turns == {"p1:ceruledge": [6]}

    # Rocky Helmet credits the [of] mon; poison credits nobody; duplicate faints count once
    events = [
        _e("turn", "3"),
        _e("move", "p1a: a", "tackle", "p2a: b"),
        _e("-damage", "p1a: a", "0 fnt", "[from] item: Rocky Helmet", "[of] p2a: b"),
        _e("faint", "p1a: a"),
        _e("turn", "4"),
        _e("-damage", "p2a: b", "0 fnt", "[from] psn"),
        _e("faint", "p2a: b"),
        _e("faint", "p2a: b"),
    ]
    facts = hr.game_facts(events, 4)
    assert facts.ko_turns == {"p2:b": [3]}
    assert facts.faint_count == {"p1": 1, "p2": 1}


@pytest.mark.parametrize("first_person", [False, True])
def test_aux_aligns_with_state_slots(first_person):
    out = hr.Converter(first_person=first_person).convert(FAINT_LOG, "aux-1", seed=3)
    for role, final_alive in (("p1", [6, 5]), ("p2", [5, 6])):
        for ts in out.states[role]:
            s = State.from_string(ts.state)
            aux = ts.aux
            assert aux["turns_left"] == 7 - ts.turn
            assert aux["final_alive"] == final_alive
            for key, side in (("own", s.side_one), ("opp", s.side_two)):
                assert len(aux[key]["ident"]) == len(side.pokemon) == 6
                for field in ("alive_end", "kos", "kos_after"):
                    assert len(aux[key][field]) == 6
                # slot i of the aux is slot i of the state (log ident == species id in this log)
                for ident, mon in zip(aux[key]["ident"], side.pokemon):
                    if ident is not None:
                        assert ident == mon.id.lower(), (role, ts.turn, key, ident, mon.id)
                for mon, alive in zip(side.pokemon, aux[key]["alive_end"]):
                    if mon.hp <= 0 and alive != -1:
                        assert alive == 0
            active = s.side_one.pokemon[int(s.side_one.active_index)]
            assert aux["own"]["ident"][0] == active.id.lower()  # slot 0 = active
    p1_t6 = next(t for t in out.states["p1"] if t.turn == 6).aux
    p1_t7 = next(t for t in out.states["p1"] if t.turn == 7).aux
    assert p1_t6["own"]["ident"][0] == "ceruledge"
    assert p1_t6["own"]["kos"][0] == 1 and p1_t6["own"]["kos_after"][0] == 1
    assert p1_t7["own"]["kos"][0] == 1 and p1_t7["own"]["kos_after"][0] == 0
    opp = p1_t6["opp"]
    if "mightyena" in opp["ident"]:
        assert opp["alive_end"][opp["ident"].index("mightyena")] == 0
    if first_person:
        # unseen slots are fillers: unknown alive_end
        assert opp["alive_end"].count(-1) == opp["ident"].count(None) > 0
    else:
        # hindsight: unrevealed slots are never-seen real members, hence alive
        assert opp["ident"].count(None) == 4 and -1 not in opp["alive_end"]
        assert sum(opp["alive_end"]) == 5
    p2_t6 = next(t for t in out.states["p2"] if t.turn == 6).aux
    assert p2_t6["own"]["alive_end"][p2_t6["own"]["ident"].index("mightyena")] == 0
    assert sum(p2_t6["own"]["alive_end"]) == 5


def test_aux_round_trips_through_writer(tmp_path):
    out = hr.Converter().convert(FAINT_LOG, "aux-2", seed=3)
    rec = recorder.GameRecorder(tmp_path, truth=True)
    hr.write_game(rec, out, {"elo": "1000-1199", "month_year": "x", "split": "train"})
    path = next(tmp_path.rglob("aux-2__p1.jsonl.gz"))
    with gzip.open(path, "rt") as handle:
        decisions = [json.loads(line) for line in handle if '"type":"decision"' in line]
    assert decisions[0]["aux"] == out.states["p1"][0].aux
