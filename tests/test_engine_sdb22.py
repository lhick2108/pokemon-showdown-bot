"""State-format compatibility of the sdb.22 engine fields (Move.max_pp, healblock / throatchop durations)."""

from __future__ import annotations

import poke_engine as pe
import pytest

HAS_SDB22 = True
try:
    pe.Move(id="tackle", max_pp=8)
except TypeError:
    HAS_SDB22 = False

pytestmark = pytest.mark.skipif(not HAS_SDB22, reason="needs a poke_engine wheel >= 0.0.48+sdb.22")


def _state(max_pp: int = 0, healblock: int = 0, throatchop: int = 0) -> pe.State:
    moves = [pe.Move(id=m, pp=16, max_pp=max_pp) for m in ("tackle", "recover", "protect", "toxic")]
    mon = pe.Pokemon(id="pikachu", moves=moves)
    side_one = pe.Side(
        pokemon=[mon],
        volatile_status_durations=pe.VolatileStatusDurations(healblock=healblock, throatchop=throatchop),
    )
    return pe.State(side_one=side_one, side_two=pe.Side(pokemon=[mon]))


def test_unknown_max_pp_and_zero_durations_serialize_like_old_wheels():
    text = _state().to_string()
    # Move: id;disabled;pp (no 4th field)
    assert "TACKLE;false;16," in text
    back = pe.State.from_string(text)
    assert back.side_one.pokemon[0].moves[0].max_pp == 0


def test_known_max_pp_and_new_durations_round_trip():
    state = _state(max_pp=24, healblock=4, throatchop=1)
    back = pe.State.from_string(state.to_string())
    assert back.side_one.pokemon[0].moves[0].max_pp == 24
    assert back.side_one.pokemon[0].moves[0].pp == 16
    assert back.side_one.volatile_status_durations.healblock == 4
    assert back.side_one.volatile_status_durations.throatchop == 1


def test_old_format_strings_still_parse():
    text = _state(max_pp=24, healblock=4, throatchop=1).to_string()
    # strip the new trailing fields back to the pre-sdb.22 format
    old = text.replace(";24,", ",").replace(";24;", ";").replace(";4;1", "")
    back = pe.State.from_string(old)
    assert back.side_one.pokemon[0].moves[0].max_pp == 0
    assert back.side_one.volatile_status_durations.healblock == 0
