"""Sleep bookkeeping that poke-env does not do (bead 589.35).

poke-env only keeps ``status == SLP`` and a counter that it bumps on *every* ``cant``
and ``move`` event of a sleeping Pokemon, so a Sleep Talk turn (``cant`` + Sleep Talk +
the called move) counts three turns, and a Rest sleep cannot be told from an ordinary
one. Showdown's real timers (data/conditions.ts, data/moves.ts):

* ordinary sleep: ``time`` starts at 2-4 and is decremented once per move attempt
  (twice with Early Bird); at 0 the Pokemon wakes and acts. ``|cant|ID|slp`` is printed
  for every attempt it stays asleep, so the number of ``cant`` events is exactly the
  engine's ``sleep_turns`` (chance to wake 0, 1/3, 1/2, 1 on attempts 1-4).
* Rest: ``time`` starts at 3: two ``cant`` turns, wakes on the third attempt. The
  engine models this with ``rest_turns`` (3 -> 2 -> 1 -> wake).

The counter survives switching (there is no ``onSwitchIn`` reset for ``slp``), so
records are kept per Pokemon identity (``p1: Name``), not per active slot.

``SleepTracker`` consumes the raw protocol events (``OpponentModelPlayerMixin`` feeds it)
and ``engine_state`` reads it through :func:`for_battle`. Without a tracker (tests that
drive a poke-env battle by hand) engine_state falls back to the poke-env counter.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

REST_SLEEP_TURNS = 3  # Rest sets time = 3: two turns lost, wakes on the third attempt
_SLOT = re.compile(r"^(p[1-4])[a-d]?: ")


def _move_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def ident_key(ident: str) -> str:
    """``"p2a: Lapras"`` / ``"p2: Lapras"`` -> ``"p2: Lapras"``."""
    return _SLOT.sub(r"\1: ", str(ident).strip())


@dataclass
class SleepRecord:
    kind: str  # "rest" | "sleep"
    turns: int = 0  # move attempts lost to sleep so far (cant|slp events)

    def engine_fields(self) -> tuple[int, int]:
        """``(sleep_turns, rest_turns)`` for the engine's Pokemon."""
        if self.kind == "rest":
            return 0, max(1, min(REST_SLEEP_TURNS, REST_SLEEP_TURNS - self.turns))
        return max(0, min(3, self.turns)), 0


class SleepTracker:
    """Per-battle sleep records, keyed by ``ident_key``."""

    def __init__(self) -> None:
        self._records: dict[str, SleepRecord] = {}
        self.turn = 0
        # side ("p1"/"p2") -> (turn Wish was used, wisher ident key); see wish_pending
        self._wishes: dict[str, tuple[int, str]] = {}

    def record(self, ident: str) -> SleepRecord | None:
        return self._records.get(ident_key(ident))

    def observe(self, event: Sequence[str]) -> None:
        """Consume one raw ``['', 'cant', 'p2a: Lapras', 'slp']``-style event."""
        if len(event) < 3:
            return
        kind = str(event[1])
        if kind == "turn":
            try:
                self.turn = int(event[2])
            except ValueError:
                pass
        elif kind == "move" and len(event) >= 4:
            key = ident_key(event[2])
            move_id = _move_id(event[3])
            if move_id == "wish" and "[still]" not in event[4:]:
                pending = self._wishes.get(key[:2])
                # A Wish used while one is pending fails (Showdown's addSlotCondition
                # refuses a duplicate), so it must not re-arm the slot.
                if pending is None or pending[0] != self.turn - 1:
                    self._wishes[key[:2]] = (self.turn, key)
        elif kind == "-status" and len(event) >= 4:
            key = ident_key(event[2])
            if event[3] == "slp":
                rest = any(str(tag).replace(" ", "").lower() == "[from]move:rest" for tag in event[4:])
                self._records[key] = SleepRecord("rest" if rest else "sleep")
            else:
                self._records.pop(key, None)
        elif kind == "cant" and len(event) >= 4 and event[3] == "slp":
            key = ident_key(event[2])
            record = self._records.get(key)
            if record is None:  # tracker joined mid-sleep
                record = self._records[key] = SleepRecord("sleep")
            record.turns += 1
        elif kind == "-curestatus" and len(event) >= 4 and event[3] == "slp":
            self._records.pop(ident_key(event[2]), None)
        elif kind == "faint":
            self._records.pop(ident_key(event[2]), None)

    def observe_events(self, events: Sequence[Sequence[str]]) -> None:
        for event in events:
            self.observe(event)

    def wish_pending(self, side: str, turn: int) -> str | None:
        """Ident key of the Pokemon whose Wish is pending for ``side`` at the start of ``turn``.

        A Wish used during turn N-1 heals at the end of turn N (the slot condition lives two
        end-of-turns), so it is pending exactly while the current turn is N; a second Wish
        used meanwhile fails (Showdown's ``addSlotCondition`` refuses a duplicate).
        """
        record = self._wishes.get(side)
        if record is not None and record[0] == turn - 1:
            return record[1]
        return None


_MAX_BATTLES = 64
_TRACKERS: OrderedDict[str, SleepTracker] = OrderedDict()


def _tag(battle_tag: str) -> str:
    return str(battle_tag).lstrip(">")


def for_battle(battle_tag: str, *, create: bool = False) -> SleepTracker | None:
    tag = _tag(battle_tag)
    tracker = _TRACKERS.get(tag)
    if tracker is None and create:
        tracker = _TRACKERS[tag] = SleepTracker()
        while len(_TRACKERS) > _MAX_BATTLES:
            _TRACKERS.popitem(last=False)
    return tracker


def observe_events(battle_tag: str, events: Sequence[Sequence[str]]) -> None:
    """Feed raw events of one battle (creates the tracker on first use)."""
    tracker = for_battle(battle_tag, create=True)
    assert tracker is not None
    tracker.observe_events(events)
