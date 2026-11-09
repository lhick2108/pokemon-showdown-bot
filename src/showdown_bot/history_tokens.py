"""Live event history for the token net, in the exporter's format (bead showdown-bot-589.68.7).

The shard exporter (``tokens.build_history``) rebuilds the net's 16 history rows from the *recorded decisions* of
a game: row ``2 (k - 1)`` is our action ``k`` decisions ago, row ``2 (k - 1) + 1`` the opponent's action of that
turn, with the hp / faint deltas between the token summaries of consecutive decisions (docs/tokens.md "Event
history"). A live bot has the same ingredients, so this module only supplies them from the live battle and then
calls the exporter's own ``build_history`` (one implementation, no drift):

* ``summary``  - ``tokens._summary`` of the tokens of the decision's state (world 0, opponent scrubbed with the
  revealed facts: hidden opponent Pokemon have species id 1 and are not part of the alive map);
* own action   - the engine choice string the bot played (``"tackle"``, ``"tackle-tera"``, ``"switch pikachu"``);
* opponent action of turn ``t`` - ``OppActionWatcher``: from the protocol events, the opponent's LAST decision of
  the turn (what the peer's recording holds under ``peer_by_turn[t]``): its move / switch at the start of the
  turn, replaced by a pivot or faint-replacement switch later in the same turn when the opponent had a choice
  (two or more Pokemon to pick from). Moves with a ``[from]`` tag (locked moves, Sleep Talk...), turns where the
  opponent could not act and turns the opponent was forced to a single option are unknown (kind 0 row), like the
  recordings, which have no decision for a forced turn.

Differences from the recordings that the live bot cannot remove: the recorded action of a Pokemon that was
asleep, flinched, fully paralysed or KO'd before moving is the *chosen* action, the live one is unknown; the
recordings skip decisions of fast (non-recorded) searches, the live bot records every non-forced decision.
``scripts/history_live_check.py`` measures the match rate field by field.

``RevealTracker`` is the mirror image for the other direction: what the OPPONENT has seen of OUR team (species
that appeared, moves used, items / abilities / Tera types revealed), as ``opp_facts``-style rows for
``reveal2`` of net-search (the side-two view is scrubbed to it).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np

from showdown_bot import tokens

N_HIST = tokens.N_HIST


def _id(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _role(ident: Any) -> str:
    return str(ident)[:2]


class OppActionWatcher:
    """The opponent's last decision of every turn, as an engine choice string (see the module docs)."""

    def __init__(self) -> None:
        self.turn = 0
        self.decisions: dict[int, str] = {}
        self._acted = False
        self._upkeep = False
        self._tera = False
        self._teamsize = 6
        self._faints = 0
        self._active_fainted = False

    def options_left(self) -> int:
        """Pokemon the opponent could switch to now (its alive bench, plus nothing for a fainted active)."""
        alive = self._teamsize - self._faints
        return alive if self._active_fainted else alive - 1

    def observe(self, event: Sequence[str], opp_role: str) -> None:
        if len(event) < 2:
            return
        kind = str(event[1])
        if kind == "turn" and len(event) >= 3:
            try:
                self.turn = int(event[2])
            except ValueError:
                return
            self._acted = False
            self._upkeep = False
            self._tera = False
        elif kind == "teamsize" and len(event) >= 4 and str(event[2]) == opp_role:
            try:
                self._teamsize = int(event[3])
            except ValueError:
                pass
        elif kind == "upkeep":
            self._upkeep = True
        elif kind == "faint" and len(event) >= 3 and _role(event[2]) == opp_role:
            self._faints += 1
            self._active_fainted = True
        elif kind == "-terastallize" and len(event) >= 3:
            if _role(event[2]) == opp_role and not self._acted:
                self._tera = True
        elif kind == "move" and len(event) >= 4 and self.turn > 0:
            if _role(event[2]) != opp_role or self._acted or self._upkeep:
                return
            if any(str(t).startswith("[from]") for t in event[4:]):
                return
            self._acted = True
            self.decisions[self.turn] = _id(event[3]) + ("-tera" if self._tera else "")
        elif kind == "switch" and len(event) >= 4 and self.turn > 0:
            if _role(event[2]) != opp_role:
                return
            choice = "switch " + _id(str(event[3]).split(",")[0])
            if not self._acted and not self._upkeep:
                self._acted = True
                self.decisions[self.turn] = choice
            elif self.options_left() >= 2:
                # a pivot / replacement the opponent chose among at least two Pokemon
                self.decisions[self.turn] = choice
            self._active_fainted = False


class RevealTracker:
    """What the Pokemon of ``role`` have shown the other side, from the protocol events."""

    def __init__(self) -> None:
        self._species_of: dict[str, str] = {}  # "p1: Nickname" -> species
        self.facts: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

    def _entry(self, species: str) -> dict[str, Any]:
        key = _id(species)
        e = self.facts.get(key)
        if e is None:
            e = self.facts[key] = {"species": key, "moves": [], "item": False, "ability": False, "tera": False}
        return e

    def _owner(self, ident: Any) -> dict[str, Any] | None:
        text = str(ident)
        name = text.split(": ", 1)[1] if ": " in text else text
        species = self._species_of.get(_role(text) + ": " + name)
        return None if species is None else self._entry(species)

    def observe(self, event: Sequence[str], role: str) -> None:
        if len(event) < 3:
            return
        kind = str(event[1])
        ident = str(event[2])
        mine = _role(ident) == role
        if kind in ("switch", "drag", "replace", "detailschange", "-formechange") and mine and len(event) >= 4:
            name = ident.split(": ", 1)[1] if ": " in ident else ident
            species = str(event[3]).split(",")[0]
            self._species_of[role + ": " + name] = species
            self._entry(species)
            return
        # item / ability reveals carry "[from] item: X" / "[from] ability: X" (owner: "[of] ident" or the target)
        tags = [str(t) for t in event[3:] if isinstance(t, str)]
        owner_ident = ident
        for t in tags:
            if t.startswith("[of] "):
                owner_ident = t[5:]
        if _role(owner_ident) == role:
            for t in tags:
                if t.startswith("[from] item:"):
                    o = self._owner(owner_ident)
                    if o is not None:
                        o["item"] = True
                elif t.startswith("[from] ability:"):
                    o = self._owner(owner_ident)
                    if o is not None:
                        o["ability"] = True
        if not mine:
            return
        o = self._owner(ident)
        if o is None:
            return
        if kind == "move" and len(event) >= 4 and not any(t.startswith("[from]") for t in tags):
            mv = _id(event[3])
            if mv and mv not in o["moves"]:
                o["moves"].append(mv)
        elif kind in ("-item", "-enditem"):
            o["item"] = True
        elif kind == "-ability":
            o["ability"] = True
        elif kind == "-terastallize":
            o["tera"] = True

    def rows(self) -> list[dict[str, Any]]:
        """``opp_facts``-style rows for ``tokens.reveal_rows``."""
        return [dict(v, moves=list(v["moves"])) for v in self.facts.values()]


class BattleHistory:
    """Per battle: the recorded decisions of this bot and the events that complete them."""

    def __init__(self, vocab: "tokens.Vocab | None" = None) -> None:
        self.vocab = vocab or tokens.Vocab()
        self.recs: list[dict[str, Any]] = []
        self.pending = False
        self.opp = OppActionWatcher()
        self.own = RevealTracker()
        self.last: tuple[np.ndarray, np.ndarray] | None = None
        self._early: list[Sequence[str]] = []

    def observe(self, events: Sequence[Sequence[str]], my_role: str | None) -> None:
        if my_role not in ("p1", "p2"):  # the role arrives with the first request: replay the early events then
            if len(self._early) < 4000:
                self._early.extend(events)
            return
        if self._early:
            early, self._early = self._early, []
            self.observe(early, my_role)
        opp_role = "p2" if my_role == "p1" else "p1"
        for event in events:
            if len(event) >= 2 and event[0] == "":
                self.opp.observe(event, opp_role)
                self.own.observe(event, my_role)

    def begin(self, turn: int, bufs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Registers a decision at ``turn`` whose state tokens are ``bufs`` (``tokens.split_buffers``) and returns
        its 16 history rows ``(hist_cat int16 [16,4], hist_cont float32 [16,5])``. A decision that was begun but
        not finished (the bot fell back to another policy within the same decision) is reused, not duplicated."""
        if self.pending and self.recs and self.recs[-1]["turn"] == int(turn) and self.last is not None:
            return self.last
        if self.pending and self.recs:
            self.recs.pop()  # the unfinished decision of an earlier request
        for r in self.recs:
            r["opp_act"] = self.vocab.parse(self.opp.decisions.get(r["turn"]))
        self.recs.append({"turn": int(turn), "sum": tokens._summary(bufs), "own_act": (-1, 0), "opp_act": (-1, 0)})
        self.pending = True
        self.last = tokens.build_history(self.recs, len(self.recs) - 1)
        return self.last

    def chosen(self, choice: str | None) -> None:
        """The engine choice string played for the decision started by ``begin``."""
        if self.recs and self.pending:
            self.recs[-1]["own_act"] = self.vocab.parse(choice)
        self.pending = False
        self.last = None

    def abandon(self) -> None:
        """Forgets a decision that was begun and never played (the bot failed)."""
        if self.pending and self.recs:
            self.recs.pop()
        self.pending = False
        self.last = None

    def our_facts(self) -> list[dict[str, Any]]:
        return self.own.rows()
