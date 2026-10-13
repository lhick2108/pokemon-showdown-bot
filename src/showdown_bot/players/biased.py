"""Scripted exploitable opponents for validating the tendency model (bead 589.40).

``BiasedSearchPlayer`` is a normal ``SearchPlayer`` whose final choice is restricted to a
class of actions when a trigger holds:

* ``bait``   - whenever the peer (our bot) has a priority-conditional move (Sucker Punch,
               Thunderclap, Upper Hand) on its active Pokemon, use a status / Protect move
               (the human behaviour that made Sucker Punch fail every time);
* ``attack`` - always use a damaging move (never switch, set up or Protect) if one is legal.

The trigger peeks at the peer's own battle (both players live in one process in the
validation scripts). If no move of the wanted class is legal the search choice stands.
"""

from __future__ import annotations

from typing import Any

from showdown_bot import opponent_tendency as tendency
from showdown_bot.players.search import SearchPlayer

BIASES = ("bait", "attack")


class BiasedSearchPlayer(SearchPlayer):
    def __init__(self, *args: Any, bias: str = "bait", **kwargs: Any):
        if bias not in BIASES:
            raise ValueError(f"bias must be one of {BIASES}, got {bias!r}")
        super().__init__(*args, **kwargs)
        self.bias = bias
        self.peer: Any = None  # the opposing in-process player
        self.trigger_turns = 0
        self.biased_turns = 0
        self.triggered_games: set[str] = set()

    def _triggered(self, battle: Any) -> bool:
        if self.bias == "attack":
            return True
        peer = self.peer
        peer_battle = getattr(peer, "_battles", {}).get(battle.battle_tag) if peer else None
        mon = getattr(peer_battle, "active_pokemon", None)
        if mon is None or getattr(mon, "fainted", False):
            return False
        return any(
            tendency._id(mid) in tendency.PRIORITY_CONDITIONAL and getattr(m, "current_pp", 1) > 0
            for mid, m in mon.moves.items()
        )

    def _order_from_policy(self, policy, built, battle, mode=None):
        if self._triggered(battle):
            self.trigger_turns += 1
            self.triggered_games.add(battle.battle_tag)
            wanted = (
                {tendency.STATUS, tendency.PROTECT}
                if self.bias == "bait"
                else {tendency.ATTACK, tendency.PRIORITY}
            )
            usable = {}
            for choice, mass in policy.items():
                if tendency.classify_choice(choice) in wanted and built.choice_to_order(choice, battle):
                    usable[choice] = mass
            if usable:
                self.biased_turns += 1
                policy = usable
        return super()._order_from_policy(policy, built, battle, mode)
