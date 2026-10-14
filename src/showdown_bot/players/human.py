"""HumanPolicyPlayer: plays like a ladder human, with no search (offline test opponent).

Each turn it samples ONE world of the opposing team (the same sampler the search bots use), builds
the engine state from its own point of view, asks the human action-policy model
(``showdown_bot.human_policy``) for P(action) over its legal actions and samples from it
(``pick="sample"``, default) or takes the most likely action (``pick="argmax"``). ``temp`` < 1
sharpens the distribution (0.5 is a more predictable human), ``temp`` > 1 flattens it.

It is meant as the opponent that the exploiting bot (``opp_model=...``) should beat by more than
it beats a plain search bot.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
from poke_env.player.battle_order import BattleOrder

from showdown_bot import human_policy as hp
from showdown_bot.players.search import SearchPlayer, load_human_model

logger = logging.getLogger(__name__)

PICKS = ("sample", "argmax")


class HumanPolicyPlayer(SearchPlayer):
    def __init__(
        self,
        *args: Any,
        model: str | Path | None = None,
        pick: str = "sample",
        elo: str | int | None = None,
        temp: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if model is None:
            raise ValueError("HumanPolicyPlayer needs model=<human-policy .npz>")
        if pick not in PICKS:
            raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
        if not temp > 0:
            raise ValueError(f"temp must be > 0, got {temp}")
        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"human policy model not found: {path}")
        self.human = load_human_model(str(path))
        self.pick = pick
        self.human_elo = elo
        self.temp = float(temp)
        self._pick_rng = random.Random(self._rng.random())
        self.human_moves = 0

    async def _choose(self, battle: Any) -> BattleOrder | None:
        legal = battle.valid_orders
        if len(legal) == 1:
            self.stats.forced += 1
            return legal[0]
        if not legal:
            return None
        trackers = self.opponent_model.update(battle)
        _slots, states, built = self._sample_worlds(battle, trackers, 1)
        state = states[0]
        from poke_engine import State

        side = State.from_string(state).side_one
        moves, bench = hp.actor_slots(side)
        orders: dict[str, BattleOrder] = {}
        for slot in range(hp.N_SLOTS):
            choice = hp.choice_of(slot, moves, bench)
            if choice is None:
                continue
            order = built.choice_to_order(choice, battle)
            if order is not None:
                orders[choice] = order
        if not orders:
            return None
        dist = self.human.predict(state, list(orders), self.human_elo)
        choices = list(dist)
        p = np.array([dist[c] for c in choices], dtype=float)
        if self.temp != 1.0:
            p = p ** (1.0 / self.temp)
        p = p / p.sum()
        if self.pick == "argmax":
            choice = choices[int(np.argmax(p))]
        else:
            choice = self._pick_rng.choices(choices, weights=p.tolist())[0]
        self.last_choice = choice
        self.last_policy = dict(zip(choices, p.tolist()))
        self.stats.searched += 1
        self.human_moves += 1
        return orders[choice]
