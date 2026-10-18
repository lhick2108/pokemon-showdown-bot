"""TokenPolicyPlayer: plays the token transformer's policy head directly, no search (bead 589.68 W2).

Each turn it samples ONE world of the opposing team (the search bots' sampler), tokenises it from
its own point of view with the opponent scrubbed to what the set tracker has seen
(``tokens.reveal_rows(_opp_facts(...))``, the training view), and takes the argmax of the policy
head over the legal actions (default), or samples from ``policy ** (1/temp)`` when ``temp`` is
given (or ``pick="sample"``). Spec: ``token-policy:model=models\\tok-d256.pt[:device=cuda][:temp=0.5]``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from poke_env.player.battle_order import BattleOrder

from showdown_bot import token_policy as tp
from showdown_bot import tokens
from showdown_bot.players.search import SearchPlayer, _opp_facts

logger = logging.getLogger(__name__)

PICKS = ("argmax", "sample")


class TokenPolicyPlayer(SearchPlayer):
    def __init__(
        self,
        *args: Any,
        model: str | Path | None = None,
        device: str = "auto",
        temp: float | None = None,
        pick: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if model is None:
            raise ValueError("TokenPolicyPlayer needs model=<token net .pt>")
        if pick is None:
            pick = "argmax" if temp is None else "sample"
        if pick not in PICKS:
            raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
        if temp is not None and not temp > 0:
            raise ValueError(f"temp must be > 0, got {temp}")
        if not tp.valid_device(device):
            raise ValueError(f"device must be auto, cpu, cuda or cuda:N, got {device!r}")
        self.policy_net = tp.TokenPolicy.load(model, device)
        self.pick = pick
        self.temp = float(temp) if temp is not None else (1.0 if pick == "sample" else None)
        self._pick_rng = random.Random(self._rng.random())
        self.token_moves = 0
        self.token_errors = 0

    async def _choose(self, battle: Any) -> BattleOrder | None:
        legal = battle.valid_orders
        if len(legal) == 1:
            self.stats.forced += 1
            return legal[0]
        if not legal:
            return None
        try:
            trackers = self.opponent_model.update(battle)
            reveal = tokens.reveal_rows(_opp_facts(battle, trackers))
            _slots, states, built = self._sample_worlds(battle, trackers, 1)
            dist = self.policy_net.distribution(states, reveal)
            orders: dict[str, BattleOrder] = {}
            for choice in dist:
                order = built.choice_to_order(choice, battle)
                if order is not None:
                    orders[choice] = order
            dist = tp.normalise(list(orders), [dist[c] for c in orders])
            if not dist:
                raise RuntimeError("no legal action with probability")
            choice = tp.pick(dist, self.temp if self.pick == "sample" else None, self._pick_rng)
        except Exception:
            self.token_errors += 1
            logger.exception("token policy failed in %s; random legal move", battle.battle_tag)
            return self._pick_rng.choice(legal)
        self.last_choice = choice
        self.last_policy = dist
        self.stats.searched += 1
        self.token_moves += 1
        return orders[choice]
