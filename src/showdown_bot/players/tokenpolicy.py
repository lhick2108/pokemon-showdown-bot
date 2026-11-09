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
from collections import OrderedDict
from pathlib import Path
from typing import Any

from poke_env.player.battle_order import BattleOrder

from showdown_bot import history_tokens
from showdown_bot import token_policy as tp
from showdown_bot import tokens
from showdown_bot.sets import _real_battle_tag
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
        hist: int = 0,
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
        self.last_states: list[str] = []
        self.hist_mode = int(hist)
        self._histories: "OrderedDict[str, history_tokens.BattleHistory]" = OrderedDict()
        self._vocab: tokens.Vocab | None = None

    # ------------------------------------------------------------------ live history
    def _history(self, tag: str, create: bool = True) -> "history_tokens.BattleHistory | None":
        h = self._histories.get(tag)
        if h is None and create:
            if self._vocab is None:
                self._vocab = tokens.Vocab()
            h = self._histories[tag] = history_tokens.BattleHistory(self._vocab)
            while len(self._histories) > 8:
                self._histories.popitem(last=False)
        return h

    def _tracks_events(self) -> bool:
        """Subclasses that need the event bookkeeping (history, side-two scrub) override."""
        return self.hist_mode > 0

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> Any:
        if self._tracks_events():
            try:
                tag = _real_battle_tag(split_messages)
                if tag is not None:
                    battle = getattr(self, "_battles", {}).get(tag)
                    self._history(tag).observe(split_messages[1:], getattr(battle, "player_role", None))
            except Exception:  # bookkeeping must never break the battle loop
                logger.debug("history tracking failed", exc_info=True)
        return await super()._handle_battle_message(split_messages)

    def _live_hist(self, battle: Any, state0: str, reveal: Any) -> Any:
        """The 16 history rows of this decision ``(hist_cat, hist_cont)`` (registers the decision)."""
        import poke_engine

        cat, cont, _names = poke_engine.state_tokens_str(state0, 1, reveal, False)
        bufs = tokens.split_buffers(cat, cont, tokens.layout())
        return self._history(battle.battle_tag).begin(int(battle.turn), bufs)

    def _hist_done(self, battle: Any, choice: str | None) -> None:
        h = self._history(battle.battle_tag, create=False)
        if h is not None:
            h.chosen(choice)

    def _hist_abandon(self, battle: Any) -> None:
        h = self._history(battle.battle_tag, create=False)
        if h is not None:
            h.abandon()

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
            self.last_states = states
            hist = self._live_hist(battle, states[0], reveal) if self.hist_mode else None
            dist = self.policy_net.distribution(states, reveal, hist)
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
            self._hist_abandon(battle)
            return self._pick_rng.choice(legal)
        self._hist_done(battle, choice)
        self.last_choice = choice
        self.last_policy = dist
        self.stats.searched += 1
        self.token_moves += 1
        return orders[choice]
