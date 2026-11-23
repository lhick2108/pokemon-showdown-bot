"""RolloutSearchPlayer: the policy-rollout search (rollout_search.py, after Ataraxos) on the server (bead 589.68.8).

Each turn it samples ``worlds`` opposing teams (the search bots' sampler), builds a rollout pool from them with what
each side has seen and our history rows (``poke_engine.SelfPlayPool.from_worlds``, engine sdb.34+), plays every
candidate action of the net's policy in every world followed by ``depth - 1`` joint steps of the net against itself,
scores the positions reached with the value head and moves the policy one KL-anchored step towards the better
candidates (``rollout_live.LiveRollout``). The move is sampled from the result (``pick=sample``, as measured in the
simulator) or its argmax. If the search raises, the token-policy argmax is played and ``rollout_errors`` counts it.

    rollout-search:model=runs/ppo7/ckpt-36750.pt:device=cuda:worlds=16:depth=4:cands=6:beta=0.01[:opp=4][:pick=argmax]
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from poke_env.player.battle_order import BattleOrder

from showdown_bot import token_policy as tp
from showdown_bot import tokens
from showdown_bot.net_search import slot_names
from showdown_bot.players.search import _opp_facts
from showdown_bot.players.tokenpolicy import TokenPolicyPlayer
from showdown_bot.rollout_live import LiveRollout

logger = logging.getLogger(__name__)

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rollout")
    return _POOL


class RolloutSearchPlayer(TokenPolicyPlayer):
    def __init__(
        self,
        *args: Any,
        model: str | Path | None = None,
        device: str = "auto",
        worlds: int = 16,
        depth: int = 4,
        cands: int = 6,
        minp: float = 0.03,
        beta: float = 0.01,
        alpha: float = 0.0,
        opp: int = 1,
        pick: str = "sample",
        rtemp: float = 1.0,
        oppmin: float = 0.0,
        oppmodel: str | None = None,
        oppshare: float = 1.0,
        opptemp: float = 1.0,
        tokthreads: int = 2,
        seed: int = 0,
        **kwargs: Any,
    ):
        # the live history rows are always needed: the rollout games start from them
        super().__init__(*args, model=model, device=device, seed=seed, hist=1, **kwargs)
        if pick not in ("sample", "argmax"):
            raise ValueError(f"pick must be sample or argmax, got {pick!r}")
        if worlds < 1:
            raise ValueError("worlds must be >= 1")
        self.worlds = worlds
        self.rollout_pick = pick
        opp_net = None
        if oppmodel:
            from showdown_bot.ppo import load_net

            opp_net = load_net(oppmodel, str(self.policy_net.device))
        self.live = LiveRollout(self.policy_net.net, self.policy_net.device, depth=depth, cands=cands, minp=minp, beta=beta,
                                alpha=alpha, opp=opp, seed=seed, threads=tokthreads,
                                rtemp=rtemp, oppmin=oppmin, opp_net=opp_net, oppshare=oppshare, opptemp=opptemp)
        self._ro_rng = random.Random(seed * 7 + 3)
        self.rollout_errors = 0
        self.rollout_searches = 0
        self.rollout_evals = 0
        self.rollout_ms = 0.0
        self.rollout_changed = 0
        self.rollout_valid = 0.0
        # SHOWDOWN_ROLLOUT_LOG=<dir>: one JSON line per searched decision in <dir>/<battle tag>__<username>.jsonl
        # (what the net and the search thought; analysis only, no effect on play)
        self._ro_log = os.environ.get("SHOWDOWN_ROLLOUT_LOG") or None

    def _log_decision(self, battle: Any, names: list[str], info: dict[str, Any], probs: Any, choice: str) -> None:
        try:
            d = Path(self._ro_log)
            d.mkdir(parents=True, exist_ok=True)
            r3 = lambda xs: [round(float(x), 4) for x in xs]  # noqa: E731
            row = {
                "tag": battle.battle_tag, "user": self.username, "turn": int(battle.turn),
                "value": round(float(info["value"]), 4), "value_sd": round(float(info["value_sd"]), 4),
                "names": list(names), "prior": r3(info["prior"]), "probs": r3(probs),
                "cand": [names[c] for c in info["cand"]], "q": r3(info.get("q", [])),
                "worlds": info.get("worlds"), "ended": info.get("ended"), "choice": choice,
                "evals": int(info["evals"]), "ms": round(float(info["ms"]), 1),
            }
            with open(d / f"{battle.battle_tag}__{self.username}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + chr(10))
        except Exception:  # logging must never cost a move
            logger.exception("rollout decision log failed")

    def _tracks_events(self) -> bool:
        return True

    async def choose_move(self, battle: Any) -> BattleOrder:
        # one decision at a time: with two games running, each search still gets the whole GPU and worker pool,
        # and the per-decision fields (_tag, last_choice, last_policy) are never interleaved between games
        lock = getattr(self, "_move_lock", None)
        if lock is None:
            lock = self._move_lock = asyncio.Lock()
        t0 = time.monotonic()
        async with lock:
            order = await super().choose_move(battle)
        dt = time.monotonic() - t0  # whole decision as the opponent sees it: wait for other games' searches + our own
        self.move_total_s = getattr(self, "move_total_s", 0.0) + dt
        self.move_max_s = max(getattr(self, "move_max_s", 0.0), dt)
        self.move_n = getattr(self, "move_n", 0) + 1
        return order

    async def _choose(self, battle: Any) -> BattleOrder | None:
        import poke_engine

        legal = battle.valid_orders
        if len(legal) == 1:
            self.stats.forced += 1
            return legal[0]
        if not legal:
            return None
        try:
            trackers = self.opponent_model.update(battle)
            reveal = tokens.reveal_rows(_opp_facts(battle, trackers))
            _slots, states, built = self._sample_worlds(battle, trackers, self.worlds)
            hist = self._live_hist(battle, states[0], reveal)
            reveal2 = tokens.reveal_rows(self._history(battle.battle_tag).our_facts())
            names = slot_names(poke_engine.state_tokens_str(states[0], 1, reveal, True)[2])
            loop = asyncio.get_running_loop()
            probs, info = await loop.run_in_executor(
                _pool(), functools.partial(self.live.search, states, reveal, reveal2, hist, int(battle.turn)))
            orders: dict[str, BattleOrder] = {}
            weight: dict[str, float] = {}
            for s, name in enumerate(names):
                if not name or probs[s] <= 0:
                    continue
                order = built.choice_to_order(name, battle)
                if order is not None:
                    orders[name] = order
                    weight[name] = float(probs[s])
            policy = tp.normalise(list(weight), list(weight.values()))
            if not policy:
                raise RuntimeError(f"rollout search produced no legal option (candidates {info.get('cand')})")
            choice = tp.pick(policy, None if self.rollout_pick == "argmax" else 1.0, self._ro_rng)
            if self._ro_log:
                self._log_decision(battle, names, info, probs, choice)
            self.rollout_searches += 1
            self.rollout_evals += int(info["evals"])
            self.rollout_ms += float(info["ms"])
            best = names[int(info["prior"].argmax())]
            self.rollout_changed += int(choice != best)
            self.rollout_valid += float(np.mean(info["worlds"])) / len(states) if "worlds" in info else 1.0
            if self.rollout_searches % 25 == 0:
                n = self.rollout_searches
                logger.warning("rollout-search: %d searches, %.0f evals and %.0f ms each, changed %.3f, valid worlds %.3f, errors %d"
                               "; whole decision avg %.2f s, worst %.2f s",
                               n, self.rollout_evals / n, self.rollout_ms / n, self.rollout_changed / n,
                               self.rollout_valid / n, self.rollout_errors,
                               getattr(self, "move_total_s", 0.0) / max(getattr(self, "move_n", 0), 1), getattr(self, "move_max_s", 0.0))
        except Exception:
            self.rollout_errors += 1
            logger.exception("rollout search failed in %s; token policy argmax", battle.battle_tag)
            saved = (self.pick, self.temp)
            self.pick, self.temp = "argmax", None
            try:
                return await super()._choose(battle)
            finally:
                self.pick, self.temp = saved
        self._hist_done(battle, choice)
        self.last_choice = choice
        self.last_policy = policy
        self.stats.searched += 1
        return orders[choice]
