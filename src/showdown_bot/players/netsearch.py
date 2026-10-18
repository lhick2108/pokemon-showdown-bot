"""NetSearchPlayer: Jaxcalibur-style net-guided search with the token transformer (bead 589.68.5).

Each turn it samples ``worlds`` opposing teams (the search bots' sampler), runs
``poke_engine.NetSearch`` (one decoupled-PUCT tree per world, every expansion evaluated by the
transformer: policy priors for both sides plus a value, batched on the GPU) for ``time_ms``, sums
the root visits of our options over the worlds and plays the argmax (``mode=argmax``) or samples
from the visit shares (``mode=mixed``). If the search raises, the token-policy argmax is played
instead and ``net_errors`` counts it. Spec (docs/net-search.md)::

    net-search:model=models\\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:max_evals=20000:batch=128
               :c_puct=2:depth=4:fpu=0:vavg=0:scrub=1:pipe=1:views=2
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from poke_env.player.battle_order import BattleOrder

from showdown_bot import net_search as ns
from showdown_bot import token_policy as tp
from showdown_bot import tokens
from showdown_bot.players.search import _opp_facts
from showdown_bot.players.tokenpolicy import TokenPolicyPlayer

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

_POOL: ThreadPoolExecutor | None = None


def _pool() -> ThreadPoolExecutor:
    """Search threads of this process: the tree work releases the GIL, so several games overlap."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="netsearch")
    return _POOL


class NetSearchPlayer(TokenPolicyPlayer):
    def __init__(
        self,
        *args: Any,
        model: str | Path | None = None,
        device: str = "auto",
        worlds: int = ns.DEFAULTS["worlds"],
        time_ms: int = ns.DEFAULTS["time_ms"],
        max_evals: int = ns.DEFAULTS["max_evals"],
        batch: int = ns.DEFAULTS["batch"],
        c_puct: float = ns.DEFAULTS["c_puct"],
        depth: int = ns.DEFAULTS["depth"],
        fpu: float = ns.DEFAULTS["fpu"],
        vavg: int = ns.DEFAULTS["vavg"],
        scrub: int = ns.DEFAULTS["scrub"],
        pipe: int = ns.DEFAULTS["pipe"],
        views: int = ns.DEFAULTS["views"],
        mode: str = "argmax",
        seed: int = 0,
        **kwargs: Any,
    ):
        super().__init__(*args, model=model, device=device, seed=seed, **kwargs)
        if mode not in ns.MODES:
            raise ValueError(f"mode must be one of {ns.MODES}, got {mode!r}")
        if worlds < 1 or time_ms < 1 or max_evals < 1 or batch < 1 or depth < 1:
            raise ValueError("worlds, time_ms, max_evals, batch and depth must be >= 1")
        self.mode = mode
        self.worlds = worlds
        self.searcher = ns.NetSearcher(
            self.policy_net, worlds=worlds, time_ms=time_ms, max_evals=max_evals, batch=batch,
            c_puct=c_puct, depth=depth, fpu=fpu, vavg=bool(vavg), scrub=bool(scrub), pipe=pipe, views=views, seed=seed,
        )
        self.net_errors = 0
        self.net_searches = 0
        self.net_evals = 0
        self.net_ms = 0.0
        self.net_batches = 0

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
            _slots, states, built = self._sample_worlds(battle, trackers, self.worlds)
            loop = asyncio.get_running_loop()
            policy, info = await loop.run_in_executor(_pool(), self.searcher.search, states, reveal)
            orders: dict[str, BattleOrder] = {}
            for choice in policy:
                order = built.choice_to_order(choice, battle)
                if order is not None:
                    orders[choice] = order
            policy = tp.normalise(list(orders), [policy[c] for c in orders])
            if not policy:
                raise RuntimeError("net search produced no legal option")
            choice = tp.pick(policy, None if self.mode == "argmax" else 1.0, self._pick_rng)
            drive = info["drive"]
            self.net_searches += 1
            self.net_evals += drive.evals
            self.net_batches += drive.batches
            self.net_ms += drive.wall_ms
        except Exception:
            self.net_errors += 1
            logger.exception("net search failed in %s; token policy argmax", battle.battle_tag)
            return await self._fallback_policy(battle)
        self.last_choice = choice
        self.last_policy = policy
        self.stats.searched += 1
        return orders[choice]

    async def _fallback_policy(self, battle: Any) -> BattleOrder | None:
        """The raw token-policy argmax (its own error handling falls back to a random move)."""
        saved = (self.pick, self.temp)
        self.pick, self.temp = "argmax", None
        try:
            return await super()._choose(battle)
        finally:
            self.pick, self.temp = saved
