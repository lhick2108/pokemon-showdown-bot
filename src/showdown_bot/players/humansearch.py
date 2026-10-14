"""HumanSearchPlayer: a search bot whose final move is blended with the human action-policy model.

A competitive human-like test opponent for the exploiting bot (``opp_model=...``). It runs the normal
SearchPlayer search (so it is strong, not a pure no-search imitation) and then picks its move from

    p_final(a) ~ search(a) ** (1 - w) * human(a) ** w        (geometric blend, ``blend="geo"``)
    p_final(a) = (1 - w) * search(a) + w * human(a)          (``blend="mix"``)

over its legal actions. ``w`` (``human_weight``) = 0 is plain search, 1 is a pure human model with
the search ignored. ``pick`` = ``sample`` (default) or ``argmax``; ``temp`` sharpens/flattens the human
distribution first, ``elo`` conditions it. How often the final move differs from the search's top
move is kept in ``blend_decisions`` / ``blend_changed`` (``blend_change_rate``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from poke_env.player.battle_order import BattleOrder

from showdown_bot.players.search import SearchPlayer, load_human_model

logger = logging.getLogger(__name__)

PICKS = ("sample", "argmax")
BLENDS = ("geo", "mix")
FLOOR = 1e-3  # a probability never drops below this in the geometric blend


def _normalise(xs: list[float]) -> list[float]:
    total = sum(xs)
    if total <= 0:
        return [1.0 / len(xs)] * len(xs)
    return [x / total for x in xs]


def blend_policies(
    search: dict[str, float],
    human: dict[str, float],
    weight: float,
    *,
    blend: str = "geo",
    temp: float = 1.0,
    floor: float = FLOOR,
) -> dict[str, float]:
    """Normalised blend of two distributions over the choices in ``search``.

    Missing/zero entries are floored (geometric blend only) so one side cannot veto a move outright.
    """
    choices = list(search)
    if not choices:
        return {}
    s = _normalise([max(search[c], 0.0) for c in choices])
    h = _normalise([max(human.get(c, 0.0), 0.0) for c in choices])
    if temp != 1.0:
        h = _normalise([x ** (1.0 / temp) for x in h])
    if blend == "mix":
        p = [(1 - weight) * a + weight * b for a, b in zip(s, h)]
    else:
        p = [max(a, floor) ** (1 - weight) * max(b, floor) ** weight for a, b in zip(s, h)]
    return dict(zip(choices, _normalise(p)))


class HumanSearchPlayer(SearchPlayer):
    def __init__(
        self,
        *args: Any,
        model: str | Path | None = None,
        human_weight: float = 0.5,
        pick: str = "sample",
        blend: str = "geo",
        elo: str | int | None = None,
        temp: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if model is None:
            raise ValueError("HumanSearchPlayer needs model=<human-policy .npz>")
        if pick not in PICKS:
            raise ValueError(f"pick must be one of {PICKS}, got {pick!r}")
        if blend not in BLENDS:
            raise ValueError(f"blend must be one of {BLENDS}, got {blend!r}")
        if not 0 <= human_weight <= 1:
            raise ValueError(f"human_weight must be in [0, 1], got {human_weight}")
        if not temp > 0:
            raise ValueError(f"temp must be > 0, got {temp}")
        path = Path(model).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"human policy model not found: {path}")
        self.human = load_human_model(str(path))
        self.human_weight = float(human_weight)
        self.pick = pick
        self.blend = blend
        self.human_elo = elo
        self.temp = float(temp)
        self._pick_rng = random.Random(self._rng.random())
        self.blend_decisions = 0
        self.blend_changed = 0

    def _order_from_policy(
        self, policy: dict[str, float], built: Any, battle: Any, mode: str | None = None
    ) -> BattleOrder | None:
        if self.human_weight <= 0 or len(policy) < 2:
            return super()._order_from_policy(policy, built, battle, mode)
        orders: dict[str, BattleOrder] = {}
        for choice in policy:
            order = built.choice_to_order(choice, battle)
            if order is not None:
                orders[choice] = order
        legal_policy = {c: policy[c] for c in orders}
        if len(legal_policy) < 2:
            return super()._order_from_policy(policy, built, battle, mode)
        try:
            human = self.human.predict(built.to_string(), list(legal_policy), self.human_elo)
        except Exception:  # the human model must never cost a decision
            logger.exception("Human model failed in %s", battle.battle_tag)
            return super()._order_from_policy(policy, built, battle, mode)
        top = max(legal_policy, key=legal_policy.get)
        final = blend_policies(legal_policy, human, self.human_weight, blend=self.blend, temp=self.temp)
        if self.pick == "argmax":
            choice = max(final, key=final.get)
        else:
            choice = self._pick_rng.choices(list(final), weights=list(final.values()))[0]
        self.blend_decisions += 1
        self.blend_changed += choice != top
        self.last_choice = choice
        self.last_policy = final
        return orders[choice]

    @property
    def blend_change_rate(self) -> float:
        return self.blend_changed / self.blend_decisions if self.blend_decisions else 0.0

    def _battle_finished_callback(self, battle: Any) -> None:
        super()._battle_finished_callback(battle)
        logger.warning(
            "humansearch %s: final move differed from search top in %d/%d blended decisions (%.0f%%), w=%.2f",
            self.username, self.blend_changed, self.blend_decisions,
            100 * self.blend_change_rate, self.human_weight,
        )
