"""Interleaved A/B testing on one ladder account (scripts/ladder.py --arm).

One login, one connection, one player object. Each NEW battle is assigned an arm (a search
configuration: evaluator path, leaf scale, ...) before it starts; the player swaps
``self.config`` to that arm's SearchConfig on every decision of that battle. This is safe
because the A/B mode runs one battle at a time (config is only read per decision).
Arm specs use the league spec syntax: ``NAME=search-net:evaluator=X.pev:scale=1.5``, or
the short form ``NAME=X.pev:scale=1.5`` (type search-net is implied by a .pev path).
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from showdown_bot import league

_SHORT = re.compile(r"^(?P<name>[^=:]+)=(?P<path>[^=]+?\.pev)(?P<rest>:.*)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Arm:
    name: str
    spec: league.BotSpec
    config: Any  # SearchConfig

    @property
    def evaluator(self) -> str | None:
        return self.config.evaluator

    @property
    def scale(self) -> float:
        return self.config.evaluator_scale


def arm_search_config(spec: league.BotSpec, default_time_ms: int = 100):
    """SearchConfig for an arm, built from defaults (never from another arm's config)."""
    from showdown_bot.players.search import SearchConfig

    if spec.type not in league.SEARCH_BOTS:
        raise ValueError(f"arm {spec.label}: only search / search-net arms are supported")
    cfg = SearchConfig(
        search_time_ms=spec.param("time_ms", default_time_ms),
        worlds=spec.param("worlds"),
        **league._quality_kwargs(spec),
    )
    fields = league.search_config_fields(spec)
    fields.pop("search_time_ms", None)
    fields.pop("worlds", None)
    cfg = dataclasses.replace(cfg, **{k: v for k, v in fields.items() if hasattr(cfg, k)})
    for attr in ("evaluator", "evaluator2"):
        path = getattr(cfg, attr)
        if path is not None:
            p = Path(path)
            if not p.is_file():
                raise ValueError(f"arm {spec.label}: {attr} file not found: {path}")
            cfg = dataclasses.replace(cfg, **{attr: str(p.resolve())})
    return cfg


def parse_arm(text: str, default_time_ms: int = 100) -> Arm:
    raw = text.strip()
    m = _SHORT.match(raw)
    if m:  # NAME=path.pev[:k=v...] -> NAME=search-net:evaluator=path.pev[:k=v...]
        raw = f"{m['name']}=search-net:evaluator={m['path']}{m['rest'] or ''}"
    spec = league.parse_spec(raw)
    return Arm(spec.label, spec, arm_search_config(spec, default_time_ms))


def parse_arms(texts: Sequence[str], default_time_ms: int = 100) -> list[Arm]:
    arms = [parse_arm(t, default_time_ms) for t in texts]
    if len(arms) < 2:
        raise ValueError("A/B testing needs at least two --arm specs")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique, got {names}")
    return arms


class ArmScheduler:
    """Balanced random arm sequence: shuffled blocks holding each arm twice, so after any
    prefix the counts differ by at most 2 and the order is unpredictable."""

    def __init__(self, names: Sequence[str], seed: int = 0, per_block: int = 2):
        self.names = list(names)
        self.rng = random.Random(seed)
        self.per_block = per_block
        self._block: list[str] = []

    def next(self) -> str:
        if not self._block:
            self._block = self.names * self.per_block
            self.rng.shuffle(self._block)
        return self._block.pop()


class ArmController:
    """Binds each battle tag to one arm, fixed at the battle's first decision."""

    def __init__(self, arms: Sequence[Arm], seed: int = 0):
        self.arms = {a.name: a for a in arms}
        self.scheduler = ArmScheduler([a.name for a in arms], seed)
        self.pending: str | None = None  # drawn for the next battle, not yet started
        self.bound: dict[str, str] = {}  # battle tag -> arm name

    def prepare_next(self) -> Arm:
        """Draw the arm of the next battle (idempotent until that battle starts)."""
        if self.pending is None:
            self.pending = self.scheduler.next()
        return self.arms[self.pending]

    def arm_for(self, tag: str) -> Arm:
        if tag not in self.bound:
            self.bound[tag] = self.prepare_next().name
            self.pending = None
        return self.arms[self.bound[tag]]

    def get(self, tag: str) -> Arm | None:
        name = self.bound.get(tag)
        return self.arms[name] if name else None


def with_arm_switch(cls: type, controller: ArmController) -> type:
    """Subclass whose every decision runs under the battle's arm config."""

    class ABPlayer(cls):  # type: ignore[valid-type, misc]
        async def choose_move(self, battle):
            arm = controller.arm_for(battle.battle_tag)
            self.config = arm.config
            self.arm_name = arm.name
            return await super().choose_move(battle)

    ABPlayer.__name__ = f"AB{cls.__name__}"
    return ABPlayer


def ab_record(game, arm: Arm | None, username: str, now: float | None = None) -> dict:
    """One ab.jsonl row from a finished LadderGame (ladder.game_from_battle)."""
    now = time.time() if now is None else now
    return {
        "battle_tag": game.battle_tag,
        "arm": arm.name if arm else None,
        "evaluator": arm.evaluator if arm else None,
        "scale": arm.scale if arm else None,
        "spec": arm.spec.text if arm else None,
        "username": username,
        "rating_before": game.rating_before,
        "rating_after": game.rating_after,
        "opponent": game.opponent,
        "opponent_rating": game.opponent_rating,
        "result": game.result,
        "turns": game.turns,
        "timestamp": now,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
    }


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def update_record_rating(path: Path, battle_tag: str, rating_after: int) -> bool:
    """Set ``rating_after`` on the ab.jsonl row of ``battle_tag`` (atomic rewrite)."""
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    hit = False
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("battle_tag") == battle_tag and row.get("rating_after") is None:
            row["rating_after"] = rating_after
            lines[i] = json.dumps(row)
            hit = True
    if hit:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return hit
