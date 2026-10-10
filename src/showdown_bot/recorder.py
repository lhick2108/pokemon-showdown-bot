"""Self-play game recorder: versioned, gzip-compressed JSONL, one file per game side.

See ``docs/selfplay-data.md`` for the schema. A recorder buffers a game's
records in memory and writes ``<run_dir>/<battle-tag>__<username>.jsonl.gz``
atomically when the game ends (``end_game``), so a crashed game leaves no
partial file. Nothing here runs unless a ``SearchPlayer`` was given a recorder.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RECORD_DIR_ENV = "SHOWDOWN_RECORD_DIR"

RESULT_WIN = "win"
RESULT_LOSS = "loss"
RESULT_TIE = "tie"
RESULTS = (RESULT_WIN, RESULT_LOSS, RESULT_TIE)


def feature_version() -> int | None:
    try:
        import poke_engine

        return int(poke_engine.FEATURE_VERSION)
    except Exception:  # engine not built: still allow recording
        return None


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "game"


def result_from_battle(battle: Any) -> str:
    """win/loss/tie from the perspective of the player owning ``battle``."""
    won = getattr(battle, "won", None)
    if won is True:
        return RESULT_WIN
    if won is False:
        return RESULT_LOSS
    return RESULT_TIE


def world_record(
    state: str, outcome: tuple[int, list[tuple[str, int, float]]]
) -> dict[str, Any]:
    """One searched world: its State string and the MCTS root statistics."""
    total, options = outcome
    score_sum = sum(score for _, _, score in options)
    return {
        "state": state,
        "total_visits": int(total),
        "avg_score": (score_sum / total) if total else None,
        "moves": [
            {"choice": choice, "visits": int(visits), "score": float(score)}
            for choice, visits, score in options
        ],
    }


class GameRecorder:
    """Collects decision records per battle and writes one gzip JSONL file per game."""

    def __init__(self, run_dir: str | os.PathLike[str], *, truth: bool = True):
        self.run_dir = Path(run_dir)
        self.truth = truth
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._games: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self.games_written = 0
        self.decisions_written = 0
        self.worlds_written = 0
        self.truth_written = 0
        self.truth_missing = 0
        self.bytes_written = 0

    def path_for(self, battle_tag: str, username: str) -> Path:
        return self.run_dir / f"{_safe_name(battle_tag)}__{_safe_name(username)}.jsonl.gz"

    def start_game(
        self,
        battle_tag: str,
        username: str,
        *,
        role: str | None = None,
        battle_format: str | None = None,
        config: dict[str, Any] | None = None,
        player: str = "search",
    ) -> None:
        header = {
            "type": "game",
            "schema_version": SCHEMA_VERSION,
            "feature_version": feature_version(),
            "battle_tag": battle_tag,
            "username": username,
            "role": role,  # "p1" / "p2": the server seat our player occupies
            "format": battle_format,
            "player": player,
            "config": config or {},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self._lock:
            self._games[(battle_tag, username)] = [header]

    def has_game(self, battle_tag: str, username: str) -> bool:
        return (battle_tag, username) in self._games

    def add_decision(self, battle_tag: str, username: str, record: dict[str, Any]) -> None:
        record = {"type": "decision", **record}
        with self._lock:
            lines = self._games.get((battle_tag, username))
            if lines is None:
                return
            lines.append(record)
            self.decisions_written += 1
            self.worlds_written += len(record.get("worlds", ()))
            if record.get("truth"):
                self.truth_written += 1
            elif self.truth:
                self.truth_missing += 1

    def end_game(
        self, battle_tag: str, username: str, result: str, turns: int | None = None
    ) -> Path | None:
        """Write the finished game; returns its path, or None if nothing was recorded."""
        with self._lock:
            lines = self._games.pop((battle_tag, username), None)
        if lines is None or len(lines) < 2:
            return None  # no searched decision: nothing worth keeping
        lines.append({"type": "result", "result": result, "turns": turns})
        path = self.path_for(battle_tag, username)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = "".join(json.dumps(line, separators=(",", ":")) + "\n" for line in lines)
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            handle.write(payload)
        os.replace(tmp, path)
        with self._lock:
            self.games_written += 1
            self.bytes_written += path.stat().st_size
        return path


def from_env() -> GameRecorder | None:
    """A recorder for ``$SHOWDOWN_RECORD_DIR``, or None when it is unset/empty."""
    raw = os.environ.get(RECORD_DIR_ENV, "").strip()
    return GameRecorder(raw) if raw else None


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_game(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse one game file into ``{"header", "decisions", "result"}``."""
    header: dict[str, Any] | None = None
    decisions: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            kind = record.get("type")
            if kind == "game":
                header = record
            elif kind == "decision":
                decisions.append(record)
            elif kind == "result":
                result = record
    if header is None:
        raise ValueError(f"{path}: missing game header")
    if header.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {header.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return {"header": header, "decisions": decisions, "result": result}


def iter_game_files(root: str | os.PathLike[str]) -> Iterator[Path]:
    root = Path(root)
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*.jsonl.gz"))


# ---------------------------------------------------------------------------
# Ground truth (self-play, both players in this process)
# ---------------------------------------------------------------------------

_PLAYERS: dict[str, Any] = {}


def _id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def register_player(player: Any) -> None:
    """Make ``player`` findable as a peer by username (for truth records)."""
    _PLAYERS[_id(player.username)] = player


def peer_battle(battle: Any) -> Any | None:
    """The opposing in-process SearchPlayer's view of the same battle, if any."""
    peer = _PLAYERS.get(_id(getattr(battle, "opponent_username", "") or ""))
    if peer is None:
        return None
    return getattr(peer, "_battles", {}).get(battle.battle_tag)


def build_truth_state(battle: Any, other: Any) -> str | None:
    """The real combined State at ``battle``'s decision, with no sampling.

    Our side comes from ``battle`` as usual; the opposing team is taken exactly
    from ``other`` (the opponent player's own battle, which knows every set).
    Public information (HP, boosts, field, conditions) still comes from
    ``battle``. Returns None if the two views are not at the same turn.
    """
    from showdown_bot import engine_state
    from showdown_bot.sets import SampledSet, to_id

    if other is None or other.turn != battle.turn:
        return None
    active = battle.opponent_active_pokemon
    by_species = {to_id(m.species): m for m in battle.opponent_team.values()}
    entries = []
    for identifier, bmon in other.team.items():
        seen = battle.opponent_team.get(identifier) or by_species.get(to_id(bmon.species))
        tera = getattr(bmon.tera_type, "name", None) or bmon.tera_type
        sampled = SampledSet(
            species=bmon.species,
            level=int(bmon.level or 100),
            role="",
            moves=tuple(bmon.moves.keys()),
            ability=bmon.ability or bmon.base_ability,
            item=(bmon.item if bmon.item and bmon.item != "unknown_item" else ""),
            tera_type=str(tera).lower() if tera else None,
        )
        slot = engine_state.OpponentSlot(to_id(bmon.species), sampled, seen)
        entries.append((0 if (seen is not None and seen is active) else 1, slot))
    entries.sort(key=lambda item: item[0])  # stable: active first
    slots = [slot for _, slot in entries][: engine_state.TEAM_SIZE]
    return engine_state.build_state(battle, slots).to_string()
