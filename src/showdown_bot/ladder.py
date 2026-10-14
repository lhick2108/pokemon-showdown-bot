"""Public-ladder helpers (credentials, per-decision time cap, game records, summary).

The runner lives in ``scripts/ladder.py``; see ``docs/ladder.md``. Credentials come
only from the environment (``SHOWDOWN_USERNAME`` / ``SHOWDOWN_PASSWORD``), optionally
read from a gitignored ``.env``; the password is never part of any repr, log line or
record written here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from showdown_bot import bench

logger = logging.getLogger(__name__)

USER_ENV = "SHOWDOWN_USERNAME"
PASS_ENV = "SHOWDOWN_PASSWORD"
DEFAULT_MOVE_CAP_S = 5.0


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def parse_env_text(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines: '#' comments, optional 'export', optional quotes."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, eq, value = line.partition("=")
        key = key.strip()
        if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()  # inline comment
        out[key] = value
    return out


def parse_env_file(path: Path) -> dict[str, str]:
    try:
        return parse_env_text(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str = field(repr=False)

    def __str__(self) -> str:
        return f"Credentials(username={self.username!r}, password=<hidden>)"

    def __reduce__(self):  # never pickled/copied with the secret by accident
        raise TypeError("Credentials must not be serialized")

    def account(self):
        from poke_env.ps_client import AccountConfiguration

        return AccountConfiguration(self.username, self.password)


def load_credentials(
    environ: Mapping[str, str] | None = None, env_file: Path | None = None
) -> Credentials:
    """Environment variables win over the .env file; both values are required."""
    environ = os.environ if environ is None else environ
    file_vars = parse_env_file(env_file) if env_file else {}

    def get(name: str) -> str:
        return (environ.get(name) or file_vars.get(name) or "").strip()

    user, password = get(USER_ENV), get(PASS_ENV)
    missing = [n for n, v in ((USER_ENV, user), (PASS_ENV, password)) if not v]
    if missing:
        raise ValueError(
            f"missing {' and '.join(missing)}: set the environment variable(s) or put them "
            "in the gitignored .env file (see docs/ladder.md)"
        )
    return Credentials(user, password)


class RedactFilter(logging.Filter):
    """Belt and braces: scrub the secret from any log record that somehow carries it."""

    def __init__(self, secrets_: Sequence[str]):
        super().__init__()
        self._secrets = [s for s in secrets_ if len(s) >= 4]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            text = record.getMessage()
        except Exception:
            return True
        if any(s in text for s in self._secrets):
            for s in self._secrets:
                text = text.replace(s, "<hidden>")
            record.msg, record.args = text, None
        return True


def install_redaction(creds: Credentials) -> RedactFilter:
    flt = RedactFilter([creds.password])
    root = logging.getLogger()
    root.addFilter(flt)
    for handler in root.handlers:
        handler.addFilter(flt)
    return flt


# ---------------------------------------------------------------------------
# Time cap
# ---------------------------------------------------------------------------

def heuristic_order(player: Any, battle: Any):
    """Cheap fallback: the strongest damaging move by base power x type effectiveness."""
    try:
        opp = battle.opponent_active_pokemon
        best, best_score = None, 0.0
        for move in battle.available_moves:
            score = float(move.base_power or 0)
            if opp is not None and move.type is not None:
                score *= opp.damage_multiplier(move)
            if score > best_score:
                best, best_score = move, score
        if best is not None:
            return player.create_order(best)
    except Exception:
        logger.exception("heuristic fallback failed; choosing at random")
    return player.choose_random_move(battle)


def with_ladder_hooks(
    cls: type,
    *,
    move_cap_s: float = DEFAULT_MOVE_CAP_S,
    on_finish: Callable[[Any, Any], None] | None = None,
    save_server_replay: bool = False,
) -> type:
    """Subclass of ``cls`` with a hard per-decision cap and a battle-finished hook."""

    class LadderPlayer(cls):  # type: ignore[valid-type, misc]
        cap_s = move_cap_s
        cap_hits = 0

        async def _handle_battle_message(self, split_messages):
            # keep every |request| (our full real sets) in order, before poke-env consumes it
            try:
                capture_requests(self.__dict__.setdefault("battle_requests", {}), split_messages,
                                 getattr(self, "battles", {}))
            except Exception:
                logger.exception("could not capture request")
            return await super()._handle_battle_message(split_messages)

        async def choose_move(self, battle):
            try:
                return await asyncio.wait_for(super().choose_move(battle), self.cap_s)
            except asyncio.TimeoutError:
                type(self).cap_hits += 1
                logger.warning(
                    "decision exceeded %.2fs cap in %s turn %s; heuristic move",
                    self.cap_s, battle.battle_tag, battle.turn,
                )
                return heuristic_order(self, battle)

        def _battle_finished_callback(self, battle):
            super()._battle_finished_callback(battle)
            if save_server_replay:
                try:
                    asyncio.ensure_future(
                        self.ps_client.send_message("/savereplay", battle.battle_tag)
                    )
                except Exception:
                    logger.exception("could not request server replay")
            if on_finish is not None:
                try:
                    on_finish(self, battle)
                except Exception:
                    logger.exception("could not record %s", battle.battle_tag)

    LadderPlayer.__name__ = f"Ladder{cls.__name__}"
    return LadderPlayer


# ---------------------------------------------------------------------------
# Records and summary
# ---------------------------------------------------------------------------

@dataclass
class LadderGame:
    battle_tag: str
    opponent: str
    result: str  # win | loss | tie
    turns: int
    rating_before: int | None = None
    rating_after: int | None = None
    opponent_rating: int | None = None
    log_file: str | None = None
    arm: str | None = None  # A/B arm name (scripts/ladder.py --arm)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


_RATING_RE = re.compile(
    r"^(?P<name>.+?)'s rating: (?P<old>\d+)\s*(?:&rarr;|->|→)\s*(?:<strong>)?(?P<new>\d+)"
)


def rating_after_from_log(replay_data: Sequence[Sequence[str]], username: str) -> int | None:
    """New rating from the server's ``|raw|NAME's rating: OLD &rarr; <strong>NEW</strong>`` line.

    (poke-env's own ``battle.rating`` reads the first four digits, i.e. the OLD rating.)
    """
    for event in replay_data or []:
        if len(event) >= 3 and event[1] == "raw":
            m = _RATING_RE.match(event[2].strip())
            if m and m.group("name").lower().replace("<strong>", "") == username.lower():
                return int(m.group("new"))
    return None


def parse_player_lines(text: str) -> list[dict]:
    """``|player|SIDE|NAME|AVATAR|RATING`` lines of a protocol log -> dicts.

    The rating field may be absent or empty (unrated); it is then None.
    """
    out = []
    for line in text.splitlines():
        if not line.startswith("|player|"):
            continue
        parts = line.split("|")  # ['', 'player', side, name, avatar, rating]
        if len(parts) < 4 or not parts[3]:
            continue
        out.append({
            "side": parts[2],
            "username": parts[3],
            "rating": _as_int(parts[5]) if len(parts) > 5 else None,
        })
    return out


def ratings_from_players(players: Sequence[Mapping], username: str) -> tuple[int | None, int | None, str | None]:
    """(our rating, opponent rating, opponent name) at battle start from |player| entries."""
    mine = theirs = opp_name = None
    for entry in players or []:
        name = str(entry.get("username", ""))
        rating = _as_int(entry.get("rating"))
        if name.lower().replace(" ", "") == username.lower().replace(" ", ""):
            mine = rating
        elif name:
            theirs, opp_name = rating, name
    return mine, theirs, opp_name


def game_from_battle(battle: Any, username: str) -> LadderGame:
    """Record of a finished poke-env battle (ratings only if the server sent them)."""
    from showdown_bot.recorder import result_from_battle

    players = getattr(battle, "_players", None) or []
    before, opp_rating, _ = ratings_from_players(players, username)
    if opp_rating is None:
        opp_rating = _as_int(getattr(battle, "opponent_rating", None))
    return LadderGame(
        battle_tag=battle.battle_tag,
        opponent=getattr(battle, "opponent_username", None) or "?",
        result=result_from_battle(battle),
        turns=int(battle.turn),
        rating_before=before,
        rating_after=rating_after_from_log(getattr(battle, "_replay_data", None), username),
        opponent_rating=opp_rating,
    )


def summarize(games: Sequence[LadderGame]) -> str:
    wins = sum(g.result == "win" for g in games)
    losses = sum(g.result == "loss" for g in games)
    ties = sum(g.result == "tie" for g in games)
    n = len(games)
    lines = [f"Ladder games: {n}  W-L-T {wins}-{losses}-{ties}"]
    if n:
        lo, hi = bench.wilson_interval(wins, n)
        lines.append(f"Win rate {wins / n:.1%}  Wilson 95% CI [{lo:.1%}, {hi:.1%}]")
        traj: list[int] = []
        for g in games:  # a game's "after" is also the next game's "before"
            for r in (g.rating_before, g.rating_after):
                if r is not None and (not traj or traj[-1] != r):
                    traj.append(r)
        if traj:
            lines.append(
                "Rating: " + " -> ".join(str(r) for r in traj)
                + f"  (net {traj[-1] - traj[0]:+d})"
            )
        else:
            lines.append("Rating: not reported by the server")
    return "\n".join(lines)


def append_game(path: Path, game: LadderGame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(game.to_dict()) + "\n")


def capture_requests(store: dict, split_messages: Sequence[Sequence[str]], battles: Mapping) -> None:
    """Remember each ``|request|`` JSON of a battle chunk as ``(position, raw_json)``.

    ``position`` is the number of protocol events poke-env had recorded when the request
    arrived, so the request can be re-inserted at the right place of the replay log.
    Only the request payload is kept (never other websocket traffic): it holds our team
    (moves, PP, item, ability, tera type, stats, HP) and the account name, no credentials.
    """
    if not split_messages or not split_messages[0] or not split_messages[0][0].startswith(">"):
        return
    tag = split_messages[0][0][1:]
    battle = battles.get(tag)
    position = len(getattr(battle, "_replay_data", None) or [])
    for msg in split_messages[1:]:
        if len(msg) >= 3 and msg[1] == "request" and msg[2].strip():
            json.loads(msg[2])  # must be valid JSON, else not ours to keep
            store.setdefault(tag, []).append((position, msg[2]))


def log_with_requests(events: Sequence[str], requests: Sequence[tuple[int, str]]) -> str:
    """Protocol events with ``|request|{json}`` lines interleaved at their recorded position."""
    out: list[str] = []
    pending = sorted(requests, key=lambda r: r[0])
    i = 0
    for index, event in enumerate(events):
        while i < len(pending) and pending[i][0] <= index:
            out.append("|request|" + pending[i][1])
            i += 1
        out.append(event)
    out.extend("|request|" + raw for _, raw in pending[i:])
    return "\n".join(out)


def safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "battle"


def max_decision_estimate_s(search_time_ms: int, worlds: int | None, parallelism: int = 1) -> float:
    """Rough worst-case wall time of one search (waves of per-world budgets)."""
    n = worlds if worlds is not None else 4 * parallelism
    waves = math.ceil(n / max(1, parallelism))
    return waves * search_time_ms / 1000.0


# ---------------------------------------------------------------------------
# Challenge mode (accept challenges from players instead of laddering)
# ---------------------------------------------------------------------------

def to_id(name: str) -> str:
    """Showdown userid: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


class ChallengePolicy:
    """Decides which incoming challenges to accept; everything else is rejected.

    Accept only ``battle_format`` challenges, optionally only from ``allow`` users, and
    only while no battle is running (one at a time). Repeats of the same challenge
    (the server reports it through both ``updatechallenges`` and a PM) are ignored for
    ``dedupe_s`` seconds, so a challenge is answered exactly once.
    """

    ACCEPT_GRACE_S = 30.0  # an accepted challenge counts as "busy" until its battle exists

    def __init__(self, battle_format: str = "gen9randombattle",
                 allow: Sequence[str] | None = None, *, dedupe_s: float = 10.0, clock=None):
        import time as _time

        self.battle_format = battle_format
        self.allow = {to_id(a) for a in allow} if allow else None
        self.dedupe_s = dedupe_s
        self._clock = clock or _time.monotonic
        self._seen: dict[tuple[str, str], float] = {}
        self._accepted_at: float | None = None
        self.last_activity = self._clock()
        self.counts = {"accepted": 0, "rejected": 0}

    def decide(self, user: str, fmt: str, busy: bool) -> tuple[str, str] | None:
        """``("accept"|"reject", reason)``, or None for a repeat already answered."""
        now = self._clock()
        key = (to_id(user), to_id(fmt))
        self._seen = {k: t for k, t in self._seen.items() if now - t < self.dedupe_s}
        if key in self._seen:
            return None
        self._seen[key] = now
        self.last_activity = now
        if to_id(fmt) != to_id(self.battle_format):
            verdict = ("reject", f"format {fmt or '?'} (only {self.battle_format})")
        elif self.allow is not None and key[0] not in self.allow:
            verdict = ("reject", "user not on the allowlist")
        elif busy or (self._accepted_at is not None
                      and now - self._accepted_at < self.ACCEPT_GRACE_S):
            verdict = ("reject", "a battle is already in progress")
        else:
            self._accepted_at = now
            verdict = ("accept", "ok")
        self.counts["accepted" if verdict[0] == "accept" else "rejected"] += 1
        return verdict

    def battle_started(self) -> None:
        self._accepted_at = None

    def touch(self) -> None:
        self.last_activity = self._clock()

    def idle_seconds(self) -> float:
        return self._clock() - self.last_activity


def with_challenge_policy(
    cls: type, policy: ChallengePolicy,
    on_decision: Callable[[str, str, str, str], None] | None = None,
) -> type:
    """Subclass of ``cls`` answering every challenge itself (``/accept`` or ``/reject``).

    poke-env's own queue silently drops other formats and never rejects, so both of its
    challenge entry points are replaced; ``accept_challenges`` is not used.
    """

    class ChallengePlayer(cls):  # type: ignore[valid-type, misc]
        async def _answer(self, user: str, fmt: str) -> None:
            if to_id(user) == to_id(self.username):
                return
            busy = any(not b.finished for b in self.battles.values())
            verdict = policy.decide(user, fmt, busy)
            if verdict is None:
                return
            action, reason = verdict
            if on_decision is not None:
                try:
                    on_decision(user, fmt, action, reason)
                except Exception:
                    logger.exception("challenge callback failed")
            if action == "accept":
                await self.ps_client.accept_challenge(to_id(user), self.next_team)
            else:
                await self.ps_client.send_message(f"/reject {to_id(user)}")

        async def _update_challenges(self, split_message):
            data = json.loads(split_message[2]) if len(split_message) > 2 else {}
            for user, fmt in (data.get("challengesFrom") or {}).items():
                await self._answer(user, fmt)

        async def _handle_challenge_request(self, split_message):
            # |pm|SENDER|RECEIVER|/challenge FORMAT|...
            if len(split_message) >= 5 and split_message[4].startswith("/challenge"):
                fmt = split_message[4][len("/challenge"):].strip().split("|")[0]
                if len(split_message) >= 6 and split_message[5]:
                    fmt = split_message[5]
                if fmt:  # an empty format is a withdrawn challenge
                    await self._answer(split_message[2].strip(), fmt)

        def _battle_finished_callback(self, battle):
            super()._battle_finished_callback(battle)
            policy.touch()

        async def _create_battle(self, split_message):
            battle = await super()._create_battle(split_message)
            policy.battle_started()
            return battle

    ChallengePlayer.__name__ = f"Challenge{cls.__name__}"
    return ChallengePlayer


