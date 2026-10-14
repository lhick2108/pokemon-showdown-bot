"""Bot league: round-robin tournament on the local server with Bradley-Terry ratings.

    python -m showdown_bot.league --bots random,maxpower,heuristic,search,foulplay \
        --games-per-pair 100 --out data/league/run1
    python -m showdown_bot.league --rate-only data/league/run1

Outputs (see docs/league.md): ``games.jsonl`` (one line per finished game, the
input of the ratings), ``replays/`` (poke-env replay HTML), ``records/`` (the
search players' decision records, via showdown_bot.recorder), ``results.md`` and
``results.json``. Foul Play runs as a separate process (never imported) and
accepts challenges, so it is always seated as p2 against our bots.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import itertools
import json
import logging
import math
import random
import re
import secrets
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from poke_env import to_id_str
from poke_env.player import Player
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench, compute
from showdown_bot import teams as teams_module
from showdown_bot.bench import FOULPLAY_NAME

logger = logging.getLogger("showdown_bot.league")

SCHEMA_VERSION = 1
TIE = "tie"
ALIASES = {"heuristic": "heuristics", "fp": FOULPLAY_NAME}
SEARCH_BOTS = ("search", "search-net")
PRIOR_WEIGHT = 0.5  # virtual tie games added between every pair of bots


# ---------------------------------------------------------------------------
# Bot specs: label=type[:key=value[:key=value...]]
# ---------------------------------------------------------------------------

SPEC_KEYS = {
    "search": {"evaluator": str, "scale": float, "time_ms": int, "worlds": int,
               "selection": str, "gamma": float, "mode": str,
               "tendency": str, "tkappa": float, "tdelta": float,
               "belief": int, "btau": float, "bfloor": float,
                "escalate": int, "escalate_gap": float, "escalate_factor": float,
                "escalate_steps": int, "escalate_cap_ms": int, "prune_noop": int, "noop_switches": int,
                "tera_margin": float, "ev2": str, "scale2": float},
    "search-net": {"evaluator": str, "scale": float, "time_ms": int, "worlds": int,
                   "selection": str, "gamma": float, "mode": str,
               "tendency": str, "tkappa": float, "tdelta": float,
               "belief": int, "btau": float, "bfloor": float,
                "escalate": int, "escalate_gap": float, "escalate_factor": float,
                "escalate_steps": int, "escalate_cap_ms": int, "prune_noop": int, "noop_switches": int,
                "tera_margin": float, "ev2": str, "scale2": float},
}


@dataclass(frozen=True)
class BotSpec:
    """A league participant: ``label`` is its identity in ratings and games.jsonl."""

    label: str
    type: str
    params: tuple[tuple[str, object], ...] = ()
    text: str = ""

    def param(self, key: str, default=None):
        return dict(self.params).get(key, default)



def _init_params(cls: type) -> set[str]:
    """Constructor keywords of ``cls`` or any base (subclasses often forward **kwargs)."""
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None:
            names.update(inspect.signature(init).parameters)
    return names

def _split_spec_params(rest: str) -> list[str]:
    """Split on ':' but keep Windows drive letters (``C:\\x``) inside a value."""
    parts: list[str] = []
    for seg in rest.split(":"):
        if parts and "=" not in seg:
            parts[-1] += ":" + seg
        else:
            parts.append(seg)
    return parts


def parse_spec(text: str) -> BotSpec:
    raw = text.strip()
    if not raw:
        raise ValueError("empty bot spec")
    head, _, rest = raw.partition(":")
    label, eq, type_name = head.partition("=")
    if not eq:
        label, type_name = head, head
    if "=" in type_name:
        raise ValueError(f"malformed bot spec {raw!r}")
    type_name = ALIASES.get(type_name.strip(), type_name.strip())
    label = ALIASES.get(label.strip(), label.strip())
    if not label or not re.fullmatch(r"[A-Za-z0-9_.+-]+", label):
        raise ValueError(f"bad bot label {label!r} in {raw!r}")
    if type_name != FOULPLAY_NAME:
        bench.get_player_class(type_name)  # ValueError for unknown types
    params: list[tuple[str, object]] = []
    if rest:
        allowed = SPEC_KEYS.get(type_name, {})
        for item in _split_spec_params(rest):
            key, eq, value = item.partition("=")
            key = key.strip()
            if not eq or not value:
                raise ValueError(f"malformed parameter {item!r} in bot spec {raw!r}")
            if key not in allowed:
                supported = ", ".join(sorted(allowed)) or "none"
                raise ValueError(
                    f"unsupported key {key!r} for {type_name} in {raw!r} (supported: {supported})"
                )
            if any(k == key for k, _ in params):
                raise ValueError(f"duplicate key {key!r} in bot spec {raw!r}")
            try:
                parsed = allowed[key](value)
            except ValueError:
                raise ValueError(f"bad value {value!r} for {key!r} in bot spec {raw!r}") from None
            if key in ("escalate", "prune_noop", "noop_switches") and parsed not in (0, 1):
                raise ValueError(f"{key} must be 0 or 1 in bot spec {raw!r}")
            if key in ("escalate_gap", "tera_margin") and not (math.isfinite(parsed) and parsed >= 0):
                raise ValueError(f"{key} must be >= 0 in bot spec {raw!r}")
            if key == "escalate_factor" and not (math.isfinite(parsed) and parsed >= 1):
                raise ValueError(f"escalate_factor must be >= 1 in bot spec {raw!r}")
            if key in ("escalate_steps", "escalate_cap_ms") and parsed < 1:
                raise ValueError(f"{key} must be >= 1 in bot spec {raw!r}")
            if key in ("time_ms", "worlds") and parsed <= 0:
                raise ValueError(f"{key} must be positive in bot spec {raw!r}")
            if key in ("scale", "scale2") and not (math.isfinite(parsed) and parsed > 0):
                raise ValueError(f"{key} must be positive in bot spec {raw!r}")
            if key == "selection" and parsed.lower() not in ("ucb1", "exp3", "rm"):
                raise ValueError(f"selection must be ucb1, exp3 or rm in bot spec {raw!r}")
            if key == "gamma" and not (math.isfinite(parsed) and 0 < parsed <= 1):
                raise ValueError(f"gamma must be in (0, 1] in bot spec {raw!r}")
            if key == "mode" and parsed.lower() not in ("mixed", "argmax", "nash"):
                raise ValueError(f"mode must be mixed, argmax or nash in bot spec {raw!r}")
            if key == "tendency" and parsed.lower() not in ("on", "off"):
                raise ValueError(f"tendency must be on or off in bot spec {raw!r}")
            if key == "belief" and parsed not in (0, 1):
                raise ValueError(f"belief must be 0 or 1 in bot spec {raw!r}")
            if key == "btau" and not (math.isfinite(parsed) and parsed >= 0):
                raise ValueError(f"btau must be >= 0 in bot spec {raw!r}")
            if key == "bfloor" and not (math.isfinite(parsed) and 0 <= parsed < 1):
                raise ValueError(f"bfloor must be in [0, 1) in bot spec {raw!r}")
            if key in ("tkappa", "tdelta") and not (math.isfinite(parsed) and parsed >= 0):
                raise ValueError(f"{key} must be >= 0 in bot spec {raw!r}")
            params.append((key, parsed.lower() if key in ("selection", "mode", "tendency") else parsed))
    keys = dict(params)
    if "scale2" in keys and "ev2" not in keys:
        raise ValueError(f"scale2 needs ev2=<path.pev> in bot spec {raw!r}")
    if "ev2" in keys:
        if "evaluator" not in keys and not os.environ.get("SHOWDOWN_SEARCH_EVALUATOR"):
            raise ValueError(f"ev2 needs a primary evaluator=<path.pev> in bot spec {raw!r}")
        if keys.get("worlds", 2) < 2:
            raise ValueError(f"the ev2 ensemble needs worlds >= 2 in bot spec {raw!r}")
    if type_name == "search-net" and not any(k == "evaluator" for k, _ in params) \
            and not os.environ.get("SHOWDOWN_SEARCH_EVALUATOR"):
        raise ValueError(f"search-net spec {raw!r} needs evaluator=<path.pev>")
    return BotSpec(label, type_name, tuple(params), raw)


# ---------------------------------------------------------------------------
# Game records (games.jsonl)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Game:
    """One finished battle between two bots; ``winner`` is a bot name or ``tie``."""

    battle_tag: str
    p1: str
    p2: str
    winner: str
    turns: int
    seed: int | None = None
    p1_user: str = ""
    p2_user: str = ""
    format: str = "gen9randombattle"
    specs: dict = field(default_factory=dict)  # label -> spec text
    # fixed-team designs (showdown_bot.teams): mode, team pair index within the matchup (the two
    # games of a pair share it) and the generated team ids seated as (p1, p2)
    team_mode: str | None = None
    team_pair: int | None = None
    team_ids: tuple[int, int] | None = None

    def score_for(self, bot: str) -> float:
        if self.winner == TIE:
            return 0.5
        return 1.0 if self.winner == bot else 0.0

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "battle_tag": self.battle_tag,
            "p1": self.p1,
            "p2": self.p2,
            "p1_user": self.p1_user,
            "p2_user": self.p2_user,
            "bots": [self.p1, self.p2],
            "winner": self.winner,
            "turns": self.turns,
            "seed": self.seed,
            "format": self.format,
            "specs": self.specs,
            **({"team_mode": self.team_mode, "team_pair": self.team_pair,
                "team_ids": list(self.team_ids) if self.team_ids else None}
               if self.team_mode else {}),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Game:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported games.jsonl schema_version: {d.get('schema_version')}")
        winner = d["winner"]
        if winner not in (TIE, d["p1"], d["p2"]):
            raise ValueError(f"winner {winner!r} is neither tie nor a participant")
        return cls(
            battle_tag=d["battle_tag"], p1=d["p1"], p2=d["p2"], winner=winner,
            turns=int(d["turns"]), seed=d.get("seed"), p1_user=d.get("p1_user", ""),
            p2_user=d.get("p2_user", ""), format=d.get("format", "gen9randombattle"),
            specs=dict(d.get("specs") or {}), team_mode=d.get("team_mode"),
            team_pair=d.get("team_pair"),
            team_ids=tuple(d["team_ids"]) if d.get("team_ids") else None,
        )


def write_games(path: Path, games: Iterable[Game]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for g in games:
            fh.write(json.dumps(g.to_dict()) + "\n")


def append_game(path: Path, game: Game) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(game.to_dict()) + "\n")


def read_games(path: Path) -> list[Game]:
    games = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                games.append(Game.from_dict(json.loads(line)))
    return games


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def roundrobin_pairs(bots: Sequence[str]) -> list[tuple[str, str]]:
    return list(itertools.combinations(bots, 2))


def seat_schedule(a: str, b: str, n: int, chunk: int = 1) -> list[tuple[str, str, int]]:
    """Chunks ``(p1, p2, games)`` covering n games, swapping seats each chunk.

    ``p1`` is the challenger (poke-env seats the challenger as p1).
    """
    out = []
    left, i = n, 0
    while left > 0:
        k = min(chunk, left)
        out.append((a, b, k) if i % 2 == 0 else (b, a, k))
        left -= k
        i += 1
    return out


# ---------------------------------------------------------------------------
# Ratings: Bradley-Terry (MM algorithm) with bootstrap CIs
# ---------------------------------------------------------------------------


def pair_tables(games: Sequence[Game], bots: Sequence[str]):
    """Return (score, count): score[i][j] = points of i vs j (tie = 0.5), count[i][j] = games."""
    idx = {b: i for i, b in enumerate(bots)}
    k = len(bots)
    score = [[0.0] * k for _ in range(k)]
    count = [[0.0] * k for _ in range(k)]
    for g in games:
        i, j = idx[g.p1], idx[g.p2]
        s = g.score_for(g.p1)
        score[i][j] += s
        score[j][i] += 1.0 - s
        count[i][j] += 1
        count[j][i] += 1
    return score, count


def fit_bt(score, count, *, prior: float = PRIOR_WEIGHT, iters: int = 1000, tol: float = 1e-10):
    """Bradley-Terry strengths by Hunter's MM, geometric mean normalised to 1.

    ``prior`` virtual tie games are added between every pair so undefeated or
    winless bots stay finite. Ties are already split 0.5/0.5 in ``score``.
    """
    k = len(score)
    if k < 2:
        return [1.0] * k
    p = [1.0] * k
    for _ in range(iters):
        new = []
        for i in range(k):
            w = prior / 2 * (k - 1)
            den = 0.0
            for j in range(k):
                if j == i:
                    continue
                w += score[i][j]
                den += (count[i][j] + prior) / (p[i] + p[j])
            new.append(w / den)
        gm = math.exp(sum(math.log(x) for x in new) / k)
        new = [x / gm for x in new]
        delta = max(abs(math.log(a / b)) for a, b in zip(new, p))
        p = new
        if delta < tol:
            break
    return p


def to_elo(strengths: Sequence[float], bots: Sequence[str], anchor: str | None = None):
    """400*log10(p), mean 0 (strengths are normalised) or ``anchor`` bot = 0."""
    elo = [400.0 * math.log10(x) for x in strengths]
    if anchor is not None:
        off = elo[list(bots).index(anchor)]
        elo = [e - off for e in elo]
    return elo


def rate(games: Sequence[Game], bots: Sequence[str] | None = None, *, anchor: str | None = None):
    bots = list(bots) if bots is not None else sorted({b for g in games for b in (g.p1, g.p2)})
    score, count = pair_tables(games, bots)
    return bots, to_elo(fit_bt(score, count), bots, anchor)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def bootstrap_elo(
    games: Sequence[Game],
    bots: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    anchor: str | None = None,
    level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs from resampling the game list with replacement."""
    rng = random.Random(seed)
    k = len(bots)
    samples: list[list[float]] = [[] for _ in range(k)]
    games = list(games)
    n = len(games)
    if n == 0:
        return {b: (float("nan"), float("nan")) for b in bots}
    for _ in range(n_boot):
        resample = [games[rng.randrange(n)] for _ in range(n)]
        score, count = pair_tables(resample, bots)
        elo = to_elo(fit_bt(score, count), bots, anchor)
        for i, e in enumerate(elo):
            samples[i].append(e)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for b, s in zip(bots, samples):
        s.sort()
        out[b] = (_quantile(s, lo_q), _quantile(s, hi_q))
    return out


def sequential_elo(games: Sequence[Game], bots: Sequence[str], k_factor: float = 16.0):
    """Plain online Elo in game order (secondary; order dependent), mean-centred."""
    r = {b: 0.0 for b in bots}
    for g in games:
        e1 = 1 / (1 + 10 ** ((r[g.p2] - r[g.p1]) / 400))
        s1 = g.score_for(g.p1)
        r[g.p1] += k_factor * (s1 - e1)
        r[g.p2] -= k_factor * (s1 - e1)
    mean = sum(r.values()) / max(1, len(r))
    return {b: v - mean for b, v in r.items()}


def head_to_head(games: Sequence[Game], bots: Sequence[str]) -> dict:
    """{(a, b): {wins, losses, ties, n, score, ci}} for a != b (a's point of view)."""
    out = {}
    for a, b in itertools.permutations(bots, 2):
        w = l = t = 0
        for g in games:
            if {g.p1, g.p2} != {a, b}:
                continue
            if g.winner == TIE:
                t += 1
            elif g.winner == a:
                w += 1
            else:
                l += 1
        n = w + l + t
        if n:
            pts = w + 0.5 * t
            out[(a, b)] = {
                "wins": w, "losses": l, "ties": t, "n": n, "score": pts / n,
                "ci": bench.wilson_interval(pts, n),
            }
    return out


def paired_head_to_head(games: Sequence[Game], bots: Sequence[str]) -> dict:
    """{(a, b): paired summary} over the fixed-team games (swap / mirror) of each bot pair.

    The unit is the team pair: the mean of a's scores in the two games sharing a ``team_pair``
    (pairs missing a game are dropped); see ``teams.paired_summary``.
    """
    out = {}
    for a, b in itertools.permutations(bots, 2):
        rows = [(g.team_pair, g.score_for(a)) for g in games
                if g.team_pair is not None and {g.p1, g.p2} == {a, b}]
        if rows:
            out[(a, b)] = teams_module.paired_summary(teams_module.pair_scores(rows))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_results(games: Sequence[Game], bots: Sequence[str], *, n_boot: int, seed: int,
                  anchor: str | None) -> dict:
    bots = list(bots)
    score, count = pair_tables(games, bots)
    elo = to_elo(fit_bt(score, count), bots, anchor)
    cis = bootstrap_elo(games, bots, n_boot=n_boot, seed=seed, anchor=anchor) if n_boot > 0 else {}
    seq = sequential_elo(games, bots)
    h2h = head_to_head(games, bots)
    paired = paired_head_to_head(games, bots)
    ratings = []
    for i, b in enumerate(bots):
        lo, hi = cis.get(b, (None, None))
        ratings.append({
            "bot": b, "elo": elo[i], "ci_low": lo, "ci_high": hi,
            "games": int(sum(count[i])), "points": sum(score[i]),
            "sequential_elo": seq[b],
        })
    ratings.sort(key=lambda r: -r["elo"])
    return {
        "paired_head_to_head": [{"a": a, "b": b, **d} for (a, b), d in paired.items()],
        "n_games": len(games), "bots": bots, "anchor": anchor, "bootstrap": n_boot,
        "bootstrap_seed": seed, "prior_weight": PRIOR_WEIGHT, "ratings": ratings,
        "head_to_head": [
            {"a": a, "b": b, **{k: v for k, v in d.items() if k != "ci"},
             "ci_low": d["ci"][0], "ci_high": d["ci"][1]}
            for (a, b), d in h2h.items()
        ],
    }


def format_markdown(res: dict) -> str:
    def f(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.0f}"

    anchor_text = f"anchor {res['anchor']} = 0" if res["anchor"] else "mean = 0"
    lines = [
        f"# League results ({res['n_games']} games)", "",
        "Bradley-Terry ratings (MM fit, ties = half a win, "
        f"{res['prior_weight']} virtual tie games per pair as prior), Elo scale "
        f"400*log10(strength), {anchor_text}. "
        f"95% CI: bootstrap over games ({res['bootstrap']} resamples, seed {res['bootstrap_seed']}).",
        "", "| Rank | Bot | BT rating | 95% CI | Games | Points | Seq. Elo |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(res["ratings"], 1):
        lines.append(
            f"| {i} | {r['bot']} | {f(r['elo'])} | [{f(r['ci_low'])}, {f(r['ci_high'])}] "
            f"| {r['games']} | {r['points']:g} | {r['sequential_elo']:+.0f} |"
        )
    lines += ["", "## Head to head", "",
              "Row bot's score vs column bot: W-L-T, score% [Wilson 95% CI].", ""]
    bots = [r["bot"] for r in res["ratings"]]
    lookup = {(h["a"], h["b"]): h for h in res["head_to_head"]}
    lines.append("| | " + " | ".join(bots) + " |")
    lines.append("|---|" + "---|" * len(bots))
    for a in bots:
        cells = []
        for b in bots:
            h = lookup.get((a, b))
            cells.append("-" if h is None else
                         f"{h['wins']}-{h['losses']}-{h['ties']}, {h['score']:.0%} "
                         f"[{h['ci_low']:.0%}, {h['ci_high']:.0%}]")
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    if res.get("paired_head_to_head"):
        lines += ["", "## Paired head to head (fixed teams)", "",
                  "Team pairs (swap / mirror): both games of a pair use the same teams with the "
                  "bots swapped; the unit is the pair's mean score. Row bot vs column bot: "
                  "score% [95% t-CI over pairs] (pairs).", ""]
        plook = {(h["a"], h["b"]): h for h in res["paired_head_to_head"]}
        lines.append("| | " + " | ".join(bots) + " |")
        lines.append("|---|" + "---|" * len(bots))
        for a in bots:
            cells = []
            for b in bots:
                h = plook.get((a, b))
                if h is None:
                    cells.append("-")
                elif h["pairs"] < 2:
                    cells.append(f"{h['mean']:.0%} ({h['pairs']} pair)")
                else:
                    cells.append(f"{h['mean']:.0%} [{h['ci_low']:.0%}, {h['ci_high']:.0%}] "
                                 f"({h['pairs']} pairs)")
            lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    comp = res.get("compute")
    if comp:
        pre = comp.get("preflight", {})
        lines += ["", "## Compute", ""]
        if pre:
            lines.append(pre["line"])
        lines += [f"- {w}" for w in comp.get("preflight_warnings", [])]
        if comp.get("bots"):
            lines += ["", "Measured MCTS iterations per second of per-world budget "
                      "(low or uneven values mean starved searches).", "",
                      "| Bot | Iter/s | Iterations | Decisions | Escalations | Fallbacks | World errors |",
                      "|---|---|---|---|---|---|---|"]
            for label, r in comp["bots"].items():
                lines.append(
                    f"| {label} | {r['iterations_per_s']:,.0f} | {r['iterations']:,} | "
                    f"{r['decisions']} | {r['escalations']} | {r['fallbacks']} | {r['world_errors']} |")
    if res.get("leaks"):
        lines += ["", "## Leak report", "", *[f"- {x}" for x in res["leaks"]]]
    return "\n".join(lines) + "\n"


def write_results(out: Path, games: Sequence[Game], bots: Sequence[str], *, n_boot: int,
                  seed: int, anchor: str | None, extra: dict | None = None) -> dict:
    res = build_results(games, bots, n_boot=n_boot, seed=seed, anchor=anchor)
    res.update(extra or {})
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "results.md").write_text(format_markdown(res), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------
# Live play
# ---------------------------------------------------------------------------


@dataclass
class LeagueConfig:
    out: Path
    battle_format: str = "gen9randombattle"
    requested_format: str = "gen9randombattle"  # the format asked for; battle_format is the
    # fixed-team variant when teams != random
    games_per_pair: int = 10
    concurrency: int = 1
    connect_timeout: float = 30.0
    game_timeout: float = 900.0
    search_time_ms: int = 100
    fp_search_time_ms: int = 100
    fp_parallelism: int = 1
    fp_threads: int = 1
    seed: int = 0
    search_workers: int | None = None  # shared MCTS pool size; None: per-player default
    parallel_pairs: int = 1
    teams: str = "random"  # random | swap | mirror (showdown_bot.teams)
    team_seed: int = 0
    port: int = 8000
    strict_compute: bool = False  # refuse to start when search_workers > cores - 2
    specs: dict = field(default_factory=dict)  # label -> BotSpec
    compute: dict = field(default_factory=dict)  # filled by run_league: preflight + throughput


class Tracker:
    """Everything created by the league, for the end-of-run leak check."""

    def __init__(self) -> None:
        self.players: list[Player] = []
        self.processes: list = []
        self.by_label: dict[str, list[Player]] = {}  # bot label -> its players (throughput)
        self.live: dict[tuple[str, str], tuple[Player, Player]] = {}  # running matchups
        self.finished: dict[tuple[str, str], list[Game]] = {}  # completed matchups

    def add(self, label: str, player: Player) -> None:
        self.players.append(player)
        self.by_label.setdefault(label, []).append(player)

    def tally(self) -> list[tuple[str, str, int, int, int]]:
        """``(a, b, a wins, b wins, ties)`` per matchup so far (running or complete)."""
        rows = []
        for (a, b), games in self.finished.items():
            rows.append((a, b, sum(g.winner == a for g in games),
                         sum(g.winner == b for g in games), sum(g.winner == TIE for g in games)))
        for (a, b), (pa, pb) in self.live.items():
            if (a, b) in self.finished:
                continue
            ties = pa.n_tied_battles
            rows.append((a, b, pa.n_won_battles, pb.n_won_battles, ties))
        return rows

    def open_connections(self) -> list[str]:
        bad = []
        for p in self.players:
            ws = getattr(p.ps_client, "websocket", None)
            if ws is not None and getattr(ws, "close_code", None) is None:
                bad.append(f"open websocket: {p.username}")
        for proc in self.processes:
            if proc.poll() is None:
                bad.append(f"running Foul Play pid {proc.pid}")
        return bad


def _quality_kwargs(spec: BotSpec) -> dict:
    """SearchConfig fields from the 589.47 decision-quality spec keys (only those given)."""
    out: dict = {}
    for key in ("escalate", "prune_noop"):
        if spec.param(key) is not None:
            out[key] = bool(spec.param(key))
    if spec.param("ev2") is not None:  # evaluator ensemble over worlds
        out["evaluator2"] = spec.param("ev2")
        out["evaluator2_scale"] = spec.param("scale2", 1.0)
    if spec.param("belief") is not None:  # bead showdown-bot-1wu
        out["belief_reweight"] = bool(spec.param("belief"))
    if spec.param("btau") is not None:
        out["belief_tau"] = spec.param("btau")
    if spec.param("bfloor") is not None:
        out["belief_floor"] = spec.param("bfloor")
    if spec.param("noop_switches") is not None:
        out["noop_vs_switches"] = bool(spec.param("noop_switches"))
    for key in ("escalate_gap", "escalate_factor", "escalate_steps", "escalate_cap_ms", "tera_margin"):
        if spec.param(key) is not None:
            out[key] = spec.param(key)
    return out


def search_config_fields(spec: BotSpec) -> dict:
    """Every SearchConfig field a search spec sets (spec keys -> field names; only those given).

    Shared with scripts/selfplay.py so a pool opponent takes the same keys as a league bot.
    """
    out: dict = {}
    for key, name in (("evaluator", "evaluator"), ("scale", "evaluator_scale"),
                      ("time_ms", "search_time_ms"), ("worlds", "worlds"),
                      ("selection", "selection"), ("gamma", "selection_gamma"),
                      ("mode", "mode"), ("tkappa", "tendency_kappa"), ("tdelta", "tendency_delta")):
        if spec.param(key) is not None:
            out[name] = spec.param(key)
    if spec.param("tendency") is not None:
        out["tendency"] = spec.param("tendency") == "on"
    out.update(_quality_kwargs(spec))
    return out


def _make_player(spec: BotSpec, cfg: LeagueConfig, recorder, concurrency: int, salt: int,
                 *, account=None, server=None, wrap=None, extra: dict | None = None) -> Player:
    """Build a spec's player. The keyword-only hooks serve scripts/ladder.py: a given
    account/server replaces the local throwaway ones, ``wrap(cls)`` may return a subclass,
    ``extra`` adds constructor keywords."""
    cls = bench.get_player_class(spec.type)
    if wrap is not None:
        cls = wrap(cls)
    kwargs = dict(
        battle_format=cfg.battle_format,
        max_concurrent_battles=concurrency,
        account_configuration=account
        or AccountConfiguration("lg" + secrets.token_hex(5), None),
        server_configuration=server or bench.local_server(cfg.port),
        log_level=logging.ERROR,
        open_timeout=cfg.connect_timeout,
        save_replays=str(cfg.out / "replays"),
    )
    kwargs.update(extra or {})
    if spec.type in SEARCH_BOTS:
        from showdown_bot.players.search import SearchConfig, get_pool

        # Explicit per-player config: the process-wide SHOWDOWN_SEARCH_* vars are not used.
        kwargs["config"] = SearchConfig(
            search_time_ms=spec.param("time_ms", cfg.search_time_ms),
            worlds=spec.param("worlds"),
            **_quality_kwargs(spec),
        )
        kwargs["recorder"] = recorder
        kwargs["seed"] = cfg.seed * 1000 + salt
        if spec.param("evaluator") is not None:
            kwargs["evaluator"] = spec.param("evaluator")
        scale = spec.param("scale", 1.0)
        if scale != 1:
            if "evaluator_scale" not in _init_params(cls):
                raise ValueError(
                    f"{spec.label}: scale={scale:g} needs SearchPlayer(evaluator_scale=...), "
                    "which this checkout does not have (branch feat/589.16-leaf-scale)"
                )
            kwargs["evaluator_scale"] = scale
        if spec.param("selection") is not None or spec.param("gamma") is not None:
            if "selection" not in _init_params(cls):
                raise ValueError(f"{spec.label}: selection/gamma need SearchPlayer(selection=...)")
            if spec.param("selection") is not None:
                kwargs["selection"] = spec.param("selection")
            if spec.param("gamma") is not None:
                kwargs["selection_gamma"] = spec.param("gamma")
        if spec.param("mode") is not None:
            if "mode" not in _init_params(cls):
                raise ValueError(f"{spec.label}: mode needs SearchPlayer(mode=...)")
            kwargs["mode"] = spec.param("mode")
        if spec.param("tendency") is not None:
            if "tendency" not in _init_params(cls):
                raise ValueError(f"{spec.label}: tendency needs SearchPlayer(tendency=...)")
            kwargs["tendency"] = spec.param("tendency") == "on"
            if spec.param("tkappa") is not None:
                kwargs["tendency_kappa"] = spec.param("tkappa")
            if spec.param("tdelta") is not None:
                kwargs["tendency_delta"] = spec.param("tdelta")
        if cfg.search_workers:
            kwargs["executor"] = get_pool(cfg.search_workers)
    return cls(**kwargs)


def _new_games(bot_name: str, player: Player, opp_name: str, opp_user: str, seen: set,
               cfg: LeagueConfig, seed: int, plan: teams_module.GamePlan | None = None) -> list[Game]:
    """Games finished since the last call, from the challenger's battles."""
    out = []
    for tag, battle in player.battles.items():
        if tag in seen or not battle.finished:
            continue
        seen.add(tag)
        role = battle.player_role or "p1"
        mine, theirs = (bot_name, opp_name) if role == "p1" else (opp_name, bot_name)
        if battle.won is True:
            winner = bot_name
        elif battle.won is False:
            winner = opp_name
        else:
            winner = TIE
        users = (player.username, opp_user) if role == "p1" else (opp_user, player.username)
        specs = {k: cfg.specs[k].text for k in (bot_name, opp_name) if k in cfg.specs}
        team = ({"team_mode": cfg.teams, "team_pair": plan.pair,
                 "team_ids": (plan.p1_team_id, plan.p2_team_id)} if plan else {})
        out.append(Game(tag, mine, theirs, winner, int(battle.turn), seed,
                        users[0], users[1], cfg.battle_format, specs, **team))
    return out


async def play_matchup(a: str, b: str, cfg: LeagueConfig, tracker: Tracker, recorder,
                       games_path: Path, pair_index: int) -> list[Game]:
    """Play cfg.games_per_pair games between two bots (one may be Foul Play)."""
    if FOULPLAY_NAME in (cfg.specs[a].type, cfg.specs[b].type):
        if cfg.teams != "random":  # Foul Play brings its own team in a normal random battle
            cfg = replace(cfg, teams="random", battle_format=cfg.requested_format)
        return await _play_foulplay_matchup(a, b, cfg, tracker, recorder, games_path, pair_index)
    plans = None
    if cfg.teams != "random":
        if cfg.concurrency != 1:
            logger.warning("--teams %s plays one game at a time; ignoring --concurrency %d",
                           cfg.teams, cfg.concurrency)
            cfg = replace(cfg, concurrency=1)
        plans = teams_module.plan_games(cfg.teams, cfg.games_per_pair,
                                        cfg.team_seed * 1000 + pair_index)
    players: list[Player] = []
    games: list[Game] = []
    try:
        pa = _make_player(cfg.specs[a], cfg, recorder, cfg.concurrency, 2 * pair_index)
        players.append(pa)
        tracker.add(a, pa)
        pb = _make_player(cfg.specs[b], cfg, recorder, cfg.concurrency, 2 * pair_index + 1)
        players.append(pb)
        tracker.add(b, pb)
        tracker.live[(a, b)] = (pa, pb)
        await bench._wait_for_player_logins(players, cfg.connect_timeout)
        by_name = {a: pa, b: pb}
        seen: set = set()  # shared: both players hold every battle of the pair
        for plan in plans or ():
            try:
                p1_player, _ = await teams_module.play_plan(
                    plan, pa, pb, cfg.game_timeout, mode=cfg.teams,
                    seed=cfg.team_seed * 1000 + pair_index)
            except TimeoutError:
                raise RuntimeError(f"{a} vs {b}: game {plan.index} timed out") from None
            p1_name, p2_name = (a, b) if plan.a_is_p1 else (b, a)
            for g in _new_games(p1_name, p1_player, p2_name, by_name[p2_name].username, seen,
                                cfg, cfg.seed, plan):
                append_game(games_path, g)
                games.append(g)
        for p1, p2, k in ([] if plans else seat_schedule(a, b, cfg.games_per_pair, cfg.concurrency)):
            waves = -(-k // cfg.concurrency)
            try:
                await asyncio.wait_for(
                    by_name[p1].battle_against(by_name[p2], n_battles=k),
                    timeout=cfg.game_timeout * waves,
                )
            except TimeoutError:
                raise RuntimeError(f"{p1} vs {p2}: {k} games timed out") from None
            for g in _new_games(p1, by_name[p1], p2, by_name[p2].username, seen, cfg, cfg.seed):
                append_game(games_path, g)
                games.append(g)
    finally:
        await asyncio.gather(*(bench._disconnect_player(p) for p in players),
                             return_exceptions=True)
    return games


async def _play_foulplay_matchup(a, b, cfg, tracker, recorder, games_path, pair_index):
    fp_label = a if cfg.specs[a].type == FOULPLAY_NAME else b
    bot_name = b if fp_label == a else a
    if cfg.specs[bot_name].type == FOULPLAY_NAME:
        raise ValueError("foulplay cannot play itself")
    n = cfg.games_per_pair
    fp_user = bench._foulplay_username()
    command = bench.build_foulplay_command(
        bench._foulplay_python_executable(), fp_user, n, cfg.battle_format,
        cfg.fp_search_time_ms, cfg.fp_parallelism, cfg.fp_threads,
    )
    log_path = bench.FOULPLAY_LOG_DIR / f"league-{fp_user}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    process, log_file = bench._launch_foulplay_process(command, log_path)
    tracker.processes.append(process)
    bot = None
    games: list[Game] = []
    try:
        bot = _make_player(cfg.specs[bot_name], cfg, recorder, 1, 2 * pair_index)
        tracker.add(bot_name, bot)
        await bench._wait_for_player_logins([bot], cfg.connect_timeout)
        seen: set = set()
        for game in range(1, n + 1):
            await bench._wait_for_foulplay_ready(
                process, log_path,
                cfg.connect_timeout if game == 1 else cfg.game_timeout, min_markers=game)
            try:
                await asyncio.wait_for(
                    bot.send_challenges(to_id_str(fp_user), 1), timeout=cfg.game_timeout)
            except TimeoutError:
                raise RuntimeError(f"Foul Play game {game}/{n} timed out") from None
            for g in _new_games(bot_name, bot, fp_label, fp_user, seen, cfg, cfg.seed):
                append_game(games_path, g)
                games.append(g)
    finally:
        bench._terminate_foulplay_process(process, log_file)
        if bot is not None:
            await bench._disconnect_player(bot)
    return games


async def run_limited(jobs: Sequence[Callable[[], Awaitable]], limit: int) -> list:
    """Run jobs with at most ``limit`` in flight; results keep job order.

    The first failure cancels the other jobs (their ``finally`` blocks clean up).
    """
    sem = asyncio.Semaphore(max(1, limit))

    async def guarded(job):
        async with sem:
            return await job()

    tasks = [asyncio.ensure_future(guarded(j)) for j in jobs]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def pair_cores(a: BotSpec, b: BotSpec, cfg: LeagueConfig) -> tuple[int, int]:
    """Rough (foul_play_cores, our_cores) one matchup keeps busy."""
    fp = cfg.fp_parallelism * cfg.fp_threads if FOULPLAY_NAME in (a.type, b.type) else 0
    ours = sum(1 for x in (a, b) if x.type in SEARCH_BOTS)
    return fp, ours


def load_report(pairs: Sequence[tuple[BotSpec, BotSpec]], cfg: LeagueConfig,
                cpu_count: int | None = None) -> tuple[str, str | None]:
    """Cores used at the peak of ``cfg.parallel_pairs`` concurrent matchups."""
    loads = sorted((pair_cores(a, b, cfg) for a, b in pairs), key=lambda x: -(x[0] + x[1]))
    top = loads[: cfg.parallel_pairs]
    fp = sum(x[0] for x in top)
    ours = sum(x[1] for x in top)
    if cfg.search_workers:
        ours = min(ours, cfg.search_workers) if ours else 0
    available = cpu_count if cpu_count is not None else os.cpu_count()
    line = (f"cores used (peak, {cfg.parallel_pairs} parallel pairs): "
            f"foulplay={fp}, search={ours}")
    warning = None
    if available is not None and fp + ours > available:
        warning = (f"WARNING: requested compute ({fp + ours} cores) exceeds "
                   f"os.cpu_count()={available}; CPU contention will skew ratings")
    return line, warning


def league_preflight(specs: Sequence[BotSpec], cfg: LeagueConfig,
                     cpu_count: int | None = None) -> tuple[dict, list[str], bool]:
    """Compute preflight: pool workers vs cores and peak concurrent world jobs vs workers.

    Peak demand = the busiest ``parallel_pairs`` matchups, each search player of a
    matchup searching ``worlds`` worlds for each of its ``concurrency`` games at once.
    Returns ``(report dict, warnings, refuse)``.
    """
    from showdown_bot.players.search import SearchConfig

    def worlds(spec: BotSpec) -> int:
        return spec.param("worlds") or SearchConfig().plan(True)[0]

    by = {s.label: s for s in specs}
    per_pair = sorted(
        (sum(worlds(x) * cfg.concurrency for x in (by[a], by[b]) if x.type in SEARCH_BOTS)
         for a, b in roundrobin_pairs(list(by))), reverse=True)
    demand = sum(per_pair[: cfg.parallel_pairs])
    workers = cfg.search_workers or SearchConfig().parallelism  # per-player default pool
    info, warnings, refuse = compute.preflight(
        workers, demand, cpu_count, strict=cfg.strict_compute and demand > 0)
    if not cfg.search_workers and demand:
        warnings.append("WARNING: no --search-workers: all search players share a default "
                        f"pool of {workers} worker(s); set --search-workers")
    return {"workers": workers, "peak_world_jobs": demand,
            "cores": cpu_count if cpu_count is not None else os.cpu_count(),
            "line": info[0]}, warnings, refuse


def compute_report(tracker: Tracker) -> dict:
    """Measured MCTS throughput per search bot, summed over all its players."""
    out = {}
    for label, players in tracker.by_label.items():
        st = [p.stats for p in players if hasattr(getattr(p, "stats", None), "iterations")]
        if not st:
            continue
        iters = sum(x.iterations for x in st)
        ms = sum(x.search_ms for x in st)
        out[label] = {
            "iterations": iters, "search_ms": ms,
            "iterations_per_s": iters / (ms / 1000) if ms else 0.0,
            "decisions": sum(x.searched for x in st),
            "escalations": sum(x.escalations for x in st),
            "fallbacks": sum(x.fallbacks for x in st),
            "world_errors": sum(x.world_errors for x in st),
        }
    return out


def emit_progress(tracker: Tracker, printer: compute.ProgressPrinter) -> None:
    """Print one flushed progress line if the finished-game count changed."""
    rows = tracker.tally()
    done = sum(w1 + w2 + t for _, _, w1, w2, t in rows)
    busy = [r for r in rows if (r[0], r[1]) not in tracker.finished]
    pick = busy or rows
    if not pick:
        printer.changed = False
        return
    a, b, w1, w2, t = pick[0]
    others = "; ".join(f"{x} {p}-{q} {y}" for x, y, p, q, _ in pick[1:4])
    printer.update(done, f"{a} v {b}", w1, w2, t, others or None)


async def _progress(tracker: Tracker, total: int, interval: float = 3.0,
                    heartbeat: float = 120.0) -> None:
    """Print a progress line whenever a game finishes (and every ``heartbeat`` s)."""
    printer = compute.ProgressPrinter(total)
    quiet_since = time.monotonic()
    while True:
        await asyncio.sleep(interval)
        emit_progress(tracker, printer)
        if printer.changed:
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > heartbeat:
            printer.last = -1  # still running, no game finished: say so
            emit_progress(tracker, printer)
            quiet_since = time.monotonic()


async def run_league(specs: Sequence[BotSpec], cfg: LeagueConfig) -> tuple[list[Game], list[str]]:
    if cfg.teams != "random":
        if cfg.games_per_pair % 2:
            raise ValueError(f"--teams {cfg.teams} plays pairs of games; --games-per-pair must be even")
        if cfg.battle_format != teams_module.FIXED_FORMAT:
            cfg.requested_format = cfg.battle_format
            cfg.battle_format = teams_module.battle_format_for(cfg.teams, cfg.battle_format)
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "replays").mkdir(exist_ok=True)
    cfg.specs = {s.label: s for s in specs}
    labels = [s.label for s in specs]
    games_path = cfg.out / "games.jsonl"
    games_path.write_text("", encoding="utf-8")
    recorder = None
    if any(s.type in SEARCH_BOTS for s in specs):
        from showdown_bot.recorder import GameRecorder

        recorder = GameRecorder(cfg.out / "records")
    tracker = Tracker()
    pairs = roundrobin_pairs(labels)
    line, warning = load_report([(cfg.specs[a], cfg.specs[b]) for a, b in pairs], cfg)
    logger.info(line)
    if warning:
        logger.warning(warning)
    pre, pre_warnings, refuse = league_preflight(specs, cfg)
    logger.info(pre["line"])
    for w in pre_warnings:
        logger.warning(w)
    if refuse:
        raise ValueError(pre_warnings[-1])
    cfg.compute = {"preflight": pre, "preflight_warnings": pre_warnings}

    def make_job(idx, a, b):
        async def job():
            start = time.perf_counter()
            logger.info("matchup %s vs %s (%d games)", a, b, cfg.games_per_pair)
            got = await play_matchup(a, b, cfg, tracker, recorder, games_path, idx)
            tracker.finished[(a, b)] = got
            logger.info("  %s %d - %d %s (%d games, %.0fs)", a, sum(g.winner == a for g in got),
                        sum(g.winner == b for g in got), b, len(got), time.perf_counter() - start)
            leaks = tracker.open_connections()
            if leaks and cfg.parallel_pairs == 1:
                logger.warning("  leaked after matchup: %s", leaks)
            return got
        return job

    total = len(pairs) * cfg.games_per_pair
    ticker = asyncio.ensure_future(_progress(tracker, total))
    try:
        results = await run_limited(
            [make_job(i, a, b) for i, (a, b) in enumerate(pairs)], cfg.parallel_pairs)
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        emit_progress(tracker, compute.ProgressPrinter(total))
    cfg.compute["bots"] = compute_report(tracker)
    for label, r in cfg.compute["bots"].items():
        logger.info("%s", compute.iterations_line(
            label, r["iterations"], r["search_ms"], decisions=r["decisions"],
            escalations=r["escalations"], fallbacks=r["fallbacks"],
            world_errors=r["world_errors"]))
    all_games = [g for got in results for g in got]
    await asyncio.sleep(0.2)
    leaks = tracker.open_connections()
    leaks += [f"pending asyncio task: {t.get_coro()}" for t in asyncio.all_tasks()
              if t is not asyncio.current_task()]
    logger.info("end of run: %d players created, %d leaks", len(tracker.players), len(leaks))
    return all_games, leaks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def split_entries(text: str) -> list[str]:
    return [t for t in (x.strip() for x in text.split(",")) if t]


def parse_bots(text: str) -> list[BotSpec]:
    """Parse ``label=type[:k=v...]`` entries (plain names are label = type)."""
    specs: list[BotSpec] = []
    for raw in split_entries(text):
        spec = parse_spec(raw)
        if any(s.label == spec.label for s in specs):
            raise ValueError(f"duplicate bot label {spec.label!r}")
        specs.append(spec)
    if len(specs) < 2:
        raise ValueError("need at least two bots")
    return specs


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bots", help="comma-separated label=type[:key=value...] or plain type. Types: "
                   + ", ".join(sorted(bench.PLAYER_REGISTRY)) + ", foulplay. search keys: "
                   "evaluator, scale, time_ms, worlds, selection (ucb1|exp3|rm), gamma, mode (mixed|argmax|nash), tendency (on|off), tkappa, tdelta. e.g. net=search-net:evaluator=m.pev:scale=4")
    p.add_argument("--games-per-pair", type=bench._positive_int, default=20)
    p.add_argument("--out", type=Path, help="output dir, e.g. data/league/<run-id>")
    p.add_argument("--rate-only", type=Path, metavar="DIR", help="recompute results from DIR/games.jsonl")
    p.add_argument("--pairing", choices=["roundrobin"], default="roundrobin")
    p.add_argument("--format", default="gen9randombattle")
    p.add_argument("--teams", choices=teams_module.MODES, default="random",
                   help="random (default): fresh random teams per game. swap / mirror: variance-"
                        "reduced pairs of games (see showdown_bot.teams) in the gen9randombattlefixed "
                        "server format; needs an even --games-per-pair and plays one game at a time "
                        "per matchup. Foul Play matchups always use random teams")
    p.add_argument("--team-seed", type=int, default=0, help="seed of the generated teams")
    p.add_argument("--port", type=int, default=8000, help="local Showdown server port "
                   "(not used for Foul Play, which stays on 8000)")
    p.add_argument("--concurrency", type=bench._positive_int, default=1,
                   help="games in flight per matchup (also the seat-swap chunk size)")
    p.add_argument("--search-time-ms", type=bench._positive_int, default=100)
    p.add_argument("--fp-search-time-ms", type=bench._positive_int, default=100)
    p.add_argument("--fp-parallelism", type=bench._positive_int, default=1)
    p.add_argument("--fp-threads", type=bench._positive_int, default=1)
    p.add_argument("--connect-timeout", type=bench._positive_seconds, default=30)
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search-workers", type=bench._positive_int, default=None,
                   help="one MCTS process pool of this size shared by all search players")
    p.add_argument("--parallel-pairs", type=bench._positive_int, default=1,
                   help="matchups played concurrently")
    p.add_argument("--strict-compute", action="store_true",
                   help="refuse to start when --search-workers > cores - 2 (room for the main "
                        "process and the node server); otherwise only warn")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--anchor", default=None, help="bot whose rating is 0 (default: mean 0)")
    return p


def rate_only(directory: Path, n_boot: int, seed: int, anchor: str | None) -> dict:
    games = read_games(directory / "games.jsonl")
    bots = sorted({b for g in games for b in (g.p1, g.p2)})
    return write_results(directory, games, bots, n_boot=n_boot, seed=seed, anchor=anchor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    anchor = ALIASES.get(args.anchor, args.anchor) if args.anchor else None
    if args.rate_only:
        print(format_markdown(rate_only(args.rate_only, args.bootstrap, args.seed, anchor)))
        return 0
    if not args.bots or not args.out:
        print("--bots and --out are required (or use --rate-only DIR)", file=sys.stderr)
        return 2
    try:
        specs = parse_bots(args.bots)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    cfg = LeagueConfig(
        out=args.out, battle_format=args.format, games_per_pair=args.games_per_pair,
        concurrency=args.concurrency, connect_timeout=args.connect_timeout,
        game_timeout=args.game_timeout, search_time_ms=args.search_time_ms,
        fp_search_time_ms=args.fp_search_time_ms, fp_parallelism=args.fp_parallelism,
        fp_threads=args.fp_threads, seed=args.seed,
        search_workers=args.search_workers, parallel_pairs=args.parallel_pairs,
        teams=args.teams, team_seed=args.team_seed, port=args.port,
        strict_compute=args.strict_compute,
    )
    if args.teams != "random" and args.games_per_pair % 2:
        print(f"error: --teams {args.teams} needs an even --games-per-pair", file=sys.stderr)
        return 2
    try:
        games, leaks = asyncio.run(run_league(specs, cfg))
    except KeyboardInterrupt:
        print("league interrupted; games.jsonl holds finished games "
              "(rate with --rate-only)", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"league error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    res = write_results(
        cfg.out, games, [s.label for s in specs], n_boot=args.bootstrap, seed=args.seed,
        anchor=anchor,
        extra={"leaks": leaks, "games_per_pair": cfg.games_per_pair, "compute": cfg.compute,
               "specs": {s.label: s.text for s in specs}},
    )
    print(format_markdown(res))
    if leaks:
        print(f"WARNING: {len(leaks)} leaked resources: {leaks}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
