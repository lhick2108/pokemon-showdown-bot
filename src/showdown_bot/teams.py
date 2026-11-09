"""Fixed-team variance reduction for duels and leagues ("duplicate bridge" for random battles).

Random battles give every game fresh random teams, so team luck dominates the variance. Two
paired designs remove most of it; both play the games of a pair back to back with seats swapped:

* ``swap``: a team pair (T1, T2); game 1 bot A has T1 as p1 and bot B has T2 as p2, game 2 the
  teams stay in their seats but the bots swap (A: T2 as p2, B: T1 as p1).
* ``mirror``: one team T for both bots; game 1 A is p1, game 2 A is p2.
* ``random``: the default, fresh server-side random teams (no pairing).

Teams come from Showdown's own gen9randombattle generator (scripts/gen_teams.js), seeded and
reproducible, and are played in the server format ``gen9randombattlefixed``
(scripts/showdown/custom-formats.ts), which has the random battle rules but takes the players'
teams. The statistics unit of a paired design is the team pair (the mean score of its two games).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from poke_env.teambuilder import Teambuilder

MODES = ("random", "swap", "mirror")
FIXED_FORMAT = "gen9randombattlefixed"
RANDOM_FORMAT = "gen9randombattle"

REPO_ROOT = Path(__file__).resolve().parents[2]
GEN_TEAMS_JS = REPO_ROOT / "scripts" / "gen_teams.js"
_SERVER_ENV = "SHOWDOWN_BOT_SERVER_DIR"


def data_format(battle_format: str) -> str:
    """The randbats data set (sets/stats) of a battle format: the fixed-team format uses
    exactly the gen9randombattle data."""
    return RANDOM_FORMAT if battle_format == FIXED_FORMAT else battle_format


def battle_format_for(mode: str, requested: str) -> str:
    """Server format to challenge in: paired modes need the fixed-team variant."""
    if mode not in MODES:
        raise ValueError(f"unknown teams mode {mode!r}; expected one of {MODES}")
    if mode == "random":
        return requested
    if requested not in (RANDOM_FORMAT, FIXED_FORMAT):
        raise ValueError(f"--teams {mode} only supports {RANDOM_FORMAT}, not {requested}")
    return FIXED_FORMAT


class PackedTeam(Teambuilder):
    """Yields one packed team verbatim (ConstantTeambuilder re-packs it, which is lossy)."""

    def __init__(self, packed: str):
        self.packed_team = packed

    def yield_team(self) -> str:
        return self.packed_team


def _server_dir() -> Path:
    env = os.environ.get(_SERVER_ENV)
    return Path(env) if env else REPO_ROOT / "vendor" / "pokemon-showdown"


def generate_teams(n: int, seed: int, server_dir: Path | None = None) -> list[str]:
    """``n`` packed gen9randombattle teams from the Showdown generator, seeded by ``seed``.

    Team i uses the generator seed ``[seed, i, 3, 4]``; the same (seed, i) always gives the
    same team (for a fixed Showdown checkout).
    """
    if n <= 0:
        return []
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("node not found on PATH (needed for Showdown team generation)")
    server = server_dir or _server_dir()
    if not (server / "dist" / "sim" / "teams.js").is_file():
        raise RuntimeError(f"no built Showdown at {server} (set {_SERVER_ENV})")
    out = subprocess.run(
        [node, str(GEN_TEAMS_JS), str(server), str(n), str(seed)],
        capture_output=True, text=True, check=True, timeout=120,
    ).stdout
    teams = [json.loads(line)["packed"] for line in out.splitlines() if line.strip()]
    if len(teams) != n:
        raise RuntimeError(f"expected {n} teams, generated {len(teams)}")
    return teams


@dataclass(frozen=True)
class GamePlan:
    """One game of a paired design, from bot A's and bot B's point of view."""

    index: int  # 0-based game number within the matchup
    pair: int  # team pair (0-based); games 2k and 2k+1 share pair k
    a_team: str  # packed team of bot A
    b_team: str
    a_team_id: int  # index into the generated team list
    b_team_id: int
    a_is_p1: bool  # p1 is the challenger

    @property
    def p1_team_id(self) -> int:
        return self.a_team_id if self.a_is_p1 else self.b_team_id

    @property
    def p2_team_id(self) -> int:
        return self.b_team_id if self.a_is_p1 else self.a_team_id


def plan_games(mode: str, n_games: int, seed: int, teams: Sequence[str] | None = None) -> list[GamePlan]:
    """The schedule of a paired design: 2 games per team pair, bot A is p1 in the first.

    ``teams`` may be given (indexed as generated); otherwise they are generated from ``seed``.
    swap needs 2 teams per pair, mirror 1.
    """
    if mode not in ("swap", "mirror"):
        raise ValueError(f"plan_games needs swap or mirror, not {mode!r}")
    if n_games <= 0 or n_games % 2:
        raise ValueError(f"--teams {mode} plays pairs of games; need an even game count, got {n_games}")
    n_pairs = n_games // 2
    per_pair = 2 if mode == "swap" else 1
    if teams is None:
        teams = generate_teams(n_pairs * per_pair, seed)
    if len(teams) < n_pairs * per_pair:
        raise ValueError(f"need {n_pairs * per_pair} teams, got {len(teams)}")
    plans = []
    for k in range(n_pairs):
        t1 = k * per_pair
        t2 = t1 + 1 if mode == "swap" else t1
        # game 1: A is p1 with T1, B is p2 with T2.  game 2: A is p2 with T2, B is p1 with T1.
        plans.append(GamePlan(2 * k, k, teams[t1], teams[t2], t1, t2, True))
        plans.append(GamePlan(2 * k + 1, k, teams[t2], teams[t1], t2, t1, False))
    return plans


def slice_plans(plans: Sequence[GamePlan], n_games: int, start: int = 0) -> list[GamePlan]:
    """The next ``n_games`` plans from ``start`` (a share of a parallel run); pairs stay whole
    as long as ``n_games`` and ``start`` are even."""
    if n_games % 2 or start % 2:
        raise ValueError("shares of a paired design must be even")
    return list(plans[start:start + n_games])


def split_pairs(n_games: int, instances: int) -> list[int]:
    """Split ``n_games`` (even) over ``instances`` workers in whole pairs (no empty shares)."""
    if n_games % 2:
        raise ValueError(f"paired designs need an even game count, got {n_games}")
    pairs = n_games // 2
    instances = max(1, min(instances, pairs))
    base, extra = divmod(pairs, instances)
    return [2 * (base + (1 if i < extra else 0)) for i in range(instances)]


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) (Numerical Recipes continued fraction)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)

    tiny = 1e-300
    f, c, d = 1.0, 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    f = d
    for m in range(1, 300):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1 + num * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + num / (c if abs(c) > tiny else tiny)
            f *= c * d
        if abs(c * d - 1) < 1e-14:
            break
    return math.exp(ln_front) * f / a


def _t_quantile(p: float, df: int) -> float:
    """Student t quantile for p > 0.5 (bisection on the exact t CDF)."""
    def cdf(t: float) -> float:
        tail = 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))
        return 1 - tail if t > 0 else tail

    lo, hi = 0.0, 1.0
    for _ in range(60):  # p is a probability < 1, so a few doublings suffice; never spin
        if cdf(hi) >= p:
            break
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def paired_summary(pair_scores: Sequence[float], level: float = 0.95) -> dict:
    """Mean score over team pairs with a t-interval (each pair is one observation).

    ``pair_scores`` holds one score per team pair (the mean of its games' 0 / 0.5 / 1 scores).
    Also returns ``binomial_se``, the standard error an unpaired design would have at the
    same mean and game count, and ``variance_ratio`` = paired variance / unpaired variance of
    the mean (< 1 means pairing helped).
    """
    n = len(pair_scores)
    if n == 0:
        nan = float("nan")
        return {"pairs": 0, "mean": nan, "sd": nan, "se": nan, "ci_low": nan, "ci_high": nan,
                "binomial_se": nan, "variance_ratio": nan}
    mean = sum(pair_scores) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in pair_scores) / (n - 1)
        se = math.sqrt(var / n)
        half = _t_quantile(0.5 + level / 2, n - 1) * se
        ci = (mean - half, mean + half)
        sd = math.sqrt(var)
    else:
        sd, se, ci = float("nan"), float("nan"), (float("nan"), float("nan"))
    bse = math.sqrt(mean * (1 - mean) / (2 * n))
    ratio = (se / bse) ** 2 if bse > 0 and n > 1 else float("nan")
    return {"pairs": n, "mean": mean, "sd": sd, "se": se, "ci_low": ci[0], "ci_high": ci[1],
            "binomial_se": bse, "variance_ratio": ratio}


def pair_scores(results: Sequence[tuple[int, float]]) -> list[float]:
    """Score per team pair from ``(pair id, game score)`` rows; pairs missing a game are dropped."""
    by_pair: dict[int, list[float]] = {}
    for pair, score in results:
        by_pair.setdefault(pair, []).append(score)
    return [sum(v) / len(v) for _, v in sorted(by_pair.items()) if len(v) == 2]


def format_paired(summary: dict) -> str:
    if not summary["pairs"]:
        return "paired: no complete team pairs"
    return (f"paired ({summary['pairs']} team pairs, unit = mean score of the pair): "
            f"{summary['mean']:.1%}, 95% t-CI [{summary['ci_low']:.1%}, {summary['ci_high']:.1%}], "
            f"SE {summary['se']:.1%} vs unpaired binomial SE {summary['binomial_se']:.1%} "
            f"(variance ratio {summary['variance_ratio']:.2f})")


# ---------------------------------------------------------------------------
# Playing one planned game
# ---------------------------------------------------------------------------


def team_info(plan: GamePlan, mode: str, seed: int, *, is_a: bool) -> dict:
    """What the recorder stores about one side's team in a planned game."""
    return {
        "mode": mode, "seed": seed, "pair": plan.pair, "game": plan.index,
        "team_id": plan.a_team_id if is_a else plan.b_team_id,
        "p1_team_id": plan.p1_team_id, "p2_team_id": plan.p2_team_id,
    }


async def play_plan(plan: GamePlan, a, b, timeout: float, *, mode: str = "", seed: int = 0):
    """Give both poke-env players their team for ``plan`` and play it (the p1 side challenges).

    Both players read ``next_team`` once per game (the challenger when challenging, the
    accepter when starting to wait), so setting the teams just before the single game keeps
    them in sync. Returns ``(p1_player, p2_player)``.
    """
    import asyncio

    a.update_team(PackedTeam(plan.a_team))
    b.update_team(PackedTeam(plan.b_team))
    a.team_info = team_info(plan, mode, seed, is_a=True)
    b.team_info = team_info(plan, mode, seed, is_a=False)
    p1, p2 = (a, b) if plan.a_is_p1 else (b, a)
    await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=timeout)
    return p1, p2
