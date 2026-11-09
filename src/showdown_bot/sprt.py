"""Sequential probability ratio test on paired games (pentanomial GSPRT, as in Fishtest / fastchess).

A *pair* is two games with the same teams and swapped seats; the candidate scores 0, 0.5, 1,
1.5 or 2 points, counted in ``counts[0..4]`` (``counts[i]`` = pairs worth i / 2 points).

Test H0: Elo = elo0 against H1: Elo = elo1 (plain logistic Elo, score = 1 / (1 + 10^(-elo/400))).
The log-likelihood ratio is the normal (generalised SPRT) approximation used by Fishtest's
``LLRcalc.LLR_alt`` and by fastchess::

    LLR = N * (s1 - s0) * (2 * s - s0 - s1) / (2 * var)

with N the number of pairs, s the mean pair score on [0, 1] (points / 2), var the (population)
variance of the pair score under the empirical distribution, and s0, s1 the scores of elo0 and
elo1. Zero counts are replaced by 1e-3 first (Fishtest's ``regularize``). The test stops when
LLR crosses ``ln(beta / (1 - alpha))`` (H0 accepted) or ``ln((1 - beta) / alpha)`` (H1 accepted).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)  # pair score on [0, 1] for 0, 0.5, 1, 1.5, 2 points
REGULARIZE = 1e-3
Z95 = 1.959963984540054


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def sprt_bounds(alpha: float, beta: float) -> tuple[float, float]:
    """(lower, upper) LLR bounds: H0 accepted at or below lower, H1 at or above upper."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def pair_index(points: float) -> int:
    """Index into the pentanomial counts of a pair worth ``points`` (0, 0.5, ..., 2)."""
    i = round(points * 2)
    if not 0 <= i <= 4 or abs(points * 2 - i) > 1e-9:
        raise ValueError(f"a pair is worth 0, 0.5, 1, 1.5 or 2 points, got {points}")
    return i


def _moments(counts, regularize: bool) -> tuple[float, float, float]:
    """(N, mean, variance) of the pair score on [0, 1] under the (regularised) counts."""
    c = [float(x) for x in counts]
    if len(c) != 5 or min(c) < 0:
        raise ValueError(f"need 5 non-negative pentanomial counts, got {counts!r}")
    n = sum(c)
    if n <= 0:
        return 0.0, 0.5, 0.0
    if regularize:
        c = [x if x > 0 else REGULARIZE for x in c]
    tot = sum(c)
    mean = sum(x * s for x, s in zip(c, PAIR_SCORES)) / tot
    var = sum(x * (s - mean) ** 2 for x, s in zip(c, PAIR_SCORES)) / tot
    return n, mean, var


def llr(counts, elo0: float, elo1: float) -> float:
    """Normalised pentanomial LLR of elo1 versus elo0 (0 with no pairs)."""
    n, mean, var = _moments(counts, True)
    if n == 0 or var <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)


def elo_estimate(counts) -> tuple[float, float, float]:
    """(elo, ci_low, ci_high): Elo of the mean pair score with a 95% normal interval (nan if N < 2)."""
    n, mean, var = _moments(counts, False)
    if n < 2:
        nan = float("nan")
        return (score_to_elo(mean) if n else nan), nan, nan
    se = math.sqrt(var / n)
    return score_to_elo(mean), score_to_elo(mean - Z95 * se), score_to_elo(mean + Z95 * se)


@dataclass(frozen=True)
class SprtConfig:
    elo0: float = 0.0
    elo1: float = 21.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 2000
    min_pairs: int = 20  # no verdict before this many pairs (the variance estimate is noise below)

    def __post_init__(self):
        if not self.elo1 > self.elo0:
            raise ValueError(f"SPRT needs elo1 > elo0, got {self.elo0},{self.elo1}")
        if not (0 < self.alpha < 1 and 0 < self.beta < 1):
            raise ValueError("SPRT alpha and beta must be in (0, 1)")

    @property
    def bounds(self) -> tuple[float, float]:
        return sprt_bounds(self.alpha, self.beta)

    @property
    def max_pairs(self) -> int:
        return self.max_games // 2


def parse_sprt(text: str, max_games: int = 2000) -> SprtConfig:
    """``elo0,elo1[,alpha,beta]``."""
    parts = [p.strip() for p in text.split(",")]
    if not 2 <= len(parts) <= 4 or not all(parts):
        raise ValueError("--sprt expects elo0,elo1[,alpha,beta]")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"--sprt: not numbers: {text!r}") from None
    kw = dict(zip(("elo0", "elo1", "alpha", "beta"), vals))
    return SprtConfig(max_games=max_games, **kw)


def decide(counts, cfg: SprtConfig) -> str | None:
    """``"H1"``, ``"H0"``, ``"max"`` (pair budget used up) or None (keep playing)."""
    pairs = sum(counts)
    if pairs >= cfg.min_pairs:
        lo, hi = cfg.bounds
        value = llr(counts, cfg.elo0, cfg.elo1)
        if value >= hi:
            return "H1"
        if value <= lo:
            return "H0"
    if pairs >= cfg.max_pairs:
        return "max"
    return None


VERDICTS = {"H1": "SPRT: H1 accepted", "H0": "SPRT: H0 accepted",
            "max": "SPRT: inconclusive at max games"}


def status_text(counts, cfg: SprtConfig) -> str:
    """``SPRT LLR 1.23 [-2.94, 2.94] pairs [0, 3, 10, 4, 1]``."""
    lo, hi = cfg.bounds
    return (f"SPRT LLR {llr(counts, cfg.elo0, cfg.elo1):.2f} [{lo:.2f}, {hi:.2f}] "
            f"pairs {list(counts)}")


def summary_text(counts, cfg: SprtConfig) -> str:
    elo, lo, hi = elo_estimate(counts)
    return (f"{status_text(counts, cfg)} ({sum(counts)} pairs, H0 elo {cfg.elo0:g} vs H1 elo "
            f"{cfg.elo1:g}, alpha {cfg.alpha:g} beta {cfg.beta:g}); "
            f"Elo {elo:+.1f} (95% CI {lo:+.1f}, {hi:+.1f})")
