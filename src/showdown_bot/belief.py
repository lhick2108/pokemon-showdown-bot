"""Soft Bayesian reweighting of the candidate opponent sets (bead showdown-bot-1wu).

Opponent inference elsewhere is HARD: a set that contradicts evidence is rejected, the rest
keep their prior occurrence weight.  With ``belief_reweight`` on, every opposing Pokemon's
``SetTracker`` also keeps a multiplicative *likelihood weight* per candidate full set, and
sampling draws sets in proportion to ``prior count x likelihood weight`` (PokAITrainer-style
belief update):

* outcome conditioning - ``damage_likelihood``: how probable the observed HP change is under
  each candidate (the engine's maximum roll gives the 16-roll damage range);
  ``speed_tie_factor``: an exact speed tie makes either move order a coin flip;
* action conditioning - ``action_factors``: the opponent's root visit share of the action it
  actually took, in each searched world, credited to the active Pokemon's set in that world.

Evidence is soft: factors are tempered, and the stored weights are floored at ``floor`` times
the pool maximum, so noisy evidence dampens a set but never eliminates it.  Hard elimination
(reveals, rejected damage windows) stays in ``SetTracker`` and is unchanged.

Everything here is pure Python and cheap; ``STATS`` accumulates the time spent.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_FLOOR = 0.05  # no set drops below this share of the most likely one
DEFAULT_TAU = 0.5  # tempering exponent for action evidence
ACTION_P_FLOOR = 0.05  # an observed action that a world gave ~no visits still counts as this
OUTLIER = 0.05  # damage-likelihood mixture weight of a uniform "model is wrong" component
SIGMA_REL = 0.02  # engine-vs-Showdown damage error, share of the maximum roll
SIGMA_HP = 0.5  # ...plus this many HP points (exact HP) or percentage points
_ROLLS = tuple((85 + i) / 100.0 for i in range(16))


@dataclass
class BeliefStats:
    """Process-wide cost accounting (seconds and call counts)."""

    damage_seconds: float = 0.0
    damage_updates: int = 0
    speed_seconds: float = 0.0
    speed_updates: int = 0
    action_seconds: float = 0.0
    action_updates: int = 0
    action_skipped: int = 0
    sample_seconds: float = 0.0
    sample_calls: int = 0

    @property
    def seconds(self) -> float:
        return self.damage_seconds + self.speed_seconds + self.action_seconds + self.sample_seconds

    def summary(self) -> str:
        return (
            f"belief_seconds={self.seconds:.3f} damage={self.damage_updates}"
            f"/{self.damage_seconds:.3f}s speed={self.speed_updates}/{self.speed_seconds:.3f}s "
            f"action={self.action_updates}(skipped {self.action_skipped})/{self.action_seconds:.3f}s "
            f"sample={self.sample_calls}/{self.sample_seconds:.3f}s"
        )


STATS = BeliefStats()


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def normalise_factors(
    factors: Sequence[float], floor: float = DEFAULT_FLOOR
) -> list[float]:
    """Factors divided by their maximum and floored: every value lands in ``[floor, 1]``."""
    if not factors:
        return []
    top = max(factors)
    if not (top > 0 and math.isfinite(top)):
        return [1.0] * len(factors)
    return [max(float(f) / top, floor) for f in factors]


def effective_weights(
    counts: Sequence[float], factors: Sequence[float], floor: float = DEFAULT_FLOOR
) -> list[float]:
    """Sampling weights: prior count x (floored, max-normalised) likelihood factor."""
    return [c * f for c, f in zip(counts, normalise_factors(factors, floor))]


def entropy_bits(weights: Sequence[float]) -> float:
    total = sum(w for w in weights if w > 0)
    if total <= 0:
        return 0.0
    return -sum(w / total * math.log2(w / total) for w in weights if w > 0)


def temper(ratio: float, tau: float) -> float:
    """``ratio ** tau`` (tau 0 = ignore the evidence, 1 = take it at face value)."""
    if ratio <= 0:
        return 0.0
    return ratio ** tau


# ---------------------------------------------------------------------------
# Outcome likelihoods
# ---------------------------------------------------------------------------


def damage_likelihood(
    high: float, observed: float, exact_hp: bool, defender_hp: float
) -> float | None:
    """Density of an observed damage figure given the engine's maximum roll ``high``.

    Units are HP points when ``exact_hp`` (we were hit) else percent of the defender's max HP;
    ``defender_hp`` is its HP before the hit in the same units.  Showdown picks one of 16 rolls
    (85%..100% of the maximum) uniformly; each roll is smeared by a Gaussian for the engine's
    arithmetic error and the observation's rounding, and mixed with a small uniform component
    for model error.  Returns None when every roll would have knocked the defender out (the
    hit then says nothing about the roll).  Crits are excluded upstream (``-crit`` spoils the
    observation) and Showdown's crit chance is the same for every candidate, so it cancels.
    """
    if not (high > 0 and math.isfinite(high)):
        return None
    if high * _ROLLS[0] >= defender_hp:
        return None
    sigma = SIGMA_REL * high + SIGMA_HP
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    total = 0.0
    for r in _ROLLS:
        value = min(high * r, defender_hp)
        z = (observed - value) / sigma
        total += math.exp(-0.5 * z * z)
    mix = total / len(_ROLLS) * norm
    uniform = 1.0 / max(defender_hp, 1.0)
    return (1.0 - OUTLIER) * mix + OUTLIER * uniform


def group_damage_factors(
    likelihoods: Sequence[float | None],
) -> list[float]:
    """Relative factors from per-group likelihoods; unknown (None) groups get the maximum."""
    known = [x for x in likelihoods if x is not None and x > 0]
    top = max(known) if known else 1.0
    return [1.0 if x is None else max(x, 0.0) / top for x in likelihoods]


def speed_tie_factor(speed: float, our_speed: float) -> float:
    """P(observed move order) relative to a non-tied candidate: a tie is a coin flip."""
    return 0.5 if abs(float(speed) - float(our_speed)) < 0.5 else 1.0


# ---------------------------------------------------------------------------
# Action conditioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldAction:
    """One searched world: the active opponent's set and its root visit shares."""

    set_key: tuple | None  # None: unmodelled (disguised, role-sampled...): no update
    shares: Mapping[str, float]  # engine choice -> share of side_two visits (sums to 1)


@dataclass
class ActionRecord:
    """What the last solve said about the opponent's action, per world."""

    turn: int
    tracker_key: str
    worlds: list[WorldAction] = field(default_factory=list)


def visit_shares(options: Sequence[tuple[str, int]]) -> dict[str, float]:
    """Visit counts per option as shares; options of the same choice are summed."""
    total = sum(max(0, int(v)) for _, v in options)
    out: dict[str, float] = {}
    if total <= 0:
        return out
    for choice, visits in options:
        out[str(choice).lower()] = out.get(str(choice).lower(), 0.0) + max(0, int(visits)) / total
    return out


def action_factors(
    worlds: Sequence[WorldAction],
    observed: str,
    tau: float = DEFAULT_TAU,
    p_floor: float = ACTION_P_FLOOR,
) -> dict[tuple, float] | None:
    """Likelihood factor per set for the opponent's observed action.

    ``p_w`` is world ``w``'s visit share of ``observed`` (floored at ``p_floor`` so a surprise
    dampens rather than eliminates).  A set's credit is the mean ``p_w`` over the worlds that
    contain it, relative to the mean over all worlds, raised to ``tau`` (DUCT visits from 2-4
    worlds are noisy).  Sets in no world get no factor (1).  Returns None when nothing can be
    learned: no world modelled, or ``observed`` is not an option in any of them (a choice
    mapping failure), or every world agrees.
    """
    observed = observed.lower()
    usable = [w for w in worlds if w.set_key is not None and w.shares]
    if not usable or not any(observed in w.shares for w in usable):
        return None
    ps = [max(w.shares.get(observed, 0.0), p_floor) for w in usable]
    mean_all = sum(ps) / len(ps)
    per_set: dict[tuple, list[float]] = {}
    for w, p in zip(usable, ps):
        per_set.setdefault(w.set_key, []).append(p)  # type: ignore[arg-type]
    if len(per_set) < 2:
        return None
    return {
        key: temper((sum(values) / len(values)) / mean_all, tau)
        for key, values in per_set.items()
    }


# ---------------------------------------------------------------------------
# Observed opponent action (from the protocol)
# ---------------------------------------------------------------------------


def _id(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


class ActionWatcher:
    """The opponent's first chosen action of each turn, as an engine choice string.

    ``observed[turn]`` is ``"<moveid>"``, ``"<moveid>-tera"`` or ``"switch <speciesid>"``.
    Moves with a ``[from]`` tag (Sleep Talk, Magic Bounce, locked moves...) and anything after
    ``|upkeep|`` (replacements) are not choices.
    """

    def __init__(self) -> None:
        self.turn = 0
        self.observed: dict[int, str] = {}
        self._acted = True
        self._upkeep = False
        self._tera = False

    def observe(self, event: Sequence[str], opp_role: str) -> None:
        if len(event) < 2:
            return
        kind = str(event[1])
        if kind == "turn" and len(event) >= 3:
            try:
                self.turn = int(event[2])
            except ValueError:
                return
            self._acted = False
            self._upkeep = False
            self._tera = False
        elif kind == "upkeep":
            self._upkeep = True
        elif kind == "-terastallize" and len(event) >= 3:
            if str(event[2]).startswith(opp_role) and not self._acted:
                self._tera = True
        elif kind in ("move", "switch") and len(event) >= 4 and not self._acted and not self._upkeep:
            if not str(event[2]).startswith(opp_role):
                return
            if kind == "move":
                if any(str(t).startswith("[from]") for t in event[4:]):
                    return
                choice = _id(event[3]) + ("-tera" if self._tera else "")
            else:
                choice = "switch " + _id(str(event[3]).split(",")[0])
            self._acted = True
            self.observed[self.turn] = choice


class Timer:
    """``with Timer("damage_seconds"):`` adds the elapsed time to ``STATS``."""

    def __init__(self, attr: str):
        self.attr = attr

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        setattr(STATS, self.attr, getattr(STATS, self.attr) + time.perf_counter() - self._start)
