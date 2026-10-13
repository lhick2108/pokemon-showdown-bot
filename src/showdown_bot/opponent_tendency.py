"""Opponent tendency model: remember how THIS opponent behaves and exploit it safely (589.40).

The root search treats the opponent as a game-theoretic adversary, so it has no memory
of a human who, say, always uses a setup move on the turn we could Sucker Punch. This
module adds that memory and the math to use it without giving up the safety of the
equilibrium:

1. ``TendencyTracker`` consumes the raw protocol (``OpponentModelPlayerMixin`` feeds it,
   like ``sleep.py``). Each turn it records the opponent's first action as a class
   (:data:`CLASSES`: attack, priority, status, protect, switch) under the *context* the
   decision was made in (:class:`Context`: could we Sucker Punch / Thunderclap, did we
   threaten a KO, could they Protect). Counts are Dirichlet counts per context.
2. ``posterior_classes`` blends the opponent's Nash class mix with the observed counts
   (posterior mean with ``kappa`` pseudo-observations of the Nash mix, so a few
   observations move it only a little) and ``blend_belief`` spreads that over the engine's
   opponent columns (within a class the Nash weights are kept).
3. ``safe_exploit`` best-responds to that belief but mixes the best response into our Nash
   strategy only as far as (a) the evidence supports (``n / (n + kappa)``) and (b) the
   worst-case loss against ANY opponent stays below ``delta`` (win probability).
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

ATTACK, PRIORITY, STATUS, PROTECT, SWITCH = "attack", "priority", "status", "protect", "switch"
CLASSES = (ATTACK, PRIORITY, STATUS, PROTECT, SWITCH)

# Moves that only work if the target is about to attack (or that punish a status move).
PRIORITY_CONDITIONAL = frozenset({"suckerpunch", "thunderclap", "upperhand"})
PROTECT_MOVES = frozenset({
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
    "burningbulwark", "obstruct", "maxguard", "endure",
})

DEFAULT_KAPPA = 4.0  # pseudo-observations of the Nash class mix
DEFAULT_DELTA = 0.05  # max worst-case win-probability loss against Nash we accept
LOW_HP = 0.35
STRONG_BP = 60


def _id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_MOVES: dict[str, Any] | None = None


def _move_data() -> Mapping[str, Any]:
    global _MOVES
    if _MOVES is None:
        try:
            from poke_env.data import GenData

            _MOVES = GenData.from_gen(9).moves
        except Exception:  # pragma: no cover - data always ships with poke-env
            _MOVES = {}
    return _MOVES


def classify_move(move_id: str, category: str | None = None, priority: int | None = None) -> str:
    """Action class of using ``move_id`` (category / priority default to the move data)."""
    mid = _id(move_id)
    if mid in PROTECT_MOVES:
        return PROTECT
    if category is None or priority is None:
        entry = _move_data().get(mid) or {}
        category = category if category is not None else entry.get("category")
        priority = priority if priority is not None else entry.get("priority", 0)
    category = str(category or "").lower()
    if category in ("", "status"):
        return STATUS
    return PRIORITY if int(priority or 0) > 0 else ATTACK


def classify_choice(choice: str) -> str:
    """Action class of an engine choice string (``"tackle"``, ``"switch x"``, ``"tackle-tera"``)."""
    choice = str(choice).strip().lower()
    if choice.startswith("switch"):
        return SWITCH
    if choice.endswith("-tera"):
        choice = choice[: -len("-tera")]
    if choice in ("", "none", "no move"):
        return STATUS
    entry = _move_data().get(_id(choice))
    if entry is None:  # hidden power variants and the like are attacks
        return ATTACK if choice.startswith("hiddenpower") else STATUS
    return classify_move(choice)


@dataclass(frozen=True)
class Context:
    """What we held when the opponent chose; the guessing-game features."""

    cond_priority: bool = False  # we can Sucker Punch / Thunderclap / Upper Hand
    threat: bool = False  # we threaten a KO: they are low, or we hold a strong super-effective hit
    can_protect: bool = False  # they have a revealed Protect-like move that is not on cooldown

    @property
    def key(self) -> str:
        return "".join(
            flag if on else "-"
            for flag, on in (("P", self.cond_priority), ("T", self.threat), ("D", self.can_protect))
        )


def context_from_battle(battle: Any, last_class: str | None = None) -> Context:
    """Context of the decision we are about to make (duck-typed poke-env battle)."""
    try:
        moves = list(battle.available_moves or [])
    except Exception:
        moves = []
    cond = any(_id(getattr(m, "id", "")) in PRIORITY_CONDITIONAL for m in moves)
    threat = False
    can_protect = False
    try:
        opp = battle.opponent_active_pokemon
        if opp is not None:
            damaging = [m for m in moves if getattr(m, "base_power", 0) and
                        str(getattr(getattr(m, "category", ""), "name", getattr(m, "category", ""))).lower() != "status"]
            hp = getattr(opp, "current_hp_fraction", 1.0)
            if damaging and hp is not None and hp <= LOW_HP:
                threat = True
            if not threat:
                for m in damaging:
                    try:
                        if m.base_power >= STRONG_BP and opp.damage_multiplier(m) >= 2:
                            threat = True
                            break
                    except Exception:
                        continue
            known = [_id(getattr(m, "id", m)) for m in getattr(opp, "moves", {}) or {}]
            can_protect = any(m in PROTECT_MOVES for m in known) and last_class != PROTECT
    except Exception:
        pass
    return Context(cond, threat, can_protect)


class TendencyTracker:
    """Per-battle record of the opponent's action class under each context."""

    def __init__(self) -> None:
        self.counts: dict[str, dict[str, float]] = {}
        self.history: list[tuple[int, str, str]] = []  # (turn, context key, class)
        self.turn = 0
        self._contexts: dict[int, Context] = {}
        self._acted = True  # nothing to record before the first |turn|
        self._upkeep = False
        self.last_class: str | None = None

    # -- protocol ------------------------------------------------------------

    def set_context(self, turn: int, context: Context) -> None:
        """Context of our decision for ``turn`` (call before the turn's events arrive)."""
        self._contexts[int(turn)] = context

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
        elif kind == "upkeep":
            self._upkeep = True
        elif kind in ("move", "switch") and len(event) >= 4 and not self._acted and not self._upkeep:
            if not str(event[2]).startswith(opp_role):
                return
            if kind == "move":
                if any(str(t).startswith("[from]") for t in event[4:]):
                    return  # Sleep Talk / Magic Bounce / Instruct: not a chosen move
                self._record(classify_move(event[3]))
            else:
                self._record(SWITCH)

    def observe_events(self, events: Sequence[Sequence[str]], opp_role: str | None) -> None:
        if opp_role not in ("p1", "p2"):
            return
        for event in events:
            self.observe(event, opp_role)

    def _record(self, cls: str) -> None:
        self._acted = True
        context = self._contexts.get(self.turn)
        if context is None:
            return  # we made no decision this turn (forced), nothing to condition on
        self.counts.setdefault(context.key, {}).setdefault(cls, 0.0)
        self.counts[context.key][cls] += 1.0
        self.history.append((self.turn, context.key, cls))
        self.last_class = cls

    # -- query ---------------------------------------------------------------

    def counts_for(self, context: Context) -> dict[str, float]:
        return dict(self.counts.get(context.key, {}))


_MAX_BATTLES = 64
_TRACKERS: OrderedDict[str, TendencyTracker] = OrderedDict()


def for_battle(battle_tag: str, *, create: bool = False) -> TendencyTracker | None:
    tag = str(battle_tag).lstrip(">")
    tracker = _TRACKERS.get(tag)
    if tracker is None and create:
        tracker = _TRACKERS[tag] = TendencyTracker()
        while len(_TRACKERS) > _MAX_BATTLES:
            _TRACKERS.popitem(last=False)
    return tracker


def observe_events(battle_tag: str, events: Sequence[Sequence[str]], our_role: str | None) -> None:
    if our_role not in ("p1", "p2"):
        return
    tracker = for_battle(battle_tag, create=True)
    assert tracker is not None
    tracker.observe_events(events, "p2" if our_role == "p1" else "p1")


# ---------------------------------------------------------------------------
# Blending math (pure numpy)
# ---------------------------------------------------------------------------


def posterior_classes(
    nash_mass: Mapping[str, float], counts: Mapping[str, float], kappa: float = DEFAULT_KAPPA
) -> dict[str, float]:
    """Posterior mean of the opponent's class mix.

    ``nash_mass`` is the Nash class distribution over the classes available now (sums to
    1); ``counts`` are observed class counts in this context. Counts of classes that are
    not available now are ignored. Each class gets ``(n_c + kappa * q_c) / (n + kappa)``:
    a Dirichlet prior of ``kappa`` pseudo-observations centred on the Nash mix.
    """
    if kappa < 0:
        raise ValueError("kappa must be >= 0")
    total_q = sum(nash_mass.values())
    q = {c: (m / total_q if total_q > 0 else 1.0 / len(nash_mass)) for c, m in nash_mass.items()}
    n_c = {c: max(0.0, float(counts.get(c, 0.0))) for c in q}
    n = sum(n_c.values())
    denom = n + kappa
    if denom <= 0:
        return q
    return {c: (n_c[c] + kappa * q[c]) / denom for c in q}


def blend_belief(
    col: np.ndarray, classes: Sequence[str], counts: Mapping[str, float], kappa: float = DEFAULT_KAPPA
) -> tuple[np.ndarray, dict[str, float], float]:
    """Opponent belief over the engine's columns.

    ``col`` is the Nash column strategy, ``classes[j]`` column j's action class. Returns
    ``(belief, class posterior, n)`` with ``n`` the evidence count over available classes.
    Within a class the Nash weights are kept (uniform if Nash gives the class no mass).
    """
    col = np.asarray(col, dtype=float)
    mass: dict[str, float] = {}
    for j, c in enumerate(classes):
        mass[c] = mass.get(c, 0.0) + float(col[j])
    post = posterior_classes(mass, counts, kappa)
    n = sum(max(0.0, float(counts.get(c, 0.0))) for c in mass)
    belief = np.zeros(len(classes))
    sizes = {c: classes.count(c) for c in mass}
    for j, c in enumerate(classes):
        within = col[j] / mass[c] if mass[c] > 1e-12 else 1.0 / sizes[c]
        belief[j] = post[c] * within
    return belief / belief.sum(), post, n


def safe_exploit(
    A: np.ndarray,
    row: np.ndarray,
    value: float,
    belief: np.ndarray,
    weight: float,
    delta: float = DEFAULT_DELTA,
) -> tuple[np.ndarray, float, int]:
    """Mix a best response to ``belief`` into the Nash row strategy, safely.

    Returns ``(strategy, lam, best_response_row)``. The strategy is
    ``(1 - lam) * row + lam * e_br`` with ``lam <= weight`` and the largest ``lam`` for
    which no opponent column holds us below ``value - delta`` (the safe-exploitation
    constraint: a wrong belief costs at most ``delta`` of win probability).
    """
    A = np.asarray(A, dtype=float)
    row = np.asarray(row, dtype=float)
    expected = A @ belief
    # best response; among near-ties prefer the row Nash already likes
    br = int(np.argmax(expected + 1e-9 * row))
    a = row @ A  # our payoff per opponent column under Nash
    b = A[br]
    lam = float(max(0.0, min(1.0, weight)))
    floor = value - delta
    for j in range(A.shape[1]):
        if b[j] < a[j]:
            allowed = (a[j] - floor) / (a[j] - b[j])
            lam = min(lam, max(0.0, allowed))
    out = (1.0 - lam) * row
    out[br] += lam
    return out, lam, br


def exploit_world(
    A: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    value: float,
    opp_classes: Sequence[str],
    counts: Mapping[str, float],
    kappa: float = DEFAULT_KAPPA,
    delta: float = DEFAULT_DELTA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Our exploiting row strategy for one world's matrix game plus diagnostics."""
    belief, post, n = blend_belief(col, list(opp_classes), counts, kappa)
    weight = n / (n + kappa) if (n + kappa) > 0 else 0.0
    strategy, lam, br = safe_exploit(A, row, value, belief, weight, delta)
    return strategy, {"post": post, "n": n, "lambda": lam, "br": br, "weight": weight}
