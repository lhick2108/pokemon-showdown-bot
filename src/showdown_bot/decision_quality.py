"""Decision-quality helpers for the root policy (bead 589.47).

Three independent, default-off adjustments to what ``SearchPlayer`` picks from the
aggregated root results; all are pure functions of plain data (unit-tested without a
server) except :func:`noop_choices`, which asks the engine.

* flat-root escalation: :func:`flat_root` says whether the best two options (tera
  siblings merged) are too close in mean score to trust, so the search should continue;
* no-op pruning: :func:`noop_choices` finds moves the engine proves have no effect
  against any reply, :func:`prune_policy` drops them;
* tera edge: :func:`fold_tera` merges an ``X-tera`` option into ``X`` unless it beats the
  plain move's mean score by a margin (the once-per-game tera is not worth a coin flip).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("showdown_bot.decision_quality")

TERA_SUFFIX = "-tera"

# Instructions that only record bookkeeping (PP, last-used move, damage-dealt memory,
# switch-out-move memory): a move whose outcomes differ from "do nothing" only in these
# leaves the battle state unchanged.
BOOKKEEPING = (
    "DecrementPP",
    "SetLastUsedMove",
    "ChangeDamageDealtDamage",
    "ChangeDamageDealtMoveCatagory",
    "ToggleDamageDealtHitSubstitute",
    "SideOneMoveSecondSwitchOutMove",
    "SideTwoMoveSecondSwitchOutMove",
)


def option_stats(results: list[tuple]) -> dict[str, tuple[int, float]]:
    """``choice -> (visits, total_score)`` summed over the worlds."""
    stats: dict[str, tuple[int, float]] = {}
    for result in results:
        if result[0] <= 0:
            continue
        for choice, visits, score in result[1]:
            v, s = stats.get(choice, (0, 0.0))
            stats[choice] = (v + int(visits), s + float(score))
    return stats


def option_means(results: list[tuple]) -> dict[str, float]:
    """Visit-weighted mean score of each choice over the worlds."""
    return {c: (s / v if v else 0.0) for c, (v, s) in option_stats(results).items()}


def plain(choice: str) -> str:
    return choice[: -len(TERA_SUFFIX)] if choice.endswith(TERA_SUFFIX) else choice


def fold_tera(
    policy: dict[str, float], means: dict[str, float], margin: float
) -> tuple[dict[str, float], list[str]]:
    """Merge ``X-tera`` into ``X`` unless its mean exceeds X's by ``margin``.

    Returns the new policy and the folded tera choices. Does nothing for ``margin <= 0``
    or when the plain sibling is absent from the policy.
    """
    if margin <= 0:
        return dict(policy), []
    out = dict(policy)
    folded: list[str] = []
    for choice in list(policy):
        if not choice.endswith(TERA_SUFFIX):
            continue
        base = plain(choice)
        if base not in out or choice not in out:
            continue
        edge = means.get(choice, 0.0) - means.get(base, 0.0)
        if edge < margin:
            out[base] += out.pop(choice)
            folded.append(choice)
    return out, folded


def prune_policy(policy: dict[str, float], noop: set[str]) -> tuple[dict[str, float], list[str]]:
    """Drop the proven no-op choices unless that would drop everything."""
    dropped = [c for c in policy if c in noop]
    if not dropped or len(dropped) == len(policy):
        return dict(policy), []
    return {c: m for c, m in policy.items() if c not in noop}, dropped


def flat_root(policy: dict[str, float], results: list[tuple], gap: float) -> bool:
    """True when the two best options (by mass, tera siblings merged) differ by < ``gap``."""
    stats = option_stats(results)
    groups: dict[str, list[float]] = {}  # base -> [mass, visits, score]
    for choice, mass in policy.items():
        g = groups.setdefault(plain(choice), [0.0, 0.0, 0.0])
        v, s = stats.get(choice, (0, 0.0))
        g[0] += mass
        g[1] += v
        g[2] += s
    if len(groups) < 2:
        return False
    top = sorted(groups.values(), key=lambda g: g[0], reverse=True)[:2]
    if min(g[1] for g in top) <= 0:
        return False
    return abs(top[0][2] / top[0][1] - top[1][2] / top[1][1]) < gap


def adjust_policy(
    policy: dict[str, float],
    results: list[tuple],
    *,
    noop: set[str] | None = None,
    tera_margin: float = 0.0,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Apply tera folding then no-op pruning; also returns what was changed."""
    means = option_means(results)
    out, folded = fold_tera(policy, means, tera_margin)
    pruned: list[str] = []
    if noop:
        out, pruned = prune_policy(out, noop)
    return out, {"tera_folded": folded, "noop_pruned": pruned}


# -- engine-side no-op detection -----------------------------------------------


def _signature(outcomes: list[Any]) -> dict[tuple[str, ...], float]:
    sig: dict[tuple[str, ...], float] = {}
    for outcome in outcomes:
        key = tuple(
            text
            for text in (str(i) for i in outcome.instruction_list)
            if not text.startswith(BOOKKEEPING)
        )
        sig[key] = sig.get(key, 0.0) + float(outcome.percentage)
    return sig


def _same(a: dict, b: dict) -> bool:
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) < 0.05 for k in set(a) | set(b))


def _opponent_replies(state: Any, switches: bool = True) -> list[str]:
    side = state.side_two
    active = int(side.active_index)
    replies = ["none"]
    for mon_index, mon in enumerate(side.pokemon):
        if mon_index == active:
            for move in mon.moves:
                if not move.disabled and move.pp > 0:
                    replies.append(str(move.id).lower())
        elif switches and mon.hp > 0:
            replies.append(str(mon.id).lower())  # the engine takes the bare id
    return replies


def noop_choices_in_state(
    state: Any, choices: list[str], switches: bool = True
) -> set[str]:
    """Move choices whose outcome equals "do nothing" against every opponent reply.

    ``generate_instructions(state, "none", reply)`` is the baseline (the reply's own
    effects, residuals, ...); a move is a no-op when its outcome distribution, ignoring
    PP / last-used bookkeeping, is the same for every reply. Any engine error counts as
    "not provably a no-op". With ``switches`` (strict, default) the opponent's possible
    switches count as replies too, so a status move into an already-statused target is
    kept (it hits the switch-in); ``switches=False`` also prunes those.
    """
    from poke_engine import generate_instructions

    candidates = [c for c in choices if not c.startswith("switch") and not c.endswith(TERA_SUFFIX)]
    replies = _opponent_replies(state, switches)
    baselines: dict[str, dict] = {}
    for reply in replies:
        try:
            baselines[reply] = _signature(generate_instructions(state, "none", reply))
        except BaseException:  # noqa: BLE001 - pyo3 PanicException is a BaseException
            continue
    if not baselines:
        return set()
    noop: set[str] = set()
    for choice in candidates:
        try:
            if all(
                _same(_signature(generate_instructions(state, choice, reply)), base)
                for reply, base in baselines.items()
            ):
                noop.add(choice)
        except BaseException:  # noqa: BLE001
            continue
    return noop


def noop_choices(
    state_strings: list[str], choices: list[str], switches: bool = True
) -> set[str]:
    """Choices that are no-ops in every sampled world."""
    from poke_engine import State

    result: set[str] | None = None
    for text in state_strings:
        try:
            found = noop_choices_in_state(State.from_string(text), choices, switches)
        except BaseException:  # noqa: BLE001
            found = set()
        result = found if result is None else result & found
        if not result:
            return set()
    return result or set()


# -- KO tie-break -----------------------------------------------------------------


def _opp_move_replies(state: Any) -> list[str]:
    mon = state.side_two.pokemon[int(state.side_two.active_index)]
    return [str(m.id).lower() for m in mon.moves if not m.disabled and int(m.pp) > 0]


def ko_probability(state: Any, mine: str, theirs: str) -> float:
    """Probability that side two's active is at 0 HP after ``mine`` vs ``theirs`` (one turn).

    ``State.apply_instructions`` returns a new State (it does not mutate).
    """
    from poke_engine import generate_instructions

    idx = int(state.side_two.active_index)
    total = 0.0
    for outcome in generate_instructions(state, mine, theirs):
        child = state.apply_instructions(outcome)
        if int(child.side_two.pokemon[idx].hp) <= 0:
            total += float(outcome.percentage) / 100.0
    return total


def is_ko_option(state: Any, choice: str) -> bool:
    """``choice`` KOs the opponent's active with >= 50% probability against a majority of its
    non-switch replies (same test as scripts/missed_kos.py). Switches never count."""
    if choice.startswith("switch"):
        return False
    replies = _opp_move_replies(state)
    if not replies:
        return False
    try:
        hits = sum(ko_probability(state, choice, t) >= 0.5 for t in replies)
    except BaseException:  # noqa: BLE001 - pyo3 PanicException is a BaseException
        return False
    return hits * 2 > len(replies)


def ko_tiebreak(
    policy: dict[str, float],
    results: list[tuple],
    states: list[Any],
    eps: float,
) -> tuple[dict[str, float], str | None]:
    """Prefer a KO option over the chosen (top-mass) option when its mean score is within ``eps``.

    ``states`` are the parsed worlds aligned with ``results``. An option is a KO option when
    it KOs in a majority of the worlds (see :func:`is_ko_option`). Returns the new policy
    (all mass on the KO option) and that option, or ``(policy, None)`` when nothing changed.
    Does nothing for ``eps <= 0`` or when the chosen option already KOs.
    """
    if eps <= 0 or not policy or not states:
        return policy, None
    means = option_means(results)
    best = max(policy, key=lambda c: policy[c])
    if best not in means:
        return policy, None
    floor = means[best] - eps
    rivals = [c for c in policy if c != best and means.get(c, -1.0) >= floor]
    if not rivals:
        return policy, None

    def votes(choice: str) -> bool:
        return sum(is_ko_option(s, choice) for s in states) * 2 > len(states)

    if votes(best):
        return policy, None
    kos = [c for c in rivals if votes(c)]
    if not kos:
        return policy, None
    pick = max(kos, key=lambda c: means[c])
    return {pick: 1.0}, pick
