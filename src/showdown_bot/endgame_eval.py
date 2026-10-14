"""Engine-only A/B of endgame mode (backs ``scripts/endgame_ab.py``).

Positions are full-information "truth" states from self-play records where few Pokemon
are left. Each position is played to the end twice with the seats swapped: bot A (e.g.
``endgame=safest``) against bot B (plain MCTS), both choosing every move by search on the
full-information state (no hidden information, so this isolates endgame search quality), the
turn outcome sampled with the engine's ``generate_instructions`` percentages. A's score per
position is the mean of its two games (win 1, tie 0.5, loss 0); the CI is over positions.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from showdown_bot import endgame as endgame_module
from showdown_bot import fork_augment as fa
from showdown_bot import recorder
from showdown_bot.players import _worker
from showdown_bot.players.search import ARGMAX, MIXED, NASH, SearchConfig, aggregate_policy, select_choice

NO_MOVE_LABELS = ("none", "no move")


def swap_sides(state: str) -> str:
    """The same position with the two sides exchanged (side two becomes side one)."""
    parts = state.split("/")
    parts[0], parts[1] = parts[1], parts[0]
    return "/".join(parts)


# ---------------------------------------------------------------------------------------------
# Position extraction
# ---------------------------------------------------------------------------------------------


def alive_counts(state: str) -> tuple[int, int]:
    from poke_engine import State

    parsed = State.from_string(state)
    return fa.side_alive(parsed, 1), fa.side_alive(parsed, 2)


def is_endgame_position(state: str, max_alive: int, max_side: int = 0) -> bool:
    """A decision state (both actives alive, no forced switch) with few Pokemon left."""
    if not fa.is_forkable(state):
        return False
    one, two = alive_counts(state)
    if one + two > max_alive or one < 1 or two < 1:
        return False
    return not (max_side and (one > max_side or two > max_side))


def extract_positions(
    sources: list[str], limit: int, max_alive: int, rng: random.Random, max_side: int = 0
) -> list[dict[str, Any]]:
    """Up to ``limit`` positions, one per game: the FIRST recorded truth state of a battle
    with total alive <= ``max_alive``. Both players' files of a battle count once."""
    files = sorted({p for r in sources for p in recorder.iter_game_files(r)})
    rng.shuffle(files)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in files:
        if len(out) >= limit:
            break
        try:
            game = recorder.read_game(path)
        except (OSError, ValueError, EOFError):
            continue
        tag = str(game["header"].get("battle_tag", path))
        if tag in seen:
            continue
        seen.add(tag)
        for i, d in enumerate(game["decisions"]):
            truth = d.get("truth")
            if truth and is_endgame_position(truth["state"], max_alive, max_side):
                one, two = alive_counts(truth["state"])
                out.append({
                    "state": truth["state"], "battle_tag": tag, "turn": d.get("turn"),
                    "decision_idx": i, "alive": [one, two], "file": os.path.basename(str(path)),
                })
                break
    return out


# ---------------------------------------------------------------------------------------------
# Move choice (one bot, full information)
# ---------------------------------------------------------------------------------------------


@dataclass
class Choice:
    choice: str  # engine option name of the side that chose (e.g. "earthquake", "switch x")
    endgame: bool = False
    depth: int | None = None
    fell_back: bool = False
    info: dict[str, Any] = field(default_factory=dict)


def _mcts_choice(text: str, cfg: SearchConfig, rng: random.Random) -> str:
    from poke_engine import State

    state = State.from_string(text)
    res = fa.search_both(state, cfg.search_time_ms, cfg.evaluator, cfg.evaluator_scale)
    options = [(o.move_choice, int(o.visits), float(o.total_score)) for o in res.side_one]
    if not options:
        return "none"
    if len(options) == 1:
        return options[0][0]
    policy = aggregate_policy([(int(res.total_visits), options)])
    choice = select_choice(policy, ARGMAX if cfg.mode == ARGMAX else MIXED, rng, cfg.mixed_threshold)
    return choice if choice is not None else options[0][0]


def choose_move(state: str, side: int, cfg: SearchConfig, rng: random.Random) -> Choice:
    """Side ``side`` (1 or 2) picks a move on the full-information ``state``."""
    text = state if side == 1 else swap_sides(state)
    if cfg.endgame != endgame_module.OFF:
        one, two = alive_counts(text)
        if endgame_module.triggered(cfg, one, two):
            try:
                world = endgame_module.parse_world(_worker.endgame_world(text, cfg.endgame_time_ms))
                if world is not None:
                    if len(world.ours) == 1:
                        return Choice(world.ours[0], True, world.depth)
                    policy, info = endgame_module.aggregate(cfg.endgame, [world], cfg.nash_floor)
                    mode = NASH if cfg.endgame == endgame_module.NASH else ARGMAX
                    choice = select_choice(policy, mode, rng) if policy else None
                    if choice is not None:
                        return Choice(choice, True, world.depth, info={"values": policy})
            except Exception:
                pass  # fall back to MCTS below
            return Choice(_mcts_choice(text, cfg, rng), False, None, fell_back=True)
    return Choice(_mcts_choice(text, cfg, rng), False)


# ---------------------------------------------------------------------------------------------
# Playout
# ---------------------------------------------------------------------------------------------


@dataclass
class GameResult:
    score: float  # A's score: 1 win, 0.5 tie (or turn cap), 0 loss
    steps: int
    capped: bool
    a_endgame_moves: int = 0  # A decisions made by expectiminimax
    a_fallbacks: int = 0
    a_depths: list[int] = field(default_factory=list)
    seconds: float = 0.0


def play_game(
    state_str: str, a_side: int, a_cfg: SearchConfig, b_cfg: SearchConfig, rng: random.Random,
    max_turns: int = 60,
) -> GameResult:
    """Play ``state_str`` out with bot A on engine side ``a_side`` (1 or 2) against bot B."""
    from poke_engine import State, generate_instructions

    started = time.perf_counter()
    state = State.from_string(state_str)
    result = GameResult(0.5, 0, False)
    for step in range(max_turns):
        done = fa.outcome(state)
        if done is not None:
            result.steps = step
            result.score = _score(done, a_side)
            break
        text = state.to_string()
        choices = {}
        for side in (1, 2):
            is_a = side == a_side
            picked = choose_move(text, side, a_cfg if is_a else b_cfg, rng)
            choices[side] = picked.choice
            if is_a:
                result.a_endgame_moves += picked.endgame
                result.a_fallbacks += picked.fell_back
                if picked.depth is not None:
                    result.a_depths.append(picked.depth)
        branches = generate_instructions(
            state, _engine_choice(choices[1]), _engine_choice(choices[2])
        )
        state = state.apply_instructions(fa.sample_outcome(branches, rng))
    else:
        done = fa.outcome(state)
        result.steps = max_turns
        if done is None:
            result.capped = True
        else:
            result.score = _score(done, a_side)
    result.seconds = time.perf_counter() - started
    return result


def _engine_choice(choice: str) -> str:
    return "none" if choice.lower() in NO_MOVE_LABELS else fa._engine_choice(choice)


def _score(outcome_side_one: str, a_side: int) -> float:
    if outcome_side_one == recorder.RESULT_TIE:
        return 0.5
    won_side_one = outcome_side_one == recorder.RESULT_WIN
    return 1.0 if won_side_one == (a_side == 1) else 0.0


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    """One game (a pool worker entry point): ``task`` has position index, seat, configs, seed."""
    rng = random.Random(f"endgame-ab-{task['seed']}-{task['index']}-{task['a_side']}")
    out = {"index": task["index"], "a_side": task["a_side"], "ok": False}
    try:
        g = play_game(
            task["state"], task["a_side"], task["a_cfg"], task["b_cfg"], rng, task["max_turns"]
        )
        out.update(ok=True, score=g.score, steps=g.steps, capped=g.capped,
                   a_endgame_moves=g.a_endgame_moves, a_fallbacks=g.a_fallbacks,
                   a_depths=g.a_depths, seconds=g.seconds)
    except Exception as exc:  # engine panic etc.: report, keep the pool alive
        out["error"] = f"{type(exc).__name__}: {exc}"
    except BaseException as exc:  # pyo3 PanicException
        out["error"] = f"engine panic: {exc!r}"
    return out


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


def paired_scores(results: list[dict[str, Any]]) -> dict[int, float]:
    """Per-position mean of A's two games (positions with both games ok)."""
    by_pos: dict[int, list[float]] = {}
    for r in results:
        if r.get("ok"):
            by_pos.setdefault(r["index"], []).append(r["score"])
    return {i: sum(v) / len(v) for i, v in by_pos.items() if len(v) == 2}


def summarize(scores: list[float], seed: int = 0, resamples: int = 4000) -> dict[str, float]:
    """Mean of the per-position pair scores with a normal 95% CI and a bootstrap 95% CI."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "boot_lo": float("nan"), "boot_hi": float("nan")}
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(resamples))
    return {
        "n": n, "mean": mean, "se": se, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
        "boot_lo": means[int(0.025 * resamples)], "boot_hi": means[int(0.975 * resamples) - 1],
    }
