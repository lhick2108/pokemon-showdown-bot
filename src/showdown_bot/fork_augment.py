"""Hypothetical-position augmentation: fork recorded truth states, edit, play out engine-only.

PokAITrainer-style augmentation (bead showdown-bot-sd0). A recorded self-play decision has an
exact "truth" State string. We perturb it (HP, status, stat stages, hazards, screens, weather,
terrain), then play the edited position to the end inside the engine with a perfect-information
search on both sides, and write the played game in the self-play record schema
(``docs/selfplay-data.md``, header ``source: "fork"``) so ``train.py --data`` reads it unchanged.

The engine's Python objects are read-only, so edits work on the serialised State string
(``engine/src/state.rs``: ``side1/side2/weather/terrain/trick_room/team_preview``, a side is
``=``-separated, a Pokemon ``,``-separated).
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
from typing import Any

from showdown_bot import recorder
from showdown_bot.players.search import MIXED, MIXED_THRESHOLD, aggregate_policy, select_choice

SOURCE = "fork"
NO_MOVE = "No Move"  # the engine's label for "this side has no choice this step" (waiting side)

# --- State string layout -------------------------------------------------------------------
# Pokemon fields (',' separated)
P_TYPE1, P_TYPE2, P_HP, P_MAXHP, P_ABILITY, P_STATUS, P_SLEEP = 2, 3, 6, 7, 8, 18, 20
# Side fields ('=' separated): 0-5 pokemon, then
S_ACTIVE, S_COND, S_BOOST0, S_FORCE = 6, 7, 11, 22
BOOST_NAMES = ("attack", "defense", "special_attack", "special_defense", "speed")
# SideConditions fields (';' separated)
C_AURORA, C_LIGHT, C_REFLECT, C_SPIKES, C_ROCKS, C_WEB, C_TAILWIND, C_TOXIC_COUNT, C_TSPIKES = (
    0, 3, 10, 12, 13, 14, 15, 16, 17,
)
# State fields ('/' separated)
ST_WEATHER, ST_TERRAIN = 2, 3

STATUSES = ("burn", "paralyze", "poison", "toxic", "sleep", "freeze")
WEATHERS = ("SUN", "RAIN", "SAND", "SNOW")
TERRAINS = ("GRASSYTERRAIN", "ELECTRICTERRAIN", "MISTYTERRAIN", "PSYCHICTERRAIN")

# Types / abilities that cannot hold a status (common cases only).
_IMMUNE_TYPES = {
    "burn": {"FIRE"},
    "paralyze": {"ELECTRIC"},
    "poison": {"POISON", "STEEL"},
    "toxic": {"POISON", "STEEL"},
    "freeze": {"ICE"},
    "sleep": set(),
}
_IMMUNE_ABILITIES = {
    "burn": {"WATERVEIL", "WATERBUBBLE", "COMATOSE", "PURIFYINGSALT"},
    "paralyze": {"LIMBER", "COMATOSE", "PURIFYINGSALT"},
    "poison": {"IMMUNITY", "PASTELVEIL", "COMATOSE", "PURIFYINGSALT"},
    "toxic": {"IMMUNITY", "PASTELVEIL", "COMATOSE", "PURIFYINGSALT"},
    "freeze": {"MAGMAARMOR", "COMATOSE", "PURIFYINGSALT"},
    "sleep": {"INSOMNIA", "VITALSPIRIT", "SWEETVEIL", "COMATOSE", "PURIFYINGSALT"},
}


# ---------------------------------------------------------------------------------------------
# State string model
# ---------------------------------------------------------------------------------------------


class ParsedState:
    """A mutable view of a State string; ``dumps()`` re-serialises it."""

    def __init__(self, text: str):
        self.parts = text.split("/")
        self.sides = [side.split("=") for side in self.parts[:2]]
        self.mons = [[m.split(",") for m in side[:6]] for side in self.sides]
        self.conds = [side[S_COND].split(";") for side in self.sides]

    def dumps(self) -> str:
        sides = []
        for side, mons, cond in zip(self.sides, self.mons, self.conds):
            fields = list(side)
            fields[:6] = [",".join(m) for m in mons]
            fields[S_COND] = ";".join(cond)
            sides.append("=".join(fields))
        return "/".join(sides + self.parts[2:])

    def active_index(self, side: int) -> int:
        return int(self.sides[side][S_ACTIVE])

    def hp(self, side: int, idx: int) -> int:
        return int(self.mons[side][idx][P_HP])

    def maxhp(self, side: int, idx: int) -> int:
        return int(self.mons[side][idx][P_MAXHP])

    def alive(self, side: int) -> list[int]:
        return [i for i in range(6) if self.maxhp(side, i) > 0 and self.hp(side, i) > 0]

    def force_switch(self, side: int) -> bool:
        return self.sides[side][S_FORCE] == "true"

    def types(self, side: int, idx: int) -> set[str]:
        mon = self.mons[side][idx]
        return {mon[P_TYPE1].upper(), mon[P_TYPE2].upper()}


def is_forkable(text: str) -> bool:
    """A mid-game decision state: both actives alive, nobody forced to switch, not over."""
    try:
        ps = ParsedState(text)
        for side in (0, 1):
            if ps.hp(side, ps.active_index(side)) <= 0 or ps.force_switch(side):
                return False
    except (IndexError, ValueError):
        return False
    return True


# ---------------------------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EditConfig:
    """Probability that each edit category fires on a fork (each is drawn independently)."""

    p_hp: float = 0.7  # rescale HP of 1-3 random living Pokemon (both sides)
    hp_low: float = 0.3  # scale ~ U(hp_low, hp_high), clamped to [1, maxhp]
    hp_high: float = 1.3
    p_faint_bench: float = 0.15  # faint one benched Pokemon
    p_status: float = 0.35  # add or remove a status on one Pokemon
    p_remove_status: float = 0.4  # of those, the chance to remove when it has one
    p_boost: float = 0.3  # +-1 / +-2 on 1-2 stat stages of one active
    p_hazard: float = 0.3  # stealth rock / spikes / toxic spikes / sticky web
    p_screen: float = 0.2  # reflect / light screen / aurora veil (snow only) / tailwind
    p_weather: float = 0.15
    p_terrain: float = 0.1

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            if f.name.startswith("p_"):
                value = getattr(self, f.name)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{f.name} must be in [0, 1], got {value}")
        if not 0.0 < self.hp_low <= self.hp_high:
            raise ValueError("need 0 < hp_low <= hp_high")


def _edit_hp(ps: ParsedState, rng: random.Random, cfg: EditConfig, meta: list) -> None:
    pool = [(s, i) for s in (0, 1) for i in ps.alive(s)]
    for side, idx in rng.sample(pool, min(len(pool), rng.randint(1, 3))):
        old, maxhp = ps.hp(side, idx), ps.maxhp(side, idx)
        new = min(max(int(round(old * rng.uniform(cfg.hp_low, cfg.hp_high))), 1), maxhp)
        if new != old:
            ps.mons[side][idx][P_HP] = str(new)
            meta.append({"kind": "hp", "side": side + 1, "slot": idx, "from": old, "to": new})


def _edit_faint_bench(ps: ParsedState, rng: random.Random, meta: list) -> None:
    options = []
    for side in (0, 1):
        active = ps.active_index(side)
        living = ps.alive(side)
        if len(living) < 2:
            continue  # keep at least the active one alive
        options += [(side, i) for i in living if i != active]
    if not options:
        return
    side, idx = rng.choice(options)
    mon = ps.mons[side][idx]
    meta.append({"kind": "faint", "side": side + 1, "slot": idx, "from": int(mon[P_HP]), "to": 0})
    mon[P_HP] = "0"
    mon[P_STATUS] = "NONE"


def _can_have(ps: ParsedState, side: int, idx: int, status: str) -> bool:
    mon = ps.mons[side][idx]
    if mon[P_STATUS].upper() != "NONE":
        return False
    if ps.types(side, idx) & _IMMUNE_TYPES[status]:
        return False
    if mon[P_ABILITY].upper() in _IMMUNE_ABILITIES[status]:
        return False
    if status == "sleep":  # sleep clause: one sleeper per side
        return not any(m[P_STATUS].upper() == "SLEEP" for m in ps.mons[side])
    return True


def _edit_status(ps: ParsedState, rng: random.Random, cfg: EditConfig, meta: list) -> None:
    pool = [(s, i) for s in (0, 1) for i in ps.alive(s)]
    side, idx = rng.choice(pool)
    mon = ps.mons[side][idx]
    old = mon[P_STATUS].upper()
    if old != "NONE" and rng.random() < cfg.p_remove_status:
        mon[P_STATUS] = "NONE"
        mon[P_SLEEP] = "0"
        meta.append({"kind": "status", "side": side + 1, "slot": idx, "from": old, "to": "NONE"})
        return
    choices = [st for st in STATUSES if _can_have(ps, side, idx, st)]
    if not choices:
        return
    new = rng.choice(choices).upper()
    mon[P_STATUS] = new
    mon[P_SLEEP] = "0"
    if idx == ps.active_index(side) and new == "TOXIC":
        ps.conds[side][C_TOXIC_COUNT] = "0"
    meta.append({"kind": "status", "side": side + 1, "slot": idx, "from": old, "to": new})


def _edit_boost(ps: ParsedState, rng: random.Random, meta: list) -> None:
    side = rng.randint(0, 1)
    for name in rng.sample(BOOST_NAMES, rng.randint(1, 2)):
        field = S_BOOST0 + BOOST_NAMES.index(name)
        old = int(ps.sides[side][field])
        new = min(6, max(-6, old + rng.choice((-2, -1, 1, 2))))
        if new != old:
            ps.sides[side][field] = str(new)
            meta.append({"kind": "boost", "side": side + 1, "stat": name, "from": old, "to": new})


def _set_cond(ps: ParsedState, side: int, field: int, name: str, new: int, meta: list) -> None:
    old = int(ps.conds[side][field])
    if new != old:
        ps.conds[side][field] = str(new)
        meta.append({"kind": "cond", "side": side + 1, "name": name, "from": old, "to": new})


def _edit_hazard(ps: ParsedState, rng: random.Random, meta: list) -> None:
    side = rng.randint(0, 1)
    field, name, top = rng.choice(
        [(C_ROCKS, "stealth_rock", 1), (C_SPIKES, "spikes", 3),
         (C_TSPIKES, "toxic_spikes", 2), (C_WEB, "sticky_web", 1)]
    )
    old = int(ps.conds[side][field])
    options = [v for v in range(top + 1) if v != old]
    _set_cond(ps, side, field, name, rng.choice(options), meta)


def _edit_screen(ps: ParsedState, rng: random.Random, meta: list) -> None:
    side = rng.randint(0, 1)
    snowing = ps.parts[ST_WEATHER].split(";")[0].upper() in ("SNOW", "HAIL")
    options = [(C_REFLECT, "reflect"), (C_LIGHT, "light_screen"), (C_TAILWIND, "tailwind")]
    if snowing:
        options.append((C_AURORA, "aurora_veil"))
    field, name = rng.choice(options)
    old = int(ps.conds[side][field])
    new = 0 if old else rng.randint(2, 5 if name != "tailwind" else 4)
    _set_cond(ps, side, field, name, new, meta)


def _edit_weather(ps: ParsedState, rng: random.Random, meta: list) -> None:
    old = ps.parts[ST_WEATHER]
    old_name = old.split(";")[0].upper()
    options = [w for w in ("NONE",) + WEATHERS if w != old_name]
    new = rng.choice(options)
    ps.parts[ST_WEATHER] = "NONE;0" if new == "NONE" else f"{new};{rng.randint(3, 5)}"
    meta.append({"kind": "weather", "from": old, "to": ps.parts[ST_WEATHER]})
    if new not in ("SNOW", "HAIL"):  # aurora veil needs snow: drop any left over
        for side in (0, 1):
            ps.conds[side][C_AURORA] = "0"


def _edit_terrain(ps: ParsedState, rng: random.Random, meta: list) -> None:
    old = ps.parts[ST_TERRAIN]
    old_name = old.split(";")[0].upper()
    options = [t for t in ("NONE",) + TERRAINS if t != old_name]
    new = rng.choice(options)
    ps.parts[ST_TERRAIN] = "NONE;0" if new == "NONE" else f"{new};{rng.randint(3, 5)}"
    meta.append({"kind": "terrain", "from": old, "to": ps.parts[ST_TERRAIN]})


def edit_state(
    text: str, rng: random.Random, cfg: EditConfig | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """A randomly perturbed copy of ``text`` and the list of edits applied (never empty).

    Raises ``ValueError`` if the input is not a forkable decision state or no edit could change it.
    """
    cfg = cfg or EditConfig()
    if not is_forkable(text):
        raise ValueError("state is not forkable (fainted active, forced switch, or malformed)")
    for _ in range(20):
        ps = ParsedState(text)
        meta: list[dict[str, Any]] = []
        if rng.random() < cfg.p_hp:
            _edit_hp(ps, rng, cfg, meta)
        if rng.random() < cfg.p_faint_bench:
            _edit_faint_bench(ps, rng, meta)
        if rng.random() < cfg.p_status:
            _edit_status(ps, rng, cfg, meta)
        if rng.random() < cfg.p_boost:
            _edit_boost(ps, rng, meta)
        if rng.random() < cfg.p_hazard:
            _edit_hazard(ps, rng, meta)
        if rng.random() < cfg.p_screen:
            _edit_screen(ps, rng, meta)
        if rng.random() < cfg.p_weather:
            _edit_weather(ps, rng, meta)
        if rng.random() < cfg.p_terrain:
            _edit_terrain(ps, rng, meta)
        if not meta:  # nothing fired (or all no-ops): force one HP edit
            _edit_hp(ps, rng, dataclasses.replace(cfg, hp_low=0.3, hp_high=0.9), meta)
        new = ps.dumps()
        if meta and new != text and is_forkable(new):
            return new, meta
    raise ValueError("could not produce a legal, changed state")


# ---------------------------------------------------------------------------------------------
# Engine-only play
# ---------------------------------------------------------------------------------------------


def side_alive(state: Any, side: int) -> int:
    pokemon = (state.side_one if side == 1 else state.side_two).pokemon
    return sum(1 for p in pokemon if p.hp > 0)


def outcome(state: Any) -> str | None:
    """``win`` / ``loss`` / ``tie`` from side one's view once a side is out of Pokemon, else None."""
    one, two = side_alive(state, 1), side_alive(state, 2)
    if one == 0 and two == 0:
        return recorder.RESULT_TIE
    if two == 0:
        return recorder.RESULT_WIN
    if one == 0:
        return recorder.RESULT_LOSS
    return None


def _engine_choice(choice: str) -> str:
    if choice == NO_MOVE:
        return "none"
    return choice[len("switch "):] if choice.startswith("switch ") else choice


def search_both(
    state: Any,
    time_ms: int,
    evaluator_path: str | None = None,
    leaf_scale: float = 1.0,
    iterations: int = 0,
) -> Any:
    """One single-threaded MCTS on the full state; returns the engine ``MctsResult`` (both sides)."""
    from poke_engine import monte_carlo_tree_search

    kwargs: dict[str, Any] = {}
    if evaluator_path is not None:
        from showdown_bot.players._worker import get_evaluator

        kwargs["evaluator"] = get_evaluator(evaluator_path, leaf_scale)
    try:
        return monte_carlo_tree_search(state, time_ms, iterations=iterations, threads=1, **kwargs)
    except Exception:
        raise
    except BaseException as exc:  # pyo3 PanicException
        raise RuntimeError(f"engine panic: {exc!r}") from None


def pick(options: list[tuple[str, int, float]], total: int, rng: random.Random, threshold: float) -> str:
    """The bot's mixed rule over one side's visit shares (``select_choice`` on the policy)."""
    policy = aggregate_policy([(total, options)])
    choice = select_choice(policy, MIXED, rng, threshold)
    return choice if choice is not None else "none"


def sample_outcome(instructions: list[Any], rng: random.Random) -> Any:
    """One branch of ``generate_instructions`` drawn by its percentages."""
    weights = [max(float(i.percentage), 0.0) for i in instructions]
    if sum(weights) <= 0:
        return instructions[0]
    return rng.choices(instructions, weights=weights)[0]


@dataclasses.dataclass
class PlayConfig:
    time_ms: int = 50
    iterations: int = 0  # > 0: fixed iteration count per search instead of time
    evaluator: str | None = None
    leaf_scale: float = 1.0
    max_turns: int = 150  # searched/played steps (a forced-switch step counts as one)
    threshold: float = MIXED_THRESHOLD
    on_cap: str = "drop"  # "drop" the game or record it as a "tie"


@dataclasses.dataclass
class ForkGame:
    result: str | None  # win/loss/tie from side one's view; None when dropped (cap reached)
    steps: int
    decisions: list[dict[str, Any]]
    forced_switch_steps: int = 0
    capped: bool = False


def play_out(state_str: str, rng: random.Random, cfg: PlayConfig | None = None) -> ForkGame:
    """Play ``state_str`` to the end: one perfect-information search per step, both sides move.

    A switch-only turn (a side's active fainted) shows up in the search as one side offering
    only ``switch <mon>`` options while the other offers the single label ``No Move`` (sent to
    the engine as ``none``); the ``force_switch`` flag is not set on the state after
    ``apply_instructions``. The mixed rule picks among the switches like any other turn. Side-one decisions with a real choice (>= 2 options) are
    recorded in the self-play ``decision`` schema (truth + one world with the search scores).
    """
    from poke_engine import State, generate_instructions

    cfg = cfg or PlayConfig()
    state = State.from_string(state_str)
    decisions: list[dict[str, Any]] = []
    forced = 0
    for step in range(cfg.max_turns):
        done = outcome(state)
        if done is not None:
            return ForkGame(done, step, decisions, forced)
        text = state.to_string()
        started = time.perf_counter()
        res = search_both(state, cfg.time_ms, cfg.evaluator, cfg.leaf_scale, cfg.iterations)
        wall_ms = (time.perf_counter() - started) * 1000.0
        opts1 = [(o.move_choice, int(o.visits), float(o.total_score)) for o in res.side_one]
        opts2 = [(o.move_choice, int(o.visits), float(o.total_score)) for o in res.side_two]
        total = int(res.total_visits)
        c1 = pick(opts1, total, rng, cfg.threshold) if opts1 else "none"
        c2 = pick(opts2, total, rng, cfg.threshold) if opts2 else "none"
        if [o[0] for o in opts1] == [NO_MOVE] or [o[0] for o in opts2] == [NO_MOVE]:
            forced += 1
        if len(opts1) >= 2 and total > 0:
            decisions.append({
                "turn": step,
                "side": "one",
                "worlds": [recorder.world_record(text, (total, opts1))],
                "policy": aggregate_policy([(total, opts1)]),
                "chosen": c1,
                "order": None,
                "compute": {
                    "search_time_ms": cfg.time_ms, "worlds_requested": 1, "worlds_searched": 1,
                    "parallelism": 1, "threads": 1, "iterations": total,
                    "wall_ms": round(wall_ms, 1),
                },
                "truth": {"state": text},
                "truth_status": "ok",
                "opp_choice": c2,
            })
        branches = generate_instructions(state, _engine_choice(c1), _engine_choice(c2))
        state = state.apply_instructions(sample_outcome(branches, rng))
    done = outcome(state)
    if done is not None:
        return ForkGame(done, cfg.max_turns, decisions, forced)
    result = recorder.RESULT_TIE if cfg.on_cap == "tie" else None
    return ForkGame(result, cfg.max_turns, decisions, forced, capped=True)


# ---------------------------------------------------------------------------------------------
# Sources and jobs
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ForkJob:
    index: int
    seed: int
    state: str
    origin: dict[str, Any]
    run_dir: str
    edit: EditConfig
    play: PlayConfig
    run_id: str = "fork"


def sample_sources(
    root: str | os.PathLike[str] | list[str | os.PathLike[str]],
    n_forks: int,
    per_game: int,
    rng: random.Random,
    *,
    min_turn: int = 1,
) -> list[dict[str, Any]]:
    """Up to ``n_forks`` source decisions: at most ``per_game`` per battle, games chosen uniformly.

    Both players' files of one battle count as the same game (deduplicated by ``battle_tag``).
    Each source is ``{"state", "file", "battle_tag", "turn", "decision_idx"}``.
    """
    roots = [root] if isinstance(root, (str, os.PathLike)) else list(root)
    files = sorted({p for r in roots for p in recorder.iter_game_files(r)})
    rng.shuffle(files)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in files:
        if len(out) >= n_forks:
            break
        try:
            game = recorder.read_game(path)
        except (OSError, ValueError, EOFError):
            continue
        tag = str(game["header"].get("battle_tag", path))
        if tag in seen:
            continue
        seen.add(tag)
        cands = [
            (i, d) for i, d in enumerate(game["decisions"])
            if d.get("truth") and (d.get("turn") or 0) >= min_turn
            and is_forkable(d["truth"]["state"])
        ]
        for i, d in rng.sample(cands, min(per_game, len(cands), n_forks - len(out))):
            out.append({
                "state": d["truth"]["state"], "file": os.path.basename(str(path)),
                "battle_tag": tag, "turn": d.get("turn"), "decision_idx": i,
            })
    return out


def run_fork(job: ForkJob) -> dict[str, Any]:
    """Edit, play and write one fork (runs in a worker process); returns a small summary."""
    rng = random.Random(job.seed)
    started = time.perf_counter()
    summary: dict[str, Any] = {"index": job.index, "ok": False}
    try:
        edited, edits = edit_state(job.state, rng, job.edit)
        game = play_out(edited, rng, job.play)
    except Exception as exc:  # engine panic / illegal edit: skip this fork, keep the pool alive
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["seconds"] = time.perf_counter() - started
        return summary
    summary.update(
        steps=game.steps, result=game.result, capped=game.capped,
        forced_switch_steps=game.forced_switch_steps, decisions=len(game.decisions),
        seconds=time.perf_counter() - started, n_edits=len(edits),
    )
    if game.result is None or not game.decisions:
        summary["dropped"] = True
        return summary
    rec = recorder.GameRecorder(job.run_dir)
    tag = f"fork-{job.run_id}-{job.index:06d}"
    rec.start_game(
        tag, "fork",
        role="p1", battle_format="gen9randombattle", player="fork-search",
        config={
            "search_time_ms": job.play.time_ms, "iterations": job.play.iterations,
            "evaluator": job.play.evaluator, "evaluator_scale": job.play.leaf_scale,
            "mode": MIXED, "mixed_threshold": job.play.threshold, "max_turns": job.play.max_turns,
        },
        extra={
            "source": SOURCE,
            "fork": {
                "seed": job.seed, "origin": job.origin, "edited_state": edited,
                "edits": edits, "capped": game.capped,
                "edit_config": dataclasses.asdict(job.edit),
            },
        },
    )
    for decision in game.decisions:
        rec.add_decision(tag, "fork", decision)
    path = rec.end_game(tag, "fork", game.result, turns=game.steps)
    summary["ok"] = path is not None
    return summary


def make_jobs(
    sources: list[dict[str, Any]],
    seed: int,
    run_dir: str | os.PathLike[str],
    edit: EditConfig,
    play: PlayConfig,
    run_id: str = "fork",
) -> list[ForkJob]:
    """One job per source; the per-fork seed depends only on ``(seed, index)`` (worker-count free)."""
    return [
        ForkJob(
            index=i, seed=(seed * 1_000_003 + i) & 0x7FFFFFFF, state=src["state"],
            origin={k: v for k, v in src.items() if k != "state"},
            run_dir=str(run_dir), edit=edit, play=play, run_id=run_id,
        )
        for i, src in enumerate(sources)
    ]
