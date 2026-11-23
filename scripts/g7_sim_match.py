"""Simulator match: a token policy net (A) vs the "gen7" MCTS search bot at a reduced budget (B).

Everything runs inside ``poke_engine.SelfPlayPool`` (no Showdown server). B is the ``best-gen7``
preset of ``showdown_bot.league`` (evaluator gen7-res-root-mlp16.pev at leaf scale 1.5, PUCT prior
prior-gen56-h32.ppol for both sides, c_puct 2, argmax of the visit share averaged over sampled
worlds) with ``--worlds`` worlds and ``--iters`` MCTS iterations per world (or ``--ms`` per world).

Per gen7 decision: ``pool.fork_rollouts`` samples ``worlds`` states the seat cannot tell from the
true one (the opponent's unseen team is resampled by the Rust sampler); a seat-2 decision has its
two sides exchanged in the state string so the searcher is always side one (as it is live); each
world is searched single-threaded in a worker process (the MCTS binding holds the GIL); the choice
names are mapped to the pool's 13 action slots with ``poke_engine.policy_actions`` (the engine's
own ``action_slot``: 0-3 move, 4-7 tera move, 8-12 switch to the n-th non-active team member).

    PYTHONPATH=".cache/engine/sdb.34;src" .venv/Scripts/python.exe scripts/g7_sim_match.py \
        --net runs/ppo7/ckpt-46000.pt --games 200 --worlds 4 --iters 2000 --procs 8

``--net gen7`` plays gen7 (budget ``--a-worlds`` / ``--a-iters`` / ``--a-ms``, default the same as
B's) against gen7: the ~50% sanity check, or a budget-vs-budget match.

Each of the ``--games`` pool slots plays exactly one counted game (no short-game bias); gen7 sits
in seat 1 in the even slots and seat 2 in the odd ones. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "models" / "gen7-res-root-mlp16.pev"
PRIOR = ROOT / "models" / "prior-gen56-h32.ppol"
LEAF_SCALE = 1.5
C_PUCT = 2.0
N_ACT = 13

# ---------------------------------------------------------------------------
# Worker process: one single-threaded MCTS per sampled world
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(evaluator: str, prior: str, scale: float) -> None:
    from poke_engine import Evaluator, PolicyPrior

    _W["ev"] = Evaluator.load(evaluator, leaf_scale=float(scale))
    _W["prior"] = PolicyPrior.load(prior)


def _search_chunk(chunk: list[tuple[int, str]], ms: int, iters: int, c_puct: float) -> tuple[list, float, float]:
    """``chunk``: (decision id, world state with the searcher as side one). Returns
    ``([(decision id, [(choice, slot, visits), ...]), ...], cpu seconds, wall seconds)``."""
    import poke_engine
    from poke_engine import State, monte_carlo_tree_search

    c0, w0 = time.process_time(), time.perf_counter()
    out = []
    for dec, text in chunk:
        try:
            state = State.from_string(text)
            slots = dict(poke_engine.policy_actions(state, 1))  # choice string -> action slot (-1: none)
            res = monte_carlo_tree_search(
                state, int(ms), iterations=int(iters), threads=1, evaluator=_W["ev"],
                prior=_W["prior"], prior_opp=_W["prior"], c_puct=float(c_puct),
            )
            out.append((dec, [(str(o.move_choice), int(slots.get(str(o.move_choice), -2)), int(o.visits))
                              for o in res.side_one]))
        except BaseException as exc:  # a pyo3 panic is a BaseException; the world is dropped
            out.append((dec, repr(exc)))
    return out, time.process_time() - c0, time.perf_counter() - w0


def swap_sides(state: str) -> str:
    """The same position with the two sides exchanged (as showdown_bot.human_policy.swap_sides)."""
    parts = state.split("/")
    parts[0], parts[1] = parts[1], parts[0]
    return "/".join(parts)


# ---------------------------------------------------------------------------
# The gen7 searcher over pool rows
# ---------------------------------------------------------------------------


class Gen7:
    def __init__(self, executor: ProcessPoolExecutor, procs: int, worlds: int, iters: int, ms: int, seed: int) -> None:
        self.ex, self.procs = executor, procs
        self.worlds, self.iters, self.ms = worlds, iters, ms
        self.seed = seed
        self.calls = 0
        self.decisions = 0
        self.cpu_s = 0.0
        self.wall_s = 0.0
        self.unmapped = 0  # the search's top choice had no legal slot (fell back to the best mapped one)
        self.no_choice = 0  # nothing usable at all (fell back to the first legal slot)
        self.legal_mismatch = 0  # the search's slot set differs from the pool row's legal slots
        self.slot_conflict = 0  # one choice name mapped to different slots in different worlds
        self.world_errors = 0

    def act(self, pool, games: list[int], sides: list[int], legal) -> list[int]:
        """One action slot per decision (``games[i]``, seat ``sides[i]``, ``legal[i]`` uint8[13])."""
        n = len(games)
        self.calls += 1
        fork = pool.fork_rollouts(games, sides, self.worlds, 1, (self.seed * 1_000_003 + self.calls) & (2**63 - 1))
        tasks: list[tuple[int, str]] = []
        for i in range(n):
            for w in range(self.worlds):
                s = fork.game_state(i * self.worlds + w)
                tasks.append((i, swap_sides(s) if sides[i] == 1 else s))
        size = max(1, min(64, math.ceil(len(tasks) / (self.procs * 4))))
        futures = [self.ex.submit(_search_chunk, tasks[k:k + size], self.ms, self.iters, C_PUCT)
                   for k in range(0, len(tasks), size)]
        per: list[list] = [[] for _ in range(n)]
        for f in futures:
            rows, cpu, wall = f.result()
            self.cpu_s += cpu
            self.wall_s += wall
            for dec, options in rows:
                if isinstance(options, str):
                    self.world_errors += 1
                    print(f"world failed: {options}", file=sys.stderr)
                else:
                    per[dec].append(options)
        self.decisions += n
        return [self._choose(per[i], legal[i]) for i in range(n)]

    def _choose(self, results: list, legal) -> int:
        # showdown_bot.players.search.aggregate_policy: mean over worlds of each choice's visit share
        policy: dict[str, float] = {}
        slot_of: dict[str, int] = {}
        usable = [r for r in results if sum(v for _, _, v in r) > 0]
        for r in usable:
            total = sum(v for _, _, v in r)
            for name, slot, visits in r:
                policy[name] = policy.get(name, 0.0) + visits / total / len(usable)
                if slot_of.setdefault(name, slot) != slot:
                    self.slot_conflict += 1
        legal_set = {s for s in range(N_ACT) if legal[s]}
        if usable and {s for s in slot_of.values() if s >= 0} != legal_set:
            self.legal_mismatch += 1
        ranked = sorted(policy.items(), key=lambda kv: kv[1], reverse=True)  # stable: ties keep engine order
        for rank, (name, _mass) in enumerate(ranked):
            if slot_of[name] in legal_set:
                if rank:
                    self.unmapped += 1
                return slot_of[name]
        self.no_choice += 1
        return min(legal_set)


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> dict:
    import numpy as np
    import poke_engine

    t_start = time.perf_counter()
    net_is_gen7 = args.net == "gen7"
    ex = ProcessPoolExecutor(args.procs, initializer=_init_worker, initargs=(str(EVALUATOR), str(PRIOR), LEAF_SCALE))
    b = Gen7(ex, args.procs, args.worlds, args.iters, args.ms, args.seed)
    a = None
    if net_is_gen7:
        a = Gen7(ex, args.procs, args.a_worlds or args.worlds, args.iters if args.a_iters is None else args.a_iters,
                 args.ms if args.a_ms is None else args.a_ms, args.seed + 7919)
    else:
        import torch

        from showdown_bot import ppo

        torch.set_num_threads(2)
        dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
        if dev == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb / total))  # as ppo.main
        torch.manual_seed(args.seed)
        batcher = ppo.TokenBatcher(dev)
        policy = ppo.Policy(ppo.load_net(args.net, dev), dev)

    n = args.games
    pool = poke_engine.SelfPlayPool(n, args.seed, args.pool_threads, args.turn_cap)
    g7_seat = np.arange(n) % 2  # gen7 is side one (seat 0) in even slots, side two in odd slots
    active = np.ones(n, bool)  # the slot's first game is still running (later games are not counted)
    score = np.full(n, np.nan)
    turns = np.zeros(n, np.int64)
    b_dec = np.zeros(n, np.int64)
    steps = 0
    while active.any():
        raw = pool.observe()
        rows = int(raw["n"])
        game = np.frombuffer(raw["game"], dtype="<i4")
        side = np.frombuffer(raw["side"], dtype=np.uint8)
        legal = np.frombuffer(raw["legal"], dtype=np.uint8).reshape(rows, N_ACT)
        actions = legal.argmax(1).astype(np.uint8)  # finished slots: first legal slot, no search
        live = active[game]
        is_b = live & (side == g7_seat[game])
        is_a = live & ~is_b
        for bot, mask in ((b, is_b), (a, is_a)):
            idx = np.nonzero(mask)[0]
            if bot is None or not len(idx):
                continue
            actions[idx] = bot.act(pool, [int(g) for g in game[idx]], [int(s) for s in side[idx]], legal[idx])
            if bot is b:
                np.add.at(b_dec, game[idx], 1)
        idx = np.nonzero(is_a)[0]
        if a is None and len(idx):
            import torch

            cat = np.frombuffer(raw["cat"], dtype="<i2").reshape(rows, batcher.cat_len)
            cont = np.frombuffer(raw["cont"], dtype="<f4").reshape(rows, batcher.cont_len)
            hc = np.frombuffer(raw["hist_cat"], dtype="<i2").reshape(rows, 16, 4)
            hf = np.frombuffer(raw["hist_cont"], dtype="<f4").reshape(rows, 16, 5)
            batch = batcher.to_device(cat[idx], cont[idx], hc[idx], hf[idx])
            actions[idx] = policy.forward(batch, greedy=not args.sample).actions
        bad = ~legal[np.arange(rows), actions].astype(bool)
        if bad.any():
            raise RuntimeError(f"illegal action chosen for rows {np.nonzero(bad)[0].tolist()}")
        _, _, fin = pool.step(actions.tobytes())
        steps += 1
        for g, w, t in fin:
            if active[g]:
                active[g] = False
                score[g] = 0.5 if w == 0 else float(w == g7_seat[g] + 1)
                turns[g] = t
        if args.progress and steps % 20 == 0:
            done = int((~active).sum())
            print(f"step {steps}: {done}/{n} games, gen7 {np.nanmean(score) if done else float('nan'):.3f}, "
                  f"{time.perf_counter() - t_start:.0f}s", file=sys.stderr, flush=True)
    ex.shutdown()
    wall = time.perf_counter() - t_start
    s = score
    se = float(s.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    p = float(np.clip(s.mean(), 1e-3, 1 - 1e-3))
    seat_score = [float(s[g7_seat == k].mean()) if (g7_seat == k).any() else float("nan") for k in (0, 1)]
    stats = pool.stats()
    out = {
        "net": args.net, "greedy": not args.sample, "worlds": args.worlds, "iters": args.iters, "ms": args.ms,
        "games": n, "gen7_score": float(s.mean()), "se": se,
        "elo": 400 * math.log10(p / (1 - p)), "elo_se": 400 / math.log(10) * se / (p * (1 - p)),
        "gen7_score_seat1": seat_score[0], "gen7_score_seat2": seat_score[1],
        "draws": int((s == 0.5).sum()), "mean_turns": float(turns.mean()),
        "cpu_s_per_decision": b.cpu_s / max(1, b.decisions),
        "search_wall_s_per_decision": b.wall_s / max(1, b.decisions),
        "decisions_per_game": float(b_dec.mean()), "gen7_decisions": b.decisions,
        "unmapped": b.unmapped + b.no_choice, "no_choice": b.no_choice, "legal_mismatch": b.legal_mismatch,
        "slot_conflict": b.slot_conflict, "world_errors": b.world_errors,
        "illegal": int(stats.get("illegal", -1)), "panics": int(stats.get("panics", -1)),
        "procs": args.procs, "seed": args.seed, "wall_s": wall,
        "wall_s_per_game": wall / n,
    }
    if a is not None:
        out["a"] = {"worlds": a.worlds, "iters": a.iters, "ms": a.ms, "decisions": a.decisions,
                    "cpu_s_per_decision": a.cpu_s / max(1, a.decisions), "unmapped": a.unmapped + a.no_choice,
                    "legal_mismatch": a.legal_mismatch, "world_errors": a.world_errors}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="token policy net (.pt), or 'gen7' for a gen7-v-gen7 match")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--worlds", type=int, default=4, help="sampled worlds per gen7 decision")
    ap.add_argument("--iters", type=int, default=10000, help="MCTS iterations per world (reproducible budget)")
    ap.add_argument("--ms", type=int, default=0, help="ms per world instead of --iters (sets iters to 0)")
    ap.add_argument("--procs", type=int, default=8, help="MCTS worker processes (at most 8)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", action="store_true", help="the net samples its policy (default: greedy)")
    ap.add_argument("--a-worlds", type=int, default=0, help="--net gen7: A's worlds (default --worlds)")
    ap.add_argument("--a-iters", type=int, default=None, help="--net gen7: A's iterations (default --iters)")
    ap.add_argument("--a-ms", type=int, default=None, help="--net gen7: A's ms per world (default --ms)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gpu-mem-gb", type=float, default=1.5, help="hard cap on this process's CUDA memory")
    ap.add_argument("--pool-threads", type=int, default=2)
    ap.add_argument("--turn-cap", type=int, default=200)
    ap.add_argument("--progress", action="store_true", help="progress lines on stderr")
    args = ap.parse_args()
    if not 1 <= args.procs <= 8:
        ap.error("--procs must be 1..8")
    if args.ms > 0:
        args.iters = 0
    if args.a_ms:
        args.a_iters = 0
    elif args.a_iters:
        args.a_ms = 0
    if args.iters <= 0 and args.ms <= 0:
        ap.error("need --iters > 0 or --ms > 0")
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
