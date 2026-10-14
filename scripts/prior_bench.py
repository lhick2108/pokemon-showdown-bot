"""Benchmark PUCT policy priors against plain UCB1: iterations/s, tree depth, root agreement.

    python scripts/prior_bench.py --data data/selfplay/gen6 --prior models/prior-gen6-h32.ppol \\
        --evaluator models/gen2-res-searchabs-mlp16.pev --scale 1.5 --states 12 --ms 1000 \\
        --c-puct 0.5 1 2

States are searched worlds taken from recorded self-play (spread over turns). For every state
and setting it runs one single-threaded search of ``--ms`` and reports iterations/s, the mean
and max depth of the expanded node (plies from the root) and whether the most-visited root
move equals the one of a long no-prior reference search (``--ref-ms``).
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path


def pick_states(root: Path, n: int, seed: int) -> list[tuple[int, str]]:
    from showdown_bot import recorder

    files = sorted(root.glob("*.jsonl.gz"))
    rng = random.Random(seed)
    rng.shuffle(files)
    out: list[tuple[int, str]] = []
    want_turns = [3, 6, 9, 12, 15, 20]
    for path in files[:400]:
        game = recorder.read_game(path)
        for d in game["decisions"]:
            if d.get("worlds") and len(d["worlds"][0].get("moves", [])) >= 5:
                t = d.get("turn") or 0
                bucket = min(range(len(want_turns)), key=lambda i: abs(want_turns[i] - t))
                if sum(1 for b, _ in out if b == bucket) < max(1, n // len(want_turns)):
                    out.append((bucket, d["worlds"][0]["state"]))
                    break
        if len(out) >= n:
            break
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prior", required=True)
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--states", type=int, default=12)
    ap.add_argument("--ms", type=int, default=1000)
    ap.add_argument("--ref-ms", type=int, default=6000)
    ap.add_argument("--c-puct", type=float, nargs="+", default=[1.0])
    ap.add_argument("--fpu", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from poke_engine import Evaluator, PolicyPrior, State, monte_carlo_tree_search

    ev = Evaluator.load(args.evaluator, leaf_scale=args.scale)
    prior = PolicyPrior.load(args.prior)
    states = pick_states(Path(args.data), args.states, args.seed)
    print(f"{len(states)} states, {args.ms} ms each, prior {Path(args.prior).name} (hidden {prior.hidden_size})")

    def top(res):
        return max(res.side_one, key=lambda o: o.visits).move_choice

    refs = []
    for _b, s in states:
        r = monte_carlo_tree_search(State.from_string(s), args.ref_ms, evaluator=ev)
        refs.append(top(r))

    settings = [("ucb1", {})] + [
        (f"puct c={c:g}", dict(prior=prior, prior_opp=prior, c_puct=c, fpu_reduction=args.fpu, prior_eps=args.eps))
        for c in args.c_puct
    ]
    rows: dict[str, list[tuple[float, float, int, int]]] = {name: [] for name, _ in settings}
    t0 = time.time()
    for rep in range(args.repeat):
        for i, (_b, s) in enumerate(states):
            for name, kw in settings:  # interleaved so machine load hits every setting alike
                res = monte_carlo_tree_search(State.from_string(s), args.ms, evaluator=ev, **kw)
                rows[name].append((res.total_visits / (args.ms / 1000), res.depth_mean, res.depth_max,
                                   int(top(res) == refs[i])))
    base = statistics.mean(r[0] for r in rows["ucb1"])
    print(f"{'setting':14s} {'iter/s':>9s} {'vs ucb1':>8s} {'depth mean':>11s} {'depth max':>10s} {'agree w/ ref':>13s}")
    for name, _ in settings:
        rs = rows[name]
        ips = statistics.mean(r[0] for r in rs)
        print(f"{name:14s} {ips:9.0f} {ips / base:8.2f} {statistics.mean(r[1] for r in rs):11.2f} "
              f"{statistics.mean(r[2] for r in rs):10.1f} {statistics.mean(r[3] for r in rs):13.2f}")
    print(f"({time.time() - t0:.0f}s of search)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
