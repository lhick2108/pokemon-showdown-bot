"""Throughput of MCTS leaf evaluation: stock evaluate() vs linear vs MLP.

Uses the same state as scripts/spike_engine.py, single thread, and dummy
(seeded random) weights; throughput does not depend on the weight values
beyond the shape of the search tree they induce.

    python -m uv run python scripts/engine_bench.py [--durations 1 2] [--hidden 64 256]
"""

from __future__ import annotations

import argparse
import random
import time

from poke_engine import (
    FEATURE_LENGTH,
    Evaluator,
    monte_carlo_tree_search,
)
from spike_engine import make_state  # scripts/ is sys.path[0] when run as a script


def random_linear(rng: random.Random) -> Evaluator:
    weights = [rng.gauss(0.0, 0.1) for _ in range(FEATURE_LENGTH)]
    return Evaluator.linear(weights, 0.0)


def random_mlp(rng: random.Random, hidden: int) -> Evaluator:
    scale1 = (2.0 / FEATURE_LENGTH) ** 0.5
    w1 = [rng.gauss(0.0, scale1) for _ in range(hidden * FEATURE_LENGTH)]
    b1 = [0.0] * hidden
    w2 = [rng.gauss(0.0, hidden**-0.5) for _ in range(hidden)]
    return Evaluator.mlp(w1, b1, w2, 0.0)


def run(state, seconds: int, evaluator) -> tuple[int, float]:
    started = time.perf_counter()
    result = monte_carlo_tree_search(
        state, duration_ms=seconds * 1000, iterations=0, threads=1, evaluator=evaluator
    )
    return result.total_visits, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durations", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 256])
    args = parser.parse_args()

    rng = random.Random(0)
    state = make_state()
    models: list[tuple[str, Evaluator | None]] = [("stock evaluate()", None)]
    models.append(("linear", random_linear(rng)))
    for h in args.hidden:
        models.append((f"mlp hidden={h}", random_mlp(rng, h)))

    # warm-up (page in code and allocator) so the first row is not penalised
    run(state, 1, None)

    rows = []
    stock_rate: dict[int, float] = {}
    for seconds in args.durations:
        for name, evaluator in models:
            iterations, elapsed = run(state, seconds, evaluator)
            rate = iterations / elapsed
            if evaluator is None:
                stock_rate[seconds] = rate
            rows.append((name, seconds, iterations, elapsed, rate, rate / stock_rate[seconds]))

    print(f"features per state: {FEATURE_LENGTH}; single thread; state from scripts/spike_engine.py")
    header = f"{'evaluator':<18} {'req':>4} {'iterations':>11} {'elapsed s':>10} {'iter/s':>12} {'vs stock':>9} {'slowdown':>9}"
    print(header)
    print("-" * len(header))
    for name, seconds, iterations, elapsed, rate, rel in rows:
        print(
            f"{name:<18} {seconds:>3}s {iterations:>11,} {elapsed:>10.3f} {rate:>12,.0f} "
            f"{rel:>8.2f}x {1.0 / rel:>8.2f}x"
        )


if __name__ == "__main__":
    main()
