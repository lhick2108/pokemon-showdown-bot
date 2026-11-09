"""Score value nets (.pev) on human first-person positions: BCE and accuracy vs the game outcome.

    python scripts/human_eval.py MODEL.pev [MODEL2.pev ...] --data data\\human\\pc1-fp-aux\\test \\
        [--max-games N] [--per-game K] [--seed S] [--json OUT.json]

Each model is loaded with ``poke_engine.Evaluator.load`` and queried exactly as the engine does
at a search root: ``Evaluator.logit(state)`` is the absolute logit of P(side one wins), including
the residual stock term (``stock_coef * evaluate``) of residual nets; ``leaf_scale`` is a
search-time gap multiplier and does not enter an absolute probability. Labels are the game
outcome (win 1, loss 0, draw 0.5) from side one's view, as in train.py's val BCE (unweighted).

The subsample is fixed by ``--seed``: per game file, ``--per-game`` decisions are drawn without
replacement (RNG seeded by seed and the file name) from those with a truth state, so every model
sees the same positions. Streaming: one game file in memory at a time; per-model state is a few
sums per turn bucket. Output: BCE / accuracy (decisive outcomes only) overall and per turn
bucket next to the constant predictor (the sample's mean outcome).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10**9)]
NAMES = ["1-5", "6-10", "11-15", "16-20", "21-30", "31+"]


def _bucket(turn: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= turn <= hi:
            return i
    return 0


def _bce(logit: float, y: float) -> float:
    # stable BCE-with-logits: max(z,0) - z*y + log(1+exp(-|z|))
    return max(logit, 0.0) - logit * y + math.log1p(math.exp(-abs(logit)))


def _battle_tag(path: Path) -> str:
    # recorder file names are <battle_tag>__<username>.jsonl.gz
    return path.name.split("__")[0]


def val_split_tags(dirs: list[str | Path], val_fraction: float, seed: int = 0) -> set[str]:
    """The val battles of ``train.split_by_battle`` over all games in ``dirs`` (same shuffle;
    assumes every game contributed samples to the training run, true for self-play records)."""
    import random as _r

    from showdown_bot import recorder

    unique = sorted({_battle_tag(p) for d in dirs for p in recorder.iter_game_files(d)})
    _r.Random(seed).shuffle(unique)
    n_val = int(round(len(unique) * val_fraction))
    n_val = min(max(n_val, 1 if len(unique) > 1 else 0), max(len(unique) - 1, 0))
    return set(unique[:n_val])


def sample_positions(data: str | Path | list, max_games: int = 0, per_game: int = 4, seed: int = 0,
                     only_tags: set[str] | None = None):
    """Yield ``(state_string, outcome, turn)`` from data directories, streaming.

    Per game, ``per_game`` decisions (0 = all) with a truth state, uniformly without
    replacement, by an RNG seeded with ``seed`` and the file name. ``only_tags``: keep only
    games of these battle tags (see ``val_split_tags``).
    """
    from showdown_bot import dataset, recorder

    dirs = data if isinstance(data, list) else [data]
    paths = (p for d in dirs for p in recorder.iter_game_files(d))
    games = 0
    for path in paths:
        if only_tags is not None and _battle_tag(path) not in only_tags:
            continue
        if max_games and games >= max_games:
            break
        try:
            game = recorder.read_game(path)
        except (OSError, ValueError, EOFError) as exc:
            print(f"skipping {path}: {exc}", file=sys.stderr)
            continue
        result = (game["result"] or {}).get("result")
        if result not in dataset.VALUE_BY_RESULT:
            continue
        games += 1
        y = float(dataset.value_target(result, 1))
        elig = [i for i, d in enumerate(game["decisions"]) if (d.get("truth") or {}).get("state")]
        if per_game > 0 and len(elig) > per_game:
            elig = sorted(random.Random(f"{seed}:{path.name}").sample(elig, per_game))
        for i in elig:
            d = game["decisions"][i]
            yield d["truth"]["state"], y, int(d.get("turn") or i + 1)


def evaluate_models(models: list[str], data: str | Path | list, max_games: int = 0, per_game: int = 4,
                    seed: int = 0, only_tags: set[str] | None = None) -> dict:
    import poke_engine

    evs = [poke_engine.Evaluator.load(str(m)) for m in models]
    nb = len(BUCKETS)
    # per model, per bucket: n, bce sum, n decisive, correct
    acc = [[[0, 0.0, 0, 0] for _ in range(nb)] for _ in evs]
    # per model: sums of squared / absolute logit difference to the first model
    agree = [[0.0, 0.0] for _ in evs]
    const = [[0, 0.0] for _ in range(nb)]  # n, sum of y
    ys: list[list[float]] = [[] for _ in range(nb)]  # only y values (floats) for the baseline
    t0 = time.perf_counter()
    for state_s, y, turn in sample_positions(data, max_games, per_game, seed, only_tags):
        state = poke_engine.State.from_string(state_s)
        b = _bucket(turn)
        const[b][0] += 1
        const[b][1] += y
        ys[b].append(y)
        z0 = None
        for ev, rows, ag in zip(evs, acc, agree):
            z = float(ev.logit(state))
            if z0 is None:
                z0 = z
            ag[0] += (z - z0) ** 2
            ag[1] += abs(z - z0)
            r = rows[b]
            r[0] += 1
            r[1] += _bce(z, y)
            if y != 0.5:
                r[2] += 1
                r[3] += (z > 0) == (y > 0.5)
    elapsed = time.perf_counter() - t0

    def cbce(vals: list[float], p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return sum(-(v * math.log(p) + (1 - v) * math.log(1 - p)) for v in vals) / max(len(vals), 1)

    all_y = [v for b in ys for v in b]
    p_all = sum(all_y) / max(len(all_y), 1)
    out: dict = {"n": len(all_y), "seconds": elapsed, "mean_outcome": p_all, "models": {}}
    rows_c = [("all", all_y)] + [(NAMES[i], ys[i]) for i in range(nb) if ys[i]]
    out["constant"] = {name: {"n": len(v), "bce": cbce(v, p_all),
                              "acc": max(sum(1 for t in v if t == 1.0), sum(1 for t in v if t == 0.0))
                              / max(sum(1 for t in v if t != 0.5), 1)} for name, v in rows_c}
    for m, rows, ag in zip(models, acc, agree):
        tot = [sum(r[k] for r in rows) for k in range(4)]
        res = {"all": {"n": tot[0], "bce": tot[1] / max(tot[0], 1), "acc": tot[3] / max(tot[2], 1),
                       "rms_vs_first": math.sqrt(ag[0] / max(tot[0], 1)),
                       "mad_vs_first": ag[1] / max(tot[0], 1)}}
        for i in range(nb):
            if rows[i][0]:
                res[NAMES[i]] = {"n": rows[i][0], "bce": rows[i][1] / rows[i][0],
                                 "acc": rows[i][3] / max(rows[i][2], 1)}
        out["models"][str(m)] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("models", nargs="+", help=".pev files")
    ap.add_argument("--data", required=True, action="append",
                    help="run directory of games (first-person human or self-play); repeatable")
    ap.add_argument("--max-games", type=int, default=0, help="stop after N games (0 = all)")
    ap.add_argument("--per-game", type=int, default=4, help="decisions per game (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-split", type=float, default=None, metavar="F",
                    help="only the val battles of train.py's split (--val-fraction F, --split-seed) "
                         "over all --data dirs, e.g. 0.1 for the held-out battles of a run trained "
                         "on exactly these dirs")
    ap.add_argument("--split-seed", type=int, default=0, help="train.py --seed of that run")
    ap.add_argument("--engine", default=None, help="engine wheel tag (engine/wheels), e.g. sdb.29")
    ap.add_argument("--json", default=None, help="also write the results here")
    args = ap.parse_args(argv)
    if args.engine:
        from showdown_bot.engines import engine_dir

        sys.path.insert(0, str(engine_dir(args.engine)))
    only = None
    if args.val_split is not None:
        only = val_split_tags(args.data, args.val_split, args.split_seed)
        print(f"val split: {len(only)} battles")
    res = evaluate_models(args.models, args.data, args.max_games, args.per_game, args.seed, only)
    n = res["n"]
    print(f"{n} positions, mean outcome {res['mean_outcome']:.3f}, {res['seconds']:.1f} s")
    first = Path(args.models[0]).stem
    for m in res["models"]:
        a = res["models"][m]["all"]
        print(f"  {Path(m).stem:32s} logit vs {first}: rms {a['rms_vs_first']:.4f} mean abs {a['mad_vs_first']:.4f}")
    names = ["all", *NAMES]
    print(f"{'bucket':>7} {'n':>8} | {'const bce':>9} {'acc':>6} | " + " | ".join(
        f"{Path(m).stem[:22]:>22}" for m in res["models"]))
    for name in names:
        if name not in res["constant"]:
            continue
        c = res["constant"][name]
        cells = []
        for m in res["models"]:
            r = res["models"][m][name]
            cells.append(f"{r['bce']:.4f} acc {r['acc']:.3f}".rjust(22))
        print(f"{name:>7} {c['n']:>8} | {c['bce']:>9.4f} {c['acc']:>6.3f} | " + " | ".join(cells))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
