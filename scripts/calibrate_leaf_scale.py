"""Suggest a learned-evaluator leaf scale k from recorded self-play games.

    python scripts/calibrate_leaf_scale.py --data runs/selfplay1 --model models/x.pev

Stock MCTS scores a leaf as sigmoid(0.0125 * (eval_leaf - eval_root)). The
learned score is sigmoid(k * (z_leaf - z_root)). For consecutive recorded
decisions of one game (states s_t, s_t+1) this script measures

    d_stock = 0.0125 * (evaluate(s_t+1) - evaluate(s_t))
    d_net   = z(s_t+1) - z(s_t)

and suggests k = std(d_stock) / std(d_net) (also a MAD-based, outlier-robust
version) so the net's score spread matches stock's. It is a starting point for a
duel sweep (k/2, k, 2k), not an optimum.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

STOCK_SCALE = 0.0125


def _mad(xs: list[float]) -> float:
    """Median absolute deviation, scaled to be comparable with a std (x1.4826)."""
    med = statistics.median(xs)
    return 1.4826 * statistics.median(abs(x - med) for x in xs)


def suggest_scale(d_stock: list[float], d_net: list[float]) -> dict[str, float]:
    """Std / MAD of each delta series and the implied scale ratios.

    Ratios are ``nan`` when the net's spread is zero.
    """
    if len(d_stock) != len(d_net) or len(d_stock) < 2:
        raise ValueError("need two equal-length delta series with at least 2 entries")
    out = {
        "n": float(len(d_stock)),
        "std_stock": statistics.pstdev(d_stock),
        "std_net": statistics.pstdev(d_net),
        "mad_stock": _mad(d_stock),
        "mad_net": _mad(d_net),
    }
    nan = float("nan")
    out["k_std"] = out["std_stock"] / out["std_net"] if out["std_net"] > 0 else nan
    out["k_mad"] = out["mad_stock"] / out["mad_net"] if out["mad_net"] > 0 else nan
    return out


def collect_deltas(data_dirs, model: str, max_games: int | None):
    import poke_engine

    from showdown_bot import recorder

    ev = poke_engine.Evaluator.load(model)
    d_stock: list[float] = []
    d_net: list[float] = []
    games = 0
    for root in data_dirs:
        for path in recorder.iter_game_files(root):
            if max_games is not None and games >= max_games:
                return d_stock, d_net, games
            try:
                game = recorder.read_game(path)
            except (OSError, ValueError, EOFError):
                continue
            games += 1
            prev = None
            for decision in game["decisions"]:
                text = (decision.get("truth") or {}).get("state")
                if not text and decision.get("worlds"):
                    text = decision["worlds"][0]["state"]
                if not text:
                    continue
                try:
                    state = poke_engine.State.from_string(text)
                    cur = (poke_engine.evaluate(state), ev.logit(state))
                except BaseException as exc:  # pyo3 PanicException
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    prev = None
                    continue
                if prev is not None:
                    d_stock.append(STOCK_SCALE * (cur[0] - prev[0]))
                    d_net.append(cur[1] - prev[1])
                prev = cur
    return d_stock, d_net, games


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, action="append", required=True,
                   help="self-play recording dir (repeatable)")
    p.add_argument("--model", type=Path, required=True, help="learned .pev file")
    p.add_argument("--max-games", type=int, default=None, help="cap on games read")
    args = p.parse_args(argv)
    if not args.model.is_file():
        print(f"no such model: {args.model}", file=sys.stderr)
        return 2
    d_stock, d_net, games = collect_deltas(args.data, str(args.model), args.max_games)
    if len(d_stock) < 2:
        print(f"only {len(d_stock)} consecutive state pairs in {games} games", file=sys.stderr)
        return 1
    r = suggest_scale(d_stock, d_net)
    print(f"games: {games}, consecutive pairs: {int(r['n'])}")
    print(f"stock d = 0.0125*dEval : std {r['std_stock']:.4f}  MAD {r['mad_stock']:.4f}")
    print(f"net   d = dz           : std {r['std_net']:.4f}  MAD {r['mad_net']:.4f}")
    print(f"suggested k (std): {r['k_std']:.3f}")
    print(f"suggested k (MAD): {r['k_mad']:.3f}")
    k = r["k_mad"] if r["k_mad"] == r["k_mad"] else r["k_std"]
    if k == k:
        print(f"sweep (duel.py --leaf-scale): {k / 2:.3f}  {k:.3f}  {2 * k:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
