"""Hindsight-gap baseline: train the MLP value net on one conversion mode, score it on both.

    python scripts/hindsight_gap.py --train data/human/pc1-aux/train --test hind=data/human/pc1-aux/test \
        --test fp=data/human/pc1-fp-aux/test --max-per-game 3 --out gap-hind.json

Same recipe as ``train.py --model mlp --hidden 64 --residual --value-target outcome --features v2
--states truth``; after fitting, the net is scored (unweighted BCE vs the game outcome, decisive
accuracy) on every ``--test`` set, overall and per turn bucket, next to the constant predictor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from showdown_bot import train as T  # noqa: E402

BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10_000)]


def score(logits: np.ndarray, y: np.ndarray, turns: np.ndarray, const_p: float) -> list[dict]:
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    q = np.clip(const_p, 1e-6, 1 - 1e-6)
    cbce = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    rows = []
    for name, lo, hi in [("all", 0, 10**9), *[(f"{a}-{b}" if b < 10_000 else f"{a}+", a, b) for a, b in BUCKETS]]:
        m = (turns >= lo) & (turns <= hi)
        if not m.any():
            continue
        dec = m & (y != 0.5)
        acc = float(((logits[dec] > 0) == (y[dec] > 0.5)).mean()) if dec.any() else float("nan")
        rows.append({"bucket": name, "n": int(m.sum()), "bce": float(bce[m].mean()),
                     "const_bce": float(cbce[m].mean()), "acc": acc})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--max-per-game", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--features", choices=("v1", "v2"), default="v2")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fv = 2 if args.features == "v2" else 1
    dtype = "float16" if fv == 2 else "float32"
    kw = dict(states="truth", value_target="outcome", residual=True, feature_version=fv,
              x_dtype=dtype, max_per_game=args.max_per_game, seed=args.seed,
              workers=args.load_workers)
    device = T.select_device("auto")
    data = T.load_data([args.train], **kw)
    tr_idx, va_idx = T.split_by_battle(data.tags, 0.1, args.seed)
    train, val = data.split_inplace(tr_idx, va_idx)
    del data
    print(f"train {len(train)} samples, val {len(val)}", flush=True)
    n_feat = train.x.shape[1] - (1 if train.e_sep is None else 0)
    a0, b0 = T.fit_stock_logistic(train.e, train.y_fit, train.w)
    model = T.build_model("mlp", n_feat, args.hidden, (a0, b0))
    result = T.fit(model, train, val, device=device, epochs=args.epochs, patience=args.patience,
                   batch_size=args.batch_size, seed=args.seed)
    model = result.model
    const_p = float((train.y * train.w).sum() / train.w.sum())
    del train, val
    out = {"train": args.train, "max_per_game": args.max_per_game, "best_epoch": result.best_epoch,
           "epochs_run": result.epochs_run, "tests": {}}
    for item in args.test:
        name, path = item.split("=", 1)
        td = T.load_data([path], **kw)
        x = T.place_x(td.x, device)
        e = None if td.e_sep is None else torch.from_numpy(td.e_sep).to(device)
        logits = T.logits_of(model, x, e).cpu().numpy()
        out["tests"][name] = score(logits, td.y, td.didx + 1, const_p)
        print(name, json.dumps(out["tests"][name][0]), flush=True)
        del td, x
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
