"""Train a value net on recorded self-play positions and export it as a ``.pev``.

    python -m showdown_bot.train --data data/selfplay/<run> [--data ...] \\
        --model mlp --hidden 64 --out models/<name>.pev

The net has exactly the engine's architecture (engine/README.md, learned.rs):
``mlp``: ``Linear(738, hidden) -> ReLU -> Linear(hidden, 1)``; ``linear``:
``Linear(738, 1)``. The output is a logit of P(side one wins), trained with
binary cross-entropy against the game outcome (1 win, 0.5 tie, 0 loss).

Data and split
    Samples come from ``showdown_bot.dataset.iter_samples`` (side one's view).
    Positions of one battle are strongly correlated, so train/val is split by
    ``battle_tag`` (both players' recordings of a battle share a tag and stay
    together). ``--states`` picks exact "truth" states, sampled opponent
    "worlds", or both (default).

Weighting (``--weighting``)
    Every position carries its battle's outcome, so long games would otherwise
    dominate the loss. ``sqrt`` (default) weights a position by
    ``1/sqrt(n_positions_in_its_battle)``: a battle of 4x the length counts 2x
    in total rather than 4x. ``game`` gives every battle equal total weight,
    ``none`` weights every position equally. Weights are normalised to mean 1
    within the train and val sets. Early stopping uses the weighted val BCE;
    the report also gives unweighted numbers.

Device: cuda when available, else cpu (``--device`` overrides).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from showdown_bot import dataset

WEIGHTINGS = ("sqrt", "game", "none")
STATE_CHOICES = ("both", "truth", "worlds")
N_BUCKETS = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Data:
    x: np.ndarray  # (N, n_in) float32
    y: np.ndarray  # (N,) float32 in {0, 0.5, 1}
    w: np.ndarray  # (N,) float32 sample weights, mean 1
    tags: list[str]  # battle_tag per sample
    states: list[str | None]  # State string per sample (for export verification)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> Data:
        w = self.w[idx]
        mean = float(w.mean()) if len(w) else 1.0
        return Data(
            self.x[idx], self.y[idx], (w / mean).astype(np.float32),
            [self.tags[i] for i in idx], [self.states[i] for i in idx],
        )


def sample_weights(tags: list[str], weighting: str) -> np.ndarray:
    """Per-sample weights (mean 1) that stop long battles dominating; see module doc."""
    n = len(tags)
    if weighting == "none" or n == 0:
        return np.ones(n, dtype=np.float32)
    counts: dict[str, int] = defaultdict(int)
    for tag in tags:
        counts[tag] += 1
    power = 0.5 if weighting == "sqrt" else 1.0
    w = np.array([counts[t] ** -power for t in tags], dtype=np.float64)
    return (w / w.mean()).astype(np.float32)


def load_data(paths: list[Path], states: str = "both", weighting: str = "sqrt") -> Data:
    """Load every sample of the given run directories into memory."""
    feats, ys, tags, strings = [], [], [], []
    load = dataset.LoadStats()
    for path in paths:
        for s in dataset.iter_samples(
            path, sides=(1,), worlds=states in ("both", "worlds"),
            truth=states in ("both", "truth"), stats=load,
        ):
            feats.append(s.features)
            ys.append(s.value)
            tags.append(s.battle_tag)
            strings.append(s.state)
    if load.bad_states:
        print(f"warning: {load.bad_states} unparsable states skipped", file=sys.stderr)
    if not feats:
        raise SystemExit("no samples found in " + ", ".join(map(str, paths)))
    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return Data(x, y, sample_weights(tags, weighting), tags, strings)


def split_by_battle(
    tags: list[str], val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (train, val) such that no battle_tag appears in both."""
    unique = sorted(set(tags))
    random.Random(seed).shuffle(unique)
    n_val = int(round(len(unique) * val_fraction))
    n_val = min(max(n_val, 1 if len(unique) > 1 else 0), max(len(unique) - 1, 0))
    val_tags = set(unique[:n_val])
    is_val = np.array([t in val_tags for t in tags], dtype=bool)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


# ---------------------------------------------------------------------------
# Model, loss, metrics
# ---------------------------------------------------------------------------


def build_model(kind: str, n_in: int, hidden: int) -> nn.Module:
    if kind == "linear":
        return nn.Linear(n_in, 1)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    raise ValueError(f"unknown model {kind!r}")


def logits_of(model: nn.Module, x: torch.Tensor, batch: int = 8192) -> torch.Tensor:
    model.eval()
    if len(x) == 0:
        return torch.zeros(0)
    with torch.no_grad():
        return torch.cat([model(x[i:i + batch]).squeeze(-1) for i in range(0, len(x), batch)])


def weighted_bce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor | None = None) -> float:
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if w is None:
        return float(loss.mean())
    return float((loss * w).sum() / w.sum())


def constant_baseline_bce(
    y: np.ndarray, w: np.ndarray, y_ref: np.ndarray, w_ref: np.ndarray
) -> float:
    """BCE on (y, w) of the best constant predictor: the (weighted) mean of y_ref."""
    p = float(np.clip((y_ref * w_ref).sum() / w_ref.sum(), 1e-6, 1 - 1e-6))
    loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return float((loss * w).sum() / w.sum())


def decisive_accuracy(logits: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Accuracy of sign(logit) on positions whose outcome is a win or a loss."""
    mask = y != 0.5
    if not mask.any():
        return float("nan"), 0
    return float(((logits[mask] > 0) == (y[mask] > 0.5)).mean()), int(mask.sum())


def calibration_buckets(logits: np.ndarray, y: np.ndarray, n: int = N_BUCKETS):
    """[(lo, hi, count, mean_predicted, mean_outcome)] over equal-width P bins."""
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    edges = np.linspace(0, 1, n + 1)
    idx = np.minimum((p * n).astype(int), n - 1)
    out = []
    for b in range(n):
        m = idx == b
        out.append((
            float(edges[b]), float(edges[b + 1]), int(m.sum()),
            float(p[m].mean()) if m.any() else float("nan"),
            float(y[m].mean()) if m.any() else float("nan"),
        ))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    model: nn.Module
    best_epoch: int
    epochs_run: int
    history: list[tuple[int, float, float]]  # (epoch, train weighted BCE, val weighted BCE)


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def fit(
    model: nn.Module,
    train: Data,
    val: Data,
    *,
    device: torch.device,
    epochs: int = 60,
    patience: int = 6,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    log=print,
) -> FitResult:
    """AdamW on weighted BCE; early stopping on weighted val BCE (best weights restored)."""
    torch.manual_seed(seed)
    model.to(device)
    xt, yt, wt = (torch.from_numpy(a).to(device) for a in (train.x, train.y, train.w))
    xv, yv, wv = (torch.from_numpy(a).to(device) for a in (val.x, val.y, val.w))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[tuple[int, float, float]] = []
    n = len(yt)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(xt[b]).squeeze(-1), yt[b], reduction="none"
            )
            loss = (loss * wt[b]).sum() / wt[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tr = weighted_bce(logits_of(model, xt), yt, wt)
        va = weighted_bce(logits_of(model, xv), yv, wv)
        history.append((epoch, tr, va))
        log(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}")
        if va < best_loss - 1e-5:
            best_loss, best_epoch, stale = va, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model, best_epoch, len(history), history)


# ---------------------------------------------------------------------------
# Export + verification
# ---------------------------------------------------------------------------


def export_pev(model: nn.Module, out: Path) -> None:
    """Write the model with ``poke_engine.write_mlp`` / ``write_linear``.

    Weight layout is torch's: ``W1`` is ``[hidden][n_in]`` row-major, which is
    what learned.rs reads.
    """
    import poke_engine

    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    if isinstance(model, nn.Linear):
        poke_engine.write_linear(
            str(out), model.weight.detach().reshape(-1).tolist(), float(model.bias.item())
        )
        return
    l1, l2 = model[0], model[2]
    poke_engine.write_mlp(
        str(out),
        l1.weight.detach().reshape(-1).tolist(),
        l1.bias.detach().tolist(),
        l2.weight.detach().reshape(-1).tolist(),
        float(l2.bias.item()),
    )


def verify_export(model: nn.Module, path: Path, states: list[str], tol: float = 1e-3) -> float:
    """Load ``path`` with ``Evaluator.load`` and compare ``evaluate`` to the torch logits.

    Returns the max absolute difference over ``states`` (raises if above ``tol``).
    """
    import poke_engine

    evaluator = poke_engine.Evaluator.load(str(path))
    model = model.to("cpu")
    parsed = [poke_engine.State.from_string(s) for s in states]
    feats = torch.tensor(
        [poke_engine.state_features(s, 1) for s in parsed], dtype=torch.float32
    )
    expected = logits_of(model, feats).numpy()
    got = np.array([evaluator.evaluate(s) for s in parsed])
    worst = float(np.abs(expected - got).max())
    if worst > tol:
        raise RuntimeError(f"exported evaluator differs from torch by {worst:.2e} (> {tol})")
    return worst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def report(model: nn.Module, train: Data, val: Data, device: torch.device) -> dict[str, float]:
    model.to(device)
    t = {k: torch.from_numpy(getattr(train, k)).to(device) for k in ("x", "y", "w")}
    v = {k: torch.from_numpy(getattr(val, k)).to(device) for k in ("x", "y", "w")}
    lt, lv = logits_of(model, t["x"]), logits_of(model, v["x"])
    out = {
        "train_bce_w": weighted_bce(lt, t["y"], t["w"]),
        "val_bce_w": weighted_bce(lv, v["y"], v["w"]),
        "train_bce": weighted_bce(lt, t["y"]),
        "val_bce": weighted_bce(lv, v["y"]),
        "base_val_bce_w": constant_baseline_bce(val.y, val.w, train.y, train.w),
        "base_val_bce": constant_baseline_bce(
            val.y, np.ones_like(val.w), train.y, np.ones_like(train.w)
        ),
    }
    lv_np = lv.cpu().numpy()
    out["val_acc"], n_dec = decisive_accuracy(lv_np, val.y)
    print("\n=== metrics (best epoch) ===")
    print(f"train BCE   weighted {out['train_bce_w']:.4f}   unweighted {out['train_bce']:.4f}")
    print(f"val BCE     weighted {out['val_bce_w']:.4f}   unweighted {out['val_bce']:.4f}")
    print(f"constant-predictor val BCE (train mean)  weighted {out['base_val_bce_w']:.4f}   "
          f"unweighted {out['base_val_bce']:.4f}   (ln2 = {math.log(2):.4f})")
    print(f"val BCE improvement over constant: "
          f"weighted {out['base_val_bce_w'] - out['val_bce_w']:+.4f}  "
          f"unweighted {out['base_val_bce'] - out['val_bce']:+.4f}")
    print(f"val accuracy on decisive positions: {out['val_acc']:.3f}  ({n_dec} positions)")
    print("calibration on val (predicted P(win) bucket: n, mean predicted, observed outcome)")
    for lo, hi, n, mp, mo in calibration_buckets(lv_np, val.y):
        if n:
            print(f"  [{lo:.1f},{hi:.1f})  n={n:7d}  pred={mp:.3f}  obs={mo:.3f}")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=Path, action="append", required=True,
                   help="self-play run directory (repeatable)")
    p.add_argument("--model", choices=("mlp", "linear"), default="mlp")
    p.add_argument("--hidden", type=int, default=64, help="MLP hidden units")
    p.add_argument("--out", type=Path, required=True, help="output .pev path")
    p.add_argument("--states", choices=STATE_CHOICES, default="both")
    p.add_argument("--weighting", choices=WEIGHTINGS, default="sqrt")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    for path in args.data:
        if not path.exists():
            print(f"no such path: {path}", file=sys.stderr)
            return 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}")
    data = load_data(args.data, args.states, args.weighting)
    tr_idx, va_idx = split_by_battle(data.tags, args.val_fraction, args.seed)
    if len(va_idx) == 0 or len(tr_idx) == 0:
        print("need at least two battles for a train/val split", file=sys.stderr)
        return 2
    train, val = data.subset(tr_idx), data.subset(va_idx)
    assert not set(train.tags) & set(val.tags)
    print(f"samples {len(data)} ({args.states}); battles {len(set(data.tags))}; "
          f"train {len(train)} samples / {len(set(train.tags))} battles; "
          f"val {len(val)} samples / {len(set(val.tags))} battles; "
          f"mean value {float(data.y.mean()):.3f}")
    model = build_model(args.model, data.x.shape[1], args.hidden)
    result = fit(
        model, train, val, device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
    )
    print(f"best epoch {result.best_epoch} of {result.epochs_run}")
    report(result.model, train, val, device)
    export_pev(result.model, args.out)
    rng = random.Random(args.seed)
    pool = [s for s in val.states if s]
    check = rng.sample(pool, min(100, len(pool)))
    worst = verify_export(result.model, args.out, check)
    print(f"\nexported {args.out}  ({args.out.stat().st_size} bytes); verified on {len(check)} "
          f"val states, max |evaluate - torch logit| = {worst:.2e}")
    print(f"total time {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
