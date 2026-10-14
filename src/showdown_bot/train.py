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
    "worlds", both (default), or ``prefer-truth`` (truth where a decision has
    one, else its sampled worlds: low memory on self-play, still uses
    ladder/league records that lack truth).

Weighting (``--weighting``)
    Every position carries its battle's outcome, so long games would otherwise
    dominate the loss. ``sqrt`` (default) weights a position by
    ``1/sqrt(n_positions_in_its_battle)``: a battle of 4x the length counts 2x
    in total rather than 4x. ``game`` gives every battle equal total weight,
    ``none`` weights every position equally. Weights are normalised to mean 1
    within the train and val sets. Early stopping uses the weighted val BCE;
    the report also gives unweighted numbers.

Sources, subsampling and source weights (``--data``, ``--val-data``)
    Each ``--data`` is ``PATH[@K][:wW]``: ``@K`` keeps at most K decisions per game
    record from that source (overrides the global ``--max-per-game``, default 0 = all;
    ``@0`` = all), ``:wW`` multiplies the weights of that source's samples by W (after
    the ``--weighting`` weights, before normalisation to mean 1), e.g.
    ``--data data/human/pc1/train@3:w0.5 --data data/selfplay/gen4@4``. The K decisions
    are drawn uniformly without replacement among a record's decisions that have a state
    under ``--states``, seeded by ``--seed`` and the record's file name (reproducible),
    BEFORE feature extraction, so memory and load time shrink proportionally (many
    positions per game make the value net memorise outcomes; fewer per game from more
    games generalise better). ``td``/``nstep`` targets look ahead only among the kept
    decisions, so avoid subsampling with them. Per-source counts (records, decisions
    kept, samples) are printed at load time, and with a non-outcome ``--value-target``
    also how many samples per source had no search value and fell back to the outcome
    (e.g. human records). ``--val-data DIR`` (repeatable, same syntax) adds extra
    held-out sets, never trained on: each epoch's log line and the final report give the
    BCE against the outcome. They do not affect early stopping unless
    ``--early-stop-on val-data`` (then the first ``--val-data`` set decides).

Value target (``--value-target``)
    ``outcome`` (default): the game outcome z. ``search``: q, the MCTS root
    average score recorded per decision (``dataset.Sample.search_value``).
    ``blend``: ``(1 - L) * z + L * q`` with ``--blend-lambda L``. Positions
    without a usable q fall back to z. ``--q-kind absolute|relative`` (default absolute): the recorded score is
    relative to the root evaluation, so absolute q adds it back,
    ``sigmoid(scale*eval(state) + logit(avg_score))``, ``--q-eval-scale``
    (0.0125). ``--q-source decision|world`` picks the
    decision's mean over worlds (default) or each world's own score. Metrics are
    reported against the training target and against the pure outcome z.

Bootstrapped targets (``--value-target td|nstep``)
    Look ahead within the same recording: ``td`` is the lambda-return over later
    decisions' absolute q ending at z (``--td-lambda``; 1 = z, 0 = next decision's q),
    ``nstep`` the q ``--td-n`` decisions later (z past the end). ``--td-include-self
    --td-mu M`` blends the decision's own q. See docs/training.md.

Residual evaluator (``--residual``)
    The stock evaluation is the starting point and the net learns only a
    correction: ``z = a * 0.0125 * evaluate(state) + net(features)``. ``a`` is a
    learnable scalar initialised from a weighted logistic fit of the outcome on
    ``0.0125 * evaluate`` (with intercept b) on the train split, or
    ``--stock-coef-init``; ``--freeze-stock-coef`` keeps it fixed. The net's
    output layer starts at zero weights and bias b, so the untrained model IS
    the calibrated stock logistic; its val BCE is printed as the baseline. The
    export is a format-2 ``.pev`` with ``stock_coef = a * 0.0125``.

Augmentation and regularisation (training batches only; validation is never touched)
    ``--augment-bench`` randomly permutes the bench slots 1..5 of each side per
    sample and batch; ``--augment-sides`` swaps own/opp with probability 1/2
    (exactly ``state_features(state, 2)``) and flips the label to ``1 - y``
    (outcome and q targets; in residual mode the stock evaluation changes sign).
    See ``showdown_bot.augment``. ``--dropout P`` adds dropout between the hidden
    ReLU and the output layer during training only; the export is unaffected.

Device: cuda when available, else cpu (``--device`` overrides).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from showdown_bot import augment, dataset, recorder

WEIGHTINGS = ("sqrt", "game", "none")
STATE_CHOICES = ("both", "truth", "worlds", "prefer-truth")
VALUE_TARGETS = ("outcome", "search", "blend", "td", "nstep")
N_BUCKETS = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One ``--data`` / ``--val-data`` entry: a directory plus its subsample / weight options."""

    path: Path
    max_per_game: int | None = None  # None = use the global default
    weight: float = 1.0


_SOURCE_RE = re.compile(r"^(.*?)(?:@(\d+))?(?::w(\d+(?:\.\d*)?|\.\d+))?$", re.DOTALL)


def parse_source(spec: str | Path | Source) -> Source:
    """Parse ``PATH[@K][:wW]`` (``@K`` decisions per record, ``:wW`` weight multiplier)."""
    if isinstance(spec, Source):
        return spec
    m = _SOURCE_RE.match(str(spec))
    path, k, w = m.groups()  # the regex always matches
    if not path:
        raise argparse.ArgumentTypeError(f"empty path in data spec {str(spec)!r}")
    weight = 1.0 if w is None else float(w)
    if weight <= 0:
        raise argparse.ArgumentTypeError(f"source weight must be > 0 in {str(spec)!r}")
    return Source(Path(path), None if k is None else int(k), weight)


@dataclass
class Data:
    x: np.ndarray  # (N, n_in) float32
    y: np.ndarray  # (N,) float32 in {0, 0.5, 1}
    w: np.ndarray  # (N,) float32 sample weights, mean 1
    tags: list[str]  # battle_tag per sample
    states: list[str | None]  # State string per sample (for export verification)
    # Search value q per sample (NaN = unavailable) and the training target built
    # from y and q. ``y`` is always the pure outcome; ``target`` None means y.
    q: np.ndarray | None = None
    target: np.ndarray | None = None
    stock_eval: np.ndarray | None = None  # poke_engine.evaluate per sample (side one)
    # residual mode: the last column of ``x`` is the raw stock evaluate(state), unless
    # ``e_sep`` holds it (used with float16 features, which cannot carry a large eval)
    residual: bool = False
    e_sep: np.ndarray | None = None
    feature_version: int = 1

    @property
    def e(self) -> np.ndarray:
        """Raw stock evaluation per sample (residual data only)."""
        assert self.residual
        return self.e_sep if self.e_sep is not None else self.x[:, -1]

    @property
    def y_fit(self) -> np.ndarray:
        """The labels the loss is computed against."""
        return self.y if self.target is None else self.target

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> Data:
        w = self.w[idx]
        mean = float(w.mean()) if len(w) else 1.0
        return Data(
            self.x[idx], self.y[idx], (w / mean).astype(np.float32),
            [self.tags[i] for i in idx], [self.states[i] for i in idx],
            None if self.q is None else self.q[idx],
            None if self.target is None else self.target[idx],
            None if self.stock_eval is None else self.stock_eval[idx],
            self.residual,
            None if self.e_sep is None else self.e_sep[idx],
            self.feature_version,
        )


def sample_weights(
    tags: list[str], weighting: str, multiplier: np.ndarray | None = None
) -> np.ndarray:
    """Per-sample weights (mean 1) that stop long battles dominating; see module doc.

    ``multiplier`` (per-sample, e.g. the per-source weight) is applied on top of the
    ``weighting`` weights before the normalisation to mean 1.
    """
    n = len(tags)
    if n == 0:
        return np.ones(0, dtype=np.float32)
    if weighting == "none":
        w = np.ones(n, dtype=np.float64)
    else:
        counts: dict[str, int] = defaultdict(int)
        for tag in tags:
            counts[tag] += 1
        power = 0.5 if weighting == "sqrt" else 1.0
        w = np.array([counts[t] ** -power for t in tags], dtype=np.float64)
    if multiplier is not None:
        w = w * np.asarray(multiplier, dtype=np.float64)
    return (w / w.mean()).astype(np.float32)


def make_targets(
    y: np.ndarray, q: np.ndarray | None, mode: str = "outcome", blend_lambda: float = 0.5
) -> tuple[np.ndarray, int]:
    """Training labels from outcome ``y`` and search value ``q`` (NaN = missing).

    ``outcome`` returns ``y`` itself; ``search`` returns q; ``blend`` returns
    ``(1 - L) * y + L * q``. Positions with missing q use ``y``. Returns
    ``(target, n_fallback)``.
    """
    if mode not in VALUE_TARGETS:
        raise ValueError(f"unknown value target {mode!r}")
    if mode == "outcome" or q is None:
        return y, 0
    missing = np.isnan(q)
    lam = 1.0 if mode == "search" else float(blend_lambda)
    target = np.where(missing, y, (1 - lam) * y + lam * np.nan_to_num(q))
    return target.astype(np.float32), int(missing.sum())


LOAD_CHUNK_FILES = 200  # game files per worker job


def _load_chunk(job: tuple) -> dict:
    """Worker: load one chunk of game files into compact numpy arrays (top level for spawn)."""
    (files, max_per_game, kw) = job
    stats = dataset.LoadStats()
    states, q_kind = kw["states"], kw["q_kind"]
    feats, ys, qs, evals, dqs, didx = [], [], [], [], [], []
    tags, strings, gids = [], [], []
    seen = set()
    for s in dataset.iter_samples(
        files[0], sides=(1,), max_per_game=max_per_game, seed=kw["seed"],
        worlds="fallback" if states == "prefer-truth" else states in ("both", "worlds"),
        truth=states in ("both", "truth", "prefer-truth"), stats=stats, q_source=kw["q_source"],
        q_eval_scale=kw["q_eval_scale"], feature_version=kw["feature_version"], files=files,
    ):
        feats.append(np.asarray(s.features, dtype=np.dtype(kw["x_dtype"])))
        ys.append(s.value)
        q = s.search_value_abs if q_kind == "absolute" else s.search_value
        qs.append(np.nan if q is None else q)
        evals.append(s.stock_eval)
        tags.append(s.battle_tag)
        strings.append(s.state)
        gids.append(s.game_id)
        didx.append(s.decision_idx)
        seen.add((s.game_id, s.decision_idx))
        dq = s.decision_q_abs if q_kind == "absolute" else s.decision_q
        dqs.append(np.nan if dq is None else dq)
    return {
        "x": np.stack(feats) if feats else None,
        "ys": np.asarray(ys, dtype=np.float32), "qs": np.asarray(qs, dtype=np.float32),
        "evals": np.asarray(evals, dtype=np.float32), "dqs": np.asarray(dqs, dtype=np.float64),
        "didx": np.asarray(didx, dtype=np.int64),
        "tags": tags, "strings": strings, "gids": gids, "decisions": len(seen),
        "stats": {"games": stats.games, "skipped_games": stats.skipped_games,
                  "bad_states": stats.bad_states, "results": dict(stats.results)},
    }


def _load_source_parallel(path: Path, k: int, workers: int, kw: dict) -> dict:
    chunk = LOAD_CHUNK_FILES
    files = list(recorder.iter_game_files(path))
    jobs = [(files[i:i + chunk], k, kw) for i in range(0, len(files), chunk)]
    out: dict = {n: [] for n in ("x", "ys", "qs", "evals", "dqs", "didx", "tags", "strings", "gids")}
    out["decisions"] = 0
    out["stats"] = {"games": 0, "skipped_games": 0, "bad_states": 0, "results": Counter()}

    def absorb(r: dict) -> None:
        if r["x"] is not None:
            out["x"].append(r["x"])
        for n in ("ys", "qs", "evals", "dqs", "didx"):
            out[n].append(r[n])
        for n in ("tags", "strings", "gids"):
            out[n] += r[n]
        out["decisions"] += r["decisions"]
        for n in ("games", "skipped_games", "bad_states"):
            out["stats"][n] += r["stats"][n]
        out["stats"]["results"].update(r["stats"]["results"])

    if not jobs:
        return out
    pending: deque = deque()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for job in jobs:  # results are consumed in submission order -> deterministic
            pending.append(ex.submit(_load_chunk, job))
            if len(pending) >= 2 * workers:
                absorb(pending.popleft().result())
        while pending:
            absorb(pending.popleft().result())
    return out


def load_data(
    paths: list[Path | str | Source],
    states: str = "both",
    weighting: str = "sqrt",
    value_target: str = "outcome",
    blend_lambda: float = 0.5,
    q_source: str = "decision",
    q_kind: str = "absolute",
    q_eval_scale: float = dataset.Q_EVAL_SCALE,
    residual: bool = False,
    td_lambda: float = 0.7,
    td_n: int = 1,
    td_mu: float = 0.0,
    feature_version: int = 1,
    x_dtype: str = "float32",
    max_per_game: int = 0,
    seed: int = 0,
    workers: int = 0,
) -> Data:
    """Load the samples of the given run directories (``Source`` / path / spec) into memory.

    ``max_per_game`` is the default cap on decisions per game record (0 = all); a
    ``Source.max_per_game`` overrides it, and ``Source.weight`` multiplies the sample
    weights. Prints per-source counts.

    ``residual`` appends the raw stock ``evaluate(state)`` as the last column of ``x``
    (with ``x_dtype="float16"`` it is kept in a separate float32 array instead).
    ``feature_version`` selects the engine feature layout (1 or 2); float16 halves the memory
    of the (N, n_in) matrix, which matters for the 2.1k-wide version 2.

    ``workers``: 0 = ``os.cpu_count() - 2`` (min 1), 1 = serial, >1 = that many processes
    each loading a chunk of game files (results are identical to serial: the per-game
    subsampling RNG depends only on ``seed`` and the file name, and chunks are joined in order).
    """
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - 2)
    np_dtype = np.dtype(x_dtype)
    feats, ys, qs, evals, tags, strings = [], [], [], [], [], []
    gids, didx, dqs = [], [], []
    load = dataset.LoadStats()
    sources = [parse_source(p) for p in paths]
    src_ids: list[int] = []
    src_records, src_decisions = [], []
    par_x: np.ndarray | None = None
    if workers > 1:
        kw = dict(states=states, q_source=q_source, q_kind=q_kind, q_eval_scale=q_eval_scale,
                  feature_version=feature_version, x_dtype=x_dtype, seed=seed)
        px, pcols = [], {n: [] for n in ("ys", "qs", "evals", "dqs", "didx")}
        for si, source in enumerate(sources):
            k = max_per_game if source.max_per_game is None else source.max_per_game
            r = _load_source_parallel(source.path, k, workers, kw)
            px += r["x"]
            for n in pcols:
                pcols[n] += r[n]
            tags += r["tags"]; strings += r["strings"]; gids += r["gids"]
            src_ids += [si] * len(r["tags"])
            load.games += r["stats"]["games"]
            load.skipped_games += r["stats"]["skipped_games"]
            load.bad_states += r["stats"]["bad_states"]
            load.results.update(r["stats"]["results"])
            src_records.append(r["stats"]["games"])
            src_decisions.append(r["decisions"])
        if px:
            par_x = np.concatenate(px)
            del px
            ys, qs, evals, dqs, didx = (
                np.concatenate(pcols[n]) for n in ("ys", "qs", "evals", "dqs", "didx"))
            feats = [None]  # non-empty marker
    for si, source in enumerate(sources if workers <= 1 else ()):
        path = source.path
        k = max_per_game if source.max_per_game is None else source.max_per_game
        games_before, seen = load.games, set()
        for s in dataset.iter_samples(
            path, sides=(1,), max_per_game=k, seed=seed,
            worlds="fallback" if states == "prefer-truth" else states in ("both", "worlds"),
            truth=states in ("both", "truth", "prefer-truth"), stats=load, q_source=q_source,
            q_eval_scale=q_eval_scale, feature_version=feature_version,
        ):
            feats.append(np.asarray(s.features, dtype=np_dtype))  # a list of Python floats costs ~8x the memory
            ys.append(s.value)
            q = s.search_value_abs if q_kind == "absolute" else s.search_value
            qs.append(np.nan if q is None else q)
            evals.append(s.stock_eval)
            tags.append(s.battle_tag)
            src_ids.append(si)
            seen.add((s.game_id, s.decision_idx))
            strings.append(s.state)
            gids.append(s.game_id)
            didx.append(s.decision_idx)
            dq = s.decision_q_abs if q_kind == "absolute" else s.decision_q
            dqs.append(np.nan if dq is None else dq)
        src_records.append(load.games - games_before)
        src_decisions.append(len(seen))
    if load.bad_states:
        print(f"warning: {load.bad_states} unparsable states skipped", file=sys.stderr)
    if not feats:
        raise SystemExit("no samples found in " + ", ".join(map(str, paths)))
    x = par_x if par_x is not None else np.stack(feats)
    del feats
    e_sep = None
    if residual and np_dtype == np.float32:
        x = np.concatenate([x, np.asarray(evals, dtype=np.float32)[:, None]], axis=1)
    elif residual:
        e_sep = np.asarray(evals, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    q = np.asarray(qs, dtype=np.float32)
    src = np.asarray(src_ids, dtype=np.int64)
    for si, source in enumerate(sources):
        k = max_per_game if source.max_per_game is None else source.max_per_game
        print(f"source {source.path}: max-per-game {k or 'all'}, weight x{source.weight:g}: "
              f"{src_records[si]} records, {src_decisions[si]} decisions kept, "
              f"{int((src == si).sum())} samples")
    if value_target in ("td", "nstep"):
        target, st = dataset.bootstrapped_targets(
            gids, didx, dqs, y, value_target, td_lambda, td_n, td_mu)
        print(f"value target {value_target}: {st['gaps']} of {st['decisions']} decisions have no "
              f"search value (gap: next available q, else outcome)")
    else:
        target, n_fallback = make_targets(y, q, value_target, blend_lambda)
    if value_target in ("td", "nstep"):
        pass
    elif value_target != "outcome":
        print(f"value target {value_target}: {n_fallback} of {len(y)} samples have no search "
              f"value and fall back to the outcome")
        for si, source in enumerate(sources):
            m = src == si
            print(f"  source {source.path}: {int(np.isnan(q[m]).sum())} of {int(m.sum())} "
                  f"samples fell back to the outcome")
    mult = np.asarray([s_.weight for s_ in sources], dtype=np.float64)[src]
    return Data(
        x, y, sample_weights(tags, weighting, mult), tags, strings, q,
        None if value_target == "outcome" else target,
        np.asarray(evals, dtype=np.float32), residual, e_sep, feature_version,
    )


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


def build_model(
    kind: str, n_in: int, hidden: int, residual: tuple[float, float] | None = None,
    freeze_a: bool = False, dropout: float = 0.0,
) -> nn.Module:
    """``n_in`` counts feature columns only. ``residual=(a, b)`` wraps the net in a ResidualNet."""
    if kind == "linear":
        net: nn.Module = nn.Linear(n_in, 1)
    elif kind == "mlp":
        layers: list[nn.Module] = [nn.Linear(n_in, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        net = nn.Sequential(*layers, nn.Linear(hidden, 1))
    else:
        raise ValueError(f"unknown model {kind!r}")
    if residual is not None:
        return ResidualNet(net, residual[0], residual[1], freeze_a)
    return net


STOCK_SCALE = 0.0125  # the stock search scale: P = sigmoid(0.0125 * evaluate)


class ResidualNet(nn.Module):
    """``z = a * 0.0125 * e + net(x)``; input is ``[x, e]`` (e = raw stock evaluate)."""

    def __init__(self, net: nn.Module, a: float, intercept: float = 0.0, freeze_a: bool = False):
        super().__init__()
        self.net = net
        self.a = nn.Parameter(torch.tensor(float(a)), requires_grad=not freeze_a)
        # zero output layer + intercept: the untrained model is the stock logistic fit
        out = net if isinstance(net, nn.Linear) else net[-1]
        nn.init.zeros_(out.weight)
        nn.init.constant_(out.bias, float(intercept))

    def forward(self, xe: torch.Tensor) -> torch.Tensor:
        return self.net(xe[:, :-1]) + STOCK_SCALE * self.a * xe[:, -1:]

    @property
    def stock_coef(self) -> float:
        return STOCK_SCALE * float(self.a.item())


def fit_stock_logistic(
    e: np.ndarray, y: np.ndarray, w: np.ndarray, fixed_a: float | None = None, iters: int = 50
) -> tuple[float, float]:
    """Weighted logistic fit ``y ~ sigmoid(a * 0.0125 * e + b)`` by Newton (IRLS).

    Returns ``(a, b)``. With ``fixed_a`` only the intercept is fitted.
    """
    s = STOCK_SCALE * e.astype(np.float64)
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    if fixed_a is not None:
        design, offset, beta = np.ones((len(s), 1)), fixed_a * s, np.zeros(1)
    else:
        design, offset, beta = np.stack([s, np.ones_like(s)], axis=1), 0.0, np.array([0.5, 0.0])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(design @ beta + offset)))
        grad = design.T @ (w * (p - y))
        hess = (design * (w * p * (1 - p))[:, None]).T @ design + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-9:
            break
    if fixed_a is not None:
        return float(fixed_a), float(beta[0])
    return float(beta[0]), float(beta[1])


def model_input(x: torch.Tensor, e: torch.Tensor | None = None) -> torch.Tensor:
    """Float32 network input: features (possibly float16), plus the stock eval column if given."""
    x = x.float()
    return x if e is None else torch.cat([x, e[:, None]], dim=1)


def logits_of(
    model: nn.Module, x: torch.Tensor, e: torch.Tensor | None = None, batch: int = 8192
) -> torch.Tensor:
    """Logits for features ``x``; ``e`` is the separately stored stock eval (float16 data)."""
    model.eval()
    if len(x) == 0:
        return torch.zeros(0)
    with torch.no_grad():
        return torch.cat([
            model(model_input(x[i:i + batch], None if e is None else e[i:i + batch])).squeeze(-1)
            for i in range(0, len(x), batch)
        ])


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
    # per epoch: weighted BCE vs the outcome on each extra ``--val-data`` set (same order)
    extra_history: list[list[float]] | None = None


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
    augmenter: augment.Augmenter | None = None,
    extra_val: list[tuple[str, Data]] | None = None,
    early_stop_on: str = "val",
) -> FitResult:
    """AdamW on weighted BCE; early stopping on weighted val BCE (best weights restored).

    ``extra_val`` are named held-out sets reported each epoch (weighted BCE vs the
    outcome); they only drive early stopping with ``early_stop_on="val-data"`` (the first
    one), otherwise they are just logged.
    """
    extra_val = extra_val or []
    if early_stop_on == "val-data" and not extra_val:
        raise ValueError("early_stop_on='val-data' needs extra_val")
    torch.manual_seed(seed)
    model.to(device)
    xt, yt, wt = (torch.from_numpy(a).to(device) for a in (train.x, train.y_fit, train.w))
    xv, yv, wv = (torch.from_numpy(a).to(device) for a in (val.x, val.y_fit, val.w))
    et = None if train.e_sep is None else torch.from_numpy(train.e_sep).to(device)
    ev = None if val.e_sep is None else torch.from_numpy(val.e_sep).to(device)
    extra_t = [
        (name, *(torch.from_numpy(a).to(device) for a in (d.x, d.y, d.w)),
         None if d.e_sep is None else torch.from_numpy(d.e_sep).to(device))
        for name, d in extra_val
    ]
    extra_history: list[list[float]] = []
    decayed = [p for n_, p in model.named_parameters() if p.requires_grad and n_ != "a"]
    groups = [{"params": decayed}]
    if isinstance(model, ResidualNet) and model.a.requires_grad:
        groups.append({"params": [model.a], "weight_decay": 0.0})  # a is not shrunk toward 0
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history: list[tuple[int, float, float]] = []
    n = len(yt)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            xb, yb = model_input(xt[b], None if et is None else et[b]), yt[b]
            if augmenter is not None:
                xb, yb = augmenter(xb, yb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(xb).squeeze(-1), yb, reduction="none"
            )
            loss = (loss * wt[b]).sum() / wt[b].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        tr = weighted_bce(logits_of(model, xt, et), yt, wt)
        va = weighted_bce(logits_of(model, xv, ev), yv, wv)
        history.append((epoch, tr, va))
        ex = [weighted_bce(logits_of(model, x_, e_), y_, w_) for _, x_, y_, w_, e_ in extra_t]
        extra_history.append(ex)
        log(f"epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}" + "".join(
            f"  val-data[{name}] {v:.4f}" for (name, *_), v in zip(extra_t, ex)))
        stop_metric = ex[0] if early_stop_on == "val-data" else va
        if stop_metric < best_loss - 1e-5:
            best_loss, best_epoch, stale = stop_metric, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model, best_epoch, len(history), history, extra_history)


def evaluate_extra(model: nn.Module, data: Data, device: torch.device) -> dict[str, float]:
    """BCE against the outcome z on an extra held-out set (weighted, unweighted, accuracy)."""
    model.to(device)
    x, y, w = (torch.from_numpy(a).to(device) for a in (data.x, data.y, data.w))
    e = None if data.e_sep is None else torch.from_numpy(data.e_sep).to(device)
    logits = logits_of(model, x, e)
    acc, n_dec = decisive_accuracy(logits.cpu().numpy(), data.y)
    return {"n": len(data), "bce_w": weighted_bce(logits, y, w), "bce": weighted_bce(logits, y),
            "acc": acc, "n_decisive": n_dec}


# ---------------------------------------------------------------------------
# Export + verification
# ---------------------------------------------------------------------------


def export_pev(model: nn.Module, out: Path, feature_version: int = 1) -> None:
    """Write the model with ``poke_engine.write_mlp`` / ``write_linear``.

    Weight layout is torch's: ``W1`` is ``[hidden][n_in]`` row-major, which is
    what learned.rs reads. ``feature_version`` is recorded in the file header.
    """
    import poke_engine

    kw = {} if feature_version == 1 else {"feature_version": feature_version}
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    stock_coef = 0.0
    if isinstance(model, ResidualNet):
        stock_coef = model.stock_coef
        if stock_coef == 0.0:
            raise ValueError("residual model with stock_coef 0 cannot be exported as residual")
        model = model.net
    if isinstance(model, nn.Linear):
        poke_engine.write_linear(
            str(out), model.weight.detach().reshape(-1).tolist(), float(model.bias.item()),
            stock_coef=stock_coef, **kw,
        )
        return
    l1, l2 = model[0], model[-1]
    poke_engine.write_mlp(
        str(out),
        l1.weight.detach().reshape(-1).tolist(),
        l1.bias.detach().tolist(),
        l2.weight.detach().reshape(-1).tolist(),
        float(l2.bias.item()),
        stock_coef=stock_coef, **kw,
    )


def verify_export(model: nn.Module, path: Path, states: list[str], tol: float = 1e-3) -> float:
    """Load ``path`` with ``Evaluator.load`` and compare ``evaluate`` to the torch logits.

    Returns the max absolute difference over ``states`` (raises if above ``tol``).
    """
    import poke_engine

    evaluator = poke_engine.Evaluator.load(str(path))
    model = model.to("cpu")
    parsed = [poke_engine.State.from_string(s) for s in states]
    version = getattr(evaluator, "feature_version", 1)
    rows = [
        poke_engine.state_features(s, 1) if version == 1
        else poke_engine.state_features(s, 1, version)
        for s in parsed
    ]
    if isinstance(model, ResidualNet):
        rows = [r + [poke_engine.evaluate(s)] for r, s in zip(rows, parsed)]
        if not evaluator.residual or abs(evaluator.stock_coef - model.stock_coef) > 1e-7:
            raise RuntimeError("exported evaluator lost its residual stock_coef")
    feats = torch.tensor(rows, dtype=torch.float32)
    expected = logits_of(model, feats).numpy()
    got = np.array([evaluator.evaluate(s) for s in parsed])
    worst = float(np.abs(expected - got).max())
    if worst > tol:
        raise RuntimeError(f"exported evaluator differs from torch by {worst:.2e} (> {tol})")
    return worst


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def q_z_correlation(data: Data) -> tuple[float, int]:
    """Pearson correlation of search value q and outcome z over positions with a q."""
    if data.q is None:
        return float("nan"), 0
    m = ~np.isnan(data.q)
    if m.sum() < 2 or data.q[m].std() == 0 or data.y[m].std() == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(data.q[m], data.y[m])[0, 1]), int(m.sum())


def stock_eval_logistic_bce(
    train: Data, val: Data, scale: float = dataset.Q_EVAL_SCALE
) -> tuple[float, float, float]:
    """The "stock eval as predictor" reference on the pure outcome z.

    Fits ``P = sigmoid(a * scale * eval + b)`` on TRAIN (weighted, same Newton fit as
    the residual net's initialisation, ``fit_stock_logistic``) against the pure outcome
    ``train.y``, then scores it on VAL with the val weights: the same samples, weights
    and label as the net's ``val BCE`` line, so the two are directly comparable. It
    equals the "stock-only logistic (fitted on train)" val BCE printed in residual mode
    (which fits ``y_fit`` and scores ``y``; they differ only when the training target
    is not the outcome). Returns ``(val weighted BCE, a, b)``; NaN when evaluations
    are unavailable.

    A net can legitimately score worse than this on outcome BCE without being weaker
    in play: nets trained on search-value targets fit q, not z, and the val split is a
    few dozen battles (noisy). Treat it as a reference point, not a pass/fail bar.
    """
    if train.stock_eval is None or val.stock_eval is None:
        return float("nan"), float("nan"), float("nan")
    ratio = scale / STOCK_SCALE  # fit_stock_logistic works in STOCK_SCALE units
    lim = 30 / STOCK_SCALE
    ev_t = np.clip(ratio * train.stock_eval.astype(np.float64), -lim, lim)
    ev_v = np.clip(ratio * val.stock_eval.astype(np.float64), -lim, lim)
    a, b = fit_stock_logistic(ev_t, train.y, train.w)
    logits = torch.from_numpy((STOCK_SCALE * a * ev_v + b).astype(np.float32))
    bce = weighted_bce(logits, torch.from_numpy(val.y), torch.from_numpy(val.w))
    return bce, a, b


def report(model: nn.Module, train: Data, val: Data, device: torch.device) -> dict[str, float]:
    model.to(device)
    t = {k: torch.from_numpy(getattr(train, k)).to(device) for k in ("x", "y", "w")}
    v = {k: torch.from_numpy(getattr(val, k)).to(device) for k in ("x", "y", "w")}
    et = None if train.e_sep is None else torch.from_numpy(train.e_sep).to(device)
    ev = None if val.e_sep is None else torch.from_numpy(val.e_sep).to(device)
    lt, lv = logits_of(model, t["x"], et), logits_of(model, v["x"], ev)
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
    soft = train.target is not None
    if soft:
        out["train_target_bce_w"] = weighted_bce(lt, torch.from_numpy(train.y_fit).to(device), t["w"])
        out["val_target_bce_w"] = weighted_bce(lv, torch.from_numpy(val.y_fit).to(device), v["w"])
        out["base_val_target_bce_w"] = constant_baseline_bce(
            val.y_fit, val.w, train.y_fit, train.w)
    out["stock_logistic_val_bce_w"], slope, icpt = stock_eval_logistic_bce(train, val)
    out["val_qz_corr"], n_q = q_z_correlation(val)
    lv_np = lv.cpu().numpy()
    out["val_acc"], n_dec = decisive_accuracy(lv_np, val.y)
    print("\n=== metrics (best epoch); BCE below is against the pure game outcome z ===")
    print(f"train BCE   weighted {out['train_bce_w']:.4f}   unweighted {out['train_bce']:.4f}")
    print(f"val BCE     weighted {out['val_bce_w']:.4f}   unweighted {out['val_bce']:.4f}")
    print(f"constant-predictor val BCE (train mean)  weighted {out['base_val_bce_w']:.4f}   "
          f"unweighted {out['base_val_bce']:.4f}   (ln2 = {math.log(2):.4f})")
    print(f"val BCE improvement over constant: "
          f"weighted {out['base_val_bce_w'] - out['val_bce_w']:+.4f}  "
          f"unweighted {out['base_val_bce'] - out['val_bce']:+.4f}")
    print(f"stock-eval-only logistic, fitted on TRAIN, scored on VAL (same samples/weights/"
          f"label z as the net's val BCE) weighted {out['stock_logistic_val_bce_w']:.4f}  "
          f"(P = sigmoid({slope:.3f}*0.0125*eval + {icpt:.3f}); a reference, not a pass/fail "
          f"bar: nets fit to search targets can score worse on z yet play stronger)")
    print(f"val accuracy on decisive positions: {out['val_acc']:.3f}  ({n_dec} positions)")
    if soft:
        print(f"training-target BCE  train weighted {out['train_target_bce_w']:.4f}   "
              f"val weighted {out['val_target_bce_w']:.4f}   (constant predictor "
              f"{out['base_val_target_bce_w']:.4f}; soft targets have entropy > 0, so "
              f"compare to the constant, not 0)")
    print(f"val q-z correlation: {out['val_qz_corr']:.3f}  ({n_q} positions with a search value)")
    print("calibration on val (predicted P(win) bucket: n, mean predicted, observed outcome)")
    for lo, hi, n, mp, mo in calibration_buckets(lv_np, val.y):
        if n:
            print(f"  [{lo:.1f},{hi:.1f})  n={n:7d}  pred={mp:.3f}  obs={mo:.3f}")
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", type=parse_source, action="append", required=True,
                   metavar="PATH[@K][:wW]",
                   help="run directory (repeatable); optional @K = at most K decisions per game "
                        "record from it, :wW = weight multiplier, e.g. data/human/pc1/train@3:w0.5")
    p.add_argument("--max-per-game", type=int, default=0, metavar="K",
                   help="default decisions kept per game record, uniformly at random (seeded by "
                        "--seed and the record), before feature extraction; 0 = all (default); "
                        "a per-source @K overrides it")
    p.add_argument("--load-workers", type=int, default=0, metavar="N",
                   help="processes for loading game files; 0 = cpu_count-2 (default), 1 = serial")
    p.add_argument("--val-data", type=parse_source, action="append", default=[],
                   metavar="PATH[@K][:wW]",
                   help="extra held-out set (repeatable), never trained on: BCE vs the outcome "
                        "reported each epoch and at the end")
    p.add_argument("--early-stop-on", choices=("val", "val-data"), default="val",
                   help="metric for early stopping / best epoch: the battle-split val set "
                        "(default) or the first --val-data set")
    p.add_argument("--model", choices=("mlp", "linear"), default="mlp")
    p.add_argument("--hidden", type=int, default=64, help="MLP hidden units")
    p.add_argument("--out", type=Path, required=True, help="output .pev path")
    p.add_argument("--states", choices=STATE_CHOICES, default="both")
    p.add_argument("--weighting", choices=WEIGHTINGS, default="sqrt")
    p.add_argument("--value-target", choices=VALUE_TARGETS, default="outcome",
                   help="outcome z (default), search value q, or blend (1-L)*z + L*q")
    p.add_argument("--blend-lambda", type=float, default=0.5,
                   help="weight L of q in the blend target (default 0.5)")
    p.add_argument("--td-lambda", type=float, default=0.7,
                   help="td target: lambda of the lambda-return over later decisions (default 0.7)")
    p.add_argument("--td-n", type=int, default=1, help="nstep target: n (default 1)")
    p.add_argument("--td-include-self", action="store_true",
                   help="td/nstep: blend the decision's own q_t: (1-mu)*G + mu*q_t")
    p.add_argument("--td-mu", type=float, default=0.0,
                   help="weight mu of the own q_t with --td-include-self (default 0)")
    p.add_argument("--q-kind", choices=dataset.Q_KINDS, default="absolute",
                   help="q = root-relative avg_score, or absolute (adds back the root's stock eval)")
    p.add_argument("--q-eval-scale", type=float, default=dataset.Q_EVAL_SCALE,
                   help="coefficient of the stock eval in the search score (default 0.0125)")
    p.add_argument("--q-source", choices=dataset.Q_SOURCES, default="decision",
                   help="q for world samples: decision mean (default) or the world's own score")
    p.add_argument("--residual", action="store_true",
                   help="learn a correction to the stock evaluation: z = a*0.0125*evaluate + net(x)")
    p.add_argument("--stock-coef-init", type=float, default=None,
                   help="residual: initial/fixed a (default: logistic fit on the train split)")
    p.add_argument("--freeze-stock-coef", action="store_true",
                   help="residual: keep a at its initial value")
    p.add_argument("--augment-bench", action="store_true",
                   help="randomly permute bench slots 1..5 of each side in training batches")
    p.add_argument("--augment-sides", action="store_true",
                   help="swap own/opp (side-two view) with p=1/2 in training batches, label 1-y")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="MLP dropout after the hidden ReLU (training only; export unaffected)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    p.add_argument("--features", choices=("v1", "v2"), default="v1",
                   help="engine feature layout: v1 (738 values) or v2 (v1 + item / ability / "
                        "move-property features, see docs/features-v2.md)")
    p.add_argument("--x-dtype", choices=("float32", "float16"), default=None,
                   help="in-memory dtype of the feature matrix (default float32 for v1, "
                        "float16 for v2: halves RAM / GPU memory)")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    started = time.perf_counter()
    for src in [*args.data, *args.val_data]:
        if not src.path.exists():
            print(f"no such path: {src.path}", file=sys.stderr)
            return 2
    if args.max_per_game < 0:
        print("--max-per-game must be >= 0", file=sys.stderr)
        return 2
    if args.early_stop_on == "val-data" and not args.val_data:
        print("--early-stop-on val-data needs --val-data", file=sys.stderr)
        return 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}")
    if not 0.0 <= args.blend_lambda <= 1.0:
        print("--blend-lambda must be in [0, 1]", file=sys.stderr)
        return 2
    if not 0.0 <= args.td_lambda <= 1.0 or not 0.0 <= args.td_mu <= 1.0 or args.td_n < 1:
        print("--td-lambda/--td-mu must be in [0, 1] and --td-n >= 1", file=sys.stderr)
        return 2
    mu = args.td_mu if args.td_include_self else 0.0
    target_desc = args.value_target + (
        f" (lambda={args.blend_lambda}, q-kind={args.q_kind}, q-source={args.q_source}, "
        f"q-eval-scale={args.q_eval_scale})" if args.value_target == "blend"
        else f" (q-kind={args.q_kind}, q-source={args.q_source}, q-eval-scale={args.q_eval_scale})"
        if args.value_target == "search"
        else f" (lambda={args.td_lambda}, mu={mu}, q-kind={args.q_kind})"
        if args.value_target == "td"
        else f" (n={args.td_n}, mu={mu}, q-kind={args.q_kind})" if args.value_target == "nstep"
        else "")
    print(f"value target: {target_desc}")
    feature_version = 2 if args.features == "v2" else 1
    x_dtype = args.x_dtype or ("float16" if feature_version == 2 else "float32")
    if feature_version == 2 and (args.augment_bench or args.augment_sides):
        print("--augment-* supports only --features v1", file=sys.stderr)
        return 2
    print(f"features: {args.features}, x dtype {x_dtype}")
    data = load_data(
        args.data, args.states, args.weighting, args.value_target, args.blend_lambda, args.q_source,
        args.q_kind, args.q_eval_scale, residual=args.residual,
        td_lambda=args.td_lambda, td_n=args.td_n, td_mu=mu,
        feature_version=feature_version, x_dtype=x_dtype,
        max_per_game=args.max_per_game, seed=args.seed, workers=args.load_workers,
    )
    extra_val: list[tuple[str, Data]] = []
    for vsrc in args.val_data:
        print(f"--- val-data {vsrc.path}")
        extra_val.append((str(vsrc.path), load_data(
            [vsrc], args.states, args.weighting, "outcome", residual=args.residual,
            feature_version=feature_version, x_dtype=x_dtype,
            max_per_game=args.max_per_game, seed=args.seed, workers=args.load_workers,
        )))
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
    n_feat = data.x.shape[1] - (1 if args.residual and data.e_sep is None else 0)
    res_init = None
    if args.residual:
        a0, b0 = fit_stock_logistic(train.e, train.y_fit, train.w, args.stock_coef_init)
        res_init = (a0, b0)
        base_lv = torch.from_numpy((STOCK_SCALE * a0 * val.e + b0).astype(np.float32))
        base_lt = torch.from_numpy((STOCK_SCALE * a0 * train.e + b0).astype(np.float32))
        yv, wv = torch.from_numpy(val.y), torch.from_numpy(val.w)
        yt, wt = torch.from_numpy(train.y), torch.from_numpy(train.w)
        print(f"stock-only logistic (a={a0:.4f}, b={b0:.4f}, fitted on train"
              f"{'' if args.stock_coef_init is None else ', a fixed'}): "
              f"val BCE weighted {weighted_bce(base_lv, yv, wv):.4f}  "
              f"unweighted {weighted_bce(base_lv, yv):.4f}  "
              f"(train weighted {weighted_bce(base_lt, yt, wt):.4f})")
    if not 0.0 <= args.dropout < 1.0:
        print("--dropout must be in [0, 1)", file=sys.stderr)
        return 2
    model = build_model(args.model, n_feat, args.hidden, res_init, args.freeze_stock_coef,
                        args.dropout)
    augmenter = None
    if args.augment_bench or args.augment_sides:
        augmenter = augment.Augmenter(args.augment_bench, args.augment_sides, args.residual,
                                      args.seed, device)
    result = fit(
        model, train, val, device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        augmenter=augmenter, extra_val=extra_val, early_stop_on=args.early_stop_on,
    )
    print(f"best epoch {result.best_epoch} of {result.epochs_run}"
          f" (early stopping on {args.early_stop_on})")
    if args.residual:
        print(f"learned a = {float(result.model.a.item()):.4f}  "
              f"(stock_coef = {result.model.stock_coef:.6f})")
    report(result.model, train, val, device)
    for name, vd in extra_val:
        m = evaluate_extra(result.model, vd, device)
        print(f"val-data {name}: BCE vs outcome weighted {m['bce_w']:.4f}  unweighted "
              f"{m['bce']:.4f}  decisive accuracy {m['acc']:.3f}  ({m['n']} samples, "
              f"{m['n_decisive']} decisive)")
    export_pev(result.model, args.out, feature_version)
    rng = random.Random(args.seed)
    pool = [s for s in val.states if s]
    check = rng.sample(pool, min(100, len(pool)))
    worst = verify_export(result.model, args.out, check)
    print(f"\nexported {args.out}  ({args.out.stat().st_size} bytes); verified on {len(check)} "
          f"val states, max |evaluate - torch logit| = {worst:.2e}")
    print(f"value target: {target_desc}")
    print(f"total time {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
