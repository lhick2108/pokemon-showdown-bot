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
    (0.0125) is only the anchor for stock-evaluator recordings; for a learned-evaluator
    recording (header config ``evaluator``) the root logit comes from that evaluator and
    the relative part is divided by its leaf scale (exact; ``--q-evaluator`` /
    ``--q-leaf-scale`` override the header). ``--q-source decision|world`` picks the
    decision's mean over worlds (default) or each world's own score. Metrics are
    reported against the training target and against the pure outcome z.

Interior-node targets (``--nodes on|only``, bead showdown-bot-rnm)
    Recordings made with ``scripts/selfplay.py --export-nodes K`` carry, per searched world, up
    to K sampled interior MCTS nodes (state, visits, absolute search value; see
    docs/selfplay-data.md). ``--nodes on`` adds one training sample per node next to the root
    samples, ``only`` trains on the nodes alone; their target is the node's search value (so it
    needs ``--value-target search|blend`` and ``--q-kind absolute``), blended with the outcome
    exactly like a root q. A node's loss weight is ``--node-weight`` times
    ``min(1, visits / --node-visit-ref)`` (a node with few visits is mostly its own leaf
    evaluation, i.e. what the net already believes: down-weighted, and nodes below
    ``--node-min-visits`` are dropped) times ``--node-depth-discount ** (depth - 1)``. Node
    samples belong to their battle for the train/val split. ``td``/``nstep`` do not combine
    with nodes. Root samples with exported values use the exact absolute root value of the
    search instead of the stock-eval reconstruction.

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

Feature version 4 and identity embeddings (``--features v4``, bead showdown-bot-589.65)
    The row is the dense v2 ++ v3 features (2379 values) followed by the 84 identity indices
    (species, item, ability, 4 moves for each of the 12 Pokemon slots, see
    ``docs/features-v4.md``) stored as numbers in the same matrix. ``EmbedMLP`` looks each index
    up in a learned table (``--embed-dim`` 8 or ``S,I,A,M``; the 4 moves are mean-pooled) and
    feeds ``[dense, concatenated slot embeddings]`` to ``Linear(., --hidden) -> ReLU ->
    Linear(hidden, 1)``. ``--residual``, ``--init``, ``--dropout`` and every ``--value-target``
    work as for v1-v3; ``--augment-*`` and ``--aux`` do not. The export is a format-3 ``.pev``
    (engine ``Evaluator.load``) plus ``<out>.vocab.json`` recording the vocabulary fingerprint.
    Human pretrain, then self-play fine-tune: see docs/training.md.

Device: cuda when available, else cpu (``--device`` overrides).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from showdown_bot import augment, auxtargets as aux, dataset, recorder

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
    # auxiliary targets (--aux on): (N, aux.N_AUX) targets and 0/1 masks, see auxtargets.py
    aux_y: np.ndarray | None = None
    aux_m: np.ndarray | None = None
    didx: np.ndarray | None = None  # decision index within its game record (turn - 1 for human data)

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

    def split_inplace(self, tr_idx: np.ndarray, va_idx: np.ndarray) -> tuple[Data, Data]:
        """(train, val) without a second full copy of ``x``.

        Val rows are copied out (small); the train rows are compacted to the front of this
        matrix in chunks (``tr_idx`` ascending, so every source row is at or after its
        destination and is read before it can be overwritten) and ``train.x`` is a view of
        that buffer. Peak is ~1x ``x`` + val; ``self.x`` must not be used afterwards.
        """
        tr_idx = np.asarray(tr_idx)
        assert len(tr_idx) < 2 or bool(np.all(np.diff(tr_idx) > 0)), "tr_idx must be ascending"
        val = self.subset(va_idx)
        step = 65536
        for a in range(0, len(tr_idx), step):
            src = self.x[tr_idx[a:a + step]]  # fancy read is a copy
            self.x[a:a + len(src)] = src
        return self.subset(tr_idx, x=self.x[:len(tr_idx)]), val

    def subset(self, idx: np.ndarray, x: np.ndarray | None = None) -> Data:
        w = self.w[idx]
        mean = float(w.mean()) if len(w) else 1.0
        return Data(
            self.x[idx] if x is None else x, self.y[idx], (w / mean).astype(np.float32),
            [self.tags[i] for i in idx], [self.states[i] for i in idx],
            None if self.q is None else self.q[idx],
            None if self.target is None else self.target[idx],
            None if self.stock_eval is None else self.stock_eval[idx],
            self.residual,
            None if self.e_sep is None else self.e_sep[idx],
            self.feature_version,
            None if self.aux_y is None else self.aux_y[idx],
            None if self.aux_m is None else self.aux_m[idx],
            None if self.didx is None else self.didx[idx],
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


def node_multiplier(sample, node_weight: float = 1.0, visit_ref: float = 64.0,
                    depth_discount: float = 1.0) -> float:
    """Loss-weight multiplier of one sample: 1 for root (world / truth) samples; for an interior
    node ``node_weight * min(1, visits / visit_ref) * depth_discount ** (depth - 1)``."""
    if sample.kind != dataset.KIND_NODE:
        return 1.0
    reach = min(1.0, sample.node_visits / visit_ref) if visit_ref > 0 else 1.0
    return float(node_weight * reach * depth_discount ** max(0, sample.node_depth - 1))


LOAD_CHUNK_FILES = 200  # game files per load job (the unit of spilling and of parallelism)
# Only about 1 in STATE_KEEP_EVERY samples keeps its State string (``Data.states``): they are
# ~3 KB each (more than the feature row itself) and only used to pick export-check states.
STATE_KEEP_EVERY = 50


def keep_state(game_id: str, decision_idx: int, every: int = STATE_KEEP_EVERY) -> bool:
    """Deterministic (hash of game id + decision index) choice of samples that keep their state."""
    return zlib.crc32(f"{game_id}:{decision_idx}".encode()) % every == 0


def _load_chunk(job: tuple) -> dict:
    """Load one chunk of game files (top level for spawn).

    The feature matrix is spilled to ``<spill_dir>/chunk-<index>.npy`` (only its path comes
    back), so the parent can assemble the final matrix without holding every chunk and the
    result in memory at once. Serial and parallel loading both go through here.
    """
    (index, files, max_per_game, kw, spill_dir) = job
    stats = dataset.LoadStats()
    states, q_kind = kw["states"], kw["q_kind"]
    feats, ys, qs, evals, dqs, didx, wm, nd = [], [], [], [], [], [], [], []
    tags, strings, gids = [], [], []
    seen = set()
    ay, am = [], []
    fin_cache: tuple[str, object] = ("", None)  # (game_id, parsed final state)
    for s in dataset.iter_samples(
        files[0], sides=(1,), max_per_game=max_per_game, seed=kw["seed"],
        nodes=kw["nodes"], node_min_visits=kw["node_min_visits"], nodes_per_world=kw["nodes_per_world"],
        worlds="fallback" if states == "prefer-truth" else states in ("both", "worlds"),
        truth=states in ("both", "truth", "prefer-truth"), stats=stats, q_source=kw["q_source"],
        q_eval_scale=kw["q_eval_scale"], feature_version=kw["feature_version"], files=files,
        q_evaluator=kw["q_evaluator"], q_leaf_scale=kw["q_leaf_scale"],
    ):
        feats.append(np.asarray(s.features, dtype=np.dtype(kw["x_dtype"])))
        ys.append(s.value)
        q = s.search_value_abs if q_kind == "absolute" else s.search_value
        qs.append(np.nan if q is None else q)
        evals.append(s.stock_eval)
        tags.append(sys.intern(s.battle_tag))
        strings.append(s.state if keep_state(s.game_id, s.decision_idx, kw["keep_every"]) else None)
        gids.append(sys.intern(s.game_id))
        didx.append(s.decision_idx)
        seen.add((s.game_id, s.decision_idx))
        dq = s.decision_q_abs if q_kind == "absolute" else s.decision_q
        dqs.append(np.nan if dq is None else dq)
        nd.append(s.kind == dataset.KIND_NODE)
        if kw["aux"]:
            import poke_engine
            if fin_cache[0] != s.game_id:
                fin_cache = (s.game_id, poke_engine.State.from_string(s.final_state)
                             if s.final_state else None)
            if fin_cache[1] is None:
                y_a, m_a = np.zeros(aux.N_AUX, np.float32), np.zeros(aux.N_AUX, np.float32)
            else:
                y_a, m_a = aux.aux_targets(poke_engine.State.from_string(s.state), fin_cache[1], s.result)
            ay.append(y_a)
            am.append(m_a)
        wm.append(node_multiplier(s, kw["node_weight"], kw["node_visit_ref"], kw["node_depth_discount"]))
    x_path = None
    if feats:
        x_path = os.path.join(spill_dir, f"chunk-{index:07d}.npy")
        np.save(x_path, np.stack(feats))
    del feats
    return {
        "x_path": x_path, "rows": len(ys),
        "ys": np.asarray(ys, dtype=np.float32), "qs": np.asarray(qs, dtype=np.float32),
        "evals": np.asarray(evals, dtype=np.float32), "dqs": np.asarray(dqs, dtype=np.float64),
        "didx": np.asarray(didx, dtype=np.int64), "wm": np.asarray(wm, dtype=np.float64),
        "nd": np.asarray(nd, dtype=bool),
        "ay": np.asarray(ay, dtype=np.float32).reshape(-1, aux.N_AUX),
        "am": np.asarray(am, dtype=np.float32).reshape(-1, aux.N_AUX),
        "tags": tags, "strings": strings, "gids": gids, "decisions": len(seen),
        "stats": {"games": stats.games, "skipped_games": stats.skipped_games,
                  "bad_states": stats.bad_states, "results": dict(stats.results)},
    }


def _load_source(path: Path, k: int, workers: int, kw: dict, spill_dir: str, job0: int) -> dict:
    """Load one source in chunks of ``LOAD_CHUNK_FILES`` files, in order (``workers`` > 1: pool)."""
    chunk = LOAD_CHUNK_FILES
    files = list(recorder.iter_game_files(path))
    jobs = [(job0 + n, files[i:i + chunk], k, kw, spill_dir)
            for n, i in enumerate(range(0, len(files), chunk))]
    out: dict = {n: [] for n in ("ys", "qs", "evals", "dqs", "didx", "wm", "nd", "ay", "am", "tags", "strings", "gids")}
    out["x_parts"] = []  # (npy path, rows) in order
    out["decisions"] = 0
    out["jobs"] = len(jobs)
    out["stats"] = {"games": 0, "skipped_games": 0, "bad_states": 0, "results": Counter()}

    def absorb(r: dict) -> None:
        if r["x_path"] is not None:
            out["x_parts"].append((r["x_path"], r["rows"]))
        for n in ("ys", "qs", "evals", "dqs", "didx", "wm", "nd", "ay", "am"):
            out[n].append(r[n])
        for n in ("tags", "strings", "gids"):
            out[n] += r[n]
        out["decisions"] += r["decisions"]
        for n in ("games", "skipped_games", "bad_states"):
            out["stats"][n] += r["stats"][n]
        out["stats"]["results"].update(r["stats"]["results"])

    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            absorb(_load_chunk(job))
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
    nodes: str = "off",
    node_min_visits: int = 16,
    node_weight: float = 1.0,
    node_visit_ref: float = 64.0,
    node_depth_discount: float = 1.0,
    nodes_per_world: int = 0,
    aux_targets: bool = False,
    q_evaluator: str | None = None,
    q_leaf_scale: float | None = None,
) -> Data:
    """Load the samples of the given run directories (``Source`` / path / spec) into memory.

    ``max_per_game`` is the default cap on decisions per game record (= per ``.jsonl.gz``
    file; 0 = all); a ``Source.max_per_game`` overrides it, and ``Source.weight`` multiplies
    the sample weights. Prints per-source counts and the feature-matrix size.

    ``residual`` appends the raw stock ``evaluate(state)`` as the last column of ``x``
    (with ``x_dtype="float16"`` it is kept in a separate float32 array instead).
    ``feature_version`` selects the engine feature layout (1, 2 or 3); float16 halves the memory
    of the (N, n_in) matrix, which matters for the 2.1k-wide version 2.

    ``workers``: 0 = ``os.cpu_count() - 2`` (min 1), 1 = serial, >1 = that many processes
    each loading a chunk of game files (results are identical to serial: the per-game
    subsampling RNG depends only on ``seed`` and the file name, and chunks are joined in order).

    Memory: each chunk's feature rows are spilled to a temporary ``.npy`` and copied into the
    final preallocated matrix, so the peak is about one copy of ``x`` (not two). ``Data.states``
    keeps a State string for only ~1 in ``STATE_KEEP_EVERY`` samples (None elsewhere).
    """
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - 2)
    np_dtype = np.dtype(x_dtype)
    tags, strings, gids = [], [], []
    cols: dict[str, list] = {n: [] for n in ("ys", "qs", "evals", "dqs", "didx", "wm", "nd", "ay", "am")}
    if nodes not in dataset.NODE_MODES:
        raise ValueError(f"nodes must be one of {dataset.NODE_MODES}")
    if nodes != "off" and (value_target not in ("search", "blend") or q_kind != "absolute"):
        raise ValueError("interior nodes are search-value targets: use --value-target search|blend "
                         "with --q-kind absolute")
    load = dataset.LoadStats()
    sources = [parse_source(p) for p in paths]
    src_records, src_decisions, src_rows = [], [], []
    x_parts: list[tuple[str, int]] = []
    kw = dict(states=states, q_source=q_source, q_kind=q_kind, q_eval_scale=q_eval_scale,
              feature_version=feature_version, x_dtype=x_dtype, seed=seed, keep_every=STATE_KEEP_EVERY,
              nodes=nodes, node_min_visits=node_min_visits, node_weight=node_weight,
              node_visit_ref=node_visit_ref, node_depth_discount=node_depth_discount,
              nodes_per_world=nodes_per_world, aux=aux_targets,
              q_evaluator=q_evaluator, q_leaf_scale=q_leaf_scale)
    with tempfile.TemporaryDirectory(prefix="pev-load-") as spill:
        job0 = 0
        for source in sources:
            k = max_per_game if source.max_per_game is None else source.max_per_game
            r = _load_source(source.path, k, workers, kw, spill, job0)
            job0 += r["jobs"]
            x_parts += r.pop("x_parts")
            for n in cols:
                cols[n] += r[n]
            tags += r["tags"]; strings += r["strings"]; gids += r["gids"]
            src_rows.append(len(r["tags"]))
            load.games += r["stats"]["games"]
            load.skipped_games += r["stats"]["skipped_games"]
            load.bad_states += r["stats"]["bad_states"]
            load.results.update(r["stats"]["results"])
            src_records.append(r["stats"]["games"])
            src_decisions.append(r["decisions"])
        if load.bad_states:
            print(f"warning: {load.bad_states} unparsable states skipped", file=sys.stderr)
        if not x_parts:
            raise SystemExit("no samples found in " + ", ".join(map(str, paths)))
        n_rows = sum(rows for _, rows in x_parts)
        n_in = np.load(x_parts[0][0], mmap_mode="r").shape[1]
        y, q, evals, dqs, didx, wmult, is_node = (
            np.concatenate(cols[n]) for n in ("ys", "qs", "evals", "dqs", "didx", "wm", "nd"))
        aux_y, aux_m = (np.concatenate(cols[n]) for n in ("ay", "am"))
        del cols
        extra = 1 if residual and np_dtype == np.float32 else 0
        x = np.empty((n_rows, n_in + extra), dtype=np_dtype)
        off = 0
        for p, rows in x_parts:  # free each spilled chunk as it is copied
            x[off:off + rows, :n_in] = np.load(p)
            os.remove(p)
            off += rows
    if extra:
        x[:, n_in] = evals
    e_sep = np.asarray(evals, dtype=np.float32) if residual and not extra else None
    src = np.repeat(np.arange(len(sources), dtype=np.int64), src_rows)
    for si, source in enumerate(sources):
        k = max_per_game if source.max_per_game is None else source.max_per_game
        gb = src_rows[si] * x.shape[1] * x.itemsize / 1e9
        print(f"source {source.path}: max-per-game {k or 'all'}, weight x{source.weight:g}: "
              f"{src_records[si]} records, {src_decisions[si]} decisions kept, "
              f"{src_rows[si]} samples, x {gb:.2f} GB")
    kept = sum(1 for s_ in strings if s_ is not None)
    print(f"loaded {len(y)} samples: x {tuple(x.shape)} {x.dtype} = {x.nbytes / 1e9:.2f} GB; "
          f"{kept} state strings kept for export checks")
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
    mult = np.asarray([s_.weight for s_ in sources], dtype=np.float64)[src] * wmult
    if nodes != "off":
        print(f"interior nodes ({nodes}): {int(is_node.sum())} node samples of {len(y)} "
              f"(min visits {node_min_visits}, weight {node_weight:g}, visit ref {node_visit_ref:g}, "
              f"depth discount {node_depth_discount:g}); mean node multiplier "
              f"{float(wmult[is_node].mean()) if is_node.any() else 0.0:.3f}")
    return Data(
        x, y, sample_weights(tags, weighting, mult), tags, strings, q,
        None if value_target == "outcome" else target,
        np.asarray(evals, dtype=np.float32), residual, e_sep, feature_version,
        aux_y if aux_targets else None, aux_m if aux_targets else None, didx,
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


V4_DENSE = 2379  # dense width of feature version 4 (poke_engine.FEATURE_LENGTH_V4)
V5_DENSE = 2191  # dense width of feature version 5 = v4 without the 188 matchup columns
N_SLOTS = 12  # identity slots: own 0..5, opp 0..5
IDS_PER_SLOT = 7  # species, item, ability, move0..3
N_IDS = N_SLOTS * IDS_PER_SLOT
EMBED_TABLES = ("species", "item", "ability", "move")


def parse_embed_dims(spec: str | int | tuple | list) -> tuple[int, int, int, int]:
    """``8`` -> (8, 8, 8, 8); ``"8"`` likewise; ``"16,8,8,8"`` -> species, item, ability, move."""
    if isinstance(spec, int):
        parts = [spec]
    elif isinstance(spec, (tuple, list)):
        parts = [int(v) for v in spec]
    else:
        parts = [int(v) for v in str(spec).split(",")]
    if len(parts) == 1:
        parts = parts * 4
    if len(parts) != 4 or any(v < 1 or v > 1024 for v in parts):
        raise ValueError(f"--embed-dim needs 1 or 4 integers in 1..1024, got {spec!r}")
    return tuple(parts)  # type: ignore[return-value]


class EmbedMLP(nn.Module):
    """Feature version 4 value net: ``relu(l1([dense, slot embeddings])) -> out``.

    Input row: ``n_dense`` dense features followed by the ``N_IDS`` identity indices (floats).
    Per slot the species, item and ability embeddings and the MEAN of the four move embeddings
    are concatenated (slot-major, own 0..5 then opp 0..5) after the dense part: exactly the
    column order of the engine's first layer (``learned.rs``, ``Embed::fill_inputs``). Index 0
    = pad (fainted), 1 = unknown, else engine enum id + 2.
    """

    def __init__(self, n_dense: int, hidden: int, dims: tuple[int, ...], vocab: tuple[int, ...],
                 dropout: float = 0.0):
        super().__init__()
        self.n_dense = n_dense
        self.dims = tuple(dims)
        self.vocab = tuple(vocab)
        self.emb = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(vocab, dims)])
        for e in self.emb:
            nn.init.normal_(e.weight, std=0.1)
        self.l1 = nn.Linear(n_dense + N_SLOTS * sum(dims), hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden, 1)

    def embed_inputs(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.round().long().view(-1, N_SLOTS, IDS_PER_SLOT)
        sp, it, ab = (self.emb[t](ids[:, :, t]) for t in range(3))
        mv = self.emb[3](ids[:, :, 3:]).mean(dim=2)
        return torch.cat([sp, it, ab, mv], dim=2).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dense, ids = x[:, :self.n_dense], x[:, self.n_dense:self.n_dense + N_IDS]
        h = torch.relu(self.l1(torch.cat([dense, self.embed_inputs(ids)], dim=1)))
        return self.out(self.drop(h))


def build_model(
    kind: str, n_in: int, hidden: int, residual: tuple[float, float] | None = None,
    freeze_a: bool = False, dropout: float = 0.0,
    embed_dims: tuple[int, ...] | None = None,
) -> nn.Module:
    """``n_in`` counts feature columns only (with ``embed_dims``: dense + the ``N_IDS`` identity
    columns). ``residual=(a, b)`` wraps the net in a ResidualNet. ``embed_dims`` (species, item,
    ability, move) builds the version-4 ``EmbedMLP``."""
    if embed_dims is not None:
        if kind != "mlp":
            raise ValueError("identity embeddings need --model mlp")
        import poke_engine

        net: nn.Module = EmbedMLP(n_in - N_IDS, hidden, embed_dims,
                                  tuple(poke_engine.identity_vocab_sizes()), dropout)
    elif kind == "linear":
        net = nn.Linear(n_in, 1)
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


def add_aux_head(model: nn.Module, hidden: int) -> None:
    """Attach the auxiliary head (12 survival logits + 1 remaining-difference output) that reads
    the SAME hidden layer as the value head. Training only: ``export_pev`` never sees it."""
    model.aux_head = nn.Linear(hidden, aux.N_AUX)


def forward_with_aux(model: nn.Module, xe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(value logit (B,), aux outputs (B, N_AUX))`` for an mlp (plain or ResidualNet) model."""
    res = isinstance(model, ResidualNet)
    net = model.net if res else model
    h = net[:-1](xe[:, :-1] if res else xe)
    z = net[-1](h)
    if res:
        z = z + STOCK_SCALE * model.a * xe[:, -1:]
    return z.squeeze(-1), model.aux_head(h)


def aux_loss(out: torch.Tensor, y: torch.Tensor, m: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Survival BCE (masked, per-sample mean over its valid slots) + remaining-difference MSE
    (tanh output vs the [-1, 1] target), weighted by the sample weights ``w``."""
    ns = 2 * aux.N_SLOTS
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, :ns], y[:, :ns], reduction="none")
    surv = (bce * m[:, :ns]).sum(1) / m[:, :ns].sum(1).clamp(min=1)
    has = (m[:, :ns].sum(1) > 0).float()
    surv = (surv * w * has).sum() / (w * has).sum().clamp(min=1e-9)
    diff = ((torch.tanh(out[:, ns]) - y[:, ns]) ** 2 * m[:, ns] * w).sum() / (m[:, ns] * w).sum().clamp(min=1e-9)
    return surv + diff


STOCK_SCALE = 0.0125  # the stock search scale: P = sigmoid(0.0125 * evaluate)


class ResidualNet(nn.Module):
    """``z = a * 0.0125 * e + net(x)``; input is ``[x, e]`` (e = raw stock evaluate)."""

    def __init__(self, net: nn.Module, a: float, intercept: float = 0.0, freeze_a: bool = False):
        super().__init__()
        self.net = net
        self.a = nn.Parameter(torch.tensor(float(a)), requires_grad=not freeze_a)
        # zero output layer + intercept: the untrained model is the stock logistic fit
        out = net if isinstance(net, nn.Linear) else net.out if isinstance(net, EmbedMLP) else net[-1]
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


GPU_X_LIMIT = 2e9  # bytes: a feature matrix larger than this stays on the CPU (batched transfer)


def place_x(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Feature matrix as a tensor: on ``device`` if small, else left on the CPU."""
    t = torch.from_numpy(arr)
    return t.to(device) if arr.nbytes <= GPU_X_LIMIT else t


def logits_of(
    model: nn.Module, x: torch.Tensor, e: torch.Tensor | None = None, batch: int = 8192
) -> torch.Tensor:
    """Logits for features ``x`` (may live on the CPU: each batch is moved to the model's
    device); ``e`` is the separately stored stock eval (float16 data)."""
    model.eval()
    if len(x) == 0:
        return torch.zeros(0)
    dev = next(model.parameters()).device
    with torch.no_grad():
        return torch.cat([
            model(model_input(
                x[i:i + batch].to(dev, non_blocking=True),
                None if e is None else e[i:i + batch].to(dev),
            )).squeeze(-1)
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
    aux_weight: float = 0.0,
) -> FitResult:
    """AdamW on weighted BCE; early stopping on weighted val BCE (best weights restored).

    ``extra_val`` are named held-out sets reported each epoch (weighted BCE vs the
    outcome); they only drive early stopping with ``early_stop_on="val-data"`` (the first
    one), otherwise they are just logged.
    """
    extra_val = extra_val or []
    use_aux = aux_weight > 0 and hasattr(model, "aux_head")
    if use_aux and (train.aux_y is None or augmenter is not None):
        raise ValueError("aux needs aux targets loaded and no --augment-*")
    if early_stop_on == "val-data" and not extra_val:
        raise ValueError("early_stop_on='val-data' needs extra_val")
    torch.manual_seed(seed)
    model.to(device)
    xt, xv = place_x(train.x, device), place_x(val.x, device)
    yt, wt = (torch.from_numpy(a).to(device) for a in (train.y_fit, train.w))
    yv, wv = (torch.from_numpy(a).to(device) for a in (val.y_fit, val.w))
    et = None if train.e_sep is None else torch.from_numpy(train.e_sep).to(device)
    ev = None if val.e_sep is None else torch.from_numpy(val.e_sep).to(device)
    if use_aux:
        ayt, amt = (torch.from_numpy(a).to(device) for a in (train.aux_y, train.aux_m))
    extra_t = [
        (name, place_x(d.x, device), *(torch.from_numpy(a).to(device) for a in (d.y, d.w)),
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
        perm = torch.randperm(n, generator=gen)  # CPU: rows of a CPU-resident x are gathered there
        perm_dev = perm.to(device)
        for i in range(0, n, batch_size):
            b = perm_dev[i:i + batch_size]
            xb = xt[perm[i:i + batch_size]] if xt.device.type == "cpu" and device.type != "cpu" else xt[b]
            xb = model_input(xb.to(device, non_blocking=True), None if et is None else et[b])
            yb = yt[b]
            if augmenter is not None:
                xb, yb = augmenter(xb, yb)
            if use_aux:
                zb, ab = forward_with_aux(model, xb)
            else:
                zb = model(xb).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(zb, yb, reduction="none")
            loss = (loss * wt[b]).sum() / wt[b].sum()
            if use_aux:
                loss = loss + aux_weight * aux_loss(ab, ayt[b], amt[b], wt[b])
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
    x = place_x(data.x, device)
    y, w = (torch.from_numpy(a).to(device) for a in (data.y, data.w))
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
    if isinstance(model, EmbedMLP):
        poke_engine.write_mlp_embed(
            str(out), model.l1.weight.detach().numpy(), model.l1.bias.detach().numpy(),
            model.out.weight.detach().numpy(), float(model.out.bias.item()),
            [e.weight.detach().numpy() for e in model.emb], model.dims, stock_coef=stock_coef,
            feature_version=feature_version,
        )
        write_vocab_record(out, model.dims, feature_version)
        return
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


def write_vocab_record(pev_path: Path, dims, feature_version: int = 4) -> Path:
    """``<pev>.vocab.json``: the vocabulary fingerprint, table sizes, embedding dims and the
    entry names (row order) the net was trained with, so a trained file can be audited against
    any engine build (the same hash is inside the ``.pev``)."""
    import json

    import poke_engine

    rec = {
        "feature_version": feature_version,
        "vocab_hash": f"{poke_engine.identity_vocab_hash():#018x}",
        "tables": list(EMBED_TABLES),
        "embed_dims": [int(d) for d in dims],
        "vocab_sizes": list(poke_engine.identity_vocab_sizes()),
        "vocab": {t: poke_engine.identity_vocab(i) for i, t in enumerate(EMBED_TABLES)},
    }
    out = Path(str(pev_path) + ".vocab.json")
    out.write_text(json.dumps(rec), encoding="utf-8")
    return out


def read_pev(path: str | Path) -> dict:
    """Parse a ``.pev`` weights file (format in ``engine/src/learned.rs``) into plain arrays.

    Returns ``feature_version``, ``kind`` ("linear" | "mlp"), ``n_in``, ``hidden``,
    ``stock_coef`` (None = not residual) and the torch-layout weights ``w1`` ``[hidden, n_in]``,
    ``b1``, ``w2`` ``[hidden]``, ``b2`` (linear: ``w1`` is ``[1, n_in]``, ``b1`` / ``w2`` unused).
    """
    import struct

    raw = Path(path).read_bytes()
    if len(raw) < 24 or raw[:4] != b"PEVN":
        raise ValueError(f"{path}: not a poke-engine evaluator file (bad magic)")
    fmt, feat_ver, kind, n_in, hidden = struct.unpack_from("<5I", raw, 4)
    off, stock_coef = 24, None
    if fmt == 3:
        return _read_pev_embed(path, raw, feat_ver, kind, n_in, hidden)
    if fmt == 2:
        flags, coef = struct.unpack_from("<If", raw, 24)
        stock_coef = float(coef) if flags & 1 else None
        off = 32
    elif fmt != 1:
        raise ValueError(f"{path}: unsupported weights format {fmt}")
    if kind not in (1, 2):
        raise ValueError(f"{path}: unknown model kind {kind}")
    body = np.frombuffer(raw, dtype="<f4", offset=off).astype(np.float32)
    need = n_in + 1 if kind == 1 else hidden * n_in + 2 * hidden + 1
    if len(body) != need:
        raise ValueError(f"{path}: weights body has {len(body)} floats, expected {need}")
    out = {"feature_version": feat_ver, "kind": "linear" if kind == 1 else "mlp", "n_in": n_in,
           "hidden": hidden, "stock_coef": stock_coef}
    if kind == 1:
        out.update(w1=body[:n_in].reshape(1, n_in), b1=None, w2=None, b2=float(body[n_in]))
    else:
        a = hidden * n_in
        out.update(w1=body[:a].reshape(hidden, n_in), b1=body[a:a + hidden],
                   w2=body[a + hidden:a + 2 * hidden], b2=float(body[a + 2 * hidden]))
    return out


def _read_pev_embed(path, raw: bytes, feat_ver: int, kind: int, n_dense: int, hidden: int) -> dict:
    """Format 3 (feature version 4 with identity embeddings), see ``engine/src/learned.rs``."""
    import struct

    if kind != 2 or feat_ver not in (4, 5):
        raise ValueError(f"{path}: format 3 needs an MLP of feature version 4 or 5")
    flags, coef = struct.unpack_from("<If", raw, 24)
    (vhash,) = struct.unpack_from("<Q", raw, 32)
    dims = struct.unpack_from("<4I", raw, 40)
    off = 56
    n_total = n_dense + N_SLOTS * sum(dims)
    body = np.frombuffer(raw, dtype="<f4", offset=off).astype(np.float32)
    a = hidden * n_total
    pos = a + 2 * hidden + 1
    tables = []
    # table row counts are not stored (the vocabulary hash pins them): use this engine's sizes
    import poke_engine

    sizes = poke_engine.identity_vocab_sizes()
    if sum(s * d for s, d in zip(sizes, dims)) != len(body) - pos:
        raise ValueError(f"{path}: embedding tables do not match this engine's vocabulary sizes")
    for s, d in zip(sizes, dims):
        tables.append(body[pos:pos + s * d].reshape(s, d))
        pos += s * d
    return {
        "feature_version": feat_ver, "kind": "mlp", "n_in": n_dense, "hidden": hidden,
        "stock_coef": float(coef) if flags & 1 else None,
        "w1": body[:a].reshape(hidden, n_total), "b1": body[a:a + hidden],
        "w2": body[a + hidden:a + 2 * hidden], "b2": float(body[a + 2 * hidden]),
        "embed_dims": tuple(dims), "tables": tables, "vocab_hash": vhash,
    }


def load_init(model: nn.Module, path: str | Path, *, model_kind: str, n_feat: int, hidden: int,
              feature_version: int, residual: bool) -> None:
    """Warm start: copy the weights of ``path`` into ``model`` (as built by ``build_model``).

    Raises ``ValueError`` naming the mismatch if the file's kind, feature version, input width,
    hidden size or residual-ness differ from the model being trained. For a residual model the
    stock coefficient ``a`` is set to ``stock_coef / 0.0125`` of the file (still trainable unless
    ``--freeze-stock-coef``) and the output bias is the file's, so the fit of ``a, b`` on this
    run's training split is NOT applied. The net has no input normalisation, so there are no
    other statistics to carry over.
    """
    pev = read_pev(path)
    problems = []
    if pev["kind"] != model_kind:
        problems.append(f"model kind {pev['kind']} (training {model_kind})")
    # partial warm start: a v4 net can seed a v5 net (v5 = v4 minus the matchup block)
    v4_to_v5 = pev["feature_version"] == 4 and feature_version == 5 and pev.get("embed_dims") is not None
    if pev["feature_version"] != feature_version and not v4_to_v5:
        problems.append(f"feature version {pev['feature_version']} (training {feature_version})")
    net_ = model.net if isinstance(model, ResidualNet) else model
    is_embed = isinstance(net_, EmbedMLP)
    if (pev.get("embed_dims") is not None) != is_embed:
        problems.append("file has identity embeddings but training without them" if is_embed is False
                        else "training with identity embeddings but the file has none")
    elif is_embed:
        import poke_engine

        if tuple(pev["embed_dims"]) != tuple(net_.dims):
            problems.append(f"embed dims {tuple(pev['embed_dims'])} (training {tuple(net_.dims)})")
        if pev["vocab_hash"] != poke_engine.identity_vocab_hash():
            problems.append("embedding vocabulary differs from this engine's")
    n_dense_file = V5_DENSE if v4_to_v5 else pev["n_in"]
    if v4_to_v5 and pev["n_in"] != V4_DENSE:
        problems.append(f"{pev['n_in']} dense inputs (a v4 net has {V4_DENSE})")
    if n_dense_file + (N_IDS if is_embed else 0) != n_feat:
        problems.append(f"{pev['n_in']} inputs (training {n_feat})")
    if model_kind == "mlp" and pev["hidden"] != hidden:
        problems.append(f"hidden {pev['hidden']} (training {hidden})")
    if (pev["stock_coef"] is not None) != residual:
        problems.append("file is residual but training without --residual" if pev["stock_coef"]
                        is not None else "file is not residual but training with --residual")
    if problems:
        raise ValueError(f"--init {path} does not match this run: " + "; ".join(problems))
    net = model.net if isinstance(model, ResidualNet) else model
    with torch.no_grad():
        if is_embed:
            w1 = torch.from_numpy(pev["w1"])
            if v4_to_v5:
                # drop the first-layer columns of the v3 matchup block (dense cols V5_DENSE..V4_DENSE)
                w1 = torch.cat([w1[:, :V5_DENSE], w1[:, V4_DENSE:]], dim=1)
            net.l1.weight.copy_(w1)
            net.l1.bias.copy_(torch.from_numpy(pev["b1"]))
            net.out.weight.copy_(torch.from_numpy(pev["w2"]).reshape(1, -1))
            net.out.bias.fill_(pev["b2"])
            for e, t in zip(net.emb, pev["tables"]):
                e.weight.copy_(torch.from_numpy(t))
        elif model_kind == "linear":
            net.weight.copy_(torch.from_numpy(pev["w1"]))
            net.bias.fill_(pev["b2"])
        else:
            l1, l2 = net[0], net[-1]
            l1.weight.copy_(torch.from_numpy(pev["w1"]))
            l1.bias.copy_(torch.from_numpy(pev["b1"]))
            l2.weight.copy_(torch.from_numpy(pev["w2"]).reshape(1, -1))
            l2.bias.fill_(pev["b2"])
        if isinstance(model, ResidualNet):
            model.a.fill_(pev["stock_coef"] / STOCK_SCALE)


def load_teacher(path: str | Path, *, n_feat: int, feature_version: int, residual: bool) -> nn.Module:
    """A ``.pev`` as a torch model (any hidden size) for ``--distill``."""
    pev = read_pev(path)
    if pev["kind"] != "mlp":
        raise ValueError(f"--distill {path}: teacher must be an MLP")
    dims = pev.get("embed_dims")
    teacher = build_model("mlp", n_feat, pev["hidden"], (1.0, 0.0) if residual else None,
                          embed_dims=tuple(dims) if dims is not None else None)
    load_init(teacher, path, model_kind="mlp", n_feat=n_feat, hidden=pev["hidden"],
              feature_version=feature_version, residual=residual)
    return teacher


def _data_e(d: Data) -> torch.Tensor | None:
    return None if d.e_sep is None else torch.from_numpy(d.e_sep)


def apply_distill(model: nn.Module, teacher_path: Path, train: Data, val: Data,
                  extra_val: list[tuple[str, Data]], *, n_feat: int, feature_version: int,
                  residual: bool, weight: float, init: str, device: torch.device) -> dict:
    """``--distill``: replace the train / val targets by ``weight * sigmoid(teacher) + (1 -
    weight) * label`` and optionally start the student from the teacher (``init``).

    Returns the original val labels and the teacher's val logits (for ``distill_report``)."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--distill-weight must be in [0, 1]")
    teacher = load_teacher(teacher_path, n_feat=n_feat, feature_version=feature_version,
                           residual=residual).to(device)
    out: dict = {"val_label": val.y_fit.copy()}
    for d, name in ((train, "train"), (val, "val")):
        t = logits_of(teacher, place_x(d.x, device), _data_e(d)).cpu().numpy().astype(np.float32)
        if name == "val":
            out["val_teacher"] = t
        p = 1.0 / (1.0 + np.exp(-t.astype(np.float64)))
        d.target = (weight * p + (1.0 - weight) * d.y_fit.astype(np.float64)).astype(np.float32)
    out["extra_teacher"] = []
    for _, d in extra_val:
        out["extra_teacher"].append(
            logits_of(teacher, place_x(d.x, device), _data_e(d)).cpu().numpy())
    t_net = teacher.net if isinstance(teacher, ResidualNet) else teacher
    s_net = model.net if isinstance(model, ResidualNet) else model
    if init != "none":
        if not isinstance(s_net, EmbedMLP) or not isinstance(t_net, EmbedMLP) \
                or tuple(s_net.dims) != tuple(t_net.dims):
            raise ValueError("--distill-init trunc|emb needs v4/v5 nets with the same --embed-dim")
        with torch.no_grad():
            for es, et in zip(s_net.emb, t_net.emb):
                es.weight.copy_(et.weight)
            if init == "trunc":
                h_s, h_t = s_net.l1.out_features, t_net.l1.out_features
                if h_s > h_t:
                    raise ValueError(f"--distill-init trunc needs --hidden <= the teacher's ({h_t})")
                idx = torch.argsort(t_net.out.weight[0].abs(), descending=True)[:h_s]
                s_net.l1.weight.copy_(t_net.l1.weight[idx])
                s_net.l1.bias.copy_(t_net.l1.bias[idx])
                s_net.out.weight.copy_(t_net.out.weight[:, idx])
                s_net.out.bias.copy_(t_net.out.bias)
                if isinstance(model, ResidualNet) and isinstance(teacher, ResidualNet):
                    model.a.fill_(float(teacher.a.item()))
    tv = out["val_teacher"]
    yv = torch.from_numpy(val.y)
    print(f"distill: teacher {teacher_path} (hidden {t_net.l1.out_features}), weight {weight}, "
          f"init {init}; teacher val BCE vs outcome weighted "
          f"{weighted_bce(torch.from_numpy(tv), yv, torch.from_numpy(val.w)):.4f} "
          f"unweighted {weighted_bce(torch.from_numpy(tv), yv):.4f}; teacher val BCE vs the "
          f"original label weighted {weighted_bce(torch.from_numpy(tv), torch.from_numpy(out['val_label']), torch.from_numpy(val.w)):.4f}")
    for (name, d), t in zip(extra_val, out["extra_teacher"]):
        print(f"distill: teacher val-data[{name}] BCE vs outcome weighted "
              f"{weighted_bce(torch.from_numpy(t), torch.from_numpy(d.y), torch.from_numpy(d.w)):.4f}")
    del teacher
    return out


def distill_report(model: nn.Module, val: Data, info: dict, device: torch.device) -> None:
    """Student vs the original labels and vs the teacher on the val split."""
    model.to(device)
    lv = logits_of(model, place_x(val.x, device), _data_e(val)).cpu()
    w = torch.from_numpy(val.w)
    tv = torch.from_numpy(info["val_teacher"])
    print(f"distill: student val BCE vs the original label weighted "
          f"{weighted_bce(lv, torch.from_numpy(info['val_label']), w):.4f} (teacher "
          f"{weighted_bce(tv, torch.from_numpy(info['val_label']), w):.4f}); "
          f"|student - teacher| logit mean {float((lv - tv).abs().mean()):.4f} "
          f"rms {float(((lv - tv) ** 2).mean().sqrt()):.4f}")


def verify_export(model: nn.Module, path: Path, states: list[str], tol: float = 1e-3) -> float:
    """Load ``path`` with ``Evaluator.load`` and compare ``evaluate`` to the torch logits.

    Returns the max absolute difference over ``states`` (raises if above ``tol``).
    """
    import poke_engine

    evaluator = poke_engine.Evaluator.load(str(path))
    model = model.to("cpu")
    parsed = [poke_engine.State.from_string(s) for s in states]
    version = getattr(evaluator, "feature_version", 1)
    rows = [dataset.state_row(poke_engine, s, 1, version) for s in parsed]
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


def aux_head_outputs(model: nn.Module, x: np.ndarray, e: np.ndarray | None, device: torch.device,
                     batch: int = 8192) -> np.ndarray:
    model.eval().to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = model_input(torch.from_numpy(x[i:i + batch]).to(device),
                             None if e is None else torch.from_numpy(e[i:i + batch]).to(device))
            outs.append(forward_with_aux(model, xb)[1].cpu())
    return torch.cat(outs).numpy()


def aux_report(model: nn.Module | None, val: Data, device: torch.device) -> dict[str, float]:
    """Held-out accuracy of the auxiliary targets (and of simple baselines) on ``val``.

    Survival is judged on the masked slots only (alive now, found in the final state). Every
    masked slot is alive in the input state, so "currently alive => survives" is the constant
    "survives" predictor (accuracy = the survive rate); ``hp>=0.5`` predicts survival only for
    healthy slots. ``model`` None (a net trained without --aux) reports the baselines only.
    """
    ns = 2 * aux.N_SLOTS
    y, m = val.aux_y, val.aux_m
    sm = m[:, :ns] > 0
    hp = np.concatenate([val.x[:, 53 * k].astype(np.float32)[:, None] for k in range(6)]
                        + [val.x[:, 363 + 53 * k].astype(np.float32)[:, None] for k in range(6)], 1)
    ys = y[:, :ns]
    out: dict[str, float] = {"n_slots": int(sm.sum())}
    out["survive_rate"] = float(ys[sm].mean())
    out["base_acc_alive_survives"] = out["survive_rate"]
    out["base_acc_hp_half"] = float(((hp[sm] >= 0.5) == (ys[sm] > 0.5)).mean())
    p = min(max(out["survive_rate"], 1e-6), 1 - 1e-6)
    out["base_bce_const"] = float(-(ys[sm] * math.log(p) + (1 - ys[sm]) * math.log(1 - p)).mean())
    own = np.zeros_like(sm)
    own[:, :6] = True
    sels = (("own", own & sm), ("opp", ~own & sm))
    for name, sel in sels:
        out[f"survive_rate_{name}"] = float(ys[sel].mean())
    dm = m[:, ns] > 0
    out["base_mse_diff_const"] = float(((y[dm, ns] - y[dm, ns].mean()) ** 2).mean())
    if model is not None and hasattr(model, "aux_head"):
        o = aux_head_outputs(model, val.x, val.e_sep if val.residual else None, device)
        lg = o[:, :ns]
        out["acc"] = float(((lg[sm] > 0) == (ys[sm] > 0.5)).mean())
        out["bce"] = float(nn.functional.binary_cross_entropy_with_logits(
            torch.from_numpy(lg[sm]), torch.from_numpy(ys[sm])))
        for name, sel in sels:
            out[f"acc_{name}"] = float(((lg[sel] > 0) == (ys[sel] > 0.5)).mean())
        out["mse_diff"] = float(((np.tanh(o[dm, ns]) - y[dm, ns]) ** 2).mean())
    print("\n=== auxiliary targets on val (masked: slots alive now and found in the final state) ===")
    print(f"survival slots {out['n_slots']}; survive rate {out['survive_rate']:.3f} "
          f"(own {out['survive_rate_own']:.3f}, opp {out['survive_rate_opp']:.3f})")
    print(f"baselines: 'alive now => survives' acc {out['base_acc_alive_survives']:.3f}; "
          f"'hp>=0.5 => survives' acc {out['base_acc_hp_half']:.3f}; constant BCE {out['base_bce_const']:.4f}")
    print(f"remaining-difference constant-mean MSE {out['base_mse_diff_const']:.4f}")
    if "acc" in out:
        print(f"aux head: survival acc {out['acc']:.3f} (own {out['acc_own']:.3f}, opp {out['acc_opp']:.3f}), "
              f"BCE {out['bce']:.4f}; remaining-difference MSE {out['mse_diff']:.4f}")
    return out


def report(model: nn.Module, train: Data, val: Data, device: torch.device) -> dict[str, float]:
    model.to(device)
    t = {k: torch.from_numpy(getattr(train, k)).to(device) for k in ("y", "w")}
    v = {k: torch.from_numpy(getattr(val, k)).to(device) for k in ("y", "w")}
    t["x"], v["x"] = place_x(train.x, device), place_x(val.x, device)
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
    p.add_argument("--q-evaluator", default=None, metavar="PATH|stock",
                   help="evaluator that anchors the recorded root-relative scores into absolute q. "
                        "Default: read per game from the header config (evaluator + evaluator_scale); "
                        "override when a header is missing or wrong. 'stock' forces the stock anchor")
    p.add_argument("--q-leaf-scale", type=float, default=None, metavar="S",
                   help="leaf scale of the recording's evaluator (default: header evaluator_scale)")
    p.add_argument("--q-source", choices=dataset.Q_SOURCES, default="decision",
                   help="q for world samples: decision mean (default) or the world's own score")
    p.add_argument("--nodes", choices=dataset.NODE_MODES, default="off",
                   help="bead rnm: train on the interior search nodes recorded with "
                        "selfplay --export-nodes: off (default), on (next to the root samples), "
                        "only; needs --value-target search|blend")
    p.add_argument("--node-min-visits", type=int, default=16,
                   help="--nodes: drop exported nodes with fewer visits (default 16)")
    p.add_argument("--nodes-per-world", type=int, default=0,
                   help="--nodes: keep at most this many of a world's exported nodes (random "
                        "subset, reproducible; 0 = all). Memory: each node is a full feature row")
    p.add_argument("--node-weight", type=float, default=1.0,
                   help="--nodes: loss weight of a fully visited node relative to a root sample")
    p.add_argument("--node-visit-ref", type=float, default=64.0,
                   help="--nodes: visits at which a node reaches full weight; fewer visits scale "
                        "the weight linearly (leaf-dominated nodes count less)")
    p.add_argument("--node-depth-discount", type=float, default=1.0,
                   help="--nodes: multiply the weight by this per joint turn below depth 1")
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
    p.add_argument("--aux", choices=("off", "on"), default="off",
                   help="on: auxiliary per-Pokemon survival + remaining-difference heads on the "
                        "value net's hidden layer (training only; the .pev is unchanged), see auxtargets.py")
    p.add_argument("--aux-weight", type=float, default=0.5,
                   help="weight of the auxiliary loss added to the value BCE (default 0.5)")
    p.add_argument("--aux-metrics", action="store_true",
                   help="load the aux targets and print the baselines even with --aux off")
    p.add_argument("--init", type=Path, default=None, metavar="MODEL.pev",
                   help="warm start: initialise weights (and the residual stock coefficient a "
                        "and output bias) from an existing .pev with the same model/hidden/"
                        "features/--residual; errors on mismatch. Typical: pretrain on human data, "
                        "then fine-tune on self-play with a lower --lr")
    p.add_argument("--distill", type=Path, default=None, metavar="TEACHER.pev",
                   help="knowledge distillation: train against sigmoid(teacher logit) on the same "
                        "rows (teacher = a residual .pev of the same --features, any hidden size; "
                        "evaluated on the GPU from the loaded feature rows). See --distill-weight, "
                        "--distill-init and docs/eval-speed.md")
    p.add_argument("--distill-weight", type=float, default=1.0, metavar="A",
                   help="target = A * sigmoid(teacher) + (1 - A) * the --value-target label (default 1)")
    p.add_argument("--distill-init", choices=("trunc", "emb", "none"), default="trunc",
                   help="student start with --distill: trunc = the teacher's embedding tables and its "
                        "--hidden hidden units of largest |w2| (needs --hidden <= teacher's), emb = "
                        "only the embedding tables, none = random (default trunc)")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto (cuda if available), cpu or cuda")
    p.add_argument("--embed-dim", default="8", metavar="D|S,I,A,M",
                   help="--features v4: learned identity embedding width, one value for all four "
                        "tables or species,item,ability,move (default 8); moves are mean-pooled")
    p.add_argument("--features", choices=("v1", "v2", "v3", "v4", "v5", "1", "2", "3", "4", "5"), default="v1",
                   help="engine feature layout: v1 (738 values) or v2 (v1 + item / ability / "
                        "move-property features, see docs/features-v2.md) or v3 (v1 + pairwise "
                        "speed / damage / revenge-kill matchup features, 926 values; '3' = 'v3') or v4 "
                        "(v2 + v3 features, 2379 values, plus per-slot species/item/ability/move "
                        "identity indices for learned embeddings, docs/features-v4.md)")
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
    if args.nodes != "off" and (args.value_target not in ("search", "blend") or args.q_kind != "absolute"):
        print("--nodes needs --value-target search|blend and --q-kind absolute", file=sys.stderr)
        return 2
    if args.nodes_per_world < 0:
        print("--nodes-per-world must be >= 0", file=sys.stderr)
        return 2
    if args.node_min_visits < 1 or args.node_weight <= 0 or args.node_visit_ref < 0             or not 0.0 < args.node_depth_discount <= 1.0:
        print("--node-min-visits >= 1, --node-weight > 0, --node-visit-ref >= 0, "
              "--node-depth-discount in (0, 1]", file=sys.stderr)
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
    feature_version = int(args.features.lstrip("v"))
    x_dtype = args.x_dtype or ("float16" if feature_version >= 2 else "float32")
    if feature_version != 1 and (args.augment_bench or args.augment_sides):
        print("--augment-* supports only --features v1", file=sys.stderr)
        return 2
    embed_dims = None
    if feature_version in (4, 5):
        try:
            embed_dims = parse_embed_dims(args.embed_dim)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.model != "mlp":
            print("--features v4/v5 needs --model mlp", file=sys.stderr)
            return 2
    print(f"features: {args.features}, x dtype {x_dtype}"
          + (f", identity embeddings (species,item,ability,move) dims {parse_embed_dims(args.embed_dim)}"
             if feature_version in (4, 5) else ""))
    data = load_data(
        args.data, args.states, args.weighting, args.value_target, args.blend_lambda, args.q_source,
        args.q_kind, args.q_eval_scale, residual=args.residual,
        td_lambda=args.td_lambda, td_n=args.td_n, td_mu=mu,
        feature_version=feature_version, x_dtype=x_dtype,
        max_per_game=args.max_per_game, seed=args.seed, workers=args.load_workers,
        nodes=args.nodes, node_min_visits=args.node_min_visits, node_weight=args.node_weight,
        node_visit_ref=args.node_visit_ref, node_depth_discount=args.node_depth_discount,
        nodes_per_world=args.nodes_per_world, aux_targets=args.aux == "on" or args.aux_metrics,
        q_evaluator=args.q_evaluator, q_leaf_scale=args.q_leaf_scale,
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
    train, val = data.split_inplace(tr_idx, va_idx)
    assert not set(train.tags) & set(val.tags)
    print(f"samples {len(data)} ({args.states}); battles {len(set(data.tags))}; "
          f"train {len(train)} samples / {len(set(train.tags))} battles; "
          f"val {len(val)} samples / {len(set(val.tags))} battles; "
          f"mean value {float(data.y.mean()):.3f}")
    n_feat = data.x.shape[1] - (1 if args.residual and data.e_sep is None else 0)
    del data  # train.x is a view of its buffer; val was copied out (no 2x peak)
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
                        args.dropout, embed_dims)
    if args.init is not None:
        try:
            load_init(model, args.init, model_kind=args.model, n_feat=n_feat, hidden=args.hidden,
                      feature_version=feature_version, residual=args.residual)
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"warm start from {args.init}"
              + (f" (a = {float(model.a.item()):.4f})" if args.residual else ""))
    orig_targets = None
    if args.distill is not None:
        if args.init is not None:
            print("--distill and --init are exclusive (use --distill-init)", file=sys.stderr)
            return 2
        try:
            orig_targets = apply_distill(model, args.distill, train, val, extra_val, n_feat=n_feat,
                                         feature_version=feature_version, residual=args.residual,
                                         weight=args.distill_weight, init=args.distill_init,
                                         device=device)
        except (ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.aux == "on":
        if args.model != "mlp" or args.augment_bench or args.augment_sides or args.aux_weight <= 0 \
                or feature_version != 1 or args.nodes != "off":
            print("--aux on needs --model mlp, --features v1, --nodes off, no --augment-*, "
                  "--aux-weight > 0", file=sys.stderr)
            return 2
        add_aux_head(model, args.hidden)
    augmenter = None
    if args.augment_bench or args.augment_sides:
        augmenter = augment.Augmenter(args.augment_bench, args.augment_sides, args.residual,
                                      args.seed, device)
    result = fit(
        model, train, val, device=device, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        augmenter=augmenter, extra_val=extra_val, early_stop_on=args.early_stop_on,
        aux_weight=args.aux_weight if args.aux == "on" else 0.0,
    )
    print(f"best epoch {result.best_epoch} of {result.epochs_run}"
          f" (early stopping on {args.early_stop_on})")
    if args.residual:
        print(f"learned a = {float(result.model.a.item()):.4f}  "
              f"(stock_coef = {result.model.stock_coef:.6f})")
    report(result.model, train, val, device)
    if orig_targets is not None:
        distill_report(result.model, val, orig_targets, device)
    if val.aux_y is not None:
        aux_report(result.model if args.aux == "on" else None, val, device)
    for name, vd in extra_val:
        m = evaluate_extra(result.model, vd, device)
        print(f"val-data {name}: BCE vs outcome weighted {m['bce_w']:.4f}  unweighted "
              f"{m['bce']:.4f}  decisive accuracy {m['acc']:.3f}  ({m['n']} samples, "
              f"{m['n_decisive']} decisive)")
    export_pev(result.model, args.out, feature_version)
    rng = random.Random(args.seed)
    # val.states only keeps a ~1/STATE_KEEP_EVERY subset of the state strings (the rest None)
    pool = [s for s in val.states if s]
    check = rng.sample(pool, min(100, len(pool)))
    if check:
        worst = verify_export(result.model, args.out, check)
        print(f"\nexported {args.out}  ({args.out.stat().st_size} bytes); verified on {len(check)} "
              f"val states, max |evaluate - torch logit| = {worst:.2e}")
    else:
        print(f"\nexported {args.out}  ({args.out.stat().st_size} bytes); WARNING: no val state "
              f"strings kept, export not verified", file=sys.stderr)
    print(f"value target: {target_desc}")
    print(f"total time {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
