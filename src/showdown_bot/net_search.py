"""Net-guided search driver (bead showdown-bot-589.68.5, docs/net-search.md).

``poke_engine.NetSearch`` (Rust) owns one PUCT tree per sampled world and hands out the token
buffers of the leaves it wants evaluated; this module runs the transformer on them (one batched
fp16 forward over both sides' views) and feeds policy priors for BOTH sides plus a value back.

* ``NetEvaluator``  - tokens (raw engine buffers) -> (policy1, policy2, value) bytes for ``feed``; the eager
                      autocast forward, or (``fast``) ``fast_forward.FastForward``: CUDA graphs, masked-token
                      compaction, pruned last layer, pinned buffers (bead 589.68.7).
* ``drive``         - the collect / forward / feed loop under a time and evaluation budget.
* ``NetSearcher``   - what a bot calls: worlds in, aggregated root visit policy out.
"""

from __future__ import annotations

import logging
import warnings
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

N_ACT = 13
MODES = ("argmax", "mixed")

DEFAULTS = {
    "worlds": 32, "time_ms": 1000, "max_evals": 20000, "batch": 128,
    "c_puct": 2.0, "depth": 4, "fpu": 0.0, "eps": 0.01, "vavg": 0, "scrub": 1, "pipe": 1, "views": 2,
    # 589.68.7 (all off = the sdb.24 behaviour): fast forward, fp16 residual stream, history rows (0 padding,
    # 1 root rows for every leaf, 2 root rows + the simulated turns), side-two scrub, tokenisation threads
    "fast": 0, "fp16res": 0, "hist": 0, "scrub2": 0, "tokthreads": 1,
}


def slot_names(choices: Sequence[str]) -> list[str]:
    """The 13 own-slot names of a ``state_tokens_str(..., with_choices=True)`` result."""
    return [str(c) for c in choices[:N_ACT]]


class NetEvaluator:
    """Runs a ``TokenNet`` on the engine's raw ``[n, views, CAT_LEN]`` / ``[n, views, CONT_LEN]`` buffers.

    ``views`` = 2: row ``2 i`` of the flattened batch is leaf ``i`` seen from side one, row ``2 i + 1``
    from side two; both are ordinary "own side" views for the net (it was trained on both roles) and
    side two's prior is its policy head on its own view. ``views`` = 1: only side one's view is run and
    side two's prior is the net's opponent-action head (``opp_slot``) on it: half the GPU work.
    The history rows are padding unless the engine hands out rows with the leaves (``hist`` > 0).

    ``fast``: the forward runs through ``FastForward`` (same function, see its docs) instead of the eager
    autocast model; ``fp16res`` keeps its residual stream in fp16 too (a little faster, diffs ~1e-3).
    """

    def __init__(self, net: Any, device: str, vavg: bool = False, views: int = 2, *, fast: bool = False,
                 fp16res: bool = False, hist: bool = False, batch: int = 256) -> None:
        import torch

        from showdown_bot import token_net, tokens

        self.torch = torch
        self.net = net
        self.device = torch.device(device)
        self.vavg = vavg and views == 2
        self.views = views
        self.use_hist = hist
        lay = tokens.layout()
        self.fields = {k: v for k, v in lay.items() if k in token_net.INPUT_FIELDS}
        self.cat_len = int(self._buffer_len("cat"))
        self.cont_len = int(self._buffer_len("cont"))
        self.n_hist = token_net.N_HIST
        self.hist_cat_w = 4
        self.hist_cont_w = net.cfg.c_hist
        self.calls = 0
        self.leaves = 0
        self.ms_prepare = 0.0
        self.ms_forward = 0.0
        self.fast = None
        if fast:
            from showdown_bot.fast_forward import FastForward

            self.fast = FastForward(net, self.device, use_hist=hist, half_res=fp16res)
            if self.device.type == "cuda":
                self.fast.warm_all(views * batch)

    def _buffer_len(self, buf: str) -> int:
        from showdown_bot import tokens

        return max(
            f["offset"] + int(np.prod(f["shape"])) for f in tokens.layout().values() if f["buffer"] == buf
        )

    def _hist_arrays(self, hist: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Engine hist bytes -> ``[views * n, 16, 4]`` int16 / ``[views * n, 16, 5]`` float32 (padding if None)."""
        v = self.views
        if hist is None:
            return (np.zeros((v * n, self.n_hist, self.hist_cat_w), np.int16),
                    np.zeros((v * n, self.n_hist, self.hist_cont_w), np.float32))
        return (np.frombuffer(hist[0], dtype="<i2").reshape(v * n, self.n_hist, self.hist_cat_w),
                np.frombuffer(hist[1], dtype="<f4").reshape(v * n, self.n_hist, self.hist_cont_w))

    def evaluate(self, cat: bytes, cont: bytes, n: int, hist: Any = None) -> tuple[bytes, bytes, bytes]:
        """``(policy1 [n,13], policy2 [n,13], value [n])`` as float32 little-endian bytes."""
        return self.finish(self.launch(cat, cont, n, hist))

    def launch(self, cat: bytes, cont: bytes, n: int, hist: Any = None) -> tuple[Any, Any, int]:
        """Prepares the batch and queues the forward (asynchronous on CUDA)."""
        v = self.views
        if self.fast is not None:
            t0 = time.perf_counter()
            hc, hf = self._hist_arrays(hist, n) if self.use_hist else (None, None)
            handle = self.fast.launch(cat, cont, v * n, hc, hf)
            self.ms_prepare += (time.perf_counter() - t0) * 1000.0
            return ("fast", handle, n)
        torch = self.torch
        t0 = time.perf_counter()
        dev = self.device
        c = torch.from_numpy(np.frombuffer(cat, dtype="<i2").reshape(v * n, self.cat_len)).to(dev, non_blocking=True)
        f = torch.from_numpy(np.frombuffer(cont, dtype="<f4").reshape(v * n, self.cont_len)).to(dev, non_blocking=True)
        b: dict[str, Any] = {}
        for name, fd in self.fields.items():
            src = c if fd["buffer"] == "cat" else f
            size = int(np.prod(fd["shape"]))
            b[name] = src[:, fd["offset"]: fd["offset"] + size].reshape(v * n, *fd["shape"])
        if hist is not None:
            hc, hf = self._hist_arrays(hist, n)
            b["hist_cat"] = torch.from_numpy(hc.copy()).to(dev, non_blocking=True).long()
            b["hist_cont"] = torch.from_numpy(hf.copy()).to(dev, non_blocking=True)
        else:
            b["hist_cat"] = torch.zeros(v * n, self.n_hist, self.hist_cat_w, dtype=torch.long, device=dev)
            b["hist_cont"] = torch.zeros(v * n, self.n_hist, self.hist_cont_w, dtype=torch.float32, device=dev)
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast(
            device_type=dev.type, dtype=torch.bfloat16 if dev.type == "cpu" else torch.float16
        ):
            o = self.net(b)
        if v == 2:
            pol = torch.softmax(o["policy"], -1).reshape(n, 2, N_ACT)
            val = torch.sigmoid(o["value"]).reshape(n, 2)
        else:
            pol = torch.stack([torch.softmax(o["policy"], -1), torch.softmax(o["opp_slot"], -1)], 1)
            val = torch.sigmoid(o["value"]).reshape(n, 1)
        self.ms_prepare += (t1 - t0) * 1000.0
        return pol, val, n

    def finish(self, handle: tuple[Any, Any, int]) -> tuple[bytes, bytes, bytes]:
        if handle[0] == "fast":
            _, fh, n = handle
            t0 = time.perf_counter()
            out = self.fast.result(fh).reshape(n, self.views, -1)
            self.ms_forward += (time.perf_counter() - t0) * 1000.0
            if self.views == 1:
                p = np.stack([out[:, 0, :N_ACT], out[:, 0, N_ACT: 2 * N_ACT]], 1)
            else:
                p = out[:, :, :N_ACT]
            v = out[:, :, 2 * N_ACT]
        else:
            pol, val, n = handle
            t0 = time.perf_counter()
            p = pol.float().cpu().numpy()
            v = val.float().cpu().numpy()
            self.ms_forward += (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self.leaves += n
        value = 0.5 * (v[:, 0] + 1.0 - v[:, 1]) if self.vavg else v[:, 0]
        return (
            np.ascontiguousarray(p[:, 0], dtype="<f4").tobytes(),
            np.ascontiguousarray(p[:, 1], dtype="<f4").tobytes(),
            np.ascontiguousarray(value, dtype="<f4").tobytes(),
        )


@dataclass
class DriveInfo:
    evals: int = 0
    batches: int = 0
    wall_ms: float = 0.0
    collect_ms: float = 0.0
    forward_ms: float = 0.0
    feed_ms: float = 0.0
    stats: dict[str, float] = field(default_factory=dict)


def drive(
    ns: Any,
    evaluate: Callable[[bytes, bytes, int], tuple[bytes, bytes, bytes]] | None = None,
    *,
    time_ms: float,
    max_evals: int,
    batch: int,
    launch: Callable[[bytes, bytes, int], Any] | None = None,
    finish: Callable[[Any], tuple[bytes, bytes, bytes]] | None = None,
    pipe: int = 1,
) -> DriveInfo:
    """collect -> evaluate -> feed until the time budget or ``max_evals`` is spent (or the trees
    have nothing left to ask for). The first batch is the roots, so every world is evaluated
    before it is expanded.

    ``evaluate`` is a synchronous evaluator. With ``launch`` / ``finish`` (asynchronous GPU
    forward) and ``pipe`` > 1 up to ``pipe`` batches are in flight: the tree is descended for the
    next batch (under virtual loss) while the GPU works on the previous one."""
    if evaluate is None and (launch is None or finish is None):
        raise ValueError("drive needs evaluate or launch + finish")
    info = DriveInfo()
    asynchronous = launch is not None and finish is not None and pipe > 1
    t_start = time.perf_counter()
    deadline = t_start + time_ms / 1000.0
    flying: list[Any] = []  # handles of in-flight batches, oldest first
    asked = 0  # leaves handed out (evaluated or in flight)
    exhausted = False
    while True:
        while (not exhausted and len(flying) < (pipe if asynchronous else 1) and asked < max_evals
               and time.perf_counter() < deadline):
            t0 = time.perf_counter()
            n, cat, cont = ns.collect(min(batch, max_evals - asked))
            t1 = time.perf_counter()
            info.collect_ms += (t1 - t0) * 1000.0
            if n == 0:
                exhausted = not flying or exhausted
                break
            asked += n
            hist = ns.take_hist() if hasattr(ns, "take_hist") else None
            if asynchronous:
                flying.append((launch(cat, cont, n) if hist is None else launch(cat, cont, n, hist), n))
                info.forward_ms += (time.perf_counter() - t1) * 1000.0
            else:
                p1, p2, v = evaluate(cat, cont, n) if hist is None else evaluate(cat, cont, n, hist)
                t2 = time.perf_counter()
                info.forward_ms += (t2 - t1) * 1000.0
                ns.feed(p1, p2, v)
                info.feed_ms += (time.perf_counter() - t2) * 1000.0
                info.evals += n
                info.batches += 1
        if not flying:
            break
        handle, n = flying.pop(0)
        t1 = time.perf_counter()
        p1, p2, v = finish(handle)
        t2 = time.perf_counter()
        info.forward_ms += (t2 - t1) * 1000.0
        ns.feed(p1, p2, v)
        info.feed_ms += (time.perf_counter() - t2) * 1000.0
        info.evals += n
        info.batches += 1
        if asked >= max_evals or time.perf_counter() >= deadline:
            exhausted = True
    info.wall_ms = (time.perf_counter() - t_start) * 1000.0
    info.stats = dict(ns.stats())
    return info


def aggregate(result: Sequence[Any], mode: str = "argmax") -> tuple[dict[str, float], dict[str, Any]]:
    """Sums the root visits of our options over the worlds. Returns ``(policy, info)``: ``policy``
    is the visit share per option (``mode`` only matters to the caller that picks from it)."""
    visits: dict[str, float] = {}
    q_sum: dict[str, float] = {}
    for rows1, _rows2, _value, _n in result:
        for name, n, q, _prior in rows1:
            visits[name] = visits.get(name, 0.0) + n
            q_sum[name] = q_sum.get(name, 0.0) + n * q
    total = sum(visits.values())
    if total <= 0:
        return {}, {"visits": 0}
    policy = {c: v / total for c, v in visits.items() if v > 0}
    q = {c: q_sum[c] / visits[c] for c in policy}
    root_values = [r[2] for r in result if r[3] > 0]
    return policy, {
        "visits": int(total),
        "q": q,
        "root_value": float(np.mean(root_values)) if root_values else 0.5,
    }


class NetSearcher:
    """Everything between "a list of world state strings" and "a visit policy over our options"."""

    def __init__(
        self,
        policy_net: Any,
        *,
        worlds: int = DEFAULTS["worlds"],
        time_ms: int = DEFAULTS["time_ms"],
        max_evals: int = DEFAULTS["max_evals"],
        batch: int = DEFAULTS["batch"],
        c_puct: float = DEFAULTS["c_puct"],
        depth: int = DEFAULTS["depth"],
        fpu: float = DEFAULTS["fpu"],
        eps: float = DEFAULTS["eps"],
        pipe: int = DEFAULTS["pipe"],
        views: int = DEFAULTS["views"],
        vavg: bool = bool(DEFAULTS["vavg"]),
        scrub: bool = bool(DEFAULTS["scrub"]),
        seed: int = 0,
        fast: bool = bool(DEFAULTS["fast"]),
        fp16res: bool = bool(DEFAULTS["fp16res"]),
        hist: int = DEFAULTS["hist"],
        scrub2: bool = bool(DEFAULTS["scrub2"]),
        tokthreads: int = DEFAULTS["tokthreads"],
    ) -> None:
        self.policy_net = policy_net
        self.hist = int(hist)
        self.scrub2 = bool(scrub2)
        self.tokthreads = int(tokthreads)
        self.evaluator = NetEvaluator(policy_net.net, policy_net.device, vavg=vavg, views=views, fast=fast,
                                      fp16res=fp16res, hist=self.hist > 0, batch=batch)
        self.worlds = worlds
        self.time_ms = time_ms
        self.max_evals = max_evals
        self.batch = batch
        self.c_puct = c_puct
        self.depth = depth
        self.fpu = fpu
        self.eps = eps
        self.scrub = scrub
        self.pipe = pipe
        self.views = views
        self.seed = seed
        self.searches = 0
        self.last: DriveInfo | None = None

    def new_search(self, states: Sequence[str], reveal: Any = None, hist: Any = None, reveal2: Any = None) -> Any:
        """``hist`` = ``(hist_cat [16,4], hist_cont [16,5])`` root rows (``self.hist`` > 0; zeros if None),
        ``reveal2`` = rows of what the opponent knows of our team (``self.scrub2``)."""
        import poke_engine

        self.searches += 1
        kw: dict[str, Any] = {}
        if self.hist:
            hc, hf = hist if hist is not None else (np.zeros((16, 4)), np.zeros((16, 5)))
            kw.update(hist=self.hist, hist_cat=np.ascontiguousarray(hc, dtype="<i2").tobytes(),
                      hist_cont=np.ascontiguousarray(hf, dtype="<f4").tobytes())
        if self.scrub2 and reveal2 is not None:
            kw["reveal2"] = reveal2
        if self.tokthreads > 1:
            kw["threads"] = self.tokthreads
        return poke_engine.NetSearch(
            list(states), c_puct=self.c_puct, fpu=self.fpu, depth=self.depth, eps=self.eps, views=self.views,
            seed=(self.seed * 1000003 + self.searches) & 0xFFFFFFFF,
            reveal=reveal if self.scrub else None, **kw,
        )

    def search(
        self, states: Sequence[str], reveal: Any = None, *, time_ms: float | None = None,
        max_evals: int | None = None, mode: str = "argmax", hist: Any = None, reveal2: Any = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ns = self.new_search(states, reveal, hist, reveal2)
        info = drive(
            ns, self.evaluator.evaluate, launch=self.evaluator.launch, finish=self.evaluator.finish, pipe=self.pipe,
            time_ms=self.time_ms if time_ms is None else time_ms,
            max_evals=self.max_evals if max_evals is None else max_evals,
            batch=self.batch,
        )
        self.last = info
        policy, extra = aggregate(ns.result(), mode)
        extra["drive"] = info
        return policy, extra
