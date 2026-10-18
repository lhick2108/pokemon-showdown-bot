"""Trainer for the token transformer (bead showdown-bot-589.68.3). Details: ``docs/token-net.md``.

    python -m showdown_bot.token_train --data data/tokens/v2 --weight selfplay=1 --weight human=0.5 \
        --epochs 4 --out models/tok-d256.pt

Streams memory-mapped shards (``showdown_bot.tokens`` export): nothing but index arrays is held in RAM.
Train/val/test is the shard ``split`` field (battle-hash based, a battle never crosses splits).
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from showdown_bot.token_net import INPUT_FIELDS, NEG, TokenNet, TokenNetConfig, spec_hash, to_tensors

TARGET_FIELDS = (
    "pol_target", "pol_valid", "z", "v_search", "v_search_valid", "opp_act_slot", "opp_act_kind", "opp_act_id",
    "opp_act_valid", "hid_ids", "hid_mask", "hid_valid", "aux_turns_left", "aux_final_alive", "aux_valid", "rating",
)
LOAD_FIELDS = INPUT_FIELDS + ("hist_cat", "hist_cont") + TARGET_FIELDS
INPUTS = tuple(dict.fromkeys(INPUT_FIELDS + ("hist_cat", "hist_cont")))
TERMS = ("pol", "val", "opp", "hid", "aux")
SPLITS = {"train": 0, "val": 1, "test": 2}
RATING_1800 = 4

# offline bars from the docs
BAR_POLICY_SELFPLAY = 0.472  # docs/policy-prior.md: h32 prior, top-1 vs visit argmax on held-out gen5/6 self-play
BAR_POLICY_HUMAN_1800 = 0.497  # docs/human-policy.md: top-1 on the 1800+ bucket of the test split (>= 2 legal actions)
BAR_POLICY_HUMAN_ALL = 0.504


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Corpus:
    """Shard directories -> per-source shard lists with the row indices of one split."""

    def __init__(self, dirs: list[Path]) -> None:
        self.dirs = dirs
        self.sources: list[dict[str, Any]] = []  # global: name, kind
        self.shards: list[dict[str, Any]] = []  # dir, name, source (global id), rows per split
        self.spec: dict[str, Any] | None = None
        for d in dirs:
            man = json.loads((d / "manifest.json").read_text())
            spec = man["spec"]
            if self.spec is None:
                self.spec = spec
            elif spec_hash(spec) != spec_hash(self.spec):
                raise SystemExit(f"{d}: token spec differs from the other data dirs")
            local = {}
            for s in man["sources"]:
                gid = len(self.sources)
                self.sources.append({"name": s["name"], "kind": s["kind"], "dir": str(d)})
                local[s["id"]] = gid
            for sh in man["shards"]:
                self.shards.append({"dir": d, "name": sh["name"], "source": local[sh["source"]], "n": sh["n"]})
        for sh in self.shards:
            split = np.load(sh["dir"] / "shards" / sh["name"] / "split.npy")
            sh["rows"] = {k: np.flatnonzero(split == v).astype(np.int32) for k, v in SPLITS.items()}

    def n(self, source: int, split: str) -> int:
        return sum(len(s["rows"][split]) for s in self.shards if s["source"] == source)

    def load(self, sh: dict[str, Any]) -> dict[str, np.ndarray]:
        d = sh["dir"] / "shards" / sh["name"]
        return {f: np.load(d / f"{f}.npy", mmap_mode="r") for f in LOAD_FIELDS}


class Stream:
    """Endless shuffled stream of one source's rows of one split: shard order shuffled each pass, ``k_open``
    shards interleaved, rows shuffled inside a shard. Memory-mapped, so RAM is only the index arrays."""

    def __init__(self, corpus: Corpus, source: int, split: str, rng: np.random.Generator, k_open: int = 6,
                 cap: int = 0) -> None:
        self.c, self.source, self.split, self.rng, self.k, self.cap = corpus, source, split, rng, k_open, cap
        self.shards = [s for s in corpus.shards if s["source"] == source and len(s["rows"][split])]
        self.order: list[int] = []
        self.active: list[list[Any]] = []  # [data, permuted rows, pos]
        self.n = sum(len(s["rows"][split]) for s in self.shards)

    def _open(self) -> None:
        while len(self.active) < self.k:
            if not self.order:
                self.order = list(self.rng.permutation(len(self.shards)))
            sh = self.shards[self.order.pop()]
            rows = self.rng.permutation(sh["rows"][self.split])
            if self.cap:  # at most ``cap`` random rows per battle (both seats share the hash) in this pass
                bh = np.load(sh["dir"] / "shards" / sh["name"] / "battle_hash.npy", mmap_mode="r")[np.sort(rows)]
                _, inv = np.unique(bh, return_inverse=True)
                inv = inv[np.argsort(np.argsort(rows))]  # back to the permuted order
                o = np.argsort(inv, kind="stable")
                first = np.searchsorted(inv[o], inv[o])
                rank = np.empty(len(rows), dtype=np.int64)
                rank[o] = np.arange(len(rows)) - first
                rows = rows[rank < self.cap]
            self.active.append([self.c.load(sh), rows, 0])
            if len(self.shards) < self.k and not self.order and len(self.active) >= len(self.shards):
                break

    def take(self, n: int) -> dict[str, np.ndarray]:
        self._open()
        rem = np.array([len(a[1]) - a[2] for a in self.active], dtype=np.float64)
        counts = self.rng.multinomial(n, rem / rem.sum())
        parts = []
        for a, c in zip(self.active, counts):
            if c == 0:
                continue
            idx = np.sort(a[1][a[2] : a[2] + c])
            a[2] += c
            parts.append({f: a[0][f][idx] for f in LOAD_FIELDS})
        self.active = [a for a in self.active if a[2] < len(a[1])]
        b = {f: np.concatenate([p[f] for p in parts]) for f in LOAD_FIELDS}
        b["src"] = np.full(len(b["z"]), self.source, dtype=np.int64)
        got = len(b["z"])
        if got < n:  # a shard ran out: top up
            more = self.take(n - got)
            b = {k: np.concatenate([b[k], more[k]]) for k in b}
        return b


class Mixer:
    """Weighted mixture of source streams; the training batch generator."""

    def __init__(self, corpus: Corpus, weights: dict[str, float], seed: int, cap: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.streams = []
        probs = []
        groups: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(corpus.sources):
            if corpus.n(i, "train"):
                key = s["name"] if s["name"] in weights else s["kind"]
                groups[key].append(i)
        for key, ids in groups.items():
            tot = sum(corpus.n(i, "train") for i in ids)
            for i in ids:
                self.streams.append(Stream(corpus, i, "train", self.rng, cap=cap))
                probs.append(weights.get(key, 1.0) * corpus.n(i, "train") / tot)
        p = np.array(probs, dtype=np.float64)
        self.p = p / p.sum()
        self.n_train = sum(s.n for s in self.streams)

    def take(self, n: int) -> dict[str, np.ndarray]:
        counts = self.rng.multinomial(n, self.p)
        parts = [s.take(int(c)) for s, c in zip(self.streams, counts) if c]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


_LIVE: list["Prefetch"] = []


class Prefetch:
    """Background thread (not a process) that keeps ``depth`` training batches ready."""

    def __init__(self, fn, n: int, depth: int = 3) -> None:
        self.q: queue.Queue = queue.Queue(depth)
        self.fn, self.n = fn, n
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        _LIVE.append(self)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                item = self.fn(self.n)
            except Exception as e:  # surfaced in the consumer
                item = e
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    pass
            if isinstance(item, Exception):
                return

    def get(self) -> dict[str, np.ndarray]:
        x = self.q.get()
        if isinstance(x, Exception):
            raise x
        return x

    def close(self) -> None:
        self.stop.set()
        self.t.join(timeout=10)
        if self in _LIVE:
            _LIVE.remove(self)


def eval_batches(corpus: Corpus, source: int, split: str, max_n: int, bs: int):
    """Evenly strided sample of at most ``max_n`` rows of ``source``/``split``, sequential reads."""
    shards = [s for s in corpus.shards if s["source"] == source]
    total = sum(len(s["rows"][split]) for s in shards)
    if not total:
        return
    step = max(1, math.ceil(total / max_n))
    buf: list[dict[str, np.ndarray]] = []
    nbuf = 0
    for sh in shards:
        rows = sh["rows"][split][::step]
        if not len(rows):
            continue
        data = corpus.load(sh)
        for i in range(0, len(rows), bs):
            idx = rows[i : i + bs]
            part = {f: data[f][idx] for f in LOAD_FIELDS}
            part["src"] = np.full(len(idx), source, dtype=np.int64)
            buf.append(part)
            nbuf += len(idx)
            if nbuf >= bs:
                yield {k: np.concatenate([p[k] for p in buf]) for k in buf[0]}
                buf, nbuf = [], 0
    if buf:
        yield {k: np.concatenate([p[k] for p in buf]) for k in buf[0]}


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


class LossCfg:
    def __init__(self, a: argparse.Namespace | None = None, **kw: Any) -> None:
        self.w = {"pol": 1.0, "val": 1.0, "opp": 0.3, "hid": 0.3, "aux": 0.05}
        self.value_target = "z"  # z | search | mix
        self.value_mix = 0.5
        self.opp_vocab_w = 0.5
        self.id_dropout = 0.0  # per-sample probability of blanking every categorical id (training only)
        self.numbers_only = False  # ablation: blank the ids always (training and evaluation)
        self.hp_coef: dict[int, tuple[float, float]] = {}  # per source: HP-sum logistic (intercept, slope)
        self.amp = "bf16"
        if a is not None:
            self.id_dropout, self.numbers_only, self.amp = a.id_dropout, a.numbers_only, a.amp
            for t in TERMS:
                self.w[t] = getattr(a, f"w_{t}")
            self.value_target, self.value_mix = a.value_target, a.value_mix
        for k, v in kw.items():
            if k in self.w:
                self.w[k] = v
            else:
                setattr(self, k, v)


def blank_ids(b: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Copy of ``b`` with every categorical id of the selected samples blanked (unknown; the known flags,
    legality and all continuous features stay): the net then only sees numbers for those samples."""
    r = rows
    out = dict(b)
    mc = b["mon_cat"].long()
    out["mon_cat"] = torch.where(r[:, None, None] & (mc > 0), torch.ones_like(mc), mc)
    out["act_cat"] = torch.where(r[:, None, None, None], torch.zeros_like(b["act_cat"]), b["act_cat"])
    hc = b["hist_cat"].clone()
    hc[..., 2:] = torch.where(r[:, None, None], torch.zeros_like(hc[..., 2:]), hc[..., 2:])
    out["hist_cat"] = hc
    return out


def hpsum_feature(nb: dict[str, np.ndarray]) -> np.ndarray:
    """Own minus opponent hp fraction sum (hidden scrubbed opponent slots count as full hp)."""
    hp = nb["mon_cont"][:, :, 0].astype(np.float32)
    hidden = (nb["mon_known"][:, 6:, 0] == 0) & (nb["meta"][:, 2:3] == 1)
    hp[:, 6:] = np.where(hidden, 1.0, hp[:, 6:])
    return hp[:, :6].sum(1) - hp[:, 6:].sum(1)


def fit_hpsum(corpus: Corpus, source: int, n: int = 20000, seed: int = 0) -> tuple[float, float]:
    """1-feature logistic z ~ sigmoid(a + b * hp-sum difference) on a train sample (Newton)."""
    st = Stream(corpus, source, "train", np.random.default_rng(seed))
    if not st.n:
        return 0.0, 0.0
    b = st.take(min(n, st.n))
    x, z = hpsum_feature(b), b["z"].astype(np.float64)
    w = np.zeros(2)
    X = np.stack([np.ones_like(x), x], 1).astype(np.float64)
    for _ in range(25):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - z) + 1e-3 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-3 * np.eye(2)
        w -= np.linalg.solve(H, g)
    return float(w[0]), float(w[1])


def value_target(b: dict[str, torch.Tensor], cfg: LossCfg) -> torch.Tensor:
    z = b["z"]
    if cfg.value_target == "z":
        return z
    ok = b["v_search_valid"] > 0
    if cfg.value_target == "search":
        return torch.where(ok, b["v_search"], z)
    return torch.where(ok, cfg.value_mix * b["v_search"] + (1 - cfg.value_mix) * z, z)


def hidden_terms(out: dict[str, torch.Tensor], b: dict[str, torch.Tensor], n_move: int):
    """Per-slot hidden-set losses, masks and metric helpers. Everything is [B, 6]."""
    hid = b["hid_ids"].long()
    kn = b["mon_known"][:, 6:12]
    ok = (b["hid_valid"] > 0)[:, None] & (b["hid_mask"] > 0) & (kn[..., 0] > 0)
    nb = hid.shape[0]

    def ce(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt = tgt.clamp(0, logits.shape[-1] - 1)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(nb, 6)

    res = {}
    for name, key, col, kcol in (("item", "hid_item", 1, 1), ("ability", "hid_ability", 2, 2), ("tera", "hid_tera", 7, 7)):
        m = ok & (kn[..., kcol] == 0)
        tgt = hid[..., col]
        pred = out[key].argmax(-1)
        res[name] = (ce(out[key], tgt), m, (pred == tgt.clamp(0, out[key].shape[-1] - 1)).float())
    # moves: softmax over the vocabulary without the revealed ones, target = uniform on the unrevealed true moves
    lg = out["hid_move"]
    rev = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, b["mon_cat"][:, 6:12, 3:7].long().clamp(0, n_move - 1), True)
    rev[..., :2] = True
    lg = lg.masked_fill(rev, NEG)
    tm = torch.zeros_like(lg).scatter_(-1, hid[..., 3:7].clamp(0, n_move - 1), 1.0) * (~rev)
    tsum = tm.sum(-1)
    m = ok & (tsum > 0)
    logp = F.log_softmax(lg, -1)
    loss = -(tm / tsum.clamp(min=1).unsqueeze(-1) * logp).sum(-1)
    top = lg.topk(4, -1).indices
    recall = tm.gather(-1, top).sum(-1) / tsum.clamp(min=1)
    res["move"] = (loss, m, recall)
    return res


def compute_loss(out: dict[str, torch.Tensor], b: dict[str, torch.Tensor], cfg: LossCfg, scale: torch.Tensor,
                 n_move: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``scale`` [len(TERMS), n_sources]: per-source multipliers. Returns (total, per-term detached values)."""
    src = b["src"]
    keep = ~b["blank"] if "blank" in b else None  # id-blanked samples train the value pathway only

    def term(name: str, per: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if keep is not None and name != "val":
            mask = mask & (keep.view(-1, *([1] * (mask.dim() - 1))))
        w = cfg.w[name] * scale[TERMS.index(name)][src]
        while w.dim() < per.dim():
            w = w.unsqueeze(-1)
        m = mask.float()
        return (per * m * w).sum() / m.sum().clamp(min=1)

    t: dict[str, torch.Tensor] = {}
    pol = -(b["pol_target"] * F.log_softmax(out["policy"], -1)).sum(-1)
    t["pol"] = term("pol", pol, b["pol_valid"] > 0)
    val = F.binary_cross_entropy_with_logits(out["value"], value_target(b, cfg), reduction="none")
    t["val"] = term("val", val, torch.ones_like(val, dtype=torch.bool))
    slot = b["opp_act_slot"].long()
    ov = b["opp_act_valid"] > 0
    m_slot = ov & (slot >= 0)
    l_slot = F.cross_entropy(out["opp_slot"], slot.clamp(min=0), reduction="none")
    kind, aid = b["opp_act_kind"].long(), b["opp_act_id"].long()
    label = torch.where(kind == 2, n_move + aid, aid).clamp(0, out["opp_vocab"].shape[-1] - 1)
    l_voc = F.cross_entropy(out["opp_vocab"], label, reduction="none")
    m_voc = ov & (kind >= 0) & (aid > 0)
    t["opp"] = term("opp", l_slot, m_slot) + cfg.opp_vocab_w * term("opp", l_voc, m_voc)
    h = hidden_terms(out, b, n_move)
    hid = sum(term("hid", l, m) for l, m, _ in h.values())
    t["hid"] = hid
    av = b["aux_valid"]
    l_t = F.smooth_l1_loss(out["aux"][:, 0], torch.log1p(b["aux_turns_left"].float().clamp(min=0)) / 4.0, reduction="none")
    l_f = F.mse_loss(torch.sigmoid(out["aux"][:, 1:3]), b["aux_final_alive"].float() / 6.0, reduction="none").mean(-1)
    t["aux"] = term("aux", l_t, av[:, 0] > 0) + term("aux", l_f, av[:, 1] > 0)
    total = sum(t.values())
    return total, {k: v.detach() for k, v in t.items()}


class Meter:
    def __init__(self) -> None:
        self.s: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    def add(self, name: str, _unused: None, mask: torch.Tensor, per: torch.Tensor | None = None) -> None:
        m = mask.float()
        x = (per * m).sum() if per is not None else torch.zeros(())
        self.s[name][0] += float(x)
        self.s[name][1] += float(m.sum())

    def get(self, name: str) -> float:
        s, n = self.s.get(name, (0.0, 0.0))
        return s / n if n else float("nan")

    def count(self, name: str) -> int:
        return int(self.s[name][1]) if name in self.s else 0


def batch_metrics(meter: Meter, out: dict[str, torch.Tensor], b: dict[str, torch.Tensor], cfg: LossCfg, n_move: int) -> None:
    nb = b["z"].shape[0]
    legal_n = (b["act_legal"][:, 0] > 0).sum(-1)
    subsets = {"": torch.ones(nb, dtype=torch.bool, device=legal_n.device)}
    if bool((b["rating"] == RATING_1800).any()):
        subsets["r1800"] = b["rating"] == RATING_1800
    pv = b["pol_valid"] > 0
    logp = F.log_softmax(out["policy"], -1)
    ce = -(b["pol_target"] * logp).sum(-1)
    top1 = (out["policy"].argmax(-1) == b["pol_target"].argmax(-1)).float()
    top3 = (out["policy"].topk(3, -1).indices == b["pol_target"].argmax(-1, keepdim=True)).any(-1).float()
    unif = 1.0 / legal_n.clamp(min=1).float()
    p = torch.sigmoid(out["value"])
    z = b["z"]
    bce = F.binary_cross_entropy_with_logits(out["value"], z, reduction="none")
    vs_ok = b["v_search_valid"] > 0
    vs = b["v_search"].clamp(1e-4, 1 - 1e-4)
    bce_vs_z = -(z * vs.log() + (1 - z) * (1 - vs).log())
    bce_vs = F.binary_cross_entropy_with_logits(out["value"], vs, reduction="none")
    decided = z != 0.5
    slot = b["opp_act_slot"].long()
    ov = b["opp_act_valid"] > 0
    m_slot = ov & (slot >= 0)
    lp_slot = F.log_softmax(out["opp_slot"], -1).gather(-1, slot.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    kind, aid = b["opp_act_kind"].long(), b["opp_act_id"].long()
    label = torch.where(kind == 2, n_move + aid, aid).clamp(0, out["opp_vocab"].shape[-1] - 1)
    lp_voc = F.log_softmax(out["opp_vocab"], -1).gather(-1, label.unsqueeze(-1)).squeeze(-1)
    m_voc = ov & (kind >= 0) & (aid > 0)
    hid = hidden_terms(out, b, n_move)
    for tag, rows in subsets.items():
        sfx = f"@{tag}" if tag else ""
        g2 = pv & (legal_n >= 2) & rows
        meter.add("pol_top1" + sfx, None, g2, top1)
        meter.add("pol_top3" + sfx, None, g2, top3)
        meter.add("pol_unif_top1" + sfx, None, g2, unif)
        meter.add("pol_ce" + sfx, None, pv & rows, ce)
        meter.add("pol_top1_all" + sfx, None, pv & rows, top1)
        meter.add("pol_sharp_top1" + sfx, None, g2 & (b["pol_target"].max(-1).values >= 0.6), top1)
        meter.add("val_bce" + sfx, None, rows, bce)
        meter.add("val_acc" + sfx, None, rows & decided, ((p > 0.5).float() == z).float())
        meter.add("val_brier" + sfx, None, rows, (p - z) ** 2)
        meter.add("val_z" + sfx, None, rows, z)
        meter.add("val_bce_vs_z" + sfx, None, rows & vs_ok, bce_vs_z)
        meter.add("val_bce_net_on_vs" + sfx, None, rows & vs_ok, bce)
        meter.add("val_bce_net_vs_target" + sfx, None, rows & vs_ok, bce_vs)
        meter.add("opp_slot_top1" + sfx, None, m_slot & rows, (out["opp_slot"].argmax(-1) == slot).float())
        meter.add("opp_slot_nll" + sfx, None, m_slot & rows, -lp_slot)
        meter.add("opp_vocab_top1" + sfx, None, m_voc & rows, (out["opp_vocab"].argmax(-1) == label).float())
        meter.add("opp_vocab_nll" + sfx, None, m_voc & rows, -lp_voc)
        for name, (loss, m, acc) in hid.items():
            r = rows[:, None].expand_as(m)
            meter.add(f"hid_{name}_{'recall4' if name == 'move' else 'acc'}" + sfx, None, m & r, acc)
            meter.add(f"hid_{name}_nll" + sfx, None, m & r, loss)
    av = b["aux_valid"]
    terr = (out["aux"][:, 0] - torch.log1p(b["aux_turns_left"].float().clamp(min=0)) / 4.0).abs()
    meter.add("aux_turns_logerr", None, av[:, 0] > 0, terr)


# ---------------------------------------------------------------------------
# Evaluation / reporting
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: TokenNet, corpus: Corpus, source: int, split: str, max_n: int, bs: int, dev: torch.device,
             cfg: LossCfg, scale: torch.Tensor, amp: str) -> tuple[Meter, float, int]:
    """Metrics for one source/split plus the weighted training-style loss (for early stopping)."""
    model.eval()
    meter = Meter()
    tot, n = 0.0, 0
    for nb in eval_batches(corpus, source, split, max_n, bs):
        b = to_tensors(nb, dev)
        allrows = torch.ones(len(nb["z"]), dtype=torch.bool, device=dev)
        if cfg.numbers_only:
            b = blank_ids(b, allrows)
        with autocast(dev, amp):
            out = model(b)
            blind = out if cfg.numbers_only else model(blank_ids(b, allrows))
        loss, _ = compute_loss(out, b, cfg, scale, model.cfg.n_move)
        z = b["z"]
        meter.add("val_bce_blind", None, torch.ones_like(z, dtype=torch.bool),
                  F.binary_cross_entropy_with_logits(blind["value"], z, reduction="none"))
        if source in cfg.hp_coef:
            a0, a1 = cfg.hp_coef[source]
            lg = torch.from_numpy(a0 + a1 * hpsum_feature(nb)).float().to(dev)
            meter.add("val_bce_hpsum", None, torch.ones_like(z, dtype=torch.bool),
                      F.binary_cross_entropy_with_logits(lg, z, reduction="none"))
        k = len(nb["z"])
        tot += float(loss) * k
        n += k
        batch_metrics(meter, out, b, cfg, model.cfg.n_move)
    model.train()
    return meter, (tot / n if n else float("nan")), n


def autocast(dev: torch.device, amp: str):
    if amp == "none" or dev.type != "cuda":
        return torch.autocast(device_type=dev.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


def base_rate_bce(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def format_meter(m: Meter, sfx: str = "") -> list[str]:
    g = lambda k: m.get(k + sfx)  # noqa: E731
    lines = [
        f"  policy  top1 {g('pol_top1'):.4f} (>=2 legal; n={m.count('pol_top1' + sfx)})  top3 {g('pol_top3'):.4f}  "
        f"uniform top1 {g('pol_unif_top1'):.4f}  CE {g('pol_ce'):.4f}  sharp(>=.6) top1 {g('pol_sharp_top1'):.4f}  top1(all) {g('pol_top1_all'):.4f}",
        f"  value   BCE {g('val_bce'):.4f}  ln2 0.6931  const-base-rate {base_rate_bce(g('val_z')):.4f}  acc {g('val_acc'):.4f}  "
        f"brier {g('val_brier'):.4f}  hp-sum logistic {g('val_bce_hpsum'):.4f}  ids-blanked {g('val_bce_blind'):.4f}  | on search-valid rows: net {g('val_bce_net_on_vs'):.4f} vs search-value-as-predictor {g('val_bce_vs_z'):.4f}"
        f" (n={m.count('val_bce_vs_z' + sfx)})",
        f"  opp act slot top1 {g('opp_slot_top1'):.4f} nll {g('opp_slot_nll'):.4f} (n={m.count('opp_slot_top1' + sfx)})  "
        f"vocab top1 {g('opp_vocab_top1'):.4f} nll {g('opp_vocab_nll'):.4f} (n={m.count('opp_vocab_top1' + sfx)})",
        f"  hidden  item acc {g('hid_item_acc'):.4f}  ability acc {g('hid_ability_acc'):.4f}  tera acc {g('hid_tera_acc'):.4f}  "
        f"moves recall@4 {g('hid_move_recall4'):.4f}  (n item {m.count('hid_item_acc' + sfx)}, move {m.count('hid_move_recall4' + sfx)})",
    ]
    return lines


class Logger:
    def __init__(self, path: Path | None) -> None:
        self.f = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a", encoding="utf8", buffering=1)

    def close(self) -> None:
        if self.f:
            self.f.close()
            self.f = None

    def __call__(self, *parts: Any) -> None:
        s = " ".join(str(p) for p in parts)
        print(s, flush=True)
        if self.f:
            self.f.write(s + "\n")


def final_report(model: TokenNet, corpus: Corpus, a: argparse.Namespace, dev: torch.device, cfg: LossCfg,
                 scale: torch.Tensor, log: Logger) -> dict[str, Any]:
    log("=" * 78)
    log(f"FINAL METRICS (best checkpoint; up to {a.final_eval_max} rows per source/split)")
    summary: dict[str, Any] = {}
    meters: dict[tuple[int, str], Meter] = {}
    for i, s in enumerate(corpus.sources):
        for split in ("val", "test"):
            if not corpus.n(i, split):
                continue
            m, loss, n = evaluate(model, corpus, i, split, a.final_eval_max, a.batch, dev, cfg, scale, a.amp)
            meters[(i, split)] = m
            log(f"[{s['name']} ({s['kind']}) {split}] n={n} weighted-loss {loss:.4f}")
            for ln in format_meter(m):
                log(ln)
            summary[f"{s['name']}/{split}"] = {k: m.get(k) for k in list(m.s)}
            if m.count("pol_top1@r1800"):
                log("  -- rating 1800+ rows --")
                for ln in format_meter(m, "@r1800"):
                    log(ln)
    log("-" * 78)
    log("BARS")

    def pooled(kind: str, splits: tuple[str, ...], key: str) -> tuple[float, int]:
        num = den = 0.0
        for (i, split), m in meters.items():
            if corpus.sources[i]["kind"] == kind and split in splits:
                n = m.count(key)
                if n:
                    num += m.get(key) * n
                    den += n
        return (num / den if den else float("nan")), int(den)

    sp, nsp = pooled("selfplay", ("val",), "pol_top1")
    log(f"(1) self-play policy top-1 vs search visit argmax, held-out val, pooled {nsp} rows: {sp:.4f}   bar {BAR_POLICY_SELFPLAY} "
        f"(h32 prior, gen5/6; different generations here)  -> {'ABOVE' if sp > BAR_POLICY_SELFPLAY else 'below'}")
    hs = ("test",) if any(corpus.n(i, "test") for i, s in enumerate(corpus.sources) if s["kind"] == "human") else ("val",)
    h18, n18 = pooled("human", hs, "pol_top1@r1800")
    hall, nall = pooled("human", hs, "pol_top1")
    log(f"(2) human policy top-1 vs played action ({hs[0]} split, >=2 legal): 1800+ {h18:.4f} (n={n18})   bar {BAR_POLICY_HUMAN_1800};"
        f" all ratings {hall:.4f} (n={nall})   bar {BAR_POLICY_HUMAN_ALL}  (bars used an elo input; this net has none)")
    for kind in ("selfplay", "human"):
        v, n = pooled(kind, ("val",) if kind == "selfplay" else hs, "val_bce")
        z, _ = pooled(kind, ("val",) if kind == "selfplay" else hs, "val_z")
        hpb, _ = pooled(kind, ("val",) if kind == "selfplay" else hs, "val_bce_hpsum")
        bl, _ = pooled(kind, ("val",) if kind == "selfplay" else hs, "val_bce_blind")
        log(f"(3) value BCE vs outcome z, {kind}, {n} rows: {v:.4f}   constant 0.5: 0.6931   constant base rate: {base_rate_bce(z):.4f}"
            f"   HP-sum logistic: {hpb:.4f}   same net with ids blanked: {bl:.4f}   -> ids {'help' if v < bl - 0.002 else ('no effect' if abs(v - bl) <= 0.002 else 'HURT (memorisation)')}")
    o, no = pooled("selfplay", ("val",), "opp_slot_nll")
    log(f"(4) opponent action (slot head) NLL self-play {o:.4f} (n={no}); see per-source blocks for top-1, hidden-set accuracy")
    summary["bars"] = {"selfplay_pol_top1": sp, "human_pol_top1_1800": h18, "human_pol_top1_all": hall}
    return summary


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", action="append", type=Path, required=True, help="token shard dir (repeatable)")
    p.add_argument("--weight", action="append", default=[], metavar="NAME=W",
                   help="sampling weight of a source kind (selfplay/human) or source name; sources of one group share it "
                        "in proportion to their size; default 1.0 per group")
    p.add_argument("--src-loss", action="append", default=[], metavar="NAME.TERM=V",
                   help="per-source loss multiplier, NAME = source name or kind, TERM in pol,val,opp,hid,aux (e.g. human.hid=0)")
    for t, d in (("pol", 1.0), ("val", 1.0), ("opp", 0.3), ("hid", 0.3), ("aux", 0.05)):
        p.add_argument(f"--w-{t}", type=float, default=d)
    p.add_argument("--value-target", choices=("z", "search", "mix"), default="search",
                   help="search = v_search where valid (self-play) else z (human): per-position targets; z = outcome only")
    p.add_argument("--value-mix", type=float, default=0.5, help="weight of v_search in --value-target mix")
    p.add_argument("--id-dropout", type=float, default=0.0,
                   help="per-sample probability of blanking all categorical ids in training; such samples train the value "
                        "head only (policy/opp/hidden/aux losses are masked for them)")
    p.add_argument("--numbers-only", action="store_true", help="ablation: blank all categorical ids in training and evaluation")
    p.add_argument("--max-per-battle", type=int, default=0,
                   help="cap on rows per battle in each pass over a shard (0 = no cap); both seats of a battle count together")
    p.add_argument("--select", choices=("value", "loss"), default="value",
                   help="early stopping / best checkpoint on val value BCE (default) or on the weighted total loss; the other "
                        "criterion's best is kept as <out>.byloss.pt / <out>.byvalue.pt")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--epoch-samples", type=int, default=0, help="samples per epoch (default: all train rows, weighted mixture)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr-frac", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500, help="warmup steps")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3, help="epochs without val improvement before stopping (0 = never)")
    p.add_argument("--amp", choices=("bf16", "fp16", "none"), default="bf16")
    p.add_argument("--eval-max", type=int, default=20000, help="val rows per source for per-epoch evaluation")
    p.add_argument("--final-eval-max", type=int, default=200000)
    p.add_argument("--out", type=Path, required=True, help="exported best net (.pt); <out>.last is the resumable checkpoint")
    p.add_argument("--log", type=Path, default=None, help="text log (default models/log-<out stem>.txt next to --out)")
    p.add_argument("--resume", action="store_true", help="continue from <out>.last if it exists")
    p.add_argument("--ckpt-every", type=int, default=2000, help="steps between resumable checkpoints")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--max-minutes", type=float, default=0, help="stop training after this long (smoke tests)")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--prefetch", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-final", action="store_true", help="skip the final metrics block")
    return p.parse_args(argv)


def kv(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def make_optimizer(model: TokenNet, a: argparse.Namespace) -> torch.optim.Optimizer:
    dec, nodec = [], []
    for n, p in model.named_parameters():
        (nodec if p.ndim < 2 or n.startswith("e_") or n.startswith("h_opp_bias") else dec).append(p)
    return torch.optim.AdamW([{"params": dec, "weight_decay": a.wd}, {"params": nodec, "weight_decay": 0.0}],
                             lr=a.lr, betas=(0.9, 0.98), eps=1e-8)


def lr_at(step: int, total: int, a: argparse.Namespace) -> float:
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    prog = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    log = Logger(a.log or (a.out.parent / f"log-{a.out.stem}.txt"))
    try:
        return run(a, log)
    finally:
        for pf in list(_LIVE):
            pf.close()
        log.close()


def run(a: argparse.Namespace, log: Logger) -> int:
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    log(f"=== token_train {time.strftime('%Y-%m-%d %H:%M:%S')} args: {json.dumps({k: str(v) for k, v in vars(a).items()})}")
    corpus = Corpus(list(a.data))
    for i, s in enumerate(corpus.sources):
        log(f"source {i} {s['name']} ({s['kind']}): train {corpus.n(i, 'train')} val {corpus.n(i, 'val')} test {corpus.n(i, 'test')}")
    weights = kv(a.weight)
    cfg = LossCfg(a)
    mixer = Mixer(corpus, weights, a.seed, cap=a.max_per_battle)
    for i in range(len(corpus.sources)):
        if corpus.n(i, "train"):
            cfg.hp_coef[i] = fit_hpsum(corpus, i)
    for i, src in enumerate(corpus.sources):  # test-only sources borrow the fit of a same-kind source
        if i not in cfg.hp_coef:
            same = [j for j in cfg.hp_coef if corpus.sources[j]["kind"] == src["kind"]]
            if same:
                cfg.hp_coef[i] = cfg.hp_coef[same[0]]
    log("hp-sum logistic baseline (intercept, slope): " + str({corpus.sources[i]["name"]: tuple(round(x, 3) for x in c) for i, c in cfg.hp_coef.items()}))
    scale = torch.ones(len(TERMS), len(corpus.sources))
    for item in a.src_loss:
        left, v = item.split("=")
        name, term = left.rsplit(".", 1)
        for i, s in enumerate(corpus.sources):
            if name in (s["name"], s["kind"]):
                scale[TERMS.index(term), i] = float(v)
    scale = scale.to(dev)

    mcfg = TokenNetConfig.from_spec(corpus.spec, d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                                    mlp_ratio=a.mlp_ratio, dropout=a.dropout)
    model = TokenNet(mcfg).to(dev)
    opt = make_optimizer(model, a)
    scaler = torch.amp.GradScaler("cuda", enabled=(a.amp == "fp16" and dev.type == "cuda"))
    log(f"model: {model.n_params() / 1e6:.2f}M params ({asdict(mcfg)})")
    epoch_samples = a.epoch_samples or mixer.n_train
    steps_per_epoch = max(1, epoch_samples // a.batch)
    total_steps = steps_per_epoch * a.epochs
    log(f"train rows {mixer.n_train}; epoch = {steps_per_epoch} steps x {a.batch}; {total_steps} steps; "
        f"mixture {[(corpus.sources[s.source]['name'], round(float(p), 3)) for s, p in zip(mixer.streams, mixer.p)]}")

    last = Path(str(a.out) + ".last")
    step, epoch, best, bad = 0, 0, float("inf"), 0
    best_by: dict[str, float] = {}
    if a.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        step, epoch, best, bad = ck["step"], ck["epoch"], ck["best"], ck["bad"]
        log(f"resumed from {last}: step {step} epoch {epoch} best {best:.4f}")

    def save_last() -> None:
        tmp = Path(str(last) + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step,
                    "epoch": epoch, "best": best, "bad": bad}, tmp)
        tmp.replace(last)

    pf = Prefetch(mixer.take, a.batch, a.prefetch)
    t0 = time.time()
    model.train()
    acc: dict[str, float] = defaultdict(float)
    n_acc = 0
    stop = False
    while epoch < a.epochs and not stop:
        # a resume restarts the epoch it was in at its last boundary; steps inside it are redone
        while step < (epoch + 1) * steps_per_epoch:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, a)
            b = to_tensors(pf.get(), dev)
            if cfg.numbers_only:
                b = blank_ids(b, torch.ones(a.batch, dtype=torch.bool, device=dev))
            elif cfg.id_dropout > 0:
                rows = torch.rand(len(b["z"]), device=dev) < cfg.id_dropout
                b = blank_ids(b, rows)
                b["blank"] = rows
            with autocast(dev, a.amp):
                out = model(b)
            loss, parts = compute_loss(out, b, cfg, scale, mcfg.n_move)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            n_acc += 1
            acc["loss"] += float(loss.detach())
            for k, v in parts.items():
                acc[k] += float(v)
            if step % a.log_every == 0:
                el = time.time() - t0
                log(f"step {step}/{total_steps} ep {epoch + 1} lr {opt.param_groups[0]['lr']:.2e} "
                    + " ".join(f"{k} {v / n_acc:.4f}" for k, v in acc.items()) + f" gn {float(gn):.2f} | {step * a.batch / max(el, 1e-9):.0f} smp/s")
                acc.clear()
                n_acc = 0
            if a.ckpt_every and step % a.ckpt_every == 0:
                save_last()
            if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
                stop = True
                break
        # ---- epoch end: validation
        if step >= (epoch + 1) * steps_per_epoch:
            epoch += 1
        vloss, vn = 0.0, 0
        vm: dict[int, Meter] = {}
        for i, s in enumerate(corpus.sources):
            if corpus.n(i, "val"):
                m, l, n = evaluate(model, corpus, i, "val", a.eval_max, a.batch, dev, cfg, scale, a.amp)
                vm[i] = m
                vloss += l * n
                vn += n
        vloss = vloss / vn if vn else float("nan")
        log(f"--- epoch {epoch} step {step} val weighted loss {vloss:.4f} ({(time.time() - t0) / 60:.1f} min)")
        for i, m in vm.items():
            log(f" [{corpus.sources[i]['name']}]")
            for ln in format_meter(m):
                log(ln)
        num = sum(m.s["val_bce"][0] for m in vm.values())
        den = sum(m.s["val_bce"][1] for m in vm.values())
        vbce = num / den if den else float("nan")
        crit = {"value": vbce, "loss": vloss}
        log(f"    criteria: val value BCE {vbce:.4f}, val weighted loss {vloss:.4f} (selecting on {a.select})")
        meta = {"epoch": epoch, "step": step, "val_loss": vloss, "val_value_bce": vbce,
                "args": {k: str(v) for k, v in vars(a).items()}}
        for name, path in (("value", Path(str(a.out) + ".byvalue.pt")), ("loss", Path(str(a.out) + ".byloss.pt"))):
            if crit[name] < best_by.get(name, float("inf")):
                best_by[name] = crit[name]
                model.export(path, extra=meta)
        if crit[a.select] < best:
            best, bad = crit[a.select], 0
            model.export(a.out, extra=meta)
            log(f"saved best -> {a.out}")
        else:
            bad += 1
            if a.patience and bad >= a.patience:
                log(f"early stop: no val improvement for {bad} epochs")
                stop = True
        save_last()
    if not a.out.exists():
        model.export(a.out, extra={"step": step})
    log(f"training done: {step} steps in {(time.time() - t0) / 60:.1f} min")
    if not a.no_final:
        best_model = TokenNet.load(a.out, device=dev, check_engine=False)
        final_report(best_model, corpus, a, dev, cfg, scale, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
