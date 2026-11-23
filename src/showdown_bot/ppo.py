"""PPO self-play trainer for the token transformer (bead showdown-bot-589.68.6, docs/selfplay-ppo.md).

The games run in Rust (``poke_engine.SelfPlayPool``: random-battle teams, per-side scrubbed token
views, real history rows); this module owns the policy side: batched fp16 forwards on the GPU, an
opponent pool (latest self / frozen supervised net / past checkpoints), GAE, clipped PPO with an
entropy bonus and a decaying KL penalty to the frozen anchor, checkpoints in the ``TokenNet.export``
format (usable as is by ``token-policy:model=`` and ``net-search``), and a metrics log.

    python -m showdown_bot.ppo train --init models/tok-d256.pt --out runs/ppo1 --minutes 25
    python -m showdown_bot.ppo bench --init models/tok-d256.pt --games 128 256 512
    python -m showdown_bot.ppo eval  --a runs/ppo1/latest.pt --b models/tok-d256.pt --games 400
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

N_ACT = 13
G7SEARCH = -100  # the reduced-budget gen7 search opponent (g7_share)
PACK = -101  # the same opponent in a game started from a saved position (pack_share)
SELF, ANCHOR = -1, 0  # opponent ids: -1 = the latest learner, 0 = the frozen anchor, k >= 1 past checkpoints


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


NEG_OK = -1e8  # logits above this are legal (token_net masks with -1e9)


@dataclass
class PPOConfig:
    # pool
    games: int = 512
    threads: int = 4
    seed: int = 1
    turn_cap: int = 200
    # rollout / update
    rollout_steps: int = 48  # pool steps per update
    epochs: int = 2
    minibatch: int = 512  # ~3 GB of activations at d256 L6; 1024 peaks at ~6.5 GB
    lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    grad_clip: float = 0.5
    gamma: float = 0.997
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.005
    target_kl: float = 0.03  # stop the epochs early when the sampled KL(old || new) exceeds 1.5x this
    adv_norm: bool = True
    # KL penalty to the frozen anchor (the supervised net): linear decay beta0 -> beta1
    kl_beta0: float = 0.1
    kl_beta1: float = 0.01
    kl_decay_updates: int = 300
    value_warmup: int = 8  # updates without a policy gradient (the critic must be good before the policy moves)
    warmup_kl: float = 1.0  # KL-to-anchor weight during the warm-up (the shared trunk moves with the value loss)
    # opponents
    p_self: float = 0.5
    p_anchor: float = 0.25  # the rest is drawn from past checkpoints (anchor if there are none yet)
    snapshot_every: int = 10
    max_past: int = 5
    # every distinct opponent net costs one more forward per pool step. With active_past > 0 only that many of the
    # pool's past checkpoints play at a time (re-drawn every update), so the step cost stops growing with max_past.
    active_past: int = 0
    seed_past: int = 0  # on --resume: refill the past pool from this many of the newest ckpt-*.pt files
    # io
    ckpt_every: int = 10
    amp: bool = True
    infer_batch: int = 2048
    window: int = 400  # games in the rolling win-rate windows
    # architecture of the learner: "init" keeps the checkpoint's, "d192L4" distils a smaller net first
    zero_history: bool = False  # feed padding history rows (what token-policy / net-search do today)
    arch: str = "init"
    distill_updates: int = 40
    # critic: "shared" = the actor's own value head (default); "asym" = a separate net (initialised from the anchor)
    # fed the UNSCRUBBED token view (the opponent's true items/abilities/moves/tera), trained only on the value loss
    critic: str = "shared"
    critic_lr: float = 5e-5
    kl_off_after: int = 0  # >0: the anchor KL penalty is exactly 0 from this update on (0: never switched off)
    # auxiliary loss: predict the opponent's action of the same turn (the ``opp_slot`` head; 0 = off)
    opp_coef: float = 0.0
    # schedules (2026-11-18 PPO review). All off by default; counted in updates from *_start.
    lr_final: float = 0.0  # >0: lr decays geometrically from lr to lr_final over lr_decay_updates
    lr_decay_start: int = 0
    lr_decay_updates: int = 0
    ent_final: float = -1.0  # >=0: ent_coef moves linearly to ent_final over ent_decay_updates
    ent_decay_start: int = 0
    ent_decay_updates: int = 0
    # a last minibatch smaller than this share of ``minibatch`` is dropped (it used to take a full Adam step)
    min_batch_frac: float = 0.0
    # >0: an exponential moving average of the learner's weights (per optimizer step), saved as ema-<name>.pt
    ema_decay: float = 0.0
    # fixed outside opponents, "name=path@share@temp,..." (e.g. a net imitating the gen7 search bot): each takes
    # `share` of all games, before the self / anchor / past split of the rest; temp 0 = its best move.
    # Two more optional numbers, "name=path@share@temp@sw@heal", give the opponent a playing STYLE (2026-11-22,
    # after the ladder opponent that pivots into resists and heals): `sw` is added to the logit of a switch whose
    # target would lose little to the other side's active Pokemon (full bonus at 0%, none from 33% of its HP),
    # `heal` to a recovery move in proportion to the HP its user is missing. A sixth number, `setup`
    # ("...@sw@heal@setup"), is added to a boosting move when the other side's active Pokemon threatens little (full
    # bonus at 0% of the user's HP, none from 50%): an opponent that takes every free turn to set up a sweep.
    # Only the opponent is changed; the learner gets no rule.
    extra_opps: str = ""
    # the gen7 search bot at a reduced budget as an opponent (showdown_bot.g7_sim; needs engine sdb.33+): share of
    # all games, sampled worlds and MCTS iterations per world for each of its decisions, worker processes
    g7_share: float = 0.0
    g7_worlds: int = 4
    g7_iters: int = 2000
    g7_procs: int = 10
    # training packs (needs g7_share > 0 and engine sdb.35+): this share of all games starts from a saved position
    # of a game the learner LOST to the gen7 search, taken up to pack_back of its decisions before the last one
    # where its own value was still >= pack_turn (reward units), and is played again with new chance rolls: against
    # the gen7 search with probability pack_g7, else against the ordinary opponent mix. The bank holds the newest
    # pack_size positions. No rule says what to play in them.
    # pack_resample: at the start of a pack game everything the learner has not seen of the opponent's team is
    # drawn again. The positions were picked by the result, so the hidden part of the saved game is worse for the
    # learner than what it can see implies (Jiang et al. 2022, SAMPLR: a curriculum that skews hidden factors
    # biases the policy as well as the value); redrawing it removes that.
    # pack_vmask (for pack_resample off): no value loss on the pack rows played against the gen7 search and their
    # advantages are centred on their own mean.
    pack_share: float = 0.0
    pack_size: int = 2000
    pack_back: int = 3
    pack_turn: float = 0.0
    pack_g7: float = 0.5
    pack_resample: bool = True
    pack_vmask: bool = False


# ---------------------------------------------------------------------------
# Pure pieces (unit tested)
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_idx: np.ndarray,
    bootstrap: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE over trajectories stored as a forest: ``next_idx[i]`` (> i) is the next decision of the same
    seat (-1: the trajectory ends or is cut here). ``values`` are in reward units (-1..1). A cut
    trajectory (``dones`` 0, no next) bootstraps with ``bootstrap[i]``. Returns ``(advantages, returns)``."""
    n = len(rewards)
    adv = np.zeros(n, np.float32)
    r, v, d, nx, bs = rewards.tolist(), values.tolist(), dones.tolist(), next_idx.tolist(), bootstrap.tolist()
    a = [0.0] * n
    for i in range(n - 1, -1, -1):
        if d[i]:
            nv, na = 0.0, 0.0
        elif nx[i] >= 0:
            nv, na = v[nx[i]], a[nx[i]]
        else:
            nv, na = bs[i], 0.0
        delta = r[i] + gamma * nv - v[i]
        a[i] = delta + gamma * lam * na
    adv[:] = a
    return adv, adv + values.astype(np.float32)


def masked_log_softmax(logits: Any, legal: Any) -> Any:
    import torch

    return torch.log_softmax(logits.float().masked_fill(~legal, -1e9), -1)


def ppo_terms(
    logits: Any,
    legal: Any,
    actions: Any,
    old_logp: Any,
    adv: Any,
    anchor_logits: Any | None,
    clip: float,
) -> dict[str, Any]:
    """Policy-side loss pieces on masked logits: clipped surrogate (to minimise), entropy, KL to the
    anchor (``KL(pi || anchor)`` over the legal actions), the sampled KL(old || new) and the clip fraction."""
    import torch

    logp_all = masked_log_softmax(logits, legal)
    logp = logp_all.gather(1, actions.long().unsqueeze(1)).squeeze(1)
    ratio = torch.exp(logp - old_logp)
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    p = logp_all.exp()
    ent = -(p * torch.where(legal, logp_all, torch.zeros_like(logp_all))).sum(-1)
    out = {
        "pg": -torch.min(s1, s2).mean(),
        "entropy": ent.mean(),
        "approx_kl": (old_logp - logp).mean(),
        "clipfrac": ((ratio - 1).abs() > clip).float().mean(),
        "logp": logp,
    }
    if anchor_logits is not None:
        qa = masked_log_softmax(anchor_logits, legal)
        out["kl_anchor"] = (p * torch.where(legal, logp_all - qa, torch.zeros_like(logp_all))).sum(-1).mean()
    return out


def value_target(returns: np.ndarray) -> np.ndarray:
    """Win probability target in [0, 1] from a return in [-1, 1]."""
    return np.clip((returns + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def sched_lr(update: int, cfg: PPOConfig) -> float:
    if cfg.lr_final <= 0 or cfg.lr_decay_updates <= 0:
        return cfg.lr
    f = min(max((update - cfg.lr_decay_start) / cfg.lr_decay_updates, 0.0), 1.0)
    return float(cfg.lr * (cfg.lr_final / cfg.lr) ** f)


def sched_ent(update: int, cfg: PPOConfig) -> float:
    if cfg.ent_final < 0 or cfg.ent_decay_updates <= 0:
        return cfg.ent_coef
    f = min(max((update - cfg.ent_decay_start) / cfg.ent_decay_updates, 0.0), 1.0)
    return float(cfg.ent_coef + (cfg.ent_final - cfg.ent_coef) * f)


def kl_beta(update: int, cfg: PPOConfig) -> float:
    if cfg.kl_off_after > 0 and update >= cfg.kl_off_after:
        return 0.0
    f = min(1.0, update / max(1, cfg.kl_decay_updates))
    return cfg.kl_beta0 + (cfg.kl_beta1 - cfg.kl_beta0) * f


# ---------------------------------------------------------------------------
# Observation decoding
# ---------------------------------------------------------------------------


class Obs:
    """A decoded ``SelfPlayPool.observe()``: numpy views of the byte buffers."""

    def __init__(self, raw: dict[str, Any], cat_len: int, cont_len: int) -> None:
        n = int(raw["n"])
        self.n = n
        self.game = np.frombuffer(raw["game"], dtype="<i4")
        self.side = np.frombuffer(raw["side"], dtype=np.uint8)
        self.turn = np.frombuffer(raw["turn"], dtype="<i4")
        self.cat = np.frombuffer(raw["cat"], dtype="<i2").reshape(n, cat_len)
        self.cont = np.frombuffer(raw["cont"], dtype="<f4").reshape(n, cont_len)
        self.hist_cat = np.frombuffer(raw["hist_cat"], dtype="<i2").reshape(n, 16, 4)
        self.hist_cont = np.frombuffer(raw["hist_cont"], dtype="<f4").reshape(n, 16, 5)
        self.legal = np.frombuffer(raw["legal"], dtype=np.uint8).reshape(n, 13) if "legal" in raw else None
        self.legal = np.frombuffer(raw["legal"], dtype=np.uint8).reshape(n, N_ACT) if "legal" in raw else None


class TokenBatcher:
    """Turns raw ``cat`` / ``cont`` rows (numpy or torch) into the dict of tensors ``TokenNet`` consumes."""

    def __init__(self, device: str, zero_history: bool = False) -> None:
        import torch

        from showdown_bot import token_net, tokens

        self.torch = torch
        self.zero_history = zero_history
        self.device = torch.device(device)
        lay = tokens.layout()
        self.fields = {k: v for k, v in lay.items() if k in token_net.INPUT_FIELDS}
        self.cat_len = max(f["offset"] + int(np.prod(f["shape"])) for f in lay.values() if f["buffer"] == "cat")
        self.cont_len = max(f["offset"] + int(np.prod(f["shape"])) for f in lay.values() if f["buffer"] == "cont")
        self.legal_off = lay["act_legal"]["offset"]

    def to_device(self, cat: Any, cont: Any, hist_cat: Any, hist_cont: Any) -> dict[str, Any]:
        t, dev = self.torch, self.device

        def up(x: Any, dt: Any) -> Any:
            x = x if isinstance(x, t.Tensor) else t.from_numpy(np.ascontiguousarray(x))
            return x.to(dev, non_blocking=True) if dt is None else x.to(dev, non_blocking=True).to(dt)

        c = up(cat, None)
        f = up(cont, t.float16)  # the net is trained on float16-rounded inputs (shards); do the same here
        b: dict[str, Any] = {}
        n = c.shape[0]
        for name, fd in self.fields.items():
            src = c if fd["buffer"] == "cat" else f
            size = int(np.prod(fd["shape"]))
            b[name] = src[:, fd["offset"] : fd["offset"] + size].reshape(n, *fd["shape"])
        b["hist_cat"] = up(hist_cat, None)
        b["hist_cont"] = up(hist_cont, t.float16)
        if self.zero_history:  # what the live bots feed today (padding rows)
            b["hist_cat"] = t.zeros_like(b["hist_cat"])
            b["hist_cont"] = t.zeros_like(b["hist_cont"])
        return b


def legal_mask(batch: dict[str, Any]) -> Any:
    return batch["act_legal"][:, 0] > 0


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def load_net(path: str | Path, device: str, dropout_off: bool = True) -> Any:
    from showdown_bot.token_net import TokenNet

    net = TokenNet.load(path, device=device)
    if dropout_off:
        zero_dropout(net)
    return net


def zero_dropout(net: Any) -> None:
    import torch.nn as nn

    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if hasattr(m, "p") and m.__class__.__name__ == "Block":
            m.p = 0.0


def small_net(teacher: Any, arch: str) -> Any:
    """A fresh smaller net with the teacher's token spec: ``d<D>L<N>[h<H>]``, e.g. ``d192L4`` = d_model 192,
    4 layers, 8 heads (default heads: 8 when it divides D, else 4)."""
    import re

    from showdown_bot.token_net import TokenNet

    m = re.fullmatch(r"d(\d+)L(\d+)(?:h(\d+))?", arch)
    if not m:
        raise ValueError(f"unknown arch {arch!r}")
    d, layers = int(m.group(1)), int(m.group(2))
    heads = int(m.group(3)) if m.group(3) else (8 if d % 8 == 0 and d >= 128 else 4)
    if d % heads:
        raise ValueError(f"arch {arch!r}: d_model {d} is not divisible by {heads} heads")
    cfg = dataclasses.replace(teacher.cfg, d_model=d, n_layers=layers, n_heads=heads, dropout=0.0)
    return TokenNet(cfg).to(next(teacher.parameters()).device)


@dataclass
class Forward:
    actions: np.ndarray
    logp: np.ndarray
    value: np.ndarray  # reward units, -1..1
    entropy: np.ndarray


class Policy:
    """Batched sampling from a TokenNet (fp16 autocast, no grad)."""

    def __init__(self, net: Any, device: str, amp: bool = True, max_batch: int = 2048) -> None:
        import torch

        self.torch = torch
        self.net = net.eval()
        self.device = torch.device(device)
        self.amp = amp and self.device.type == "cuda"
        self.max_batch = max_batch
        self.temp = 1.0  # sampling temperature (fixed outside opponents only); 0 = argmax
        self.style_sw = 0.0  # logit bonus for switching to something that resists (extra_opps, see PPOConfig)
        self.style_heal = 0.0  # logit bonus for recovery moves when hurt
        self.style_setup = 0.0  # logit bonus for boosting moves when the other side threatens little
        self.style_enc = 0.0  # logit bonus for the move Encore whenever it is legal
        self._enc_id = None

    def value(self, batch: dict[str, Any]) -> np.ndarray:
        """Value head only, in reward units (-1..1)."""
        torch = self.torch
        n = batch["mon_cat"].shape[0]
        out = []
        for s in range(0, n, self.max_batch):
            sub = {k: v[s : s + self.max_batch] for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                o = self.net(sub)
            out.append((2.0 * torch.sigmoid(o["value"].float()) - 1.0).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def forward(self, batch: dict[str, Any], greedy: bool = False, probs_out: bool = False) -> Any:
        torch = self.torch
        n = batch["mon_cat"].shape[0]
        acts, logps, vals, ents, probs = [], [], [], [], []
        for s in range(0, n, self.max_batch):
            sub = {k: v[s : s + self.max_batch] for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                o = self.net(sub)
            legal = legal_mask(sub)
            logits = o["policy"]
            if self.style_sw or self.style_heal or self.style_setup or self.style_enc:
                # action features of the acting side (engine/src/prior.rs): 46 recovery share of a move, 47 its
                # boost stages / 6, 54 the best damage the other active does to a switch target (share of its
                # HP / 1.5), 56 the same against our own active, 57 own active's HP
                logits = logits.float().clone()
                ac = sub["act_cont"][:, 0].float()
                if self.style_sw:
                    logits[:, 8:] += self.style_sw * (1.0 - ac[:, 8:, 54] * 1.5 / 0.33).clamp(0.0, 1.0)
                if self.style_heal:
                    logits[:, :8] += self.style_heal * (ac[:, :8, 46] > 0).float() * (1.0 - ac[:, :8, 57]).clamp(0.0, 1.0)
                if self.style_setup:
                    logits[:, :8] += self.style_setup * (ac[:, :8, 47] > 0).float() * (1.0 - ac[:, :8, 56] * 1.5 / 0.5).clamp(0.0, 1.0)
                if self.style_enc:
                    if self._enc_id is None:
                        import poke_engine
                        self._enc_id = [str(x).lower() for x in poke_engine.identity_vocab(3)].index("encore")
                    logits[:, :8] += self.style_enc * (sub["act_cat"][:, 0, :8, 0] == self._enc_id).float()
            if self.temp > 0 and self.temp != 1.0:
                logits = logits.float() / self.temp
            lp = masked_log_softmax(logits, legal)
            p = lp.exp()
            a = p.argmax(-1) if (greedy or self.temp <= 0) else torch.multinomial(p, 1).squeeze(1)
            acts.append(a)
            logps.append(lp.gather(1, a.unsqueeze(1)).squeeze(1))
            vals.append(2.0 * torch.sigmoid(o["value"].float()) - 1.0)
            ents.append(-(p * torch.where(legal, lp, torch.zeros_like(lp))).sum(-1))
            if probs_out:
                probs.append(p)
        cat = lambda xs: torch.cat(xs) if len(xs) > 1 else xs[0]  # noqa: E731
        res = Forward(
            actions=cat(acts).to(torch.uint8).cpu().numpy(),
            logp=cat(logps).cpu().numpy(),
            value=cat(vals).cpu().numpy(),
            entropy=cat(ents).cpu().numpy(),
        )
        return (res, cat(probs)) if probs_out else res


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@dataclass
class Rollout:
    cat: np.ndarray
    cont: np.ndarray  # float16
    hist_cat: np.ndarray
    hist_cont: np.ndarray  # float16
    actions: np.ndarray
    logp: np.ndarray
    value: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_idx: np.ndarray
    bootstrap: np.ndarray
    seat: np.ndarray
    opp: np.ndarray
    stats: dict[str, float] = field(default_factory=dict)
    opp_act: np.ndarray | None = None  # int16: the opponent's action slot of the same pool step (-1: it had no decision)
    cat_t: np.ndarray | None = None  # unscrubbed (critic) token view, asymmetric critic only
    cont_t: np.ndarray | None = None  # float16

    def __len__(self) -> int:
        return len(self.actions)


class Arena:
    """The pool plus the policy/opponent bookkeeping that turns it into PPO rollouts."""

    def __init__(self, cfg: PPOConfig, learner: Any, anchor: Any, device: str, critic: Any = None) -> None:
        import poke_engine

        self.cfg = cfg
        self.device = device
        self.pool = poke_engine.SelfPlayPool(cfg.games, cfg.seed, cfg.threads, cfg.turn_cap)
        self.batcher = TokenBatcher(device, cfg.zero_history)
        self.rng = random.Random(cfg.seed + 17)
        self.learner = Policy(learner, device, cfg.amp, cfg.infer_batch)
        self.critic = Policy(critic, device, cfg.amp, cfg.infer_batch) if critic is not None else None
        self.opponents: dict[int, Policy] = {ANCHOR: Policy(anchor, device, cfg.amp, cfg.infer_batch)}
        self.extra: list[tuple[int, str, float]] = []  # (opponent id <= -2, name, share of all games)
        for k, item in enumerate(x for x in cfg.extra_opps.split(",") if x.strip()):
            name, rest = item.strip().split("=", 1)
            parts = rest.split("@")
            pol = Policy(load_net(parts[0], device), device, cfg.amp, cfg.infer_batch)
            pol.temp = float(parts[2]) if len(parts) > 2 else 1.0
            pol.style_sw = float(parts[3]) if len(parts) > 3 else 0.0
            pol.style_heal = float(parts[4]) if len(parts) > 4 else 0.0
            pol.style_setup = float(parts[5]) if len(parts) > 5 else 0.0
            pol.style_enc = float(parts[6]) if len(parts) > 6 else 0.0
            self.opponents[-2 - k] = pol
            self.extra.append((-2 - k, name, float(parts[1]) if len(parts) > 1 else 0.1))
        self.g7 = None
        if cfg.g7_share > 0:
            from concurrent.futures import ProcessPoolExecutor

            from showdown_bot import g7_sim

            ex = ProcessPoolExecutor(cfg.g7_procs, initializer=g7_sim._init_worker,
                                     initargs=(str(g7_sim.EVALUATOR), str(g7_sim.PRIOR), g7_sim.LEAF_SCALE))
            self.g7 = g7_sim.Gen7(ex, cfg.g7_procs, cfg.g7_worlds, cfg.g7_iters, 0, cfg.seed)
            self.extra.append((G7SEARCH, "g7", cfg.g7_share))
        self.time_g7 = 0.0
        if cfg.pack_share > 0 and self.g7 is None:
            raise ValueError("pack_share needs g7_share > 0")
        self.bank: deque[tuple[int, int]] = deque()  # (saved position id, learner's seat), oldest first
        self.cand: dict[int, list[tuple[int, float]]] = {}  # game -> (position id, learner's value) per decision
        self.pack_started = 0
        self.pack_added = 0
        self.pack_decisions = 0
        self.is_pack = np.zeros(cfg.games, bool)  # the game in this slot started from a saved position
        self.pack_new: set[int] = set()  # pack games whose first learner decision has not been seen yet
        self.pack_v0: deque[float] = deque(maxlen=cfg.window)  # learner's win chance at the start of gen7 pack games
        self.past_ids: list[int] = []
        self.active_ids: list[int] = []  # the past checkpoints in play this update (active_past > 0)
        self._next_id = 1
        n = cfg.games
        self.opp = np.full(n, SELF, np.int32)
        self.lseat = np.full(n, -1, np.int8)  # learner's seat in games against an opponent (0/1); -1: both seats
        for g in range(n):
            self._assign(g)
        self.open: dict[tuple[int, int], int] = {}
        self.results = {k: deque(maxlen=cfg.window) for k in ("anchor", "past", "self", "pack", *(e[1] for e in self.extra))}
        self.extra_name = {e[0]: e[1] for e in self.extra}
        self.extra_name[PACK] = "pack"
        self.games_done = 0
        self.decisions = 0
        self.turns_done = 0
        self.draws = 0
        self.time_infer = 0.0
        self.time_pool = 0.0

    # --- opponents
    def _assign(self, g: int) -> None:
        c = self.cfg
        if self.extra:
            r = self.rng.random()
            for oid, _, share in self.extra:
                if r < share:
                    self.opp[g], self.lseat[g] = oid, self.rng.randrange(2)
                    return
                r -= share
        r = self.rng.random()
        if r < c.p_self:
            self.opp[g], self.lseat[g] = SELF, -1
            return
        past = self.past_ids
        if r < c.p_self + c.p_anchor or not past:
            self.opp[g] = ANCHOR
        else:
            self.opp[g] = self.rng.choice(self.active_ids or past)
        self.lseat[g] = self.rng.randrange(2)

    def rotate_active(self) -> None:
        """Re-draws which past checkpoints are in play (``active_past``). Games in progress against a checkpoint
        that leaves play continue against one that stays, as they do when a snapshot is dropped."""
        k = self.cfg.active_past
        if k <= 0 or len(self.past_ids) <= k:
            self.active_ids = []
            return
        self.active_ids = self.rng.sample(self.past_ids, k)
        act = np.asarray(self.active_ids, np.int32)
        stale = np.nonzero((self.opp > ANCHOR) & ~np.isin(self.opp, act))[0]
        if len(stale):
            self.opp[stale] = act[np.asarray([self.rng.randrange(k) for _ in range(len(stale))])]

    def add_snapshot(self, net: Any) -> int:
        """Freezes a copy of ``net`` as a new past opponent (dropping a random old one when full)."""
        pid = self._next_id
        self._next_id += 1
        copy_net = copy.deepcopy(net).eval()
        for p in copy_net.parameters():
            p.requires_grad_(False)
        if len(self.past_ids) >= self.cfg.max_past:
            drop = self.rng.choice(self.past_ids)
            self.past_ids.remove(drop)
            del self.opponents[drop]
            if drop in self.active_ids:
                self.active_ids[self.active_ids.index(drop)] = pid
            # games still playing the dropped snapshot continue against the new one
            self.opp[self.opp == drop] = pid
        self.opponents[pid] = Policy(copy_net, self.device, self.cfg.amp, self.cfg.infer_batch)
        self.past_ids.append(pid)
        return pid

    # --- privileged critic view
    def truth_tokens(self, obs: Obs, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Token buffers of ``rows`` from the TRUE game state (nothing scrubbed): same encoding and layout as
        the actor's view, but the opponent's items, abilities, moves, Tera type and bench are real.
        Python-only path (``game_state`` + ``state_tokens_str``)."""
        import poke_engine

        cat = np.zeros((len(rows), self.batcher.cat_len), np.int16)
        cont = np.zeros((len(rows), self.batcher.cont_len), np.float16)
        cache: dict[int, str] = {}
        for j, r in enumerate(rows):
            g = int(obs.game[r])
            st = cache.get(g)
            if st is None:
                st = cache[g] = self.pool.game_state(g)
            c, f, _ = poke_engine.state_tokens_str(st, int(obs.side[r]) + 1, None)
            cat[j] = np.frombuffer(c, "<i2")
            cont[j] = np.frombuffer(f, "<f4").astype(np.float16)
        return cat, cont

    def critic_values(self, obs: Obs, rows: np.ndarray, truth: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        cat_t, cont_t = truth if truth is not None else self.truth_tokens(obs, rows)
        b = self.batcher.to_device(cat_t, cont_t, obs.hist_cat[rows], obs.hist_cont[rows])
        return self.critic.value(b)

    # --- one pool step
    def _decode(self) -> Obs:
        return Obs(self.pool.observe(), self.batcher.cat_len, self.batcher.cont_len)

    def step(self, store: dict[str, list] | None) -> int:
        """Observe, choose actions for every row, step the pool. Learner decisions go to ``store``."""
        import torch

        t0 = time.perf_counter()
        obs = self._decode()
        n = obs.n
        if n == 0:
            raise RuntimeError("the pool has no decisions to make")
        opp_of_row = self.opp[obs.game]
        is_learner = (opp_of_row == SELF) | (obs.side == self.lseat[obs.game])
        batch = self.batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont)
        actions = np.zeros(n, np.uint8)
        logp = np.zeros(n, np.float32)
        value = np.zeros(n, np.float32)
        t1 = time.perf_counter()
        groups: list[tuple[Policy, np.ndarray]] = []
        li = np.nonzero(is_learner)[0]
        if len(li):
            groups.append((self.learner, li))
        gi = None
        for oid in np.unique(opp_of_row[~is_learner]):
            idx = np.nonzero((opp_of_row == oid) & ~is_learner)[0]
            if int(oid) in (G7SEARCH, PACK):
                gi = idx if gi is None else np.concatenate([gi, idx])
                continue
            groups.append((self.opponents[int(oid)], idx))
        if gi is not None:  # CPU search (worker processes); blocks this step until every decision is searched
            tg = time.perf_counter()
            actions[gi] = self.g7.act(self.pool, obs.game[gi].tolist(), obs.side[gi].tolist(), obs.legal[gi])
            self.time_g7 += time.perf_counter() - tg
        for pol, idx in groups:
            sel = torch.from_numpy(idx).to(self.batcher.device)
            sub = {k: v[sel] for k, v in batch.items()}
            f = pol.forward(sub)
            actions[idx], logp[idx], value[idx] = f.actions, f.logp, f.value
        truth = None
        if store is not None and len(li) and self.critic is not None:
            # the critic's view must come from the state the action was chosen in, i.e. BEFORE pool.step
            # (afterwards game_state is the next state, or a fresh game when this one just ended)
            truth = self.truth_tokens(obs, li)
            value[li] = self.critic_values(obs, li, truth)
        if self.cfg.pack_share > 0 and len(li):
            self.pack_decisions += int(self.is_pack[obs.game[li]].sum())
            if self.pack_new:
                for r in li:
                    g = int(obs.game[r])
                    if g in self.pack_new:
                        self.pack_new.discard(g)
                        if opp_of_row[r] == PACK:
                            self.pack_v0.append(0.5 * (float(value[r]) + 1.0))
            # save every learner decision of an ordinary game against the gen7 search; which one is kept (if any)
            # is decided when the game ends
            sv = li[opp_of_row[li] == G7SEARCH]
            if len(sv):
                ids = self.pool.snap(obs.game[sv].tolist())
                for r, i in zip(sv, ids):
                    self.cand.setdefault(int(obs.game[r]), []).append((int(i), float(value[r])))
        t2 = time.perf_counter()
        _, _, finished = self.pool.step(actions.tobytes())
        t3 = time.perf_counter()
        self.time_infer += t2 - t1
        self.time_pool += (t1 - t0) + (t3 - t2)
        self.decisions += n

        if store is not None and len(li):
            base = sum(len(a) for a in store["actions"])
            if truth is not None:
                store["cat_t"].append(truth[0])
                store["cont_t"].append(truth[1])
            store["cat"].append(obs.cat[li].copy())
            store["cont"].append(obs.cont[li].astype(np.float16))
            store["hist_cat"].append(obs.hist_cat[li].copy())
            store["hist_cont"].append(obs.hist_cont[li].astype(np.float16))
            store["actions"].append(actions[li].copy())
            store["logp"].append(logp[li].copy())
            store["value"].append(value[li].copy())
            store["seat"].append(obs.side[li].copy())
            store["opp"].append(opp_of_row[li].copy())
            # the other seat's action in this step (its own slot numbering = the opponent action tokens' order)
            row_of = np.full(2 * int(obs.game.max()) + 2, -1, np.int64)
            row_of[2 * obs.game.astype(np.int64) + obs.side] = np.arange(n)
            other = row_of[2 * obs.game[li].astype(np.int64) + (1 - obs.side[li].astype(np.int64))]
            store["opp_act"].append(np.where(other >= 0, actions[np.maximum(other, 0)].astype(np.int16), -1).astype(np.int16))
            nli = len(li)
            store["reward"].append(np.zeros(nli, np.float32))
            store["done"].append(np.zeros(nli, np.uint8))
            nxt = np.full(nli, -1, np.int64)
            store["next"].append(nxt)
            for j in range(nli):
                key = (int(obs.game[li[j]]), int(obs.side[li[j]]))
                prev = self.open.get(key)
                if prev is not None:
                    store["_next_flat"][prev] = base + j
                self.open[key] = base + j
            store["_n"] = base + nli
        # finished games: close both seats' trajectories with the result, then reassign opponents
        loads: list[tuple[int, int, int]] = []
        drops: list[int] = []  # dropped after the loads: a position picked for loading may be pushed out of the bank in the same step
        for g, w, turns in finished:
            self.games_done += 1
            self.turns_done += turns
            if w == 0:
                self.draws += 1
            for seat in (0, 1):
                i = self.open.pop((g, seat), None)
                if i is not None and store is not None:
                    store["_reward_flat"][i] = 0.0 if w == 0 else (1.0 if w == seat + 1 else -1.0)
                    store["_done_flat"][i] = 1
            o = int(self.opp[g])
            if self.is_pack[g]:
                # pack games count only in their own tally (they start from positions the learner went on to lose)
                if o == PACK:
                    ls = int(self.lseat[g])
                    self.results["pack"].append(0.5 if w == 0 else (1.0 if w == ls + 1 else 0.0))
                self.is_pack[g] = False
                self.pack_new.discard(g)
            elif o != SELF:
                ls = int(self.lseat[g])
                score = 0.5 if w == 0 else (1.0 if w == ls + 1 else 0.0)
                self.results[self.extra_name.get(o) or ("anchor" if o == ANCHOR else "past")].append(score)
            else:
                self.results["self"].append(0.5 if w == 0 else float(w == 1))
            if self.cfg.pack_share > 0:
                cands = self.cand.pop(g, None)
                if cands:
                    keep = -1
                    if o == G7SEARCH and w != 0 and w != int(self.lseat[g]) + 1:
                        ok = [k for k, (_, v) in enumerate(cands) if v >= self.cfg.pack_turn]
                        if ok:
                            keep = self.rng.randint(max(0, ok[-1] - self.cfg.pack_back), ok[-1])
                            self.bank.append((cands[keep][0], int(self.lseat[g])))
                            self.pack_added += 1
                    drop = [i for k, (i, _) in enumerate(cands) if k != keep]
                    while len(self.bank) > self.cfg.pack_size:
                        drop.append(self.bank.popleft()[0])
                    drops.extend(drop)
                if self.bank and self.rng.random() < self.cfg.pack_share:
                    pid, seat = self.bank[self.rng.randrange(len(self.bank))]
                    if self.rng.random() < self.cfg.pack_g7:
                        self.opp[g] = PACK
                    else:
                        self._assign(g)
                        if self.opp[g] == G7SEARCH:
                            self.opp[g] = PACK
                    if self.opp[g] != SELF:
                        self.lseat[g] = seat
                    self.is_pack[g] = True
                    self.pack_new.add(g)
                    loads.append((g, pid, seat))
                    continue
            self._assign(g)
        if loads:
            # the pool has already dealt a new game into each finished slot; put the saved position there instead
            self.pack_started += len(loads)
            self.pool.load([x[0] for x in loads], [x[1] for x in loads], self.pack_started,
                           [x[2] if self.cfg.pack_resample else -1 for x in loads])
        if drops:
            self.pool.drop_snaps(drops)
        return n

    # --- a full rollout
    def rollout(self, steps: int) -> Rollout:
        store: dict[str, Any] = {k: [] for k in ("cat", "cont", "hist_cat", "hist_cont", "actions", "logp", "value", "seat", "opp", "reward", "done", "next", "opp_act", "cat_t", "cont_t")}
        store["_n"] = 0
        store["_next_flat"] = _Flat()
        store["_reward_flat"] = _Flat()
        store["_done_flat"] = _Flat()
        self.open = {}
        t0 = time.perf_counter()
        d0, g0 = self.decisions, self.games_done
        for _ in range(steps):
            self.step(store)
        # bootstrap values of the cut trajectories from their next observation
        n = store["_n"]
        reward = np.zeros(n, np.float32)
        done = np.zeros(n, np.uint8)
        nxt = np.full(n, -1, np.int64)
        for i, v in store["_reward_flat"].items():
            reward[i] = v
        for i, v in store["_done_flat"].items():
            done[i] = v
        for i, v in store["_next_flat"].items():
            nxt[i] = v
        value = np.concatenate(store["value"]) if n else np.zeros(0, np.float32)
        boot = value.copy()  # fallback: the transition's own value
        if self.open:
            obs = self._decode()
            want = dict(self.open)
            is_l = (self.opp[obs.game] == SELF) | (obs.side == self.lseat[obs.game])
            rows = [r for r in range(obs.n) if is_l[r] and (int(obs.game[r]), int(obs.side[r])) in want]
            if rows:
                idx = np.array(rows)
                batch = self.batcher.to_device(obs.cat[idx], obs.cont[idx], obs.hist_cat[idx], obs.hist_cont[idx])
                bv = self.critic_values(obs, idx) if self.critic is not None else self.learner.forward(batch).value
                for j, r in enumerate(rows):
                    boot[want[(int(obs.game[r]), int(obs.side[r]))]] = bv[j]
        wall = time.perf_counter() - t0
        ro = Rollout(
            cat=np.concatenate(store["cat"]),
            cont=np.concatenate(store["cont"]),
            hist_cat=np.concatenate(store["hist_cat"]),
            hist_cont=np.concatenate(store["hist_cont"]),
            actions=np.concatenate(store["actions"]),
            logp=np.concatenate(store["logp"]),
            value=value,
            reward=reward,
            done=done,
            next_idx=nxt,
            bootstrap=boot,
            seat=np.concatenate(store["seat"]),
            opp=np.concatenate(store["opp"]),
            opp_act=np.concatenate(store["opp_act"]),
            cat_t=np.concatenate(store["cat_t"]) if store["cat_t"] else None,
            cont_t=np.concatenate(store["cont_t"]) if store["cont_t"] else None,
        )
        ro.stats = {
            "rollout_s": wall,
            "decisions": float(self.decisions - d0),
            "games": float(self.games_done - g0),
            "decisions_per_s": (self.decisions - d0) / max(wall, 1e-9),
        }
        self.open = {}
        return ro


class _Flat(dict):
    """dict used as a sparse array (index -> value)."""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class Trainer:
    def __init__(self, cfg: PPOConfig, learner: Any, anchor: Any, device: str, out: Path, critic: Any = None) -> None:
        import torch

        if cfg.critic not in ("shared", "asym"):
            raise ValueError(f"critic must be 'shared' or 'asym', got {cfg.critic!r}")
        self.torch = torch
        self.cfg = cfg
        self.device = device
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.learner = learner.to(device).train()
        self.anchor = anchor.to(device).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        params = [p for p in self.learner.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
        self.critic = None
        if cfg.critic == "asym":
            # a separate trunk + value head on the unscrubbed view, starting from the anchor's weights
            self.critic = (critic if critic is not None else copy.deepcopy(anchor)).to(device).train()
            for p in self.critic.parameters():
                p.requires_grad_(True)
            zero_dropout(self.critic)
            self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
            self.scaler_c = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
        self.arena = Arena(cfg, learner, anchor, device, self.critic)
        self.ema = None
        if cfg.ema_decay > 0:
            self.ema = copy.deepcopy(self.learner).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)
        self.update = 0
        self.log_path = out / "metrics.jsonl"
        self.t_start = time.perf_counter()
        self.total_decisions = 0

    # --- checkpoints
    def save(self, name: str) -> Path:
        path = self.out / name
        self.learner.export(path, extra={"ppo": dataclasses.asdict(self.cfg), "update": self.update, "decisions": self.total_decisions})
        if self.ema is not None:
            self.ema.export(self.out / ("ema-" + name), extra={"ema_of": name, "ema_decay": self.cfg.ema_decay, "update": self.update})
        return path

    def save_state(self) -> None:
        torch = self.torch
        st = {"opt": self.opt.state_dict(), "update": self.update, "decisions": self.total_decisions}
        if self.critic is not None:
            # the critic is NOT part of latest.pt (that stays an actor-only TokenNet.export for token-policy / net-search)
            self.critic.export(self.out / "critic.pt", extra={"update": self.update, "role": "asym-critic"})
            st["opt_c"] = self.opt_c.state_dict()
        torch.save(st, self.out / "state.pt")

    def resume(self) -> None:
        torch = self.torch
        st = torch.load(self.out / "state.pt", map_location="cpu", weights_only=False)
        self.opt.load_state_dict(st["opt"])
        if self.critic is not None and (self.out / "critic.pt").exists():
            blob = torch.load(self.out / "critic.pt", map_location="cpu", weights_only=False)
            self.critic.load_state_dict(blob["state_dict"])
            if "opt_c" in st:
                self.opt_c.load_state_dict(st["opt_c"])
        self.update = int(st["update"])
        self.total_decisions = int(st["decisions"])
        if self.ema is not None and (self.out / "ema-latest.pt").exists():
            blob = torch.load(self.out / "ema-latest.pt", map_location="cpu", weights_only=False)
            self.ema.load_state_dict(blob["state_dict"])

    # --- one update
    def train_step(self) -> dict[str, float]:
        torch = self.torch
        cfg = self.cfg
        self.learner.eval()  # rollouts use eval mode (dropout is zero anyway)
        self.arena.rotate_active()
        ro = self.arena.rollout(cfg.rollout_steps)
        n = len(ro)
        self.total_decisions += int(ro.stats["decisions"])
        adv, ret = compute_gae(ro.reward, ro.value, ro.done, ro.next_idx, ro.bootstrap, cfg.gamma, cfg.lam)
        y = value_target(ret)
        warm = self.update < cfg.value_warmup
        asym = self.critic is not None
        beta = kl_beta(self.update, cfg)
        lr_now = sched_lr(self.update, cfg)
        ent_now = sched_ent(self.update, cfg)
        for g in self.opt.param_groups:
            g["lr"] = lr_now
        dev = self.arena.batcher.device
        is_pack = ro.opp == PACK
        vmask = cfg.pack_share > 0 and cfg.pack_vmask and bool(is_pack.any()) and not bool(is_pack.all())
        if vmask:
            adv = adv.copy()
            adv[is_pack] -= adv[is_pack].mean() - adv[~is_pack].mean()
        vw_t = torch.from_numpy((~is_pack).astype(np.float32)).to(dev) if vmask else None
        adv_t = torch.from_numpy(adv).to(dev)
        if cfg.adv_norm and n > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)
        y_t = torch.from_numpy(y).to(dev)
        act_t = torch.from_numpy(ro.actions).to(dev)
        logp_t = torch.from_numpy(ro.logp).to(dev)
        use_opp = cfg.opp_coef > 0 and ro.opp_act is not None
        opp_t = torch.from_numpy(ro.opp_act.astype(np.int64)).to(dev) if use_opp else None
        need_anchor = warm or beta > 0  # the anchor forward is skipped once its KL weight is 0
        self.learner.train()
        zero_dropout(self.learner)
        t0 = time.perf_counter()
        agg: dict[str, float] = {}
        cnt = 0
        stop = False
        for ep in range(cfg.epochs):
            perm = np.random.permutation(n)
            for s in range(0, n, cfg.minibatch):
                idx = np.sort(perm[s : s + cfg.minibatch])
                if len(idx) < 8 or (s > 0 and len(idx) < cfg.min_batch_frac * cfg.minibatch):
                    continue
                b = self.arena.batcher.to_device(ro.cat[idx], ro.cont[idx], ro.hist_cat[idx], ro.hist_cont[idx])
                it = torch.from_numpy(idx).to(dev)
                amp_on = cfg.amp and dev.type == "cuda"
                if asym:
                    # separate critic on the unscrubbed view: value loss only; the actor never sees it
                    bt = self.arena.batcher.to_device(ro.cat_t[idx], ro.cont_t[idx], ro.hist_cat[idx], ro.hist_cont[idx])
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_on):
                        co = self.critic(bt)
                    vl = torch.nn.functional.binary_cross_entropy_with_logits(co["value"].float(), y_t[it])
                    if torch.isfinite(vl):
                        self.opt_c.zero_grad(set_to_none=True)
                        self.scaler_c.scale(vl).backward()
                        self.scaler_c.unscale_(self.opt_c)
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
                        self.scaler_c.step(self.opt_c)
                        self.scaler_c.update()
                        agg["value_loss"] = agg.get("value_loss", 0.0) + float(vl.detach())
                    del co, bt
                    if warm:  # critic-only warm-up: the policy does not move at all
                        cnt += 1
                        continue
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp_on):
                    o = self.learner(b)
                    if need_anchor:
                        with torch.no_grad():
                            ao = self.anchor(b)
                legal = legal_mask(b)
                terms = ppo_terms(o["policy"], legal, act_t[it], logp_t[it], adv_t[it],
                                  ao["policy"] if need_anchor else o["policy"].detach(), cfg.clip)
                opp_l = None
                if use_opp:
                    tgt = opp_t[it]
                    ol = o["opp_slot"].float()
                    # rows whose opponent decided this turn and whose action is legal in our view of it
                    ok = (tgt >= 0) & (ol.gather(1, tgt.clamp(min=0).unsqueeze(1)).squeeze(1) > NEG_OK)
                    if bool(ok.any()):
                        opp_l = torch.nn.functional.cross_entropy(ol[ok], tgt[ok])
                        terms["opp_nll"] = opp_l.detach()
                        terms["opp_top1"] = (ol[ok].argmax(-1) == tgt[ok]).float().mean()
                if asym:
                    vl = torch.zeros((), device=dev)  # logged above
                    loss = terms["pg"] - ent_now * terms["entropy"] + beta * terms["kl_anchor"]
                else:
                    if vw_t is not None:
                        vw = vw_t[it]
                        vl = (torch.nn.functional.binary_cross_entropy_with_logits(o["value"].float(), y_t[it], reduction="none") * vw).sum() / vw.sum().clamp(min=1.0)
                    else:
                        vl = torch.nn.functional.binary_cross_entropy_with_logits(o["value"].float(), y_t[it])
                    if warm:
                        # value-only warm-up: the shared trunk moves with the value loss, so keep the policy on the anchor
                        loss = cfg.vf_coef * vl + cfg.warmup_kl * terms["kl_anchor"]
                    else:
                        loss = terms["pg"] - ent_now * terms["entropy"] + beta * terms["kl_anchor"] + cfg.vf_coef * vl
                if opp_l is not None:
                    loss = loss + cfg.opp_coef * opp_l
                if not torch.isfinite(loss):
                    logger.warning("non-finite loss; skipping the minibatch")
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(self.learner.parameters(), cfg.grad_clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                if self.ema is not None:
                    with torch.no_grad():
                        for pe, pl in zip(self.ema.parameters(), self.learner.parameters()):
                            pe.lerp_(pl.detach(), 1.0 - cfg.ema_decay)
                cnt += 1
                for k, v in (("loss", loss), ("pg", terms["pg"]), ("value_loss", vl), ("entropy", terms["entropy"]),
                             ("kl_anchor", terms["kl_anchor"]), ("approx_kl", terms["approx_kl"]),
                             ("clipfrac", terms["clipfrac"]), ("grad_norm", gn)):
                    agg[k] = agg.get(k, 0.0) + float(v.detach())
                if "opp_nll" in terms:
                    agg["opp_nll"] = agg.get("opp_nll", 0.0) + float(terms["opp_nll"])
                    agg["opp_top1"] = agg.get("opp_top1", 0.0) + float(terms["opp_top1"])
                if not warm and float(terms["approx_kl"].detach()) > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        train_s = time.perf_counter() - t0
        self.learner.eval()
        self.update += 1
        m = {k: v / max(cnt, 1) for k, v in agg.items()}
        res = self.arena.results
        ev = float(1.0 - np.var(y - (ro.value + 1) * 0.5) / (np.var(y) + 1e-8))
        m.update(
            update=self.update,
            samples=n,
            minibatches=cnt,
            beta=beta,
            lr=lr_now,
            ent_coef=ent_now,
            warmup=int(warm),
            asym=int(asym),
            explained_var=ev,
            train_s=train_s,
            rollout_s=ro.stats["rollout_s"],
            decisions_per_s=ro.stats["decisions_per_s"],
            e2e_decisions_per_s=ro.stats["decisions"] / max(ro.stats["rollout_s"] + train_s, 1e-9),
            games_per_s=ro.stats["games"] / max(ro.stats["rollout_s"], 1e-9),
            e2e_games_per_s=ro.stats["games"] / max(ro.stats["rollout_s"] + train_s, 1e-9),
            games_done=self.arena.games_done,
            mean_turns=self.arena.turns_done / max(self.arena.games_done, 1),
            draw_rate=self.arena.draws / max(self.arena.games_done, 1),
            wr_anchor=float(np.mean(res["anchor"])) if res["anchor"] else float("nan"),
            n_anchor=len(res["anchor"]),
            wr_past=float(np.mean(res["past"])) if res["past"] else float("nan"),
            wr_self_p1=float(np.mean(res["self"])) if res["self"] else float("nan"),
            **{f"wr_x_{nm}": (float(np.mean(res[nm])) if res[nm] else float("nan")) for _, nm, _ in self.arena.extra},
            adv_mean=float(adv.mean()),
            **({"wr_x_pack": float(np.mean(res["pack"])) if res["pack"] else float("nan"), "pack_bank": len(self.arena.bank),
                "pack_v0": float(np.mean(self.arena.pack_v0)) if self.arena.pack_v0 else float("nan"),
                "pack_rows": self.arena.pack_decisions / max(self.arena.decisions, 1), "pack_started": self.arena.pack_started,
                "pack_added": self.arena.pack_added} if cfg.pack_share > 0 else {}),
            wall_min=(time.perf_counter() - self.t_start) / 60.0,
            decisions_total=self.total_decisions,
            gpu_mb=(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else 0.0,
        )
        ps = self.arena.pool.stats()
        m.update(pool_panics=ps["panics"], pool_illegal=ps["illegal"])
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(m) + "\n")
        if self.update % cfg.snapshot_every == 0:
            self.arena.add_snapshot(self.learner)
        if self.update % cfg.ckpt_every == 0:
            self.save(f"ckpt-{self.update:05d}.pt")
            self.save("latest.pt")
            self.save_state()
        return m


def fmt(m: dict[str, float]) -> str:
    g = lambda k, f="{:.3f}": f.format(m[k]) if k in m and m[k] == m[k] else "n/a"  # noqa: E731
    return (
        f"u{int(m['update']):4d} {m['wall_min']:5.1f}m | n={int(m['samples']):6d} dec/s={m['decisions_per_s']:6.0f} "
        f"(e2e {m['e2e_decisions_per_s']:5.0f}) games/s={m['games_per_s']:5.1f} | wr_anchor={g('wr_anchor')} (n={int(m['n_anchor'])}) wr_past={g('wr_past')} "
        f"self_p1={g('wr_self_p1')}{''.join(f' {k[5:]}={g(k)}' for k in m if k.startswith('wr_x_'))} | kl_anchor={g('kl_anchor', '{:.4f}')} approx_kl={g('approx_kl', '{:.4f}')} "
        f"ent={g('entropy')} vloss={g('value_loss')} ev={g('explained_var', '{:.2f}')} clip={g('clipfrac', '{:.2f}')} "
        f"turns={m['mean_turns']:.1f} draws={m['draw_rate']:.3f} gpuMB={m['gpu_mb']:.0f}"
    )


# ---------------------------------------------------------------------------
# Distillation to a smaller net
# ---------------------------------------------------------------------------


def distill(teacher: Any, student: Any, cfg: PPOConfig, device: str, updates: int, out: Path | None = None) -> list[dict[str, float]]:
    """Trains ``student`` (fresh, smaller) to match ``teacher``'s policy and value on states reached by the
    teacher playing itself in the pool. Returns the per-update log."""
    import torch

    from showdown_bot import token_net  # noqa: F401

    batcher = TokenBatcher(device, cfg.zero_history)
    import poke_engine

    pool = poke_engine.SelfPlayPool(cfg.games, cfg.seed + 101, cfg.threads, cfg.turn_cap)
    tpol = Policy(teacher, device, cfg.amp, cfg.infer_batch)
    student.train()
    zero_dropout(student)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.startswith("cuda"))
    log: list[dict[str, float]] = []
    dev = batcher.device
    for u in range(updates):
        cats, conts, hcs, hfs, pols, vals = [], [], [], [], [], []
        for _ in range(cfg.rollout_steps // 2 + 1):
            obs = Obs(pool.observe(), batcher.cat_len, batcher.cont_len)
            batch = batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont)
            f, probs = tpol.forward(batch, probs_out=True)
            cats.append(obs.cat.copy())
            conts.append(obs.cont.astype(np.float16))
            hcs.append(obs.hist_cat.copy())
            hfs.append(obs.hist_cont.astype(np.float16))
            pols.append(probs.float().cpu().numpy())
            vals.append(((f.value + 1) * 0.5).astype(np.float32))
            pool.step(f.actions.tobytes())
        cat, cont, hc, hf = (np.concatenate(x) for x in (cats, conts, hcs, hfs))
        pol, val = np.concatenate(pols), np.concatenate(vals)
        n = len(cat)
        tot = {"kl": 0.0, "vbce": 0.0}
        k = 0
        for _ in range(2):
            perm = np.random.permutation(n)
            for s in range(0, n, cfg.minibatch):
                idx = np.sort(perm[s : s + cfg.minibatch])
                b = batcher.to_device(cat[idx], cont[idx], hc[idx], hf[idx])
                tp = torch.from_numpy(pol[idx]).to(dev)
                tv = torch.from_numpy(val[idx]).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=cfg.amp and dev.type == "cuda"):
                    o = student(b)
                legal = legal_mask(b)
                lp = masked_log_softmax(o["policy"], legal)
                kl = (tp * (torch.log(tp.clamp_min(1e-9)) - lp)).sum(-1).mean()
                vb = torch.nn.functional.binary_cross_entropy_with_logits(o["value"].float(), tv)
                loss = kl + 0.5 * vb
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                tot["kl"] += float(kl)
                tot["vbce"] += float(vb)
                k += 1
        row = {"distill_update": u + 1, "samples": n, "kl_teacher": tot["kl"] / k, "value_bce": tot["vbce"] / k}
        log.append(row)
        logger.info("distill %s", row)
        if out is not None:
            with open(out / "distill.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    student.eval()
    return log


# ---------------------------------------------------------------------------
# Evaluation and benchmarks
# ---------------------------------------------------------------------------


def play_match(a: Any, b: Any, games: int, device: str, seed: int = 7, threads: int = 4, greedy: bool = False,
               pool_games: int = 128, turn_cap: int = 200, zero_hist: tuple[bool, bool] = (False, False)) -> dict[str, float]:
    """``a`` vs ``b`` in the pool, seats alternating per game slot; returns a's score (1 / 0.5 / 0).
    ``zero_hist`` = (a, b): that side sees padding history rows (what a live bot without history tracking feeds)."""
    import poke_engine

    batcher = TokenBatcher(device)
    pa, pb = Policy(a, device), Policy(b, device)
    pool = poke_engine.SelfPlayPool(pool_games, seed, threads, turn_cap)
    a_seat = np.arange(pool_games) % 2
    scores: list[float] = []
    done = 0
    while done < games:
        obs = Obs(pool.observe(), batcher.cat_len, batcher.cont_len)
        batch = batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont)
        is_a = obs.side == a_seat[obs.game]
        actions = np.zeros(obs.n, np.uint8)
        import torch

        for pol, mask, zh in ((pa, is_a, zero_hist[0]), (pb, ~is_a, zero_hist[1])):
            idx = np.nonzero(mask)[0]
            if len(idx):
                sel = torch.from_numpy(idx).to(batcher.device)
                sub = {k: v[sel] for k, v in batch.items()}
                if zh:
                    sub["hist_cat"] = torch.zeros_like(sub["hist_cat"])
                    sub["hist_cont"] = torch.zeros_like(sub["hist_cont"])
                actions[idx] = pol.forward(sub, greedy=greedy).actions
        _, _, fin = pool.step(actions.tobytes())
        for g, w, _t in fin:
            scores.append(0.5 if w == 0 else float(w == a_seat[g] + 1))
            a_seat[g] = 1 - a_seat[g]
            done += 1
    s = np.array(scores[:games])
    se = float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else float("nan")
    p = float(np.clip(s.mean(), 1e-3, 1 - 1e-3))
    return {"games": len(s), "score": float(s.mean()), "se": se, "elo": 400 * math.log10(p / (1 - p)),
            "elo_se": 400 / math.log(10) * se / (p * (1 - p)), "stats": pool.stats()}


def bench_pool(net: Any, device: str, sizes: Sequence[int], steps: int = 40, threads: int = 4, seed: int = 3) -> list[dict[str, float]]:
    """Decisions per second of the whole loop (observe + upload + forward + sample + step) per pool size."""
    import poke_engine
    import torch

    batcher = TokenBatcher(device)
    pol = Policy(net, device)
    rows = []
    for g in sizes:
        pool = poke_engine.SelfPlayPool(g, seed, threads, 200)
        for w in range(3):  # warm-up
            obs = Obs(pool.observe(), batcher.cat_len, batcher.cont_len)
            f = pol.forward(batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont))
            pool.step(f.actions.tobytes())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_obs = t_fwd = t_step = 0.0
        dec = 0
        t0 = time.perf_counter()
        for _ in range(steps):
            a = time.perf_counter()
            obs = Obs(pool.observe(), batcher.cat_len, batcher.cont_len)
            b_ = time.perf_counter()
            f = pol.forward(batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont))
            c = time.perf_counter()
            pool.step(f.actions.tobytes())
            d = time.perf_counter()
            t_obs += b_ - a
            t_fwd += c - b_
            t_step += d - c
            dec += obs.n
        wall = time.perf_counter() - t0
        rows.append({"games": g, "threads": threads, "decisions": dec, "wall_s": wall, "decisions_per_s": dec / wall,
                     "rows_per_step": dec / steps, "ms_observe": 1000 * t_obs / steps, "ms_forward": 1000 * t_fwd / steps,
                     "ms_step": 1000 * t_step / steps})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _device(arg: str) -> str:
    import torch

    return "cuda" if arg == "auto" and torch.cuda.is_available() else ("cpu" if arg == "auto" else arg)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="PPO self-play training")
    tr.add_argument("--init", required=True, help="supervised TokenNet checkpoint (also the frozen anchor)")
    tr.add_argument("--out", required=True)
    tr.add_argument("--minutes", type=float, default=0.0, help="wall-clock budget (0: use --updates)")
    tr.add_argument("--updates", type=int, default=100)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--resume", action="store_true")
    tr.add_argument("--resume-from", help="learner weights to start from instead of --init (anchor stays --init)")
    for f in dataclasses.fields(PPOConfig):
        if f.name == "betas":
            continue
        kind = type(f.default)
        if kind is bool:
            tr.add_argument("--" + f.name.replace("_", "-"), type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            tr.add_argument("--" + f.name.replace("_", "-"), type=kind, default=None)
    bn = sub.add_parser("bench", help="decisions/s of the pool + net loop")
    bn.add_argument("--init", required=True)
    bn.add_argument("--games", type=int, nargs="+", default=[64, 128, 256, 512])
    bn.add_argument("--steps", type=int, default=40)
    bn.add_argument("--threads", type=int, default=4)
    bn.add_argument("--device", default="auto")
    ev = sub.add_parser("eval", help="a vs b in the pool")
    ev.add_argument("--a", required=True)
    ev.add_argument("--b", required=True)
    ev.add_argument("--games", type=int, default=400)
    ev.add_argument("--greedy", action="store_true")
    ev.add_argument("--zero-hist-a", action="store_true", help="a sees padding history rows")
    ev.add_argument("--zero-hist-b", action="store_true", help="b sees padding history rows")
    ev.add_argument("--threads", type=int, default=4)
    ev.add_argument("--device", default="auto")
    for sp in (tr, bn, ev):
        sp.add_argument("--gpu-mem-gb", type=float, default=6.0, help="hard cap on this process's CUDA memory")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    torch.set_num_threads(2)
    dev = _device(args.device)
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.gpu_mem_gb / total))

    if args.cmd == "bench":
        net = load_net(args.init, dev)
        for r in bench_pool(net, dev, args.games, args.steps, args.threads):
            print(json.dumps(r))
        return
    if args.cmd == "eval":
        a, b = load_net(args.a, dev), load_net(args.b, dev)
        print(json.dumps(play_match(a, b, args.games, dev, threads=args.threads, greedy=args.greedy,
                                    zero_hist=(args.zero_hist_a, args.zero_hist_b))))
        return

    cfg = PPOConfig()
    for f in dataclasses.fields(PPOConfig):
        v = getattr(args, f.name, None)
        if v is not None and f.name != "betas":
            setattr(cfg, f.name, v)
    out = Path(args.out)
    anchor = load_net(args.init, dev)
    learner = load_net(args.resume_from or args.init, dev)
    (out).mkdir(parents=True, exist_ok=True)
    if cfg.arch != "init" and not args.resume_from:
        student = small_net(anchor, cfg.arch)
        logger.info("distilling %s into %s (%d params)", args.init, cfg.arch, student.n_params())
        distill(anchor, student, cfg, dev, cfg.distill_updates, out)
        student.export(out / "distilled.pt", extra={"distilled_from": args.init})
        learner = student
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg) | {"init": args.init, "device": dev}, indent=1))
    t = Trainer(cfg, learner, anchor, dev, out)
    if args.resume and (out / "state.pt").exists():
        t.resume()
        if cfg.seed_past > 0:
            # a restart used to begin with an empty past pool (all non-self games against the weak anchor until
            # snapshots built up again); refill it from the checkpoints on disk, oldest first
            old = sorted(q for q in out.glob("ckpt-*.pt") if int(q.stem.split("-")[1]) < t.update)[-cfg.seed_past:]
            for q in old:
                t.arena.add_snapshot(load_net(str(q), dev))
            logger.info("past pool seeded from %s", [q.stem for q in old])
    t.arena.add_snapshot(learner)
    deadline = time.perf_counter() + args.minutes * 60 if args.minutes > 0 else None
    logger.info("learner %d params, anchor %d params, device %s", learner.n_params(), anchor.n_params(), dev)
    try:
        while True:
            m = t.train_step()
            logger.info(fmt(m))
            if deadline is not None:
                if time.perf_counter() >= deadline:
                    break
            elif t.update >= args.updates:
                break
    finally:
        p = t.save("latest.pt")
        t.save_state()
        logger.info("saved %s", p)


if __name__ == "__main__":
    main()
