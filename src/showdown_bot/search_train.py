"""Search in the self-play loop (bead showdown-bot-589.68.8, "option 3"): train the token net on the output of
its own net-guided search, AlphaZero style.

    python -m showdown_bot.search_train --init runs/ppo7/ckpt-03000.pt --out runs/st1 [--evals 96] [--games 256]

Each pool step every game with a decision gets one PUCT tree (``SelfPlayPool.net_search``, engine sdb.30+),
rooted at the game's TRUE state; the net sees every leaf through each side's own scrubbed view with that side's
history, exactly as in ``SelfPlayPool.observe``. After ``--evals`` net evaluations per tree both seats sample
their action from the root visit counts. A decision is stored with its visit distribution, the tree's root value,
the other seat's action and (once the game ends) the result; the net is then trained on

    policy: cross entropy to the visit distribution        value: BCE to (1 - mix) * outcome + mix * root value
    opp_slot: cross entropy to the other seat's action (the aux head net-search uses with views=1)

The tree knows the opponent's hidden set (one determinisation = the truth), so the targets are those of a search
over the true world: one sample of what the live bot averages over its sampled worlds.

``--p-raw`` of the games are a running check of the improvement operator: one seat samples from the raw net
policy (the root prior) instead of the visits; ``search_vs_raw`` in the metrics is the search seat's score.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from showdown_bot.ppo import N_ACT, NEG_OK, Obs, TokenBatcher, legal_mask, load_net, masked_log_softmax

logger = logging.getLogger(__name__)


class Chunk:
    """The decisions of one pool step; ``z`` (outcome for the row's seat, 0 / 0.5 / 1) is NaN until the game ends."""

    def __init__(self, obs: Obs, pi: np.ndarray, root_v: np.ndarray, opp_act: np.ndarray, keep: np.ndarray) -> None:
        self.cat = obs.cat[keep].copy()
        self.cont = obs.cont[keep].astype(np.float16)
        self.hist_cat = obs.hist_cat[keep].copy()
        self.hist_cont = obs.hist_cont[keep].astype(np.float16)
        self.pi = pi[keep].astype(np.float32)
        self.root_v = root_v[keep].astype(np.float32)
        self.opp_act = opp_act[keep].astype(np.int16)
        self.z = np.full(len(self.pi), np.nan, np.float32)


class SearchArena:
    def __init__(self, net: Any, device: str, *, games: int, evals: int, batch: int, c_puct: float, depth: int,
                 temp: float, p_raw: float, seed: int, threads: int, tokthreads: int, turn_cap: int = 200,
                 fast: bool = True, worlds: int = 0, stats: int = 0) -> None:
        import poke_engine

        # worlds > 0 (sdb.32): each seat searches `worlds` trees rooted at teams its opponent could have, given
        # what the seat has seen, and the visits are summed, as the live bot does. worlds = 0 is the first form:
        # one tree per game on the true state, which lets the search see the opponent's hidden sets.
        if worlds and not hasattr(poke_engine.SelfPlayPool, "net_search_worlds"):
            raise RuntimeError("--worlds needs engine sdb.32+ (SelfPlayPool.net_search_worlds)")
        self.worlds, self.stats = int(worlds), int(stats)

        if not hasattr(poke_engine.SelfPlayPool, "net_search"):
            raise RuntimeError("search_train needs engine sdb.30+ (SelfPlayPool.net_search)")
        self.net = net
        self.device = device
        self.pool = poke_engine.SelfPlayPool(games, seed, threads, turn_cap)
        self.batcher = TokenBatcher(device)
        self.evals, self.batch, self.c_puct, self.depth, self.temp = evals, batch, c_puct, depth, temp
        self.tokthreads = tokthreads
        self.fast = fast
        self.rng = np.random.default_rng(seed + 11)
        self.seed = seed
        self.p_raw = p_raw
        self.raw_seat = np.full(games, -1, np.int8)  # the seat that plays the raw policy (-1: both seats search)
        for g in range(games):
            self._assign(g)
        self.pending: dict[tuple[int, int], list[tuple[Chunk, int]]] = {}
        self.evaluator: Any = None
        self.steps = 0
        self.games_done = 0
        self.turns_done = 0
        self.vs_raw: deque[float] = deque(maxlen=2000)
        self.evals_done = 0
        self.t_search = 0.0

    def _assign(self, g: int) -> None:
        self.raw_seat[g] = self.rng.integers(2) if self.rng.random() < self.p_raw else -1

    def refresh(self) -> None:
        """Builds the evaluator, or re-reads the net's current weights into it (the fast forward keeps fp16 copies)."""
        from showdown_bot.net_search import NetEvaluator

        self.net.eval()
        if self.evaluator is None:
            self.evaluator = NetEvaluator(self.net, self.device, views=2, fast=self.fast, fp16res=self.fast, hist=True,
                                          batch=self.batch)
        elif self.evaluator.fast is not None:
            self.evaluator.fast.sync()  # in place: rebuilding would capture every CUDA graph again (GBs)

    def step(self, out: list[Chunk] | None) -> int:
        from showdown_bot.net_search import drive

        obs = Obs(self.pool.observe(), self.batcher.cat_len, self.batcher.cont_len)
        n = obs.n
        if n == 0:
            raise RuntimeError("the pool has no decisions to make")
        games = np.unique(obs.game)
        self.steps += 1
        t0 = time.perf_counter()
        seed = (self.seed * 1000003 + self.steps) & 0xFFFFFFFF
        side = obs.side.astype(np.int64)
        legal = obs.legal > 0
        ev = self.evaluator
        if self.worlds:
            k = self.worlds
            ns = self.pool.net_search_worlds(obs.game.astype(np.int64).tolist(), side.tolist(), k, c_puct=self.c_puct,
                                             depth=self.depth, views=2, hist=2, seed=seed, threads=self.tokthreads,
                                             stats=self.stats)
            # the same evaluations per decision as the true-state form spends per game seat pair
            info = drive(ns, ev.evaluate, launch=ev.launch, finish=ev.finish, pipe=2, time_ms=1e9,
                         max_evals=self.evals * n, batch=self.batch)
            vis, _q, prior, val, _tot = ns.root_arrays()
            rows = np.arange(n)
            vis = np.frombuffer(vis, "<u4").reshape(n, k, 2, N_ACT)[rows, :, side].sum(1)
            prior = np.frombuffer(prior, "<f4").reshape(n, k, 2, N_ACT)[rows, :, side].mean(1)
            val1 = np.frombuffer(val, "<f4").reshape(n, k).mean(1)
            v = vis.astype(np.float64) * legal
            pr = prior.astype(np.float64) * legal
        else:
            ns = self.pool.net_search(games.tolist(), c_puct=self.c_puct, depth=self.depth, views=2, hist=2,
                                      seed=seed, threads=self.tokthreads)
            info = drive(ns, ev.evaluate, launch=ev.launch, finish=ev.finish, pipe=2, time_ms=1e9,
                         max_evals=self.evals * len(games), batch=self.batch)
            vis, _q, prior, val, _tot = ns.root_arrays()
            nw = len(games)
            vis = np.frombuffer(vis, "<u4").reshape(nw, 2, N_ACT)
            prior = np.frombuffer(prior, "<f4").reshape(nw, 2, N_ACT)
            w = np.searchsorted(games, obs.game)
            val1 = np.frombuffer(val, "<f4")[w]
            v = vis[w, side].astype(np.float64) * legal
            pr = prior[w, side].astype(np.float64) * legal
        self.t_search += time.perf_counter() - t0
        self.evals_done += info.evals
        ok = v.sum(1) > 0  # rows the tree gave visits (a stuck / terminal root has none)
        uni = legal / np.maximum(legal.sum(1, keepdims=True), 1)
        pr = np.where(pr.sum(1, keepdims=True) > 0, pr / np.maximum(pr.sum(1, keepdims=True), 1e-12), uni)
        pi = np.where(ok[:, None], v / np.maximum(v.sum(1, keepdims=True), 1e-12), pr)
        play = pi if self.temp == 1.0 else pi ** (1.0 / self.temp)
        play = play / play.sum(1, keepdims=True)
        is_raw = obs.side == self.raw_seat[obs.game]
        play = np.where(is_raw[:, None], pr, play)
        cum = play.cumsum(1)
        actions = (self.rng.random(n)[:, None] * cum[:, -1:] >= cum).sum(1).clip(0, N_ACT - 1).astype(np.uint8)
        bad = ~legal[np.arange(n), actions]
        if bad.any():  # numeric edge: fall back to the most likely legal slot
            actions[bad] = play[bad].argmax(1)
        root_v = np.where(side == 0, val1, 1.0 - val1)
        row_of = np.full(2 * int(obs.game.max()) + 2, -1, np.int64)
        row_of[2 * obs.game.astype(np.int64) + side] = np.arange(n)
        other = row_of[2 * obs.game.astype(np.int64) + (1 - side)]
        opp_act = np.where(other >= 0, actions[np.maximum(other, 0)].astype(np.int16), -1)
        _, _, finished = self.pool.step(actions.tobytes())

        if out is not None:
            keep = np.nonzero(ok & ~is_raw & (legal.sum(1) > 1))[0]  # forced moves carry no policy signal
            ch = Chunk(obs, pi, root_v, opp_act, keep)
            out.append(ch)
            for j, r in enumerate(keep):
                self.pending.setdefault((int(obs.game[r]), int(obs.side[r])), []).append((ch, j))
        for g, win, turns in finished:
            self.games_done += 1
            self.turns_done += turns
            for seat in (0, 1):
                z = 0.5 if win == 0 else float(win == seat + 1)
                for ch, j in self.pending.pop((g, seat), ()):
                    ch.z[j] = z
            rs = int(self.raw_seat[g])
            if rs >= 0:
                self.vs_raw.append(0.5 if win == 0 else float(win != rs + 1))
            self._assign(g)
        return n


class Window:
    """The newest ``cap`` stored decisions (whole chunks); training samples the ones whose game has ended."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.chunks: deque[Chunk] = deque()
        self.rows = 0

    def extend(self, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            self.chunks.append(c)
            self.rows += len(c.z)
        while self.rows > self.cap and len(self.chunks) > 1:
            self.rows -= len(self.chunks.popleft().z)

    def arrays(self) -> dict[str, np.ndarray]:
        done = [np.isfinite(c.z) for c in self.chunks]
        return {k: np.concatenate([getattr(c, k)[d] for c, d in zip(self.chunks, done)])
                for k in ("cat", "cont", "hist_cat", "hist_cont", "pi", "root_v", "opp_act", "z")}


def train_on(net: Any, opt: Any, scaler: Any, data: dict[str, np.ndarray], batcher: TokenBatcher, *, steps: int,
             minibatch: int, value_mix: float, vf_coef: float, opp_coef: float, grad_clip: float,
             rng: np.random.Generator) -> dict[str, float]:
    import torch

    dev = batcher.device
    n = len(data["z"])
    agg: dict[str, float] = {}
    net.train()
    for _ in range(steps):
        idx = np.sort(rng.choice(n, size=min(minibatch, n), replace=False))
        b = batcher.to_device(data["cat"][idx], data["cont"][idx], data["hist_cat"][idx], data["hist_cont"][idx])
        pi = torch.from_numpy(data["pi"][idx]).to(dev)
        y = torch.from_numpy((1.0 - value_mix) * data["z"][idx] + value_mix * data["root_v"][idx]).to(dev)
        tgt = torch.from_numpy(data["opp_act"][idx].astype(np.int64)).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            o = net(b)
        legal = legal_mask(b)
        lp = masked_log_softmax(o["policy"], legal)
        pol = -(pi * torch.where(legal, lp, torch.zeros_like(lp))).sum(-1).mean()
        ent_t = -(pi * torch.log(pi.clamp_min(1e-9))).sum(-1).mean()
        vl = torch.nn.functional.binary_cross_entropy_with_logits(o["value"].float(), y)
        loss = pol + vf_coef * vl
        ol = o["opp_slot"].float()
        okk = (tgt >= 0) & (ol.gather(1, tgt.clamp(min=0).unsqueeze(1)).squeeze(1) > NEG_OK)
        opp = torch.zeros((), device=dev)
        if bool(okk.any()):
            opp = torch.nn.functional.cross_entropy(ol[okk], tgt[okk])
            loss = loss + opp_coef * opp
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        for k, x in (("loss", loss), ("policy_ce", pol), ("kl_to_search", pol - ent_t), ("value_loss", vl), ("opp_nll", opp),
                     ("top1_agree", (lp.argmax(-1) == pi.argmax(-1)).float().mean())):
            agg[k] = agg.get(k, 0.0) + float(x.detach())
    net.eval()
    return {k: x / max(steps, 1) for k, x in agg.items()}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--evals", type=int, default=96, help="net evaluations per tree per decision")
    ap.add_argument("--batch", type=int, default=256, help="leaves per forward (the fast forward's graphs for 2x this many rows take GPU memory)")
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0, help="actions are sampled from visits ** (1 / temp)")
    ap.add_argument("--p-raw", type=float, default=0.15, help="share of games where one seat plays the raw policy")
    ap.add_argument("--gen-steps", type=int, default=24, help="pool steps per iteration")
    ap.add_argument("--train-steps", type=int, default=12, help="minibatches per iteration")
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--window", type=int, default=400_000, help="replay window in decisions")
    ap.add_argument("--min-rows", type=int, default=30_000, help="finished decisions before training starts")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--value-mix", type=float, default=0.25, help="weight of the tree's root value in the value target")
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--opp-coef", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=600)
    ap.add_argument("--ckpt-every", type=int, default=50, help="iterations")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tokthreads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--worlds", type=int, default=0,
                    help="trees per decision on sampled opponent teams (0: one tree per game on the true state)")
    ap.add_argument("--stats", type=int, default=0, choices=(0, 1), help="with --worlds: shared information-set statistics")
    ap.add_argument("--no-train", action="store_true", help="only generate (measures search_vs_raw)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(args.init, args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    arena = SearchArena(net, args.device, games=args.games, evals=args.evals, batch=args.batch, c_puct=args.c_puct,
                        depth=args.depth, temp=args.temp, p_raw=args.p_raw, seed=args.seed, threads=args.threads,
                        tokthreads=args.tokthreads, fast=not args.no_fast, worlds=args.worlds, stats=args.stats)
    win = Window(args.window)
    rng = np.random.default_rng(args.seed + 5)
    t_start = time.perf_counter()
    trained = 0
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))
    for it in range(1, args.iters + 1):
        t0 = time.perf_counter()
        arena.refresh()
        t1 = time.perf_counter()
        e0, g0 = arena.evals_done, arena.games_done
        chunks: list[Chunk] = []
        dec = sum(arena.step(chunks) for _ in range(args.gen_steps))
        t2 = time.perf_counter()
        win.extend(chunks)
        data = win.arrays()
        m: dict[str, float] = {}
        if not args.no_train and len(data["z"]) >= args.min_rows:
            m = train_on(net, opt, scaler, data, arena.batcher, steps=args.train_steps, minibatch=args.minibatch,
                         value_mix=args.value_mix, vf_coef=args.vf_coef, opp_coef=args.opp_coef,
                         grad_clip=args.grad_clip, rng=rng)
            trained += 1
        t3 = time.perf_counter()
        vr = np.array(arena.vs_raw)
        p = float(np.clip(vr.mean(), 1e-3, 1 - 1e-3)) if len(vr) else float("nan")
        m.update(
            iter=it, trained_iters=trained, rows_ready=int(len(data["z"])), rows_window=win.rows,
            games_done=arena.games_done, games_per_s=(arena.games_done - g0) / max(t2 - t1, 1e-9),
            decisions_per_s=dec / max(t2 - t1, 1e-9), evals_per_s=(arena.evals_done - e0) / max(t2 - t1, 1e-9),
            refresh_s=t1 - t0, gen_s=t2 - t1, train_s=t3 - t2,
            search_vs_raw=float(vr.mean()) if len(vr) else float("nan"), n_vs_raw=len(vr),
            search_vs_raw_elo=400 * math.log10(p / (1 - p)) if len(vr) else float("nan"),
            mean_turns=arena.turns_done / max(arena.games_done, 1), minutes=(t3 - t_start) / 60,
        )
        with (out / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(m) + "\n")
        if it % 5 == 0 or it == 1:
            logger.info("it %d  games %d  ready %d  %.0f evals/s  %.0f dec/s  search-vs-raw %.3f (n=%d)  kl %.4f  agree %.3f  v %.4f",
                        it, arena.games_done, m["rows_ready"], m["evals_per_s"], m["decisions_per_s"], m["search_vs_raw"],
                        m["n_vs_raw"], m.get("kl_to_search", float("nan")), m.get("top1_agree", float("nan")),
                        m.get("value_loss", float("nan")))
        if trained and trained % args.ckpt_every == 0 and m.get("loss") is not None:
            net.export(out / f"ckpt-{trained:05d}.pt", extra={"search_train": vars(args), "iter": trained, "games": arena.games_done})
            net.export(out / "latest.pt", extra={"search_train": vars(args), "iter": trained, "games": arena.games_done})
        if (t3 - t_start) / 60 >= args.minutes:
            break
    if trained:
        net.export(out / "latest.pt", extra={"search_train": vars(args), "iter": trained, "games": arena.games_done})


if __name__ == "__main__":
    main()
