"""Policy-rollout search with a KL anchor, measured against the bare net in the simulator (bead showdown-bot-589.68.8).

    python -m showdown_bot.rollout_search --init runs/ppo7/ckpt-36750.pt --out runs/ro1 --arms b0.1,b0.3,greedy

The search of Ataraxos (Sokota et al. 2025, arXiv 2511.07312, SI S3.7) moved to random battles. At a decision of
the searching seat:

  1. sample ``--worlds`` teams the opponent could have, given what the seat has seen (the engine's set sampler);
  2. for each candidate action (legal, prior >= ``--minp``, at most ``--cands``) and each world, play that action
     and then ``--depth - 1`` further joint steps in which the NET plays both seats, each from its own scrubbed
     view and history (``SelfPlayPool.fork_rollouts``, engine sdb.33+); every chance event is sampled;
  3. score the position reached with the net's value head (the seat's view; the result itself if the game ended)
     and average over the worlds: q(a) in -1..1;
  4. take one closed-form step of magnetic mirror descent away from the net's policy p (their Eq. 7-8, the magnet
     uniform over the candidates):  p'(a)  ~  [ exp(q(a)) * u(a)^alpha * p(a)^beta ] ^ (1 / (alpha + beta))
     and sample the move from p'.

Neither seat of a rollout sees more than it would in the real game, so there is no strategy fusion; the opponent
of a rollout does know its own (sampled) team, as the real opponent knows its real one.

Each game of the pool belongs to one arm and has one searching seat; the other seat samples from the net's policy.
Arms: ``b<beta>`` the search with that anchor strength; ``greedy`` plays the net's most likely action and ``t<T>``
samples from p^(1/T) (controls: how much of a gain is just a sharper policy).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from showdown_bot.ppo import N_ACT, Forward, Obs, Policy, TokenBatcher, load_net

logger = logging.getLogger(__name__)
RETIRE = 254


class Arm:
    """``b<beta>``, ``t<temperature>`` or ``greedy``; a ``b`` arm takes options after slashes (2026-11-17):
    ``w<worlds>``, ``d<depth>``, ``o<k>`` (the opponent's first reply is enumerated over its k most likely
    actions and weighted by their probability, instead of sampled) and ``h`` (sequential halving over the
    candidates at the same total number of rollouts), e.g. ``b0.01/w64/d8/o3/h``."""

    def __init__(self, spec: str, worlds: int = 16, depth: int = 4) -> None:
        self.spec = spec
        head, *opts = spec.split("/")
        self.kind = "greedy" if head == "greedy" else head[0]
        if self.kind not in ("b", "t", "greedy"):
            raise ValueError(f"unknown arm {spec!r} (b<beta>, t<temperature>, greedy)")
        self.x = float(head[1:]) if self.kind in ("b", "t") else 0.0
        self.worlds, self.depth, self.opp, self.halve = worlds, depth, 1, False
        for o in opts:
            if o == "h":
                self.halve = True
            elif o[:1] == "w":
                self.worlds = int(o[1:])
            elif o[:1] == "d":
                self.depth = int(o[1:])
            elif o[:1] == "o":
                self.opp = int(o[1:])
            else:
                raise ValueError(f"unknown option {o!r} in arm {spec!r} (w<worlds>, d<depth>, o<k>, h)")
        self.res: list[float] = []
        self.turns = 0
        self.evals = 0
        self.searches = 0

    def summary(self) -> dict[str, Any]:
        n = len(self.res)
        if n == 0:
            return {"arm": self.spec, "n": 0}
        r = np.asarray(self.res)
        s, se = float(r.mean()), float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.5

        def elo(p: float) -> float:
            p = min(max(p, 1e-4), 1 - 1e-4)
            return 400.0 * math.log10(p / (1 - p))

        return {"arm": self.spec, "n": n, "score": round(s, 4), "elo": round(elo(s), 1),
                "lo": round(elo(s - 1.96 * se), 1), "hi": round(elo(s + 1.96 * se), 1),
                "evals": round(self.evals / max(self.searches, 1), 1)}


class RolloutArena:
    def __init__(self, net: Any, device: str, arms: Sequence[Arm], *, games: int, worlds: int, depth: int, cands: int,
                 minp: float, alpha: float, seed: int, threads: int, turn_cap: int = 200, max_fork: int = 7000, pad: int = 0) -> None:
        import poke_engine

        if not hasattr(poke_engine.SelfPlayPool, "fork_rollouts"):
            raise RuntimeError("rollout_search needs engine sdb.33+ (SelfPlayPool.fork_rollouts)")
        self.pool = poke_engine.SelfPlayPool(games, seed, threads, turn_cap)
        self.batcher = TokenBatcher(device)
        self.policy = Policy(net, device, max_batch=4096)
        self.arms = list(arms)
        self.worlds, self.depth, self.cands, self.minp, self.alpha = worlds, depth, cands, minp, alpha
        self.rng = np.random.default_rng(seed + 17)
        self.seed = seed
        self.arm_of = np.arange(games) % len(self.arms)
        self.seat = self.rng.integers(0, 2, games).astype(np.int8)
        self.steps = 0
        self.evals = 0
        self.searches = 0
        self.forks = 0
        self.max_fork = max_fork
        self.pad = pad
        self.diag = {"changed": 0.0, "kl": 0.0, "spread": 0.0, "valid": 0.0, "ncand": 0.0, "ended": 0.0}

    def _forward(self, pool: Any) -> tuple[Obs, Any, Any]:
        obs = Obs(pool.observe(), self.batcher.cat_len, self.batcher.cont_len)
        if obs.n == 0:
            return obs, None, None
        n = obs.n
        extra = (-n) % self.pad if self.pad > 1 else 0
        if extra:  # round the batch up to a multiple of ``pad`` rows (copies of row 0, dropped again below): a few
            # batch shapes instead of thousands, which the GPU allocator handles far better beside another job
            ix = np.concatenate([np.arange(n), np.zeros(extra, np.int64)])
            take = lambda x: x if x is None else x[ix]  # noqa: E731
            batch = self.batcher.to_device(take(obs.cat), take(obs.cont), take(obs.hist_cat), take(obs.hist_cont))
        else:
            batch = self.batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont)
        f, probs = self.policy.forward(batch, probs_out=True)
        if extra:
            f = Forward(f.actions[:n], f.logp[:n], f.value[:n], f.entropy[:n])
            probs = probs[:n]
        self.evals += n
        return obs, f, probs.float().cpu().numpy()

    def rollouts(self, games: np.ndarray, sides: np.ndarray, cand: np.ndarray, *, worlds: int, depth: int,
                 opp: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Per candidate [m, C]: the sum over worlds of its value (seat's view, -1..1) and the number of worlds.

        With ``opp`` > 1 every (world, candidate) is copied once per opponent reply: copy j plays the opponent's
        j-th most likely first action and the copies are averaged with those probabilities as weights."""
        m, c = cand.shape
        k, o = worlds, max(opp, 1)
        r = c * o
        self.forks += 1
        seed = (self.seed * 1000003 + self.steps * 131 + self.forks) & 0xFFFFFFFF
        fp = self.pool.fork_rollouts(games.tolist(), sides.tolist(), k, r, seed=seed)
        n = m * k * r
        j = np.arange(n)
        dec, ci, oi = j // (k * r), (j % r) // o, j % o
        seat = sides[dec]
        q = np.full(n, np.nan, np.float64)
        wt = np.ones(n, np.float64)
        valid = np.zeros(n, bool)
        ended = 0
        for t in range(depth + 1):
            obs, f, probs = self._forward(fp)
            if obs.n == 0:
                break
            g = obs.game
            is_seat = obs.side == seat[g]
            if t == depth:  # the value of the position reached: the seat's own if it is to move, else the opponent's
                v = np.where(is_seat, f.value, -f.value).astype(np.float64)
                q[g[~is_seat]] = v[~is_seat]
                q[g[is_seat]] = v[is_seat]
                break
            acts = f.actions.copy()
            if t == 0:
                has_seat = np.zeros(n, bool)
                has_seat[g[is_seat]] = True
                gs = g[is_seat]
                a = cand[dec[gs], ci[gs]]
                ok = (a >= 0) & (obs.legal[np.nonzero(is_seat)[0], np.maximum(a, 0)] > 0)
                acts[is_seat] = np.where(ok, a, RETIRE).astype(np.uint8)
                valid[gs] = ok
                if o > 1:
                    ri = np.nonzero(~is_seat)[0]
                    go = g[ri]
                    has_opp = np.zeros(n, bool)
                    has_opp[go] = True
                    if len(ri):
                        pp = probs[ri].astype(np.float64) * (obs.legal[ri] > 0)
                        ao = np.argsort(-pp, axis=1)[np.arange(len(ri)), oi[go]]
                        w = pp[np.arange(len(ri)), ao]
                        acts[ri] = np.where(w > 0, ao, RETIRE).astype(np.uint8)
                        wt[go] = w
                    # no opponent decision this turn: the copies would be identical, keep the first
                    sr = np.nonzero(is_seat)[0]
                    lone = ~has_opp[g[sr]] & (oi[g[sr]] > 0)
                    acts[sr[lone]] = RETIRE
                    wt[g[sr[lone]]] = 0.0
                acts[~is_seat & ~has_seat[g]] = RETIRE
            _, _, fin = fp.step(acts.tobytes())
            for gi, w, _turns in fin:
                q[gi] = 0.0 if w == 0 else (1.0 if w == seat[gi] + 1 else -1.0)
                ended += 1
        valid &= ~np.isnan(q) & (wt > 0)
        qv = np.where(valid, q * wt, 0.0).reshape(m, k, c, o).sum(3)
        ws = np.where(valid, wt, 0.0).reshape(m, k, c, o).sum(3)
        got = ws > 0
        self.diag["ended"] += ended / max(int(valid.sum()), 1) * m
        return np.where(got, qv / np.maximum(ws, 1e-12), 0.0).sum(1), got.sum(1)

    def search(self, games: np.ndarray, sides: np.ndarray, cand: np.ndarray, prior: np.ndarray,
               arm: Arm) -> tuple[np.ndarray, np.ndarray]:
        """q[m, C] and the worlds behind each candidate (0 = not to be played) under the arm's settings."""
        m, c = cand.shape
        qs, cnt = np.zeros((m, c)), np.zeros((m, c), np.int64)

        def run(rows: np.ndarray, cd: np.ndarray, worlds: int) -> tuple[np.ndarray, np.ndarray]:
            a, b = np.zeros(cd.shape), np.zeros(cd.shape, np.int64)
            per = max(1, self.max_fork // (worlds * cd.shape[1] * arm.opp))
            for s0 in range(0, len(rows), per):
                x = rows[s0:s0 + per]
                a[s0:s0 + per], b[s0:s0 + per] = self.rollouts(games[x], sides[x], cd[s0:s0 + per], worlds=worlds,
                                                              depth=arm.depth, opp=arm.opp)
            return a, b

        every = np.arange(m)
        if not arm.halve or c < 3:
            qs, cnt = run(every, cand, arm.worlds)
            return qs / np.maximum(cnt, 1), cnt
        # sequential halving: the same c * worlds rollouts, in rounds; each round keeps the better half by the
        # search's own objective (mean value + beta * log prior)
        rounds = int(math.ceil(math.log2(c)))
        live = cand.copy()
        for rd in range(rounds):
            width = int((live >= 0).sum(1).max())
            if width == 0:
                break
            idx = np.argsort(live < 0, axis=1, kind="stable")[:, :width]
            a, b = run(every, np.take_along_axis(live, idx, 1), max(1, round(c * arm.worlds / (rounds * width))))
            np.add.at(qs, (every[:, None], idx), a)
            np.add.at(cnt, (every[:, None], idx), b)
            if rd == rounds - 1:
                break
            score = qs / np.maximum(cnt, 1) + arm.x * np.log(np.maximum(prior, 1e-9))
            score[(live < 0) | (cnt == 0)] = -np.inf
            keep = np.maximum(((live >= 0).sum(1) + 1) // 2, 2)
            rank = np.argsort(np.argsort(-score, axis=1), axis=1)
            live = np.where(rank < keep[:, None], live, -1)
        cnt = np.where(live >= 0, cnt, 0)
        return qs / np.maximum(cnt, 1), cnt

    def step(self) -> None:
        obs, f, p = self._forward(self.pool)
        if obs.n == 0:
            raise RuntimeError("the pool has no decisions to make")
        self.steps += 1
        n = obs.n
        legal = obs.legal > 0
        p = p.astype(np.float64) * legal
        p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        actions = f.actions.copy()
        arm = self.arm_of[obs.game]
        mine = (obs.side == self.seat[obs.game]) & (legal.sum(1) > 1)
        kinds = np.array([a.kind for a in self.arms])[arm]
        xs = np.array([a.x for a in self.arms])[arm]

        gr = mine & (kinds == "greedy")
        actions[gr] = p[gr].argmax(1)
        for r in np.nonzero(mine & (kinds == "t"))[0]:
            w = p[r] ** (1.0 / xs[r])
            actions[r] = self.rng.choice(N_ACT, p=w / w.sum())

        rows = np.nonzero(mine & (kinds == "b"))[0]
        if len(rows):
            pr = p[rows]
            order = np.argsort(-pr, axis=1)[:, : self.cands]
            top = np.take_along_axis(pr, order, 1)
            keep = top >= self.minp
            keep[:, 0] = True
            c = int(keep.sum(1).max())
            cand = np.where(keep, order, -1)[:, :c]
            q, cnt = np.zeros(cand.shape), np.zeros(cand.shape, np.int64)
            for ai in np.unique(arm[rows]):
                sel = np.nonzero(arm[rows] == ai)[0]
                am, e0 = self.arms[ai], self.evals
                cd = cand[sel]
                q[sel], cnt[sel] = self.search(obs.game[rows[sel]].astype(np.int64), obs.side[rows[sel]].astype(np.int64),
                                               cd, np.where(cd >= 0, p[rows[sel][:, None], np.maximum(cd, 0)], 0.0), am)
                am.evals += self.evals - e0
                am.searches += len(sel)
            for i, r in enumerate(rows):
                ok = (cand[i] >= 0) & (cnt[i] > 0)
                if ok.sum() < 2:
                    continue
                a, qa = cand[i][ok], q[i][ok]
                beta = xs[r]
                lg = (qa + beta * np.log(np.maximum(p[r, a], 1e-9)) + self.alpha * math.log(1.0 / len(a))) / (self.alpha + beta)
                w = np.exp(lg - lg.max())
                w /= w.sum()
                actions[r] = a[self.rng.choice(len(a), p=w)]
                base = p[r, a] / p[r, a].sum()
                self.searches += 1
                self.diag["changed"] += float(a[w.argmax()] != a[base.argmax()])
                self.diag["kl"] += float((w * np.log(np.maximum(w, 1e-12) / np.maximum(base, 1e-12))).sum())
                self.diag["spread"] += float(qa.max() - qa.min())
                self.diag["valid"] += float(cnt[i][ok].mean())
                self.diag["ncand"] += float(len(a))

        _, _, finished = self.pool.step(actions.astype(np.uint8).tobytes())
        for g, win, turns in finished:
            a = self.arms[self.arm_of[g]]
            a.res.append(0.5 if win == 0 else float(win == self.seat[g] + 1))
            a.turns += turns
            self.seat[g] = self.rng.integers(0, 2)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="b0.1,b0.3,greedy")
    ap.add_argument("--games", type=int, default=48, help="games in play at once (split evenly over the arms)")
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--depth", type=int, default=4, help="joint steps per rollout, the candidate action included")
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--minp", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.0, help="weight of the uniform magnet (the paper: beta / 10)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-games", type=int, default=0, help="stop when every arm has this many results (0: time only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-fork", type=int, default=7000, help="rollout games forked at once (memory, GPU batch)")
    ap.add_argument("--pad", type=int, default=0, help="round every net batch up to a multiple of this many rows (0 = off)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import torch

    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(a.init, a.device)
    arms = [Arm(s, a.worlds, a.depth) for s in a.arms.split(",")]
    ar = RolloutArena(net, a.device, arms, games=a.games, worlds=a.worlds, depth=a.depth, cands=a.cands, minp=a.minp,
                      alpha=a.alpha, seed=a.seed, threads=a.threads, max_fork=a.max_fork, pad=a.pad)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    t0 = last = time.time()
    while True:
        ar.step()
        now = time.time()
        done = a.max_games and all(len(x.res) >= a.max_games for x in arms)
        if now - last >= 120 or done or now - t0 >= a.minutes * 60:
            last = now
            s = max(ar.searches, 1)
            m = {"minutes": round((now - t0) / 60, 1), "steps": ar.steps, "evals": ar.evals,
                 "evals_per_search": round(ar.evals / s, 1), "evals_per_s": round(ar.evals / (now - t0), 1),
                 **{k: round(v / s, 4) for k, v in ar.diag.items()}, "arms": [x.summary() for x in arms]}
            with (out / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps(m) + "\n")
            logger.info("%s", json.dumps(m))
            if done or now - t0 >= a.minutes * 60:
                break


if __name__ == "__main__":
    main()
