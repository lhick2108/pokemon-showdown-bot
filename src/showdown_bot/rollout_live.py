"""The policy-rollout search of ``rollout_search.py`` for ONE live decision (bead showdown-bot-589.68.8, engine sdb.34+).

``rollout_search.py`` measures the search inside the simulator, where the pool owns the games. Here the game is a live
battle: the caller hands over the worlds its set sampler drew (serialized states, side one = us), what each side has
seen of the other (reveal rows) and our own history rows; ``SelfPlayPool.from_worlds`` turns each world into a game
whose two seats observe exactly what they would in self-play, so the rollouts run on the path the net was trained on.

    p, info = LiveRollout(net, "cuda", worlds=16, depth=4, cands=6, beta=0.01).search(states, reveal, reveal2, hist, turn)

``p`` is the searched policy over the 13 action slots (zeros for slots that were not candidates).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Sequence

import numpy as np

from showdown_bot.ppo import N_ACT, Obs, Policy, TokenBatcher

RETIRE = 254


class LiveRollout:
    def __init__(self, net: Any, device: str, *, depth: int = 4, cands: int = 6, minp: float = 0.03, beta: float = 0.01,
                 alpha: float = 0.0, opp: int = 1, seed: int = 0, threads: int = 2, rtemp: float = 1.0,
                 oppmin: float = 0.0, opp_net: Any = None, oppshare: float = 1.0, opptemp: float = 1.0) -> None:
        import poke_engine

        if not hasattr(poke_engine.SelfPlayPool, "from_worlds"):
            raise RuntimeError("rollout-search needs engine sdb.34+ (SelfPlayPool.from_worlds)")
        if depth < 1 or cands < 1 or opp < 1 or not beta > 0 or alpha < 0:
            raise ValueError("need depth, cands, opp >= 1, beta > 0 and alpha >= 0")
        self.pe = poke_engine
        self.batcher = TokenBatcher(str(device))
        self.policy = Policy(net, str(device), max_batch=4096)
        self.depth, self.cands, self.minp, self.beta, self.alpha, self.opp = depth, cands, minp, beta, alpha, opp
        self.seed, self.threads = seed, threads
        # rtemp: temperature of BOTH sides' moves inside the rollouts (1 = the net's policy as trained, 0 = its top
        # move). oppmin (with opp > 1): weight of the WORST enumerated opponent reply against the probability-
        # weighted mean, per world (0 = the mean; a hedge against an opponent that does not play like the net).
        if rtemp < 0 or not 0 <= oppmin <= 1:
            raise ValueError("need rtemp >= 0 and 0 <= oppmin <= 1")
        self.rtemp, self.oppmin = rtemp, oppmin
        # opp_net: a SECOND net that plays the opponent's moves inside the rollouts, in the first ``oppshare`` of the
        # worlds (the rest keep our own net), so the candidates are scored against more than one way of playing.
        # Our own moves and every value still come from the main net.
        if not 0 <= oppshare <= 1:
            raise ValueError("need 0 <= oppshare <= 1")
        self.opp_policy = Policy(opp_net, str(device), max_batch=4096) if opp_net is not None else None
        self.oppshare = oppshare
        # opptemp: temperature of the second net's moves (0 = its best move). An imitation net trained on search visit
        # counts is much flatter than the bot it imitates, which plays the top move (2026-11-18: tok-d256 top-1 62% on
        # gen7's moves, but a temperature-1 sample matched only 39%).
        self.opptemp = opptemp
        self.rng = np.random.default_rng(seed + 29)
        self.searches = 0
        self.evals = 0
        self.lock = threading.Lock()  # one search at a time per player (several battles share it)

    def _forward(self, pool: Any) -> tuple[Obs, Any, Any]:
        obs = Obs(pool.observe(), self.batcher.cat_len, self.batcher.cont_len)
        if obs.n == 0:
            return obs, None, None
        f, probs = self.policy.forward(self.batcher.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont), probs_out=True)
        self.evals += obs.n
        return obs, f, probs.float().cpu().numpy()

    def _pool(self, states: Sequence[str], reveal: Any, reveal2: Any, hist: Any, turn: int, reps: int) -> Any:
        hc, hf = hist if hist is not None else (np.zeros((16, 4)), np.zeros((16, 5)))
        return self.pe.SelfPlayPool.from_worlds(
            list(states), reveal, reveal2, np.ascontiguousarray(hc, dtype="<i2").tobytes(),
            np.ascontiguousarray(hf, dtype="<f4").tobytes(), int(turn), reps=reps,
            seed=(self.seed * 1000003 + self.searches * 7919 + reps) & 0xFFFFFFFF, threads=self.threads)

    def root_prior(self, states: Sequence[str], reveal: Any, reveal2: Any, hist: Any, turn: int) -> tuple[np.ndarray, np.ndarray]:
        """Our policy over the 13 slots averaged over the worlds, and the share of worlds in which each is legal."""
        obs, f, probs = self._forward(self._pool(states, reveal, reveal2, hist, turn, 1))
        mine = obs.side == 0 if obs.n else np.zeros(0, bool)
        if not mine.any():
            raise RuntimeError("no world in which the engine sees a decision for us")
        legal = obs.legal[mine] > 0
        p = probs[mine].astype(np.float64) * legal
        p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        v = np.asarray(f.value)[mine].astype(np.float64)  # the value head at the root, our view, per world
        self.root_value = (float(v.mean()), float(v.std()))
        return p.mean(0), legal.mean(0)

    def rollouts(self, pool: Any, k: int, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """q[C] (mean value of each candidate over the worlds, our view, -1..1), worlds behind each, games ended.

        Game ``(w * C + c) * O + j`` of ``pool``: world ``w``, candidate ``c``, opponent reply ``j`` (``opp`` > 1:
        the opponent's j-th most likely first action, weighted by its probability)."""
        c, o = len(cand), self.opp
        r = c * o
        n = k * r
        j = np.arange(n)
        ci, oi = (j % r) // o, j % o
        q = np.full(n, np.nan, np.float64)
        wt = np.ones(n, np.float64)
        valid = np.zeros(n, bool)
        ended = 0
        for t in range(self.depth + 1):
            obs, f, probs = self._forward(pool)
            if obs.n == 0:
                break
            g = obs.game
            is_seat = obs.side == 0
            if t == self.depth:  # the value of the position reached: ours if we are to move, else minus the opponent's
                v = np.where(is_seat, f.value, -f.value).astype(np.float64)
                q[g[~is_seat]] = v[~is_seat]
                q[g[is_seat]] = v[is_seat]
                break
            acts = f.actions.copy()
            if self.opp_policy is not None:
                alt = np.nonzero(~is_seat & ((g // r) < round(k * self.oppshare)))[0]
                if len(alt):
                    sub = lambda x: None if x is None else x[alt]  # noqa: E731
                    f2, p2 = self.opp_policy.forward(
                        self.batcher.to_device(obs.cat[alt], obs.cont[alt], sub(obs.hist_cat), sub(obs.hist_cont)), probs_out=True)
                    self.evals += len(alt)
                    acts[alt] = f2.actions
                    probs[alt] = p2.float().cpu().numpy()
                    if self.opptemp != 1.0:
                        pa = probs[alt].astype(np.float64) * (obs.legal[alt] > 0)
                        if self.opptemp == 0:
                            acts[alt] = pa.argmax(1).astype(np.uint8)
                        else:
                            pa = pa ** (1.0 / self.opptemp)
                            ca = np.cumsum(pa, axis=1)
                            ua = self.rng.random(len(pa)) * ca[:, -1]
                            acts[alt] = np.minimum((ca < ua[:, None]).sum(1), N_ACT - 1).astype(np.uint8)
            if self.rtemp != 1.0:
                pl = probs.astype(np.float64) * (obs.legal > 0)
                if self.rtemp == 0:
                    acts = pl.argmax(1).astype(np.uint8)
                else:
                    pl = pl ** (1.0 / self.rtemp)
                    cs = np.cumsum(pl, axis=1)
                    u = self.rng.random(len(pl)) * cs[:, -1]
                    acts = np.minimum((cs < u[:, None]).sum(1), N_ACT - 1).astype(np.uint8)
            if t == 0:
                has_seat = np.zeros(n, bool)
                has_seat[g[is_seat]] = True
                gs = g[is_seat]
                a = cand[ci[gs]]
                ok = obs.legal[np.nonzero(is_seat)[0], a] > 0
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
                    sr = np.nonzero(is_seat)[0]
                    lone = ~has_opp[g[sr]] & (oi[g[sr]] > 0)  # no opponent decision: the copies are identical
                    acts[sr[lone]] = RETIRE
                    wt[g[sr[lone]]] = 0.0
                acts[~is_seat & ~has_seat[g]] = RETIRE
            _, _, fin = pool.step(acts.tobytes())
            for gi, w, _turns in fin:
                q[gi] = 0.0 if w == 0 else (1.0 if w == 1 else -1.0)
                ended += 1
        valid &= ~np.isnan(q) & (wt > 0)
        qv = np.where(valid, q * wt, 0.0).reshape(k, c, o).sum(2)
        ws = np.where(valid, wt, 0.0).reshape(k, c, o).sum(2)
        got = ws > 0
        cnt = got.sum(0)
        per = np.where(got, qv / np.maximum(ws, 1e-12), 0.0)
        if self.oppmin > 0 and o > 1:
            worst = np.where(valid, q, np.inf).reshape(k, c, o).min(2)
            per = np.where(got, (1 - self.oppmin) * per + self.oppmin * np.where(np.isfinite(worst), worst, 0.0), 0.0)
        return per.sum(0) / np.maximum(cnt, 1), cnt, ended

    def search(self, states: Sequence[str], reveal: Any, reveal2: Any, hist: Any, turn: int) -> tuple[np.ndarray, dict[str, Any]]:
        with self.lock:
            t0 = time.perf_counter()
            e0 = self.evals
            self.searches += 1
            p, legal = self.root_prior(states, reveal, reveal2, hist, turn)
            order = np.argsort(-p)[: self.cands]
            keep = (p[order] >= self.minp) & (p[order] > 0)
            keep[0] = True
            cand = order[keep]
            out = np.zeros(N_ACT)
            info: dict[str, Any] = {"prior": p, "legal": legal, "cand": cand.tolist(),
                                    "value": self.root_value[0], "value_sd": self.root_value[1]}
            if len(cand) < 2:
                out[cand[0]] = 1.0
            else:
                pool = self._pool(states, reveal, reveal2, hist, turn, len(cand) * self.opp)
                q, cnt, ended = self.rollouts(pool, len(states), cand)
                ok = cnt > 0
                info.update(q=q.tolist(), worlds=cnt.tolist(), ended=ended)
                if ok.sum() < 2:
                    out[cand[0]] = 1.0
                else:
                    a, qa = cand[ok], q[ok]
                    lg = (qa + self.beta * np.log(np.maximum(p[a], 1e-9)) + self.alpha * math.log(1.0 / len(a))) / (self.alpha + self.beta)
                    w = np.exp(lg - lg.max())
                    out[a] = w / w.sum()
            info["evals"] = self.evals - e0
            info["ms"] = (time.perf_counter() - t0) * 1000.0
            return out, info
