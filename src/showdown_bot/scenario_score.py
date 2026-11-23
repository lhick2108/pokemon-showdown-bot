"""Scenario packs: where does the net (and the rollout search on it) give up win chance? (bead showdown-bot-589.68.8)

    python -m showdown_bot.scenario_score --init runs/ppo7/ckpt-36750.pt --out runs/sc1 --per-tag 60 \
        --opp self,models/tok-d256.pt@0.25

The net plays itself in the simulator. Decisions are sorted into scenarios by plain facts of the position (the
opponent's active mon is boosted, ours is at low HP with a switch available, a boosting move is legal, ...); the tags
only choose what gets measured, they never tell a bot what to play. For each chosen decision an ANSWER KEY is built:
every legal action is played in ``--worlds`` sampled worlds (teams the opponent could have, as in the rollout search)
and each copy is then played to the END of the game, our seat sampling from the net and the opponent played by each
``--opp`` model in turn (``self`` = the same net sampling; ``PATH@T`` = a second net at temperature T). The mean
result of an action is its value under that continuation, with no value head involved.

Against the key, per scenario and per opponent model:
  * regret of the net's most likely action, of a sample from its policy, and of the rollout search's choice
    (``--arm``, default b0.01/w64/d4 on the top ``--cands``): value of the key's best action minus the value of the
    action taken. The best action is picked on half of the worlds and both are valued on the other half (and the
    reverse), so noise in the key does not show up as regret;
  * the value head against the key (the key's value of the net's own policy): bias and correlation.

Limits: the key is only as good as the continuation it is played under (hence more than one ``--opp``); positions
come from the net's own games, not from games against other bots.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from showdown_bot.net_search import slot_names
from showdown_bot.ppo import N_ACT, Policy, load_net
from showdown_bot.rollout_search import RETIRE, Arm, RolloutArena

logger = logging.getLogger(__name__)

# Move lists used ONLY to label positions and actions for the report.
SETUP = {"swordsdance", "nastyplot", "dragondance", "calmmind", "quiverdance", "bulkup", "shellsmash", "agility",
         "rockpolish", "irondefense", "curse", "coil", "bellydrum", "tailglow", "shiftgear", "trailblaze", "torchsong",
         "victorydance", "tidyup", "workup", "growth", "honeclaws", "autotomize", "amnesia", "cosmicpower",
         "filletaway", "clangoroussoul", "noretreat", "geomancy", "takeheart", "acidarmor", "cottonguard", "stockpile"}
PHAZE = {"roar", "whirlwind", "dragontail", "circlethrow", "haze", "clearsmog", "encore", "taunt", "spectralthief",
         "topsyturvy", "yawn", "perishsong"}
RECOVER = {"recover", "roost", "softboiled", "slackoff", "moonlight", "morningsun", "synthesis", "shoreup", "milkdrink",
           "strengthsap", "wish", "rest", "healorder", "protect", "substitute", "spikyshield", "banefulbunker",
           "kingsshield", "silktrap", "burningbulwark", "detect", "painsplit", "leechseed"}
HAZARD = {"stealthrock", "spikes", "toxicspikes", "stickyweb", "rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
STATUS = {"toxic", "willowisp", "thunderwave", "spore", "sleeppowder", "glare", "nuzzle", "hypnosis", "stunspore",
          "knockoff", "trick", "switcheroo"}
PIVOT = {"uturn", "voltswitch", "flipturn", "partingshot", "teleport", "chillyreception", "batonpass", "shedtail"}


def action_class(name: str) -> str:
    if name.startswith("switch "):
        return "switch"
    tera = name.endswith("-tera")
    mv = name[:-5] if tera else name
    for label, group in (("setup", SETUP), ("phaze", PHAZE), ("recover", RECOVER), ("hazard", HAZARD),
                         ("status", STATUS), ("pivot", PIVOT)):
        if mv in group:
            return label + ("+tera" if tera else "")
    return "attack+tera" if tera else "attack"


def _alive(side: Any) -> int:
    return sum(1 for p in side.pokemon if int(p.hp) > 0)


def tags_of(state: Any, side: int, turn: int, names: Sequence[str], legal: np.ndarray) -> list[str]:
    """Scenario labels of a decision of ``side`` (0 = side one)."""
    me, op = (state.side_one, state.side_two) if side == 0 else (state.side_two, state.side_one)
    act = me.pokemon[int(me.active_index)]
    lg = [names[i] for i in range(N_ACT) if legal[i]]
    moves = {n[:-5] if n.endswith("-tera") else n for n in lg if not n.startswith("switch ")}
    can_switch = any(n.startswith("switch ") for n in lg)
    out = ["any"]
    if int(act.hp) <= 0 or not moves:
        return out + ["replace"] if can_switch else out
    hp = int(act.hp) / max(int(act.maxhp), 1)
    ob = max(int(op.attack_boost), int(op.special_attack_boost), int(op.speed_boost))
    mb = max(int(me.attack_boost), int(me.special_attack_boost), int(me.speed_boost))
    if ob >= 1:
        out.append("opp_boosted")
        if moves & PHAZE:
            out.append("opp_boosted+answer")
    if ob >= 2:
        out.append("opp_boosted2")
    if mb >= 1:
        out.append("we_boosted")
    if moves & SETUP:
        out.append("setup_legal")
        if hp >= 0.75 and ob < 1:
            out.append("setup_healthy")
    if hp <= 0.33 and can_switch:
        out.append("low_hp")
    if any(n.endswith("-tera") for n in lg):
        out.append("tera_legal")
    sc = me.side_conditions
    if int(sc.stealth_rock) + int(sc.spikes) + int(sc.toxic_spikes) + int(sc.sticky_web) > 0 and can_switch:
        out.append("hazards_on_us")
    if moves & RECOVER and hp <= 0.6:
        out.append("recover_legal")
    a, b = _alive(me), _alive(op)
    if a <= 2 and b <= 2:
        out.append("endgame")
    if a < b:
        out.append("behind")
    elif a > b:
        out.append("ahead")
    if turn <= 3:
        out.append("opening")
    return out


class ScenarioScorer(RolloutArena):
    def __init__(self, net: Any, device: str, opps: Sequence[tuple[str, Any, float]], arm: Arm, *, games: int, key_worlds: int,
                 cands: int, minp: float, alpha: float, seed: int, threads: int, max_fork: int) -> None:
        super().__init__(net, device, [arm], games=games, worlds=arm.worlds, depth=arm.depth, cands=cands, minp=minp,
                         alpha=alpha, seed=seed, threads=threads, max_fork=max_fork)
        self.arm = arm
        self.key_worlds = key_worlds
        # (label, Policy or None for the main net, temperature)
        self.opps = [(lab, None if nt is None else Policy(nt, device, max_batch=4096), t) for lab, nt, t in opps]

    def playout(self, games: np.ndarray, sides: np.ndarray, cand: np.ndarray, opp: Any, opptemp: float) -> tuple[np.ndarray, np.ndarray]:
        """Results [m, worlds, C] in -1..1 for the seat (nan = not played) of each candidate played to the game's end."""
        m, c = cand.shape
        k = self.key_worlds
        self.forks += 1
        seed = (self.seed * 1000003 + self.steps * 131 + self.forks) & 0xFFFFFFFF
        fp = self.pool.fork_rollouts(games.tolist(), sides.tolist(), k, c, seed=seed)
        n = m * k * c
        j = np.arange(n)
        dec, ci = j // (k * c), j % c
        seat = sides[dec]
        res = np.full(n, np.nan, np.float64)
        valid = np.zeros(n, bool)
        for t in range(3000):
            obs, f, probs = self._forward(fp)
            if obs.n == 0:
                break
            g = obs.game
            is_seat = obs.side == seat[g]
            acts = f.actions.copy()
            alt = np.nonzero(~is_seat)[0]
            if len(alt) and (opp is not None or opptemp != 1.0):
                if opp is not None:
                    sub = lambda x: None if x is None else x[alt]  # noqa: E731
                    f2, p2 = opp.forward(self.batcher.to_device(obs.cat[alt], obs.cont[alt], sub(obs.hist_cat), sub(obs.hist_cont)),
                                         probs_out=True)
                    self.evals += len(alt)
                    acts[alt] = f2.actions
                    pa = p2.float().cpu().numpy().astype(np.float64)
                else:
                    pa = probs[alt].astype(np.float64)
                if opptemp != 1.0:
                    pa = pa * (obs.legal[alt] > 0)
                    if opptemp == 0:
                        acts[alt] = pa.argmax(1).astype(np.uint8)
                    else:
                        pa = pa ** (1.0 / opptemp)
                        cs = np.cumsum(pa, axis=1)
                        u = self.rng.random(len(pa)) * cs[:, -1]
                        acts[alt] = np.minimum((cs < u[:, None]).sum(1), N_ACT - 1).astype(np.uint8)
            if t == 0:
                has_seat = np.zeros(n, bool)
                has_seat[g[is_seat]] = True
                gs = g[is_seat]
                a = cand[dec[gs], ci[gs]]
                ok = (a >= 0) & (obs.legal[np.nonzero(is_seat)[0], np.maximum(a, 0)] > 0)
                acts[is_seat] = np.where(ok, a, RETIRE).astype(np.uint8)
                valid[gs] = ok
                acts[~is_seat & ~has_seat[g]] = RETIRE
            _, _, fin = fp.step(acts.tobytes())
            for gi, w, _turns in fin:
                res[gi] = 0.0 if w == 0 else (1.0 if w == seat[gi] + 1 else -1.0)
        valid &= ~np.isnan(res)
        return np.where(valid, res, np.nan).reshape(m, k, c), valid.reshape(m, k, c)

    def advance(self, want: Any, limit: int) -> list[dict[str, Any]]:
        """One step of the main games; up to ``limit`` decisions that ``want(tags)`` accepts are scored first."""
        import poke_engine

        obs, f, p = self._forward(self.pool)
        if obs.n == 0:
            raise RuntimeError("the pool has no decisions to make")
        self.steps += 1
        legal = obs.legal > 0
        p = p.astype(np.float64) * legal
        p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
        picked: list[tuple[int, list[str], list[str]]] = []
        for r in self.rng.permutation(obs.n):
            if len(picked) >= limit:
                break
            if legal[r].sum() < 2:
                continue
            ss = self.pool.game_state(int(obs.game[r]))
            side = int(obs.side[r])
            names = slot_names(poke_engine.state_tokens_str(ss, 1, None, True)[2] if side == 0
                               else poke_engine.state_tokens_str(ss, 2, None, True)[2])
            tg = tags_of(poke_engine.State.from_string(ss), side, int(obs.turn[r]), names, legal[r])
            if want(tg):
                picked.append((int(r), tg, names))
        out: list[dict[str, Any]] = []
        if picked:
            rows = np.array([x[0] for x in picked])
            games, sides = obs.game[rows].astype(np.int64), obs.side[rows].astype(np.int64)
            # the rollout search's own choice
            pr = p[rows]
            order = np.argsort(-pr, axis=1)[:, : self.cands]
            top = np.take_along_axis(pr, order, 1)
            keep = top >= self.minp
            keep[:, 0] = True
            cd = np.where(keep, order, -1)
            q4, cnt4 = self.search(games, sides, cd, np.where(cd >= 0, top, 0.0), self.arm)
            # the answer key: every legal action to the end of the game
            full = np.where(legal[rows], np.arange(N_ACT)[None, :], -1)
            keys = {}
            per = max(1, self.max_fork // (self.key_worlds * N_ACT))
            for lab, pol, temp in self.opps:
                parts = [self.playout(games[s:s + per], sides[s:s + per], full[s:s + per], pol, temp)[0]
                         for s in range(0, len(rows), per)]
                keys[lab] = np.concatenate(parts, 0)
            for i, (r, tg, names) in enumerate(picked):
                ok = (cd[i] >= 0) & (cnt4[i] > 0)
                a = cd[i][ok]
                if len(a) >= 2:
                    lg = q4[i][ok] + self.arm.x * np.log(np.maximum(p[r, a], 1e-9))
                    choice = int(a[lg.argmax()])
                else:
                    choice = int(p[r].argmax())
                rec = {"tags": tg, "turn": int(obs.turn[r]), "names": names, "legal": legal[r].astype(int).tolist(),
                       "prior": np.round(p[r], 4).tolist(), "value": round(float(f.value[r]), 4), "ro_choice": choice,
                       "ro_cand": a.tolist(), "ro_q": np.round(q4[i][ok], 4).tolist(), "key": {}}
                for lab, kq in keys.items():
                    x = kq[i]  # [worlds, 13]
                    h = [np.nanmean(x[0::2], 0), np.nanmean(x[1::2], 0)]
                    rec["key"][lab] = {"q": [None if np.isnan(v) else round(float(v), 4) for v in np.nanmean(x, 0)],
                                       "qa": [None if np.isnan(v) else round(float(v), 4) for v in h[0]],
                                       "qb": [None if np.isnan(v) else round(float(v), 4) for v in h[1]],
                                       "n": (~np.isnan(x)).sum(0).tolist()}
                out.append(rec)
        _, _, _fin = self.pool.step(f.actions.astype(np.uint8).tobytes())
        return out


def summarise(recs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per opponent model and tag: regrets (win-probability points per decision) and the value head against the key."""
    labs = list(recs[0]["key"]) if recs else []
    out: dict[str, Any] = {}
    for lab in labs:
        rows: dict[str, list[tuple[float, ...]]] = {}
        miss: dict[str, dict[str, int]] = {}
        for rc in recs:
            kq = rc["key"][lab]
            lg = [i for i in range(N_ACT) if rc["legal"][i] and kq["qa"][i] is not None and kq["qb"][i] is not None]
            if len(lg) < 2:
                continue
            qa = np.array([kq["qa"][i] for i in lg])
            qb = np.array([kq["qb"][i] for i in lg])
            p = np.array([rc["prior"][i] for i in lg])
            p = p / max(p.sum(), 1e-12)
            net = int(p.argmax())
            ro = lg.index(rc["ro_choice"]) if rc["ro_choice"] in lg else net

            def reg(c: int) -> float:  # best on one half, valued on the other; -1..1 -> win probability
                return 0.25 * ((qb[qa.argmax()] - qb[c]) + (qa[qb.argmax()] - qa[c]))

            samp = 0.25 * ((qb[qa.argmax()] - (p * qb).sum()) + (qa[qb.argmax()] - (p * qa).sum()))
            keyv = float((p * (qa + qb) / 2).sum())
            best = int((qa + qb).argmax())
            row = (reg(net), samp, reg(ro), float(ro != net), rc["value"], keyv, float(best != net))
            bc = action_class(rc["names"][lg[best]])
            for tg in rc["tags"]:
                rows.setdefault(tg, []).append(row)
                if best != net:
                    d = miss.setdefault(tg, {})
                    d[bc] = d.get(bc, 0) + 1
        tab = {}
        for tg, rr in sorted(rows.items()):
            x = np.array(rr)
            n = len(x)
            se = lambda v: float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0  # noqa: E731
            tab[tg] = {"n": n, "net_regret": round(float(x[:, 0].mean()), 4), "net_se": round(se(x[:, 0]), 4),
                       "sample_regret": round(float(x[:, 1].mean()), 4),
                       "ro_regret": round(float(x[:, 2].mean()), 4), "ro_se": round(se(x[:, 2]), 4),
                       "ro_gain": round(float((x[:, 0] - x[:, 2]).mean()), 4), "ro_gain_se": round(se(x[:, 0] - x[:, 2]), 4),
                       "ro_changed": round(float(x[:, 3].mean()), 3),
                       "value_bias": round(float((x[:, 4] - x[:, 5]).mean()), 4),
                       "value_corr": round(float(np.corrcoef(x[:, 4], x[:, 5])[0, 1]), 3) if n > 2 else None,
                       "key_best_not_net": round(float(x[:, 6].mean()), 3),
                       "key_best_classes": dict(sorted(miss.get(tg, {}).items(), key=lambda kv: -kv[1]))}
        out[lab] = tab
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--opp", default="self", help="comma list: self[@T] or PATH[@T] (T = temperature of the opponent's moves)")
    ap.add_argument("--arm", default="b0.01/w64/d4")
    ap.add_argument("--cands", type=int, default=6)
    ap.add_argument("--minp", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--key-worlds", type=int, default=128)
    ap.add_argument("--per-tag", type=int, default=60, help="decisions wanted per scenario tag")
    ap.add_argument("--per-step", type=int, default=4, help="decisions scored per step of the main games")
    ap.add_argument("--skip", type=int, default=3, help="main steps played between scored steps (spreads the positions)")
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--minutes", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-fork", type=int, default=7000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net = load_net(args.init, args.device)
    opps = []
    for spec in args.opp.split(","):
        path, _, t = spec.partition("@")
        opps.append((spec, None if path == "self" else load_net(path, args.device), float(t) if t else 1.0))
    sc = ScenarioScorer(net, args.device, opps, Arm(args.arm), games=args.games, key_worlds=args.key_worlds, cands=args.cands,
                        minp=args.minp, alpha=args.alpha, seed=args.seed, threads=args.threads, max_fork=args.max_fork)
    have: dict[str, int] = {}
    recs: list[dict[str, Any]] = []

    def want(tags: Sequence[str]) -> bool:
        return any(have.get(t, 0) < args.per_tag for t in tags)

    t0 = time.time()
    last = t0
    step = 0
    with open(out / "decisions.jsonl", "a", encoding="utf-8") as fh:
        while len(recs) < args.max_decisions:
            step += 1
            new = sc.advance(want, args.per_step if step % (args.skip + 1) == 0 else 0)
            for rc in new:
                for t in rc["tags"]:
                    have[t] = have.get(t, 0) + 1
                fh.write(json.dumps(rc) + "\n")
            fh.flush()
            recs.extend(new)
            now = time.time()
            if new and now - last > 120:
                last = now
                logger.info("%d decisions, %.0f s, %.2f M evals, tags %s", len(recs), now - t0, sc.evals / 1e6,
                            {k: v for k, v in sorted(have.items())})
                (out / "summary.json").write_text(json.dumps(summarise(recs), indent=1), encoding="utf-8")
            full = len(have) >= 8 and all(v >= args.per_tag for v in have.values())
            if full or (args.minutes and now - t0 > args.minutes * 60):
                break
    summ = summarise(recs)
    (out / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    for lab, tab in summ.items():
        print(f"== opponent {lab}: regret in win-probability points per decision")
        print(f"{'tag':20s} {'n':>4s} {'net':>7s} {'sample':>7s} {'rollout':>8s} {'gain':>7s} {'±':>6s} {'chg':>5s} {'v bias':>7s} {'v corr':>7s}")
        for tg, r in tab.items():
            print(f"{tg:20s} {r['n']:4d} {100 * r['net_regret']:7.2f} {100 * r['sample_regret']:7.2f} {100 * r['ro_regret']:8.2f} "
                  f"{100 * r['ro_gain']:7.2f} {100 * r['ro_gain_se']:6.2f} {r['ro_changed']:5.2f} {r['value_bias']:7.3f} {r['value_corr']!s:>7s}")
    print("SCENARIODONE", len(recs), "decisions", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
