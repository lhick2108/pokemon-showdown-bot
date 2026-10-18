"""Offline diagnosis: why is the token transformer a weak search root prior? (docs/token-prior-diag.md)

Held-out self-play decisions (val split of the token trainer: ``tokens.split_of(battle_hash(tag), 50)``) with the
search's aggregated visit distribution as the target. Variants scored against it:

  a   the .ppol prior as the live search uses it (``PolicyPrior.priors`` on every searched world, mean)
  b   the token net as the LIVE path computes it (``TokenPolicy.predict_states`` over the sampled worlds, opponent
      scrubbed with the recorded ``opp_facts``, history rows padding, mean over worlds)
  bt  the live tokenisation on the truth state (still padded history): isolates "worlds vs truth state"
  c   the TRAINING export (``tokens.game_samples``: truth state, real 16-row event history) through the net
  d   (c) with the history zeroed (padding), as live play
  e   token tensors of the live path, saved to ``--out-npz`` and diffed against another engine wheel's run
      (``--other-npz``)

Run once per engine wheel (the wheel on PYTHONPATH decides)::

    PYTHONPATH="<wheel>;<worktree>\\src" .venv\\Scripts\\python.exe scripts\\diag_tok_prior.py \\
        --out-json r21.json --out-npz t21.npz
    PYTHONPATH="<whl23>;<worktree>\\src" ... --out-json r23.json --out-npz t23.npz --other-npz t21.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

N_ACT = 13
EPS = 0.03  # the engine mixes this much uniform into every prior (prior_eps)
KINDS = ("move", "tera", "switch")


def kind_of_slot(j: int) -> int:
    return 0 if j < 4 else (1 if j < 8 else 2)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def mix(p: np.ndarray, legal: np.ndarray, eps: float = EPS) -> np.ndarray:
    """What the engine does with a root prior: (1-eps) p + eps/k over the legal options."""
    k = legal.sum(-1, keepdims=True)
    p = np.where(legal, p, 0.0)
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    return np.where(legal, (1 - eps) * p + eps / np.maximum(k, 1), 0.0)


def entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(q)).sum(-1)


def temper(p: np.ndarray, legal: np.ndarray, temp: float) -> np.ndarray:
    z = np.where(legal, np.log(np.clip(p, 1e-9, 1.0)) / temp, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def score(p: np.ndarray, t: np.ndarray, legal: np.ndarray) -> dict[str, Any]:
    """Metrics of prior ``p`` [n,13] against the visit distribution ``t`` [n,13] on ``legal`` [n,13] rows."""
    n = len(p)
    pm = mix(p, legal)
    am_t = t.argmax(-1)
    am_p = p.argmax(-1)
    hit = am_p == am_t
    ce = -(t * np.log(np.clip(pm, 1e-9, 1))).sum(-1)
    h_t = entropy(t)
    kl = ce - h_t
    out: dict[str, Any] = {
        "n": n,
        "top1": float(hit.mean()),
        "ce": float(ce.mean()),
        "kl": float(kl.mean()),
        "entropy": float(entropy(p).mean()),
        "maxprob": float(p.max(-1).mean()),
        "frac_gt08": float((p.max(-1) > 0.8).mean()),
        "mass_on_target_argmax": float(p[np.arange(n), am_t].mean()),
    }
    kt = np.array([kind_of_slot(int(j)) for j in am_t])
    for k, name in enumerate(KINDS):
        m = kt == k
        out[f"top1_{name}"] = float(hit[m].mean()) if m.any() else None
        out[f"n_{name}"] = int(m.sum())
        # how often the prior's argmax is of this kind
    kp = np.array([kind_of_slot(int(j)) for j in am_p])
    for k, name in enumerate(KINDS):
        out[f"pred_share_{name}"] = float((kp == k).mean())
    # temperature at which the prior best matches the visits (cross-entropy), a sharpness diagnostic
    best = (1.0, out["ce"])
    for temp in (0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0):
        pt = mix(temper(p, legal, temp), legal)
        c = float((-(t * np.log(np.clip(pt, 1e-9, 1))).sum(-1)).mean())
        if c < best[1] - 1e-12:
            best = (temp, c)
    out["best_temp"] = best[0]
    out["ce_at_best_temp"] = best[1]
    return out


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def pick_games(root: Path, gen: str, per_gen: int, per_game: int, seed: int, val_permille: int):
    """Val-split battles of ``root/gen``: lists of recording files per battle, shuffled deterministically."""
    from showdown_bot import tokens

    groups = tokens._group_files(root / gen)
    val = []
    for files in groups:
        tag = files[0].name.split("__")[0]
        if tokens.split_of(tokens.battle_hash(tag), val_permille) == tokens.SPLIT_VAL:
            val.append(files)
    random.Random(seed).shuffle(val)
    return val


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    """One record per sampled decision: the decision dict, the exporter sample (real history), the role."""
    from showdown_bot import tokens

    vocab, fields, lay = tokens.Vocab(), tokens._shard_fields(), tokens.layout()
    opts = tokens.Options(reconstruct_q=False, val_permille=args.val_permille)
    rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for gen in args.gens.split(","):
        games = pick_games(Path(args.data), gen, args.per_gen, args.per_game, args.seed, args.val_permille)
        got = 0
        for files in games:
            if got >= args.per_gen:
                break
            recs = [g for g in map(tokens._read, files) if g is not None]
            for gi, g in enumerate(recs):
                others = [o for oi, o in enumerate(recs) if oi != gi]
                peer = others[0] if len(others) == 1 else None
                stats: Counter = Counter()
                samples = tokens.game_samples(g, peer, source_idx=0, kind="selfplay", opts=opts, vocab=vocab,
                                              fields=fields, lay=lay, stats=stats)
                by_idx = {int(s["decision_idx"]): s for s in samples}
                cand = []
                for idx, d in enumerate(g["decisions"]):
                    s = by_idx.get(idx)
                    if s is None or not int(s["pol_valid"]) or int(s["act_legal"][0].sum()) < 2:
                        continue
                    if not d.get("worlds") or not (d.get("truth") or {}).get("state") or d.get("opp_facts") is None:
                        continue
                    cand.append((idx, d, s))
                for idx, d, s in rng.sample(cand, min(args.per_game, len(cand))):
                    rows.append({"gen": gen, "tag": g["header"]["battle_tag"], "role": g["header"].get("role"),
                                 "idx": idx, "d": d, "s": s})
                    got += 1
        print(f"{gen}: {got} decisions from {len(games)} val battles available", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import poke_engine
    from poke_engine import State

    from showdown_bot import token_net, tokens
    from showdown_bot.token_policy import TokenPolicy

    t0 = time.time()
    rows = collect(args)
    print(f"collected {len(rows)} decisions in {time.time() - t0:.0f}s; engine {getattr(poke_engine, '__version__', '?')}",
          file=sys.stderr)
    tp = TokenPolicy.load(args.net, args.device)
    ppol = poke_engine.PolicyPrior.load(args.ppol)
    lay = tokens.layout()

    n = len(rows)
    T = np.zeros((n, N_ACT), np.float32)  # search visit distribution on the 13 slots (the exporter's pol_target)
    L = np.zeros((n, N_ACT), bool)
    A = np.zeros((n, N_ACT), np.float32)  # ppol, mean over worlds
    B = np.zeros((n, N_ACT), np.float32)  # token net, live path (worlds, mean)
    BT = np.zeros((n, N_ACT), np.float32)  # live tokenisation, truth state
    C = np.zeros((n, N_ACT), np.float32)  # training export tokens, real history
    D = np.zeros((n, N_ACT), np.float32)  # same, history zeroed
    spread = np.zeros(n, np.float32)  # mean TV distance of a world's token policy to the world mean
    unmatched = Counter()
    ppol_unmatched = Counter()
    dists: list[dict[str, float]] = []
    tok_live: list[dict[str, np.ndarray]] = []  # world 0, live tokenisation
    tok_truth: list[dict[str, np.ndarray]] = []
    sum_check = Counter()

    for i, r in enumerate(rows):
        d, s = r["d"], r["s"]
        T[i] = s["pol_target"]
        L[i] = s["act_legal"][0] > 0
        truth = d["truth"]["state"]
        worlds = [w["state"] for w in d["worlds"]]
        reveal = tokens.reveal_rows(d["opp_facts"])
        # --- b: exactly TokenPolicy.distribution's path, but keep the 13-vector
        names, pol, legal = tp.predict_states(worlds, reveal)
        vec = np.zeros(N_ACT, np.float64)
        for nm, p, lg in zip(names, pol, legal):
            vec += np.where(lg, p, 0.0) / len(names)
        B[i] = vec / max(vec.sum(), 1e-12)
        spread[i] = float(np.mean([0.5 * np.abs(np.where(lg, p, 0) - B[i]).sum() for p, lg in zip(pol, legal)]))
        # the dict the live bot ships, checked against the engine's own root option names
        dist = tp.distribution(worlds, reveal)
        dists.append(dist)
        eng_names = {c for c, _s in poke_engine.policy_actions(State.from_string(worlds[0]), 1)}
        for k in dist:
            if k not in eng_names:
                unmatched["prior_key_not_a_root_option"] += 1
        for k in eng_names:
            if k not in dist:
                unmatched["root_option_without_prior"] += 1
        sum_check["decisions"] += 1
        slot_names = names[0]
        if any(k not in slot_names for k in dist):
            unmatched["prior_key_not_a_slot_name"] += 1
        if abs(sum(dist.values()) - 1) > 1e-6:
            unmatched["dist_not_normalised"] += 1
        # recorded policy keys vs the token slot names
        miss = [k for k in d["policy"] if k not in slot_names]
        unmatched["policy_keys_unmapped"] += len(miss)
        # --- bt: truth state through the live tokeniser
        nm_t, pol_t, leg_t = tp.predict_states([truth], reveal)
        BT[i] = np.where(leg_t[0], pol_t[0], 0.0)
        if nm_t[0] != s_names(s, truth):
            sum_check["truth_names_differ_from_export"] += 0
        # --- a: ppol on each world
        acc = np.zeros(N_ACT)
        for w in worlds:
            st = State.from_string(w)
            pri = ppol.priors(st, 1)
            sl = {c: sl_ for c, sl_ in poke_engine.policy_actions(st, 1)}
            for c, p in pri:
                j = sl.get(c, -1)
                if j < 0:
                    ppol_unmatched["slot_missing"] += 1
                else:
                    acc[j] += p / len(worlds)
        A[i] = acc / max(acc.sum(), 1e-12)
        # --- tensors for (e)
        cat, cont, _ch = poke_engine.state_tokens_str(worlds[0], 1, reveal, True)
        tok_live.append(tokens.split_buffers(cat, cont, lay))
        cat, cont, _ch = poke_engine.state_tokens_str(truth, 1, reveal, True)
        tok_truth.append(tokens.split_buffers(cat, cont, lay))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", file=sys.stderr)

    # --- c / d: the exporter's samples through the net (batched)
    for ids in chunks(list(range(n)), 128):
        bt = {k: np.stack([rows[i]["s"][k] for i in ids]) for k in token_net.INPUT_FIELDS}
        out = tp.net.predict(bt, device=tp.device)["policy"]
        C[ids] = out
        bt0 = dict(bt)
        bt0["hist_cat"] = np.zeros_like(bt["hist_cat"])
        bt0["hist_cont"] = np.zeros_like(bt["hist_cont"])
        D[ids] = tp.net.predict(bt0, device=tp.device)["policy"]
    # sanity: the live tokeniser on the truth state + padding must equal the exporter with zeroed history
    d_vs_bt = float(np.abs(D - BT).max())

    # --- end-to-end mapping check: a one-hot root prior must steer the engine's root visits
    e2e = end_to_end(rows, args)
    seff = search_effect(rows, dists, args) if args.search else None

    res: dict[str, Any] = {
        "engine": getattr(poke_engine, "__version__", "?"),
        "n": n,
        "by_gen": dict(Counter(r["gen"] for r in rows)),
        "by_role": dict(Counter(r["role"] for r in rows)),
        "target": {"entropy": float(entropy(T).mean()), "maxprob": float(T.max(-1).mean()),
                   "frac_gt08": float((T.max(-1) > 0.8).mean()),
                   "argmax_kind": {k: float(np.mean([kind_of_slot(int(j)) == q for j in T.argmax(-1)]))
                                   for q, k in enumerate(KINDS)}},
        "variants": {name: score(P, T, L) for name, P in
                     (("a_ppol", A), ("b_live", B), ("bt_live_truth", BT), ("c_train_export", C), ("d_export_nohist", D))},
        "by_gen_top1": {},
        "by_role_top1": {},
        "world_spread_tv": float(spread.mean()),
        "max_abs_diff_d_vs_bt": d_vs_bt,
        "unmatched": dict(unmatched),
        "ppol_unmatched": dict(ppol_unmatched),
        "decisions_checked": sum_check["decisions"],
        "end_to_end": e2e,
        "search_effect": seff,
        "seconds": time.time() - t0,
    }
    for name, P in (("a_ppol", A), ("b_live", B), ("c_train_export", C)):
        for grp, key in (("by_gen_top1", "gen"), ("by_role_top1", "role")):
            for v in sorted({str(r[key]) for r in rows}):
                m = np.array([str(r[key]) == v for r in rows])
                res[grp].setdefault(name, {})[v] = float((P[m].argmax(-1) == T[m].argmax(-1)).mean())
    # mix of live prior with the .ppol (what search would see if both were averaged) is not needed; keep arrays
    if args.out_npz:
        dump = {f"live__{k}": np.stack([t[k] for t in tok_live]) for k in tok_live[0]}
        dump.update({f"truth__{k}": np.stack([t[k] for t in tok_truth]) for k in tok_truth[0]})
        dump.update(T=T, L=L, A=A, B=B, BT=BT, C=C, D=D)
        np.savez_compressed(args.out_npz, **dump)
    if args.other_npz:
        res["token_drift"] = drift(args.out_npz, args.other_npz)
        res["policy_drift"] = policy_drift(args.out_npz, args.other_npz)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))
    return res


def s_names(s, truth):
    return None


def end_to_end(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    """Ship a one-hot root prior (on option j) to the real search; the top-visited option must be j."""
    import poke_engine
    from poke_engine import State, monte_carlo_tree_search

    ok = tot = 0
    bad: list[str] = []
    r_ = random.Random(1)
    for r in r_.sample(rows, min(args.e2e, len(rows))):
        st = State.from_string(r["d"]["worlds"][0]["state"])
        opts = [c for c, _s in poke_engine.policy_actions(st, 1)]
        j = r_.choice(opts)
        res = monte_carlo_tree_search(st, duration_ms=60, threads=1, root_prior_s1={j: 1.0}, c_puct=30.0, prior_eps=0.0)
        vis = {}
        for o in res.side_one:
            vis[getattr(o, "move_choice", None)] = getattr(o, "visits", None)
        best = max(vis, key=lambda k: vis[k] or 0) if vis else None
        tot += 1
        ok += int(best == j)
        if best != j and len(bad) < 5:
            bad.append(f"want {j!r} got {best!r} keys {list(vis)[:4]}")
    return {"checked": tot, "steered_to_prior_option": ok, "failures": bad}


def search_effect(rows: list[dict[str, Any]], dists: list[dict[str, float]], args: argparse.Namespace) -> dict[str, Any]:
    """Real single-world, single-thread searches under different root priors (live config: evaluator, c_puct, ppol
    interior). Reference = the recorded aggregated visit share (4 worlds x 1 s, ppol prior: biased towards the ppol)."""
    from poke_engine import State, monte_carlo_tree_search

    from showdown_bot.players import _worker

    ev = _worker.get_evaluator(args.evaluator, 1.5)
    ppol = _worker.get_prior(args.ppol)
    rng = random.Random(7)
    pick = rng.sample(range(len(rows)), min(args.search, len(rows)))
    variants = {"ppol_root": None, "tok_root(+ppol interior)": "tok", "tok_root_only(UCB1 interior)": "tok_only",
                "tok_T0.7": "tok_sharp", "uniform_root": "uniform", "no_prior(UCB1)": "ucb1"}
    acc: dict[str, dict[str, list[float]]] = {v: {"hit": [], "mass": [], "ent": [], "n": []} for v in variants}
    first: dict[str, list[str]] = {v: [] for v in variants}
    for i in pick:
        r, dist = rows[i], dists[i]
        st_s = r["d"]["worlds"][0]["state"]
        pol = r["d"]["policy"]
        ref_arg = max(pol, key=pol.get)
        for v, mode in variants.items():
            st = State.from_string(st_s)
            kw: dict[str, Any] = dict(evaluator=ev, c_puct=2.0, prior_eps=0.03)
            if mode in (None, "tok", "tok_sharp", "uniform"):
                kw["prior"], kw["prior_opp"] = ppol, ppol
            if mode in ("tok", "tok_only"):
                kw["root_prior_s1"] = dist
            elif mode == "tok_sharp":
                w = {k: x ** (1 / 0.7) for k, x in dist.items()}
                z = sum(w.values())
                kw["root_prior_s1"] = {k: x / z for k, x in w.items()}
            elif mode == "uniform":
                kw["root_prior_s1"] = {k: 1.0 for k in dist}
            if mode == "ucb1":
                kw = dict(evaluator=ev)
            res = monte_carlo_tree_search(st, duration_ms=args.search_ms, threads=1, **kw)
            vis = {str(o.move_choice): float(o.visits) for o in res.side_one}
            tot = sum(vis.values()) or 1.0
            share = {k: x / tot for k, x in vis.items()}
            top = max(share, key=share.get)
            acc[v]["hit"].append(float(top == ref_arg))
            acc[v]["mass"].append(share.get(ref_arg, 0.0))
            acc[v]["ent"].append(float(-sum(p * np.log(max(p, 1e-12)) for p in share.values())))
            acc[v]["n"].append(tot)
            first[v].append(top)
    out: dict[str, Any] = {}
    for v in variants:
        out[v] = {k: float(np.mean(x)) for k, x in acc[v].items()}
        out[v]["argmax_agree_with_ppol_root_search"] = float(np.mean([a == b for a, b in zip(first[v], first["ppol_root"])]))
    out["n_decisions"] = len(pick)
    out["search_ms"] = args.search_ms
    return out


def drift(a_path: str, b_path: str) -> dict[str, Any]:
    """Fraction of rows where any token field differs between two runs' dumps, per field, for the live tensors."""
    a, b = np.load(a_path), np.load(b_path)
    out: dict[str, Any] = {}
    for kind in ("live", "truth"):
        per, anyd = {}, None
        for k in a.files:
            if not k.startswith(kind + "__") or k not in b.files:
                continue
            x, y = a[k], b[k]
            if x.shape != y.shape:
                per[k.split("__", 1)[1]] = "shape differs"
                continue
            diff = (x != y).reshape(len(x), -1).any(-1)
            per[k.split("__", 1)[1]] = float(diff.mean())
            anyd = diff if anyd is None else (anyd | diff)
        out[kind] = {"any_field": float(anyd.mean()) if anyd is not None else None,
                     "fields_differing": {k: v for k, v in per.items() if v not in (0.0,)}}
    return out


def policy_drift(a_path: str, b_path: str) -> dict[str, Any]:
    a, b = np.load(a_path), np.load(b_path)
    return {"mean_TV_live_policy": float(0.5 * np.abs(a["B"] - b["B"]).sum(-1).mean()),
            "argmax_changed": float((a["B"].argmax(-1) != b["B"].argmax(-1)).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=r"C:\Users\Lewis\pokemon-showdown-bot\data\selfplay")
    ap.add_argument("--gens", default="gen7,gen8,gen9")
    ap.add_argument("--per-gen", type=int, default=1000)
    ap.add_argument("--per-game", type=int, default=5)
    ap.add_argument("--val-permille", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net", default=r"C:\Users\Lewis\pokemon-showdown-bot\models\tok-d256.pt")
    ap.add_argument("--ppol", default=r"C:\Users\Lewis\pokemon-showdown-bot\models\prior-gen56-h32.ppol")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--search", type=int, default=0, help="decisions for the real-search prior comparison (0 = skip)")
    ap.add_argument("--search-ms", type=int, default=300)
    ap.add_argument("--evaluator", default="C:/Users/Lewis/pokemon-showdown-bot/models/gen7-res-root-mlp16.pev")
    ap.add_argument("--e2e", type=int, default=20)
    ap.add_argument("--out-json")
    ap.add_argument("--out-npz")
    ap.add_argument("--other-npz")
    args = ap.parse_args(argv)
    res = run(args)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
