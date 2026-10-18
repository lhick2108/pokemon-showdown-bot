"""Offline check of the net-guided search (docs/net-search.md): held-out self-play decisions, the recorded search's
aggregated visit distribution as the target (same loader and metrics as ``scripts/diag_tok_prior.py``).

Per decision it compares, by top-1 agreement with the recorded visit argmax:

  raw   the token net's policy (live path: recorded worlds, opponent scrubbed with the recorded ``opp_facts``)
  ns    ``NetSearcher`` on the recorded worlds (repeated up to ``--worlds``; every copy is its own tree)

and reports throughput (evals/s, batches per decision) with the latency breakdown. Decisions are cached in
``--cache`` so config sweeps do not reload the recordings::

    PYTHONPATH="<whl24>;<worktree>\\src" .venv\\Scripts\\python.exe scripts\\net_search_offline.py \\
        --per-gen 100 --per-game 2 --time-ms 1000 --depth 4 --out-json ns_d4.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_tok_prior as dtp  # noqa: E402

N_ACT = 13


def load_rows(args: argparse.Namespace) -> list[dict]:
    cache = Path(args.cache)
    if cache.is_file():
        return pickle.loads(cache.read_bytes())
    raw = dtp.collect(args)
    rows = []
    for r in raw:
        d, s = r["d"], r["s"]
        rows.append({
            "gen": r["gen"], "role": r["role"], "tag": r["tag"], "idx": r["idx"],
            "worlds": [w["state"] for w in d["worlds"]], "opp_facts": d["opp_facts"],
            "T": np.asarray(s["pol_target"], np.float32), "L": np.asarray(s["act_legal"][0] > 0),
        })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(rows))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"C:\Users\Lewis\pokemon-showdown-bot\data\selfplay")
    ap.add_argument("--gens", default="gen7,gen8,gen9")
    ap.add_argument("--per-gen", type=int, default=100)
    ap.add_argument("--per-game", type=int, default=2)
    ap.add_argument("--val-permille", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net", default=r"C:\Users\Lewis\pokemon-showdown-bot\models\tok-d256.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache", default=".cache/netsearch_offline_rows.pkl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--time-ms", type=int, default=1000)
    ap.add_argument("--max-evals", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--c-puct", type=float, default=2.0)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--fpu", type=float, default=0.0)
    ap.add_argument("--vavg", type=int, default=0)
    ap.add_argument("--scrub", type=int, default=1)
    ap.add_argument("--pipe", type=int, default=1)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import poke_engine

    from showdown_bot import net_search as nsr
    from showdown_bot import tokens
    from showdown_bot.token_policy import TokenPolicy

    t0 = time.time()
    rows = load_rows(args)
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)
    print(f"{n} decisions ({time.time() - t0:.0f}s); engine {getattr(poke_engine, '__version__', '?')}", file=sys.stderr)
    tp = TokenPolicy.load(args.net, args.device)
    searcher = nsr.NetSearcher(
        tp, worlds=args.worlds, time_ms=args.time_ms, max_evals=args.max_evals, batch=args.batch,
        c_puct=args.c_puct, depth=args.depth, fpu=args.fpu, vavg=bool(args.vavg), scrub=bool(args.scrub), pipe=args.pipe, views=args.views,
    )
    T = np.stack([r["T"] for r in rows])
    L = np.stack([r["L"] for r in rows])
    RAW = np.zeros((n, N_ACT), np.float32)
    NS = np.zeros((n, N_ACT), np.float32)
    evals, batches, wall, coll, fwd, feed, tree, tokm = [], [], [], [], [], [], [], []
    maxd, depthn = [], []
    for w in range(args.warmup):  # CUDA / cudnn warm-up, not counted
        searcher.search(rows[w % n]["worlds"] * 2, tokens.reveal_rows(rows[w % n]["opp_facts"]), time_ms=300)
    for i, r in enumerate(rows):
        reveal = tokens.reveal_rows(r["opp_facts"])
        names, pol, legal = tp.predict_states(r["worlds"], reveal)
        vec = np.zeros(N_ACT)
        for p, lg in zip(pol, legal):
            vec += np.where(lg, p, 0.0) / len(pol)
        RAW[i] = vec / max(vec.sum(), 1e-12)
        slot = {c: j for j, c in enumerate(names[0]) if c}
        worlds = (r["worlds"] * (args.worlds // len(r["worlds"]) + 1))[: args.worlds]
        policy, info = searcher.search(worlds, reveal)
        for c, p in policy.items():
            if c in slot:
                NS[i, slot[c]] = p
        d = info["drive"]
        evals.append(d.evals); batches.append(d.batches); wall.append(d.wall_ms)
        coll.append(d.collect_ms); fwd.append(d.forward_ms); feed.append(d.feed_ms)
        tree.append(d.stats["tree_ms"]); tokm.append(d.stats["tokenise_ms"]); maxd.append(d.stats["max_depth"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s) ns top1 so far "
                  f"{np.mean(NS[:i + 1].argmax(-1) == T[:i + 1].argmax(-1)):.3f}", file=sys.stderr)

    am_t = T.argmax(-1)

    def top1(P):
        return float((P.argmax(-1) == am_t).mean())

    kinds = np.array([dtp.kind_of_slot(int(j)) for j in am_t])
    res = {
        "args": vars(args), "n": n,
        "top1_raw": top1(RAW), "top1_ns": top1(NS),
        "top1_ns_by_kind": {k: (float((NS[kinds == q].argmax(-1) == am_t[kinds == q]).mean()) if (kinds == q).any() else None)
                            for q, k in enumerate(dtp.KINDS)},
        "top1_raw_by_kind": {k: (float((RAW[kinds == q].argmax(-1) == am_t[kinds == q]).mean()) if (kinds == q).any() else None)
                             for q, k in enumerate(dtp.KINDS)},
        "ns_agrees_with_raw": float((NS.argmax(-1) == RAW.argmax(-1)).mean()),
        "ns_changed_raw_to_right": float(((NS.argmax(-1) == am_t) & (RAW.argmax(-1) != am_t)).mean()),
        "ns_changed_raw_to_wrong": float(((NS.argmax(-1) != am_t) & (RAW.argmax(-1) == am_t)).mean()),
        "score_raw": dtp.score(RAW, T, L), "score_ns_visits": dtp.score(NS, T, L),
        "by_gen": {g: {"n": int(sum(r["gen"] == g for r in rows)),
                       "raw": float(np.mean([RAW[i].argmax() == am_t[i] for i, r in enumerate(rows) if r["gen"] == g])),
                       "ns": float(np.mean([NS[i].argmax() == am_t[i] for i, r in enumerate(rows) if r["gen"] == g]))}
                   for g in sorted({r["gen"] for r in rows})},
        "throughput": {
            "evals_per_decision": float(np.mean(evals)), "batches_per_decision": float(np.mean(batches)),
            "wall_ms": float(np.mean(wall)), "evals_per_s": float(np.sum(evals) / (np.sum(wall) / 1000.0)),
            "collect_ms": float(np.mean(coll)), "rust_tree_ms": float(np.mean(tree)),
            "tokenise_ms": float(np.mean(tokm)), "forward_ms": float(np.mean(fwd)), "feed_ms": float(np.mean(feed)),
            "max_depth_mean": float(np.mean(maxd)),
        },
    }
    print(json.dumps(res, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
