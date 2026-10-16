"""Offline diagnostic for ``mode=bayes``: argmax vs per-world-Nash average vs the Bayesian solve.

Re-runs the ladder root search (one MCTS per recorded world, joint matrices exported) for
a sample of recorded ladder decisions, then derives all three move choices from the SAME
per-world matrices:

  argmax  visit-share average over worlds (the ladder default), best move
  nash    per-world matrix-game solutions averaged (``aggregate_nash``), best move
  bayes   one strategy shared by all worlds, per-world best-responding opponent
          (``aggregate_bayes``), best move

and scores each pick under the Bayesian objective ``sum_w p_w min_j (x^T M_w)_j``, with
``M_w`` the shrunk root matrix: how much expected win probability argmax / nash leave on
the table *under the model*. (Bayes maximises this by construction, so the gap is a
model-internal quantity, not a measured win-rate gain.)

Also reports cell-visit statistics, a pseudocount sensitivity table, and a turn-by-turn
printout for the Incineroar loss.

    set PYTHONPATH=<worktree>\\src
    python scripts\\bayes_offline.py --n 150 --workers 4 --cache bayes_cache.pkl
"""

from __future__ import annotations

import argparse
import functools
import glob
import gzip
import json
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from showdown_bot import nash as nash_module
from showdown_bot.players import _worker
from showdown_bot.players.search import (
    aggregate_bayes, aggregate_nash, aggregate_policy,
)

MAIN = Path(r"C:\Users\Lewis\pokemon-showdown-bot")
DEFAULT_DIRS = [MAIN / "data" / "ladder" / "gen7" / "records", MAIN / "data" / "ladder" / "gen7b" / "records"]
EVALUATOR = MAIN / "models" / "gen7-res-root-mlp16.pev"
PRIOR = MAIN / "models" / "prior-gen56-h32.ppol"
FLOOR = 0.03
DEFAULT_PC = nash_module.DEFAULT_PSEUDOCOUNT


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def load_decisions(dirs, incineroar_tag: str) -> list[dict]:
    out = []
    for d in dirs:
        for path in sorted(glob.glob(str(Path(d) / "*.jsonl.gz"))):
            tag = Path(path).name.split("__")[0]
            rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
            result = next((r.get("result") for r in rows if r.get("type") == "result"), None)
            for idx, r in enumerate(rows):
                if r.get("type") != "decision" or len(r.get("worlds") or []) < 2:
                    continue
                policy = r.get("policy") or {}
                facts = r.get("opp_facts") or []
                out.append({
                    "key": f"{tag}#{idx}", "game": tag, "turn": int(r["turn"]), "result": result,
                    "states": [w["state"] for w in r["worlds"]],
                    "policy": policy, "chosen": r.get("chosen"),
                    "opp": [f.get("species") for f in facts[:1]],
                    "has_incineroar": any(f.get("species") == "incineroar" for f in facts),
                    "incin_game": incineroar_tag in tag,
                })
    return out


def nontrivial(dec: dict) -> bool:
    """Not forced: at least two choices carry real mass and the best does not dominate."""
    masses = sorted(dec["policy"].values(), reverse=True)
    return len(masses) >= 2 and masses[0] < 0.9 and masses[1] >= FLOOR


def pick_sample(decisions: list[dict], n: int, seed: int) -> list[dict]:
    incin = [d for d in decisions if d["incin_game"] and len(d["policy"]) >= 2]
    pool = [d for d in decisions if not d["incin_game"] and nontrivial(d)]
    rng = random.Random(seed)
    rng.shuffle(pool)
    return incin + pool[:n]


# ---------------------------------------------------------------------------
# search (once per decision-world)
# ---------------------------------------------------------------------------


def run_search(decisions: list[dict], workers: int, time_ms: int, cache: Path | None) -> dict:
    results: dict = {}
    if cache is not None and cache.is_file():
        with open(cache, "rb") as fh:
            results = pickle.load(fh)
    jobs = [(d["key"], k, s) for d in decisions for k, s in enumerate(d["states"])
            if (d["key"], k) not in results]
    print(f"search: {len(jobs)} world searches ({len(results)} cached), {workers} workers, {time_ms} ms")
    if jobs:
        fn = functools.partial(
            _worker.search_world, opp_names=True, prior_path=str(PRIOR), prior_opp="self",
            c_puct=2.0, prior_eps=0.03, prior_fpu=0.0,
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(fn, state, time_ms, 1, str(EVALUATOR), 1.5, "ucb1", 0.1, True, 0.0, 1.0)
                for _key, _k, state in jobs
            ]
            for n, ((key, k, _s), fut) in enumerate(zip(jobs, futs), 1):
                try:
                    results[(key, k)] = tuple(fut.result())
                except Exception as exc:  # keep going: one broken world drops out
                    print(f"  search failed for {key} world {k}: {exc!r}")
                if n % 40 == 0:
                    print(f"  {n}/{len(jobs)}")
                    if cache is not None:
                        with open(cache, "wb") as fh:
                            pickle.dump(results, fh)
    if cache is not None:
        with open(cache, "wb") as fh:
            pickle.dump(results, fh)
    return results


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def world_results(dec: dict, searched: dict) -> list[tuple]:
    return [searched[(dec["key"], k)] for k in range(len(dec["states"])) if (dec["key"], k) in searched]


def best(policy: dict[str, float]) -> str | None:
    return max(policy, key=policy.get) if policy else None


def matrices(results: list[tuple], pc: float):
    mats, names = [], []
    for r in results:
        if r[0] <= 0 or len(r) < 4 or r[3] is None or not r[1]:
            continue
        k2, cells = r[3]
        mats.append(nash_module.shrunk_matrix(len(r[1]), int(k2), cells, pc))
        names.append([c for c, _v, _s in r[1]])
    if not mats:
        return [], []
    common = [c for c in names[0] if all(c in n for n in names)]
    return [M[[n.index(c) for c in common], :] for M, n in zip(mats, names)], common


def objective(policy: dict[str, float], mats, common) -> float:
    """Bayesian objective of a policy (mass on rows outside ``common`` is dropped)."""
    x = np.array([policy.get(c, 0.0) for c in common])
    if x.sum() <= 0:
        return float("nan")
    return nash_module.bayes_value(x / x.sum(), mats)


def analyse(dec: dict, searched: dict, pc: float) -> dict | None:
    results = world_results(dec, searched)
    mats, common = matrices(results, pc)
    if not mats or len(common) < 2:
        return None
    pol_arg = aggregate_policy(results)
    pol_nash, _ = aggregate_nash(results, pc, FLOOR)
    pol_bayes, info = aggregate_bayes(results, pc, FLOOR)
    if not pol_arg or not pol_nash or not pol_bayes:
        return None
    picks = {"argmax": best(pol_arg), "nash": best(pol_nash), "bayes": best(pol_bayes)}
    pure = {c: objective({c: 1.0}, mats, common) for c in common}
    # cell visit statistics (raw, un-shrunk) for the worlds that carried a matrix
    visits = []
    for r in results:
        if len(r) >= 4 and r[3] is not None:
            k1, k2 = len(r[1]), int(r[3][0])
            grid = np.zeros((k1, k2))
            for i, j, v, _m in r[3][1]:
                grid[i, j] += v
            visits.append(grid)
    return {
        "dec": dec, "common": common, "picks": picks, "pure": pure, "n_worlds": len(mats),
        "mixed": {"argmax": objective(pol_arg, mats, common), "nash": objective(pol_nash, mats, common),
                  "bayes": objective(pol_bayes, mats, common)},
        "policies": {"argmax": pol_arg, "nash": pol_nash, "bayes": pol_bayes},
        "bayes_value": info["value"], "world_values": info["world_values"],
        "row_mismatch": info["row_mismatch"], "visits": visits,
    }


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarise(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n=== {n} decisions analysed ===")
    for a, b in (("argmax", "nash"), ("argmax", "bayes"), ("nash", "bayes")):
        d = sum(r["picks"][a] != r["picks"][b] for r in rows)
        print(f"  {a:>6} != {b:<5}: {d:3d}/{n} = {100 * d / n:4.1f}%")
    print(f"  row-set mismatch across worlds in {sum(r['row_mismatch'] for r in rows)} decisions")
    for who in ("argmax", "nash"):
        gaps = np.array([r["bayes_value"] - r["pure"][r["picks"][who]] for r in rows])
        dis = np.array([r["picks"][who] != r["picks"]["bayes"] for r in rows])
        print(f"  Bayes-objective gap, bayes mixed minus {who} PURE pick:")
        print(f"     all     : mean {gaps.mean():.4f}  median {pct(gaps, 50):.4f}  p90 {pct(gaps, 90):.4f}  max {gaps.max():.4f}")
        if dis.any():
            g = gaps[dis]
            print(f"     disagree: n={dis.sum()}  mean {g.mean():.4f}  median {pct(g, 50):.4f}  p90 {pct(g, 90):.4f}  max {g.max():.4f}")
        mg = np.array([r["bayes_value"] - r["mixed"][who] for r in rows])
        print(f"     vs {who} full mixed policy: median {pct(mg, 50):.4f}  p90 {pct(mg, 90):.4f}  max {mg.max():.4f}")
    bpure = np.array([r["bayes_value"] - r["pure"][r["picks"]["bayes"]] for r in rows])
    print(f"  bayes mixed minus bayes PURE argmax pick (cost of not mixing): median {pct(bpure, 50):.4f} p90 {pct(bpure, 90):.4f}")
    # visits
    cell = np.concatenate([g[g > 0].ravel() for r in rows for g in r["visits"]])
    allc = np.concatenate([g.ravel() for r in rows for g in r["visits"]])
    rowtot = np.concatenate([g.sum(axis=1) for r in rows for g in r["visits"]])
    print("  cell visits (raw, per world root matrix):")
    print(f"     cells total {allc.size}, never visited {100 * (allc == 0).mean():.1f}%")
    print(f"     visited cells: median {pct(cell, 50):.0f}  p10 {pct(cell, 10):.0f}  p90 {pct(cell, 90):.0f}")
    print(f"     share of ALL cells with <{DEFAULT_PC:g} visits (pseudocount-dominated): {100 * (allc < DEFAULT_PC).mean():.1f}%")
    print(f"     share of ALL cells with <50 visits: {100 * (allc < 50).mean():.1f}%")
    print(f"     row visit totals: median {pct(rowtot, 50):.0f}  p10 {pct(rowtot, 10):.0f}")
    sizes = [(g.shape[0], g.shape[1]) for r in rows for g in r["visits"]]
    print(f"     matrix size: median {int(np.median([s[0] for s in sizes]))} x {int(np.median([s[1] for s in sizes]))}")


def sensitivity(decisions: list[dict], searched: dict, pcs: list[float]) -> None:
    print("\n=== pseudocount sensitivity (same matrices, bayes pick) ===")
    base: dict = {}
    for pc in pcs:
        picks = {}
        for d in decisions:
            res = world_results(d, searched)
            pol, _ = aggregate_bayes(res, pc, FLOOR)
            ag, _ = aggregate_nash(res, pc, FLOOR)
            picks[d["key"]] = (best(pol), best(ag), best(aggregate_policy(res)))
        base[pc] = picks
    ref = base[pcs[0]]
    for pc in pcs:
        picks = base[pc]
        valid = [k for k in picks if picks[k][0] and ref[k][0]]
        chg = sum(picks[k][0] != ref[k][0] for k in valid)
        dis_arg = sum(picks[k][0] != picks[k][2] for k in valid)
        dis_nash = sum(picks[k][0] != picks[k][1] for k in valid)
        print(f"  pc={pc:>5g}: bayes pick changed vs pc={pcs[0]:g}: {chg}/{len(valid)} ({100 * chg / len(valid):.1f}%); "
              f"bayes!=argmax {dis_arg} ({100 * dis_arg / len(valid):.1f}%), bayes!=nash {dis_nash} ({100 * dis_nash / len(valid):.1f}%)")


def incineroar_report(rows: list[dict]) -> None:
    print("\n=== Incineroar game, turn by turn ===")
    for r in sorted((r for r in rows if r["dec"]["incin_game"]), key=lambda r: r["dec"]["turn"]):
        d = r["dec"]
        top = sorted(r["pure"].items(), key=lambda kv: -kv[1])[:3]
        flag = "" if len({r["picks"]["argmax"], r["picks"]["nash"], r["picks"]["bayes"]}) == 1 else "  <-- DIFFER"
        print(f"T{d['turn']:>2} opp={d['opp'][0]} incin_in_play={d['has_incineroar']} played={d['chosen']}  "
              f"argmax={r['picks']['argmax']} nash={r['picks']['nash']} bayes={r['picks']['bayes']}{flag}")
        pb = sorted(r["policies"]["bayes"].items(), key=lambda kv: -kv[1])[:3]
        print(f"      bayes value {r['bayes_value']:.3f} worlds {[round(v, 3) for v in r['world_values']]}; "
              f"bayes x {[(c, round(p, 2)) for c, p in pb]}")
        print(f"      pure-pick values: argmax {r['pure'].get(r['picks']['argmax'], float('nan')):.3f} "
              f"nash {r['pure'].get(r['picks']['nash'], float('nan')):.3f} bayes {r['pure'].get(r['picks']['bayes'], float('nan')):.3f}; "
              f"top pure {[(c, round(v, 3)) for c, v in top]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="non-Incineroar decisions to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-ms", type=int, default=1000)
    ap.add_argument("--cache", type=Path, default=None, help="pickle of search results (resumable)")
    ap.add_argument("--dirs", type=Path, nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--incineroar-game", default="2699130927", help="battle tag fragment of the Incineroar loss")
    ap.add_argument("--pcs", type=float, nargs="*", default=[DEFAULT_PC, 1.0, 20.0])
    args = ap.parse_args()

    decisions = pick_sample(load_decisions(args.dirs, args.incineroar_game), args.n, args.seed)
    print(f"{len(decisions)} decisions ({sum(d['incin_game'] for d in decisions)} from the Incineroar game)")
    searched = run_search(decisions, args.workers, args.time_ms, args.cache)

    rows = [a for d in decisions if (a := analyse(d, searched, args.pcs[0])) is not None]
    generic = [r for r in rows if not r["dec"]["incin_game"]]
    summarise(generic)
    sens_decs = [r["dec"] for r in generic]
    sensitivity(sens_decs, searched, args.pcs)
    disagree = [r for r in generic if r["picks"]["bayes"] != r["picks"]["argmax"]]
    print("\n=== disagreements (bayes != argmax), largest Bayes-objective gaps ===")
    for r in sorted(disagree, key=lambda r: -(r["bayes_value"] - r["pure"][r["picks"]["argmax"]]))[:10]:
        d = r["dec"]
        print(f"  {d['key'][-30:]} T{d['turn']}: argmax={r['picks']['argmax']} ({r['pure'][r['picks']['argmax']]:.3f}) "
              f"nash={r['picks']['nash']} ({r['pure'][r['picks']['nash']]:.3f}) "
              f"bayes={r['picks']['bayes']} (pure {r['pure'][r['picks']['bayes']]:.3f}, mixed {r['bayes_value']:.3f}) "
              f"world values {[round(v, 2) for v in r['world_values']]}")
    print("\n=== Incineroar game: summary of the same statistics (all decisions of that game) ===")
    incin_rows = [r for r in rows if r["dec"]["incin_game"]]
    if incin_rows:
        incineroar_report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
