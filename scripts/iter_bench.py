"""Iterations per 1 s single-threaded search on recorded w16 world states, for one spec.

    python scripts/iter_bench.py SPEC N [--engine TAG]

The engine is the one on PYTHONPATH, or with `--engine TAG` the wheel
`engine/wheels/poke_engine-0.0.48+TAG-*.whl` (unpacked to `.cache/engine/TAG`, see
`showdown_bot.engines.engine_dir`). States: the first 40 `data/ladder/w16/records` files,
shuffled with seed 7, first N. Single process; keep N small on a loaded machine.
"""
import argparse
import glob
import gzip
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("spec")
ap.add_argument("n", type=int)
ap.add_argument("--engine", default=None, help="engine wheel tag, e.g. sdb.19")
ap.add_argument("--ms", type=int, default=1000)
ap.add_argument("--records", default=r"C:\Users\Lewis\pokemon-showdown-bot\data\ladder\w16\records")
ap.add_argument("--json", default=None, help="append {spec, engine, its: [per state]} as a JSON line")
args = ap.parse_args()

if args.engine:
    from showdown_bot.engines import engine_dir

    sys.path.insert(0, str(engine_dir(args.engine)))

import importlib.metadata as m  # noqa: E402

from showdown_bot import league  # noqa: E402
from showdown_bot.ladder_ab import arm_search_config  # noqa: E402
from showdown_bot.players._worker import search_world  # noqa: E402
import poke_engine  # noqa: E402

cfg = arm_search_config(league.parse_spec(args.spec), args.ms)
states = []
for f in sorted(glob.glob(str(Path(args.records) / "*.jsonl.gz")))[:40]:
    for line in gzip.open(f, "rt", encoding="utf-8"):
        d = json.loads(line)
        for w in d.get("worlds") or []:
            if w.get("state"):
                states.append(w["state"])
random.Random(7).shuffle(states)
its = []
for s in states[:args.n]:
    out = search_world(s, args.ms, 1, evaluator_path=cfg.evaluator, evaluator_scale=cfg.evaluator_scale,
                       prior_path=cfg.prior, prior_opp=cfg.prior_opp, c_puct=cfg.c_puct,
                       prior_eps=cfg.prior_eps, human_path=cfg.prior_human)
    its.append(out[0])
try:
    ver = m.version("poke_engine")
except m.PackageNotFoundError:
    ver = "?"
print(f"{args.spec.split(':')[0]:10s} engine {ver:16s} ({Path(poke_engine.__file__).parent.parent.name}) "
      f"n={len(its)} median {statistics.median(its):>9,.0f} mean {statistics.mean(its):>9,.0f}", flush=True)
if args.json:
    with open(args.json, "a", encoding="utf-8") as f:
        f.write(json.dumps({"spec": args.spec, "engine": ver, "its": its}) + "\n")
