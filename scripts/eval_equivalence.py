"""Bit-exact equivalence of learned evaluations between two engine wheels.

    python scripts/eval_equivalence.py --engine sdb.28 --engine sdb.29 [--n 2000] [MODEL ...]

Runs itself once per engine (a subprocess each, so two poke_engine builds never share a
process), evaluating ``Evaluator.logit`` of every value net (.pev) and
``PolicyPrior.option_logits`` (both sides) of every prior (.ppol) on the same recorded ladder
states (the iter_bench pool, shuffled with seed 7), and compares the float32 bit patterns.
Exit code 0 = identical everywhere. Single process at a time, a few seconds per model.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "models/gen7-res-root-mlp16.pev", "models/ft-v5-h128.pev", "models/ft-v4-h128.pev",
    "models/human-v4-h128.pev", "models/prior-gen789-h32.ppol", "models/prior-gen56-h32.ppol",
]


def load_states(records: str, n: int) -> list[str]:
    states = []
    for f in sorted(glob.glob(str(Path(records) / "*.jsonl.gz")))[:40]:
        for line in gzip.open(f, "rt", encoding="utf-8"):
            d = json.loads(line)
            for w in d.get("worlds") or []:
                if w.get("state"):
                    states.append(w["state"])
    random.Random(7).shuffle(states)
    return states[:n]


def worker(engine: str, models: list[str], records: str, n: int) -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from showdown_bot.engines import engine_dir

    sys.path.insert(0, str(engine_dir(engine)))
    import poke_engine as pe

    states = [pe.State.from_string(s) for s in load_states(records, n)]
    out = {"engine": pe.__file__, "models": {}}
    for m in models:
        bits = []
        if m.endswith(".ppol"):
            pr = pe.PolicyPrior.load(m)
            for s in states:
                for side in (1, 2):
                    bits += [struct.unpack("<I", struct.pack("<f", z))[0] for _, _, z in pr.option_logits(s, side)]
        else:
            ev = pe.Evaluator.load(m)
            bits = [struct.unpack("<I", struct.pack("<f", ev.logit(s)))[0] for s in states]
        out["models"][m] = bits
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--engine", action="append", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--records", default=r"C:\Users\Lewis\pokemon-showdown-bot\data\ladder\w16\records")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    models = [str(REPO / m) if not Path(m).is_absolute() else m for m in (args.models or DEFAULT_MODELS)]
    if args.worker:
        print(json.dumps(worker(args.engine[0], models, args.records, args.n)))
        return 0
    res = []
    for e in args.engine:
        cmd = [sys.executable, __file__, "--worker", "--engine", e, "--n", str(args.n),
               "--records", args.records, *models]
        res.append(json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout))
        print(f"{e}: {res[-1]['engine']}")
    ok = True
    for m in models:
        a = res[0]["models"][m]
        for e, r in zip(args.engine[1:], res[1:]):
            b = r["models"][m]
            diff = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            ok &= diff == 0
            print(f"{Path(m).name:32s} {len(a):6d} values  {args.engine[0]} vs {e}: "
                  f"{'identical' if diff == 0 else f'{diff} DIFFER'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
