"""Run scripts/ladder.py with the promoted config on a pinned engine wheel.

    python scripts/ladder_best.py --check                 # verify engine, models and spec; no connection
    python scripts/ladder_best.py --games 300             # public ladder, best:worlds=16
    python scripts/ladder_best.py --engine sdb.27 --games 3 --dry-run
    python scripts/ladder_best.py --games 600 --arm self=best:worlds=16 \
        --arm human=best:worlds=16:prior_opp=human:prior_human=models/human-policy.npz

The main .venv keeps an old engine (Windows locks the .pyd while jobs run), so the
wheel engine/wheels/poke_engine-0.0.48+<engine>-*.whl is unzipped once to
.cache/engine/<engine> and put first on PYTHONPATH with src/. Every other argument
goes to scripts/ladder.py; --bot (unless --arm is given), --move-cap and
--search-workers get the defaults below unless given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# sdb.27 is the engine stack789v5 was SPRT-tested on (2026-10-20); sdb.28 adds fidelity fixes only
ENGINE = "sdb.28"
DEFAULTS = {"--bot": "best:worlds=16", "--move-cap": "10", "--search-workers": "16"}

CHECK = """
import importlib.metadata as m, sys
from pathlib import Path
import poke_engine
from showdown_bot import ladder_ab, league
print("engine", m.version("poke_engine"), "from", poke_engine.__file__)
for text in sys.argv[1:]:
    spec = league.parse_spec(text)
    for key in ("evaluator", "prior", "prior_human"):
        value = spec.param(key)
        assert value is None or Path(value).is_file(), f"{spec.label}: missing {key} file {value}"
    cfg = ladder_ab.arm_search_config(spec, 1000)
    print("spec  ", spec.text)
    print("      ", {k: getattr(cfg, k) for k in ("worlds", "mode", "evaluator", "prior", "prior_opp",
                                                  "prior_human", "c_puct")})
"""


def engine_dir(tag: str) -> Path:
    wheels = sorted((REPO / "engine" / "wheels").glob(f"poke_engine-0.0.48+{tag}-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel for {tag} in engine/wheels, found {len(wheels)}")
    out = REPO / ".cache" / "engine" / tag
    if not (out / "poke_engine").is_dir():
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", default=ENGINE, help=f"engine wheel tag (default {ENGINE})")
    p.add_argument("--check", action="store_true", help="load engine, models and specs, then exit")
    args, rest = p.parse_known_args(argv)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(engine_dir(args.engine)), str(REPO / "src")])
    arms = [rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--arm"]
    bots = arms or [next((rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--bot"), DEFAULTS["--bot"])]
    for flag, value in DEFAULTS.items():
        if flag == "--bot" and arms:
            continue
        if flag not in rest and not any(a.startswith(flag + "=") for a in rest):
            rest += [flag, value]

    if args.check:
        return subprocess.call([sys.executable, "-c", CHECK, *bots], env=env, cwd=REPO)
    return subprocess.call([sys.executable, str(REPO / "scripts" / "ladder.py"), *rest], env=env, cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
