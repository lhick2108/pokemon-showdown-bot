"""Run scripts/ladder.py with the promoted config on a pinned engine wheel.

    python scripts/ladder_best.py --check                 # verify engine, models and spec; no connection
    python scripts/ladder_best.py --games 300             # public ladder, best:worlds=16
    python scripts/ladder_best.py --engine sdb.28 --games 3 --dry-run

The main .venv keeps an old engine (Windows locks the .pyd while jobs run), so the
wheel engine/wheels/poke_engine-0.0.48+<engine>-*.whl is unzipped once to
.cache/engine/<engine> and put first on PYTHONPATH with src/. Every other argument
goes to scripts/ladder.py; --bot, --move-cap and --search-workers get the defaults
below unless given.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE = "sdb.27"  # the engine stack789v5 was validated on (4-world SPRT, 2026-10-20)
DEFAULTS = {"--bot": "best:worlds=16", "--move-cap": "10", "--search-workers": "16"}


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
    p.add_argument("--check", action="store_true", help="load engine, models and spec, then exit")
    args, rest = p.parse_known_args(argv)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(engine_dir(args.engine)), str(REPO / "src")])
    bot = next((rest[i + 1] for i, a in enumerate(rest[:-1]) if a == "--bot"), DEFAULTS["--bot"])
    for flag, value in DEFAULTS.items():
        if flag not in rest and not any(a.startswith(flag + "=") for a in rest):
            rest += [flag, value]

    if args.check:
        probe = (
            "import importlib.metadata as m, poke_engine, showdown_bot.league as L\n"
            f"spec = L.parse_spec({bot!r})\n"
            "from pathlib import Path\n"
            "for k in ('evaluator', 'prior'):\n"
            "    v = spec.param(k)\n"
            "    assert v is None or Path(v).is_file(), f'missing {k}: {v}'\n"
            "from showdown_bot import ladder_ab\n"
            "cfg = ladder_ab.arm_search_config(spec, 1000)\n"
            "print('engine', m.version('poke_engine'), 'from', poke_engine.__file__)\n"
            "print('spec  ', spec.text)\n"
            "print('config', cfg)\n"
        )
        return subprocess.call([sys.executable, "-c", probe], env=env, cwd=REPO)
    return subprocess.call([sys.executable, str(REPO / "scripts" / "ladder.py"), *rest], env=env, cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
