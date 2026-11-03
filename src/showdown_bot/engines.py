"""Pinned poke-engine wheels: engine/wheels/poke_engine-0.0.48+<tag>-*.whl unzipped to .cache/engine/<tag>.

The main .venv keeps an old engine (Windows locks the .pyd while jobs run), so runs that need a
specific engine put ``engine_dir(tag)`` first on the path: scripts/ladder_best.py for the whole
process, or a league spec key ``engine=<tag>`` for one bot's search worker pool (players.search
``get_pool``). Worker pools only run the searches; the main process still builds the state strings
with its own engine, so a pool engine should be the same or newer than the main one (state
strings are backward compatible, not forward).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def engine_dir(tag: str) -> Path:
    wheels = sorted((REPO / "engine" / "wheels").glob(f"poke_engine-0.0.48+{tag}-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"expected one wheel for engine {tag!r} in engine/wheels, found {len(wheels)}")
    out = REPO / ".cache" / "engine" / tag
    if not (out / "poke_engine").is_dir():
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(out)
    return out
