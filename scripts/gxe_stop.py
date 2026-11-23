"""Stop a running ladder.py as soon as the account's GXE reaches a target.

    python scripts/gxe_stop.py --log data/ladder/gen7b.log --match gen7b --target 90.5

Polls the ladder log; after each finished game ("[n/N] ..." line) it reads the public
rating JSON (pokemonshowdown.com/users/<user>.json) and, at or above the target, kills the
ladder process tree straight away — between games, while it is queueing — so no battle is
abandoned.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def gxe(user: str, fmt: str) -> float | None:
    url = f"https://pokemonshowdown.com/users/{user}.json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gxe-stop"}),
                                    timeout=20) as r:
            data = json.load(r)
        return float(data["ratings"][fmt]["gxe"])
    except Exception as exc:  # network hiccup: try again next game
        print(f"gxe fetch failed: {exc}", flush=True)
        return None


def ladder_pids(match: str) -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         f"Where-Object {{ $_.CommandLine -match 'ladder.py' -and $_.CommandLine -match '{match}' }} | "
         "ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--match", required=True, help="substring of the ladder.py command line")
    p.add_argument("--target", type=float, default=90.5)
    p.add_argument("--user", default="porygonzplaya")
    p.add_argument("--format", default="gen9randombattle")
    args = p.parse_args()
    log = Path(args.log)

    def finished() -> list[str]:
        return [l for l in log.read_text(encoding="utf-8", errors="replace").splitlines()
                if re.match(r"\[\d+/\d+\]", l)]

    seen = len(finished())  # only react to games that finish after we start
    while True:
        lines = finished()
        if len(lines) > seen:
            seen = len(lines)
            time.sleep(2)  # let the server settle the rating
            value = gxe(args.user, args.format)
            print(f"{time.strftime('%H:%M:%S')} after game {seen}: gxe {value}", flush=True)
            if value is not None and value >= args.target:
                pids = ladder_pids(args.match)
                for pid in pids:
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
                print(f"target {args.target} reached: stopped ladder pids {pids}", flush=True)
                return
        if not ladder_pids(args.match):
            print("ladder process gone; exiting", flush=True)
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
