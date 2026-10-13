"""Local dry run of challenge mode (docs/ladder.md): needs the local server on 127.0.0.1:8000.

Starts `scripts/ladder.py --accept-challenges --dry-run --games 1`, then a throwaway player
challenges it (1) in gen8randombattle (must be rejected) and (2) in gen9randombattle (must be
accepted and played to the end). Exits 0 on success. At most one game, one worker.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from showdown_bot import bench

REPO = Path(__file__).resolve().parent.parent
TIMEOUT_S = 240


async def main() -> int:
    out = REPO / "data" / "challenges" / ("dryrun-" + time.strftime("%Y%m%d-%H%M%S"))
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "scripts" / "ladder.py"), "--accept-challenges",
         "--dry-run", "--games", "1", "--bot", "search:time_ms=200:worlds=2", "--idle-timeout", "4", "--out", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    lines: list[str] = []
    loop = asyncio.get_running_loop()
    bot_name = None
    challenger = None
    try:
        async def read_until(pattern: str, timeout: float) -> str:
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                line = await loop.run_in_executor(None, proc.stdout.readline)
                if not line:
                    raise RuntimeError("bot exited early:\n" + "".join(lines))
                lines.append(line)
                print("bot>", line.rstrip(), flush=True)
                if re.search(pattern, line):
                    return line
            raise TimeoutError(pattern)

        m = re.search(r"Logged in as (\S+)", await read_until(r"Logged in as", 60))
        bot_name = m.group(1)
        challenger = bench.get_player_class("heuristics")(
            battle_format="gen8randombattle", max_concurrent_battles=1,
            account_configuration=AccountConfiguration("chl" + secrets.token_hex(4), None),
            server_configuration=LocalhostServerConfiguration, log_level=logging.ERROR,
        )
        await bench._wait_for_player_logins([challenger], 30)

        print(f"--- {challenger.username} challenges {bot_name} in gen8randombattle", flush=True)
        await challenger.ps_client.challenge(bot_name, "gen8randombattle", None)
        await read_until(r"REJECT", 30)
        await challenger.ps_client.send_message(f"/cancelchallenge {bot_name}")
        await asyncio.sleep(1)

        print(f"--- {challenger.username} challenges {bot_name} in gen9randombattle", flush=True)
        challenger._format = "gen9randombattle"
        await challenger.ps_client.challenge(bot_name, "gen9randombattle", None)
        await read_until(r"ACCEPT", 30)
        await read_until(r"^\[1/1\]", TIMEOUT_S)
        await read_until(r"Challenges accepted", 60)
        rc = await loop.run_in_executor(None, lambda: proc.wait(timeout=60))
        text = "".join(lines)
        ok = rc == 0 and "REJECT" in text and "ACCEPT" in text
        print("DRY RUN", "PASSED" if ok else "FAILED", flush=True)
        return 0 if ok else 1
    finally:
        if challenger is not None:
            await bench._disconnect_player(challenger)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

