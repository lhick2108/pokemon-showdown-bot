"""Play gen9randombattle on the PUBLIC Showdown ladder with our search player.

    python scripts/ladder.py --games 1
    python scripts/ladder.py --games 20 --bot search-net:evaluator=models/x.pev:scale=1.5
    python scripts/ladder.py --games 3 --dry-run     # local server 127.0.0.1:8000, throwaway account

Credentials: SHOWDOWN_USERNAME / SHOWDOWN_PASSWORD from the environment or the
gitignored .env in the repo root (never printed, logged or written to records).
One battle at a time by default; every decision is hard-capped (--move-cap, default
5 s) with a heuristic fallback. See docs/ladder.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import sys
import time
from pathlib import Path

from poke_env.ps_client import (
    AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration,
)

from showdown_bot import bench, ladder, league

REPO = Path(__file__).resolve().parent.parent
logger = logging.getLogger("ladder")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--games", type=bench._positive_int, required=True,
                   help="ladder games to play, then stop")
    p.add_argument("--bot", default="search", help="league spec syntax (default: search)")
    p.add_argument("--dry-run", action="store_true",
                   help="ladder on the LOCAL server 127.0.0.1:8000 with throwaway accounts")
    p.add_argument("--max-concurrent", type=bench._positive_int, default=1)
    p.add_argument("--out", type=Path, default=None, help="default data/ladder/<run-id>")
    p.add_argument("--move-cap", type=bench._positive_seconds, default=ladder.DEFAULT_MOVE_CAP_S,
                   help="hard cap in seconds per decision (default 5)")
    p.add_argument("--save-replays", action="store_true",
                   help="ask the server to save each replay (/savereplay); default off")
    p.add_argument("--env-file", type=Path, default=REPO / ".env")
    p.add_argument("--game-timeout", type=bench._positive_seconds, default=1800,
                   help="forfeit a battle still running after this many seconds (last resort)")
    p.add_argument("--connect-timeout", type=bench._positive_seconds, default=30)
    p.add_argument("--max-reconnects", type=int, default=5)
    return p


class Run:
    def __init__(self, args, out: Path, creds):
        self.args, self.out, self.creds = args, out, creds
        self.games: list[ladder.LadderGame] = []
        self.players: list = []
        self.spec = league.parse_spec(args.bot)
        if args.dry_run:
            self.server = LocalhostServerConfiguration
        else:
            self.server = ShowdownServerConfiguration

    def account(self) -> AccountConfiguration:
        if self.args.dry_run:
            return AccountConfiguration("ldr" + secrets.token_hex(4), None)
        return self.creds.account()

    def on_finish(self, player, battle) -> None:
        game = ladder.game_from_battle(battle, player.username)
        log = self.out / "logs" / f"{ladder.safe_filename(battle.battle_tag)}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        reqs = getattr(player, "battle_requests", {}).get(battle.battle_tag, [])
        events = [f">{battle.battle_tag}", *battle._build_replay_events()]
        # +1: the header line shifts event positions
        text = ladder.log_with_requests(events, [(pos + 1, raw) for pos, raw in reqs])
        log.write_text(text + "\n", encoding="utf-8")
        game.log_file = str(log.relative_to(self.out))
        ladder.append_game(self.out / "games.jsonl", game)
        self.games.append(game)
        rating = (f"{game.rating_before}->{game.rating_after}"
                  if game.rating_after is not None else "n/a")
        print(f"[{len(self.games)}/{self.args.games}] {game.battle_tag} vs {game.opponent}: "
              f"{game.result} in {game.turns} turns, rating {rating}", flush=True)

    def recorder(self):
        # per-decision search records (worlds, policy) so ladder blunders can be replayed
        if getattr(self, "_recorder", None) is None:
            from showdown_bot.recorder import GameRecorder
            self._recorder = GameRecorder(self.out / "records", truth=False)
        return self._recorder

    def make_player(self):
        cfg = league.LeagueConfig(out=self.out, connect_timeout=self.args.connect_timeout)
        player = league._make_player(
            self.spec, cfg, self.recorder(), self.args.max_concurrent, secrets.randbits(16),
            account=self.account(), server=self.server,
            wrap=lambda cls: ladder.with_ladder_hooks(
                cls, move_cap_s=self.args.move_cap, on_finish=self.on_finish,
                save_server_replay=self.args.save_replays,
            ),
            extra={"save_replays": False},
        )
        self.players.append(player)
        return player

    async def unfinished(self, player) -> list[str]:
        return [t for t, b in player.battles.items() if not b.finished]

    async def forfeit(self, player) -> None:
        for tag in await self.unfinished(player):
            logger.warning("FORFEIT (last resort) %s", tag)
            print(f"WARNING: forfeiting {tag}", flush=True)
            try:
                await player.ps_client.send_message("/forfeit", tag)
            except Exception:
                logger.exception("forfeit failed for %s", tag)

    @staticmethod
    def disconnected(player) -> bool:
        ws = getattr(player.ps_client, "websocket", None)
        return ws is not None and getattr(ws, "close_code", None) is not None

    async def play(self) -> None:
        args = self.args
        player = self.make_player()
        opponent = None
        opp_task = None
        if args.dry_run:
            cls = bench.get_player_class("heuristics")
            opponent = cls(
                battle_format="gen9randombattle", max_concurrent_battles=1,
                account_configuration=AccountConfiguration("ldo" + secrets.token_hex(4), None),
                server_configuration=LocalhostServerConfiguration, log_level=logging.ERROR,
            )
            await bench._wait_for_player_logins([opponent], args.connect_timeout)
            opp_task = asyncio.create_task(opponent.ladder(args.games + 2))
        try:
            await bench._wait_for_player_logins([player], args.connect_timeout)
            print(f"Logged in as {player.username} on "
                  f"{'LOCAL dry-run server' if args.dry_run else 'public server'}; "
                  f"bot {self.spec.label}; move cap {args.move_cap:g}s", flush=True)
            reconnects = 0
            while len(self.games) < args.games:
                remaining = args.games - len(self.games)
                chunk = 1 if args.max_concurrent == 1 else remaining
                task = asyncio.create_task(player.ladder(chunk))
                deadline = time.monotonic() + args.game_timeout * chunk
                reason = None
                while not task.done():
                    await asyncio.sleep(1.0)
                    if self.disconnected(player):
                        reason = "disconnect"
                        break
                    if time.monotonic() > deadline:
                        reason = "timeout"
                        break
                if reason is None:
                    exc = task.exception()
                    if exc is not None:
                        reason = f"error {type(exc).__name__}"
                if reason is None:
                    continue
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                logger.warning("ladder interrupted: %s", reason)
                print(f"WARNING: ladder interrupted ({reason})", flush=True)
                if reason == "timeout":
                    await self.forfeit(player)
                if reconnects >= args.max_reconnects:
                    print("giving up: too many reconnects", flush=True)
                    break
                reconnects += 1
                await bench._disconnect_player(player)
                await asyncio.sleep(min(60, 5 * 2 ** reconnects))
                player = self.make_player()  # same account, fresh connection
                await bench._wait_for_player_logins([player], args.connect_timeout)
                # the server re-sends rooms of battles still running: let them finish
                end = time.monotonic() + 300
                await asyncio.sleep(3)
                while await self.unfinished(player) and time.monotonic() < end:
                    await asyncio.sleep(1)
                await self.forfeit(player)
        finally:
            if opp_task is not None:
                opp_task.cancel()
                await asyncio.gather(opp_task, return_exceptions=True)
            for p in self.players + ([opponent] if opponent else []):
                await bench._disconnect_player(p)

    def summary(self) -> str:
        caps = sum(getattr(type(p), "cap_hits", 0) for p in self.players[:1])
        text = ladder.summarize(self.games) + f"\nDecisions over the time cap: {caps}"
        stats = [p for p in self.players if hasattr(p, "stats")]
        if stats:
            s = stats[-1].stats
            text += f"\nSearch fallbacks: {s.fallbacks} of {s.decisions} decisions"
        return text


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    creds = None
    if not args.dry_run:
        try:
            creds = ladder.load_credentials(env_file=args.env_file)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        ladder.install_redaction(creds)
        if "bot" not in creds.username.lower():
            print("note: the account name should say it is a bot (politeness rule, docs/ladder.md)",
                  file=sys.stderr)
    try:
        league.parse_spec(args.bot)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-dry" if args.dry_run else "")
    out = args.out or REPO / "data" / "ladder" / run_id
    out.mkdir(parents=True, exist_ok=True)
    run = Run(args, out, creds)
    try:
        asyncio.run(run.play())
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    summary = run.summary()
    print(summary)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return 0 if len(run.games) >= args.games else 1


if __name__ == "__main__":
    raise SystemExit(main())
