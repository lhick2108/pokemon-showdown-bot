"""Live history vs the exporter's history on real games (bead showdown-bot-589.68.7, docs/net-search.md).

Two token-policy bots (``hist=1``: the live rows of ``history_tokens``) play each other on the local Showdown server
(``scripts/start_server.ps1``, port 8000) and record every decision with ``recorder.GameRecorder`` (truth state, worlds,
revealed facts, chosen action) exactly like a self-play search bot. For every decision the rows the bot actually fed
its net are compared with what the shard exporter (``tokens.game_samples``) builds from the two recordings of the
same game, field by field. Run it with the engine wheel and src on PYTHONPATH::

    PYTHONPATH="<wheel>;<worktree>\\src" .venv\\Scripts\\python.exe scripts\\history_live_check.py --games 12 \\
        --model models\\tok-d256.pt --out-json history_live.json

The match table counts every (decision, row) pair of the 16 rows: ``pad`` rows (both all-zero) count as matches
and are reported separately from the ``real`` rows (at least one side non-zero).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

CAT_FIELDS = ("side", "kind", "action_id", "actor_species")
CONT_FIELDS = ("turns_ago", "actor_hp_change", "other_hp_change", "actor_side_fainted", "other_side_fainted")


def make_player_class():
    from showdown_bot import recorder as recorder_module
    from showdown_bot import tokens
    from showdown_bot.players.search import _opp_facts
    from showdown_bot.players.tokenpolicy import TokenPolicyPlayer

    class RecordingTokenPlayer(TokenPolicyPlayer):
        """Token-policy bot that records its decisions like a search bot and keeps the history rows it used."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.live_rows: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
            self._n_dec: dict[str, int] = {}

        def _live_hist(self, battle: Any, state0: str, reveal: Any) -> Any:
            rows = super()._live_hist(battle, state0, reveal)
            self._pending_rows = (np.array(rows[0]), np.array(rows[1]))
            return rows

        async def _choose(self, battle: Any) -> Any:
            order = await super()._choose(battle)
            if len(battle.valid_orders) > 1 and getattr(self, "_pending_rows", None) is not None and self.last_choice:
                tag = battle.battle_tag
                idx = self._n_dec.get(tag, 0)
                self._n_dec[tag] = idx + 1
                self.live_rows[(tag, idx)] = self._pending_rows
                self._pending_rows = None
                self._record(battle)
            return order

        def _record(self, battle: Any) -> None:
            rec = self._recorder
            tag = battle.battle_tag
            if not rec.has_game(tag, self.username):
                rec.start_game(tag, self.username, role=getattr(battle, "player_role", None), battle_format=self.format)
            truth, status = self._truth_state(battle)
            trackers = self.opponent_model.update(battle)
            rec.add_decision(tag, self.username, {
                "turn": battle.turn, "side": "one",
                "worlds": [{"state": s, "total_visits": 0, "avg_score": None, "moves": []} for s in self.last_states],
                "policy": self.last_policy, "chosen": self.last_choice,
                "truth": {"state": truth} if truth else None, "truth_status": status,
                "opp_facts": _opp_facts(battle, trackers),
            })

        def _battle_finished_callback(self, battle: Any) -> None:
            super()._battle_finished_callback(battle)

    return RecordingTokenPlayer


async def play(args: argparse.Namespace, rec_dir: Path) -> tuple[Any, Any]:
    from poke_env import AccountConfiguration

    from showdown_bot import bench
    from showdown_bot.recorder import GameRecorder

    cls = make_player_class()
    server = bench.local_server(args.port)
    recorder = GameRecorder(rec_dir, truth=True)
    players = []
    for i in range(2):
        players.append(cls(
            model=args.model, device=args.device, hist=1, battle_format="gen9randombattle", server_configuration=server,
            account_configuration=AccountConfiguration(f"hl{i}" + secrets.token_hex(3), None),
            seed=args.seed + i, log_level=40, recorder=recorder,
        ))
    await asyncio.wait_for(players[0].battle_against(players[1], n_battles=args.games), timeout=args.timeout)
    return players


def compare(args: argparse.Namespace, rec_dir: Path, players: list[Any]) -> dict[str, Any]:
    from collections import defaultdict

    from showdown_bot import recorder as recorder_module
    from showdown_bot import tokens

    vocab, fields, lay = tokens.Vocab(), tokens._shard_fields(), tokens.layout()
    opts = tokens.Options(reconstruct_q=False)
    by_name = {p.username: p for p in players}
    files: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(rec_dir.glob("*.jsonl.gz")):
        files[f.name.split("__")[0]].append(f)
    cat_ok = Counter()
    cat_n = Counter()
    cont_ok = Counter()
    cont_n = Counter()
    rows_exact = Counter()
    kinds = Counter()
    decisions = 0
    exact_decisions = 0
    misc = Counter()
    examples: list[str] = []
    for tag, fl in files.items():
        if len(fl) != 2:
            misc["games_without_two_files"] += 1
            continue
        games = [recorder_module.read_game(f) for f in fl]
        for gi, g in enumerate(games):
            peer = games[1 - gi]
            user = g["header"]["username"]
            player = by_name[user]
            stats: Counter = Counter()
            samples = tokens.game_samples(g, peer, source_idx=0, kind="selfplay", opts=opts, vocab=vocab, fields=fields,
                                          lay=lay, stats=stats)
            by_idx = {int(s["decision_idx"]): s for s in samples}
            for idx, d in enumerate(g["decisions"]):
                s = by_idx.get(idx)
                live = player.live_rows.get((g["header"]["battle_tag"], idx))
                if s is None or live is None:
                    misc["decision_without_sample_or_live_rows"] += 1
                    continue
                decisions += 1
                ec, ef = np.asarray(s["hist_cat"]), np.asarray(s["hist_cont"], np.float32)
                lc, lf = live[0].astype(np.int16), live[1].astype(np.float32)
                lf16 = lf.astype(np.float16).astype(np.float32)  # the shards store float16
                for k, name in enumerate(CAT_FIELDS):
                    real = (ec[:, 0] > 0) | (lc[:, 0] > 0)
                    cat_n[name] += 16
                    cat_ok[name] += int((ec[:, k] == lc[:, k]).sum())
                    cat_n[name + "@real"] += int(real.sum())
                    cat_ok[name + "@real"] += int(((ec[:, k] == lc[:, k]) & real).sum())
                for k, name in enumerate(CONT_FIELDS):
                    real = (ec[:, 0] > 0) | (lc[:, 0] > 0)
                    close = np.abs(ef[:, k] - lf16[:, k]) <= 2e-3
                    cont_n[name] += 16
                    cont_ok[name] += int(close.sum())
                    cont_n[name + "@real"] += int(real.sum())
                    cont_ok[name + "@real"] += int((close & real).sum())
                same_rows = (ec == lc).all(1) & (np.abs(ef - lf16) <= 2e-3).all(1)
                rows_exact["rows"] += 16
                rows_exact["rows_match"] += int(same_rows.sum())
                real = (ec[:, 0] > 0) | (lc[:, 0] > 0)
                rows_exact["real_rows"] += int(real.sum())
                rows_exact["real_rows_match"] += int((same_rows & real).sum())
                for r in np.flatnonzero(real & ~same_rows):
                    # which kind of disagreement: own/opponent row, unknown on one side only, other
                    side = int(ec[r, 0] or lc[r, 0])
                    who = "own" if side == 1 else "opp"
                    if ec[r, 1] != lc[r, 1] and (ec[r, 1] == 0 or lc[r, 1] == 0):
                        kinds[f"{who}: action unknown on {'exporter' if ec[r, 1] == 0 else 'live'} side only"] += 1
                    elif ec[r, 1] != lc[r, 1] or ec[r, 2] != lc[r, 2]:
                        kinds[f"{who}: different action"] += 1
                    elif ec[r, 3] != lc[r, 3]:
                        kinds[f"{who}: different actor species"] += 1
                    else:
                        kinds[f"{who}: only hp / faint numbers differ"] += 1
                    if len(examples) < 12 and kinds.total() <= 40:
                        examples.append(f"{tag} dec {idx} row {r}: exporter {ec[r].tolist()} {np.round(ef[r], 3).tolist()} | "
                                        f"live {lc[r].tolist()} {np.round(lf16[r], 3).tolist()}")
                exact_decisions += int(same_rows.all())
    def rate(ok: Counter, n: Counter) -> dict[str, float]:
        return {k: (ok[k] / n[k] if n[k] else float("nan")) for k in n}
    return {
        "decisions": decisions, "decisions_all_16_rows_equal": exact_decisions,
        "cat_match": rate(cat_ok, cat_n), "cont_match": rate(cont_ok, cont_n),
        "rows": dict(rows_exact),
        "row_match_all": rows_exact["rows_match"] / max(rows_exact["rows"], 1),
        "row_match_real": rows_exact["real_rows_match"] / max(rows_exact["real_rows"], 1),
        "mismatch_kinds": dict(kinds), "examples": examples, "misc": dict(misc),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"C:\Users\Lewis\pokemon-showdown-bot\models\tok-d256.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--keep", default=None, help="keep the recordings in this directory")
    ap.add_argument("--out-json")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="histlive-"))
    t0 = time.time()
    try:
        players = asyncio.run(play(args, tmp))
        print(f"played {sum(p.n_finished_battles for p in players) // 2} games in {time.time() - t0:.0f}s", file=sys.stderr)
        res = compare(args, tmp, players)
    finally:
        if args.keep:
            shutil.copytree(tmp, args.keep, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(res, indent=1))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
