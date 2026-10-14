"""Convert human gen9randombattle logs (PokeChamp jsonl.gz) into value + policy training data.

    python scripts/human_replays.py --input data/external/pokechamp/gen9randombattle.jsonl.gz \
        --out data/human/run1 [--limit 2000] [--min-elo 1400] [--workers 6] [--seed 0]

Output (see docs/human-replays.md):
    <out>/<split>/shard-NNNNN/<battle_id>__p1.jsonl.gz   value records, self-play format
    <out>/policy/<split>/shard-NNNNN.jsonl.gz            one row per human decision
    <out>/manifest.json                                  counts, skip reasons, throughput

Train on it unchanged:  train.py --data <out>/train --value-target outcome --states truth
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from showdown_bot import human_replays as hr  # noqa: E402


def elo_floor(bucket: str) -> int:
    """Lower bound of an elo bucket string ("1000-1199" -> 1000, "1800+" -> 1800)."""
    digits = "".join(ch if ch.isdigit() else " " for ch in str(bucket)).split()
    return int(digits[0]) if digits else 0


def iter_games(path: Path, min_elo: int, split: str, limit: int | None):
    taken = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and taken >= limit:
                return
            record = json.loads(line)
            if elo_floor(record.get("elo", "")) < min_elo:
                continue
            if split != "all" and record.get("split") != split:
                continue
            taken += 1
            yield {
                "text": record["text"], "battle_id": record["battle_id"],
                "elo": record.get("elo"), "month_year": record.get("month_year"),
                "split": record.get("split", "train"),
            }


def iter_jobs(args, out: Path):
    buffers: dict[str, list] = {}
    counters: Counter = Counter()

    def make(split: str, games: list) -> dict:
        index = counters[split]
        counters[split] += 1
        name = f"shard-{index:05d}"
        return {
            "games": games, "seed": args.seed, "split": split, "shard": name,
            "shard_dir": str(out / split / name),
            "policy_path": None if args.no_policy else str(out / "policy" / split / f"{name}.jsonl.gz"),
        }

    for game in iter_games(Path(args.input), args.min_elo, args.split, args.limit):
        buf = buffers.setdefault(game["split"], [])
        buf.append(game)
        if len(buf) >= args.shard_games:
            yield make(game["split"], buf)
            buffers[game["split"]] = []
    for split, buf in buffers.items():
        if buf:
            yield make(split, buf)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", required=True, help="PokeChamp gen9randombattle.jsonl.gz")
    parser.add_argument("--out", required=True, help="output run directory, e.g. data/human/run1")
    parser.add_argument("--limit", type=int, default=None, help="stop after N games (after filters)")
    parser.add_argument("--min-elo", type=int, default=0,
                        help="keep games whose elo bucket starts at or above this (1000..1800)")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-games", type=int, default=250, help="games per output shard")
    parser.add_argument("--no-policy", action="store_true", help="skip the policy JSONL")
    parser.add_argument("--resume", action="store_true", help="skip shards that already finished")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    totals: Counter = Counter()
    skips: Counter = Counter()
    examples: dict[str, list[str]] = {}
    started = time.perf_counter()
    done_games = 0
    pending = set()
    jobs = iter(iter_jobs(args, out))
    exhausted = False

    def absorb(result: dict) -> None:
        nonlocal done_games
        done_games += result["games"]
        for key in ("games", "ok", "decisions", "actions", "illegal_actions", "hp_checked",
                    "hp_mismatch"):
            totals[key] += result[key]
        totals["worker_seconds"] += result["seconds"]
        skips.update(result["skips"])
        for reason, ids in result["examples"].items():
            examples.setdefault(reason, [])
            examples[reason] += ids[: max(0, 5 - len(examples[reason]))]
        elapsed = time.perf_counter() - started
        skipped = sum(skips.values())
        print(
            f"[{done_games:>7} games] ok {totals['ok']} skipped {skipped} "
            f"({100 * skipped / max(done_games, 1):.1f}%) | {done_games / elapsed:.1f} games/s "
            f"({1000 * elapsed / done_games:.0f} s per 1000) | decisions {totals['decisions']}",
            flush=True,
        )

    with ProcessPoolExecutor(max_workers=args.workers, initializer=hr._init_worker) as pool:
        while True:
            while not exhausted and len(pending) < args.workers * 2:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                done_marker = Path(job["shard_dir"]) / ".done"
                if args.resume and done_marker.exists():
                    absorb(json.loads(done_marker.read_text()))
                    continue
                pending.add(pool.submit(hr.process_chunk, job))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                absorb(future.result())

    elapsed = time.perf_counter() - started
    manifest = {
        "args": vars(args), "games": totals["games"], "ok": totals["ok"],
        "skipped": dict(skips), "skip_examples": examples,
        "decisions": totals["decisions"], "policy_rows": totals["actions"],
        "illegal_actions": totals["illegal_actions"],
        "hp_checked": totals["hp_checked"], "hp_mismatch": totals["hp_mismatch"],
        "wall_seconds": elapsed,
        "seconds_per_1000_games": 1000 * elapsed / max(totals["games"], 1),
        "schema_version": hr.recorder.SCHEMA_VERSION,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("args", "skip_examples")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
