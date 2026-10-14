#!/usr/bin/env python
"""Fixed evaluation gauntlet: one candidate bot against a fixed opponent set.

    python scripts/gauntlet.py --candidate "search-net:evaluator=models/x.pev:scale=1.5:mode=argmax:time_ms=1000:worlds=4" \
        --games 100 --out data/gauntlet/x.jsonl --league-args "--search-workers 8"

Each opponent is played in chunks by a fresh ``python -m showdown_bot.league`` subprocess (2 bots,
``--games-per-pair`` = chunk size, own output dir), so a stalled or crashed chunk is killed after
``--chunk-timeout`` and its finished games are still kept. Every finished game is appended to the
``--out`` jsonl at once; rerunning with the same ``--out`` resumes (only the missing games are
played). Prints per-opponent score (win = 1, tie = 0.5) with a Wilson 95% CI and the overall mean
of the per-opponent scores with a bootstrap 95% CI. See docs/league.md.

Opponent set: ``--opponents-file FILE.json`` is ``{"label": "league spec", ...}`` or
``{"opponents": {...}}``; the default is DEFAULT_OPPONENTS. ``--foulplay`` adds Foul Play (needs
scripts/setup_foulplay.ps1 and the local server on port 8000).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_OPPONENTS = {
    "gen2": "search-net:evaluator=models/gen2-res-searchabs-mlp16.pev:scale=1.5:mode=argmax:time_ms=1000:worlds=4",
    "search": "search:mode=argmax:time_ms=1000:worlds=4",
    "human": "human:model=models/human-policy-heldout.npz",
}
FOULPLAY_SPEC = "foulplay"
CAND = "cand"


def wilson(points: float, n: int) -> tuple[float, float]:
    """95% Wilson interval of a score with ``points`` out of ``n`` (ties may make it fractional)."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.96
    p = points / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - margin), min(1.0, centre + margin)


def load_opponents(path: Path | None, foulplay: bool) -> dict[str, str]:
    opps = dict(DEFAULT_OPPONENTS)
    if path:
        data = json.loads(path.read_text(encoding="utf-8"))
        opps = dict(data.get("opponents", data))
    if foulplay:
        opps["foulplay"] = FOULPLAY_SPEC
    for label in opps:
        if label == CAND or any(c in label for c in ",=:"):
            raise SystemExit(f"bad opponent label {label!r}")
    return opps


def read_log(path: Path) -> tuple[dict | None, dict[str, list[float]], dict[str, int]]:
    """(header, {opponent: [score per game]}, {opponent: chunks started})."""
    header, scores, chunks = None, {}, {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if "header" in rec:
                header = rec["header"]
            elif "score" in rec:
                scores.setdefault(rec["opponent"], []).append(rec["score"])
            elif "chunk" in rec:
                chunks[rec["opponent"]] = max(chunks.get(rec["opponent"], 0), rec["chunk"] + 1)
    return header, scores, chunks


def append(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def run_chunk(opp: str, spec: str, cand_spec: str, n: int, idx: int, args, out_dir: Path) -> list[dict]:
    """Play ``n`` games in one league subprocess; return its finished games as score records."""
    chunk_dir = out_dir / f"{opp}-{idx:03d}"
    cmd = [sys.executable, "-m", "showdown_bot.league",
           "--bots", f"{CAND}={cand_spec},opp={spec}",
           "--games-per-pair", str(n), "--out", str(chunk_dir),
           "--seed", str(args.seed * 1000 + idx), "--bootstrap", "10",
           *shlex.split(args.league_args)]
    log = open(out_dir / f"{opp}-{idx:03d}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        proc.wait(timeout=args.chunk_timeout)
    except subprocess.TimeoutExpired:
        print(f"  chunk {opp}-{idx} timed out after {args.chunk_timeout:.0f}s; killing", flush=True)
        proc.kill()
        proc.wait()
    finally:
        log.close()
    games_path = chunk_dir / "games.jsonl"
    recs = []
    if games_path.exists():
        for line in games_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            if CAND not in g.get("bots", []):
                continue
            score = 0.5 if g["winner"] == "tie" else float(g["winner"] == CAND)
            recs.append({"opponent": opp, "chunk": idx, "battle_tag": g["battle_tag"],
                         "score": score, "turns": g["turns"]})
    return recs


def summarize(scores: dict[str, list[float]], order: list[str], boot: int = 2000, seed: int = 0) -> dict:
    rows = {}
    for opp in order:
        s = scores.get(opp, [])
        if s:
            rows[opp] = {"n": len(s), "score": sum(s) / len(s), "ci": wilson(sum(s), len(s))}
    mean = ci = None
    if rows:
        mean = sum(r["score"] for r in rows.values()) / len(rows)
        rng = random.Random(seed)
        means = []
        for _ in range(boot):
            means.append(sum(sum(rng.choices(scores[o], k=len(scores[o]))) / len(scores[o])
                             for o in rows) / len(rows))
        means.sort()
        ci = (means[int(0.025 * boot)], means[min(boot - 1, int(0.975 * boot))])
    return {"opponents": rows, "mean": mean, "mean_ci": ci}


def format_summary(summary: dict, target: int) -> str:
    lines = [f"{'opponent':<12} {'games':>6} {'score':>7}   95% CI (Wilson)"]
    for opp, r in summary["opponents"].items():
        lines.append(f"{opp:<12} {r['n']:>3}/{target:<3} {r['score']:>7.3f}   [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
    if summary["mean"] is not None:
        lo, hi = summary["mean_ci"]
        lines.append(f"{'MEAN':<12} {'':>7} {summary['mean']:>7.3f}   [{lo:.3f}, {hi:.3f}]  (mean of per-opponent scores, bootstrap CI)")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate", help="candidate bot, league spec syntax (type[:key=value...])")
    p.add_argument("--out", type=Path, required=True, help="results jsonl (appended; rerun to resume)")
    p.add_argument("--games", type=int, default=100, help="games per opponent (default 100)")
    p.add_argument("--chunk", type=int, default=10, help="games per league subprocess (even; default 10)")
    p.add_argument("--opponents-file", type=Path, help="JSON {label: spec} replacing the default opponent set")
    p.add_argument("--foulplay", action="store_true", help="also play Foul Play (needs setup_foulplay.ps1)")
    p.add_argument("--only", help="comma-separated opponent labels to play")
    p.add_argument("--league-args", default="", help="extra league arguments, e.g. '--search-workers 8 --concurrency 2'")
    p.add_argument("--chunk-timeout", type=float, default=3600.0, help="kill a chunk after this many seconds")
    p.add_argument("--max-failures", type=int, default=3, help="give up on an opponent after this many chunks that finished no game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", action="store_true", help="only print the summary of --out")
    args = p.parse_args(argv)

    opps = load_opponents(args.opponents_file, args.foulplay)
    if args.only:
        keep = [x.strip() for x in args.only.split(",")]
        opps = {k: v for k, v in opps.items() if k in keep}
    header, scores, chunks = read_log(args.out)
    if args.report:
        order = list(header["opponents"]) if header else list(scores)
        print(format_summary(summarize(scores, order), header["games"] if header else args.games))
        return 0
    if not args.candidate:
        p.error("--candidate is required")
    if header is None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        header = {"candidate": args.candidate, "games": args.games, "opponents": opps,
                  "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        append(args.out, {"header": header})
    elif header["candidate"] != args.candidate:
        raise SystemExit(f"{args.out} was started for a different candidate:\n  {header['candidate']}")
    else:  # resume; opponents added later are appended to the header record log
        new = {k: v for k, v in opps.items() if k not in header["opponents"]}
        for k, v in header["opponents"].items():
            if k in opps and opps[k] != v:
                raise SystemExit(f"opponent {k!r} spec changed since the run started")
        if new:
            append(args.out, {"header": {**header, "opponents": {**header["opponents"], **new}}})
            header["opponents"].update(new)

    work_dir = args.out.with_suffix("")  # chunk dirs + logs live next to the jsonl
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk = max(1, args.chunk)
    for opp, spec in opps.items():
        failures = 0
        while len(scores.get(opp, [])) < args.games and failures < args.max_failures:
            have = len(scores.get(opp, []))
            n = min(chunk, args.games - have)
            idx = chunks.get(opp, 0)
            chunks[opp] = idx + 1
            append(args.out, {"chunk": idx, "opponent": opp, "games": n})
            t0 = time.time()
            recs = run_chunk(opp, spec, args.candidate, n, idx, args, work_dir)
            for r in recs:
                append(args.out, r)
                scores.setdefault(opp, []).append(r["score"])
            failures = failures + 1 if not recs else 0
            print(f"{opp}: chunk {idx}: {len(recs)}/{n} games in {time.time() - t0:.0f}s; "
                  f"{len(scores.get(opp, []))}/{args.games}", flush=True)
        if len(scores.get(opp, [])) < args.games:
            print(f"WARNING: {opp} incomplete ({len(scores.get(opp, []))}/{args.games})", flush=True)
    print(format_summary(summarize(scores, list(opps)), args.games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
