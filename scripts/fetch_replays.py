#!/usr/bin/env python
"""Download high-rated public gen9randombattle replays into the human_replays.py input format.

    python scripts/fetch_replays.py --out data/external/top/gen9randombattle-top.jsonl.gz \
        [--min-rating 2000] [--days 365] [--delay 1.0]

Two passes over https://replay.pokemonshowdown.com (public JSON API, one request per --delay s):

1. ``search.json?sort=rating&page=N`` for N = 1..: the highest-rated uploads (the search stops
   after about 100 pages, roughly rating 2330 and up).
2. ``search.json?before=T`` walking back in upload time for --days, keeping replays rated
   >= --min-rating (most uploads are low-rated, so this pass scans many pages per kept game).

Each kept replay's ``<id>.json`` is fetched and appended as one line
``{"text", "battle_id", "elo", "month_year", "split", "players", "rating", "uploadtime"}``;
``elo`` is the rating as a string so ``human_replays.py --min-elo`` works on it. Ids already in
--out are skipped, so the script resumes after an interruption. Private / password replays are
never listed by the search and are skipped if seen.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://replay.pokemonshowdown.com"
FORMAT = "gen9randombattle"
HEADERS = {"User-Agent": "showdown-bot-research/0.1 (replay dataset; polite crawler)"}


class Client:
    """Thread-safe polite client: keep-alive sessions (one per thread, so up to --connections
    requests in flight), with request STARTS spaced at least ``delay`` seconds apart overall.

    Backs off on its own: a 429/5xx doubles the delay (cap 8 s) and so does a slowdown (mean
    latency of the last 50 requests above 3x the first 50's); the delay creeps back toward the
    target after 1000 clean requests. Changes are printed.
    """

    def __init__(self, delay: float):
        import threading

        self.target = self.delay = delay
        self.next_slot = 0.0
        self.requests = 0
        self.lock = threading.Lock()
        self.local = threading.local()
        self.lat: list[float] = []
        self.base_lat: float | None = None
        self.clean = 0

    def _session(self):
        import requests

        sess = getattr(self.local, "session", None)
        if sess is None:
            sess = self.local.session = requests.Session()
            sess.headers.update(HEADERS)
        return sess

    def _slow_down(self, why: str) -> None:  # caller holds the lock
        self.delay = min(8.0, max(self.delay * 2, 0.5))
        self.clean = 0
        print(f"  backing off ({why}): delay {self.delay:.2f}s", flush=True)

    def _wait_turn(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_slot)
            self.next_slot = start + self.delay
            self.requests += 1
        if start > now:
            time.sleep(start - now)

    def get(self, path: str) -> object | None:
        import requests

        for attempt in range(4):
            self._wait_turn()
            t0 = time.monotonic()
            try:
                resp = self._session().get(BASE + path, timeout=30)
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
                continue
            lat = time.monotonic() - t0
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                with self.lock:
                    self._slow_down(f"HTTP {resp.status_code}")
                time.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            with self.lock:
                self.lat.append(lat)
                if len(self.lat) == 50:
                    mean = sum(self.lat) / 50
                    if self.base_lat is None:
                        self.base_lat = mean
                    elif mean > 3 * self.base_lat:
                        self._slow_down(f"latency {mean:.2f}s vs {self.base_lat:.2f}s")
                    self.lat = []
                self.clean += 1
                if self.clean >= 1000 and self.delay > self.target:
                    self.delay = max(self.target, self.delay / 2)
                    self.clean = 0
                    print(f"  easing back: delay {self.delay:.2f}s", flush=True)
            try:
                return resp.json()
            except ValueError:
                time.sleep(10 * (attempt + 1))
        return None


def oldest_upload(out: Path) -> int | None:
    """Where the time pass left off: the upload time of the LAST record in --out (the time pass
    writes newest to oldest; the rating pass's records come first and span all dates, so the
    overall minimum would jump past the gap)."""
    last = None
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        try:
            for line in handle:
                last = int(json.loads(line)["uploadtime"])
        except (EOFError, json.JSONDecodeError, gzip.BadGzipFile):
            pass
    return last


def repair(out: Path) -> None:
    """Rewrite --out without a truncated tail (a killed run leaves a cut-off gzip member, and
    records appended after it would be unreadable)."""
    if not out.exists():
        return
    rows, broken = [], False
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        try:
            for line in handle:
                json.loads(line)
                rows.append(line)
        except (EOFError, json.JSONDecodeError, gzip.BadGzipFile):
            broken = True
    if broken:
        tmp = out.with_suffix(".repair")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            handle.writelines(rows)
        tmp.replace(out)
        print(f"repaired {out}: kept {len(rows)} complete records", flush=True)


def existing_ids(out: Path) -> set[str]:
    if not out.exists():
        return set()
    ids = set()
    with gzip.open(out, "rt", encoding="utf-8") as handle:
        try:
            for line in handle:
                ids.add(json.loads(line)["battle_id"])
        except (EOFError, json.JSONDecodeError, gzip.BadGzipFile):
            pass  # a truncated tail from an interrupted run; the ids read so far are kept
    return ids


def record(rep: dict) -> dict:
    t = datetime.fromtimestamp(int(rep["uploadtime"]), tz=timezone.utc)
    return {
        "text": rep["log"], "battle_id": rep["id"], "elo": str(rep.get("rating") or 0),
        "month_year": t.strftime("%B%Y"), "split": "train", "players": rep.get("players"),
        "rating": rep.get("rating"), "uploadtime": rep["uploadtime"],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--min-rating", type=int, default=2000)
    p.add_argument("--days", type=float, default=365, help="how far back the time pass walks")
    p.add_argument("--delay", type=float, default=1.0, help="minimum seconds between request starts")
    p.add_argument("--connections", type=int, default=1, help="replay fetches in flight at once")
    p.add_argument("--max-pages", type=int, default=100000, help="cap on time-pass search pages")
    p.add_argument("--continue-time", action="store_true",
                   help="skip the rating pass and start the time pass at the oldest upload already in --out "
                        "(extend an earlier crawl further back, e.g. --days 1500)")
    p.add_argument("--user", default=None,
                   help="only this player's public uploads (search.json?user=...), all pages; "
                        "skips both passes above and ignores --min-rating")
    args = p.parse_args(argv)
    if args.user:
        args.min_rating = 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    repair(out)
    seen = existing_ids(out)
    client = Client(args.delay)
    kept = 0
    print(f"resuming with {len(seen)} replays already in {out}", flush=True)

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(max(1, args.connections))

    def take(entries: list[dict], handle) -> int:
        nonlocal kept
        ids = []
        for e in entries:
            rid = e.get("id")
            if not rid or rid in seen or (e.get("rating") or 0) < args.min_rating or e.get("private"):
                continue
            ids.append(rid)
            seen.add(rid)
        n = 0
        # map keeps page order (newest to oldest), so the file's last record stays the oldest
        for rep in pool.map(lambda rid: client.get(f"/{rid}.json"), ids):
            if not rep or rep.get("password") or not rep.get("log"):
                continue
            handle.write(json.dumps(record(rep), ensure_ascii=False) + "\n")
            kept += 1
            n += 1
            if kept % 100 == 0:
                handle.flush()
                print(f"  kept {kept} (requests {client.requests}, delay {client.delay:.2f}s)", flush=True)
        return n

    start_before = oldest_upload(out) if args.continue_time and out.exists() else None
    with gzip.open(out, "at", encoding="utf-8") as handle:
        if args.user:
            user = urllib.parse.quote(args.user)
            for page in range(1, 10000):
                res = client.get(f"/search.json?user={user}&format={FORMAT}&page={page}")
                if not res:
                    break
                n = take(res, handle)
                print(f"user page {page}: {len(res)} listed, {n} new, kept {kept}", flush=True)
                if len(res) < 51:
                    break
            print(f"done: kept {kept} new replays, {client.requests} requests", flush=True)
            return 0
        # pass 1: highest-rated uploads
        for page in range(1, 1 if args.continue_time else 1000):
            res = client.get(f"/search.json?format={FORMAT}&sort=rating&page={page}")
            if not res:
                break
            n = take(res, handle)
            low = min((e.get("rating") or 0) for e in res)
            print(f"rating page {page}: {len(res)} listed, {n} new, lowest rating {low}", flush=True)
            if low < args.min_rating:
                break
        # pass 2: walk back in upload time
        stop = time.time() - args.days * 86400
        before = start_before
        if before:
            print(f"continuing the time pass before {datetime.fromtimestamp(before, tz=timezone.utc):%Y-%m-%d}", flush=True)
        # the next search page only needs this page's oldest upload time, so it is requested in
        # the background while this page's replays download (keeps all connections busy)
        pager = ThreadPoolExecutor(1)

        def page_query(b):
            return client.get(f"/search.json?format={FORMAT}" + (f"&before={b}" if b else ""))

        nxt = pager.submit(page_query, before)
        for page in range(args.max_pages):
            res = nxt.result()
            if not res:
                break
            oldest = min(int(e["uploadtime"]) for e in res)
            last_page = oldest <= stop or (before is not None and oldest >= before)
            if not last_page:
                nxt = pager.submit(page_query, oldest)
            n = take(res, handle)
            if page % 25 == 0 or n:
                day = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"time page {page}: back to {day}, {n} new, kept {kept}", flush=True)
            if last_page:
                break
            before = oldest
    print(f"done: kept {kept} new replays, {client.requests} requests, total {len(seen)} ids seen", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
