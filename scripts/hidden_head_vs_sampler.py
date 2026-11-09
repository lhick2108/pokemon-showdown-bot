"""Offline go/no-go: transformer hidden-set heads vs the live set sampler, on held-out human games.

    PYTHONPATH="<wheel>;<worktree>\\src" .venv\\Scripts\\python.exe scripts\\hidden_head_vs_sampler.py \
        --input data/external/pokechamp/gen9randombattle.jsonl.gz --model models/tok-d256.pt \
        --games 300 --out-json hidden_head.json

Method (docs/hidden-head-vs-sampler.md)
---------------------------------------
Each PokeChamp ``test``-split game is replayed once per seat in the live way (``human_replays.states_for_role``
with ``FirstPerson``): an ``OpponentModel`` consumes the log event by event and at every ``|turn|`` the set
trackers know only what that seat had seen. A capture hook records, per turn,

* the sampler's marginal for every revealed opponent Pokemon: the count-weighted distribution over the
  full randbats sets still consistent with the evidence (``SetTracker._matching_full_sets``, exactly what
  ``SetTracker.sample`` draws from; the role-based fallback marginal when no full set matches),
* the revealed facts (``_opp_facts``, the scrub of the token view).

The tracker state at the end of the game gives the ground truth: an item / ability / tera type / move that
was unrevealed at turn T and is revealed by the end of the game. The head is run on the token view the live
bot feeds it (state with the opponent scrubbed to the revealed facts, real event history rebuilt as in the
shard exporter); ``--hist pad`` uses padding history rows instead.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from showdown_bot import engine_state as es
from showdown_bot import human_replays as hr
from showdown_bot import token_net, tokens
from showdown_bot.players.search import _opp_facts, default_randbats_data
from showdown_bot.sets import OpponentModel, to_id

TERA_NAMES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
              "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy", "stellar", "none"]
EPS = 1e-3  # probability floor for log-loss (applied identically to every method)
ATTRS = ("item", "ability", "tera", "move")


def norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ---------------------------------------------------------------------------------------------------------
# sampler marginals
# ---------------------------------------------------------------------------------------------------------


def _addp(d: dict[str, float], k: str, w: float) -> None:
    d[k] = d.get(k, 0.0) + w


def _normed(d: dict[str, float]) -> dict[str, float]:
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t > 0 else {}


def sampler_marginals(tr: Any) -> dict[str, Any]:
    """What ``tr.sample`` would draw, as marginals: item / ability / tera -> {id: p}, move -> {id: P(move in set)}."""
    matches = tr._matching_full_sets()
    item: dict[str, float] = {}
    abil: dict[str, float] = {}
    tera: dict[str, float] = {}
    mv: dict[str, float] = {}
    if matches:
        tot = 0.0
        for _, fs in matches:
            w = max(0.0, float(fs.count))
            tot += w
            _addp(item, to_id(fs.item) if fs.item else "none", w)
            _addp(abil, to_id(fs.ability), w)
            _addp(tera, to_id(fs.tera_type), w)
            for m in set(to_id(x) for x in fs.moves):
                _addp(mv, m, w)
        if tot <= 0:
            matches = []
        else:
            return {"item": _normed(item), "ability": _normed(abil), "tera": _normed(tera),
                    "move": {k: v / tot for k, v in mv.items()}, "fallback": False}
    # role-based fallback (no full set matches the evidence): role mixture of per-role option marginals
    cands = tr.candidate_roles()
    for rname, rp in cands.items():
        role = tr._roles[rname]
        for kind, opts, out in (("items", role.items, item), ("abilities", role.abilities, abil),
                                ("teraTypes", role.tera_types, tera)):
            ws = {to_id(o) if o else "none": role.option_weight(kind, o) for o in opts}
            z = sum(ws.values())
            for k, w in ws.items():
                _addp(out, k, rp * (w / z if z > 0 else 0.0))
        mw = role.weights["moves"]
        for m in role.moves:
            _addp(mv, to_id(m), rp * min(1.0, mw.get(to_id(m), 0.0)))
    return {"item": _normed(item), "ability": _normed(abil), "tera": _normed(tera), "move": mv, "fallback": True}


class CaptureModel(OpponentModel):
    """``OpponentModel`` that snapshots every tracker at each ``update`` (one per ``|turn|``) and at ``forget``."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.turns: list[dict[str, Any]] = []
        self.final: dict[str, dict[str, Any]] = {}

    def update(self, battle: Any) -> dict[str, Any]:
        trackers = super().update(battle)
        mons = {}
        for ident, tr in trackers.items():
            mon = battle.opponent_team.get(ident)
            if mon is None:
                continue
            mons[ident] = {
                "species": str(mon.species).lower(), "item": tr._revealed_item, "ability": tr._revealed_ability,
                "tera": tr._revealed_tera, "moves": [to_id(m) for m in tr.revealed_moves],
                "prov": bool(tr.provisional_observations), "marg": sampler_marginals(tr),
                "n_sets": len(tr._matching_full_sets()),
            }
        self.turns.append({"turn": battle.turn, "mons": mons, "facts": _opp_facts(battle, trackers)})
        return trackers

    def forget(self, battle_tag: str) -> None:
        st = self._battles.get(str(battle_tag))
        if st is not None:
            self.final = {
                ident: {"item": tr._revealed_item, "ability": tr._revealed_ability, "tera": tr._revealed_tera,
                        "moves": [to_id(m) for m in tr.revealed_moves]}
                for ident, tr in st.trackers.items()
            }
        super().forget(battle_tag)


# ---------------------------------------------------------------------------------------------------------
# games
# ---------------------------------------------------------------------------------------------------------


def iter_test_games(path: Path, n: int, min_elo: int, stride: int, scan_limit: int):
    """Every ``stride``-th test-split game (deterministic spread over the file), up to ``n``."""
    seen = got = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if scan_limit and seen >= scan_limit:
                return
            rec = json.loads(line)
            if rec.get("split") != "test" or hr_elo(rec) < min_elo:
                continue
            seen += 1
            if (seen - 1) % stride:
                continue
            yield {"text": rec["text"], "battle_id": str(rec["battle_id"]), "elo": rec.get("elo")}
            got += 1
            if got >= n:
                return


def hr_elo(rec: dict[str, Any]) -> int:
    digits = "".join(ch if ch.isdigit() else " " for ch in str(rec.get("elo", ""))).split()
    return int(digits[0]) if digits else 0


def replay_game(text: str, battle_id: str, conv: hr.Converter, seed: int):
    """-> {role: (turn_states, capture_model)} for both seats (first-person, live-style trackers)."""
    log = hr.sanitize_log(text)
    if log.skip:
        raise hr.SkipGame(log.skip)
    digest = hashlib.sha256(f"{seed}:{battle_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    events = log.events
    teams: dict[str, list] = {}
    for role, other in (("p1", "p2"), ("p2", "p1")):
        teams[other] = hr.hindsight_team(events, role, other, conv.data, conv.sampler, rng, conv.model)
    actions = hr.extract_actions(events)
    facts = hr.game_facts(events, log.turns)
    out = {}
    for role in ("p1", "p2"):
        cap = CaptureModel(conv.data, damage=False)  # damage-roll inference needs our real stats (unknown here)
        fp = hr.FirstPerson(cap, conv.sampler, random.Random(rng.getrandbits(64)))
        stats: Counter = Counter()
        ts = hr.states_for_role(events, role, teams[role], None, actions, stats, fp, facts)
        if len(ts) != len(cap.turns):
            raise hr.SkipGame("capture_mismatch")
        out[role] = (ts, cap)
    return out


# ---------------------------------------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------------------------------------


def make_examples(cap: CaptureModel) -> list[dict[str, Any]]:
    """(turn index, ident, attr, truth) for every attribute unrevealed at that turn and revealed by the end."""
    ex = []
    for i, tn in enumerate(cap.turns):
        for ident, m in tn["mons"].items():
            fin = cap.final.get(ident)
            if fin is None or m["prov"]:
                continue
            for a in ("item", "ability", "tera"):
                if m[a] is None and fin[a] is not None:
                    ex.append({"i": i, "ident": ident, "attr": a, "truth": norm(fin[a]) or "none"})
            now = m["moves"]
            later = [x for x in fin["moves"] if x not in now]
            if later and len(now) < 4 and set(now) <= set(fin["moves"]):
                ex.append({"i": i, "ident": ident, "attr": "move", "truth": later[0], "later": later})
    return ex


# ---------------------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------------------


def score(dist: dict[str, float], truth: str, later: list[str] | None = None) -> dict[str, float]:
    """top-1 hit, log-loss (nats, floored), and whether the truth is outside the support."""
    if not dist:
        return {"top1": 0.0, "nll": -math.log(EPS), "out": 1.0, "p": 0.0}
    top = max(dist, key=dist.__getitem__)
    p = dist.get(truth, 0.0)
    hit = float(top in later) if later else float(top == truth)
    return {"top1": hit, "nll": -math.log(max(p, EPS)), "out": float(p <= 0.0), "p": p}


def restrict(head: dict[str, float], support: dict[str, float]) -> dict[str, float]:
    d = {k: head.get(k, 0.0) for k in support}
    return _normed(d) if sum(d.values()) > 0 else {}


def product(head: dict[str, float], samp: dict[str, float]) -> dict[str, float]:
    d = {k: head.get(k, 0.0) * p for k, p in samp.items()}
    return _normed(d) if sum(d.values()) > 0 else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/external/pokechamp/gen9randombattle.jsonl.gz")
    ap.add_argument("--model", default="models/tok-d256.pt")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--stride", type=int, default=20, help="take every Nth test game (spread over the file)")
    ap.add_argument("--min-elo", type=int, default=0)
    ap.add_argument("--scan-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hist", choices=("real", "pad"), default="real")
    ap.add_argument("--max-pos-per-side", type=int, default=0, help="subsample positions per game side (0 = all)")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    import poke_engine as pe
    import torch

    torch.set_num_threads(args.threads)
    net = token_net.TokenNet.load(args.model, device=args.device)
    lay = tokens.layout()
    vocab = tokens.Vocab()
    id_item = {norm(n): i for i, n in enumerate(pe.identity_vocab(1))}
    id_abil = {norm(n): i for i, n in enumerate(pe.identity_vocab(2))}
    id_move = {norm(n): i for i, n in enumerate(pe.identity_vocab(3))}
    inv_item = {i: n for n, i in id_item.items()}
    inv_abil = {i: n for n, i in id_abil.items()}
    inv_move = {i: n for n, i in id_move.items()}
    tera_id = {n: i for i, n in enumerate(TERA_NAMES)}

    conv = hr.Converter(default_randbats_data())
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    skips: Counter = Counter()
    n_games = 0
    t0 = time.time()
    for g in iter_test_games(Path(args.input), args.games, args.min_elo, args.stride, args.scan_limit):
        try:
            sides = replay_game(g["text"], g["battle_id"], conv, args.seed)
        except hr.SkipGame as e:
            skips[e.reason] += 1
            continue
        except Exception as e:  # noqa: BLE001
            skips[f"error:{type(e).__name__}"] += 1
            continue
        n_games += 1
        chosen = {r: {t.turn: t.chosen for t in ts} for r, (ts, _) in sides.items()}
        for role, (ts, cap) in sides.items():
            other = "p2" if role == "p1" else "p1"
            ex = make_examples(cap)
            if not ex:
                continue
            if args.max_pos_per_side and len(ex) > args.max_pos_per_side:
                ex = rng.sample(ex, args.max_pos_per_side)
            # token view of every turn (needed for the history rows)
            recs = []
            ok = True
            for i, t in enumerate(ts):
                reveal = tokens.reveal_rows(cap.turns[i]["facts"])
                try:
                    cat, cont, names = pe.state_tokens_str(t.state, 1, reveal, True)
                except BaseException:  # noqa: BLE001
                    ok = False
                    break
                b = tokens.split_buffers(cat, cont, lay)
                recs.append({"t": b, "turn": int(t.turn), "sum": tokens._summary(b),
                             "own_act": vocab.parse(t.chosen), "opp_act": vocab.parse(chosen[other].get(t.turn))})
            if not ok:
                skips["token_error"] += 1
                continue
            need = sorted({e["i"] for e in ex})
            bufs = [recs[i]["t"] for i in need]
            if args.hist == "real":
                hh = [tokens.build_history(recs, i) for i in need]
                batch = token_net.batch_from_engine(bufs, np.stack([h[0] for h in hh]), np.stack([h[1] for h in hh]))
            else:
                batch = token_net.batch_from_engine(bufs)
            out = net.predict(batch, device=args.device)
            pos = {i: k for k, i in enumerate(need)}
            for e in ex:
                tn = cap.turns[e["i"]]
                m = tn["mons"][e["ident"]]
                k = pos[e["i"]]
                sid = vocab.species.get(norm(m["species"]))
                mon_cat = recs[e["i"]]["t"]["mon_cat"]
                slot = next((s for s in range(6) if sid is not None and int(mon_cat[6 + s, 0]) == sid), None)
                if slot is None:
                    skips["slot_not_found"] += 1
                    continue
                a = e["attr"]
                samp = m["marg"]["move" if a == "move" else a]
                if a == "item":
                    p = out["hid_item"][k, slot]
                    head = {inv_item[j]: float(p[j]) for j in range(len(p)) if j in inv_item and j >= 2}
                elif a == "ability":
                    p = out["hid_ability"][k, slot]
                    head = {inv_abil[j]: float(p[j]) for j in range(len(p)) if j in inv_abil and j >= 2}
                elif a == "tera":
                    p = out["hid_tera"][k, slot]
                    head = {TERA_NAMES[j]: float(p[j]) for j in range(min(len(p), 20))}
                    head.pop("none", None)  # a used Tera type is never "none"
                    head = _normed(head)
                else:
                    p = out["hid_move"][k, slot].copy()
                    rev = set(int(x) for x in mon_cat[6 + slot, 3:7]) | {0, 1}
                    for j in rev:
                        p[j] = 0.0
                    p = p / max(p.sum(), 1e-12)
                    head = {inv_move[j]: float(p[j]) for j in np.argsort(-p)[:60] if j in inv_move and p[j] > 0}
                    head = _normed(head)
                    # sampler: next-move distribution = P(move in set) over the unrevealed moves, renormalised
                    samp = _normed({mm: pp for mm, pp in samp.items() if mm not in m["moves"]})
                later = e.get("later")
                row = {"game": g["battle_id"], "attr": a, "n_sets": m["n_sets"], "fallback": m["marg"]["fallback"],
                       "truth": e["truth"], "species": m["species"], "turn": tn["turn"]}
                for name, d in (("sampler", samp), ("head", head), ("head_in_support", restrict(head, samp)),
                                ("product", product(head, samp))):
                    for kk, vv in score(d, e["truth"], later).items():
                        row[f"{name}.{kk}"] = vv
                row["sampler.top"] = max(samp, key=samp.__getitem__) if samp else ""
                row["head.top"] = max(head, key=head.__getitem__) if head else ""
                row["sampler.maxp"] = max(samp.values()) if samp else 0.0
                rows.append(row)
        if n_games % 25 == 0:
            print(f"[{time.time() - t0:.0f}s] games {n_games} rows {len(rows)} skips {dict(skips)}", file=sys.stderr)

    report(rows, n_games, skips, args)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps({"args": vars(args), "games": n_games, "skips": dict(skips), "rows": rows}))
    return 0


METHODS = ("sampler", "head", "head_in_support", "product")


def boot_diff(rows: list[dict[str, Any]], a: str, b: str, key: str, n: int = 400) -> tuple[float, float, float]:
    """Game-clustered bootstrap of mean(a.key) - mean(b.key)."""
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by[r["game"]].append(r[f"{a}.{key}"] - r[f"{b}.{key}"])
    games = list(by)
    sums = np.array([sum(by[g]) for g in games])
    cnt = np.array([len(by[g]) for g in games])
    mean = sums.sum() / cnt.sum()
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(games), len(games))
        bs.append(sums[idx].sum() / cnt[idx].sum())
    return float(mean), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def report(rows: list[dict[str, Any]], n_games: int, skips: Counter, args: Any) -> None:
    print(f"\ngames {n_games}  examples {len(rows)}  skips {dict(skips)}  hist={args.hist}")
    hdr = f"{'attr':8s} {'n':>6s} {'games':>5s} | " + " | ".join(f"{m:>15s}" for m in METHODS)
    for key, label in (("top1", "top-1 accuracy"), ("nll", "log-loss (nats, p floored at 1e-3)")):
        print(f"\n{label}\n{hdr}")
        for a in list(ATTRS) + ["all3"]:
            rs = [r for r in rows if (r["attr"] != "move" if a == "all3" else r["attr"] == a)]
            if not rs:
                continue
            ng = len({r["game"] for r in rs})
            print(f"{a:8s} {len(rs):6d} {ng:5d} | " + " | ".join(
                f"{np.mean([r[f'{m}.{key}'] for r in rs]):15.4f}" for m in METHODS))
    print("\nsampler support excludes the truth (impossible worlds), fraction; role-based fallback rate")
    for a in ATTRS:
        rs = [r for r in rows if r["attr"] == a]
        if rs:
            print(f"{a:8s} out-of-support {np.mean([r['sampler.out'] for r in rs]):.4f}  fallback {np.mean([r['fallback'] for r in rs]):.4f}"
                  f"  head p(truth)=0 (<1e-3): {np.mean([r['head.p'] < EPS for r in rs]):.4f}")
    print("\ngame-clustered 95% CI of (head - sampler) top-1 and (head_in_support - sampler) top-1, product - sampler")
    for a in list(ATTRS) + ["all3"]:
        rs = [r for r in rows if (r["attr"] != "move" if a == "all3" else r["attr"] == a)]
        if len(rs) > 10:
            out = []
            for m in ("head", "head_in_support", "product"):
                d, lo, hi = boot_diff(rs, m, "sampler", "top1")
                out.append(f"{m} {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
            print(f"{a:8s} " + "  ".join(out))
    print("\nby sampler uncertainty (n_sets consistent sets; 1 = sampler is certain), top-1 sampler vs head")
    for a in ("item", "ability", "tera"):
        for lo, hi in ((1, 1), (2, 4), (5, 20), (21, 10**9)):
            rs = [r for r in rows if r["attr"] == a and lo <= r["n_sets"] <= hi]
            if len(rs) >= 20:
                print(f"{a:8s} n_sets {lo}-{hi if hi < 10**9 else 'inf':<4} n {len(rs):5d}  sampler {np.mean([r['sampler.top1'] for r in rs]):.3f}"
                      f"  head {np.mean([r['head.top1'] for r in rs]):.3f}  head_in_support {np.mean([r['head_in_support.top1'] for r in rs]):.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
