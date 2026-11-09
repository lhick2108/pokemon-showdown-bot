"""Token shards for the transformer policy+value net (bead showdown-bot-589.68.1).

Reads self-play recordings (``docs/selfplay-data.md``) and converted human replays
(``docs/human-replays.md``) and writes memory-mappable shards of fixed-shape token arrays plus
per-sample targets. Layout, dtypes, target definitions and masks: ``docs/tokens.md``.

    python -m showdown_bot.tokens export --source gen9=selfplay:data/selfplay/gen9 \
        --source human=human:data/human/pc1-fp-aux/train --out data/tokens/sample --max-records 200
    python -m showdown_bot.tokens info data/tokens/sample

The engine side is ``poke_engine.state_tokens_str`` (engine/src/tokens.rs); nothing here
recomputes a feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SHARD_FORMAT = 1
N_ACT = 13
N_HIST = 16  # event-history tokens: last 8 decisions x (own action, opponent action)
HIST_DECISIONS = N_HIST // 2
HIST_CAT_W = 4  # side, kind, action id, actor species id
HIST_CONT_W = 5  # turns_ago/16, d_hp_self, d_hp_other, fainted_self/6, fainted_other/6
KIND_NAMES = ("move", "tera_move", "switch")  # action kinds 0, 1, 2 in opp_act_kind; -1 unknown
HID_W = 8
N_TEAM = 6
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
N_RATING = 5  # rating buckets 0-4; 5 = unknown
RATING_UNKNOWN = 5

SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "val", SPLIT_TEST: "test"}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def battle_hash(tag: str) -> int:
    """Stable signed 63-bit hash of a battle tag (the train/val split and grouping key)."""
    return int.from_bytes(hashlib.blake2b(str(tag).encode(), digest_size=8).digest(), "little") >> 1


def split_of(bhash: int, val_permille: int) -> int:
    return SPLIT_VAL if bhash % 1000 < val_permille else SPLIT_TRAIN


# ---------------------------------------------------------------------------
# Engine layout
# ---------------------------------------------------------------------------


def _engine():
    import poke_engine

    if not hasattr(poke_engine, "state_tokens_str"):
        raise RuntimeError("poke_engine has no state_tokens_str: build the wheel from engine/ (docs/tokens.md)")
    return poke_engine


def layout() -> dict[str, dict[str, Any]]:
    """``{field: {"buffer", "offset", "shape", "dtype"}}`` of the engine token buffers."""
    pe = _engine()
    return {
        name: {"buffer": buf, "offset": off, "shape": list(shape), "dtype": dtype}
        for name, buf, off, shape, dtype in pe.token_layout()
    }


def split_buffers(cat: bytes, cont: bytes, lay: dict[str, dict[str, Any]] | None = None) -> dict[str, np.ndarray]:
    """Views of the two engine buffers as named arrays (``int16`` and ``float32``)."""
    lay = lay or layout()
    bufs = {"cat": np.frombuffer(cat, dtype="<i2"), "cont": np.frombuffer(cont, dtype="<f4")}
    out = {}
    for name, f in lay.items():
        n = int(np.prod(f["shape"]))
        out[name] = bufs[f["buffer"]][f["offset"] : f["offset"] + n].reshape(f["shape"])
    return out


def token_spec() -> dict[str, Any]:
    """Shapes, vocabulary sizes, version and hash of the token representation."""
    pe = _engine()
    return {
        "token_version": dict(pe.token_constants())["TOKEN_VERSION"],
        "shard_format": SHARD_FORMAT,
        "vocab_hash": f"{pe.identity_vocab_hash():#018x}",
        "vocab_sizes": dict(zip(("species", "item", "ability", "move"), pe.identity_vocab_sizes())),
        "n_types": 20,
        "constants": dict(pe.token_constants()),
        "fields": layout(),
        "hist": {"n": N_HIST, "cat_w": HIST_CAT_W, "cont_w": HIST_CONT_W},
        "n_actions": N_ACT,
    }


# shard arrays: name -> (dtype, trailing shape), see docs/tokens.md
def _shard_fields() -> dict[str, tuple[str, tuple[int, ...]]]:
    f = {k: (v["dtype"], tuple(v["shape"])) for k, v in layout().items()}
    for k, (dt, shp) in list(f.items()):
        if dt == "float32":
            f[k] = ("float16", shp)
    f["mon_known"] = ("int8", f["mon_known"][1])
    f["act_legal"] = ("int8", f["act_legal"][1])
    f.update(
        hist_cat=("int16", (N_HIST, HIST_CAT_W)),
        hist_cont=("float16", (N_HIST, HIST_CONT_W)),
        pol_target=("float32", (N_ACT,)),
        pol_valid=("int8", ()),
        z=("float32", ()),
        v_search=("float32", ()),
        v_search_valid=("int8", ()),
        opp_act_slot=("int8", ()),
        opp_act_kind=("int8", ()),
        opp_act_id=("int16", ()),
        opp_act_valid=("int8", ()),
        hid_ids=("int16", (N_TEAM, HID_W)),
        hid_mask=("int8", (N_TEAM,)),
        truth_team=("int16", (N_TEAM,)),
        hid_valid=("int8", ()),
        aux_turns_left=("int16", ()),
        aux_final_alive=("int8", (2,)),
        aux_kos=("int8", (12,)),
        aux_alive_end=("int8", (12,)),
        aux_valid=("int8", (3,)),
        source=("int8", ()),
        split=("int8", ()),
        battle_hash=("int64", ()),
        turn=("int16", ()),
        decision_idx=("int16", ()),
        rating=("int8", ()),
        state_kind=("int8", ()),
        reveal_src=("int8", ()),
    )
    return f


# ---------------------------------------------------------------------------
# Choice parsing
# ---------------------------------------------------------------------------


class Vocab:
    """Name -> embedding row lookups for species and moves (``identity_vocab``)."""

    def __init__(self) -> None:
        pe = _engine()
        self.species = {_norm(n): i for i, n in enumerate(pe.identity_vocab(0)) if i >= 2}
        self.move = {_norm(n): i for i, n in enumerate(pe.identity_vocab(3)) if i >= 2}

    def parse(self, choice: str | None) -> tuple[int, int]:
        """``(kind, id)``: kind 0 move / 1 tera move / 2 switch / -1 unknown, id = move or species row."""
        if not choice:
            return -1, 0
        c = choice.strip().lower()
        if c.startswith("switch "):
            return 2, self.species.get(_norm(c[7:]), 0)
        tera = c.endswith("-tera")
        if tera:
            c = c[: -len("-tera")]
        return (1 if tera else 0), self.move.get(_norm(c), 0)


def reveal_rows(opp_facts: Sequence[dict[str, Any]] | None) -> list[tuple[str, list[str], bool, bool, bool]] | None:
    """The recorder's ``opp_facts`` as the ``opp_reveal`` argument of ``state_tokens_str``."""
    if opp_facts is None:
        return None
    return [
        (str(f.get("species") or ""), [str(m) for m in (f.get("moves") or [])],
         bool(f.get("item")), bool(f.get("ability")), bool(f.get("tera")))
        for f in opp_facts
    ]


def opp_slot_of(kind: int, aid: int, act_cat: np.ndarray, act_legal: np.ndarray) -> int:
    """Slot of the opponent action ``(kind, id)`` among the opponent's 13 action tokens, or -1."""
    if kind < 0 or aid <= 0:
        return -1
    opp_cat, opp_leg = act_cat[1], act_legal[1]
    if kind in (0, 1):
        for s in range(4 * kind, 4 * kind + 4):
            if opp_leg[s] and opp_cat[s, 0] == aid:
                return s
    else:
        for s in range(8, N_ACT):
            if opp_leg[s] and opp_cat[s, 1] == aid:
                return s
    return -1


# ---------------------------------------------------------------------------
# Per-game sample building
# ---------------------------------------------------------------------------


class Options:
    def __init__(self, **kw: Any) -> None:
        self.state = kw.get("state", "auto")  # auto | truth | world
        self.scrub = kw.get("scrub", True)  # hide unrevealed opponent info using opp_facts
        self.human_view = kw.get("human_view", "as-is")  # as-is | species
        self.val_permille = kw.get("val_permille", 50)
        self.max_per_game = kw.get("max_per_game", 0)
        self.seed = kw.get("seed", 0)
        self.reconstruct_q = kw.get("reconstruct_q", True)  # rebuild v_search from avg_score when not exported


def _summary(t: dict[str, np.ndarray]) -> dict[str, Any]:
    """Public per-decision quantities used for the event history."""
    ids, cont = t["mon_cat"][:, 0].astype(int), t["mon_cont"]
    alive = {side: {} for side in (0, 1)}
    for m in range(12):
        if cont[m, 1] > 0.5 and ids[m] > 1:
            alive[m // 6][int(ids[m])] = float(cont[m, 0])
    return {"alive": alive, "active": (int(ids[0]), int(ids[6]))}


def _hp_delta(before: dict[str, Any], after: dict[str, Any], side: int) -> float:
    x = before["active"][side]
    if x <= 1 or x not in before["alive"][side]:
        return 0.0
    return after["alive"][side].get(x, 0.0) - before["alive"][side][x]


def _fainted(before: dict[str, Any], after: dict[str, Any], side: int) -> int:
    return sum(1 for sp in before["alive"][side] if sp not in after["alive"][side])


def build_history(recs: list[dict[str, Any]], i: int) -> tuple[np.ndarray, np.ndarray]:
    """Event tokens for decision ``i`` from the previous ``HIST_DECISIONS`` decisions of one game."""
    hc = np.zeros((N_HIST, HIST_CAT_W), dtype=np.int16)
    hf = np.zeros((N_HIST, HIST_CONT_W), dtype=np.float32)
    for k in range(1, HIST_DECISIONS + 1):
        j = i - k
        if j < 0:
            break
        a, b = recs[j], recs[j + 1]
        sb, sa = a["sum"], b["sum"]
        ago = min(max(recs[i]["turn"] - a["turn"], 0), 16) / 16.0
        d_own, d_opp = _hp_delta(sb, sa, 0), _hp_delta(sb, sa, 1)
        f_own, f_opp = _fainted(sb, sa, 0), _fainted(sb, sa, 1)
        for n, (side, kind, aid) in enumerate(((1, *a["own_act"]), (2, *a["opp_act"]))):
            row = 2 * (k - 1) + n
            hc[row] = (side, kind + 1, aid, sb["active"][side - 1])
            self_d, other_d = (d_own, d_opp) if side == 1 else (d_opp, d_own)
            self_f, other_f = (f_own, f_opp) if side == 1 else (f_opp, f_own)
            hf[row] = (ago, self_d, other_d, self_f / 6.0, other_f / 6.0)
    return hc, hf


def _truth_or_world(decision: dict[str, Any], mode: str) -> tuple[str | None, int]:
    """``(state string, kind)``; kind 0 = truth state, 1 = sampled world."""
    truth = (decision.get("truth") or {}).get("state")
    worlds = decision.get("worlds") or []
    if mode == "world" and worlds:
        return worlds[0]["state"], 1
    if mode in ("auto", "truth") and truth:
        return truth, 0
    if mode in ("auto", "world") and worlds:
        return worlds[0]["state"], 1
    return None, 1


def _search_value(decision: dict[str, Any]) -> float | None:
    vals = [float(w["search_value"]) for w in (decision.get("worlds") or []) if w.get("search_value") is not None]
    return sum(vals) / len(vals) if vals else None


def _anchor(header: dict[str, Any], stats: Counter) -> tuple[Any, float] | None:
    """``(evaluator or None for stock, leaf scale)`` the recording's search used, or None if unusable."""
    from showdown_bot import dataset

    try:
        path, scale = dataset.recording_anchor(header)
        return (None if path is None else dataset._anchor_evaluator(path)), float(scale)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        stats["anchor_unusable"] += 1
        return None


def reconstructed_value(decision: dict[str, Any], anchor: tuple[Any, float]) -> float | None:
    """Mean absolute P(side one wins) over the worlds, rebuilt from ``avg_score`` (dataset.py rules)."""
    from showdown_bot import dataset

    pe = _engine()
    ev, scale = anchor
    vals = []
    for w in decision.get("worlds") or []:
        score = dataset._valid_score(w)
        if score is None:
            continue
        st = pe.State.from_string(w["state"])
        vals.append(
            dataset.absolute_search_value(pe.evaluate(st), score)
            if ev is None
            else dataset.learned_absolute_search_value(ev, st, score, scale)
        )
    return sum(vals) / len(vals) if vals else None


def _empty_sample(fields: dict[str, tuple[str, tuple[int, ...]]]) -> dict[str, np.ndarray]:
    return {k: np.zeros(shp, dtype=dt) for k, (dt, shp) in fields.items()}


def game_samples(
    game: dict[str, Any],
    peer: dict[str, Any] | None,
    *,
    source_idx: int,
    kind: str,
    opts: Options,
    vocab: Vocab,
    fields: dict[str, tuple[str, tuple[int, ...]]],
    lay: dict[str, dict[str, Any]],
    stats: Counter,
    split: int | None = None,
) -> list[dict[str, np.ndarray]]:
    """All samples (one per decision with a state) of one recorded game side."""
    pe = _engine()
    header, decisions, result = game["header"], game["decisions"], game["result"] or {}
    tag = str(header.get("battle_tag", ""))
    bh = battle_hash(tag)
    out_split = split if split is not None else split_of(bh, opts.val_permille)
    z = {"win": 1.0, "tie": 0.5, "loss": 0.0}.get(result.get("result"))
    if z is None:
        stats["games_no_result"] += 1
        return []
    peer_by_turn = {d.get("turn"): d for d in (peer["decisions"] if peer else [])}
    from showdown_bot.human_policy import elo_id

    rating = elo_id(header.get("elo")) if kind == "human" else RATING_UNKNOWN
    n_turns = result.get("turns")
    anchor = _anchor(header, stats) if (kind == "selfplay" and opts.reconstruct_q) else None

    recs: list[dict[str, Any]] = []
    for idx, d in enumerate(decisions):
        state, skind = _truth_or_world(d, opts.state)
        if state is None:
            continue
        reveal = None
        if kind == "selfplay" and opts.scrub and d.get("opp_facts") is not None:
            reveal = reveal_rows(d["opp_facts"])
        elif kind == "human" and opts.human_view == "species":
            ident = ((d.get("aux") or {}).get("opp") or {}).get("ident") or []
            reveal = [(str(s), [], False, False, False) for s in ident if s]
        try:
            cat, cont, names = pe.state_tokens_str(state, 1, reveal, True)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # includes pyo3 PanicException on a state the engine rejects
            stats["token_errors"] += 1
            continue
        t = split_buffers(cat, cont, lay)
        peer_d = peer_by_turn.get(d.get("turn"))
        own_act = vocab.parse(d.get("chosen"))
        opp_act = vocab.parse(peer_d.get("chosen") if peer_d else None)
        recs.append(
            {"idx": idx, "d": d, "state": state, "skind": skind, "t": t, "names": names, "reveal": reveal,
             "turn": int(d.get("turn") or 0), "sum": _summary(t), "own_act": own_act, "opp_act": opp_act,
             "peer": peer_d}
        )

    keep = list(range(len(recs)))
    if opts.max_per_game and len(keep) > opts.max_per_game:
        import random

        keep = sorted(random.Random(f"{opts.seed}:{tag}:{header.get('username')}").sample(keep, opts.max_per_game))

    samples = []
    for i in keep:
        r = recs[i]
        d, t = r["d"], r["t"]
        s = _empty_sample(fields)
        for name in lay:
            s[name][...] = t[name]
        hc, hf = build_history(recs, i)  # looks backwards only: decisions i-1 .. i-8
        s["hist_cat"][...] = hc
        s["hist_cont"][...] = hf
        # --- policy
        names = r["names"]
        slot_of = {n: j for j, n in enumerate(names[:N_ACT]) if n}
        legal = s["act_legal"][0]
        if kind == "selfplay" and d.get("policy"):
            pol = np.zeros(N_ACT, dtype=np.float32)
            dropped = 0.0
            for choice, p in d["policy"].items():
                j = slot_of.get(choice)
                if j is None:
                    dropped += float(p)
                else:
                    pol[j] += float(p)
            total = float(pol.sum())
            if total > 0.95 and dropped < 0.05:
                s["pol_target"][...] = pol / total
                s["pol_valid"] = 1
            else:
                stats["policy_unmapped"] += 1
        elif kind == "human" and d.get("chosen"):
            j = slot_of.get(d["chosen"])
            if j is not None and legal[j]:
                s["pol_target"][j] = 1.0
                s["pol_valid"] = 1
            else:
                stats["human_action_not_legal"] += 1
        # --- values
        s["z"] = z
        q = _search_value(d)
        if q is not None:
            s["v_search_valid"] = 1
        elif opts.reconstruct_q and anchor is not None and d.get("worlds"):
            try:
                q = reconstructed_value(d, anchor)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                q = None
                stats["q_errors"] += 1
            if q is not None:
                s["v_search_valid"] = 2
        if q is not None:
            s["v_search"] = q
        # --- opponent next action
        kd, aid = r["opp_act"]
        if kd >= 0:
            s["opp_act_kind"] = kd
            s["opp_act_id"] = aid
            s["opp_act_valid"] = 1
            s["opp_act_slot"] = opp_slot_of(kd, aid, t["act_cat"], t["act_legal"])
        else:
            s["opp_act_kind"] = -1
            s["opp_act_slot"] = -1
        # --- hidden sets (self-play truth)
        truth = (d.get("truth") or {}).get("state")
        if kind == "selfplay" and truth:
            try:
                h = np.frombuffer(pe.state_hidden_targets(r["state"], truth, 1), dtype="<i2")
                s["hid_ids"][...] = h[: N_TEAM * HID_W].reshape(N_TEAM, HID_W)
                s["hid_mask"][...] = h[N_TEAM * HID_W : N_TEAM * HID_W + N_TEAM]
                s["truth_team"][...] = h[N_TEAM * HID_W + N_TEAM :]
                s["hid_valid"] = 1
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                stats["hidden_errors"] += 1
        # --- aux
        meta = t["meta"]
        order = [[int(meta[0])] + [k for k in range(6) if k != int(meta[0])],
                 [int(meta[1])] + [k for k in range(6) if k != int(meta[1])]]
        s["aux_kos"][...] = -1
        s["aux_alive_end"][...] = -1
        aux = d.get("aux")
        if aux:
            s["aux_turns_left"] = int(aux.get("turns_left", 0))
            s["aux_final_alive"][...] = [int(x) for x in aux.get("final_alive", (0, 0))]
            s["aux_valid"][0:2] = 1
            for side_no, key in enumerate(("own", "opp")):
                blk = aux.get(key) or {}
                for slot, k in enumerate(order[side_no]):
                    if k < len(blk.get("kos", ())):
                        s["aux_kos"][side_no * 6 + slot] = min(int(blk["kos"][k]), 127)
                    if k < len(blk.get("alive_end", ())):
                        s["aux_alive_end"][side_no * 6 + slot] = int(blk["alive_end"][k])
            s["aux_valid"][2] = 1
        elif n_turns is not None and d.get("turn") is not None:
            s["aux_turns_left"] = max(int(n_turns) - int(d["turn"]), 0)
            s["aux_valid"][0] = 1
        # --- meta
        s["source"] = source_idx
        s["split"] = out_split
        s["battle_hash"] = bh
        s["turn"] = r["turn"]
        s["decision_idx"] = min(r["idx"], 32767)
        s["rating"] = rating
        s["state_kind"] = r["skind"]
        s["reveal_src"] = 1 if r["reveal"] is not None else 0
        samples.append(s)
    stats["games"] += 1
    stats["samples"] += len(samples)
    return samples


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------


def write_shard(path: Path, samples: Sequence[dict[str, np.ndarray]], fields: dict[str, tuple[str, tuple[int, ...]]]) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    info: dict[str, Any] = {}
    for name, (dt, shp) in fields.items():
        arr = np.zeros((n, *shp), dtype=dt)
        for i, s in enumerate(samples):
            v = s[name]
            if dt == "float16":
                v = np.clip(v, -65000, 65000)
            arr[i] = v
        np.save(path / f"{name}.npy", arr)
        info[name] = {"shape": [n, *shp], "dtype": dt}
    return info


def _group_files(root: Path) -> list[list[Path]]:
    """Game files grouped by battle (the two recordings of one battle share a name prefix)."""
    from showdown_bot import recorder

    groups: dict[tuple[Path, str], list[Path]] = {}
    for p in recorder.iter_game_files(root):
        groups.setdefault((p.parent, p.name.split("__")[0]), []).append(p)
    return [sorted(v) for _, v in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))]


def _read(path: Path) -> dict[str, Any] | None:
    from showdown_bot import recorder

    try:
        return recorder.read_game(path)
    except (OSError, ValueError, EOFError):
        return None


def _task(args: tuple) -> dict[str, Any]:
    out_dir, shard_name, source_idx, kind, groups, opts, max_rows = args
    vocab, fields, lay = Vocab(), _shard_fields(), layout()
    stats: Counter = Counter()
    samples: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    for files in groups:
        games = [g for g in map(_read, files) if g is not None]
        for gi, g in enumerate(games):
            others = [o for oi, o in enumerate(games) if oi != gi]
            peer = others[0] if len(others) == 1 else None
            split = None
            if kind == "human":
                split = SPLIT_TEST if g["header"].get("split") == "test" else None
            samples.extend(
                game_samples(g, peer, source_idx=source_idx, kind=kind, opts=opts, vocab=vocab,
                             fields=fields, lay=lay, stats=stats, split=split)
            )
        if max_rows and len(samples) >= max_rows:
            samples = samples[:max_rows]
            break
    info = write_shard(Path(out_dir) / "shards" / shard_name, samples, fields) if samples else {}
    return {"shard": shard_name, "source": source_idx, "n": len(samples), "fields": info, "stats": dict(stats),
            "seconds": time.perf_counter() - t0}


def coverage(shard_dir: Path) -> Counter:
    """Counts of samples per target that is available (read back from a written shard)."""
    a = {k: np.load(shard_dir / f"{k}.npy") for k in (
        "pol_valid", "v_search_valid", "opp_act_valid", "opp_act_slot", "hid_valid", "aux_valid",
        "reveal_src", "hid_mask", "split")}
    return Counter(
        n=len(a["pol_valid"]),
        pol_valid=int(a["pol_valid"].sum()),
        v_search_valid=int((a["v_search_valid"] > 0).sum()),
        v_search_exported=int((a["v_search_valid"] == 1).sum()),
        v_search_reconstructed=int((a["v_search_valid"] == 2).sum()),
        opp_act_valid=int(a["opp_act_valid"].sum()),
        opp_act_slot_found=int((a["opp_act_slot"] >= 0).sum()),
        hid_valid=int(a["hid_valid"].sum()),
        hid_slots_matched=int(a["hid_mask"].sum()),
        aux_turns_left_valid=int(a["aux_valid"][:, 0].sum()),
        aux_full_valid=int(a["aux_valid"][:, 2].sum()),
        reveal_scrubbed=int(a["reveal_src"].sum()),
        train=int((a["split"] == SPLIT_TRAIN).sum()),
        val=int((a["split"] == SPLIT_VAL).sum()),
        test=int((a["split"] == SPLIT_TEST).sum()),
    )


def export(
    sources: Sequence[tuple[str, str, Path]],
    out: Path,
    *,
    workers: int = 2,
    games_per_shard: int = 100,
    max_records: int = 0,
    max_games: int = 0,
    opts: Options | None = None,
) -> dict[str, Any]:
    """Write shards for ``sources`` = ``[(name, kind, path)]`` (kind ``selfplay`` or ``human``)."""
    opts = opts or Options()
    out.mkdir(parents=True, exist_ok=True)
    spec = token_spec()
    shard_infos: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for sidx, (name, kind, path) in enumerate(sources):
        groups = _group_files(Path(path))
        if max_games:
            groups = groups[:max_games]
        chunks = [groups[i : i + games_per_shard] for i in range(0, len(groups), games_per_shard)]
        tasks = [(str(out), f"{name}-{i:05d}", sidx, kind, c, opts, 0) for i, c in enumerate(chunks)]
        total = 0
        if workers <= 1:
            results: Iterator[dict[str, Any]] = (_task(t) for t in tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(workers)
            results = pool.imap(_task, tasks)
        try:
            for res in results:
                if max_records and total + res["n"] > max_records:
                    res = _trim_shard(out, res, max_records - total)
                total += res["n"]
                if res["n"]:
                    shard_infos.append(res)
                print(f"[{name}] {res['shard']}: {res['n']} samples ({total} total) {res['seconds']:.1f}s", flush=True)
                if max_records and total >= max_records:
                    break
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
    cov: dict[str, Counter] = {}
    for r in shard_infos:
        c = cov.setdefault(sources[r["source"]][0], Counter())
        c.update(coverage(out / "shards" / r["shard"]))
        for k, v in r["stats"].items():
            c["stat_" + k] += v
    manifest = {
        "shard_format": SHARD_FORMAT,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": time.perf_counter() - t0,
        "spec": spec,
        "sources": [{"id": i, "name": n, "kind": k, "path": str(p)} for i, (n, k, p) in enumerate(sources)],
        "options": vars(opts),
        "shards": [{"name": r["shard"], "source": r["source"], "n": r["n"], "fields": r["fields"]} for r in shard_infos],
        "n_samples": sum(r["n"] for r in shard_infos),
        "coverage": {k: dict(v) for k, v in cov.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _trim_shard(out: Path, res: dict[str, Any], keep: int) -> dict[str, Any]:
    """Cut a finished shard down to its first ``keep`` rows (for ``--max-records``)."""
    d = out / "shards" / res["shard"]
    for name, f in res["fields"].items():
        a = np.load(d / f"{name}.npy")[:keep]
        np.save(d / f"{name}.npy", a)
        f["shape"][0] = keep
    res["n"] = keep
    return res


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class TokenDataset:
    """Memory-mapped view of an exported directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())

    @property
    def shards(self) -> list[dict[str, Any]]:
        return self.manifest["shards"]

    def load(self, shard: str | dict[str, Any], mmap: bool = True) -> dict[str, np.ndarray]:
        name = shard["name"] if isinstance(shard, dict) else shard
        d = self.path / "shards" / name
        return {p.stem: np.load(p, mmap_mode="r" if mmap else None) for p in sorted(d.glob("*.npy"))}

    def __len__(self) -> int:
        return int(self.manifest["n_samples"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_source(text: str) -> tuple[str, str, Path]:
    try:
        name, rest = text.split("=", 1)
        kind, path = rest.split(":", 1)
    except ValueError:
        raise argparse.ArgumentTypeError("--source must look like NAME=KIND:PATH (KIND selfplay or human)")
    if kind not in ("selfplay", "human"):
        raise argparse.ArgumentTypeError("source kind must be selfplay or human")
    return name, kind, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write token shards")
    e.add_argument("--source", action="append", type=_parse_source, required=True, metavar="NAME=KIND:PATH")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--workers", type=int, default=2, help="processes (default 2)")
    e.add_argument("--games-per-shard", type=int, default=100)
    e.add_argument("--max-records", type=int, default=0, help="stop each source after N samples (0 = all)")
    e.add_argument("--max-games", type=int, default=0, help="read at most N battles per source (0 = all)")
    e.add_argument("--max-per-game", type=int, default=0, help="keep at most N decisions per game side (0 = all)")
    e.add_argument("--state", choices=("auto", "truth", "world"), default="auto",
                   help="self-play input state: truth when recorded else first world (auto)")
    e.add_argument("--no-scrub", action="store_true", help="do not hide unrevealed opponent info (self-play)")
    e.add_argument("--human-view", choices=("as-is", "species"), default="as-is")
    e.add_argument("--val-permille", type=int, default=50, help="battles hashed into the val split, per mille")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--no-reconstruct-q", action="store_true",
                   help="leave v_search missing when the recording has no exported search_value")
    i = sub.add_parser("info", help="print a manifest summary")
    i.add_argument("path", type=Path)
    sub.add_parser("spec", help="print token_spec() as JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "spec":
        print(json.dumps(token_spec(), indent=1))
        return 0
    if args.cmd == "info":
        ds = TokenDataset(args.path)
        m = ds.manifest
        print(f"samples {m['n_samples']} in {len(m['shards'])} shards, vocab {m['spec']['vocab_hash']}, {m['seconds']:.1f}s")
        for name, c in m["coverage"].items():
            n = max(c.get("n", 0), 1)
            print(f"  {name}: n={c.get('n', 0)} " + " ".join(
                f"{k}={v / n:.1%}" for k, v in c.items() if k not in ("n",) and not k.startswith("stat_")))
            print("    stats:", {k[5:]: v for k, v in c.items() if k.startswith("stat_")})
        return 0
    opts = Options(state=args.state, scrub=not args.no_scrub, human_view=args.human_view,
                   val_permille=args.val_permille, max_per_game=args.max_per_game, seed=args.seed,
                   reconstruct_q=not args.no_reconstruct_q)
    manifest = export(args.source, args.out, workers=args.workers, games_per_shard=args.games_per_shard,
                      max_records=args.max_records, max_games=args.max_games, opts=opts)
    print(f"wrote {manifest['n_samples']} samples to {args.out} in {manifest['seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
