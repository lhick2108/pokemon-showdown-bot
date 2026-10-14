"""Train and evaluate the human action-policy model (``showdown_bot.human_policy``).

Streams ``data/human/*/policy/{train,test}/shard-*.jsonl.gz`` through DataLoader workers (each
reads a few shards at a time, shuffles within that group and featurises batches lazily), so
memory stays flat however many rows are used.

    python scripts/train_human_policy.py --data data/human/pc1/policy --out models/human-policy.npz \
        --epochs 2 --workers 8
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from showdown_bot import human_policy as hp


def shard_files(root: str | Path, split: str) -> list[str]:
    return sorted(glob.glob(str(Path(root) / split / "shard-*.jsonl.gz")))


def read_rows(path: str) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def encode_row(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int, int] | None:
    """``(features, mask, label slot, elo id, n legal)`` or ``None`` if unusable."""
    if not row.get("action_legal", True):
        return None
    feats, mask, moves, bench = hp.encode(row["state"], row["legal"])
    slot = hp.slot_of(row["action"], moves, bench)
    if slot is None or mask[slot] <= 0:
        return None
    return feats, mask, slot, hp.elo_id(row.get("elo")), int(mask.sum())


class BatchStream(IterableDataset):
    """IterableDataset yielding ready batches ``(x, mask, y, elo, n_legal)`` (torch tensors).

    Rows with fewer than ``min_legal`` legal slots are skipped (a forced action carries no
    information). With ``shuffle`` the shard order depends on ``seed + epoch`` and rows are
    shuffled inside groups of ``group`` shards; otherwise order is fixed.
    """

    def __init__(self, files, batch=1024, group=4, shuffle=True, seed=0, epoch=0,
                 min_legal=2, max_rows=0):
        self.files = list(files)
        self.batch, self.group, self.shuffle = batch, group, shuffle
        self.seed, self.epoch, self.min_legal, self.max_rows = seed, epoch, min_legal, max_rows

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        files = list(self.files)
        if self.shuffle:
            random.Random(f"{self.seed}:{self.epoch}").shuffle(files)
        files = files[wid::nw]
        rng = random.Random(f"{self.seed}:{self.epoch}:{wid}")
        emitted = 0
        for g in range(0, len(files), self.group):
            rows = [r for f in files[g : g + self.group] for r in read_rows(f)]
            if self.shuffle:
                rng.shuffle(rows)
            buf: list[tuple] = []
            for row in rows:
                enc = encode_row(row)
                if enc is None or enc[4] < self.min_legal:
                    continue
                buf.append(enc)
                if len(buf) == self.batch:
                    yield _collate(buf)
                    emitted += len(buf)
                    buf = []
                    if self.max_rows and emitted * nw >= self.max_rows:
                        return
            if buf:
                yield _collate(buf)


def _collate(buf: list[tuple]) -> tuple:
    x = torch.from_numpy(np.stack([b[0] for b in buf]).astype(np.float16))
    mask = torch.from_numpy(np.stack([b[1] for b in buf]))
    y = torch.tensor([b[2] for b in buf], dtype=torch.long)
    elo = torch.tensor([b[3] for b in buf], dtype=torch.long)
    n = torch.tensor([b[4] for b in buf], dtype=torch.long)
    return x, mask, y, elo, n


def make_model(h1: int, h2: int, mu: np.ndarray, sd: np.ndarray):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
            self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
            self.fc1 = nn.Linear(hp.N_FEATURES, h1)
            self.fm = nn.Linear(hp.N_SLOTS, h1, bias=False)
            self.emb = nn.Embedding(hp.N_ELO + 1, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, hp.N_SLOTS)
            nn.init.zeros_(self.emb.weight)

        def forward(self, x, mask, elo):
            h = self.fc1((x.float() - self.mu) / self.sd) + self.fm(mask) + self.emb(elo)
            h = torch.relu(h)
            h = torch.relu(self.fc2(h))
            z = self.fc3(h)
            return z.masked_fill(mask <= 0, -1e9)

    return Net()


def export(model: Any, meta: dict[str, Any]) -> hp.HumanPolicy:
    """Fold the input standardisation into layer 1 and wrap the weights for numpy inference."""
    w = model.fc1.weight.detach().cpu().double().numpy()  # (h1, F)
    sd = model.sd.cpu().double().numpy()
    mu = model.mu.cpu().double().numpy()
    w1 = (w / sd[None, :]).T
    b1 = model.fc1.bias.detach().cpu().double().numpy() - (w * (mu / sd)[None, :]).sum(axis=1)
    arrays = {
        "w1": w1, "b1": b1,
        "wm": model.fm.weight.detach().cpu().double().numpy().T,
        "elo": model.emb.weight.detach().cpu().double().numpy(),
        "w2": model.fc2.weight.detach().cpu().double().numpy().T,
        "b2": model.fc2.bias.detach().cpu().double().numpy(),
        "w3": model.fc3.weight.detach().cpu().double().numpy().T,
        "b3": model.fc3.bias.detach().cpu().double().numpy(),
    }
    return hp.HumanPolicy({k: v.astype(np.float32) for k, v in arrays.items()}, meta)


def feature_stats(files: list[str], n_shards: int = 24, per_shard: int = 1500, seed: int = 0):
    rng = random.Random(seed)
    xs = []
    for f in rng.sample(files, min(n_shards, len(files))):
        for i, row in enumerate(read_rows(f)):
            if i >= per_shard:
                break
            enc = encode_row(row)
            if enc is not None:
                xs.append(enc[0])
    x = np.stack(xs)
    return x.mean(axis=0), np.maximum(x.std(axis=0), 0.1)


def evaluate(model: Any, files: list[str], workers: int, device: Any, *, elo_known: bool = True,
             max_rows: int = 0, prior: np.ndarray | None = None) -> dict[str, Any]:
    """Overall and per-elo-bucket top-1 / top-3 / NLL against uniform (and slot-prior) baselines."""
    import torch
    from torch.utils.data import DataLoader

    stream = BatchStream(files, shuffle=False, min_legal=2, max_rows=max_rows)
    loader = DataLoader(stream, batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None)
    acc: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    model.eval()
    with torch.no_grad():
        for x, mask, y, elo, n in loader:
            x, mask, y, elo = x.to(device), mask.to(device), y.to(device), elo.to(device)
            z = model(x, mask, elo if elo_known else torch.full_like(elo, hp.N_ELO))
            logp = torch.log_softmax(z, dim=1)
            nll = -logp.gather(1, y[:, None])[:, 0]
            top3 = (z.topk(3, dim=1).indices == y[:, None]).any(dim=1).float()
            top1 = (z.argmax(dim=1) == y).float()
            n_f = n.to(device).float()
            uni_nll = torch.log(n_f)
            uni_top1 = 1.0 / n_f
            uni_top3 = torch.clamp(3.0 / n_f, max=1.0)
            cols = {"n": torch.ones_like(nll), "top1": top1, "top3": top3, "nll": nll,
                    "uni_top1": uni_top1, "uni_top3": uni_top3, "uni_nll": uni_nll}
            if prior is not None:
                p = torch.tensor(prior, device=device, dtype=torch.float32)[None, :] * mask
                p = p / p.sum(dim=1, keepdim=True)
                cols["prior_nll"] = -torch.log(p.gather(1, y[:, None])[:, 0].clamp_min(1e-9))
                cols["prior_top1"] = (p.argmax(dim=1) == y).float()
            for bucket in (-1, *range(hp.N_ELO)):
                sel = torch.ones_like(elo, dtype=torch.bool) if bucket < 0 else elo == bucket
                if sel.any():
                    for k, v in cols.items():
                        acc[bucket][k] += float(v[sel].sum())
    model.train()
    out: dict[str, Any] = {}
    for bucket, sums in acc.items():
        n = sums["n"]
        name = "all" if bucket < 0 else hp.ELO_BUCKETS[bucket]
        out[name] = {k: (n if k == "n" else v / n) for k, v in sums.items()}
    return out


def format_table(res: dict[str, Any]) -> str:
    keys = [k for k in ("n", "top1", "top3", "nll", "uni_top1", "uni_top3", "uni_nll", "prior_top1", "prior_nll")
            if k in res["all"]]
    lines = ["bucket      " + " ".join(f"{k:>10}" for k in keys)]
    for name in ("all", *hp.ELO_BUCKETS):
        if name in res:
            lines.append(f"{name:<11} " + " ".join(
                f"{res[name][k]:>10.0f}" if k == "n" else f"{res[name][k]:>10.4f}" for k in keys))
    return "\n".join(lines)


def max_damage_choice(state_string: str, choices: list[str]) -> str | None:
    """The legal non-Tera move with the largest expected damage (capped at the target's HP).

    Baseline for "plays the obvious attack"; ``None`` if no legal move deals damage.
    """
    import poke_engine

    state = poke_engine.State.from_string(state_string)
    target = state.side_two.pokemon[hp._active_pos(state.side_two)]
    hp_left = float(target.hp)
    best, best_dmg = None, 0.0
    for choice in choices:
        if choice.startswith("switch") or choice.endswith("-tera"):
            continue
        rolls = poke_engine.calculate_damage(state, choice, "none", True)[0]
        dmg = min(sum(rolls) / max(1, len(rolls)), hp_left)
        if dmg > best_dmg:
            best, best_dmg = choice, dmg
    return best


def baseline_report(model_path: str, data: str, rows: int = 30000, seed: int = 0) -> dict[str, Any]:
    """Top-1 of the model, uniform and "max damage" on a test sample, by decision type.

    The sample takes an even slice of rows from spread-out test shards. "all" counts a switch
    decision as a miss for max-damage; "moves" restricts to decisions where the human attacked or
    used a status move (max-damage can only be right there).
    """
    model = hp.HumanPolicy.load(model_path)
    files = shard_files(data, "test")
    per = max(1, rows // max(1, len(files)))
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    rng = random.Random(seed)
    for f in files:
        got = 0
        for row in read_rows(f):
            if got >= per:
                break
            if rng.random() > 0.35 or not row.get("action_legal", True) or len(row["legal"]) < 2:
                continue
            got += 1
            choices = [c for c in row["legal"]]
            dist = model.predict(row["state"], choices, row.get("elo"))
            kind = "switch" if row["action"].startswith("switch") else "move"
            md = max_damage_choice(row["state"], choices)
            for key in ("all", kind):
                a = acc[key]
                a["n"] += 1
                a["model_top1"] += max(dist, key=dist.get) == row["action"]
                a["uniform_top1"] += 1.0 / len(choices)
                a["maxdmg_top1"] += md == row["action"]
                a["model_nll"] += -math.log(max(dist[row["action"]], 1e-9))
                a["uniform_nll"] += math.log(len(choices))
    return {k: {m: (v if m == "n" else v / a["n"]) for m, v in a.items()} for k, a in acc.items()}


def opponent_view_report(model_path: str, data: str, games: int = 600, seed: int = 0) -> dict[str, Any]:
    """Accuracy of the live-search path: predict a player's action from the OTHER player's state.

    In a search world the opponent is read off a state built from our side, whose opposing team is
    a sample. The same situation exists in the replay data: for every (game, turn) both seats have
    a row, each with its own hindsight-sampled picture of the other team. This predicts seat B's
    real action from seat A's row (state swapped, side two as actor, exactly as ``SearchPlayer``
    does) and compares with predicting it from B's own row. Choices that A's sampled picture of
    B's team cannot express (a move B really has but A's hindsight team lacks) are dropped from the
    legal list; ``coverage`` is the share of decisions whose real action survived.
    """
    model = hp.HumanPolicy.load(model_path)
    files = shard_files(data, "test")
    rng = random.Random(seed)
    rng.shuffle(files)
    acc: dict[str, float] = defaultdict(float)
    seen_games = 0
    for f in files:
        by_tag: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
        for row in read_rows(f):
            by_tag[row["battle_tag"]][int(row["turn"])][row["role"]] = row
        tags = sorted(by_tag)
        rng.shuffle(tags)
        for tag in tags[: max(1, (games - seen_games) // 4)]:
            seen_games += 1
            for pair in by_tag[tag].values():
                if len(pair) != 2:
                    continue
                for src, dst in (("p1", "p2"), ("p2", "p1")):
                    a, b = pair[src], pair[dst]
                    if not b.get("action_legal", True) or len(b["legal"]) < 2:
                        continue
                    acc["n"] += 1
                    own = model.predict(b["state"], b["legal"], b.get("elo"))
                    acc["own_top1"] += max(own, key=own.get) == b["action"]
                    acc["own_nll"] += -math.log(max(own[b["action"]], 1e-9))
                    acc["uniform_nll"] += math.log(len(b["legal"]))
                    state = a["state"]
                    cols = [c for c in b["legal"] if _expressible(state, c)]
                    if b["action"] not in cols or len(cols) < 2:
                        continue
                    acc["covered"] += 1
                    dist = model.predict(state, cols, b.get("elo"), side=2)
                    acc["cov_view_top1"] += max(dist, key=dist.get) == b["action"]
                    acc["cov_view_nll"] += -math.log(max(dist[b["action"]], 1e-9))
                    acc["cov_view_uniform_nll"] += math.log(len(cols))
                    sub = model.predict(b["state"], cols, b.get("elo"))
                    acc["cov_own_top1"] += max(sub, key=sub.get) == b["action"]
                    acc["cov_own_nll"] += -math.log(max(sub[b["action"]], 1e-9))
        if seen_games >= games:
            break
    n, c = max(1.0, acc["n"]), max(1.0, acc["covered"])
    return {
        "decisions": acc["n"], "coverage": acc["covered"] / n,
        "own_row": {"top1": acc["own_top1"] / n, "nll": acc["own_nll"] / n, "uniform_nll": acc["uniform_nll"] / n},
        "covered_decisions": acc["covered"],
        "own_row_on_covered": {"top1": acc["cov_own_top1"] / c, "nll": acc["cov_own_nll"] / c},
        "other_players_state_on_covered": {
            "top1": acc["cov_view_top1"] / c, "nll": acc["cov_view_nll"] / c,
            "uniform_nll": acc["cov_view_uniform_nll"] / c},
    }


_EXPRESS_CACHE: dict[str, tuple[list[str], list[str]]] = {}


def _expressible(state_string: str, choice: str) -> bool:
    """Is ``choice`` one of the swapped state's side-two actor slots?"""
    import poke_engine

    slots = _EXPRESS_CACHE.get(state_string)
    if slots is None:
        slots = hp.actor_slots(poke_engine.State.from_string(state_string).side_two)
        if len(_EXPRESS_CACHE) > 4096:
            _EXPRESS_CACHE.clear()
        _EXPRESS_CACHE[state_string] = slots
    return hp.slot_of(choice, *slots) is not None


def main(argv: list[str] | None = None) -> int:
    import torch
    from torch.utils.data import DataLoader

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/human/pc1/policy")
    p.add_argument("--out", help="output .npz")
    p.add_argument("--report", help="skip training: print the baseline comparison table for this model .npz")
    p.add_argument("--report-rows", type=int, default=30000)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--h1", type=int, default=1024)
    p.add_argument("--h2", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-train-shards", type=int, default=0, help="use only the first N shards (random subset)")
    p.add_argument("--eval-shards", type=int, default=0, help="test shards to evaluate (0 = all)")
    p.add_argument("--elo-dropout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    args.workers = min(args.workers, 8)
    if args.report:
        print(json.dumps(baseline_report(args.report, args.data, args.report_rows), indent=1))
        print(json.dumps(opponent_view_report(args.report, args.data), indent=1))
        return 0
    if not args.out:
        p.error("--out is required unless --report is given")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    train = shard_files(args.data, "train")
    test = shard_files(args.data, "test")
    if args.max_train_shards:
        train = random.Random(args.seed).sample(train, min(args.max_train_shards, len(train)))
    if args.eval_shards:
        test = random.Random(1).sample(test, min(args.eval_shards, len(test)))
    print(f"train shards {len(train)}, test shards {len(test)}, device {device}", flush=True)

    mu, sd = feature_stats(train)
    model = make_model(args.h1, args.h2, mu, sd).to(device)
    print("params", sum(q.numel() for q in model.parameters()), flush=True)

    # ~10.5k rows per shard, ~85% of them with >= 2 legal actions
    steps_per_epoch = int(len(train) * 9000 / args.batch)
    total = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total + 10, pct_start=0.05, anneal_strategy="cos")
    slot_counts = np.ones(hp.N_SLOTS)
    step, epoch, started = 0, 0, time.time()
    run_loss, run_n = 0.0, 0
    done = False
    meta: dict[str, Any] = {"h1": args.h1, "h2": args.h2, "args": vars(args)}
    while not done:
        stream = BatchStream(train, batch=args.batch, shuffle=True, seed=args.seed, epoch=epoch)
        loader = DataLoader(stream, batch_size=None, num_workers=args.workers, prefetch_factor=4)
        for x, mask, y, elo, _n in loader:
            x, mask, y, elo = x.to(device, non_blocking=True), mask.to(device), y.to(device), elo.to(device)
            if epoch == 0:
                slot_counts += np.bincount(y.cpu().numpy(), minlength=hp.N_SLOTS)
            drop = torch.rand(elo.shape, device=device) < args.elo_dropout
            elo_in = torch.where(drop, torch.full_like(elo, hp.N_ELO), elo)
            loss = torch.nn.functional.cross_entropy(model(x, mask, elo_in), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item() * len(y)
            run_n += len(y)
            if step % 500 == 0:
                print(f"step {step}/{total} epoch {epoch} loss {run_loss / run_n:.4f} "
                      f"{step * args.batch / (time.time() - started):.0f} rows/s", flush=True)
                run_loss, run_n = 0.0, 0
            if step >= total:
                done = True
                break
        epoch += 1
        meta["steps"] = step
        export(model, meta).save(args.out)
        print(f"epoch {epoch} done at step {step}; saved {args.out}", flush=True)

    prior = slot_counts / slot_counts.sum()
    meta.update(train_rows_seen=step * args.batch, prior=prior.tolist(), seconds=time.time() - started)
    policy = export(model, meta)
    policy.save(args.out)
    res = evaluate(model, test, args.workers, device, prior=prior)
    res_unknown = evaluate(model, test, args.workers, device, elo_known=False, prior=prior)
    print("TEST (elo bucket known to the model)\n" + format_table(res), flush=True)
    print("TEST (elo unknown)\n" + format_table(res_unknown), flush=True)
    meta["test"], meta["test_elo_unknown"] = res, res_unknown
    export(model, meta).save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps({"test": res, "test_unknown_elo": res_unknown}, indent=1))
    return 0
