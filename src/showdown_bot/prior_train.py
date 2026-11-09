"""Train the MCTS policy prior (``.ppol``) by distilling root visit shares (expert iteration).

    python -m showdown_bot.prior_train --data data/selfplay/gen6 --max-games 1500 \\
        --hidden 32 --out models/prior-gen6-h32.ppol

Data: self-play records (``docs/selfplay-data.md``). Every searched world of a decision is one
sample: input = the engine's v2 features of the world's state from our (side one's) view,
target = the world's root visit share over our options, mapped onto the engine's 13 action
slots (4 moves, 4 tera moves, 5 switch slots; ``poke_engine.policy_actions`` is the single
source of that mapping, so training and the Rust search cannot disagree).

Each option also carries the engine's action features (``poke_engine.policy_action_features``:
damage / KO / effectiveness / property flags of that move, matchup of a switch target, ...):
the v2 state features aggregate the four moves, so the state alone cannot say which slot is
which move. Net (``engine/src/prior.rs``)::

    hr = relu(W1 x + b1);  logit(s) = b2[s] + W2[s] . hr + v . relu(P hr + c + A af(s))

Loss = cross-entropy of the masked softmax over the legal options against the visit share
(``--target-temp T`` trains against ``visits ** (1/T)`` renormalised instead; reports always
score the raw visit share).
Train/validation are split by battle tag. The export goes through ``poke_engine.PolicyPrior``
(Rust serialiser) and is checked against the torch logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

N_ACTIONS = 13
F_DAMAGE_FRAC = 13  # index of the damage-fraction action feature (heuristic baseline)


def battle_tag(path: Path) -> str:
    return path.name.split("__", 1)[0]


def is_val(tag: str, val_frac: float, seed: int) -> bool:
    h = int(hashlib.sha1(f"{seed}:{tag}".encode()).hexdigest()[:8], 16)
    return (h % 10_000) < val_frac * 10_000


def load_file(args: tuple[str, int, int, int, int]) -> dict[str, Any] | None:
    """Samples of one recording: features (float16), 13-slot visits and legal mask."""
    path, feat_ver, max_dec, min_visits, seed = args
    import poke_engine
    from poke_engine import State

    from showdown_bot import recorder

    try:
        game = recorder.read_game(path)
    except Exception:
        return None
    decisions = [d for d in game["decisions"] if d.get("worlds")]
    rng = random.Random(f"{seed}:{path}")
    if max_dec and len(decisions) > max_dec:
        decisions = rng.sample(decisions, max_dec)
    feats, afs, visits, mask, states = [], [], [], [], []
    for d in decisions:
        for w in d["worlds"]:
            moves = w.get("moves") or []
            if len(moves) < 2 or (w.get("total_visits") or 0) < min_visits:
                continue
            try:
                st = State.from_string(w["state"])
                acts = poke_engine.policy_action_features(st, 1)
            except Exception:
                continue
            slot_of = {c: s for c, s, _f in acts}
            if any(slot_of.get(m["choice"], -1) < 0 for m in moves):
                continue
            v = np.zeros(N_ACTIONS, np.float32)
            legal = np.zeros(N_ACTIONS, np.bool_)
            af = np.zeros((N_ACTIONS, len(acts[0][2])), np.float16)
            for c, s, f in acts:
                if s >= 0:
                    legal[s] = True
                    af[s] = f
            for m in moves:
                v[slot_of[m["choice"]]] += float(m["visits"])
            if v.sum() <= 0:
                continue
            feats.append(np.asarray(poke_engine.state_features(st, 1, feat_ver), np.float16))
            afs.append(af)
            visits.append(v)
            mask.append(legal)
            states.append(w["state"])
    if not feats:
        return None
    return {"x": np.stack(feats), "af": np.stack(afs), "v": np.stack(visits), "m": np.stack(mask),
            "states": states}


def build(paths: list[Path], feat_ver: int, max_dec: int, min_visits: int, seed: int, workers: int):
    jobs = [(str(p), feat_ver, max_dec, min_visits, seed) for p in paths]
    out = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(load_file, jobs, chunksize=8)):
            if r is not None:
                out.append(r)
            if (i + 1) % 500 == 0:
                print(f"  loaded {i + 1}/{len(jobs)} files", flush=True)
    return out


def to_device(parts: list[dict[str, Any]], dev: str) -> dict[str, Any]:
    """Pack the per-file parts into preallocated device tensors, freeing each part as it goes
    (a concatenate would need twice the memory)."""
    import torch

    n = sum(len(p["x"]) for p in parts)
    n_in, n_af = parts[0]["x"].shape[1], parts[0]["af"].shape[2]
    x = torch.empty((n, n_in), dtype=torch.float16, device=dev)
    af = torch.empty((n, N_ACTIONS, n_af), dtype=torch.float16, device=dev)
    v = np.concatenate([p["v"] for p in parts])
    m = np.concatenate([p["m"] for p in parts])
    states: list[str] = []
    i = 0
    while parts:
        part = parts.pop(0)
        k = len(part["x"])
        x[i : i + k] = torch.from_numpy(part["x"]).to(dev)
        af[i : i + k] = torch.from_numpy(part["af"]).to(dev)
        states += part["states"]
        i += k
    return {"x": x, "af": af, "v": v, "m": m, "states": states, "n": n}


def metrics(logits: np.ndarray, v: np.ndarray, m: np.ndarray, af: np.ndarray | None = None) -> dict[str, float]:
    z = np.where(m, logits, -1e9)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) * m
    p = p / p.sum(1, keepdims=True)
    t = v / v.sum(1, keepdims=True)
    ce = -(t * np.log(np.maximum(p, 1e-9))).sum(1)
    k = m.sum(1)
    uniform_ce = -(t * np.log(1.0 / k)[:, None]).sum(1)
    ent = -(t * np.log(np.maximum(t, 1e-9))).sum(1)
    top = p.argmax(1)
    best = np.where(m, v, -1).argmax(1)
    sharp = t.max(1) >= 0.6
    extra = {}
    if af is not None:
        # heuristic baseline: the legal option with the highest estimated damage fraction
        # (switches / status moves score 0, ties go to the lowest slot)
        score = np.where(m, af[:, :, F_DAMAGE_FRAC].astype(np.float32), -1.0)
        extra["top1_max_damage_heuristic"] = float((score.argmax(1) == best).mean())
    return {
        **extra,
        "n": int(len(v)),
        "ce": float(ce.mean()),
        "ce_uniform": float(uniform_ce.mean()),
        "target_entropy": float(ent.mean()),
        "top1": float((top == best).mean()),
        "top1_uniform_baseline": float((1.0 / k).mean()),
        "top1_sharp": float((top == best)[sharp].mean()) if sharp.any() else float("nan"),
        "frac_sharp": float(sharp.mean()),
        "mass_on_best": float(p[np.arange(len(p)), best].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="self-play dir (repeatable)")
    ap.add_argument("--out", required=True, help="output .ppol")
    ap.add_argument("--hidden", type=int, default=32, choices=(16, 32, 64, 128))
    ap.add_argument("--max-games", type=int, default=0, help="record files to read (0: all)")
    ap.add_argument("--max-per-game", type=int, default=16, help="decisions kept per record (0: all)")
    ap.add_argument("--min-visits", type=int, default=2000, help="skip worlds searched less than this")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--target-temp", type=float, default=1.0, help="train on visits**(1/T); 1 = raw share")
    ap.add_argument("--feature-version", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args(argv)

    import torch

    t0 = time.time()
    files: list[Path] = []
    for d in args.data:
        files += sorted(Path(d).glob("*.jsonl.gz"))
    random.Random(args.seed).shuffle(files)
    if args.max_games:
        files = files[: args.max_games]
    tr_files = [f for f in files if not is_val(battle_tag(f), args.val_frac, args.seed)]
    va_files = [f for f in files if is_val(battle_tag(f), args.val_frac, args.seed)]
    print(f"{len(files)} records: {len(tr_files)} train, {len(va_files)} val (split by battle)", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = to_device(build(tr_files, args.feature_version, args.max_per_game, args.min_visits, args.seed, args.workers), dev)
    va = to_device(build(va_files, args.feature_version, args.max_per_game, args.min_visits, args.seed, args.workers), dev)
    print(f"samples: train {tr['n']}, val {va['n']}  ({time.time() - t0:.0f}s)", flush=True)

    torch.manual_seed(args.seed)
    n_in = tr["x"].shape[1]
    n_af, n_emb = tr["af"].shape[2], 16

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(n_in, args.hidden)
            self.l2 = torch.nn.Linear(args.hidden, N_ACTIONS)
            self.p = torch.nn.Linear(args.hidden, n_emb)
            self.a = torch.nn.Linear(n_af, n_emb, bias=False)
            self.v = torch.nn.Linear(n_emb, 1, bias=False)

        def forward(self, x, af):
            hr = torch.relu(self.l1(x))
            z = torch.relu(self.p(hr)[:, None, :] + self.a(af))
            return self.l2(hr) + self.v(z).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def tensors(d):
        # x / af stay float16 on the device (cast per batch)
        return d["x"], d["af"], torch.from_numpy(d["v"]).to(dev), torch.from_numpy(d["m"]).to(dev)

    xtr, aftr, vtr, mtr = tensors(tr)
    xva, afva, vva, mva = tensors(va)

    def target(v, m):
        t = (v.clamp_min(0) ** (1.0 / args.target_temp)) * m
        return t / t.sum(1, keepdim=True)

    def loss_fn(logits, v, m):
        z = logits.masked_fill(~m, -1e9)
        logp = torch.log_softmax(z, 1)
        return -(target(v, m) * logp).sum(1).mean()

    def predict(x, af, bs=8192):
        net.eval()
        with torch.no_grad():
            out = torch.cat([net(x[i : i + bs].float(), af[i : i + bs].float()) for i in range(0, len(x), bs)])
        net.train()
        return out

    best, best_state, bad = math.inf, None, 0
    n = len(xtr)
    for epoch in range(args.epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            loss = loss_fn(net(xtr[idx].float(), aftr[idx].float()), vtr[idx], mtr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        pv = predict(xva, afva)
        val = loss_fn(pv, vva, mva)
        mv = metrics(pv.cpu().numpy(), va["v"], va["m"])
        print(f"epoch {epoch + 1:2d} train {tot / n:.4f} val {float(val):.4f} top1 {mv['top1']:.3f} "
              f"ce_raw {mv['ce']:.4f}", flush=True)
        if float(val) < best - 1e-4:
            best, bad = float(val), 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    net.load_state_dict(best_state)

    final = metrics(predict(xva, afva).cpu().numpy(), va["v"], va["m"], afva.cpu().numpy())
    k = min(n, 200_000)
    train_m = metrics(predict(xtr[:k], aftr[:k]).cpu().numpy(), tr["v"][:k], tr["m"][:k], aftr[:k].cpu().numpy())
    print("VAL  ", json.dumps(final))
    print("TRAIN", json.dumps(train_m))

    # export through the Rust serialiser and check it against torch
    import poke_engine
    from poke_engine import PolicyPrior, State

    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    flat = lambda k: sd[k].reshape(-1).tolist()  # noqa: E731
    prior = PolicyPrior.mlp(
        flat("l1.weight"), flat("l1.bias"), flat("l2.weight"), flat("l2.bias"),
        flat("p.weight"), flat("p.bias"), flat("a.weight"), flat("v.weight"), args.feature_version,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    prior.save(args.out)
    prior = PolicyPrior.load(args.out)
    worst = 0.0
    sample = random.Random(1).sample(range(len(va["states"])), min(40, len(va["states"])))
    ref = predict(xva[sample], afva[sample]).cpu().numpy()
    for row, k in zip(ref, sample):
        for _choice, slot, z in prior.option_logits(State.from_string(va["states"][k]), 1):
            worst = max(worst, abs(z - float(row[slot])))
    print(f"rust-vs-torch max |logit diff| over {len(sample)} val states: {worst:.2e}")
    if worst > 5e-2:
        print("ERROR: exported net does not match torch", file=sys.stderr)
        return 1
    meta = {"args": vars(args), "val": final, "train": train_m, "n_train": len(xtr), "n_val": len(xva)}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
