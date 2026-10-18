"""Train the tiny transformer on self-play positions.

    python learn/train.py                  # all 10000 files (both gens), 3 epochs
    python learn/train.py --files 500 --epochs 3      # quick smoke test

Needs `poke_engine` importable (it parses the engine's state strings).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyTransformer                      # noqa: E402
from tokenizer import Vocab, N_NUM, OWN0, OPP0, policy_target, tokenize_raw  # noqa: E402

DATA_ROOT = r"C:\Users\Lewis\pokemon-showdown-bot\data\selfplay"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def battle_id(path: str) -> str:
    """Files are '<battle-tag>__<username>.jsonl.gz'. Both players' views of one battle share the tag,
    and we must keep a whole battle on one side of the train/val split (see README)."""
    return os.path.basename(path).split("__")[0]


def pick_files(n_files: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    per_gen = n_files // 2
    files = []
    for gen in ("gen8", "gen9"):
        fs = sorted(glob.glob(os.path.join(DATA_ROOT, gen, "*.jsonl.gz")))
        rng.shuffle(fs)
        files += fs[:per_gen]
    return files


def load_positions(files):
    """Parse files -> flat per-position arrays (strings not yet turned into ids)."""
    import poke_engine
    cat_rows, nums, targets, legals, values, turns, games, meta = [], [], [], [], [], [], [], []
    for fi, path in enumerate(files):
        rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
        result = rows[-1]
        if result.get("type") != "result":
            continue
        won = 1.0 if result["result"] == "win" else 0.0
        for r in rows:
            if r.get("type") != "decision" or r.get("truth_status") != "ok" or not r.get("truth"):
                continue
            state = poke_engine.State.from_string(r["truth"]["state"])
            persp = r["side"]                       # 'one' in this data: side_one is "us"
            tgt, legal = policy_target(state, r["policy"], persp)
            if tgt.sum() == 0:
                continue
            cats, num = tokenize_raw(state, persp)
            cat_rows.append(cats)
            nums.append(num)
            targets.append(tgt)
            legals.append(legal)
            values.append(won)
            turns.append(r["turn"])
            games.append(battle_id(path))
            meta.append(fi)
    return dict(cats=np.stack(cat_rows), nums=np.stack(nums), target=np.stack(targets),
                legal=np.stack(legals), value=np.asarray(values, np.float32),
                turn=np.asarray(turns), game=np.asarray(games), file=np.asarray(meta))


def to_device(d, idx, vocab, dev):
    return dict(
        cats=torch.from_numpy(vocab.encode(d["cats"][idx]).astype(np.int32)).to(dev),
        nums=torch.from_numpy(d["nums"][idx]).to(dev),
        target=torch.from_numpy(d["target"][idx]).to(dev),
        legal=torch.from_numpy(d["legal"][idx]).to(dev),
        value=torch.from_numpy(d["value"][idx]).to(dev),
        turn=torch.from_numpy(d["turn"][idx]).to(dev),
    )


def policy_loss(logits, target):
    """Cross-entropy against the search's visit shares (a soft target). Masked slots have target 0."""
    logp = F.log_softmax(logits, dim=-1).masked_fill(target == 0, 0.0)
    return -(target * logp).sum(-1)       # (B,)


@torch.no_grad()
def evaluate(model, D, bs=1024):
    model.eval()
    n = len(D["value"])
    vp, pl, top1 = [], [], []
    for i in range(0, n, bs):
        s = slice(i, i + bs)
        v, lg = model(D["cats"][s].long(), D["nums"][s], D["legal"][s])
        vp.append(torch.sigmoid(v))
        pl.append(policy_loss(lg, D["target"][s]))
        top1.append((lg.argmax(-1) == D["target"][s].argmax(-1)).float())
    vp, pl, top1 = torch.cat(vp), torch.cat(pl), torch.cat(top1)
    m = D["turn"] >= 3
    y = D["value"]
    eps = 1e-6
    bce = -(y * (vp + eps).log() + (1 - y) * (1 - vp + eps).log())
    return dict(v_bce=bce[m].mean().item(), v_acc=((vp > 0.5).float() == y)[m].float().mean().item(),
                p_ce=pl.mean().item(), p_top1=top1.mean().item())


def hp_baseline(tr, va):
    """Logistic regression on ONE feature: (sum of our hp fractions) - (sum of their hp fractions)."""
    def feat(D):
        hp = D["nums"][:, :, 0]
        return hp[:, OWN0:OWN0 + 6].sum(1) - hp[:, OPP0:OPP0 + 6].sum(1)
    xt, yt = feat(tr)[tr["turn"] >= 3], tr["value"][tr["turn"] >= 3]
    w = torch.zeros(2, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(w[0] * xt + w[1], yt)
        loss.backward()
        return loss
    opt.step(closure)
    m = va["turn"] >= 3
    xv, yv = feat(va)[m], va["value"][m]
    logit = w[0] * xv + w[1]
    return (F.binary_cross_entropy_with_logits(logit, yv).item(),
            ((logit > 0).float() == yv).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min_count", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--emb_lr_mult", type=float, default=1.0,
                    help="scale the embedding table's learning rate (small = harder to memorise teams)")
    ap.add_argument("--no_cats", action="store_true",
                    help="ablation: blank every categorical id (model sees only the numbers)")
    ap.add_argument("--field_drop", type=float, default=0.5,
                    help="chance of blanking each categorical field during training (regulariser)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print(f"parsing {a.files} files ...")
    data = load_positions(pick_files(a.files))
    N = len(data["value"])
    print(f"{N} positions from {len(set(data['game']))} battles  ({time.time() - t0:.0f}s)")

    # ---- split BY BATTLE: every position of a battle lands entirely in train or in val
    games = sorted(set(data["game"]))
    random.Random(1).shuffle(games)
    val_games = set(games[: len(games) // 10])
    is_val = np.asarray([g in val_games for g in data["game"]])
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"train {len(tr_idx)} positions / val {len(va_idx)} positions")

    # ---- vocab from TRAIN positions only (val-only strings become <unk>, like at deployment)
    vocab = Vocab.build(data["cats"][tr_idx].reshape(len(tr_idx), -1), a.min_count)
    print(f"vocab size {len(vocab)}")
    tr = to_device(data, tr_idx, vocab, dev)
    va = to_device(data, va_idx, vocab, dev)
    if a.no_cats:
        tr["cats"].zero_(); va["cats"].zero_()
    tr_cpu = dict(nums=tr["nums"].cpu(), turn=tr["turn"].cpu(), value=tr["value"].cpu())
    va_cpu = dict(nums=va["nums"].cpu(), turn=va["turn"].cpu(), value=va["value"].cpu())

    # ---- baselines
    mv = va_cpu["turn"] >= 3
    p = va_cpu["value"][mv].mean().item()
    print("\nBASELINES (val)")
    print(f"  value constant 0.5        : BCE {np.log(2):.4f}  acc {max(p, 1 - p):.4f} (majority class)")
    hb, ha = hp_baseline(tr_cpu, va_cpu)
    print(f"  value HP-sum logistic     : BCE {hb:.4f}  acc {ha:.4f}")
    n_legal = va["legal"].sum(-1)
    uni = (n_legal.log()).mean().item()
    ent = -(va["target"] * va["target"].clamp_min(1e-9).log()).sum(-1).mean().item()
    print(f"  policy uniform-over-legal : CE {uni:.4f}  top1 {(1 / n_legal).mean().item():.4f}")
    print(f"  (policy CE floor = entropy of search targets: {ent:.4f})")

    model = TinyTransformer(len(vocab), dropout=a.dropout).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.emb.weight.numel()
    print(f"\nmodel params: {n_params:,} (embedding table {n_emb:,}, rest {n_params - n_emb:,})")
    emb_p = list(model.emb.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": a.lr}, {"params": emb_p, "lr": a.lr * a.emb_lr_mult}],
                            lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((len(tr_idx) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * a.emb_lr_mult], total_steps=steps, pct_start=0.1)

    print("\nepoch | train loss | val value BCE / acc | val policy CE / top1 | time")
    n_tr = len(tr_idx)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot, cnt = 0.0, 0
        for i in range(0, n_tr, a.bs):
            b = perm[i:i + a.bs]
            cats = tr["cats"][b].long()
            if a.field_drop > 0:   # randomly blank categorical fields so the model can't memorise exact teams
                cats = cats.masked_fill(torch.rand(cats.shape, device=dev) < a.field_drop, 0)
            v, lg = model(cats, tr["nums"][b], tr["legal"][b],
                          move_ids=tr["cats"][b][:, OWN0, 4:8].long())
            loss = F.binary_cross_entropy_with_logits(v, tr["value"][b]) \
                + policy_loss(lg, tr["target"][b]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
            cnt += len(b)
        m = evaluate(model, va)
        mt = evaluate(model, {k: v[::9] for k, v in tr.items()})   # same metrics on (a sample of) TRAIN
        print(f"{ep:5d} | {tot / cnt:10.4f} | {m['v_bce']:.4f} / {m['v_acc']:.4f}     | "
              f"{m['p_ce']:.4f} / {m['p_top1']:.4f}      | {time.time() - t0:.0f}s"
              f"   [train-sample value BCE {mt['v_bce']:.4f} acc {mt['v_acc']:.4f}]", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict(), "itos": vocab.itos}, os.path.join(OUT_DIR, "model.pt"))
    # keep the validation file list so inspect_attention.py can load a few val positions
    json.dump({"val_games": sorted(val_games), "files": a.files}, open(os.path.join(OUT_DIR, "split.json"), "w"))
    print(f"\nsaved to {OUT_DIR}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
