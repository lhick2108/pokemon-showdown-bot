"""Look inside the trained model: what does the SUMMARY token attend to?

    python learn/inspect_attention.py [--n 3]

How to read the printout
------------------------
One table per position.  Rows are the 14 tokens (SUMMARY, our 6 mons, their 6 mons, FIELD).
Columns h0..h3 are the 4 attention heads of the LAST layer: the number is how much of the
SUMMARY token's attention budget went to that row's token (each head's column sums to 1.00).
  * A column with one big number = that head "looks at" one specific token.
  * Spread-out numbers (~0.07 each) = the head is mostly averaging everything.
  * Different heads may specialise: e.g. one watches our active mon, another the opponent's.
Caveat: attention weights are a hint about information flow, not a full explanation -
earlier layers already mixed tokens together before the last layer looks at them.

Below the table: the model's P(win) next to what actually happened, and the top-3 actions
of the model vs the top-3 of the search (visit share).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyTransformer                                   # noqa: E402
from tokenizer import Vocab, action_slots, OWN0, OPP0, FIELD        # noqa: E402
import train as T                                                    # noqa: E402


def token_label(i, cats):
    if i == 0:
        return "SUMMARY"
    if i == FIELD:
        return f"FIELD ({cats[i][0].split('=')[1]}/{cats[i][1].split('=')[1]})"
    side = "own" if i < OPP0 else "opp"
    active = " *active*" if i in (OWN0, OPP0) else ""
    return f"{side}  {cats[i][0].split('=')[1]}{active}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="how many positions to show")
    a = ap.parse_args()
    ck = torch.load(os.path.join(T.OUT_DIR, "model.pt"), map_location="cpu")
    vocab = Vocab(ck["itos"])
    model = TinyTransformer(len(vocab))
    model.load_state_dict(ck["model"])
    model.eval()
    split = json.load(open(os.path.join(T.OUT_DIR, "split.json")))
    val_games = set(split["val_games"])

    # find a few validation battles (the model never trained on these)
    files = [f for f in T.pick_files(split["files"]) if T.battle_id(f) in val_games][:6]
    data = T.load_positions(files)
    import poke_engine  # noqa: F401  (needed to re-parse states for action names)

    # choose positions in the middle of games, spread across different battles
    idx = [i for i in range(len(data["turn"])) if data["turn"][i] in (6, 7)]
    seen, picks = set(), []
    for i in idx:
        if data["game"][i] not in seen:
            seen.add(data["game"][i])
            picks.append(i)
    picks = picks[: a.n]

    for i in picks:
        cats_raw = data["cats"][i]
        cats = torch.from_numpy(vocab.encode(cats_raw)).unsqueeze(0)
        nums = torch.from_numpy(data["nums"][i]).unsqueeze(0)
        legal = torch.from_numpy(data["legal"][i]).unsqueeze(0)
        with torch.no_grad():
            v, logits, attns = model(cats, nums, legal, return_attn=True)
        last = attns[-1][0]                      # (heads, 14, 14): row 0 = SUMMARY's attention
        print("=" * 72)
        print(f"battle {data['game'][i]}   turn {data['turn'][i]}")
        print(f"{'token':34s} " + " ".join(f"  h{h}" for h in range(last.shape[0])))
        for t in range(14):
            row = " ".join(f"{last[h, 0, t].item():4.2f}" for h in range(last.shape[0]))
            print(f"{token_label(t, cats_raw):34s} {row}")
        print(f"\nmodel P(win) = {torch.sigmoid(v).item():.2f}    actual result: {'WIN' if data['value'][i] else 'LOSS'}")

        # action names: moves of our active, then bench mons -- recovered from the raw token strings
        moves = [c.split("=")[1] for c in cats_raw[OWN0][4:8] if c]
        bench = [cats_raw[OWN0 + 1 + j][0].split("=")[1] for j in range(5)]
        names = moves + [""] * (4 - len(moves)) + ["switch " + b for b in bench]
        p_model = torch.softmax(logits[0], -1).numpy()
        p_search = data["target"][i]
        def top3(p):
            return ", ".join(f"{names[k]} {p[k]:.2f}" for k in np.argsort(-p)[:3] if p[k] > 0)
        print(f"model  top-3: {top3(p_model)}")
        print(f"search top-3: {top3(p_search)}\n")


if __name__ == "__main__":
    main()
