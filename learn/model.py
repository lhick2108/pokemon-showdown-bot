"""A tiny transformer written from scratch (no nn.TransformerEncoder / nn.MultiheadAttention).

Notation used in shape comments:
    B = batch size, T = 14 tokens, D = d_model (64), H = heads (4), Dh = D / H = 16
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import N_CAT, N_NUM, N_TOKENS, OWN0


class SelfAttention(nn.Module):
    """Multi-head self-attention, spelled out step by step.

    Every token asks a question (query), advertises what it holds (key), and carries a
    payload (value).  Token i's new vector is a weighted average of all tokens' values,
    where the weight on token j is how well i's query matches j's key.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)   # x -> queries
        self.k = nn.Linear(d_model, d_model)   # x -> keys
        self.v = nn.Linear(d_model, d_model)   # x -> values
        self.out = nn.Linear(d_model, d_model)  # mixes the heads back together

    def forward(self, x):                      # x: (B, T, D)
        B, T, D = x.shape
        # Project, then split D into H heads of size Dh: (B, T, D) -> (B, H, T, Dh)
        def split(t):
            return t.view(B, T, self.h, self.dh).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        # Scores: for every head, every (query token i, key token j) pair -> (B, H, T, T).
        # Dividing by sqrt(Dh) keeps the dot products from growing with head size,
        # which would make the softmax saturate.
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        attn = scores.softmax(dim=-1)          # each row sums to 1: "how much do I look at j?"
        mixed = attn @ v                       # weighted average of values: (B, H, T, Dh)
        mixed = mixed.transpose(1, 2).reshape(B, T, D)   # glue heads back: (B, T, D)
        return self.out(mixed), attn           # attn is returned so we can look at it later


class Block(nn.Module):
    """One transformer layer:  x = x + Attention(LN(x));  x = x + MLP(LN(x)).

    * residual ("x + ...") lets each layer make a small edit instead of rewriting the vector,
      and gives gradients a straight path back.
    * LayerNorm rescales each token's vector to mean 0 / variance 1 (pre-norm variant).
    * the MLP processes each token on its own (attention is where tokens talk to each other).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # (B, T, D) -> (B, T, D)
        a, attn = self.attn(self.ln1(x))
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, attn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        D = d_model
        # --- token embedding: four ingredients that get SUMMED into one D-vector per token
        self.emb = nn.Embedding(vocab_size, D, padding_idx=0)  # (a) categorical ids -> vectors
        self.num_proj = nn.Linear(N_NUM, D)                    # (b) numeric features -> vector
        self.slot_emb = nn.Embedding(N_TOKENS, D)              # (c) which of the 14 positions
        self.side_emb = nn.Embedding(3, D)                     # (d) 0 summary/field, 1 own, 2 opp
        self.register_buffer("slot_ids", torch.arange(N_TOKENS))
        side = torch.tensor([0] + [1] * 6 + [2] * 6 + [0])
        self.register_buffer("side_ids", side)

        self.blocks = nn.ModuleList(Block(D, n_heads, dropout) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(D)

        # --- heads
        self.value_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        # move logit i = f([own active token, embedding of move i]) ; see forward()
        self.move_head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 1))
        # switch logit j = g(bench token j)
        self.switch_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))

    def forward(self, cats, nums, legal=None, return_attn: bool = False, move_ids=None):
        """cats (B,14,11) long, nums (B,14,17) float, legal (B,9) 0/1 or None.
        move_ids: optional (B,4) un-blanked move ids (training blanks some fields as a regulariser).
        Returns value_logit (B,), policy_logits (B,9) [4 moves, 5 switches], optionally attn list."""
        # (a) look up every categorical id: (B,14,11,D), then add them up over the 11 fields.
        # <pad> embeds to zeros (padding_idx=0), so empty fields contribute nothing.
        x = self.emb(cats).sum(dim=2)                            # (B, 14, D)
        x = x + self.num_proj(nums)                              # (b)
        x = x + self.slot_emb(self.slot_ids) + self.side_emb(self.side_ids)   # (c), (d): (14,D) broadcasts

        attns = []
        for blk in self.blocks:
            x, a = blk(x)
            attns.append(a)                                      # each (B, H, 14, 14)
        x = self.ln_f(x)                                         # (B, 14, D)

        # value head: read the SUMMARY token (position 0) -> one logit; sigmoid(logit) = P(win)
        value_logit = self.value_head(x[:, 0]).squeeze(-1)       # (B,)

        # policy head
        active = x[:, OWN0]                                      # (B, D) our active mon
        if move_ids is None:                                     # (B, 4) which moves it has
            move_ids = cats[:, OWN0, 4:8]
        move_vecs = self.emb(move_ids)                           # (B, 4, D)
        pair = torch.cat([active.unsqueeze(1).expand(-1, 4, -1), move_vecs], dim=-1)  # (B, 4, 2D)
        move_logits = self.move_head(pair).squeeze(-1)           # (B, 4)
        bench = x[:, OWN0 + 1: OWN0 + 6]                         # (B, 5, D)
        switch_logits = self.switch_head(bench).squeeze(-1)      # (B, 5)
        logits = torch.cat([move_logits, switch_logits], dim=-1)  # (B, 9)
        if legal is not None:                                    # illegal slots -> -inf -> prob 0
            logits = logits.masked_fill(legal == 0, float("-inf"))
        if return_attn:
            return value_logit, logits, attns
        return value_logit, logits
