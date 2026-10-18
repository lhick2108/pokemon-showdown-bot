"""Transformer policy / value / auxiliary net on the token representation (bead showdown-bot-589.68.3).

Architecture, heads, losses and CLI: ``docs/token-net.md``. Input layout: ``docs/tokens.md`` (the batch
dict has the shard field names; see ``INPUT_FIELDS``).

Token sequence (94 tokens, fixed layout so shapes are static): ``[CLS, global, side x2, mon x12,
pair x36, history x16, own action x13, opp action x13]``. Pre-norm encoder, attention through
``F.scaled_dot_product_attention`` with a key mask (pad history rows, empty matchup pairs, illegal
action tokens are not attended to).

    net = TokenNet(TokenNetConfig())
    out = net(batch)               # dict of logits
    probs = net.predict(batch)     # {"policy", "value", "opp_policy", ...} numpy, no grad, eval mode
    net.export("models/tok.pt"); net = TokenNet.load("models/tok.pt")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MON, N_SIDE, N_PAIR, N_HIST, N_ACT = 12, 2, 36, 16, 13
N_TOK = 1 + 1 + N_SIDE + N_MON + N_PAIR + N_HIST + 2 * N_ACT  # 94
# token index ranges
T_CLS, T_GLOB = 0, 1
T_SIDE = 2
T_MON = T_SIDE + N_SIDE
T_PAIR = T_MON + N_MON
T_HIST = T_PAIR + N_PAIR
T_ACT = T_HIST + N_HIST  # own actions, then opponent actions
NEG = -1e9

INPUT_FIELDS = (
    "mon_cat", "mon_known", "mon_cont", "side_cont", "glob", "match_pair", "match_agg",
    "act_cat", "act_legal", "act_cont", "meta", "hist_cat", "hist_cont",
)
SPEC_KEYS = ("token_version", "shard_format", "vocab_hash", "vocab_sizes", "fields", "hist", "n_actions")


def spec_hash(spec: dict[str, Any]) -> str:
    """Short hash of the token spec (version, vocab hash/sizes, field layout) a net was trained on."""
    keep = {k: spec[k] for k in SPEC_KEYS if k in spec}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TokenNetConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    # vocabularies (tables 0 pad, 1 unknown, v + 2 variants), from the shard spec
    n_species: int = 1471
    n_item: int = 260
    n_ability: int = 324
    n_move: int = 887
    n_tera: int = 20  # raw PokemonType discriminants 0-17, 18 Stellar, 19 none
    # continuous widths
    c_mon: int = 173
    c_side: int = 62
    c_glob: int = 15
    c_pair: int = 5
    c_agg: int = 8
    c_act: int = 58
    c_hist: int = 5
    known_w: int = 8
    spec_hash: str = ""  # token spec the net was trained on ("" = unchecked)

    @staticmethod
    def from_spec(spec: dict[str, Any], **kw: Any) -> "TokenNetConfig":
        v = spec["vocab_sizes"]
        c = spec["constants"]
        return TokenNetConfig(
            n_species=int(v["species"]), n_item=int(v["item"]), n_ability=int(v["ability"]), n_move=int(v["move"]),
            c_mon=int(c["C_MON"]), c_side=int(c["C_SIDE"]), c_glob=int(c["C_GLOB"]), c_pair=int(c["N_PAIR_CH"]),
            c_agg=int(c["N_AGG"]), c_act=int(c["N_AFEAT"]), known_w=int(c["KNOWN_W"]),
            spec_hash=spec_hash(spec), **kw,
        )


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ratio: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ratio * d)
        self.fc2 = nn.Linear(ratio * d, d)
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.n2(x)))))


class TokenNet(nn.Module):
    def __init__(self, cfg: TokenNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or TokenNetConfig()
        d = cfg.d_model
        assert d % cfg.n_heads == 0
        # identity embeddings (tied to the heads that predict the same vocabulary)
        self.e_species = nn.Embedding(cfg.n_species, d)
        self.e_item = nn.Embedding(cfg.n_item, d)
        self.e_ability = nn.Embedding(cfg.n_ability, d)
        self.e_move = nn.Embedding(cfg.n_move, d)
        self.e_movepos = nn.Parameter(torch.zeros(4, d))
        self.e_slot = nn.Embedding(N_MON, d)  # mon slot (own 0-5, opp 6-11), also the pair slots
        self.e_pos = nn.Embedding(N_TOK, d)  # one learned position per token = token type + index
        self.e_hside = nn.Embedding(3, d)
        self.e_hkind = nn.Embedding(4, d)
        self.cls = nn.Parameter(torch.zeros(d))
        # continuous projections
        self.l_mon = nn.Linear(cfg.c_mon, d)
        self.l_known = nn.Linear(cfg.known_w, d, bias=False)
        self.l_side = nn.Linear(cfg.c_side, d)
        self.l_glob = nn.Linear(cfg.c_glob + cfg.c_agg, d)
        self.l_pair = nn.Linear(cfg.c_pair, d)
        self.l_hist = nn.Linear(cfg.c_hist, d)
        self.l_act = nn.Linear(cfg.c_act, d)
        self.n_in = nn.LayerNorm(d)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.n_layers))
        self.n_out = nn.LayerNorm(d)
        # heads
        self.h_pol = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.h_val = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.h_opp_slot = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.h_opp_vocab = nn.Linear(d, d)  # CLS -> tied to move / species embeddings
        self.h_opp_bias = nn.Parameter(torch.zeros(cfg.n_move + cfg.n_species))
        self.h_item = nn.Linear(d, d)
        self.h_abil = nn.Linear(d, d)
        self.h_move = nn.Linear(d, d)
        self.h_tera = nn.Linear(d, cfg.n_tera)
        self.h_aux = nn.Linear(d, 3)  # log turns left, final alive own / opp (fractions)
        self.apply(self._init)
        nn.init.zeros_(self.h_pol[-1].weight)
        nn.init.zeros_(self.h_val[-1].weight)
        nn.init.zeros_(self.h_opp_slot[-1].weight)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encode
    def encode(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Token states ``[B, N_TOK, d]`` after the encoder and the key mask ``[B, N_TOK]`` (True = usable)."""
        dev = b["mon_cat"].device
        bs = b["mon_cat"].shape[0]
        dt = self.l_mon.weight.dtype
        cat = b["mon_cat"].long()
        ar = torch.arange(N_MON, device=dev)

        def f(name: str) -> torch.Tensor:
            return b[name].to(dt)

        mon = (
            self.e_species(cat[..., 0]) + self.e_item(cat[..., 1]) + self.e_ability(cat[..., 2])
            + (self.e_move(cat[..., 3:7]) + self.e_movepos).sum(-2)
            + self.l_mon(f("mon_cont")) + self.l_known(f("mon_known")) + self.e_slot(ar)
        )
        side = self.l_side(f("side_cont"))
        glob = self.l_glob(torch.cat([f("glob"), f("match_agg")], -1)).unsqueeze(1)
        # matchup pairs: pair index own * 6 + opp
        pi = torch.arange(N_PAIR, device=dev)
        pair_x = f("match_pair")
        pair = self.l_pair(pair_x) + self.e_slot(pi // 6) + self.e_slot(6 + pi % 6)
        pair_ok = (pair_x != 0).any(-1)
        # history
        hc = b["hist_cat"].long()
        kind = hc[..., 1]
        hid = torch.where(
            (kind == 3).unsqueeze(-1), self.e_species(hc[..., 2]), self.e_move(hc[..., 2].clamp(max=self.cfg.n_move - 1))
        )  # kind 3 = switch -> species row
        hist = (self.e_hside(hc[..., 0]) + self.e_hkind(kind) + hid + self.e_species(hc[..., 3])
                + self.l_hist(f("hist_cont")))
        hist_ok = hc[..., 0] > 0
        # actions: [B, 2, 13, ...]
        ac = b["act_cat"].long()
        act = self.e_move(ac[..., 0]) + self.e_species(ac[..., 1]) + self.l_act(f("act_cont"))
        act = act + self.e_hside(torch.tensor([1, 2], device=dev)).unsqueeze(1)
        act_ok = b["act_legal"] > 0
        cls = (self.cls).expand(bs, 1, -1).to(dt)
        x = torch.cat([cls, glob, side, mon, pair, hist, act.flatten(1, 2)], 1) + self.e_pos.weight
        ok = torch.cat(
            [torch.ones(bs, 1 + 1 + N_SIDE + N_MON, dtype=torch.bool, device=dev), pair_ok, hist_ok, act_ok.flatten(1, 2)], 1
        )
        x = self.n_in(x)
        mask = ok[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.n_out(x), ok

    # ----------------------------------------------------------------- forward
    def forward(self, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h, _ = self.encode(b)
        h = h.float()
        cfg = self.cfg
        cls = h[:, T_CLS]
        legal = b["act_legal"] > 0
        own = h[:, T_ACT : T_ACT + N_ACT]
        opp = h[:, T_ACT + N_ACT : T_ACT + 2 * N_ACT]
        out: dict[str, torch.Tensor] = {}
        out["policy"] = self.h_pol(own).squeeze(-1).float().masked_fill(~legal[:, 0], NEG)
        out["value"] = self.h_val(cls).squeeze(-1)  # logit of P(own side wins)
        out["opp_slot"] = self.h_opp_slot(opp).squeeze(-1).float().masked_fill(~legal[:, 1], NEG)
        z = self.h_opp_vocab(cls)
        out["opp_vocab"] = torch.cat([z @ self.e_move.weight.float().T, z @ self.e_species.weight.float().T], -1) + self.h_opp_bias
        om = h[:, T_MON + 6 : T_MON + 12]  # opponent mon tokens
        out["hid_item"] = self.h_item(om) @ self.e_item.weight.float().T
        out["hid_ability"] = self.h_abil(om) @ self.e_ability.weight.float().T
        out["hid_move"] = self.h_move(om) @ self.e_move.weight.float().T
        out["hid_tera"] = self.h_tera(om)
        out["aux"] = self.h_aux(cls)
        return {k: v.float() for k, v in out.items()}

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, batch: dict[str, Any], device: str | torch.device | None = None, amp: bool = True) -> dict[str, np.ndarray]:
        """Eval-mode forward on numpy (or tensor) inputs; returns probabilities as numpy arrays.

        ``policy`` [B,13] softmax over the legal own actions, ``value`` [B] P(win), ``opp_policy`` [B,13]
        softmax over the legal opponent action tokens, ``opp_vocab`` [B, n_move + n_species] softmax,
        ``hid_*`` softmaxes over the opponent's hidden item / ability / tera / move for each opponent slot.
        """
        was = self.training
        self.eval()
        dev = torch.device(device) if device is not None else next(self.parameters()).device
        t = to_tensors(batch, dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16 if dev.type == "cpu" else torch.float16, enabled=amp):
            o = self(t)
        res = {
            "policy": torch.softmax(o["policy"], -1),
            "value": torch.sigmoid(o["value"]),
            "opp_policy": torch.softmax(o["opp_slot"], -1),
            "opp_vocab": torch.softmax(o["opp_vocab"], -1),
            "hid_item": torch.softmax(o["hid_item"], -1),
            "hid_ability": torch.softmax(o["hid_ability"], -1),
            "hid_tera": torch.softmax(o["hid_tera"], -1),
            "hid_move": torch.softmax(o["hid_move"], -1),
            "aux": o["aux"],
        }
        self.train(was)
        return {k: v.float().cpu().numpy() for k, v in res.items()}

    # ------------------------------------------------------------------ export
    def export(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        """Writes ``path`` (state_dict + config + spec hash) and ``path`` + ``.json`` (config only)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"config": asdict(self.cfg), "n_params": self.n_params(), "extra": extra or {}}
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}, **meta}, path)
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(meta, indent=1))

    @staticmethod
    def load(path: str | Path, device: str | torch.device = "cpu", expect_spec: dict[str, Any] | None = None,
             check_engine: bool = True) -> "TokenNet":
        """Loads an exported net. The token spec hash must match ``expect_spec`` (a ``token_spec()`` dict) or,
        when ``check_engine`` and the engine is importable, the installed engine's spec."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TokenNetConfig(**blob["config"])
        want = None
        if expect_spec is not None:
            want = spec_hash(expect_spec)
        elif check_engine and cfg.spec_hash:
            try:
                from showdown_bot import tokens

                want = spec_hash(tokens.token_spec())
            except Exception:  # no engine wheel: cannot check
                want = None
        if want is not None and cfg.spec_hash and want != cfg.spec_hash:
            raise ValueError(f"token spec mismatch: net trained on {cfg.spec_hash}, current {want}")
        net = TokenNet(cfg)
        net.load_state_dict(blob["state_dict"])
        return net.to(device).eval()


def to_tensors(batch: dict[str, Any], device: torch.device | str, fields: tuple[str, ...] | None = None) -> dict[str, torch.Tensor]:
    """numpy / tensor dict -> device tensors (floats as float32 so autocast decides the compute type)."""
    out = {}
    for k in fields or batch.keys():
        v = batch[k]
        t = v if isinstance(v, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(v))
        if t.dtype in (torch.float16, torch.float64):
            t = t.float()
        out[k] = t.to(device, non_blocking=True)
    return out


def batch_from_engine(bufs: list[dict[str, np.ndarray]], hist_cat: np.ndarray | None = None,
                      hist_cont: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Stacks ``tokens.split_buffers`` results (one per state) into a batch; the history defaults to padding."""
    b = {k: np.stack([x[k] for x in bufs]) for k in INPUT_FIELDS if k in bufs[0]}
    n = len(bufs)
    b["hist_cat"] = hist_cat if hist_cat is not None else np.zeros((n, N_HIST, 4), np.int16)
    b["hist_cont"] = hist_cont if hist_cont is not None else np.zeros((n, N_HIST, 5), np.float32)
    return b


def bench_latency(cfg: TokenNetConfig | None = None, sizes: tuple[int, ...] = (1, 16, 64, 256), device: str = "cuda",
                  iters: int = 50, warmup: int = 10) -> list[dict[str, float]]:
    """Median forward latency (ms) per batch size: fp16 autocast. Inputs are built on
    the device beforehand; the policy logits are copied back to the host inside the timing, as a bot pays for."""
    import time

    cfg = cfg or TokenNetConfig()
    dev = torch.device(device)
    net = TokenNet(cfg).to(dev).eval()
    rows = []
    for n in sizes:
        b = {
            "mon_cat": torch.zeros(n, N_MON, 7, dtype=torch.long), "mon_known": torch.ones(n, N_MON, cfg.known_w),
            "mon_cont": torch.randn(n, N_MON, cfg.c_mon), "side_cont": torch.randn(n, 2, cfg.c_side),
            "glob": torch.randn(n, cfg.c_glob), "match_pair": torch.randn(n, N_PAIR, cfg.c_pair),
            "match_agg": torch.randn(n, cfg.c_agg), "act_cat": torch.zeros(n, 2, N_ACT, 2, dtype=torch.long),
            "act_legal": torch.ones(n, 2, N_ACT, dtype=torch.long), "act_cont": torch.randn(n, 2, N_ACT, cfg.c_act),
            "meta": torch.zeros(n, 4, dtype=torch.long), "hist_cat": torch.ones(n, N_HIST, 4, dtype=torch.long),
            "hist_cont": torch.randn(n, N_HIST, cfg.c_hist),
        }
        b = {k: v.to(dev) for k, v in b.items()}
        row: dict[str, float] = {"batch": n}
        for name, m, inp, amp in (("fp16_autocast_ms", net, b, True),):
            ts = []
            with torch.no_grad():
                for i in range(warmup + iters):
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                        o = m(inp)
                    _ = o["policy"].float().cpu()
                    if i >= warmup:
                        ts.append((time.perf_counter() - t0) * 1e3)
            row[name] = float(np.median(ts))
        rows.append(row)
    return rows


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="TokenNet utilities")
    ap.add_argument("cmd", choices=("bench", "params"))
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    conf = TokenNetConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)
    print(f"{TokenNet(conf).n_params():,} parameters")
    if args.cmd == "bench":
        torch.set_num_threads(2)
        for r in bench_latency(conf, device=args.device):
            print(r)
