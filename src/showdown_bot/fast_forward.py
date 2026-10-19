"""Fast inference path of the token transformer for net-search (bead showdown-bot-589.68.7, docs/net-search.md).

``FastForward`` computes the same function as ``TokenNet.forward`` (policy and opponent-action heads, value) from the
engine's raw token buffers, with these exact (up to float rounding) changes:

* **history tokens dropped** when the history is padding (``hist`` not given): they are masked as attention keys, and
  their own outputs are never read, so removing them changes nothing;
* **masked-token compaction** (``compact``): illegal action tokens, empty matchup pairs and padded history rows are
  never attended to. Rows are sorted by their number of usable matchup/history tokens and processed in ``groups``
  groups, each truncated to its own longest row (rounded up to ``kstep``): the attention sequence shrinks from 94 tokens
  to about 45-60;
* **last-layer pruning** (``prune_last``): the heads read the CLS token and the 26 action tokens only, so the last
  block computes queries, the projection and the MLP for those 27 tokens only (keys and values still see all);
* **fp16 weights** cast once (matmuls in fp16, residual stream and LayerNorm in fp32: the same arithmetic autocast does);
* **CUDA graphs** (``graphs``): one graph per (rows, group length) capturing the whole forward, so the ~250 kernel
  launches of the eager model cost one launch (works without triton / ``torch.compile``);
* **pinned host buffers** and non-blocking copies: the engine's bytes are copied into a ring of pinned buffers and
  uploaded asynchronously, the result comes back through a pinned buffer and an event.

``__call__(cat_bytes, cont_bytes, n)`` returns ``(handle)``; ``result(handle)`` waits and returns the
``[rows, 27]`` float32 array ``[policy(13), opponent policy(13), value]`` (softmax / sigmoid already applied).
"""

from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn.functional as F

from showdown_bot import token_net, tokens
from showdown_bot.token_net import N_ACT, N_HIST, N_MON, N_PAIR, N_SIDE, NEG, T_ACT, T_GLOB, T_HIST, T_MON, T_PAIR, T_SIDE

N_HEAD = 1 + 2 * N_ACT  # CLS + own + opponent action tokens: the only outputs the heads read
OUT_W = 2 * N_ACT + 1


def _bucket(n: int, step: int, hi: int) -> int:
    return min(hi, ((max(n, 1) + step - 1) // step) * step)


class _W:
    """fp16 copies of a Linear's parameters."""

    def __init__(self, lin: torch.nn.Linear, dtype: torch.dtype) -> None:
        self.w = lin.weight.detach().to(dtype).contiguous()
        self.b = None if lin.bias is None else lin.bias.detach().to(dtype).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.w.dtype), self.w, self.b)


class FastForward:
    def __init__(
        self,
        net: token_net.TokenNet,
        device: str | torch.device,
        *,
        compact: bool = True,
        prune_last: bool = True,
        graphs: bool = True,
        groups: int = 2,
        kstep: int = 8,
        row_step: int = 128,
        use_hist: bool = False,
        half: bool = True,
        half_res: bool = False,
        pinned: bool = True,
        warm_rows: tuple[int, ...] = (),
    ) -> None:
        self.net = net.eval()
        self.dev = torch.device(device)
        self.cuda = self.dev.type == "cuda"
        self.compact = compact
        self.prune_last = prune_last
        self.graphs = graphs and self.cuda
        self.groups = max(1, groups) if compact else 1
        self.kstep = kstep
        self.row_step = row_step
        self.use_hist = use_hist
        self.pinned = pinned and self.cuda
        self.dt = torch.float16 if (half and self.cuda) else torch.float32
        self.rt = self.dt if half_res else torch.float32  # dtype of the residual stream
        lay = tokens.layout()
        self.fields = {k: v for k, v in lay.items() if k in token_net.INPUT_FIELDS}
        self.cat_len = max(f["offset"] + int(np.prod(f["shape"])) for f in lay.values() if f["buffer"] == "cat")
        self.cont_len = max(f["offset"] + int(np.prod(f["shape"])) for f in lay.values() if f["buffer"] == "cont")
        self.off_leg = self.fields["act_legal"]["offset"]
        self.off_pair = self.fields["match_pair"]["offset"]
        cfg = net.cfg
        self.cfg = cfg
        self.d = cfg.d_model
        dt = self.dt
        n = self.net
        self.lin = {
            name: _W(getattr(n, name), dt)
            for name in ("l_mon", "l_known", "l_side", "l_glob", "l_pair", "l_hist", "l_act")
        }
        self.blocks = [
            {"qkv": _W(b.qkv, dt), "proj": _W(b.proj, dt), "fc1": _W(b.fc1, dt), "fc2": _W(b.fc2, dt),
             "n1": self._ln(b.n1), "n2": self._ln(b.n2), "heads": b.heads}
            for b in n.blocks
        ]
        self.n_in = self._ln(n.n_in)
        self.n_out = self._ln(n.n_out)
        self.heads = {name: [_W(m, dt) for m in seq if isinstance(m, torch.nn.Linear)]
                      for name, seq in (("pol", n.h_pol), ("val", n.h_val), ("opp", n.h_opp_slot))}
        # token order inside the model: [CLS, own actions, opp actions | glob, side x2, mon x12, pair x36, (hist x16)]
        head_pos = [0] + list(range(T_ACT, T_ACT + 2 * N_ACT))
        rem_pos = [T_GLOB] + list(range(T_SIDE, T_SIDE + N_SIDE)) + list(range(T_MON, T_MON + N_MON)) \
            + list(range(T_PAIR, T_PAIR + N_PAIR))
        if use_hist:
            rem_pos += list(range(T_HIST, T_HIST + N_HIST))
        pos = n.e_pos.weight.detach()
        self.pos_head = pos[head_pos].to(self.dev)
        self.pos_rem = pos[rem_pos].to(self.dev)
        self.n_rem = len(rem_pos)
        self.n_fixed = 1 + N_SIDE + N_MON  # glob + side + mon tokens are always usable
        self.ar_mon = torch.arange(N_MON, device=self.dev)
        self.ar_pair = torch.arange(N_PAIR, device=self.dev)
        self.side_ids = torch.tensor([1, 2], device=self.dev)
        self._static: dict[int, dict] = {}
        self._graph: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self._pool = torch.cuda.graph_pool_handle() if self.graphs else None
        self._ring: list[dict] = []
        self._lock = threading.RLock()  # one launch at a time: the static buffers are shared by all launches
        self._ring_i = 0
        self.captures = 0
        self.launches = 0
        if self.graphs and warm_rows:
            for r in warm_rows:
                self.warm(r)

    def _ln(self, ln: torch.nn.LayerNorm):
        w, b, eps, d = ln.weight.detach().to(self.rt), ln.bias.detach().to(self.rt), ln.eps, ln.normalized_shape
        return lambda x: F.layer_norm(x.to(self.rt), d, w, b, eps)

    # ------------------------------------------------------------------ buffers
    def _bufs(self, rows: int) -> dict:
        s = self._static.get(rows)
        if s is None:
            dev = self.dev
            s = {
                "cat": torch.zeros(rows, self.cat_len, dtype=torch.int16, device=dev),
                "cont": torch.zeros(rows, self.cont_len, dtype=torch.float32, device=dev),
                "order": torch.arange(rows, device=dev),
                "out": torch.zeros(rows, OUT_W, dtype=torch.float32, device=dev),
            }
            if self.use_hist:
                s["hist_cat"] = torch.zeros(rows, N_HIST, 4, dtype=torch.int16, device=dev)
                s["hist_cont"] = torch.zeros(rows, N_HIST, self.cfg.c_hist, dtype=torch.float32, device=dev)
            self._static[rows] = s
        return s

    def _pin(self, rows: int) -> dict:
        """A pinned host staging set for ``rows`` rows: one that is not in flight (a small ring)."""
        for r in self._ring:
            if r["rows"] == rows and not r["busy"]:
                return r
        r = {
            "rows": rows, "busy": False,
            "cat": torch.zeros(rows, self.cat_len, dtype=torch.int16, pin_memory=self.pinned),
            "cont": torch.zeros(rows, self.cont_len, dtype=torch.float32, pin_memory=self.pinned),
            "order": torch.zeros(rows, dtype=torch.int64, pin_memory=self.pinned),
            "out": torch.zeros(rows, OUT_W, dtype=torch.float32, pin_memory=self.pinned),
            "free": torch.cuda.Event(),
        }
        if self.use_hist:
            r["hist_cat"] = torch.zeros(rows, N_HIST, 4, dtype=torch.int16, pin_memory=self.pinned)
            r["hist_cont"] = torch.zeros(rows, N_HIST, self.cfg.c_hist, dtype=torch.float32, pin_memory=self.pinned)
        r["free"].record()
        self._ring.append(r)
        return r

    # ------------------------------------------------------------------ the model
    def _field(self, cat: torch.Tensor, cont: torch.Tensor, name: str) -> torch.Tensor:
        fd = self.fields[name]
        src = cat if fd["buffer"] == "cat" else cont
        size = int(np.prod(fd["shape"]))
        return src[:, fd["offset"]: fd["offset"] + size].reshape(src.shape[0], *fd["shape"])

    def _forward(self, cat: torch.Tensor, cont: torch.Tensor, order: torch.Tensor, k_rem: int,
                 hist_cat: torch.Tensor | None, hist_cont: torch.Tensor | None) -> torch.Tensor:
        """Rows ``order`` of the (static) batch -> ``[len(order), OUT_W]``."""
        n = self.net
        d = self.d
        cat = cat.index_select(0, order)
        cont = cont.index_select(0, order)
        b = cat.shape[0]
        mc = self._field(cat, cont, "mon_cat").long()
        f32 = torch.float32
        mon = (
            n.e_species(mc[..., 0]) + n.e_item(mc[..., 1]) + n.e_ability(mc[..., 2])
            + (n.e_move(mc[..., 3:7]) + n.e_movepos).sum(-2)
            + self.lin["l_mon"](self._field(cat, cont, "mon_cont")) + self.lin["l_known"](self._field(cat, cont, "mon_known").to(f32))
            + n.e_slot(self.ar_mon)
        )
        side = self.lin["l_side"](self._field(cat, cont, "side_cont"))
        glob = self.lin["l_glob"](torch.cat([self._field(cat, cont, "glob"), self._field(cat, cont, "match_agg")], -1)).unsqueeze(1)
        pair_x = self._field(cat, cont, "match_pair")
        pair = self.lin["l_pair"](pair_x) + n.e_slot(self.ar_pair // 6) + n.e_slot(6 + self.ar_pair % 6)
        pair_ok = (pair_x != 0).any(-1)
        ac = self._field(cat, cont, "act_cat").long()
        act = n.e_move(ac[..., 0]) + n.e_species(ac[..., 1]) + self.lin["l_act"](self._field(cat, cont, "act_cont"))
        act = act + n.e_hside(self.side_ids).unsqueeze(1)
        act_ok = self._field(cat, cont, "act_legal") > 0
        parts = [glob.to(f32), side.to(f32), mon.to(f32), pair.to(f32)]
        oks = [torch.ones(b, 1 + N_SIDE + N_MON, dtype=torch.bool, device=cat.device), pair_ok]
        if self.use_hist:
            hc = hist_cat.index_select(0, order).long()
            hf = hist_cont.index_select(0, order)
            kind = hc[..., 1]
            hid = torch.where((kind == 3).unsqueeze(-1), n.e_species(hc[..., 2]),
                              n.e_move(hc[..., 2].clamp(max=self.cfg.n_move - 1)))
            hist = n.e_hside(hc[..., 0]) + n.e_hkind(kind) + hid + n.e_species(hc[..., 3]) + self.lin["l_hist"](hf)
            parts.append(hist.to(f32))
            oks.append(hc[..., 0] > 0)
        x_rem = torch.cat(parts, 1) + self.pos_rem
        ok_rem = torch.cat(oks, 1)
        if self.compact and k_rem < self.n_rem:
            perm = torch.sort((~ok_rem).to(torch.int8), dim=1, stable=True).indices[:, :k_rem]
            x_rem = x_rem.gather(1, perm.unsqueeze(-1).expand(-1, -1, d))
            ok_rem = ok_rem.gather(1, perm)
        cls = n.cls.expand(b, 1, -1)
        x_head = torch.cat([cls.to(f32), act.flatten(1, 2).to(f32)], 1) + self.pos_head
        ok_head = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=cat.device), act_ok.flatten(1, 2)], 1)
        x = self.n_in(torch.cat([x_head, x_rem], 1))
        mask = torch.cat([ok_head, ok_rem], 1)[:, None, None, :]
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            x = self._block(blk, x, mask, prune=self.prune_last and i == last)
        h = self.n_out(x[:, :N_HEAD])
        cls_h, own, opp = h[:, 0], h[:, 1: 1 + N_ACT], h[:, 1 + N_ACT:]
        legal = ok_head[:, 1:]
        pol = self._mlp(self.heads["pol"], own).squeeze(-1).float().masked_fill(~legal[:, :N_ACT], NEG)
        oppl = self._mlp(self.heads["opp"], opp).squeeze(-1).float().masked_fill(~legal[:, N_ACT:], NEG)
        val = self._mlp(self.heads["val"], cls_h).squeeze(-1).float()
        return torch.cat([torch.softmax(pol, -1), torch.softmax(oppl, -1), torch.sigmoid(val).unsqueeze(-1)], 1)

    @staticmethod
    def _mlp(layers: list[_W], x: torch.Tensor) -> torch.Tensor:
        return layers[1](F.gelu(layers[0](x)))

    def _block(self, blk: dict, x: torch.Tensor, mask: torch.Tensor, prune: bool) -> torch.Tensor:
        b, t, d = x.shape
        h = blk["heads"]
        qkv = blk["qkv"](blk["n1"](x)).view(b, t, 3, h, d // h)
        m = N_HEAD if prune else t
        q = qkv[:, :m, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        xo = (x[:, :m] if prune else x) + blk["proj"](a.transpose(1, 2).reshape(b, m, d))
        return xo + blk["fc2"](F.gelu(blk["fc1"](blk["n2"](xo))))

    # ------------------------------------------------------------------ graphs
    def _run_group(self, s: dict, rows: int, g: int, rg: int, k_rem: int) -> None:
        order = s["order"][g * rg:(g + 1) * rg]
        key = (rows, g, k_rem) if self.graphs else None

        def body() -> None:
            out = self._forward(s["cat"], s["cont"], order, k_rem, s.get("hist_cat"), s.get("hist_cont"))
            s["out"].index_copy_(0, order, out)

        if not self.graphs:
            with torch.no_grad():
                body()
            return
        gr = self._graph.get(key)
        if gr is None:
            with torch.no_grad():
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        body()
                torch.cuda.current_stream().wait_stream(stream)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, pool=self._pool):
                    body()
            self._graph[key] = gr
            self.captures += 1
        gr.replay()

    def warm_all(self, top_rows: int) -> None:
        """Captures the graphs of every row bucket up to ``top_rows`` rows."""
        if not self.graphs:
            return
        top = ((top_rows + self.row_step - 1) // self.row_step) * self.row_step
        for rows in range(self.row_step, top + 1, self.row_step):
            self.warm(((rows + self.groups - 1) // self.groups) * self.groups)

    def warm(self, rows: int) -> None:
        """Captures every (group, length) graph for ``rows`` rows (so no capture happens in a search)."""
        if not self.graphs:
            return
        s = self._bufs(rows)
        rg = rows // self.groups
        span = self.n_rem - self.n_fixed
        ks = [self.n_fixed + min(span, self.kstep * i) for i in range(1, (span + self.kstep - 1) // self.kstep + 1)]
        for g in range(self.groups):
            for k in ks:
                self._run_group(s, rows, g, rg, k)
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ public
    def counts(self, cont: np.ndarray, n_rows: int, hist_cat: np.ndarray | None) -> np.ndarray:
        """Usable non-head tokens per row (always-usable ones included), from the host buffer."""
        pr = (cont[:n_rows, self.off_pair: self.off_pair + N_PAIR * 5].reshape(n_rows, N_PAIR, 5) != 0).any(-1).sum(1)
        c = self.n_fixed + pr
        if self.use_hist and hist_cat is not None:
            c = c + (hist_cat[:n_rows, :, 0] > 0).sum(1)
        return c

    def launch(self, cat: bytes, cont: bytes, n_rows: int, hist_cat: np.ndarray | None = None,
               hist_cont: np.ndarray | None = None) -> tuple:
        """Queues the forward of ``n_rows`` rows; returns a handle for ``result``."""
        with self._lock:
            return self._launch(cat, cont, n_rows, hist_cat, hist_cont)

    def _launch(self, cat: bytes, cont: bytes, n_rows: int, hist_cat: np.ndarray | None,
                hist_cont: np.ndarray | None) -> tuple:
        self.launches += 1
        if self.use_hist and hist_cat is None:
            hist_cat = np.zeros((n_rows, N_HIST, 4), np.int16)
            hist_cont = np.zeros((n_rows, N_HIST, self.cfg.c_hist), np.float32)
        cat_np = np.frombuffer(cat, dtype="<i2").reshape(n_rows, self.cat_len)
        cont_np = np.frombuffer(cont, dtype="<f4").reshape(n_rows, self.cont_len)
        if not self.cuda:  # CPU: plain eager on the exact rows, no padding / grouping
            return ("cpu", self._cpu(cat_np, cont_np, hist_cat, hist_cont), n_rows)
        rows = _bucket(n_rows, self.row_step, 1 << 30)
        rows = ((rows + self.groups - 1) // self.groups) * self.groups
        s = self._bufs(rows)
        p = self._pin(rows)
        p["busy"] = True
        p["free"].synchronize()  # the previous upload from this staging set is done
        p["cat"][:n_rows].numpy()[...] = cat_np
        p["cont"][:n_rows].numpy()[...] = cont_np
        if self.use_hist and hist_cat is not None:
            p["hist_cat"][:n_rows].numpy()[...] = hist_cat
            p["hist_cont"][:n_rows].numpy()[...] = hist_cont
        cnt = np.zeros(rows, np.int64)
        cnt[:n_rows] = self.counts(cont_np, n_rows, hist_cat)
        order = np.argsort(cnt, kind="stable")
        p["order"].numpy()[...] = order
        s["cat"][:n_rows].copy_(p["cat"][:n_rows], non_blocking=True)
        s["cont"][:n_rows].copy_(p["cont"][:n_rows], non_blocking=True)
        if self.use_hist and hist_cat is not None:
            s["hist_cat"][:n_rows].copy_(p["hist_cat"][:n_rows], non_blocking=True)
            s["hist_cont"][:n_rows].copy_(p["hist_cont"][:n_rows], non_blocking=True)
        s["order"].copy_(p["order"], non_blocking=True)
        rg = rows // self.groups
        for g in range(self.groups):
            grp = cnt[order[g * rg:(g + 1) * rg]]
            k = _bucket(int(grp.max()) - self.n_fixed, self.kstep, self.n_rem - self.n_fixed) + self.n_fixed \
                if self.compact else self.n_rem
            self._run_group(s, rows, g, rg, min(k, self.n_rem))
        p["out"].copy_(s["out"], non_blocking=True)
        p["free"].record()
        return ("cuda", p, n_rows)

    def _cpu(self, cat_np: np.ndarray, cont_np: np.ndarray, hist_cat: np.ndarray | None, hist_cont: np.ndarray | None) -> np.ndarray:
        rows = cat_np.shape[0]
        cat = torch.from_numpy(cat_np.copy())
        cont = torch.from_numpy(cont_np.copy())
        s = {"cat": cat, "cont": cont, "order": torch.arange(rows)}
        if self.use_hist and hist_cat is not None:
            s["hist_cat"], s["hist_cont"] = torch.from_numpy(hist_cat), torch.from_numpy(hist_cont)
        cnt = self.counts(cont_np, rows, hist_cat)
        order = torch.from_numpy(np.argsort(cnt, kind="stable").copy())
        out = torch.zeros(rows, OUT_W)
        rg = (rows + self.groups - 1) // self.groups
        with torch.no_grad():
            for g in range(self.groups):
                idx = order[g * rg:(g + 1) * rg]
                if len(idx) == 0:
                    continue
                k = _bucket(int(cnt[idx.numpy()].max()) - self.n_fixed, self.kstep, self.n_rem - self.n_fixed) + self.n_fixed \
                    if self.compact else self.n_rem
                o = self._forward(cat, cont, idx, min(k, self.n_rem), s.get("hist_cat"), s.get("hist_cont"))
                out.index_copy_(0, idx, o)
        return out.numpy()

    def result(self, handle: tuple) -> np.ndarray:
        kind, p, n_rows = handle
        if kind == "cpu":
            return p[:n_rows]
        p["free"].synchronize()
        out = p["out"][:n_rows].numpy().copy()
        with self._lock:
            p["busy"] = False
        return out
