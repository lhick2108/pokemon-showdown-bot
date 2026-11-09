"""FastForward (bead showdown-bot-589.68.7) computes the same function as ``TokenNet.forward``: tiny random nets on the
CPU (exact up to float rounding, with padded and with real history rows), the real net with CUDA graphs against the
eager autocast model, and the evaluator wiring (``fast=1`` against ``fast=0`` through ``NetSearcher``)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from showdown_bot import net_search as nsr, token_net, tokens  # noqa: E402
from showdown_bot.fast_forward import FastForward  # noqa: E402
from showdown_bot.token_net import TokenNet, TokenNetConfig  # noqa: E402
from showdown_bot.token_policy import TokenPolicy  # noqa: E402

STATES = Path(__file__).resolve().parents[1] / "engine" / "data" / "gen9randombattle.txt"
MODEL = "models/tok-d256.pt"


def real_model() -> Path | None:
    env = os.environ.get("TOK_NET_PATH")
    if env and Path(env).is_file():
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / MODEL).is_file():
            return parent / MODEL
    return None


@pytest.fixture(scope="module")
def states() -> list[str]:
    return [line.strip() for line in STATES.read_text().splitlines() if line.strip()][:12]


@pytest.fixture(scope="module")
def tiny() -> TokenNet:
    torch.manual_seed(0)
    cfg = TokenNetConfig.from_spec(tokens.token_spec(), d_model=32, n_layers=3, n_heads=4)
    net = TokenNet(cfg)
    with torch.no_grad():  # non-trivial heads
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.15)
    return net.eval()


def raw_buffers(states: list[str], reveals: list) -> tuple[bytes, bytes, int]:
    """Engine-layout byte buffers [n, CAT_LEN] / [n, CONT_LEN] for (state, reveal) pairs, side one's view."""
    cats, conts = [], []
    for s, rv in zip(states, reveals):
        cat, cont, _ = pe.state_tokens_str(s, 1, rv, False)
        cats.append(np.frombuffer(cat, "<i2"))
        conts.append(np.frombuffer(cont, "<f4"))
    return np.stack(cats).tobytes(), np.stack(conts).tobytes(), len(states)


def reference(net: TokenNet, cat: bytes, cont: bytes, n: int, hist=None) -> np.ndarray:
    """``[n, 27]``: policy probs, opponent probs, value, from the eager model in float32."""
    lay = tokens.layout()
    c = np.frombuffer(cat, "<i2").reshape(n, -1)
    f = np.frombuffer(cont, "<f4").reshape(n, -1)
    b = {}
    for name, fd in lay.items():
        if name not in token_net.INPUT_FIELDS:
            continue
        src = c if fd["buffer"] == "cat" else f
        size = int(np.prod(fd["shape"]))
        b[name] = torch.from_numpy(src[:, fd["offset"]: fd["offset"] + size].reshape(n, *fd["shape"]).copy())
    if hist is None:
        b["hist_cat"] = torch.zeros(n, token_net.N_HIST, 4, dtype=torch.long)
        b["hist_cont"] = torch.zeros(n, token_net.N_HIST, 5)
    else:
        b["hist_cat"] = torch.from_numpy(hist[0].astype(np.int64))
        b["hist_cont"] = torch.from_numpy(hist[1].astype(np.float32))
    with torch.no_grad():
        o = net(b)
    return torch.cat([torch.softmax(o["policy"], -1), torch.softmax(o["opp_slot"], -1),
                      torch.sigmoid(o["value"]).unsqueeze(-1)], 1).numpy()


def fake_history(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    hc = np.zeros((n, 16, 4), np.int16)
    hf = np.zeros((n, 16, 5), np.float32)
    for i in range(n):
        k = int(rng.integers(0, 17))  # leading rows real, the rest padding
        for r in range(k):
            side = 1 + r % 2
            kind = int(rng.integers(0, 4))
            hc[i, r] = (side, kind, int(rng.integers(2, 800)), int(rng.integers(2, 1400)))
            hf[i, r] = (rng.random(), rng.normal() * 0.3, rng.normal() * 0.3, rng.integers(0, 3) / 6, rng.integers(0, 3) / 6)
    return hc, hf


@pytest.mark.parametrize("compact,prune,groups,half_res", [(False, False, 1, False), (True, False, 1, False),
                                                           (True, True, 2, False), (True, True, 3, True)])
def test_cpu_equivalent_to_eager_padded_history(tiny, states, compact, prune, groups, half_res):
    n = len(states)
    reveals = [None if i % 2 else [] for i in range(n)]  # scrubbed rows mask the matchup pairs of the unseen bench
    cat, cont, _ = raw_buffers(states, reveals)
    ff = FastForward(tiny, "cpu", compact=compact, prune_last=prune, graphs=False, groups=groups, half=False,
                     half_res=half_res)
    out = ff.result(ff.launch(cat, cont, n))
    ref = reference(tiny, cat, cont, n)
    assert out.shape == (n, 27)
    assert np.abs(out - ref).max() < 1e-4, np.abs(out - ref).max()
    # the masked-token compaction really shortened the sequence
    if compact:
        assert ff.counts(np.frombuffer(cont, "<f4").reshape(n, -1), n, None).min() < ff.n_rem


def test_cpu_equivalent_with_real_history(tiny, states):
    rng = np.random.default_rng(1)
    n = len(states)
    cat, cont, _ = raw_buffers(states, [[] if i % 3 else None for i in range(n)])
    hist = fake_history(n, rng)
    ff = FastForward(tiny, "cpu", use_hist=True, compact=True, prune_last=True, graphs=False, groups=2, half=False)
    out = ff.result(ff.launch(cat, cont, n, hist[0], hist[1]))
    ref = reference(tiny, cat, cont, n, hist)
    assert np.abs(out - ref).max() < 1e-4
    # history changes the output (the rows are used), and padding rows are ignored
    pad = reference(tiny, cat, cont, n)
    assert np.abs(ref - pad).max() > 1e-3


def test_row_order_and_batch_size_do_not_matter(tiny, states):
    n = len(states)
    cat, cont, _ = raw_buffers(states, [None] * n)
    ff = FastForward(tiny, "cpu", compact=True, prune_last=True, graphs=False, groups=2, half=False)
    full = ff.result(ff.launch(cat, cont, n))
    c = np.frombuffer(cat, "<i2").reshape(n, -1)
    f = np.frombuffer(cont, "<f4").reshape(n, -1)
    perm = np.random.default_rng(2).permutation(n)
    shuffled = ff.result(ff.launch(c[perm].tobytes(), f[perm].tobytes(), n))
    assert np.abs(shuffled - full[perm]).max() < 1e-5
    part = ff.result(ff.launch(c[:5].tobytes(), f[:5].tobytes(), 5))
    assert np.abs(part - full[:5]).max() < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_graphs_match_eager_autocast(tiny, states):
    import copy

    net = copy.deepcopy(tiny).to("cuda")
    n = len(states) * 2
    ss = states * 2
    cat, cont, _ = raw_buffers(ss, [None if i % 2 else [] for i in range(n)])
    ff = FastForward(net, "cuda", compact=True, prune_last=True, graphs=True, groups=2)
    out = ff.result(ff.launch(cat, cont, n))
    out2 = ff.result(ff.launch(cat, cont, n))  # a replay of the captured graphs
    assert np.abs(out - out2).max() == 0
    ref = reference(tiny, cat, cont, n)
    assert np.abs(out - ref).max() < 2e-2  # fp16 matmuls against the fp32 reference


@pytest.mark.skipif(not torch.cuda.is_available() or real_model() is None, reason="needs CUDA and the real net")
def test_real_net_fast_vs_eager_through_the_evaluator(states):
    tp = TokenPolicy.load(real_model(), "cuda")
    slow = nsr.NetEvaluator(tp.net, tp.device, views=2)
    fast = nsr.NetEvaluator(tp.net, tp.device, views=2, fast=True, batch=8)
    sts = states[:8]
    # two views per leaf: side one's tokens then side two's
    cats, conts = [], []
    for s in sts:
        for side in (1, 2):
            cat, cont, _ = pe.state_tokens_str(s, side, None, False)
            cats.append(np.frombuffer(cat, "<i2"))
            conts.append(np.frombuffer(cont, "<f4"))
    cat, cont = np.stack(cats).tobytes(), np.stack(conts).tobytes()
    a = [np.frombuffer(x, "<f4") for x in slow.evaluate(cat, cont, len(sts))]
    b = [np.frombuffer(x, "<f4") for x in fast.evaluate(cat, cont, len(sts))]
    for x, y in zip(a, b):
        assert np.abs(x - y).max() < 5e-3, np.abs(x - y).max()
    assert all(int(np.argmax(x[i * 13:(i + 1) * 13])) == int(np.argmax(y[i * 13:(i + 1) * 13]))
               for x, y in zip(a[:2], b[:2]) for i in range(len(sts)) if np.sort(x[i * 13:(i + 1) * 13])[-1]
               - np.sort(x[i * 13:(i + 1) * 13])[-2] > 0.02)
