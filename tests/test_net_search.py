"""Net-guided search (bead showdown-bot-589.68.5): spec parsing, the slot mapping of both sides, a smoke
search with a dummy evaluator, the real transformer on a real state, and a smoke game vs random when a local
Showdown server is up. The Rust unit tests are in engine/src/netsearch.rs."""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not hasattr(pe, "NetSearch"), reason="engine wheel without NetSearch (sdb.24)")

from showdown_bot import league, net_search as nsr, token_policy, tokens  # noqa: E402
from showdown_bot.token_net import TokenNet, TokenNetConfig  # noqa: E402

STATES = Path(__file__).resolve().parents[1] / "engine" / "data" / "gen9randombattle.txt"
MODEL = "models/tok-d256.pt"
N = 13


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
    return [line.strip() for line in STATES.read_text().splitlines() if line.strip()][:8]


@pytest.fixture(scope="module")
def tiny_net(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    cfg = TokenNetConfig.from_spec(tokens.token_spec(), d_model=32, n_layers=2, n_heads=4)
    path = tmp_path_factory.mktemp("tok") / "tiny.pt"
    TokenNet(cfg).export(path)
    return path


def f32(a) -> bytes:
    return np.ascontiguousarray(a, dtype="<f4").tobytes()


def slot_policy(names: list[str], hot: str) -> np.ndarray:
    p = np.zeros(N, np.float32)
    p[names.index(hot)] = 1.0
    return p


# ------------------------------------------------------------------ spec


def test_spec_parses():
    s = league.parse_spec(
        r"ns=net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:max_evals=3000:batch=64"
        r":c_puct=2:depth=4:fpu=0.1:vavg=1:scrub=0:pipe=2:views=1:mode=mixed")
    assert (s.label, s.type) == ("ns", "net-search")
    assert s.param("model") == r"models\tok-d256.pt" and s.param("device") == "cuda"
    assert (s.param("worlds"), s.param("time_ms"), s.param("max_evals"), s.param("batch")) == (32, 1000, 3000, 64)
    assert (s.param("c_puct"), s.param("depth"), s.param("fpu"), s.param("vavg"), s.param("scrub")) == (2, 4, 0.1, 1, 0)
    assert s.param("mode") == "mixed"
    league.parse_spec("net-search:model=m.pt")


@pytest.mark.parametrize("bad", [
    "net-search",                                       # model is required
    "net-search:device=cuda",
    "net-search:model=m.pt:device=tpu",
    "net-search:model=m.pt:worlds=0",
    "net-search:model=m.pt:depth=0",
    "net-search:model=m.pt:batch=0",
    "net-search:model=m.pt:vavg=2",
    "net-search:model=m.pt:pipe=0",
    "net-search:model=m.pt:views=3",
    "net-search:model=m.pt:mode=nash",
    "net-search:model=m.pt:c_puct=0",
    "net-search:model=m.pt:evaluator=x.pev",            # search keys are not net-search keys
    "net-search:model=m.pt:time_ms=500:time_ms=900",    # duplicate key
])
def test_spec_rejects(bad):
    with pytest.raises(ValueError):
        league.parse_spec(bad)


def test_documented_sprt_specs_parse():
    league.parse_spec(r"ns=net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:depth=4:c_puct=2")
    league.parse_spec(r"tokpol=token-policy:model=models\tok-d256.pt:device=cuda")
    league.parse_spec(r"best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000"
                      r":worlds=4:evaluator=models\gen7-res-root-mlp16.pev")


# ------------------------------------------------------------------ slot mapping and the engine class


def test_slot_names_of_both_sides(states):
    """The 13 own-slot names of side one's and side two's token view are the option names NetSearch reports."""
    for s in states[:4]:
        own1 = nsr.slot_names(pe.state_tokens_str(s, 1, None, True)[2])
        own2 = nsr.slot_names(pe.state_tokens_str(s, 2, None, True)[2])
        ns = pe.NetSearch([s])
        ns.collect(1)
        (rows1, rows2, _v, _n), = ns.result()
        # a side that waits (opponent's forced switch) has the single option "No Move" and no action slot
        assert {r[0] for r in rows1} - {"No Move"} == {c for c in own1 if c}
        assert {r[0] for r in rows2} - {"No Move"} == {c for c in own2 if c}
        # the token legal flags of each view agree with those names
        n, cat, _cont = ns.collect(1)  # root pending -> nothing new
        assert n == 0


def test_priors_follow_each_sides_own_policy(states):
    s = states[0]
    own1 = nsr.slot_names(pe.state_tokens_str(s, 1, None, True)[2])
    own2 = nsr.slot_names(pe.state_tokens_str(s, 2, None, True)[2])
    hot1 = [c for c in own1 if c][-1]
    hot2 = [c for c in own2 if c][-1]
    ns = pe.NetSearch([s], eps=0.01)
    n, cat, cont = ns.collect(4)
    assert n == 1
    ns.feed(f32(slot_policy(own1, hot1)[None]), f32(slot_policy(own2, hot2)[None]), f32([0.5]))
    (rows1, rows2, _v, _n), = ns.result()
    assert max(rows1, key=lambda r: r[3])[0] == hot1 and max(rows1, key=lambda r: r[3])[3] > 0.9
    assert max(rows2, key=lambda r: r[3])[0] == hot2 and max(rows2, key=lambda r: r[3])[3] > 0.9


def test_feed_validates_shapes(states):
    ns = pe.NetSearch([states[0]])
    n, _c, _f = ns.collect(4)
    with pytest.raises(ValueError):
        ns.feed(f32(np.ones((n, N))), f32(np.ones((n, N))), f32(np.ones(n + 1) * 0.5))
    with pytest.raises(ValueError):
        pe.NetSearch([])
    with pytest.raises(ValueError):
        pe.NetSearch([states[0]], depth=0)


def dummy_eval(cat: bytes, cont: bytes, n: int):
    """Uniform policies, value 0.5 plus a deterministic wobble: enough to drive the trees."""
    rng = np.random.default_rng(len(cat) + n)
    return (f32(np.ones((n, N), np.float32)), f32(np.ones((n, N), np.float32)),
            f32(0.5 + 0.1 * rng.standard_normal(n)))


def test_smoke_search_with_dummy_evaluator(states):
    legal = {c for c, _s in pe.policy_actions(pe.State.from_string(states[0]), 1)}
    ns = pe.NetSearch([states[0]] * 4, depth=3, seed=3)
    info = nsr.drive(ns, dummy_eval, time_ms=5000, max_evals=600, batch=32)
    assert info.evals == 600 and info.batches >= 600 // 32
    st = ns.stats()
    assert st["max_depth"] >= 2 and st["evals"] == 600
    assert ns.pending() == 0
    policy, extra = nsr.aggregate(ns.result())
    assert set(policy) <= legal
    assert abs(sum(policy.values()) - 1.0) < 1e-6
    assert extra["visits"] > 600  # visits count the revisits of known leaves too
    # every root option got an explicit visit statistic
    for rows1, _rows2, v, nv in ns.result():
        assert nv > 0 and 0.0 <= v <= 1.0
        assert sum(r[1] for r in rows1) == nv - 1


def test_stats_keys(states):
    ns = pe.NetSearch([states[0]])
    assert {"evals", "batches", "collisions", "terminal", "leaf_reuse", "max_depth", "nodes", "tree_ms",
            "tokenise_ms", "feed_ms"} <= set(ns.stats())


# ------------------------------------------------------------------ the transformer


def test_tiny_net_evaluator_shapes(tiny_net, states):
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    searcher = nsr.NetSearcher(pol, worlds=2, time_ms=2000, max_evals=60, batch=16)
    policy, extra = searcher.search(states[:2])
    legal = {c for s in states[:2] for c, _x in pe.policy_actions(pe.State.from_string(s), 1)}
    assert policy and set(policy) <= legal and extra["drive"].evals == 60


def test_evaluator_matches_predict_for_both_views(tiny_net, states):
    """NetEvaluator's sliced device batch gives the same numbers as TokenPolicy's per-state path."""
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    ev = nsr.NetEvaluator(pol.net, "cpu")
    s = states[2]
    ns = pe.NetSearch([s])
    n, cat, cont = ns.collect(1)
    p1, p2, v = ev.evaluate(cat, cont, n)
    p1 = np.frombuffer(p1, "<f4").reshape(n, N)
    p2 = np.frombuffer(p2, "<f4").reshape(n, N)
    for side, got in ((1, p1), (2, p2)):
        names, want, legal = pol.predict_states([s], None) if side == 1 else _predict_side2(pol, s)
        assert np.allclose(got[0], want[0], atol=2e-2), (side, got[0], want[0])
    assert 0.0 <= np.frombuffer(v, "<f4")[0] <= 1.0


def _predict_side2(pol, s):
    cat, cont, ch = pe.state_tokens_str(s, 2, None, True)
    from showdown_bot import token_net

    b = token_net.batch_from_engine([tokens.split_buffers(cat, cont)])
    out = pol.net.predict(b, device="cpu")
    return [list(ch[:N])], out["policy"], b["act_legal"][:, 0] > 0


@pytest.mark.skipif(real_model() is None, reason="models/tok-d256.pt not available")
def test_real_net_on_a_real_state(states):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pol = token_policy.TokenPolicy.load(real_model(), dev)
    s = states[3]
    legal = {c for c, _x in pe.policy_actions(pe.State.from_string(s), 1)}
    searcher = nsr.NetSearcher(pol, worlds=4, time_ms=4000, max_evals=300, batch=64, depth=3)
    policy, extra = searcher.search([s] * 4)
    assert set(policy) <= legal and abs(sum(policy.values()) - 1) < 1e-6
    raw = pol.distribution([s], None)
    # the search stays in the neighbourhood of the raw net: it does not pick an option the net finds absurd
    best = max(policy, key=policy.get)
    assert raw.get(best, 0) > 0.02, (best, raw)
    assert 0 < extra["root_value"] < 1


# ------------------------------------------------------------------ 589.68.7: fast forward, history, scrub2, threads

has25 = hasattr(getattr(pe, "NetSearch", None), "take_hist")
needs25 = pytest.mark.skipif(not has25, reason="engine wheel without sdb.25 (history / reveal2 / threads)")

OLD_SPEC = r"net-search:model=models	ok-d256.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4"
NEW_SPEC = OLD_SPEC + ":pipe=2:fast=1:hist=2:scrub2=1:tokthreads=2"


def test_new_keys_parse_and_old_spec_is_unchanged():
    old, new = league.parse_spec("old=" + OLD_SPEC), league.parse_spec("new=" + NEW_SPEC)
    assert [k for k, _ in old.params if k in ("fast", "hist", "scrub2", "tokthreads", "fp16res")] == []
    assert (new.param("fast"), new.param("hist"), new.param("scrub2"), new.param("tokthreads")) == (1, 2, 1, 2)
    league.parse_spec("tp=token-policy:model=m.pt:hist=1")


@pytest.mark.parametrize("bad", [
    "net-search:model=m.pt:hist=3", "net-search:model=m.pt:fast=2", "net-search:model=m.pt:scrub2=2",
    "net-search:model=m.pt:tokthreads=0", "net-search:model=m.pt:tokthreads=5", "net-search:model=m.pt:fp16res=3",
    "token-policy:model=m.pt:hist=2", "token-policy:model=m.pt:fast=1", "search-net:evaluator=x.pev:hist=1",
])
def test_new_keys_reject(bad):
    with pytest.raises(ValueError):
        league.parse_spec(bad)


@needs25
def test_engine_rejects_bad_new_arguments(states):
    with pytest.raises(ValueError):
        pe.NetSearch([states[0]], hist=3)
    with pytest.raises(ValueError):
        pe.NetSearch([states[0]], threads=9)
    with pytest.raises(ValueError):
        pe.NetSearch([states[0]], hist=1, hist_cat=b"\x00" * 5, hist_cont=b"\x00" * 7)


@needs25
def test_history_rows_reach_the_evaluator(tiny_net, states):
    """hist=1 gives every leaf the root rows (side two mirrored); hist=2 pushes the simulated turns."""
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    hc = np.zeros((16, 4), np.int16)
    hf = np.zeros((16, 5), np.float32)
    hc[0] = (1, 1, 77, 5)
    hc[1] = (2, 3, 66, 6)
    hf[0] = (0.0625, -0.1, 0.0, 0.0, 0.0)
    hf[1] = (0.0625, 0.0, -0.1, 0.0, 0.0)
    for mode in (1, 2):
        seen = []
        searcher = nsr.NetSearcher(pol, worlds=2, time_ms=5000, max_evals=80, batch=16, depth=3, hist=mode)
        orig = searcher.evaluator.evaluate

        def spy(cat, cont, n, hist=None, _o=orig, _s=seen):
            _s.append(None if hist is None else (np.frombuffer(hist[0], "<i2").reshape(n, 2, 16, 4).copy(),
                                                 np.frombuffer(hist[1], "<f4").reshape(n, 2, 16, 5).copy()))
            return _o(cat, cont, n, hist)

        searcher.evaluator.evaluate = spy
        policy, extra = searcher.search(states[:2], hist=(hc, hf))
        assert policy and extra["drive"].evals == 80
        assert all(h is not None for h in seen)
        first_cat = seen[0][0]  # the roots
        assert (first_cat[:, 0, 0] == (1, 1, 77, 5)).all() and (first_cat[:, 0, 1] == (2, 3, 66, 6)).all()
        # side two sees the turn mirrored: its own action (the opponent's row) first, with the side ids flipped
        assert (first_cat[:, 1, 0] == (1, 3, 66, 6)).all() and (first_cat[:, 1, 1] == (2, 1, 77, 5)).all()
        deeper = np.concatenate([h[0] for h in seen[1:]])
        if mode == 1:
            assert (deeper[:, 0, 0] == (1, 1, 77, 5)).all()
        else:
            assert (deeper[:, 0, 2] == (1, 1, 77, 5)).any(), "the root rows move down behind the simulated turn"
            assert (deeper[:, 0, 0, 0] == 1).all() and (deeper[:, 0, 1, 0] == 2).all()


@needs25
def test_scrub2_and_threads_change_only_what_they_should(states):
    s = states[0]
    base = pe.NetSearch([s])
    n, cat, cont = base.collect(1)
    scr = pe.NetSearch([s], reveal2=[])
    n2, cat2, cont2 = scr.collect(1)
    from showdown_bot import tokens as tk

    lay = tk.layout()
    meta = lay["meta"]["offset"]
    c1 = np.frombuffer(cat, "<i2").reshape(2, -1)
    c2 = np.frombuffer(cat2, "<i2").reshape(2, -1)
    assert (c1[0] == c2[0]).all()  # side one's view is untouched
    assert c1[1][meta + 2] == 0 and c2[1][meta + 2] == 1  # side two's view of our team is scrubbed
    t1 = pe.NetSearch([s] * 8, seed=4)
    t3 = pe.NetSearch([s] * 8, seed=4, threads=3)
    assert t1.collect(8)[1:] == t3.collect(8)[1:]


@needs25
def test_search_with_everything_on_tiny_net(tiny_net, states):
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    searcher = nsr.NetSearcher(pol, worlds=4, time_ms=5000, max_evals=120, batch=16, depth=3, fast=True, hist=2,
                               scrub2=True, tokthreads=2, pipe=2)
    hist = (np.zeros((16, 4), np.int16), np.zeros((16, 5), np.float32))
    policy, extra = searcher.search(states[:4], hist=hist, reveal2=[("x", [], False, False, False)])
    legal = {c for s in states[:4] for c, _x in pe.policy_actions(pe.State.from_string(s), 1)}
    assert policy and set(policy) <= legal and extra["drive"].evals == 120
    assert abs(sum(policy.values()) - 1) < 1e-6


def test_fast_evaluator_equals_eager_on_a_tiny_net(tiny_net, states):
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    slow = nsr.NetEvaluator(pol.net, "cpu")
    fast = nsr.NetEvaluator(pol.net, "cpu", fast=True)
    cats, conts = [], []
    for s in states[:6]:
        for side in (1, 2):
            c, f, _ = pe.state_tokens_str(s, side, None, False)
            cats.append(np.frombuffer(c, "<i2"))
            conts.append(np.frombuffer(f, "<f4"))
    cat, cont = np.stack(cats).tobytes(), np.stack(conts).tobytes()
    a = [np.frombuffer(x, "<f4") for x in slow.evaluate(cat, cont, 6)]
    b = [np.frombuffer(x, "<f4") for x in fast.evaluate(cat, cont, 6)]
    for x, y in zip(a, b):
        assert np.abs(x - y).max() < 3e-2  # bf16 autocast on the CPU against float32


# ------------------------------------------------------------------ smoke game


def server_up(port: int = 8000) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not server_up(), reason="no local Showdown server on :8000")
def test_net_search_plays_a_game_vs_random(tiny_net):
    from poke_env import AccountConfiguration, RandomPlayer

    from showdown_bot import bench
    from showdown_bot.players.netsearch import NetSearchPlayer

    path = real_model() or tiny_net
    dev = "cuda" if (real_model() and torch.cuda.is_available()) else "cpu"
    server = bench.local_server(8000)
    bot = NetSearchPlayer(
        model=str(path), device=dev, worlds=4, time_ms=300, max_evals=200, batch=32, depth=3,
        battle_format="gen9randombattle", server_configuration=server,
        account_configuration=AccountConfiguration("ns" + secrets.token_hex(4), None), seed=1, log_level=40,
    )
    rnd = RandomPlayer(
        battle_format="gen9randombattle", server_configuration=server,
        account_configuration=AccountConfiguration("rp" + secrets.token_hex(4), None), log_level=40,
    )

    async def play():
        await asyncio.wait_for(bot.battle_against(rnd, n_battles=1), timeout=600)

    asyncio.run(play())
    assert bot.n_finished_battles == 1
    assert bot.net_searches > 0 and bot.net_errors == 0
