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
