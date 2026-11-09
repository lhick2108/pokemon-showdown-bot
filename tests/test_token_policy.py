"""Token policy bot and transformer root prior (bead showdown-bot-589.68 W2): spec parsing, legal-action masking,
prior vector shape / normalisation, the prior reaching the engine's MCTS, and a smoke game against a random bot
when a local Showdown server is up."""

from __future__ import annotations

import asyncio
import random
import secrets
import socket
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not hasattr(pe, "state_tokens_str"), reason="engine wheel without state_tokens")

from showdown_bot import league, token_policy, tokens  # noqa: E402
from showdown_bot.token_net import TokenNet, TokenNetConfig  # noqa: E402

STATES = Path(__file__).resolve().parents[1] / "engine" / "data" / "gen9randombattle.txt"
MODEL = "models/tok-d256.pt"


def real_model() -> Path | None:
    """models/tok-d256.pt of this checkout or of the main checkout (a worktree has no models/)."""
    env = __import__("os").environ.get("TOK_NET_PATH")
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
    """A small random net exported with the installed engine's token spec (no trained model needed)."""
    torch.manual_seed(0)
    cfg = TokenNetConfig.from_spec(tokens.token_spec(), d_model=32, n_layers=2, n_heads=4)
    path = tmp_path_factory.mktemp("tok") / "tiny.pt"
    TokenNet(cfg).export(path)
    return path


def legal_choices(state: str) -> set[str]:
    return {c for c, _slot in pe.policy_actions(pe.State.from_string(state), 1)}


# ------------------------------------------------------------------ spec parsing


def test_token_policy_spec_parses():
    s = league.parse_spec(r"tok=token-policy:model=models\tok-d256.pt:device=cuda")
    assert (s.label, s.type) == ("tok", "token-policy")
    assert s.param("model") == r"models\tok-d256.pt" and s.param("device") == "cuda"
    s = league.parse_spec("token-policy:model=m.pt:temp=0.5:device=CPU")
    assert s.param("temp") == 0.5 and s.param("device") == "cpu"
    s = league.parse_spec(r"t=token-policy:model=C:\Users\x\tok.pt:device=cuda:1")
    assert s.param("model") == r"C:\Users\x\tok.pt" and s.param("device") == "cuda:1"


@pytest.mark.parametrize("bad", [
    "token-policy",                                   # model is required
    "token-policy:device=cuda",
    "token-policy:model=m.pt:device=tpu",
    "token-policy:model=m.pt:temp=0",
    "token-policy:model=m.pt:evaluator=x.pev",        # search keys are not token-policy keys
])
def test_token_policy_spec_rejects(bad):
    with pytest.raises(ValueError):
        league.parse_spec(bad)


def test_search_net_tok_prior_spec():
    s = league.parse_spec(r"x=search-net:evaluator=e.pev:tok_prior=models\tok-d256.pt:tok_device=cpu:c_puct=2")
    assert s.param("tok_prior") == r"models\tok-d256.pt" and s.param("tok_device") == "cpu"
    # tok_prior stands alone (root-only PUCT) and with a .ppol
    league.parse_spec("x=search-net:evaluator=e.pev:tok_prior=t.pt")
    league.parse_spec("x=search-net:evaluator=e.pev:prior=p.ppol:tok_prior=t.pt")
    with pytest.raises(ValueError):
        league.parse_spec("x=search-net:evaluator=e.pev:tok_device=cpu")  # needs tok_prior
    with pytest.raises(ValueError):
        league.parse_spec("x=search-net:evaluator=e.pev:c_puct=2")        # still needs a prior of some kind
    with pytest.raises(ValueError):
        league.parse_spec("x=search-net:evaluator=e.pev:tok_prior=t.pt:tok_device=npu")
    cfg = league.search_config_fields(s)
    assert cfg["tok_prior"].endswith("tok-d256.pt") and cfg["tok_device"] == "cpu" and cfg["c_puct"] == 2


def test_documented_specs_parse():
    best = (r"search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:"
            r"evaluator=models\gen7-res-root-mlp16.pev")
    league.parse_spec(best)
    league.parse_spec(best.replace("search-net:", "cand=search-net:") + r":tok_prior=models\tok-d256.pt:tok_device=cuda")
    league.parse_spec(r"tokpol=token-policy:model=models\tok-d256.pt:device=cuda")


# ------------------------------------------------------------------ distribution / masking


def test_normalise_drops_unnamed_and_zero():
    d = token_policy.normalise(["a", "", "b", "c"], [2.0, 5.0, 0.0, 6.0])
    assert set(d) == {"a", "c"} and abs(sum(d.values()) - 1) < 1e-12 and abs(d["a"] - 0.25) < 1e-12
    assert token_policy.normalise(["a"], [0.0]) == {}


def test_pick_argmax_and_temperature():
    d = {"a": 0.1, "b": 0.7, "c": 0.2}
    assert token_policy.pick(d, None, random.Random(0)) == "b"
    rng = random.Random(1)
    draws = [token_policy.pick(d, 1.0, rng) for _ in range(2000)]
    assert 0.6 < draws.count("b") / 2000 < 0.8
    cold = [token_policy.pick(d, 0.05, rng) for _ in range(200)]
    assert cold.count("b") > 190


def test_distribution_is_legal_and_normalised(tiny_net, states):
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    assert pol.device == "cpu"
    for s in states:
        for reveal in (None, []):
            dist = pol.distribution([s], reveal)
            assert dist and abs(sum(dist.values()) - 1) < 1e-5
            assert set(dist) <= legal_choices(s)
            assert all(p > 0 for p in dist.values())


def test_batched_over_worlds_is_the_mean(tiny_net, states):
    pol = token_policy.TokenPolicy.load(tiny_net, "cpu")
    a, b = states[0], states[0]
    names, probs, legal = pol.predict_states([a, b], None)
    assert probs.shape == (2, 13) and legal.shape == (2, 13) and len(names) == 2
    assert np.allclose(probs.sum(1), 1, atol=1e-4)
    assert (probs[~legal] < 1e-6).all()  # illegal slots carry no mass
    one = pol.distribution([a], None)
    two = pol.distribution([a, b], None)
    assert one.keys() == two.keys() and all(abs(one[k] - two[k]) < 1e-5 for k in one)


def test_spec_mismatch_is_refused(tmp_path, states):
    cfg = TokenNetConfig.from_spec(tokens.token_spec(), d_model=32, n_layers=2, n_heads=4)
    cfg.spec_hash = "0" * 16
    path = tmp_path / "bad.pt"
    TokenNet(cfg).export(path)
    with pytest.raises(ValueError, match="spec mismatch"):
        token_policy.TokenPolicy.load(path, "cpu")


def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        token_policy.TokenPolicy.load("no/such/net.pt", "cpu")


def test_trained_model_distribution(states):
    path = real_model()
    if path is None:
        pytest.skip("models/tok-d256.pt not available")
    pol = token_policy.TokenPolicy.load(path, "cpu")
    for s in states:
        dist = pol.distribution([s], None)
        assert set(dist) <= legal_choices(s) and abs(sum(dist.values()) - 1) < 1e-5


# ------------------------------------------------------------------ the prior reaches the engine


def test_root_prior_steers_mcts(states):
    state = pe.State.from_string(states[0])
    options = [o for o, _ in pe.policy_actions(state, 1)]
    assert len(options) >= 2
    plain = pe.monte_carlo_tree_search(state, 300, threads=1)
    base = {o.move_choice: o.visits for o in plain.side_one}
    assert set(base) == set(options)  # choice strings are the same spelling as policy_actions
    favourite = min(base, key=base.get)  # the option plain UCB1 likes least
    prior = {c: (1.0 if c == favourite else 0.0) for c in options}
    res = pe.monte_carlo_tree_search(state, 300, threads=1, root_prior_s1=prior, c_puct=4.0, prior_eps=0.03)
    visits = {o.move_choice: o.visits for o in res.side_one}
    assert set(visits) == set(options)
    # PUCT gives the prior's favourite far more of the visits than UCB1 does
    assert visits[favourite] / sum(visits.values()) > 5 * base[favourite] / sum(base.values()), (visits, base)


def test_worker_accepts_token_root_prior(states):
    from showdown_bot.players import _worker

    state = pe.State.from_string(states[0])
    options = [o for o, _ in pe.policy_actions(state, 1)]
    prior = {c: 1.0 / len(options) for c in options}
    total, rows = _worker.search_world(states[0], 100, 1, root_prior_s1=prior)[:2]
    assert total > 0 and {c for c, _v, _s in rows} == set(options)


# ------------------------------------------------------------------ SearchConfig / player wiring


def test_search_config_validates_tok_prior(tiny_net):
    from showdown_bot.players.search import SearchConfig, SearchNetPlayer

    cfg = SearchConfig(tok_prior=str(tiny_net), tok_device="cpu", search_time_ms=50, worlds=2)
    player = SearchNetPlayer(config=cfg, evaluator=_any_pev(), battle_format="gen9randombattle",
                             start_listening=False)
    assert player.tok_net is not None and player.config.tok_prior == player.tok_net.path
    kw = player._prior_kwargs()
    assert kw["prior_path"] is None and "root_prior_s1" not in kw  # none computed yet
    player._tok_root = {"tackle": 1.0}
    assert player._prior_kwargs()["root_prior_s1"] == {"tackle": 1.0}
    with pytest.raises(FileNotFoundError):
        SearchNetPlayer(config=SearchConfig(tok_prior="nope.pt"), evaluator=_any_pev(),
                        battle_format="gen9randombattle", start_listening=False)


def _any_pev() -> str:
    p = Path(__file__).resolve()
    for parent in p.parents:
        found = sorted((parent / "models").glob("*.pev")) if (parent / "models").is_dir() else []
        if found:
            return str(found[0])
    pytest.skip("no .pev evaluator available")


# ------------------------------------------------------------------ smoke game


def server_up(port: int = 8000) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not server_up(), reason="no local Showdown server on :8000")
def test_token_policy_plays_a_game_vs_random(tiny_net):
    from poke_env import AccountConfiguration, RandomPlayer

    from showdown_bot import bench
    from showdown_bot.players.tokenpolicy import TokenPolicyPlayer

    path = real_model() or tiny_net
    server = bench.local_server(8000)
    bot = TokenPolicyPlayer(
        model=str(path), device="cpu", battle_format="gen9randombattle", server_configuration=server,
        account_configuration=AccountConfiguration("tp" + secrets.token_hex(4), None), seed=1, log_level=40,
    )
    rnd = RandomPlayer(
        battle_format="gen9randombattle", server_configuration=server,
        account_configuration=AccountConfiguration("rp" + secrets.token_hex(4), None), log_level=40,
    )

    async def play():
        await asyncio.wait_for(bot.battle_against(rnd, n_battles=1), timeout=240)

    asyncio.run(play())
    assert bot.n_finished_battles == 1
    assert bot.token_moves > 0 and bot.token_errors == 0
