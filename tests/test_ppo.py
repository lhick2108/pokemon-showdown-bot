"""PPO trainer pieces and the Rust self-play pool API (bead showdown-bot-589.68.6).

The Rust unit tests (team legality, reveal scrub, history, determinism) are in engine/src/selfplay.rs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not hasattr(pe, "SelfPlayPool"), reason="engine wheel without SelfPlayPool (sdb.26)")

from showdown_bot import league, ppo, token_net, tokens  # noqa: E402

LAY = tokens.layout() if hasattr(pe, "token_layout") else {}
CAT_LEN = max((f["offset"] + int(np.prod(f["shape"])) for f in LAY.values() if f["buffer"] == "cat"), default=0)
CONT_LEN = max((f["offset"] + int(np.prod(f["shape"])) for f in LAY.values() if f["buffer"] == "cont"), default=0)


def legal_of(obs: ppo.Obs) -> np.ndarray:
    off = LAY["act_legal"]["offset"]
    return obs.cat[:, off : off + 13] > 0


def play(pool, steps: int, seed: int = 0) -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(seed)
    fin = []
    for _ in range(steps):
        obs = ppo.Obs(pool.observe(), CAT_LEN, CONT_LEN)
        acts = np.array([rng.choice(np.nonzero(l)[0]) for l in legal_of(obs)], np.uint8)
        _, _, f = pool.step(acts.tobytes())
        fin += f
    return fin


# ------------------------------------------------------------------ pure loss pieces


def test_gae_chain_and_bootstrap():
    # two decisions of one seat: r=0 then a win (+1); values 0.2, 0.5
    r = np.array([0.0, 1.0], np.float32)
    v = np.array([0.2, 0.5], np.float32)
    adv, ret = ppo.compute_gae(r, v, np.array([0, 1], np.uint8), np.array([1, -1]), v.copy(), 1.0, 1.0)
    assert np.allclose(adv, [1.0 - 0.2, 1.0 - 0.5]) and np.allclose(ret, [1.0, 1.0])
    # a cut trajectory bootstraps; lambda 0 is the one-step TD error
    adv0, _ = ppo.compute_gae(np.zeros(2, np.float32), v, np.zeros(2, np.uint8), np.array([1, -1]), np.array([0, 0.9], np.float32), 0.9, 0.0)
    assert np.allclose(adv0, [0.9 * 0.5 - 0.2, 0.9 * 0.9 - 0.5], atol=1e-6)


def test_gae_independent_trajectories_do_not_mix():
    r = np.array([0, 0, 1, -1], np.float32)
    v = np.zeros(4, np.float32)
    nxt = np.array([2, 3, -1, -1])
    adv, _ = ppo.compute_gae(r, v, np.array([0, 0, 1, 1], np.uint8), nxt, v, 1.0, 0.95)
    assert np.allclose(adv, [0.95, -0.95, 1, -1])


def test_value_target_and_beta_schedule():
    assert np.allclose(ppo.value_target(np.array([-1.5, -1, 0, 1, 2])), [0, 0, 0.5, 1, 1])
    cfg = ppo.PPOConfig(kl_beta0=0.1, kl_beta1=0.01, kl_decay_updates=10)
    assert ppo.kl_beta(0, cfg) == pytest.approx(0.1) and ppo.kl_beta(10, cfg) == pytest.approx(0.01)
    assert ppo.kl_beta(100, cfg) == pytest.approx(0.01) and 0.01 < ppo.kl_beta(5, cfg) < 0.1


def test_ppo_terms_values():
    logits = torch.zeros(2, 13)
    legal = torch.zeros(2, 13, dtype=torch.bool)
    legal[:, :4] = True
    legal[1, 4:6] = True
    act = torch.tensor([0, 1])
    old = torch.log(torch.tensor([0.25, 1 / 6]))
    t = ppo.ppo_terms(logits, legal, act, old, torch.tensor([1.0, -1.0]), logits.clone(), 0.2)
    # ratio is exactly 1, entropy is ln(k) of the legal set, no KL to an identical anchor
    assert t["pg"].item() == pytest.approx(0.0, abs=1e-6)
    assert t["entropy"].item() == pytest.approx((np.log(4) + np.log(6)) / 2, abs=1e-5)
    assert t["kl_anchor"].item() == pytest.approx(0.0, abs=1e-6) and t["clipfrac"].item() == 0
    other = logits.clone()
    other[:, 0] = 2.0
    t2 = ppo.ppo_terms(logits, legal, act, old, torch.tensor([1.0, -1.0]), other, 0.2)
    assert t2["kl_anchor"].item() > 0.05
    # illegal actions never carry probability: a huge logit on an illegal slot changes nothing
    spiky = logits.clone()
    spiky[:, 12] = 50.0
    t3 = ppo.ppo_terms(spiky, legal, act, old, torch.tensor([1.0, -1.0]), None, 0.2)
    assert t3["entropy"].item() == pytest.approx(t["entropy"].item(), abs=1e-5)


def test_ppo_clipping_stops_the_gradient():
    legal = torch.ones(1, 13, dtype=torch.bool)
    logits = torch.zeros(1, 13, requires_grad=True)
    act = torch.tensor([3])
    # new policy already far above the old one for a positive advantage: clipped, no gradient
    old = torch.log(torch.tensor([1 / 13])) - 1.0
    t = ppo.ppo_terms(logits, legal, act, old, torch.tensor([1.0]), None, 0.2)
    t["pg"].backward()
    assert logits.grad.abs().max().item() == 0.0
    # negative advantage with a ratio above 1 + eps is NOT clipped
    logits.grad = None
    t = ppo.ppo_terms(logits, legal, act, old, torch.tensor([-1.0]), None, 0.2)
    t["pg"].backward()
    assert logits.grad.abs().max().item() > 0


# ------------------------------------------------------------------ pool API


def test_pool_observe_step_api():
    pool = pe.SelfPlayPool(16, 5, 2, 200)
    obs = ppo.Obs(pool.observe(), CAT_LEN, CONT_LEN)
    assert obs.n >= 16 and obs.cat.shape == (obs.n, CAT_LEN) and obs.cont.shape == (obs.n, CONT_LEN)
    assert obs.hist_cat.shape == (obs.n, 16, 4) and obs.hist_cont.shape == (obs.n, 16, 5)
    assert set(np.unique(obs.side)) <= {0, 1} and obs.game.max() < 16
    assert legal_of(obs).sum(-1).min() >= 2, "only real decisions are listed"
    assert obs.legal is None or (obs.legal > 0).tolist() == legal_of(obs).tolist()
    assert np.isfinite(obs.cont).all()
    with pytest.raises(ValueError):
        pool.step(bytes(obs.n + 1))
    fin = play(pool, 400, seed=1)
    st = pool.stats()
    assert st["illegal"] == 0 and st["panics"] == 0
    assert len(fin) == st["games"] and len(fin) > 5
    assert all(w in (0, 1, 2) and 1 <= t <= 200 for _, w, t in fin)
    assert pe.SelfPlayPool(4, 1, 1, 200).game_state(0).count("/") > 3


def test_pool_is_deterministic_and_thread_independent():
    def run(threads: int) -> tuple[bytes, list]:
        pool = pe.SelfPlayPool(12, 77, threads, 200)
        fin = play(pool, 60, seed=3)
        return pool.observe()["cat"], fin

    a, b, c = run(1), run(1), run(3)
    assert a == b and a == c


def test_rewards_and_done_flags():
    pool = pe.SelfPlayPool(16, 9, 2, 200)
    rng = np.random.default_rng(0)
    seen = 0
    for _ in range(500):
        obs = ppo.Obs(pool.observe(), CAT_LEN, CONT_LEN)
        acts = np.array([rng.choice(np.nonzero(l)[0]) for l in legal_of(obs)], np.uint8)
        rew, done, fin = pool.step(acts.tobytes())
        rew, done = np.frombuffer(rew, "<f4"), np.frombuffer(done, np.uint8)
        assert len(rew) == len(done) == obs.n
        assert (rew[done == 0] == 0).all() and np.isin(rew, [-1, 0, 1]).all()
        winners = {g: w for g, w, _ in fin}
        for i in np.nonzero(done)[0]:
            w = winners[int(obs.game[i])]
            assert rew[i] == (0 if w == 0 else (1 if w == obs.side[i] + 1 else -1))
            seen += 1
    assert seen > 10


def test_team_generator_makes_legal_distinct_teams():
    states = pe.selfplay_teams(3, 40)
    assert len(states) == 40 and len(set(states)) == 40
    assert pe.selfplay_teams(3, 40) == states, "seeded"
    n_species, n_sets, dropped, dropped_sets = pe.selfplay_data_info()
    assert n_species > 450 and n_sets > 4000 and dropped_sets == 0
    for s in states[:10]:
        st = pe.State.from_string(s)
        assert len(st.side_one.pokemon) == 6 and len(st.side_two.pokemon) == 6
        assert len({p.id for p in st.side_one.pokemon}) == 6


def test_fresh_observation_hides_the_opposing_bench():
    pool = pe.SelfPlayPool(6, 12, 1, 200)
    obs = ppo.Obs(pool.observe(), CAT_LEN, CONT_LEN)
    known = LAY["mon_known"]["offset"]
    cat = LAY["mon_cat"]["offset"]
    k = obs.cat[:, known : known + 12 * 8].reshape(obs.n, 12, 8)
    c = obs.cat[:, cat : cat + 12 * 7].reshape(obs.n, 12, 7)
    assert (k[:, 6, 0] == 1).all() and (k[:, 6, 1:] == 0).all(), "only the opposing lead's species is known at the start"
    assert (k[:, 7:, :] == 0).all() and (c[:, 7:, :] == 1).all(), "the opposing bench is entirely unknown"
    assert (k[:, :6, 0] == 1).all() and (k[:, :6, 1:] == 1).all(), "own side is fully known"
    assert (obs.hist_cat == 0).all(), "no history before the first turn"


# ------------------------------------------------------------------ trainer end to end (tiny net, CPU)


def tiny_net(seed: int = 0):
    torch.manual_seed(seed)
    cfg = token_net.TokenNetConfig.from_spec(tokens.token_spec(), d_model=32, n_layers=1, n_heads=2)
    return token_net.TokenNet(cfg)


def test_full_snapshot_pool_drops_one_and_keeps_games_playable(tmp_path):
    cfg = ppo.PPOConfig(games=32, threads=2, rollout_steps=4, epochs=1, minibatch=64, value_warmup=0, max_past=2,
                        p_self=0.0, p_anchor=0.0, amp=False, window=50)
    learner, anchor = tiny_net(0), tiny_net(1)
    tr = ppo.Trainer(cfg, learner, anchor, "cpu", tmp_path)
    a = tr.arena
    for _ in range(2):
        a.add_snapshot(learner)
    for g in range(cfg.games):
        a._assign(g)
    for _ in range(3):
        a.add_snapshot(learner)
        assert len(a.past_ids) == 2
        live = set(np.unique(a.opp).tolist()) - {ppo.SELF, ppo.ANCHOR}
        assert live <= set(a.opponents), "every game's opponent is still in the pool"
        a.step(None)


def test_trainer_updates_and_exports_a_loadable_checkpoint(tmp_path):
    cfg = ppo.PPOConfig(games=16, threads=2, rollout_steps=6, epochs=1, minibatch=64, value_warmup=1, ckpt_every=2,
                        snapshot_every=1, amp=False, window=50)
    learner, anchor = tiny_net(0), tiny_net(1)
    ppo.zero_dropout(learner)
    tr = ppo.Trainer(cfg, learner, anchor, "cpu", tmp_path)
    tr.arena.add_snapshot(learner)
    before = [p.detach().clone() for p in learner.parameters()]
    rows = [tr.train_step() for _ in range(3)]
    for m in rows:
        assert all(np.isfinite(v) for k, v in m.items() if k not in ("wr_anchor", "wr_past", "wr_self_p1")), m
        assert m["samples"] > 0 and m["pool_illegal"] == 0 and m["pool_panics"] == 0
        assert 0 <= m["entropy"] <= np.log(13)
    assert any((a != b).any() for a, b in zip(before, learner.parameters())), "weights moved"
    log = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["update"] for r in log] == [1, 2, 3]
    ck = tmp_path / "latest.pt"
    assert ck.is_file() and list(tmp_path.glob("ckpt-*.pt"))
    # same format as token-trained nets: loads through TokenNet.load / the token-policy loader
    net = token_net.TokenNet.load(ck)
    assert net.cfg.d_model == 32
    from showdown_bot import token_policy

    tp = token_policy.TokenPolicy.load(ck, device="cpu")
    assert tp.net.cfg.spec_hash == learner.cfg.spec_hash


def test_ppo_only_ever_samples_legal_actions():
    net = tiny_net(2)
    pool = pe.SelfPlayPool(24, 4, 2, 200)
    bat = ppo.TokenBatcher("cpu")
    pol = ppo.Policy(net, "cpu", amp=False)
    for _ in range(25):
        obs = ppo.Obs(pool.observe(), CAT_LEN, CONT_LEN)
        f = pol.forward(bat.to_device(obs.cat, obs.cont, obs.hist_cat, obs.hist_cont))
        assert np.isfinite(f.logp).all() and np.isfinite(f.value).all()
        assert legal_of(obs)[np.arange(obs.n), f.actions].all()
        pool.step(f.actions.tobytes())
    assert pool.stats()["illegal"] == 0


def test_distillation_reduces_the_policy_gap():
    teacher, student = tiny_net(3), tiny_net(4)
    cfg = ppo.PPOConfig(games=16, threads=2, rollout_steps=8, minibatch=64, amp=False)
    log = ppo.distill(teacher, student, cfg, "cpu", 4)
    assert len(log) == 4 and all(np.isfinite(r["kl_teacher"]) for r in log)
    assert log[-1]["kl_teacher"] < log[0]["kl_teacher"]


def test_small_arch_builds_with_the_teachers_spec():
    teacher = tiny_net(5)
    s = ppo.small_net(teacher, "d192L4")
    assert s.cfg.d_model == 192 and s.cfg.n_layers == 4 and s.cfg.spec_hash == teacher.cfg.spec_hash
    with pytest.raises(ValueError):
        ppo.small_net(teacher, "nope")


def test_documented_evaluation_specs_parse():
    cand = r"cand=token-policy:model=runs\ppo1\latest.pt:device=cuda"
    best = (r"best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:"
            r"evaluator=models\gen7-res-root-mlp16.pev")
    league.parse_spec(cand)
    league.parse_spec(best)
    league.parse_spec(r"cand=net-search:model=runs\ppo1\latest.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4")


# ------------------------------------------------------------------ asymmetric critic


def test_kl_beta_switches_off():
    cfg = ppo.PPOConfig(kl_beta0=0.1, kl_beta1=0.02, kl_decay_updates=10, kl_off_after=5)
    assert ppo.kl_beta(4, cfg) > 0 and ppo.kl_beta(5, cfg) == 0.0 and ppo.kl_beta(50, cfg) == 0.0
    assert ppo.kl_beta(50, ppo.PPOConfig(kl_beta1=0.0, kl_decay_updates=10)) == 0.0  # kl-beta1 0 already reaches beta 0


def _asym_cfg(**kw):
    base = dict(games=16, threads=2, rollout_steps=70, epochs=1, minibatch=64, value_warmup=1, ckpt_every=2,
                snapshot_every=1, amp=False, window=50, critic="asym")
    base.update(kw)
    return ppo.PPOConfig(**base)


def test_critic_view_is_unscrubbed(tmp_path):
    tr = ppo.Trainer(_asym_cfg(), tiny_net(0), tiny_net(1), "cpu", tmp_path)
    a = tr.arena
    for _ in range(4):
        a.step(None)
    obs = a._decode()
    rows = np.arange(obs.n)
    cat_t, cont_t = a.truth_tokens(obs, rows)
    assert cat_t.shape == obs.cat.shape and cont_t.shape == obs.cont.shape
    known = LAY["mon_known"]["offset"]
    k_act = obs.cat[:, known : known + 12 * 8].reshape(obs.n, 12, 8)
    k_tru = cat_t[:, known : known + 12 * 8].reshape(obs.n, 12, 8)
    assert (k_tru[:, :6] == k_act[:, :6]).all(), "own side identical in both views"
    assert (k_tru[:, 6:, :] == 1).all(), "the whole opposing team is known in the critic view"
    assert (k_act[:, 6:, :] == 0).any(), "while the actor view still hides part of it"
    assert (cat_t != obs.cat).any()


def test_shared_mode_has_no_critic(tmp_path):
    cfg = ppo.PPOConfig(games=8, threads=1, rollout_steps=3, epochs=1, minibatch=32, value_warmup=0, amp=False, window=20)
    tr = ppo.Trainer(cfg, tiny_net(0), tiny_net(1), "cpu", tmp_path)
    assert tr.critic is None and tr.arena.critic is None
    ro = tr.arena.rollout(3)
    assert ro.cat_t is None and ro.cont_t is None
    tr.train_step()
    tr.save_state()
    assert not (tmp_path / "critic.pt").exists()


def test_asym_trainer_trains_critic_only_in_warmup_and_saves_separately(tmp_path):
    cfg = _asym_cfg()
    learner, anchor = tiny_net(0), tiny_net(1)
    ppo.zero_dropout(learner)
    tr = ppo.Trainer(cfg, learner, anchor, "cpu", tmp_path)
    tr.arena.add_snapshot(learner)
    actor0 = [p.detach().clone() for p in learner.parameters()]
    crit0 = [p.detach().clone() for p in tr.critic.parameters()]
    assert all((a == b).all() for a, b in zip(crit0, anchor.parameters())), "critic starts at the anchor"
    m1 = tr.train_step()  # warm-up update
    assert m1["warmup"] == 1 and m1["asym"] == 1
    assert all((a == b).all() for a, b in zip(actor0, learner.parameters())), "actor untouched during warm-up"
    assert any((a != b).any() for a, b in zip(crit0, tr.critic.parameters())), "critic trained"
    rows = [m1] + [tr.train_step() for _ in range(2)]
    for m in rows:
        assert np.isfinite(m["value_loss"]) and np.isfinite(m["explained_var"]) and m["pool_illegal"] == 0
    assert any((a != b).any() for a, b in zip(actor0, learner.parameters())), "actor moves after warm-up"
    # actor checkpoint stays loadable by the policy loaders; the critic is a separate file
    from showdown_bot import token_policy

    token_policy.TokenPolicy.load(tmp_path / "latest.pt", device="cpu")
    assert (tmp_path / "critic.pt").is_file() and (tmp_path / "state.pt").is_file()
    # resume restores the critic and its optimizer
    tr2 = ppo.Trainer(cfg, tiny_net(0), tiny_net(1), "cpu", tmp_path)
    tr2.resume()
    assert tr2.update == 2  # state.pt is written every ckpt_every=2 updates
    saved = torch.load(tmp_path / "critic.pt", weights_only=False)["state_dict"]
    for a, b in zip(saved.values(), tr2.critic.state_dict().values()):
        assert torch.allclose(a.cpu().float(), b.cpu().float())
    assert len(tr2.opt_c.state_dict()["state"]) > 0
