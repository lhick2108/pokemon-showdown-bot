"""Token transformer (bead showdown-bot-589.68.3): shapes, masks, overfit, export/load, shard streaming and
a tiny end-to-end training run. Synthetic data, tiny config, CPU: no engine wheel needed."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from showdown_bot import token_net, token_train  # noqa: E402
from showdown_bot.token_net import N_ACT, TokenNet, TokenNetConfig  # noqa: E402

SPEC = {
    "token_version": 2, "shard_format": 1, "vocab_hash": "0x1", "n_actions": 13,
    "vocab_sizes": {"species": 60, "item": 20, "ability": 25, "move": 40},
    "constants": {"C_MON": 173, "C_SIDE": 62, "C_GLOB": 15, "N_PAIR_CH": 5, "N_AGG": 8, "N_AFEAT": 58, "KNOWN_W": 8},
    "fields": {}, "hist": {"n": 16, "cat_w": 4, "cont_w": 5},
}


def tiny_cfg(**kw) -> TokenNetConfig:
    return TokenNetConfig.from_spec(SPEC, d_model=32, n_layers=2, n_heads=4, **kw)


def fake(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    r = np.random.default_rng(seed)
    nsp, nit, nab, nmv = 60, 20, 25, 40
    legal = (r.random((n, 2, N_ACT)) < 0.5).astype(np.int8)
    legal[:, :, 0] = 1
    pol = r.random((n, N_ACT)).astype(np.float32) * legal[:, 0]
    pol /= pol.sum(1, keepdims=True)
    # a learnable signal: the best action is the legal slot with the largest first act_cont feature
    act_cont = r.normal(size=(n, 2, N_ACT, 58)).astype(np.float16)
    score = np.where(legal[:, 0] > 0, act_cont[:, 0, :, 0].astype(np.float32), -9)
    best = score.argmax(1)
    pol = np.eye(N_ACT, dtype=np.float32)[best]
    mon_cat = np.stack([r.integers(0, nsp, (n, 12)), r.integers(0, nit, (n, 12)), r.integers(0, nab, (n, 12))]
                       + [r.integers(0, nmv, (n, 12)) for _ in range(4)], -1).astype(np.int16)
    b = dict(
        mon_cat=mon_cat,
        mon_known=r.integers(0, 2, (n, 12, 8)).astype(np.int8),
        mon_cont=r.normal(size=(n, 12, 173)).astype(np.float16),
        side_cont=r.normal(size=(n, 2, 62)).astype(np.float16),
        glob=r.normal(size=(n, 15)).astype(np.float16),
        match_pair=(r.normal(size=(n, 36, 5)) * (r.random((n, 36, 1)) < 0.7)).astype(np.float16),
        match_agg=r.normal(size=(n, 8)).astype(np.float16),
        act_cat=np.stack([r.integers(0, nmv, (n, 2, N_ACT)), r.integers(0, nsp, (n, 2, N_ACT))], -1).astype(np.int16),
        act_legal=legal, act_cont=act_cont, meta=np.zeros((n, 4), np.int16),
        hist_cat=np.stack([r.integers(0, 3, (n, 16)), r.integers(0, 4, (n, 16)), r.integers(0, nmv, (n, 16)),
                           r.integers(0, nsp, (n, 16))], -1).astype(np.int16),
        hist_cont=r.normal(size=(n, 16, 5)).astype(np.float16),
        pol_target=pol, pol_valid=np.ones(n, np.int8),
        z=(act_cont[:, 0, 0, 1] > 0).astype(np.float32),  # learnable from the inputs
        v_search=r.random(n).astype(np.float32), v_search_valid=(r.random(n) < 0.5).astype(np.int8),
        opp_act_slot=np.where(r.random(n) < 0.5, np.argmax(legal[:, 1], 1), -1).astype(np.int8),
        opp_act_kind=r.integers(-1, 3, n).astype(np.int8), opp_act_id=r.integers(0, nmv, n).astype(np.int16),
        opp_act_valid=np.ones(n, np.int8),
        hid_ids=np.stack([r.integers(0, nsp, (n, 6)), r.integers(0, nit, (n, 6)), r.integers(0, nab, (n, 6))]
                         + [r.integers(2, nmv, (n, 6)) for _ in range(4)] + [r.integers(0, 20, (n, 6))], -1).astype(np.int16),
        hid_mask=np.ones((n, 6), np.int8), hid_valid=np.ones(n, np.int8),
        aux_turns_left=r.integers(0, 40, n).astype(np.int16), aux_final_alive=r.integers(0, 7, (n, 2)).astype(np.int8),
        aux_valid=np.ones((n, 3), np.int8), rating=np.full(n, 4, np.int8), battle_hash=(np.arange(n) // 5).astype(np.int64),
    )
    b["opp_act_slot"][legal[np.arange(n), 1, np.clip(b["opp_act_slot"], 0, 12)] == 0] = -1
    return b


def tensors(b, fields=None):
    t = token_net.to_tensors(b, "cpu")
    t["src"] = torch.zeros(len(b["z"]), dtype=torch.long)
    return t


def test_forward_shapes_and_masks():
    cfg = tiny_cfg()
    net = TokenNet(cfg)
    b = tensors(fake(5))
    out = net(b)
    assert out["policy"].shape == (5, 13) and out["opp_slot"].shape == (5, 13) and out["value"].shape == (5,)
    assert out["opp_vocab"].shape == (5, cfg.n_move + cfg.n_species)
    assert out["hid_item"].shape == (5, 6, cfg.n_item) and out["hid_move"].shape == (5, 6, cfg.n_move)
    assert all(torch.isfinite(v).all() for v in out.values())
    legal = b["act_legal"][:, 0] > 0
    assert (out["policy"][~legal] <= -1e8).all() and (out["policy"][legal] > -1e8).all()
    assert torch.allclose(torch.softmax(out["policy"], -1).sum(-1), torch.ones(5))
    assert N_ACT == 13 and token_net.N_TOK == 94


def test_illegal_action_and_pad_history_tokens_are_ignored():
    torch.manual_seed(0)
    net = TokenNet(tiny_cfg()).eval()
    for h in (net.h_pol, net.h_val):
        h[-1].weight.data.normal_(std=0.1)  # the init is zero: make the heads sensitive
    a = fake(4, 1)
    a["hist_cat"][:, 5, 0] = 0  # a padding history row
    b = {k: v.copy() for k, v in a.items()}
    illegal = a["act_legal"] == 0
    for s in range(2):
        b["act_cont"][:, s][illegal[:, s]] = 5.0  # garbage in the illegal action tokens
        b["act_cat"][:, s][illegal[:, s]] = 1
    b["hist_cont"][:, 5] = 7.0  # garbage in the padding history row
    with torch.no_grad():
        oa, ob = net(tensors(a)), net(tensors(b))
    legal = a["act_legal"][:, 0] > 0
    assert torch.allclose(oa["policy"][legal], ob["policy"][legal], atol=1e-5)
    assert torch.allclose(oa["value"], ob["value"], atol=1e-5)


def test_overfits_a_small_batch():
    torch.manual_seed(0)
    net = TokenNet(tiny_cfg())
    b = tensors(fake(48, 2))
    cfg = token_train.LossCfg(None)
    scale = torch.ones(len(token_train.TERMS), 1)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    first = last = None
    for i in range(60):
        loss, parts = token_train.compute_loss(net(b), b, cfg, scale, net.cfg.n_move)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or float(loss.detach())
        last = float(loss.detach())
    assert last < 0.6 * first, (first, last)
    assert all(torch.isfinite(v) for v in parts.values())


def test_value_target_modes():
    b = tensors(fake(10, 3))
    z = token_train.value_target(b, token_train.LossCfg(None, value_target="z"))
    s = token_train.value_target(b, token_train.LossCfg(None, value_target="search"))
    m = token_train.value_target(b, token_train.LossCfg(None, value_target="mix", value_mix=0.25))
    ok = b["v_search_valid"] > 0
    assert torch.equal(z, b["z"]) and torch.equal(s[~ok], b["z"][~ok]) and torch.equal(s[ok], b["v_search"][ok])
    assert torch.allclose(m[ok], 0.25 * b["v_search"][ok] + 0.75 * b["z"][ok]) and torch.equal(m[~ok], b["z"][~ok])


def test_export_load_roundtrip_and_spec_check(tmp_path):
    torch.manual_seed(0)
    net = TokenNet(tiny_cfg()).eval()
    b = fake(6, 4)
    p = net.predict(b, "cpu", amp=False)
    assert np.allclose(p["policy"].sum(1), 1, atol=1e-5) and (p["value"] >= 0).all() and (p["value"] <= 1).all()
    assert np.allclose(p["opp_policy"].sum(1), 1, atol=1e-5)
    net.export(tmp_path / "n.pt", extra={"note": "x"})
    assert json.loads((tmp_path / "n.pt.json").read_text())["n_params"] == net.n_params()
    net2 = TokenNet.load(tmp_path / "n.pt", expect_spec=SPEC)
    p2 = net2.predict(b, "cpu", amp=False)
    assert all(np.allclose(p[k], p2[k], atol=1e-6) for k in p)
    bad = dict(SPEC, vocab_sizes={"species": 61, "item": 20, "ability": 25, "move": 40})
    with pytest.raises(ValueError, match="spec mismatch"):
        TokenNet.load(tmp_path / "n.pt", expect_spec=bad)


def test_default_config_param_count():
    n = TokenNet(TokenNetConfig()).n_params()
    assert 4e6 < n < 8e6, n


def write_shards(root, sizes=(300, 200), kinds=("selfplay", "human")):
    (root / "shards").mkdir(parents=True)
    shards, sources = [], []
    for si, (kind, n) in enumerate(zip(kinds, sizes)):
        sources.append({"id": si, "name": f"s{si}", "kind": kind, "path": "x"})
        for k in range(2):
            name = f"s{si}-{k:05d}"
            d = root / "shards" / name
            d.mkdir()
            data = fake(n // 2, seed=10 * si + k)
            m = len(data["z"])
            data["split"] = (np.arange(m) % 10 == 0).astype(np.int8)  # 10% val
            for f, v in data.items():
                np.save(d / f"{f}.npy", v)
            shards.append({"name": name, "source": si, "n": m, "fields": {}})
    (root / "manifest.json").write_text(json.dumps({"spec": SPEC, "sources": sources, "shards": shards, "n_samples": sum(s["n"] for s in shards)}))


def test_streams_and_mixer(tmp_path):
    write_shards(tmp_path)
    c = token_train.Corpus([tmp_path])
    assert [c.n(i, "train") for i in (0, 1)] == [270, 180] and [c.n(i, "val") for i in (0, 1)] == [30, 20]
    mix = token_train.Mixer(c, {"selfplay": 3.0, "human": 1.0}, 0)
    assert np.allclose(mix.p, [0.75, 0.25])
    seen = []
    for _ in range(30):
        b = mix.take(32)
        assert b["z"].shape == (32,) and set(b["src"].tolist()) <= {0, 1}
        seen.append(float((b["src"] == 0).mean()))
    assert 0.6 < np.mean(seen) < 0.9
    # one pass over a split covers every row exactly once (no leakage between train and val rows)
    rows = sum(len(b["z"]) for b in token_train.eval_batches(c, 0, "val", 10**6, 16))
    assert rows == 30


def test_end_to_end_train_resume_and_report(tmp_path, capsys):
    write_shards(tmp_path / "d")
    out = tmp_path / "m" / "t.pt"
    args = ["--data", str(tmp_path / "d"), "--weight", "selfplay=1", "--weight", "human=0.5", "--d-model", "32", "--layers", "1",
            "--heads", "2", "--batch", "32", "--epochs", "2", "--epoch-samples", "320", "--device", "cpu", "--amp", "none",
            "--log-every", "5", "--warmup", "5", "--final-eval-max", "100", "--eval-max", "100", "--prefetch", "1",
            "--out", str(out), "--src-loss", "human.hid=0", "--id-dropout", "0.5", "--max-per-battle", "3"]
    assert token_train.main(args) == 0
    text = (tmp_path / "m" / "log-t.txt").read_text()
    assert "FINAL METRICS" in text and "(1) self-play policy top-1" in text and "(3) value BCE" in text
    assert out.exists() and (tmp_path / "m" / "t.pt.last").exists()
    assert TokenNet.load(out, check_engine=False).cfg.d_model == 32
    # resume: continues from the stored epoch, nothing left to do
    assert token_train.main(args + ["--resume", "--no-final"]) == 0
    assert "resumed from" in (tmp_path / "m" / "log-t.txt").read_text()


def test_blank_ids_and_cap_per_battle(tmp_path):
    b = tensors(fake(8, 5))
    rows = torch.tensor([True, False] * 4)
    o = token_train.blank_ids(b, rows)
    assert (o["mon_cat"][rows] <= 1).all() and (o["act_cat"][rows] == 0).all() and (o["hist_cat"][rows][..., 2:] == 0).all()
    assert torch.equal(o["mon_cat"][~rows], b["mon_cat"][~rows]) and torch.equal(o["mon_cont"], b["mon_cont"])
    # blanked samples train the value head only
    net = TokenNet(tiny_cfg())
    b["blank"] = torch.ones(8, dtype=torch.bool)
    loss, parts = token_train.compute_loss(net(b), b, token_train.LossCfg(None), torch.ones(5, 1), net.cfg.n_move)
    assert float(parts["pol"]) == 0 and float(parts["hid"]) == 0 and float(parts["val"]) > 0
    write_shards(tmp_path)
    c = token_train.Corpus([tmp_path])
    st = token_train.Stream(c, 0, "train", np.random.default_rng(0), cap=2)
    st._open()
    for a in st.active:  # fake battles have 5 consecutive rows; the cap keeps at most 2 of each
        assert len(a[1]) <= 2 * ((150 + 4) // 5)
    hp = token_train.fit_hpsum(c, 0, 100)
    assert np.isfinite(hp).all()
