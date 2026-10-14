"""Human action-policy model: slots, numpy forward pass, perspective swap, training pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")

from showdown_bot import human_policy as hp  # noqa: E402

STATE = (Path(__file__).parent / "fixtures" / "mcts_deep_chain.state").read_text().strip()


def _random_policy(h1=16, h2=8, seed=0) -> hp.HumanPolicy:
    rng = np.random.default_rng(seed)
    return hp.HumanPolicy({
        "w1": rng.normal(0, 0.05, (hp.N_FEATURES, h1)), "b1": rng.normal(0, 0.1, h1),
        "wm": rng.normal(0, 0.5, (hp.N_SLOTS, h1)), "elo": rng.normal(0, 0.1, (hp.N_ELO + 1, h1)),
        "w2": rng.normal(0, 0.3, (h1, h2)), "b2": rng.normal(0, 0.1, h2),
        "w3": rng.normal(0, 0.3, (h2, hp.N_SLOTS)), "b3": rng.normal(0, 0.1, hp.N_SLOTS),
    })


def _actor(state):
    return hp.actor_slots(pe.State.from_string(state).side_one)


def test_slot_roundtrip_and_tera_and_switch():
    moves, bench = ["a", "b", "c", "d"], ["x", "y", "z", "u", "v"]
    for slot in range(hp.N_SLOTS):
        choice = hp.choice_of(slot, moves, bench)
        assert hp.slot_of(choice, moves, bench) == slot
    assert hp.slot_of("b-tera", moves, bench) == 5
    assert hp.slot_of("switch z", moves, bench) == 10
    assert hp.slot_of("nosuchmove", moves, bench) is None
    assert hp.slot_of("switch nobody", moves, bench) is None
    assert hp.choice_of(3, ["a"], bench) is None  # empty move slot
    assert hp.legal_mask(["a", "switch x", "bogus"], moves, bench).tolist() == [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def test_elo_ids():
    assert hp.elo_id("1000-1199") == 0 and hp.elo_id("1800+") == 4
    assert hp.elo_id(1500) == 2 and hp.elo_id(1050) == 0 and hp.elo_id(2300) == 4
    assert hp.elo_id(None) == hp.N_ELO and hp.elo_id("garbage") == hp.N_ELO


def test_fixture_state_actor_slots():
    moves, bench = _actor(STATE)
    assert moves == ["hypervoice", "expandingforce", "dazzlinggleam", "calmmind"]
    assert bench[0] == "clefable"


def test_probs_are_masked_and_normalised():
    policy = _random_policy()
    moves, bench = _actor(STATE)
    choices = ["hypervoice", "calmmind", "switch clefable"]
    feats, mask, _m, _b = hp.encode(STATE, choices)
    assert feats.shape == (hp.N_FEATURES,) and mask.sum() == 3
    p = policy.probs(feats, mask, 0)[0]
    assert p.sum() == pytest.approx(1.0)
    assert p[mask == 0].max() == 0.0
    # an empty mask gives all zeros, not NaN
    assert not np.isnan(policy.probs(feats, np.zeros(hp.N_SLOTS), 0)).any()


def test_predict_distribution_over_given_choices():
    policy = _random_policy()
    dist = policy.predict(STATE, ["hypervoice", "calmmind", "switch clefable", "notamove"], "1400-1599")
    assert set(dist) == {"hypervoice", "calmmind", "switch clefable", "notamove"}
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["notamove"] == 0.0 and dist["hypervoice"] > 0
    # elo conditioning changes the output
    other = policy.predict(STATE, ["hypervoice", "calmmind", "switch clefable"], 1900)
    same = policy.predict(STATE, ["hypervoice", "calmmind", "switch clefable"], 1100)
    assert other != same
    # nothing usable: uniform over the unusable (no crash, normalised)
    odd = policy.predict(STATE, ["bogus1", "bogus2"])
    assert sum(odd.values()) == pytest.approx(1.0)


def test_side_two_prediction_is_the_swapped_actor_view():
    policy = _random_policy()
    swapped = hp.swap_sides(STATE)
    assert hp.swap_sides(swapped) == STATE
    moves2, _bench2 = _actor(swapped)  # what was side two's moves in STATE
    choices = [m for m in moves2]
    a = policy.predict(STATE, choices, side=2)
    b = policy.predict(swapped, choices, side=1)
    assert a == b and sum(a.values()) == pytest.approx(1.0)


def test_save_load_roundtrip(tmp_path):
    policy = _random_policy()
    policy.meta = {"note": "x"}
    path = tmp_path / "m.npz"
    policy.save(path)
    loaded = hp.HumanPolicy.load(path)
    assert loaded.meta == {"note": "x"}
    feats, mask, *_ = hp.encode(STATE, ["hypervoice", "calmmind"])
    assert np.allclose(policy.probs(feats, mask, 1), loaded.probs(feats, mask, 1))


def test_inference_is_fast():
    import time

    policy = _random_policy(h1=1024, h2=512)
    choices = ["hypervoice", "calmmind", "switch clefable"]
    policy.predict_many([STATE] * 4, [choices] * 4, side=2)  # warm up
    start = time.perf_counter()
    for _ in range(5):
        policy.predict_many([STATE] * 4, [choices] * 4, side=2)
    per_call_ms = (time.perf_counter() - start) / 5 * 1000
    assert per_call_ms < 100  # generous; real cost is a few ms for 4 worlds


# --- training pipeline ---------------------------------------------------------------


def _write_shard(path, n=48):
    moves, bench = _actor(STATE)
    legal = ["hypervoice", "hypervoice-tera", "expandingforce", "calmmind", "switch clefable"]
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({
            "battle_tag": f"b{i // 8}", "role": "p1", "turn": i, "elo": hp.ELO_BUCKETS[i % 5],
            "result": "win", "state": STATE,
            "action": legal[int(rng.integers(0, len(legal)))], "legal": legal, "action_legal": i % 11 != 0,
        })
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_stream_train_export_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    from showdown_bot import human_policy_train as tr

    for k in range(2):
        _write_shard(tmp_path / f"shard-{k:05d}.jsonl.gz")
    files = sorted(str(p) for p in tmp_path.glob("shard-*.jsonl.gz"))
    batches = list(tr.BatchStream(files, batch=16, shuffle=True, seed=1))
    x, mask, y, elo, n = batches[0]
    assert x.shape == (16, hp.N_FEATURES) and mask.shape == (16, hp.N_SLOTS)
    assert (n == 5).all()
    assert all(float(mask[i, y[i]]) == 1.0 for i in range(len(y)))  # label is always legal
    total = sum(len(b[2]) for b in batches)
    assert 80 <= total <= 96  # 96 rows minus the action_legal=False ones
    # deterministic order without shuffle
    again = list(tr.BatchStream(files, batch=16, shuffle=False))
    assert torch.equal(again[0][2], list(tr.BatchStream(files, batch=16, shuffle=False))[0][2])

    mu, sd = tr.feature_stats(files, n_shards=2, per_shard=20)
    model = tr.make_model(16, 8, mu, sd)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(25):
        for x, mask, y, elo, _n in tr.BatchStream(files, batch=32, shuffle=True, seed=2):
            loss = torch.nn.functional.cross_entropy(model(x, mask, elo), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    exported = tr.export(model, {"h1": 16})
    x, mask, y, elo, _n = batches[0]
    with torch.no_grad():
        z = torch.log_softmax(model(x, mask, elo), dim=1).numpy()
    p = exported.probs(x.float().numpy(), mask.numpy(), elo.numpy())
    legal = mask.numpy() > 0
    assert np.allclose(np.log(p[legal]), z[legal], atol=1e-3)
    res = tr.evaluate(model, files, 0, torch.device("cpu"))
    assert res["all"]["n"] > 0 and 0 <= res["all"]["top1"] <= 1 and "uni_nll" in res["all"]
