"""Transformer tokens (bead showdown-bot-589.68.1): engine ``state_tokens_str`` layout, determinism,
own/opp symmetry, vocabulary hash shared with v4, action slots shared with ``policy_actions``,
and a round trip through the shard writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
pytestmark = pytest.mark.skipif(not hasattr(pe, "state_tokens_str"), reason="engine wheel without state_tokens")

from showdown_bot import recorder, tokens  # noqa: E402

STATES = Path(__file__).resolve().parents[1] / "engine" / "data" / "gen9randombattle.txt"
V4_HASH = 0x41BB936119EB13F2


@pytest.fixture(scope="module")
def states() -> list[str]:
    return [line.strip() for line in STATES.read_text().splitlines() if line.strip()][:12]


def tok(state: str, side: int = 1, reveal=None):
    cat, cont, names = pe.state_tokens_str(state, side, reveal, True)
    return tokens.split_buffers(cat, cont), names


def test_spec_and_layout_are_consistent():
    spec = tokens.token_spec()
    c = spec["constants"]
    assert spec["vocab_hash"] == f"{V4_HASH:#018x}" == f"{pe.identity_vocab_hash():#018x}"
    assert list(spec["vocab_sizes"].values()) == pe.identity_vocab_sizes()
    assert spec["fields"]["mon_cont"]["shape"] == [12, c["C_MON"]]
    for buf, total in (("cat", c["CAT_LEN"]), ("cont", c["CONT_LEN"])):
        fields = sorted((f["offset"], int(np.prod(f["shape"]))) for f in spec["fields"].values() if f["buffer"] == buf)
        pos = 0
        for off, n in fields:  # contiguous, no gaps or overlaps
            assert off == pos
            pos += n
        assert pos == total
    # shard fields cover every engine field
    assert set(spec["fields"]) <= set(tokens._shard_fields())


def test_shapes_dtypes_finite_and_deterministic(states):
    for s in states:
        for side in (1, 2):
            a, names_a = tok(s, side)
            b, names_b = tok(s, side)
            assert names_a == names_b and len(names_a) == 26
            for k in a:
                assert np.array_equal(a[k], b[k])
            assert a["mon_cat"].shape == (12, 7) and a["mon_cat"].dtype == np.int16
            assert a["mon_cont"].dtype == np.float32 and np.isfinite(a["mon_cont"]).all()
            assert np.isfinite(a["act_cont"]).all() and np.abs(a["mon_cont"]).max() < 65000
    st = pe.State.from_string(states[0])
    cat, cont, _ = pe.state_tokens(st, 1)
    ref, _ = tok(states[0], 1)
    got = tokens.split_buffers(cat, cont)
    assert all(np.array_equal(got[k], ref[k]) for k in ref)


def test_ids_are_the_v4_identity(states):
    for s in states:
        st = pe.State.from_string(s)
        for side in (1, 2):
            t, _ = tok(s, side)
            assert t["mon_cat"].reshape(-1).tolist() == pe.state_identity(st, side)


def test_own_opp_symmetry(states):
    for s in states:
        a, _ = tok(s, 1)
        b, _ = tok(s, 2)
        assert np.array_equal(a["mon_cat"][:6], b["mon_cat"][6:]) and np.array_equal(a["mon_cat"][6:], b["mon_cat"][:6])
        assert np.array_equal(a["mon_known"][:6], b["mon_known"][6:])
        assert np.array_equal(a["act_cat"][0], b["act_cat"][1]) and np.array_equal(a["act_legal"][0], b["act_legal"][1])
        assert np.array_equal(a["act_cont"][0], b["act_cont"][1])
        n = 173 - 2  # everything but the [is_opp, is_active] flags
        assert np.array_equal(a["mon_cont"][:6, :n], b["mon_cont"][6:, :n])
        assert np.array_equal(a["mon_cont"][6:, :n], b["mon_cont"][:6, :n])
        # the v2 speed-ratio features are computed from the other direction: equal up to rounding
        assert np.allclose(a["side_cont"][0], b["side_cont"][1], atol=1e-5)
        assert np.allclose(a["side_cont"][1], b["side_cont"][0], atol=1e-5)
        assert np.array_equal(a["glob"], b["glob"])
        assert a["mon_cont"][:6, -2].sum() == 0 and a["mon_cont"][6:, -2].min() == 1  # is_opp flag
        assert a["mon_cont"][0, -1] == 1 and a["mon_cont"][6, -1] == 1  # active flag


def test_action_tokens_use_policy_actions_slots(states):
    for s in states:
        st = pe.State.from_string(s)
        for side in (1, 2):
            t, names = tok(s, side)
            side_no = 0
            rows = pe.policy_action_features(st, side)
            legal_slots = sorted(slot for _c, slot, _f in rows if slot >= 0)
            assert np.flatnonzero(t["act_legal"][side_no]).tolist() == legal_slots
            for choice, slot, feats in rows:
                if slot >= 0:
                    assert names[slot] == choice
                    assert np.allclose(t["act_cont"][side_no, slot], feats, atol=1e-6)
            assert [n for n in names[:13] if n] == [c for c, sl in sorted(pe.policy_actions(st, side), key=lambda r: r[1]) if sl >= 0]


def test_reveal_scrub_hides_unrevealed_opponent_info(states):
    s = states[0]
    full, _ = tok(s, 1)
    opp_active = pe.State.from_string(s).side_two
    species = str(opp_active.pokemon[int(str(opp_active.active_index)[-1])].id)
    hidden, _ = tok(s, 1, [(species, [], False, False, False)])
    assert hidden["meta"][2] == 1 and full["meta"][2] == 0
    assert hidden["mon_known"][6].tolist()[:3] == [1, 0, 0]
    assert (hidden["mon_known"][7:, 0] == 0).all()
    assert hidden["mon_cat"][6, 1] == 1 and hidden["mon_cat"][6, 3:].tolist() == [1, 1, 1, 1]  # unknown item/moves
    assert (hidden["mon_cat"][7:] == 1).all()  # never-revealed bench: all unknown
    assert hidden["mon_cat"][6, 0] == full["mon_cat"][6, 0]  # the active species is public
    assert np.array_equal(hidden["mon_cat"][:6], full["mon_cat"][:6])  # own side untouched
    assert (hidden["match_pair"].reshape(6, 6, 5)[:, 1:] == 0).all()
    # revealing a move keeps it and reports it as known
    mv = pe.identity_vocab(3)[int(full["mon_cat"][6, 4])]
    one, _ = tok(s, 1, [(species, [mv.lower()], False, False, False)])
    assert one["mon_cat"][6, 3] == full["mon_cat"][6, 4] and one["mon_known"][6, 3] == 1


def test_hidden_targets_follow_species(states):
    s = states[0]
    h = np.frombuffer(pe.state_hidden_targets(s, s, 1), dtype="<i2")
    ids = h[:48].reshape(6, 8)
    mask = h[48:54]
    t, _ = tok(s, 1)
    assert mask.tolist() == [1] * 6
    alive = t["mon_cat"][6:, 0] > 0  # a fainted Pokemon is pad in the input, its species stays a target
    assert (ids[alive, 0] == t["mon_cat"][6:, 0][alive]).all()
    assert (np.diff(ids[:, 3:7], axis=1) >= 0).all()  # moves sorted


# ---------------------------------------------------------------------------
# shard round trip
# ---------------------------------------------------------------------------


def _write_battle(root, states, tag):
    rec = recorder.GameRecorder(root)
    users = ("alice", "bob")
    for u in users:
        rec.start_game(tag, u, role="p1", battle_format="gen9randombattle", config={})
    for turn, s in enumerate(states[:4], start=1):
        st = pe.State.from_string(s)
        opts = [c for c, slot in pe.policy_actions(st, 1) if slot >= 0]
        side2 = st.side_two
        species = str(side2.pokemon[int(str(side2.active_index)[-1])].id)
        facts = [{"species": species, "moves": [], "item": None, "ability": None, "tera": None}]
        for u in users:
            rec.add_decision(tag, u, {
                "turn": turn, "side": "one", "policy": {c: 1.0 / len(opts) for c in opts}, "chosen": opts[0],
                "worlds": [{"state": s, "total_visits": 10, "avg_score": 0.5, "search_value": 0.6,
                            "moves": [{"choice": c, "visits": 1, "score": 0.5} for c in opts]}],
                "truth": {"state": s}, "truth_status": "ok", "opp_facts": facts,
            })
    for u, r in zip(users, (recorder.RESULT_WIN, recorder.RESULT_LOSS)):
        rec.end_game(tag, u, r, turns=6)


def test_export_round_trip(tmp_path, states):
    src = tmp_path / "sp"
    for i in range(3):
        _write_battle(src, states[i : i + 4] if len(states) >= i + 4 else states, f"battle-gen9randombattle-{i}")
    out = tmp_path / "out"
    man = tokens.export([("sp", "selfplay", src)], out, workers=1, games_per_shard=2,
                        opts=tokens.Options(val_permille=500))
    assert man["spec"]["vocab_hash"] == f"{V4_HASH:#018x}"
    ds = tokens.TokenDataset(out)
    assert len(ds) == man["n_samples"] == 6 * 4  # 3 battles x 2 recordings x 4 decisions
    fields = tokens._shard_fields()
    seen_hash: dict[int, set[int]] = {}
    for sh in ds.shards:
        a = ds.load(sh)
        assert set(a) == set(fields)
        for k, (dt, shp) in fields.items():
            assert a[k].dtype == np.dtype(dt) and a[k].shape == (sh["n"], *shp), k
            assert isinstance(a[k], np.memmap)
        assert (a["pol_valid"] == 1).all() and np.allclose(a["pol_target"].sum(1), 1.0)
        assert (a["pol_target"] * (1 - a["act_legal"][:, 0])).sum() == 0
        assert (a["v_search_valid"] == 1).all() and np.allclose(a["v_search"], 0.6)
        assert set(np.unique(a["z"])) <= {0.0, 1.0}
        assert (a["hid_valid"] == 1).all() and (a["hid_mask"] == 1).all()
        assert (a["opp_act_valid"] == 1).all() and (a["opp_act_kind"] >= 0).all()
        assert (a["reveal_src"] == 1).all() and (a["mon_known"][:, 6:, 0].sum(1) >= 1).all()
        # history: first decision of a game has none, later ones see the previous decisions
        first = a["decision_idx"] == 0
        assert (a["hist_cat"][first][:, :, 0] == 0).all()
        assert (a["hist_cat"][~first][:, 0, 0] == 1).all() and (a["hist_cat"][~first][:, 1, 0] == 2).all()
        for h, sp in zip(a["battle_hash"], a["split"]):
            seen_hash.setdefault(int(h), set()).add(int(sp))
        # the same input regenerates the same row through the engine
    assert all(len(v) == 1 for v in seen_hash.values()), "a battle must live in exactly one split"
    # reading twice is deterministic: re-export gives identical shards
    out2 = tmp_path / "out2"
    tokens.export([("sp", "selfplay", src)], out2, workers=1, games_per_shard=2, opts=tokens.Options(val_permille=500))
    for sh in ds.shards:
        a, b = ds.load(sh), tokens.TokenDataset(out2).load(sh)
        assert all(np.array_equal(a[k], b[k]) for k in a)


def test_battle_hash_is_stable():
    assert tokens.battle_hash("battle-x-1") == tokens.battle_hash("battle-x-1")
    assert 0 <= tokens.battle_hash("battle-x-1") < 2**63
