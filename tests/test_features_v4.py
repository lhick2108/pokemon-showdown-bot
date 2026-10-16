"""Feature version 4 (dense v2 ++ v3 plus identity indices) and the embedding value net
(bead showdown-bot-589.65): vocabulary stability, unknown / none handling, export round trip,
older .pev files, warm start and loading through the dataset path."""

from __future__ import annotations

import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from test_recorder import fake_world  # noqa: E402,F401
from test_train import state_strings  # noqa: E402,F401
from test_train_subsample import _write_games  # noqa: E402

from showdown_bot import dataset, train  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(pe, "state_identity"), reason="engine wheel without feature version 4")

# Golden fingerprint of the vocabularies (engine enums in declaration order). If this fails you
# changed an enum: appending variants at the END keeps old nets' meaning only if you also keep
# the hash story straight -- update these constants AND docs/features-v4.md, and retrain v4 nets.
GOLDEN_SIZES = [1471, 260, 324, 887]
GOLDEN_HASH = 0x41BB936119EB13F2


def test_vocab_is_stable():
    assert pe.identity_vocab_sizes() == GOLDEN_SIZES
    assert pe.identity_vocab_hash() == GOLDEN_HASH
    for t, size in enumerate(GOLDEN_SIZES):
        names = pe.identity_vocab(t)
        assert len(names) == size
        assert names[:3] == ["<pad>", "<unknown>", "NONE"]
        assert len(set(names)) == len(names)
    assert pe.identity_vocab(1)[3] == "UNKNOWNITEM"
    assert "LEFTOVERS" in pe.identity_vocab(1) and "TACKLE" in pe.identity_vocab(3)
    assert pe.IDENTITY_LENGTH == 84 and pe.IDENTITY_IDS_PER_SLOT == 7
    assert pe.FEATURE_LENGTH_V4 == 738 + 1453 + 188 == pe.feature_length(4)
    assert 4 in pe.FEATURE_VERSIONS


def test_dense_part_is_v2_then_v3(state_strings):  # noqa: F811
    s = pe.State.from_string(state_strings[0])
    v2, v3, v4 = (pe.state_features(s, 1, v) for v in (2, 3, 4))
    assert len(v4) == pe.FEATURE_LENGTH_V4
    assert v4[:2191] == v2
    assert v4[2191:] == v3[738:]
    names = pe.state_feature_names(4)
    assert len(names) == len(v4) and names[2191] == "match.speed.own0.opp0"
    # side two's view is the mirror image of the same state
    assert pe.state_features(s, 2, 4) != v4


def test_identity_indices(state_strings):  # noqa: F811
    sizes = pe.identity_vocab_sizes()
    for ss in state_strings:
        s = pe.State.from_string(ss)
        for side in (1, 2):
            ids = pe.state_identity(s, side)
            assert len(ids) == 84
            for slot in range(12):
                row = ids[slot * 7:(slot + 1) * 7]
                limits = [sizes[0], sizes[1], sizes[2]] + [sizes[3]] * 4
                assert all(0 <= v < lim for v, lim in zip(row, limits))
                assert all(v == 0 for v in row) or row[0] >= 1  # fainted slots are all pad
        a, b = pe.state_identity(s, 1), pe.state_identity(s, 2)
        assert a[:42] == b[42:] and a[42:] == b[:42]  # perspective swaps the halves
    # the opponent's team in the fixture is only partly known: unknown entries use index 1
    names = pe.identity_vocab(3)
    ids = pe.state_identity(pe.State.from_string(state_strings[0]), 1)
    own_moves = ids[3:7]  # own active slot
    assert all(m >= 2 and names[m] != "NONE" for m in own_moves)


def test_fainted_and_unknown_pokemon(state_strings):  # noqa: F811
    ss = state_strings[0]
    assert ss.startswith("GREATTUSK,80,GROUND,FIGHTING,GROUND,FIGHTING,315,315,")
    base = pe.state_identity(pe.State.from_string(ss), 1)
    assert not any(v == 0 for v in base)  # nobody fainted
    # our first Pokemon faints: its whole slot becomes the pad identity, the rest is unchanged
    dead = ss.replace(",315,315,", ",0,315,", 1)
    ids = pe.state_identity(pe.State.from_string(dead), 1)
    assert base[:7] != ids[:7] and ids[:7] == [0] * 7
    assert ids[7:] == base[7:]
    # item: unknown -> index 1, "no item" is a real entry (2), a named item is >= 3
    i_unknown = pe.state_identity(pe.State.from_string(ss.replace(",LEFTOVERS,", ",UNKNOWNITEM,", 1)), 1)[1]
    i_none = pe.state_identity(pe.State.from_string(ss.replace(",LEFTOVERS,", ",NONE,", 1)), 1)[1]
    assert (i_unknown, i_none) == (1, 2) and base[1] >= 3
    assert pe.identity_vocab(1)[base[1]] == "LEFTOVERS"


def _perturbed(hidden=8, residual=False, dims=(4, 3, 2, 5), seed=5):
    torch.manual_seed(seed)
    model = train.build_model(
        "mlp", pe.FEATURE_LENGTH_V4 + 84, hidden, residual=(0.9, -0.1) if residual else None,
        embed_dims=dims)
    net = model.net if residual else model
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return model


@pytest.mark.parametrize("residual", [False, True])
def test_export_round_trip_matches_torch(tmp_path, state_strings, residual):  # noqa: F811
    model = _perturbed(residual=residual)
    out = tmp_path / "v4.pev"
    train.export_pev(model, out, 4)
    ev = pe.Evaluator.load(str(out))
    assert ev.feature_version == 4 and ev.kind == "mlp" and ev.hidden_size == 8
    assert list(ev.embed_dims) == [4, 3, 2, 5]
    assert ev.residual is residual
    assert out.read_bytes()[4:8] == (3).to_bytes(4, "little")  # format 3
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3
    rec = (tmp_path / "v4.pev.vocab.json").read_text()
    assert f"{GOLDEN_HASH:#018x}" in rec
    parsed = train.read_pev(out)
    assert parsed["embed_dims"] == (4, 3, 2, 5) and parsed["vocab_hash"] == GOLDEN_HASH
    assert [t.shape[0] for t in parsed["tables"]] == GOLDEN_SIZES


def test_wide_net_h128_matches_torch(tmp_path, state_strings):  # noqa: F811
    model = _perturbed(hidden=128, dims=(8, 8, 8, 8))
    out = tmp_path / "w.pev"
    train.export_pev(model, out, 4)
    assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3


def test_unknown_none_and_pad_are_distinct_inputs(tmp_path, state_strings):  # noqa: F811
    """The tables are used: changing only an item identity changes the logit, and unknown,
    none and a named item are three different inputs."""
    model = _perturbed()
    out = tmp_path / "v4.pev"
    train.export_pev(model, out, 4)
    ev = pe.Evaluator.load(str(out))
    ss = state_strings[0]
    z = [ev.evaluate(pe.State.from_string(ss.replace(",LEFTOVERS,", f",{item},", 1)))
         for item in ("LEFTOVERS", "UNKNOWNITEM", "NONE")]
    assert len(set(z)) == 3
    # fainted Pokemon: their species / item / moves do not reach the embeddings
    dead = ss.replace(",315,315,", ",0,315,", 1)
    assert ev.evaluate(pe.State.from_string(dead)) == pytest.approx(
        train.logits_of(model, torch.tensor([dataset.state_row(pe, pe.State.from_string(dead), 1, 4)])).item(),
        abs=1e-3)


def test_old_pev_files_still_load_and_are_identical(tmp_path, state_strings):  # noqa: F811
    for version, n_in in ((1, pe.FEATURE_LENGTH), (2, pe.FEATURE_LENGTH_V2), (3, pe.FEATURE_LENGTH_V3)):
        torch.manual_seed(version)
        model = train.build_model("mlp", n_in, 8, residual=(0.7, 0.0))
        out = tmp_path / f"v{version}.pev"
        train.export_pev(model, out, version)
        raw = out.read_bytes()
        assert raw[4:8] == (2).to_bytes(4, "little")  # still format 2 (residual), not 3
        ev = pe.Evaluator.load(str(out))
        assert ev.feature_version == version and ev.embed_dims is None
        assert train.verify_export(model, out, state_strings, tol=1e-3) < 1e-3
    # a plain v1 file is format 1 as before
    plain = train.build_model("mlp", pe.FEATURE_LENGTH, 4)
    out = tmp_path / "plain.pev"
    train.export_pev(plain, out, 1)
    assert out.read_bytes()[4:8] == (1).to_bytes(4, "little")
    assert pe.Evaluator.load(str(out)).embed_dims is None


def test_vocab_mismatch_is_rejected(tmp_path):
    out = tmp_path / "v4.pev"
    train.export_pev(_perturbed(), out, 4)
    raw = bytearray(out.read_bytes())
    raw[32] ^= 0xFF  # vocabulary hash
    bad = tmp_path / "bad.pev"
    bad.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="vocabulary"):
        pe.Evaluator.load(str(bad))


@pytest.mark.parametrize("residual", [False, True])
def test_init_round_trip_v4(tmp_path, state_strings, residual):  # noqa: F811
    src = _perturbed(residual=residual)
    path = tmp_path / "src.pev"
    train.export_pev(src, path, 4)
    n_feat = pe.FEATURE_LENGTH_V4 + 84
    fresh = train.build_model("mlp", n_feat, 8, residual=(0.3, 0.0) if residual else None,
                              embed_dims=(4, 3, 2, 5))
    train.load_init(fresh, path, model_kind="mlp", n_feat=n_feat, hidden=8, feature_version=4,
                    residual=residual)
    again = tmp_path / "again.pev"
    train.export_pev(fresh, again, 4)
    assert again.read_bytes() == path.read_bytes()
    # mismatches name the problem
    wrong = train.build_model("mlp", n_feat, 8, residual=(0.3, 0.0) if residual else None,
                              embed_dims=(8, 8, 8, 8))
    with pytest.raises(ValueError, match="embed dims"):
        train.load_init(wrong, path, model_kind="mlp", n_feat=n_feat, hidden=8, feature_version=4,
                        residual=residual)
    plain = train.build_model("mlp", pe.FEATURE_LENGTH_V4, 8, residual=(0.3, 0.0) if residual else None)
    with pytest.raises(ValueError, match="identity embeddings"):
        train.load_init(plain, path, model_kind="mlp", n_feat=pe.FEATURE_LENGTH_V4, hidden=8,
                        feature_version=4, residual=residual)


def test_parse_embed_dims():
    assert train.parse_embed_dims("8") == (8, 8, 8, 8)
    assert train.parse_embed_dims(6) == (6, 6, 6, 6)
    assert train.parse_embed_dims("16,8,4,12") == (16, 8, 4, 12)
    for bad in ("0", "1,2", "a", "8,8,8,2000"):
        with pytest.raises(ValueError):
            train.parse_embed_dims(bad)
    a = train._parser().parse_args(["--data", "d", "--out", "o.pev", "--features", "v4",
                                    "--embed-dim", "16,8,8,8"])
    assert a.features == "v4" and a.embed_dim == "16,8,8,8"


def test_dataset_and_load_data_v4(tmp_path, state_strings):  # noqa: F811
    _write_games(tmp_path / "g", state_strings, n_games=3, n_dec=4, searched=True, prefix="s")
    samples = list(dataset.iter_samples(tmp_path / "g", feature_version=4))
    assert samples and all(len(s.features) == pe.FEATURE_LENGTH_V4 + 84 for s in samples)
    data = train.load_data([tmp_path / "g"], states="both", feature_version=4, x_dtype="float16",
                           workers=1)
    assert data.x.shape[1] == pe.FEATURE_LENGTH_V4 + 84 and data.x.dtype.name == "float16"
    ids = data.x[:, pe.FEATURE_LENGTH_V4:]
    assert ids.max() < max(GOLDEN_SIZES)
    assert (ids.astype("int64") == ids).all()  # float16 holds the indices exactly
    # v1 is untouched
    v1 = train.load_data([tmp_path / "g"], states="both", feature_version=1, workers=1)
    assert v1.x.shape[1] == pe.FEATURE_LENGTH


def test_cli_smoke_train_v4(tmp_path, state_strings, capsys):  # noqa: F811
    root = tmp_path / "games"
    _write_games(root, state_strings, n_games=12, n_dec=4, searched=True, prefix="s")
    out = tmp_path / "m.pev"
    args = ["--data", str(root), "--out", str(out), "--features", "v4", "--embed-dim", "4",
            "--hidden", "8", "--epochs", "2", "--device", "cpu", "--load-workers", "1",
            "--val-fraction", "0.25"]
    assert train.main(args) == 0
    cap = capsys.readouterr()
    assert "exported" in cap.out + cap.err
    ev = pe.Evaluator.load(str(out))
    assert ev.feature_version == 4 and list(ev.embed_dims) == [4, 4, 4, 4]
    # residual + search target + warm start from that file
    out2 = tmp_path / "m2.pev"
    res = tmp_path / "res.pev"
    assert train.main(args[:4] + ["--features", "v4", "--embed-dim", "4", "--hidden", "8", "--epochs", "1",
                                  "--device", "cpu", "--load-workers", "1", "--val-fraction", "0.25",
                                  "--residual", "--value-target", "search", "--out", str(res)]) == 0
    assert train.main(args[:4] + ["--features", "v4", "--embed-dim", "4", "--hidden", "8", "--epochs", "1",
                                  "--device", "cpu", "--load-workers", "1", "--val-fraction", "0.25",
                                  "--residual", "--value-target", "search", "--init", str(res),
                                  "--lr", "1e-4", "--out", str(out2)]) == 0
    assert pe.Evaluator.load(str(out2)).residual
