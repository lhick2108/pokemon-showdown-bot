"""Auxiliary value-net targets: slot order, masking, end-of-game derivation, training."""

from __future__ import annotations

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")
torch = pytest.importorskip("torch")

from showdown_bot import auxtargets as aux  # noqa: E402
from showdown_bot import train  # noqa: E402

_B = pe.State().side_one


def _side(ids, hps, active):
    return pe.Side(
        pokemon=[pe.Pokemon(id=i, hp=h, maxhp=100) for i, h in zip(ids, hps)],
        side_conditions=_B.side_conditions, active_index=str(active),
        volatile_status_durations=_B.volatile_status_durations, wish=_B.wish,
        future_sight=_B.future_sight, last_used_move=_B.last_used_move,
        switch_out_move_second_saved_move=_B.switch_out_move_second_saved_move,
    )


def _state(own, opp, own_active=0, opp_active=0):
    return pe.State(side_one=_side(*zip(*own), own_active), side_two=_side(*zip(*opp), opp_active))


OWN = list(zip("ABCDEF", [10, 0, 30, 40, 50, 60]))
OPP = list(zip("abcdef", [100, 100, 0, 100, 100, 100]))


def test_slot_order_matches_feature_order():
    st = _state(OWN, OPP, own_active=3, opp_active=1)
    assert aux.slot_ids(st.side_one) == ["D", "A", "B", "C", "E", "F"]
    assert aux.slot_ids(st.side_two) == ["b", "a", "c", "d", "e", "f"]
    x = pe.state_features(st, 1)
    own_hp = [x[53 * k] for k in range(6)]
    opp_hp = [x[363 + 53 * k] for k in range(6)]
    assert np.allclose(own_hp, np.array(aux.slot_hp(st.side_one)) / 100, atol=1e-6)
    assert np.allclose(opp_hp, np.array(aux.slot_hp(st.side_two)) / 100, atol=1e-6)


def test_win_survival_diff_and_masks():
    st = _state(OWN, OPP, own_active=3)
    # last stored state: we won with A and E alive; theirs all fainted
    ref = _state(list(zip("ABCDEF", [5, 0, 0, 0, 50, 0])), list(zip("abcdef", [0] * 6)))
    y, m = aux.aux_targets(st, ref, "win")
    # slots: D A B C E F (B already fainted -> masked)
    assert m.tolist() == [1, 1, 0, 1, 1, 1] + [1, 1, 0, 1, 1, 1] + [1]
    assert y[:6].tolist() == [0, 1, 0, 0, 1, 0]       # D fainted later; A, E survive
    assert y[6:12].tolist() == [0] * 6                 # the loser has nobody left
    assert y[12] == pytest.approx(2 / 6)


def test_loss_and_tie():
    st = _state(OWN, OPP)
    ref = _state(list(zip("ABCDEF", [0] * 6)), list(zip("abcdef", [0, 0, 0, 7, 0, 9])))
    y, m = aux.aux_targets(st, ref, "loss")
    assert y[:6].tolist() == [0] * 6
    # opp slots a b c d e f; c already fainted (masked); d, f survive
    assert m[6:12].tolist() == [1, 1, 0, 1, 1, 1]
    assert y[6:12].tolist() == [0, 0, 0, 1, 0, 1]
    assert y[12] == pytest.approx(-2 / 6)
    y, m = aux.aux_targets(st, ref, "tie")
    assert not m.any()


def test_unknown_species_is_masked():
    st = _state(OWN, OPP)
    ref = _state(OWN, list(zip("abcdeX", [1] * 6)))  # their sixth was a different (sampled) mon
    _, m = aux.aux_targets(st, ref, "win")
    assert m[11] == 0 and m[10] == 1


def test_aux_loss_and_forward_shapes():
    torch.manual_seed(0)
    net = train.build_model("mlp", 5, 4, residual=(1.0, 0.0))
    train.add_aux_head(net, 4)
    xe = torch.randn(7, 6)
    z, a = train.forward_with_aux(net, xe)
    assert torch.allclose(z, net(xe).squeeze(-1), atol=1e-6)
    assert a.shape == (7, aux.N_AUX)
    y = torch.zeros(7, aux.N_AUX)
    m = torch.ones(7, aux.N_AUX)
    loss = train.aux_loss(a, y, m, torch.ones(7))
    loss.backward()
    assert net.aux_head.weight.grad is not None
    # fully masked rows contribute nothing
    assert train.aux_loss(a, y, torch.zeros_like(m), torch.ones(7)).item() == 0.0


def test_aux_export_is_value_head_only(tmp_path):
    net = train.build_model("mlp", 738, 8, residual=(1.0, 0.0))
    plain = tmp_path / "a.pev"
    train.export_pev(net, plain)
    train.add_aux_head(net, 8)
    withaux = tmp_path / "b.pev"
    train.export_pev(net, withaux)
    assert plain.read_bytes() == withaux.read_bytes()
