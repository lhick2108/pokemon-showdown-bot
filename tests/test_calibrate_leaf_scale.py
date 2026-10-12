from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_leaf_scale", Path(__file__).resolve().parents[1] / "scripts" / "calibrate_leaf_scale.py"
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


def test_suggest_scale_recovers_known_ratio():
    net = [0.1, -0.2, 0.05, 0.3, -0.15, 0.0, 0.25]
    stock = [4.0 * x for x in net]
    r = cal.suggest_scale(stock, net)
    assert r["k_std"] == pytest.approx(4.0)
    assert r["k_mad"] == pytest.approx(4.0)
    assert r["n"] == len(net)


def test_suggest_scale_zero_net_spread_is_nan_and_bad_input_raises():
    r = cal.suggest_scale([1.0, -1.0, 2.0], [0.0, 0.0, 0.0])
    assert math.isnan(r["k_std"]) and math.isnan(r["k_mad"])
    with pytest.raises(ValueError):
        cal.suggest_scale([1.0], [1.0])
    with pytest.raises(ValueError):
        cal.suggest_scale([1.0, 2.0], [1.0])
