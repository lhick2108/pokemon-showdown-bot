"""SPRT math (src/showdown_bot/sprt.py).

Reference values: the LLR bounds ln(19) = 2.9444 for alpha = beta = 0.05 are the (-2.94, 2.94)
printed by Fishtest / fastchess; elo 21 -> score 0.530185 and elo 190.85 <-> 75% follow from
the logistic formula; the LLR examples are derived by hand from the Fishtest / fastchess
normal approximation (see the comments), not copied from a Fishtest run.
"""

import math

import pytest

from showdown_bot import sprt


def test_bounds_match_fishtest_defaults():
    lo, hi = sprt.sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(-2.9444, abs=1e-4)
    assert hi == pytest.approx(2.9444, abs=1e-4)
    lo, hi = sprt.sprt_bounds(0.05, 0.10)
    assert lo == pytest.approx(math.log(0.10 / 0.95))
    assert hi == pytest.approx(math.log(0.90 / 0.05))


def test_elo_score_roundtrip():
    assert sprt.elo_to_score(0) == 0.5
    assert sprt.elo_to_score(21) == pytest.approx(0.530185, abs=1e-6)
    assert sprt.score_to_elo(0.75) == pytest.approx(400 * math.log10(3), abs=1e-9)
    assert sprt.score_to_elo(sprt.elo_to_score(-35.5)) == pytest.approx(-35.5)
    assert sprt.score_to_elo(1.0) == math.inf


def test_llr_hand_derived():
    # counts [1,2,4,2,1]: N=10, mean 0.5, var = 2*(0.25 + 2*0.0625)/10 = 0.075
    # LLR = 10 * (0.530185-0.5) * (1 - 1.030185) / (2 * 0.075) = -0.060741
    assert sprt.llr([1, 2, 4, 2, 1], 0, 21) == pytest.approx(-0.060741, abs=1e-5)
    # counts [2,10,40,30,18]: N=100, mean 0.63, var 0.0581
    # LLR = 100 * 0.0301847 * (1.26 - 1.0301847) / (2 * 0.0581) = 5.9698
    assert sprt.llr([2, 10, 40, 30, 18], 0, 21) == pytest.approx(5.9698, abs=1e-3)


def test_llr_sign_and_scaling():
    assert sprt.llr([0, 0, 0, 0, 0], 0, 21) == 0.0
    assert sprt.llr([10, 20, 40, 20, 10], 0, 21) < 0  # even score favours H0
    assert sprt.llr([2, 10, 40, 30, 18], 0, 21) > 0
    # doubling every count doubles N and the LLR (same distribution)
    a = sprt.llr([2, 10, 40, 30, 18], 0, 21)
    assert sprt.llr([4, 20, 80, 60, 36], 0, 21) == pytest.approx(2 * a)


def test_llr_zero_counts_are_regularised_not_nan():
    v = sprt.llr([0, 0, 5, 0, 0], 0, 21)
    assert math.isfinite(v)


def test_pair_index():
    assert [sprt.pair_index(p) for p in (0, 0.5, 1, 1.5, 2)] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        sprt.pair_index(0.25)


def test_elo_estimate():
    elo, lo, hi = sprt.elo_estimate([2, 10, 40, 30, 18])
    assert elo == pytest.approx(sprt.score_to_elo(0.63))
    # se = sqrt(0.0581/100) = 0.024104
    assert lo == pytest.approx(sprt.score_to_elo(0.63 - 1.96 * 0.024104), abs=0.05)
    assert lo < elo < hi
    assert math.isnan(sprt.elo_estimate([0, 0, 1, 0, 0])[1])


def test_parse_sprt():
    c = sprt.parse_sprt("0,21")
    assert (c.elo0, c.elo1, c.alpha, c.beta, c.max_games) == (0, 21, 0.05, 0.05, 2000)
    c = sprt.parse_sprt("-1,3,0.1,0.2", max_games=500)
    assert (c.elo0, c.elo1, c.alpha, c.beta, c.max_games) == (-1, 3, 0.1, 0.2, 500)
    for bad in ("5", "1,2,3", "a,b", "5,5", "0,21,2,0.1"):
        with pytest.raises(ValueError):
            sprt.parse_sprt(bad)


def test_decide():
    cfg = sprt.SprtConfig(max_games=400)
    assert sprt.decide([0, 0, 1, 0, 0], cfg) is None  # below min_pairs
    assert sprt.decide([2, 10, 40, 30, 18], cfg) == "H1"  # LLR 5.97 >= 2.94
    assert sprt.decide([30, 40, 60, 40, 30], sprt.SprtConfig(max_games=400)) == "max"  # 200 pairs
    # strongly negative: H0
    assert sprt.decide([30, 30, 30, 10, 0], cfg) == "H0"
