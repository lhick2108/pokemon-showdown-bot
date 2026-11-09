"""Fixed-team designs: format mapping, schedules, team generation, paired statistics."""

import math

import pytest

from showdown_bot import sets, teams
from showdown_bot.league import Game, paired_head_to_head


def test_fixed_format_maps_to_randbats_data(monkeypatch):
    seen = []

    def fake_read(path, url, refresh, max_age=None):
        seen.append(url)
        if "full" in url:
            raise OSError("no full file")
        return {}

    monkeypatch.setattr(sets, "_read_cached_json", fake_read)
    try:
        sets.RandbatsData.load(teams.FIXED_FORMAT, cache_dir="unused")
    except Exception:
        pass  # an empty payload may not parse; only the requested urls matter
    assert seen and all("gen9randombattlefixed" not in u for u in seen)
    assert any(u.endswith("/gen9randombattle.json") for u in seen)
    assert teams.data_format(teams.FIXED_FORMAT) == "gen9randombattle"
    assert teams.data_format("gen9randombattle") == "gen9randombattle"


def test_battle_format_for():
    assert teams.battle_format_for("random", "gen9randombattle") == "gen9randombattle"
    assert teams.battle_format_for("swap", "gen9randombattle") == "gen9randombattlefixed"
    assert teams.battle_format_for("mirror", "gen9randombattlefixed") == "gen9randombattlefixed"
    with pytest.raises(ValueError):
        teams.battle_format_for("swap", "gen8randombattle")
    with pytest.raises(ValueError):
        teams.battle_format_for("bogus", "gen9randombattle")


FAKE = [f"T{i}" for i in range(20)]


def test_swap_plan_is_duplicate_bridge():
    plans = teams.plan_games("swap", 6, 0, FAKE)
    assert len(plans) == 6
    for k in range(3):
        g1, g2 = plans[2 * k], plans[2 * k + 1]
        assert g1.pair == g2.pair == k
        assert (g1.a_team, g1.b_team) == (FAKE[2 * k], FAKE[2 * k + 1])
        assert (g2.a_team, g2.b_team) == (g1.b_team, g1.a_team)  # teams swap between the bots
        assert g1.a_is_p1 and not g2.a_is_p1  # and the seats swap too
        # the same team always sits in the same seat
        assert (g1.p1_team_id, g1.p2_team_id) == (g2.p1_team_id, g2.p2_team_id)
    assert len({p.a_team_id for p in plans}) == 6  # fresh teams for every pair


def test_mirror_plan_gives_both_the_same_team():
    plans = teams.plan_games("mirror", 4, 0, FAKE)
    for p in plans:
        assert p.a_team == p.b_team == FAKE[p.pair]
        assert p.a_team_id == p.b_team_id == p.pair
    assert [p.a_is_p1 for p in plans] == [True, False, True, False]


def test_plan_rejects_odd_and_unknown():
    with pytest.raises(ValueError):
        teams.plan_games("swap", 5, 0, FAKE)
    with pytest.raises(ValueError):
        teams.plan_games("random", 4, 0, FAKE)
    with pytest.raises(ValueError):
        teams.plan_games("swap", 4, 0, FAKE[:3])


def test_split_pairs_keeps_pairs_together():
    assert teams.split_pairs(20, 4) == [6, 6, 4, 4]
    assert teams.split_pairs(4, 8) == [2, 2]
    assert all(s % 2 == 0 for s in teams.split_pairs(30, 4))
    assert sum(teams.split_pairs(30, 4)) == 30
    with pytest.raises(ValueError):
        teams.split_pairs(5, 2)
    plans = teams.plan_games("swap", 10, 0, FAKE)
    shares = teams.split_pairs(10, 2)
    got = [teams.slice_plans(plans, s, st) for s, st in zip(shares, [0, shares[0]])]
    assert [p for sh in got for p in sh] == plans


def test_packed_team_is_verbatim():
    packed = "Brute Bonnet||Leftovers|Protosynthesis|spore||85,85,85,85,85,85|N|||80|,,,,,Poison"
    assert teams.PackedTeam(packed).yield_team() == packed


def test_generate_teams_is_seeded_and_reproducible():
    try:
        a = teams.generate_teams(3, 11)
    except (RuntimeError, FileNotFoundError):
        pytest.skip("no built Showdown checkout (set SHOWDOWN_BOT_SERVER_DIR)")
    assert a == teams.generate_teams(3, 11)
    assert a != teams.generate_teams(3, 12)
    assert len(set(a)) == 3
    assert teams.generate_teams(2, 11) == a[:2]  # team i depends only on (seed, i)
    assert all(t.count("]") == 5 for t in a)  # six Pokemon, packed


def test_paired_summary_known_values():
    s = teams.paired_summary([1.0, 0.5, 0.0, 0.5, 1.0, 0.5])
    assert s["pairs"] == 6 and s["mean"] == pytest.approx(0.583333, abs=1e-5)
    var = sum((x - s["mean"]) ** 2 for x in [1.0, 0.5, 0.0, 0.5, 1.0, 0.5]) / 5
    assert s["se"] == pytest.approx(math.sqrt(var / 6))
    # t(0.975, df=5) = 2.5706
    assert s["ci_high"] - s["mean"] == pytest.approx(2.5706 * s["se"], rel=1e-3)
    assert s["ci_low"] < s["mean"] < s["ci_high"]


def test_t_quantile_matches_tables():
    assert teams._t_quantile(0.975, 2) == pytest.approx(4.303, rel=1e-3)
    assert teams._t_quantile(0.975, 10) == pytest.approx(2.228, rel=1e-3)
    assert teams._t_quantile(0.975, 1) == pytest.approx(12.706, rel=1e-3)
    assert teams._t_quantile(0.975, 30) == pytest.approx(2.042, rel=1e-3)


def test_paired_summary_variance_ratio_shows_benefit():
    # every pair splits 1-1: the pairs carry no team luck, so the paired variance is zero
    flat = teams.paired_summary([0.5] * 10)
    assert flat["se"] == 0 and flat["variance_ratio"] == 0
    # pairs that are all-or-nothing behave like a coin per pair: ratio ~ 2 vs binomial over games
    swingy = teams.paired_summary([1.0, 0.0] * 5)
    assert swingy["variance_ratio"] > 1


def test_paired_summary_degenerate():
    assert teams.paired_summary([])["pairs"] == 0
    one = teams.paired_summary([1.0])
    assert math.isnan(one["ci_low"]) and one["mean"] == 1.0


def test_pair_scores_drops_incomplete_pairs():
    rows = [(0, 1.0), (0, 0.0), (1, 1.0), (1, 1.0), (2, 0.5)]
    assert teams.pair_scores(rows) == [0.5, 1.0]


def _game(tag, p1, p2, winner, pair, ids=(0, 1)):
    return Game(tag, p1, p2, winner, 10, team_mode="swap", team_pair=pair, team_ids=ids)


def test_game_roundtrip_keeps_team_fields_and_old_lines_load():
    g = _game("t1", "a", "b", "a", 3, (6, 7))
    d = g.to_dict()
    assert d["team_mode"] == "swap" and d["team_pair"] == 3 and d["team_ids"] == [6, 7]
    assert Game.from_dict(d) == g
    plain = Game("t2", "a", "b", "b", 5).to_dict()
    assert "team_pair" not in plain  # random games keep the old schema
    back = Game.from_dict(plain)
    assert back.team_pair is None and back.team_mode is None


def test_league_paired_head_to_head_uses_pairs_as_units():
    games = [
        _game("1", "a", "b", "a", 0), _game("2", "b", "a", "b", 0),   # a wins 1, loses 1: 0.5
        _game("3", "a", "b", "a", 1), _game("4", "b", "a", "a", 1),   # a wins both: 1
        _game("5", "a", "b", "a", 2), _game("6", "b", "a", "b", 2),   # split: 0.5
        Game("7", "a", "b", "a", 5),                                  # random game: ignored
        _game("8", "a", "b", "a", 3),                                 # incomplete pair: dropped
    ]
    out = paired_head_to_head(games, ["a", "b"])
    s = out[("a", "b")]
    assert s["pairs"] == 3
    assert s["mean"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert out[("b", "a")]["mean"] == pytest.approx(1 - s["mean"])
