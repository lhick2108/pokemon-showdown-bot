import asyncio
import json
import math
import random
from collections import Counter

import pytest

from showdown_bot import ab_stats, ladder, ladder_ab


@pytest.fixture
def pev(tmp_path):
    a, b = tmp_path / "a.pev", tmp_path / "b.pev"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    return a, b


def test_parse_arm_short_and_long(pev):
    a, b = pev
    arm = ladder_ab.parse_arm(f"gen2={a}:scale=1.5")
    assert arm.name == "gen2" and arm.scale == 1.5 and arm.evaluator == str(a.resolve())
    arm2 = ladder_ab.parse_arm(f"v2s=search-net:evaluator={b}:scale=2.0:time_ms=200")
    assert arm2.scale == 2.0 and arm2.config.search_time_ms == 200
    # arms do not inherit from each other
    assert arm.config.search_time_ms == 100


def test_parse_arms_errors(pev):
    a, b = pev
    with pytest.raises(ValueError, match="at least two"):
        ladder_ab.parse_arms([f"x={a}"])
    with pytest.raises(ValueError, match="unique"):
        ladder_ab.parse_arms([f"x={a}", f"x={b}"])
    with pytest.raises(ValueError):
        ladder_ab.parse_arm(f"x={a}:bogus=1")
    with pytest.raises(ValueError, match="not found"):
        ladder_ab.parse_arm("x=missing/none.pev:scale=1")


def test_scheduler_balanced_and_seeded():
    s = ladder_ab.ArmScheduler(["a", "b"], seed=3)
    seq = [s.next() for _ in range(200)]
    for n in range(1, 201):
        c = Counter(seq[:n])
        assert abs(c["a"] - c["b"]) <= 2
    assert Counter(seq) == {"a": 100, "b": 100}
    s2 = ladder_ab.ArmScheduler(["a", "b"], seed=3)
    assert [s2.next() for _ in range(200)] == seq
    s3 = ladder_ab.ArmScheduler(["a", "b"], seed=4)
    assert seq != [s3.next() for _ in range(200)]


def test_assignment_fixed_within_battle_and_switches_config(pev):
    arms = ladder_ab.parse_arms([f"a={pev[0]}:scale=1.5", f"b={pev[1]}:scale=2"])
    ctl = ladder_ab.ArmController(arms, seed=1)
    seen = []

    class Base:
        config = None

        async def choose_move(self, battle):
            seen.append((battle.battle_tag, self.config.evaluator_scale))
            return "m"

    class B:
        def __init__(self, tag):
            self.battle_tag = tag

    p = ladder_ab.with_arm_switch(Base, ctl)()
    first = ctl.prepare_next()
    assert ctl.prepare_next() is first  # idempotent until the battle starts
    for _ in range(3):  # three decisions of battle 1
        asyncio.run(p.choose_move(B("battle-1")))
    assert set(seen) == {("battle-1", first.scale)}
    assert ctl.get("battle-1") is first and ctl.pending is None
    # a second battle draws the next arm from the scheduler; battle 1 stays put
    ctl.prepare_next()
    asyncio.run(p.choose_move(B("battle-2")))
    asyncio.run(p.choose_move(B("battle-1")))
    assert ctl.get("battle-1") is first
    assert seen[-1] == ("battle-1", first.scale)
    counts = Counter(ctl.get(t).name for t in ("battle-1", "battle-2"))
    for i in range(3, 203):
        counts[ctl.arm_for(f"battle-{i}").name] += 1
    assert abs(counts["a"] - counts["b"]) <= 2


def test_ab_record_fields(pev, tmp_path):
    arms = ladder_ab.parse_arms([f"a={pev[0]}:scale=1.5", f"b={pev[1]}"])
    game = ladder.LadderGame("battle-gen9randombattle-1", "Opp", "win", 21, 1500, 1512, 1488)
    rec = ladder_ab.ab_record(game, arms[0], "Me-bot", now=1000.0)
    for key in ("battle_tag", "arm", "evaluator", "scale", "username", "rating_before",
                "rating_after", "opponent", "opponent_rating", "result", "turns", "timestamp"):
        assert key in rec
    assert (rec["arm"], rec["scale"], rec["opponent_rating"], rec["result"], rec["turns"]) == (
        "a", 1.5, 1488, "win", 21)
    path = tmp_path / "d" / "ab.jsonl"
    ladder_ab.append_record(path, rec)
    ladder_ab.append_record(path, rec)
    rows = ab_stats.read_records(path)
    assert len(rows) == 2 and rows[0] == json.loads(json.dumps(rec))
    assert ladder.LadderGame("t", "o", "win", 1, arm="a").to_dict()["arm"] == "a"


def test_report_math_known_rates():
    recs = [{"arm": "a", "result": "win" if i < 40 else "loss", "opponent_rating": 1500,
             "rating_before": 1500} for i in range(100)]
    recs += [{"arm": "b", "result": "win" if i < 60 else "loss", "opponent_rating": 1500,
              "rating_before": 1500} for i in range(100)]
    s = ab_stats.arm_summary(recs)
    assert s["a"]["wins"] == 40 and s["b"]["rate"] == 0.6
    assert s["b"]["ci"][0] == pytest.approx(0.5020, abs=2e-3)
    assert s["b"]["ci"][1] == pytest.approx(0.6906, abs=2e-3)
    d, lo, hi = ab_stats.diff_ci(40, 100, 60, 100)
    se = math.sqrt(2 * 0.24 / 100)
    assert (d, lo, hi) == pytest.approx((0.2, 0.2 - 1.96 * se, 0.2 + 1.96 * se))
    assert lo > 0
    assert ab_stats.games_needed(0.4, 0.6) == pytest.approx(1.96**2 * 0.48 / 0.04)
    e = ab_stats.adjusted_effects(recs, ["a", "b"])["effects"]["b"]
    assert e["effect"] == pytest.approx(0.2, abs=1e-4)  # equal ratings: matches the raw diff
    assert 0 < e["lo"] < 0.2 < e["hi"]
    text = ab_stats.format_report(recs)
    assert "b - a" in text and "games per arm" in text and "equal ratings" in text


def test_logistic_adjusts_for_opponent_strength():
    # b faces weaker opponents but is exactly as strong: the raw gap is large, the adjusted ~ 0
    rng = random.Random(2)
    recs = []
    for arm, shift in (("a", 0), ("b", -100)):
        for _ in range(3000):
            opp = 1500 + shift + rng.gauss(0, 150)
            p = 1 / (1 + 10 ** ((opp - 1500) / 400))
            recs.append({"arm": arm, "result": "win" if rng.random() < p else "loss",
                         "opponent_rating": opp, "rating_before": 1500})
    raw = ab_stats.arm_summary(recs)
    assert raw["b"]["rate"] - raw["a"]["rate"] > 0.1
    e = ab_stats.adjusted_effects(recs, ["a", "b"])["effects"]["b"]
    assert abs(e["effect"]) < 0.04 < raw["b"]["rate"] - raw["a"]["rate"]


def test_ties_and_missing_ratings_are_skipped():
    recs = [{"arm": "a", "result": "tie", "opponent_rating": None, "rating_before": None},
            {"arm": "a", "result": "win", "opponent_rating": None, "rating_before": None},
            {"arm": "b", "result": "loss", "opponent_rating": None, "rating_before": None}]
    s = ab_stats.arm_summary(recs)
    assert s["a"]["games"] == 1 and s["a"]["ties"] == 1
    assert ab_stats.adjusted_effects(recs, ["a", "b"])["effects"] == {}
    assert "skipped" in ab_stats.format_report(recs)


# --- |player| rating parsing and ab_report backfill ----------------------------------

_LOG_P1 = "|player|p1|Me-bot|169|1819\n|player|p2|Foe|swimmerf|1822\n|turn|1\n"
_LOG_P2 = "|player|p1|Foe|swimmerf|1822\n|player|p2|Me-bot|169|1819\n"


def test_parse_player_lines_orders_and_missing():
    for text in (_LOG_P1, _LOG_P2):
        players = ladder.parse_player_lines(text)
        assert ladder.ratings_from_players(players, "me-bot") == (1819, 1822, "Foe")
    assert ladder.parse_player_lines("|player|p1|A|1\n")[0]["rating"] is None  # no field
    assert ladder.parse_player_lines("|player|p1|A|1|\n")[0]["rating"] is None  # unrated ''
    unrated = ladder.parse_player_lines("|player|p1|Me-bot|1|\n|player|p2|Foe|2|\n")
    assert ladder.ratings_from_players(unrated, "Me-bot") == (None, None, "Foe")


def test_game_from_battle_uses_players_for_opponent_rating():
    class B:
        battle_tag = "battle-x-1"
        turn = 3
        opponent_username = "Foe"
        opponent_rating = None
        _players = [{"username": "Foe", "rating": "1822"},
                    {"username": "Me-bot", "rating": "1819"}]
        _replay_data = []
        won = True
        lost = False

    g = ladder.game_from_battle(B(), "Me-bot")
    assert (g.rating_before, g.opponent_rating) == (1819, 1822)


def _rec(tag, **kw):
    r = {"battle_tag": tag, "arm": "a", "username": "Me-bot", "rating_before": None,
         "rating_after": None, "opponent_rating": None, "result": "win"}
    r.update(kw)
    return r


def test_backfill_ratings(tmp_path):
    logs = tmp_path / "run1" / "logs"
    logs.mkdir(parents=True)
    (logs / "battle-x-1.log").write_text(_LOG_P2)
    ab = tmp_path / "ab.jsonl"
    recs = [_rec("battle-x-1"), _rec("battle-x-2"), _rec("battle-x-1", opponent_rating=1500)]
    assert ab_stats.default_log_dirs(ab) == [logs.resolve()]
    out = ab_stats.backfill_ratings(recs, ab_stats.default_log_dirs(ab))
    assert (out[0]["rating_before"], out[0]["opponent_rating"]) == (1819, 1822)
    assert out[0]["rating_after"] is None
    assert out[1]["opponent_rating"] is None
    assert out[2]["opponent_rating"] == 1500 and out[2]["rating_before"] == 1819
    assert recs[0]["opponent_rating"] is None  # inputs untouched
    prox = ab_stats.backfill_ratings([_rec("battle-x-2", rating_before=1700)], [logs],
                                     proxy_opp_rating=True)
    assert prox[0]["opponent_rating"] == 1700
