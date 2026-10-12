import json
import math
import random

import pytest

from showdown_bot import league
from showdown_bot.league import TIE, Game


def synth(strengths, n_per_pair, seed=0, tie_rate=0.0):
    rng = random.Random(seed)
    bots = list(strengths)
    games = []
    for a, b in league.roundrobin_pairs(bots):
        for g in range(n_per_pair):
            p1, p2 = (a, b) if g % 2 == 0 else (b, a)
            if rng.random() < tie_rate:
                w = TIE
            else:
                pa = strengths[p1] / (strengths[p1] + strengths[p2])
                w = p1 if rng.random() < pa else p2
            games.append(Game(f"battle-{len(games)}", p1, p2, w, 20, 0))
    return bots, games


TRUE = {"a": 4.0, "b": 2.0, "c": 1.0, "d": 0.5}


def elo_of(s):
    gm = math.exp(sum(math.log(v) for v in s.values()) / len(s))
    return {k: 400 * math.log10(v / gm) for k, v in s.items()}


def test_bt_recovers_known_strengths():
    bots, games = synth(TRUE, 3000)
    got = dict(zip(*league.rate(games, bots)))
    for k, v in elo_of(TRUE).items():
        assert got[k] == pytest.approx(v, abs=25)
    assert sum(got.values()) == pytest.approx(0, abs=1e-6)


def test_ties_count_as_half():
    ties = [Game(f"t{i}", "x", "y", TIE, 5) for i in range(50)]
    got = dict(zip(*league.rate(ties)))
    assert got["x"] == pytest.approx(0, abs=1e-6)
    assert got["y"] == pytest.approx(0, abs=1e-6)
    # 30 wins + 20 ties scores 80% for x; the same score without ties gives the same rating
    mixed = [Game(f"m{i}", "x", "y", "x" if i < 30 else TIE, 5) for i in range(50)]
    plain = [Game(f"p{i}", "x", "y", "x" if i < 40 else "y", 5) for i in range(50)]
    assert dict(zip(*league.rate(mixed)))["x"] == pytest.approx(
        dict(zip(*league.rate(plain)))["x"], abs=1e-6
    )


def test_prior_keeps_undefeated_finite():
    games = [Game(f"g{i}", "w", "l", "w", 5) for i in range(40)]
    games += [Game(f"h{i}", "w", "m", "w", 5) for i in range(40)]
    games += [Game(f"i{i}", "m", "l", "m" if i % 2 else "l", 5) for i in range(40)]
    got = dict(zip(*league.rate(games)))
    assert all(math.isfinite(v) for v in got.values())
    assert got["w"] > got["m"] and got["w"] > got["l"]


def test_bootstrap_ci_contains_truth_and_is_seeded():
    bots, games = synth(TRUE, 1500, seed=3)
    ci1 = league.bootstrap_elo(games, bots, n_boot=100, seed=7)
    ci2 = league.bootstrap_elo(games, bots, n_boot=100, seed=7)
    ci3 = league.bootstrap_elo(games, bots, n_boot=100, seed=8)
    assert ci1 == ci2 and ci1 != ci3
    for k, v in elo_of(TRUE).items():
        lo, hi = ci1[k]
        assert lo < v < hi and hi - lo < 80


def test_games_jsonl_round_trip(tmp_path):
    games = [
        Game("battle-gen9randombattle-1", "random", "search", "search", 31, 5, "u1", "u2"),
        Game("battle-gen9randombattle-2", "search", "random", TIE, 300, 5),
    ]
    path = tmp_path / "games.jsonl"
    league.write_games(path, games)
    assert league.read_games(path) == games
    row = json.loads(path.read_text().splitlines()[0])
    assert row["bots"] == ["random", "search"] and row["winner"] == "search"
    with pytest.raises(ValueError):
        Game.from_dict({**row, "winner": "nobody"})
    with pytest.raises(ValueError):
        Game.from_dict({**row, "schema_version": 99})


def test_roundrobin_and_seat_alternation():
    assert league.roundrobin_pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]
    sched = league.seat_schedule("a", "b", 5)
    assert [s[0] for s in sched] == ["a", "b", "a", "b", "a"]
    assert sum(s[2] for s in sched) == 5
    assert league.seat_schedule("a", "b", 5, chunk=2) == [
        ("a", "b", 2), ("b", "a", 2), ("a", "b", 1)
    ]


def test_parse_bots_aliases_and_errors():
    assert league.parse_bots("random,heuristic,fp") == ["random", "heuristics", "foulplay"]
    with pytest.raises(ValueError):
        league.parse_bots("random")
    with pytest.raises(ValueError):
        league.parse_bots("random,nope")


def test_rate_only_on_fixture_dir(tmp_path):
    _, games = synth({"a": 3.0, "b": 1.0, "c": 1.0}, 40, seed=1)
    league.write_games(tmp_path / "games.jsonl", games)
    res = league.rate_only(tmp_path, n_boot=20, seed=0, anchor="c")
    md = (tmp_path / "results.md").read_text()
    assert "BT rating" in md and "Head to head" in md
    by = {r["bot"]: r for r in res["ratings"]}
    assert by["c"]["elo"] == pytest.approx(0, abs=1e-9)
    assert by["a"]["elo"] > by["b"]["elo"]
    assert json.loads((tmp_path / "results.json").read_text())["n_games"] == len(games)
