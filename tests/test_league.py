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
    assert [s.label for s in league.parse_bots("random,heuristic,fp")] == [
        "random", "heuristics", "foulplay"
    ]
    with pytest.raises(ValueError):
        league.parse_bots("random")
    with pytest.raises(ValueError):
        league.parse_bots("random,nope")


def test_parse_spec_good():
    specs = league.parse_bots(
        "stock=search,net64=search-net:evaluator=models/a.pev:scale=4,"
        "n2=search-net:evaluator=C:\m\a.pev:scale=8:time_ms=50:worlds=3,foulplay"
    )
    stock, net, n2, fp = specs
    assert (stock.label, stock.type, stock.params) == ("stock", "search", ())
    assert net.type == "search-net" and net.param("evaluator") == "models/a.pev"
    assert net.param("scale") == 4.0 and net.text.startswith("net64=search-net")
    assert n2.param("evaluator") == "C:\m\a.pev"
    assert (n2.param("time_ms"), n2.param("worlds")) == (50, 3)
    assert fp.type == "foulplay"
    assert league.parse_spec("search:time_ms=20").label == "search"


def test_parse_spec_selection_keys():
    spec = league.parse_spec("rm=search:selection=RM:gamma=0.2")
    assert spec.param("selection") == "rm" and spec.param("gamma") == 0.2
    for bad in ("x=search:selection=foo", "x=search:gamma=0", "x=search:gamma=1.5"):
        with pytest.raises(ValueError):
            league.parse_spec(bad)


@pytest.mark.parametrize("bad", [
    "x=search:bogus=1", "x=search:time_ms", "x=search:time_ms=abc", "x=search:time_ms=0",
    "x=search:scale=-1", "x=nope", "=search", "x=search:time_ms=1:time_ms=2",
    "x=random:time_ms=5", "x=search-net", "a b=search", "x=y=search",
])
def test_parse_spec_malformed(bad, monkeypatch):
    monkeypatch.delenv("SHOWDOWN_SEARCH_EVALUATOR", raising=False)
    with pytest.raises(ValueError):
        league.parse_spec(bad)


def test_duplicate_labels_rejected():
    with pytest.raises(ValueError):
        league.parse_bots("a=search,a=random")


def test_specs_in_games_and_results(tmp_path):
    specs = {"a": "a=search", "b": "b=random"}
    games = [Game(f"t{i}", "a", "b", "a" if i % 2 else "b", 9, 0, specs=specs) for i in range(10)]
    path = tmp_path / "games.jsonl"
    league.write_games(path, games)
    assert json.loads(path.read_text().splitlines()[0])["specs"] == specs
    assert league.read_games(path)[0].specs == specs
    res = league.write_results(tmp_path, games, ["a", "b"], n_boot=5, seed=0, anchor=None,
                               extra={"specs": specs})
    assert json.loads((tmp_path / "results.json").read_text())["specs"] == res["specs"] == specs


def test_scale_guard_without_evaluator_scale(tmp_path):
    spec = league.parse_spec("n=search-net:evaluator=x.pev:scale=4")
    cfg = league.LeagueConfig(out=tmp_path)
    import inspect
    from showdown_bot.players.search import SearchNetPlayer
    if "evaluator_scale" not in inspect.signature(SearchNetPlayer.__init__).parameters:
        with pytest.raises(ValueError, match="evaluator_scale"):
            league._make_player(spec, cfg, None, 1, 0)


def test_run_limited_respects_limit_and_order():
    import asyncio

    state = {"now": 0, "peak": 0}

    def job(i):
        async def run():
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01 * (5 - i))
            state["now"] -= 1
            return i
        return run

    out = asyncio.run(league.run_limited([job(i) for i in range(5)], 2))
    assert out == [0, 1, 2, 3, 4] and state["peak"] == 2
    state.update(now=0, peak=0)
    asyncio.run(league.run_limited([job(i) for i in range(5)], 1))
    assert state["peak"] == 1


def test_run_limited_failure_cancels_others():
    import asyncio

    cancelled = []

    async def slow():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    async def boom():
        await asyncio.sleep(0.01)
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        asyncio.run(league.run_limited([slow, boom, slow], 3))
    assert len(cancelled) == 2


def test_load_report_warns_when_oversubscribed(tmp_path):
    specs = league.parse_bots("random,search,foulplay")
    by = {s.label: s for s in specs}
    pairs = [(by[a], by[b]) for a, b in league.roundrobin_pairs(list(by))]
    cfg = league.LeagueConfig(out=tmp_path, parallel_pairs=3, fp_parallelism=2, fp_threads=2)
    line, warn = league.load_report(pairs, cfg, cpu_count=4)
    assert "foulplay=8" in line and warn and "exceeds" in warn
    cfg1 = league.LeagueConfig(out=tmp_path, parallel_pairs=1)
    assert league.load_report(pairs, cfg1, cpu_count=64)[1] is None


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


def test_parse_spec_mode_key():
    spec = league.parse_spec("nash=search:mode=Nash:time_ms=50")
    assert spec.param("mode") == "nash" and spec.param("time_ms") == 50
    with pytest.raises(ValueError):
        league.parse_spec("x=search:mode=bogus")
