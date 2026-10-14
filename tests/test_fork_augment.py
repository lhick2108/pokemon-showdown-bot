"""Hypothetical-position forks: edits, engine-only playout, record round trip (no server)."""

from __future__ import annotations

import random

import pytest

pe = pytest.importorskip("poke_engine")

from showdown_bot import dataset, fork_augment as fa, recorder  # noqa: E402


def _mon(name, hp=200, types=("normal", "typeless"), moves=("tackle", "thunderbolt"), **kw):
    return pe.Pokemon(
        id=name, level=80, types=types, base_types=types, hp=hp, maxhp=200,
        attack=150, defense=120, special_attack=150, special_defense=120, speed=130,
        moves=[pe.Move(id=m, pp=16) for m in moves], **kw,
    )


def _side(names):
    mons = [_mon(n) for n in names] + [pe.Pokemon.create_fainted()] * (6 - len(names))
    return pe.Side(pokemon=mons)


@pytest.fixture(scope="module")
def state_string():
    one = _side(["pikachu", "charizard", "blastoise"])
    two = _side(["venusaur", "gengar", "machamp"])
    return pe.State(side_one=one, side_two=two).to_string()


def _snapshot(text):
    ps = fa.ParsedState(text)
    return ps.dumps()


def test_parsed_state_round_trips(state_string):
    assert fa.ParsedState(state_string).dumps() == state_string
    assert fa.is_forkable(state_string)


def test_edits_change_state_and_stay_legal(state_string):
    cfg = fa.EditConfig(p_hp=0.9, p_faint_bench=0.5, p_status=0.8, p_boost=0.8, p_hazard=0.8,
                        p_screen=0.8, p_weather=0.5, p_terrain=0.5)
    kinds = set()
    for seed in range(60):
        rng = random.Random(seed)
        new, meta = fa.edit_state(state_string, rng, cfg)
        assert new != state_string and meta
        kinds.update(m["kind"] for m in meta)
        state = pe.State.from_string(new)  # parses
        for side, ps_side in ((1, state.side_one), (2, state.side_two)):
            active = ps_side.pokemon[int(ps_side.active_index)]
            assert active.hp > 0 and fa.side_alive(state, side) >= 1
            for mon in ps_side.pokemon:
                assert 0 <= mon.hp <= mon.maxhp
        ps = fa.ParsedState(new)
        for side in (0, 1):
            sleepers = [m for m in ps.mons[side] if m[fa.P_STATUS].upper() == "SLEEP"]
            assert len(sleepers) <= 1
            for boost in ps.sides[side][fa.S_BOOST0:fa.S_BOOST0 + 5]:
                assert -6 <= int(boost) <= 6
    assert {"hp", "faint", "status", "boost", "cond", "weather", "terrain"} <= kinds


def test_hp_scale_respects_bounds(state_string):
    cfg = fa.EditConfig(p_hp=1.0, p_faint_bench=0, p_status=0, p_boost=0, p_hazard=0, p_screen=0,
                        p_weather=0, p_terrain=0, hp_low=0.3, hp_high=1.3)
    for seed in range(30):
        new, meta = fa.edit_state(state_string, random.Random(seed), cfg)
        for m in meta:
            assert m["kind"] == "hp" and 1 <= m["to"] <= 200


def test_status_respects_type_immunity():
    one = pe.Side(pokemon=[_mon("charizard", types=("fire", "flying")), _mon("blastoise")]
                  + [pe.Pokemon.create_fainted()] * 4)
    two = _side(["venusaur"])
    text = pe.State(side_one=one, side_two=two).to_string()
    cfg = fa.EditConfig(p_hp=0, p_faint_bench=0, p_status=1.0, p_boost=0, p_hazard=0, p_screen=0,
                        p_weather=0, p_terrain=0)
    for seed in range(80):
        try:
            new, meta = fa.edit_state(text, random.Random(seed), cfg)
        except ValueError:
            continue
        ps = fa.ParsedState(new)
        assert ps.mons[0][0][fa.P_STATUS].upper() != "BURN"


def test_edit_is_reproducible_and_rejects_unforkable(state_string):
    a = fa.edit_state(state_string, random.Random(7))
    b = fa.edit_state(state_string, random.Random(7))
    assert a == b
    ps = fa.ParsedState(state_string)
    ps.mons[0][0][fa.P_HP] = "0"
    with pytest.raises(ValueError):
        fa.edit_state(ps.dumps(), random.Random(0))


def test_outcome_detection(state_string):
    state = pe.State.from_string(state_string)
    assert fa.outcome(state) is None
    ps = fa.ParsedState(state_string)
    for m in ps.mons[1]:
        m[fa.P_HP] = "0"
    assert fa.outcome(pe.State.from_string(ps.dumps())) == recorder.RESULT_WIN
    for m in ps.mons[0]:
        m[fa.P_HP] = "0"
    assert fa.outcome(pe.State.from_string(ps.dumps())) == recorder.RESULT_TIE


def test_play_out_terminates_with_forced_switches(state_string):
    cfg = fa.PlayConfig(iterations=400, max_turns=150)
    total_forced = 0
    for seed in range(6):
        rng = random.Random(seed)
        edited, _ = fa.edit_state(state_string, rng)
        game = fa.play_out(edited, rng, cfg)
        assert game.result in (recorder.RESULT_WIN, recorder.RESULT_LOSS, recorder.RESULT_TIE)
        assert 1 <= game.steps <= 150 and game.decisions
        total_forced += game.forced_switch_steps
        for d in game.decisions:
            assert d["truth"]["state"] == d["worlds"][0]["state"]
            assert d["chosen"] in d["policy"]
            assert len(d["worlds"][0]["moves"]) >= 2
    assert total_forced > 0  # 3v3 playouts always involve fainted-active switch turns


def test_turn_cap_drops_or_ties(state_string):
    dropped = fa.play_out(state_string, random.Random(0), fa.PlayConfig(iterations=200, max_turns=1))
    assert dropped.capped and dropped.result is None
    tied = fa.play_out(
        state_string, random.Random(0), fa.PlayConfig(iterations=200, max_turns=1, on_cap="tie")
    )
    assert tied.capped and tied.result == recorder.RESULT_TIE


def test_fork_record_loads_through_dataset(tmp_path, state_string):
    src = tmp_path / "src"
    rec = recorder.GameRecorder(src)
    for i in range(3):
        tag = f"battle-{i}"
        rec.start_game(tag, "alice", role="p1")
        for turn in range(4):
            rec.add_decision(tag, "alice", {
                "turn": turn + 1, "side": "one", "worlds": [], "policy": {}, "chosen": None,
                "truth": {"state": state_string}, "truth_status": "ok",
            })
        rec.end_game(tag, "alice", "win", turns=5)
    sources = fa.sample_sources(src, 6, 2, random.Random(0))
    assert len(sources) == 6
    per_battle = {}
    for s in sources:
        per_battle[s["battle_tag"]] = per_battle.get(s["battle_tag"], 0) + 1
    assert max(per_battle.values()) <= 2

    out = tmp_path / "forks"
    jobs = fa.make_jobs(sources, 3, out, fa.EditConfig(), fa.PlayConfig(iterations=300))
    summaries = [fa.run_fork(job) for job in jobs]
    assert all(s["ok"] for s in summaries), summaries
    games = [recorder.read_game(p) for p in recorder.iter_game_files(out)]
    assert len(games) == 6
    for g in games:
        assert g["header"]["source"] == "fork"
        assert g["header"]["fork"]["edits"] and g["header"]["fork"]["origin"]["battle_tag"]
        assert g["result"]["result"] in recorder.RESULTS
    samples = list(dataset.iter_samples(out, truth=True, worlds=False))
    assert samples
    assert all(s.search_value is not None and 0.0 <= s.search_value <= 1.0 for s in samples)
    assert all(s.kind == dataset.KIND_TRUTH for s in samples)


def test_same_seed_gives_same_sources_and_edits(tmp_path, state_string):
    rec = recorder.GameRecorder(tmp_path)
    for i in range(4):
        tag = f"battle-{i}"
        rec.start_game(tag, "alice")
        for turn in range(3):
            rec.add_decision(tag, "alice", {"turn": turn + 1, "worlds": [], "policy": {},
                                            "truth": {"state": state_string}})
        rec.end_game(tag, "alice", "loss", turns=3)

    def plan(seed):
        sources = fa.sample_sources(tmp_path, 5, 2, random.Random(seed))
        jobs = fa.make_jobs(sources, seed, tmp_path / "o", fa.EditConfig(), fa.PlayConfig())
        return [(j.seed, j.origin, fa.edit_state(j.state, random.Random(j.seed))) for j in jobs]

    assert plan(11) == plan(11)
    assert plan(11) != plan(12)
