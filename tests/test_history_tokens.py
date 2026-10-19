"""Live history rows (bead showdown-bot-589.68.7): the protocol-event watchers, the reveal tracker, and the
equivalence of ``BattleHistory`` with the shard exporter's ``build_history`` on recorded self-play games."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pe = pytest.importorskip("poke_engine")

from showdown_bot import history_tokens as ht, tokens  # noqa: E402


def ev(*parts: str) -> list[str]:
    return [""] + list(parts)


def feed(w: ht.OppActionWatcher, events: list[list[str]], role: str = "p2") -> None:
    for e in events:
        w.observe(e, role)


# ------------------------------------------------------------------ opponent action watcher


def test_watcher_records_the_first_action_of_a_turn_and_tera():
    w = ht.OppActionWatcher()
    feed(w, [
        ev("switch", "p2a: Pikachu", "Pikachu, L88, M", "100/100"),  # turn 0 lead: not a decision
        ev("turn", "1"),
        ev("move", "p1a: Garchomp", "Earthquake", "p2a: Pikachu"),
        ev("move", "p2a: Pikachu", "Volt Switch", "p1a: Garchomp"),
        ev("turn", "2"),
        ev("-terastallize", "p2a: Pikachu", "Electric"),
        ev("move", "p2a: Pikachu", "Thunderbolt", "p1a: Garchomp"),
        ev("turn", "3"),
        ev("switch", "p2a: Gengar", "Gengar, L84, F", "100/100"),
    ])
    assert w.decisions == {1: "voltswitch", 2: "thunderbolt-tera", 3: "switch gengar"}


def test_watcher_ignores_locked_moves_and_cant():
    w = ht.OppActionWatcher()
    feed(w, [
        ev("turn", "4"),
        ev("move", "p2a: Dragonite", "Outrage", "p1a: X", "[from]lockedmove"),
        ev("turn", "5"),
        ev("cant", "p2a: Dragonite", "slp"),
        ev("turn", "6"),
        ev("move", "p1a: X", "Protect", "p1a: X"),
        ev("move", "p2a: Dragonite", "Dragon Claw", "p1a: X"),
    ])
    assert w.decisions == {6: "dragonclaw"}


def test_watcher_pivot_and_replacement_overwrite_when_there_was_a_choice():
    w = ht.OppActionWatcher()
    feed(w, [
        ev("teamsize", "p2", "6"),
        ev("turn", "1"),
        ev("move", "p2a: Scizor", "U-turn", "p1a: X"),
        ev("switch", "p2a: Lucario", "Lucario, L80", "100/100"),  # pivot: 5 bench -> a choice
        ev("turn", "2"),
        ev("move", "p1a: X", "Surf", "p2a: Lucario"),
        ev("faint", "p2a: Lucario"),
        ev("upkeep"),
        ev("switch", "p2a: Rotom", "Rotom-Wash, L84", "100/100"),  # replacement among 4 alive
        ev("turn", "3"),
    ])
    assert w.decisions == {1: "switch lucario", 2: "switch rotomwash"}


def test_watcher_forced_replacement_is_not_a_decision():
    w = ht.OppActionWatcher()
    feed(w, [ev("teamsize", "p2", "2"), ev("turn", "1"), ev("move", "p2a: A", "Tackle", "p1a: X"), ev("faint", "p2a: A")])
    # one Pokemon left: the replacement is forced, the recordings hold no decision for it
    feed(w, [ev("upkeep"), ev("switch", "p2a: B", "B, L80", "100/100")])
    assert w.decisions == {1: "tackle"}


def test_watcher_only_sees_the_opponent():
    w = ht.OppActionWatcher()
    feed(w, [ev("turn", "1"), ev("move", "p1a: Mine", "Tackle", "p2a: Theirs")], role="p2")
    assert w.decisions == {}


# ------------------------------------------------------------------ what the opponent has seen of us


def test_reveal_tracker():
    r = ht.RevealTracker()
    for e in [
        ev("switch", "p1a: Garchomp", "Garchomp, L80, M", "100/100"),
        ev("move", "p1a: Garchomp", "Earthquake", "p2a: X"),
        ev("move", "p1a: Garchomp", "Earthquake", "p2a: X"),
        ev("move", "p1a: Garchomp", "Sleep Talk", "p1a: Garchomp"),
        ev("move", "p1a: Garchomp", "Rock Slide", "p2a: X", "[from]move: Sleep Talk"),
        ev("-heal", "p1a: Garchomp", "90/100", "[from] item: Leftovers"),
        ev("-terastallize", "p1a: Garchomp", "Steel"),
        ev("switch", "p1a: Pikachu", "Pikachu, L88", "100/100"),
        ev("-weather", "SunnyDay", "[from] ability: Drought", "[of] p1a: Pikachu"),
        ev("switch", "p2a: Foe", "Foe, L80", "100/100"),
        ev("move", "p2a: Foe", "Tackle", "p1a: Pikachu"),
    ]:
        r.observe(e, "p1")
    rows = {x["species"]: x for x in r.rows()}
    assert set(rows) == {"garchomp", "pikachu"}  # the bench that never appeared is not listed
    assert rows["garchomp"]["moves"] == ["earthquake", "sleeptalk"]
    assert rows["garchomp"]["item"] and rows["garchomp"]["tera"] and not rows["garchomp"]["ability"]
    assert rows["pikachu"]["ability"] and not rows["pikachu"]["item"] and rows["pikachu"]["moves"] == []
    # the rows go straight into the engine's reveal argument
    assert tokens.reveal_rows(r.rows())[0][0] == "garchomp"


# ------------------------------------------------------------------ BattleHistory against the exporter


def selfplay_dir() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        d = parent / "data" / "selfplay" / "gen7"
        if d.is_dir() and any(d.glob("*.jsonl.gz")):
            return d
    return None


@pytest.mark.skipif(selfplay_dir() is None, reason="no recorded self-play games")
def test_battle_history_matches_the_exporter_on_recorded_games():
    from showdown_bot import recorder

    root = selfplay_dir()
    vocab, fields, lay = tokens.Vocab(), tokens._shard_fields(), tokens.layout()
    opts = tokens.Options(reconstruct_q=False)
    groups = tokens._group_files(root)
    checked = rows_checked = 0
    for files in groups[:40]:
        recs = [g for g in map(tokens._read, files) if g is not None]
        if len(recs) != 2:
            continue
        for gi, g in enumerate(recs):
            peer = recs[1 - gi]
            samples = tokens.game_samples(g, peer, source_idx=0, kind="selfplay", opts=opts, vocab=vocab,
                                          fields=fields, lay=lay, stats=Counter())
            by_idx = {int(s["decision_idx"]): s for s in samples}
            h = ht.BattleHistory(vocab)
            # the opponent's decision of each turn: the peer's last recorded choice of that turn
            h.opp.decisions = {pd.get("turn"): pd.get("chosen") for pd in peer["decisions"]}
            for idx, d in enumerate(g["decisions"]):
                s = by_idx.get(idx)
                if s is None:
                    continue
                state, _ = tokens._truth_or_world(d, "auto")
                reveal = tokens.reveal_rows(d["opp_facts"]) if d.get("opp_facts") is not None else None
                cat, cont, _names = pe.state_tokens_str(state, 1, reveal, True)
                hc, hf = h.begin(int(d.get("turn") or 0), tokens.split_buffers(cat, cont, lay))
                assert (hc == s["hist_cat"]).all(), (files[0].name, idx)
                assert np.abs(hf.astype(np.float16).astype(np.float32) - s["hist_cont"].astype(np.float32)).max() < 1e-3
                h.chosen(d.get("chosen"))
                checked += 1
                rows_checked += int((hc[:, 0] > 0).sum())
    assert checked > 100 and rows_checked > 500


def test_begin_is_idempotent_until_chosen_and_abandon_drops_it():
    vocab = tokens.Vocab()
    h = ht.BattleHistory(vocab)
    states = [line.strip() for line in (Path(__file__).resolve().parents[1] / "engine" / "data" / "gen9randombattle.txt")
              .read_text().splitlines() if line.strip()]
    lay = tokens.layout()

    def bufs(i: int):
        cat, cont, _ = pe.state_tokens_str(states[i], 1, None, False)
        return tokens.split_buffers(cat, cont, lay)

    a = h.begin(1, bufs(0))
    assert (a[0] == 0).all()  # the first decision of a game has no history
    again = h.begin(1, bufs(0))  # a fallback inside the same decision
    assert len(h.recs) == 1 and again[0] is a[0]
    h.chosen("tackle")
    b = h.begin(2, bufs(1))
    assert (b[0][:2, 0] == [1, 2]).all() and b[0][0, 1] == 1  # our move, then the opponent's unknown row
    h.abandon()
    assert len(h.recs) == 1
    c = h.begin(2, bufs(2))
    assert len(h.recs) == 2 and (c[0][:2, 0] == [1, 2]).all()
    # the role arrives late: early events are replayed
    h2 = ht.BattleHistory(vocab)
    h2.observe([ev("switch", "p1a: Garchomp", "Garchomp, L80", "100/100")], None)
    h2.observe([ev("turn", "1")], "p1")
    assert [r["species"] for r in h2.our_facts()] == ["garchomp"]
