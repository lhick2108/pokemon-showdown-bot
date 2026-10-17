import json
import asyncio
import logging
import time

import pytest

from showdown_bot import ladder

SECRET = "hunter2-very-secret"


def test_parse_env_text():
    text = (
        "# comment\n\nSHOWDOWN_USERNAME=MyBot-bot\n"
        f"export SHOWDOWN_PASSWORD=\"{SECRET}\"  # trailing\n"
        "OTHER='a b' \nbad line\nX=1 # inline\n"
    )
    got = ladder.parse_env_text(text)
    assert got["SHOWDOWN_USERNAME"] == "MyBot-bot"
    assert got["SHOWDOWN_PASSWORD"] == SECRET
    assert got["OTHER"] == "a b"
    assert got["X"] == "1"
    assert "bad line" not in got


def test_load_credentials_env_beats_file_and_missing(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"SHOWDOWN_USERNAME=filebot\nSHOWDOWN_PASSWORD={SECRET}\n")
    creds = ladder.load_credentials({"SHOWDOWN_USERNAME": "envbot"}, env)
    assert (creds.username, creds.password) == ("envbot", SECRET)
    with pytest.raises(ValueError, match="SHOWDOWN_PASSWORD") as err:
        ladder.load_credentials({"SHOWDOWN_USERNAME": "x"}, tmp_path / "nope.env")
    assert SECRET not in str(err.value)
    with pytest.raises(ValueError):
        ladder.load_credentials({}, None)


def test_password_never_in_repr_str_or_log(tmp_path, caplog):
    creds = ladder.Credentials("bot", SECRET)
    assert SECRET not in repr(creds) and SECRET not in str(creds)
    assert SECRET not in f"{creds!r} {creds}"
    flt = ladder.RedactFilter([SECRET])
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "login with %s", (SECRET,), None)
    assert flt.filter(rec)
    assert SECRET not in rec.getMessage()
    with pytest.raises(TypeError):
        import pickle
        pickle.dumps(creds)


def test_rating_after_parsing():
    data = [["", "raw", "Me-bot's rating: 1000 &rarr; <strong>1021</strong><br />(+21 for winning)"]]
    assert ladder.rating_after_from_log(data, "me-bot") == 1021
    assert ladder.rating_after_from_log(data, "other") is None
    assert ladder.rating_after_from_log([], "me-bot") is None


def test_summarize_trajectory_and_wilson():
    g = [
        ladder.LadderGame("a", "o", "win", 10, 1000, 1020),
        ladder.LadderGame("b", "o", "loss", 12, 1020, None),
        ladder.LadderGame("c", "o", "win", 9, None, 1030),
    ]
    text = ladder.summarize(g)
    assert "W-L-T 2-1-0" in text and "Wilson" in text
    assert "1000 -> 1020 -> 1030" in text and "+30" in text
    assert "0" in ladder.summarize([])


class _Move:
    def __init__(self, bp):
        self.base_power, self.type = bp, None


class _Battle:
    battle_tag = "battle-x"
    turn = 3
    opponent_active_pokemon = None
    available_moves = [_Move(40), _Move(90), _Move(60)]


class _Base:
    async def choose_move(self, battle):
        await asyncio.sleep(10)
        return "slow"

    def create_order(self, move):
        return ("move", move.base_power)

    def choose_random_move(self, battle):
        return "random"

    def _battle_finished_callback(self, battle):
        pass


def test_time_cap_falls_back_to_heuristic_quickly():
    cls = ladder.with_ladder_hooks(_Base, move_cap_s=0.2)
    start = time.perf_counter()
    order = asyncio.run(cls().choose_move(_Battle()))
    assert order == ("move", 90)
    assert time.perf_counter() - start < 2
    assert cls.cap_hits == 1


def test_time_cap_passes_fast_decisions_and_hook_runs():
    class Fast(_Base):
        async def choose_move(self, battle):
            return "fast"

    seen = []
    cls = ladder.with_ladder_hooks(Fast, move_cap_s=1, on_finish=lambda p, b: seen.append(b))
    p = cls()
    assert asyncio.run(p.choose_move(_Battle())) == "fast"
    p._battle_finished_callback("battle")
    assert seen == ["battle"] and cls.cap_hits == 0


# --- rating arrives after the finish callback; names with non-cp1252 characters ---------

STAR = "Star☆Bot❤"


def test_rating_line_with_unicode_name_and_chunk_parsing():
    raw = ["", "raw", f"{STAR}'s rating: 2259 &rarr; <strong>2272</strong><br />(+13 for winning)"]
    assert ladder.rating_after_from_log([raw], STAR) == 2272
    chunk = [[">battle-gen9randombattle-1"], ["", "win", STAR], raw]
    assert ladder.rating_from_messages(chunk, STAR) == ("battle-gen9randombattle-1", 2272)
    # the opponent's line or a chunk without one yields nothing
    assert ladder.rating_from_messages(chunk, "someone") == (None, None)
    assert ladder.rating_from_messages([[">b"], ["", "win", STAR]], STAR) == (None, None)
    assert ladder.rating_from_messages([], STAR) == (None, None)


def test_rating_patched_when_it_arrives_after_finish_callback():
    """Real server order: |win| (finish callback, record written) then |raw| rating line."""
    events = []

    class P(_Base):
        username = STAR

        async def _handle_battle_message(self, split_messages):
            for m in split_messages[1:]:
                if m[1] == "win":
                    self._battle_finished_callback("battle")
                events.append(("seen", m[1]))

    cls = ladder.with_ladder_hooks(
        P, move_cap_s=1,
        on_finish=lambda p, b: events.append("finish"),
        on_rating=lambda p, tag, new: events.append(("rating", tag, new)),
    )
    chunk = [[">battle-x"], ["", "win", STAR],
             ["", "raw", f"{STAR}'s rating: 2259 &rarr; <strong>2272</strong>"]]
    asyncio.run(cls()._handle_battle_message(chunk))
    assert events.index("finish") < events.index(("rating", "battle-x", 2272))


def test_update_game_rating_rewrites_only_matching_row(tmp_path):
    path = tmp_path / "games.jsonl"
    for tag in ("a", "b"):
        ladder.append_game(path, ladder.LadderGame(tag, STAR, "win", 5, 1000, None))
    assert ladder.update_game_rating(path, "b", 1013)
    assert not ladder.update_game_rating(path, "zzz", 1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["rating_after"] for r in rows] == [None, 1013]
    assert rows[0]["opponent"] == STAR


def test_safe_print_survives_cp1252_stdout(tmp_path):
    path = tmp_path / "out.txt"
    with path.open("w", encoding="cp1252") as fh:
        with pytest.raises(UnicodeEncodeError):
            print(STAR, file=fh)
        ladder.safe_print(f"vs {STAR}: win", file=fh, flush=True)
    assert "vs Star" in path.read_text(encoding="cp1252")


def test_configure_stdio_makes_stdout_utf8_replace(tmp_path, monkeypatch):
    import sys
    path = tmp_path / "out.txt"
    with path.open("w", encoding="cp1252") as fh:
        monkeypatch.setattr(sys, "stdout", fh)
        monkeypatch.setattr(sys, "stderr", fh)
        ladder.configure_stdio()
        print(STAR)
    assert STAR in path.read_text(encoding="utf-8")
