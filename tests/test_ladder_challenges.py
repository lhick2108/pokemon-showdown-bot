import asyncio
import json
from types import SimpleNamespace

from showdown_bot import ladder
from showdown_bot.ladder import ChallengePolicy, to_id, with_challenge_policy

FMT = "gen9randombattle"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_to_id():
    assert to_id(" Some-User_1 ") == "someuser1"


def test_accepts_target_format_only():
    p = ChallengePolicy(FMT)
    assert p.decide("Alice", FMT, busy=False)[0] == "accept"
    action, reason = p.decide("Bob", "gen9ou", busy=False)
    assert action == "reject" and "gen9ou" in reason
    assert p.decide("Carl", "", busy=False)[0] == "reject"
    assert p.counts == {"accepted": 1, "rejected": 2}


def test_format_match_ignores_case_and_punctuation():
    assert ChallengePolicy(FMT).decide("a", "Gen9 RandomBattle", False)[0] == "accept"


def test_allowlist():
    p = ChallengePolicy(FMT, allow=["Alice", "b-ob"])
    assert p.decide("alice", FMT, False)[0] == "accept"
    p2 = ChallengePolicy(FMT, allow=["Alice", "b-ob"])
    assert p2.decide(" Bob", FMT, False)[0] == "accept"
    action, reason = p2.decide("mallory", FMT, False)
    assert action == "reject" and "allowlist" in reason


def test_rejects_while_busy_and_during_accept_grace():
    clock = Clock()
    p = ChallengePolicy(FMT, clock=clock)
    assert p.decide("a", FMT, busy=True)[0] == "reject"
    assert p.decide("b", FMT, busy=False)[0] == "accept"
    # accepted but battle not created yet: another challenger is rejected
    assert p.decide("c", FMT, busy=False) == ("reject", "a battle is already in progress")
    p.battle_started()
    clock.t += 1
    assert p.decide("d", FMT, busy=False)[0] == "accept"


def test_accept_grace_expires():
    clock = Clock()
    p = ChallengePolicy(FMT, clock=clock)
    assert p.decide("a", FMT, False)[0] == "accept"
    clock.t += p.ACCEPT_GRACE_S + 1  # the accepted battle never came
    assert p.decide("b", FMT, False)[0] == "accept"


def test_repeat_of_same_challenge_answered_once():
    clock = Clock()
    p = ChallengePolicy(FMT, clock=clock)
    assert p.decide("a", "gen9ou", False)[0] == "reject"
    assert p.decide("a", "gen9ou", False) is None
    clock.t += p.dedupe_s + 1
    assert p.decide("a", "gen9ou", False)[0] == "reject"


def test_idle_clock_resets_on_challenge_and_touch():
    clock = Clock()
    p = ChallengePolicy(FMT, clock=clock)
    clock.t += 50
    assert p.idle_seconds() == 50
    p.decide("a", "gen9ou", False)
    assert p.idle_seconds() == 0
    clock.t += 5
    p.touch()
    assert p.idle_seconds() == 0


class FakeClient:
    def __init__(self):
        self.sent = []

    async def accept_challenge(self, user, team):
        self.sent.append(("accept", user))

    async def send_message(self, message, room=""):
        self.sent.append((message, room))


class BasePlayer:
    username = "BotAccount"
    next_team = None

    def __init__(self):
        self.ps_client = FakeClient()
        self.battles = {}

    def _battle_finished_callback(self, battle):
        pass

    async def _create_battle(self, split_message):
        return SimpleNamespace(finished=False)


def make(allow=None):
    policy = ChallengePolicy(FMT, allow)
    seen = []
    cls = with_challenge_policy(BasePlayer, policy, lambda *a: seen.append(a))
    return cls(), policy, seen


def run(coro):
    return asyncio.run(coro)


def test_update_challenges_accepts_and_rejects():
    player, _, seen = make()
    msg = ["", "updatechallenges",
           json.dumps({"challengesFrom": {"alice": FMT, "bob": "gen9ou"}, "challengeTo": None})]
    run(player._update_challenges(msg))
    assert player.ps_client.sent == [("accept", "alice"), ("/reject bob", "")]
    assert [s[2] for s in seen] == ["accept", "reject"]


def test_pm_challenge_and_dedupe_with_updatechallenges():
    player, _, _ = make()
    run(player._handle_challenge_request(["", "pm", " Alice", "BotAccount", f"/challenge {FMT}", FMT]))
    run(player._update_challenges(["", "updatechallenges",
                                   json.dumps({"challengesFrom": {"alice": FMT}})]))
    assert player.ps_client.sent == [("accept", "alice")]  # answered once


def test_pm_withdrawn_challenge_and_own_pm_ignored():
    player, _, _ = make()
    run(player._handle_challenge_request(["", "pm", " Alice", "BotAccount", "/challenge", ""]))
    run(player._handle_challenge_request(["", "pm", " BotAccount", "Alice", f"/challenge {FMT}", FMT]))
    assert player.ps_client.sent == []


def test_second_challenge_rejected_while_battle_running():
    player, _, _ = make()
    player.battles["battle-x"] = SimpleNamespace(finished=False)
    run(player._update_challenges(["", "updatechallenges",
                                   json.dumps({"challengesFrom": {"alice": FMT}})]))
    assert player.ps_client.sent == [("/reject alice", "")]


def test_allowlist_in_player():
    player, _, _ = make(allow=["alice"])
    run(player._update_challenges(["", "updatechallenges",
                                   json.dumps({"challengesFrom": {"mallory": FMT}})]))
    assert player.ps_client.sent == [("/reject mallory", "")]


def test_battle_creation_clears_accept_grace():
    player, policy, _ = make()
    run(player._update_challenges(["", "updatechallenges",
                                   json.dumps({"challengesFrom": {"alice": FMT}})]))
    assert policy._accepted_at is not None
    run(player._create_battle([]))
    assert policy._accepted_at is None


def test_ladder_module_unchanged_api():
    for name in ("with_ladder_hooks", "summarize", "LadderGame", "heuristic_order"):
        assert hasattr(ladder, name)
