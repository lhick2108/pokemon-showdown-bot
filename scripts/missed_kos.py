"""List decisions where a move would likely KO the opponent's active but the bot chose otherwise.

    python scripts/missed_kos.py RECORD.jsonl.gz

For each searched decision and each recorded world: for every one of our move options,
simulate it against each of the opponent's non-switch options (generate_instructions) and
call it a KO when the opponent's active faints with >= 50% probability against a majority
of those options. Prints the KO options, the bot's choice, and the search's mean score
(visits-weighted, averaged over worlds) of each.
"""

from __future__ import annotations

import sys
from collections import defaultdict

from poke_engine import State, generate_instructions

from showdown_bot import recorder


def opp_active_hp_after(state_str: str, mine: str, theirs: str) -> float:
    """Probability that side two's active ends the turn at 0 HP."""
    st = State.from_string(state_str)
    idx = int(st.side_two.active_index)
    p_ko = 0.0
    for o in generate_instructions(State.from_string(state_str), mine, theirs):
        child = st.apply_instructions(o)
        if int(child.side_two.pokemon[idx].hp) <= 0:
            p_ko += o.percentage / 100.0
    return p_ko


def opp_options(state_str: str) -> list[str]:
    st = State.from_string(state_str)
    mon = st.side_two.pokemon[int(st.side_two.active_index)]
    return [m.id.lower() for m in mon.moves if not m.disabled and int(m.pp) > 0]


def main() -> None:
    game = recorder.read_game(sys.argv[1])
    flagged = 0
    for d in game["decisions"]:
        if not d.get("worlds") or d.get("chosen") is None:
            continue
        ko_votes: dict[str, int] = defaultdict(int)
        score: dict[str, list[float]] = defaultdict(list)
        for w in d["worlds"]:
            theirs = opp_options(w["state"])
            for m in w["moves"]:
                c = m["choice"]
                score[c].append(m["score"] / max(m["visits"], 1))
                if c.startswith("switch") or not theirs:
                    continue
                try:
                    hits = sum(opp_active_hp_after(w["state"], c, t) >= 0.5 for t in theirs)
                except Exception:
                    continue
                if hits * 2 > len(theirs):
                    ko_votes[c] += 1
        n = len(d["worlds"])
        kos = sorted(c for c, v in ko_votes.items() if v * 2 > n)
        chosen = d["chosen"]
        if kos and chosen not in kos:
            flagged += 1
            mean = {c: round(sum(v) / len(v), 3) for c, v in score.items()}
            print(f"turn {d['turn']}: chose {chosen} ({mean.get(chosen)}), KO options "
                  + ", ".join(f"{c} ({mean.get(c)})" for c in kos)
                  + f"  policy top: {sorted(d['policy'].items(), key=lambda kv: -kv[1])[:3]}")
    print(f"{flagged} decisions with a likely KO not taken")


if __name__ == "__main__":
    main()
