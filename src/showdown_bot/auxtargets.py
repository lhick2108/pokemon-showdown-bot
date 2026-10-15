"""Auxiliary training targets for the value net (KataGo-style, bead showdown-bot-589.56).

Two sub-components of the game outcome, derived from the recording's last stored state:

* survival: for each of the 12 feature slots (6 own then 6 opp, in the SAME slot order as
  ``state_features``: slot 0 = the active Pokemon, then the rest in team order), whether
  that Pokemon is still alive at the END of the game. Masked when the slot is empty or
  already fainted in the input state, when the game was a tie, or when the Pokemon cannot
  be found in the reference state (a sampled, never-revealed opposing Pokemon that is not on
  the real team).
* final remaining-Pokemon difference (own alive minus opp alive at the end) / 6.

The recordings do not store the terminal state. The game end is reconstructed from the
result plus the LAST stored decision state (``final_state``; the truth state when there is
one): the loser has no Pokemon left by definition, the winner keeps the ones alive in the
last stored state. This slightly overestimates the winner's survivors (a Pokemon can still
faint in the last unrecorded turns).
"""

from __future__ import annotations

import numpy as np

N_SLOTS = 6
N_AUX = 2 * N_SLOTS + 1  # 12 survival slots + the remaining-Pokemon difference (last column)


def slot_ids(side) -> list[str | None]:
    """Pokemon ids in feature-slot order (None = empty slot); slot 0 is the active Pokemon."""
    team = list(side.pokemon)
    active = int(side.active_index)
    order = [active] + [i for i in range(len(team)) if i != active]
    ids: list[str | None] = [team[i].id for i in order if i < len(team)][:N_SLOTS]
    ids += [None] * (N_SLOTS - len(ids))
    return ids


def slot_hp(side) -> list[int]:
    team = list(side.pokemon)
    active = int(side.active_index)
    order = [active] + [i for i in range(len(team)) if i != active]
    hp = [team[i].hp for i in order if i < len(team)][:N_SLOTS]
    return hp + [0] * (N_SLOTS - len(hp))


def _alive(side) -> dict[str, bool]:
    return {p.id: p.hp > 0 for p in side.pokemon}


def aux_targets(state, ref, result: str) -> tuple[np.ndarray, np.ndarray]:
    """``(y, mask)``, both length ``N_AUX``, from the input ``state`` (side one = ours), the
    ``ref`` state (the recording's last stored state) and the recording player's ``result``."""
    y = np.zeros(N_AUX, dtype=np.float32)
    m = np.zeros(N_AUX, dtype=np.float32)
    if result not in ("win", "loss"):
        return y, m
    won = result == "win"
    sides = ((state.side_one, ref.side_one, won), (state.side_two, ref.side_two, not won))
    alive_ref = []
    for k, (cur, fin, side_won) in enumerate(sides):
        ids, hps, fin_alive = slot_ids(cur), slot_hp(cur), _alive(fin)
        alive_ref.append(sum(fin_alive.values()))
        for j, (pid, hp) in enumerate(zip(ids, hps)):
            if pid is None or hp <= 0 or pid not in fin_alive:
                continue
            m[k * N_SLOTS + j] = 1.0
            y[k * N_SLOTS + j] = float(fin_alive[pid]) if side_won else 0.0
    own_n, opp_n = alive_ref
    y[-1] = max(1, own_n) / N_SLOTS if won else -max(1, opp_n) / N_SLOTS
    m[-1] = 1.0
    return y, m
