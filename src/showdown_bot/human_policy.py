"""Human action-policy model: P(action | state, legal actions) learned from ladder games.

Training data are the policy rows of ``showdown_bot.human_replays`` (``data/human/*/policy``):
an engine ``State`` string whose side one is the ACTOR (a hindsight-completed, truth-like state:
both teams full, exact HP, ...), the action the human took and the legal action list.

The action space is a fixed set of ``N_SLOTS`` = 13 slots relative to the actor's side:

    0-3    the active Pokemon's moves in state order            ("closecombat")
    4-7    the same moves with Terastallization                 ("closecombat-tera")
    8-12   switches to the bench Pokemon at list positions 1-5  ("switch kingambit")

The active Pokemon is always list position 0 in states built by ``engine_state.build_state`` and
by the replay converter, and ``poke_engine.state_features`` lists the bench in that same order, so
a slot means the same thing in the features and in the output.

The network takes the engine's v2 features (2191 values, side one's view) plus the 13-bit legal
mask and an elo-bucket embedding (bucket ``N_ELO`` = unknown), and outputs 13 logits that are
masked to the legal slots. The numpy forward pass here is all that is needed at decision time
(``HumanPolicy``); training lives in ``showdown_bot.human_policy_train``.

To read the opponent's (side two's) behaviour from a live world, swap the two sides of the state
string (``swap_sides``) and treat it as an actor-view state: that is the exact input the model was
trained on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

FEATURE_VERSION = 2
N_STATE_FEATURES = 2191  # poke_engine.state_features v2
N_DAMAGE = 20  # damage-calculator features, see damage_features
N_FEATURES = N_STATE_FEATURES + N_DAMAGE
N_MOVES = 4
N_BENCH = 5
N_SLOTS = 2 * N_MOVES + N_BENCH
TERA_OFFSET = N_MOVES
SWITCH_OFFSET = 2 * N_MOVES

ELO_BUCKETS = ("1000-1199", "1200-1399", "1400-1599", "1600-1799", "1800+")
N_ELO = len(ELO_BUCKETS)  # the "unknown" id
ELO_ID = {name: i for i, name in enumerate(ELO_BUCKETS)}


def elo_id(elo: str | int | None) -> int:
    """Bucket id for a bucket name or a numeric rating (``None`` / unparsable: unknown)."""
    if elo is None:
        return N_ELO
    if isinstance(elo, str) and elo in ELO_ID:
        return ELO_ID[elo]
    try:
        rating = float(elo)
    except (TypeError, ValueError):
        return N_ELO
    for i, bound in enumerate((1200, 1400, 1600, 1800)):
        if rating < bound:
            return i
    return N_ELO - 1


def swap_sides(state: str) -> str:
    """The same position with the two sides exchanged (side two becomes side one)."""
    parts = state.split("/")
    parts[0], parts[1] = parts[1], parts[0]
    return "/".join(parts)


# ---------------------------------------------------------------------------
# State -> slots
# ---------------------------------------------------------------------------


def _active_pos(side: Any) -> int:
    return int(str(side.active_index)[-1])


def actor_slots(side: Any) -> tuple[list[str], list[str]]:
    """``(move ids, bench ids)`` of an actor side, lowercase, in slot order.

    Bench ids are the species ids of the Pokemon at list positions other than the active one's, in
    list order (positions 1-5 for every state this project builds).
    """
    pos = _active_pos(side)
    mons = list(side.pokemon)
    moves = [str(m.id).lower() for m in mons[pos].moves]
    bench = [str(m.id).lower() for i, m in enumerate(mons) if i != pos]
    return moves, bench


def slot_of(choice: str, moves: Sequence[str], bench: Sequence[str]) -> int | None:
    """Slot of an engine choice string, ``None`` if it is not one of this actor's actions."""
    choice = str(choice).strip().lower()
    if choice.startswith("switch "):
        name = choice[7:]
        return SWITCH_OFFSET + bench.index(name) if name in bench[:N_BENCH] else None
    tera = choice.endswith("-tera")
    if tera:
        choice = choice[: -len("-tera")]
    if choice in moves[:N_MOVES]:
        return (TERA_OFFSET if tera else 0) + moves.index(choice)
    return None


def choice_of(slot: int, moves: Sequence[str], bench: Sequence[str]) -> str | None:
    """Inverse of :func:`slot_of` (``None`` when the slot is empty for this state)."""
    if slot < N_MOVES:
        return moves[slot] if slot < len(moves) else None
    if slot < SWITCH_OFFSET:
        i = slot - TERA_OFFSET
        return moves[i] + "-tera" if i < len(moves) else None
    i = slot - SWITCH_OFFSET
    return "switch " + bench[i] if i < len(bench) else None


def legal_mask(choices: Iterable[str], moves: Sequence[str], bench: Sequence[str]) -> np.ndarray:
    mask = np.zeros(N_SLOTS, dtype=np.float32)
    for choice in choices:
        slot = slot_of(choice, moves, bench)
        if slot is not None:
            mask[slot] = 1.0
    return mask


def encode(state_string: str, choices: Iterable[str]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """``(features, legal mask, move ids, bench ids)`` for an actor-view state string."""
    import poke_engine

    state = poke_engine.State.from_string(state_string)
    moves, bench = actor_slots(state.side_one)
    base = poke_engine.state_features(state, 1, FEATURE_VERSION)
    feats = np.concatenate([np.asarray(base, dtype=np.float32), damage_features(state)])
    return feats, legal_mask(choices, moves, bench), moves, bench


def _avg_and_min(rolls: Sequence[int]) -> tuple[float, float]:
    return (sum(rolls) / len(rolls), float(min(rolls))) if rolls else (0.0, 0.0)


def damage_features(state: Any) -> np.ndarray:
    """Damage-calculator features of the actor's (side one) position, ``N_DAMAGE`` = 20 values.

    The raw engine features hold stats, types and move properties but not the product that human
    players read off the screen: how hard each move hits the opposing active. Per own move slot
    (state order): average damage as a fraction of the target's current HP (capped at 3), "can KO"
    (even the minimum roll kills), and damage relative to the best own move. Per opposing move
    slot: the same two numbers for the damage the actor would take.
    """
    out = np.zeros(N_DAMAGE, dtype=np.float32)
    try:
        import poke_engine

        side, other = state.side_one, state.side_two
        mine = side.pokemon[_active_pos(side)]
        theirs = other.pokemon[_active_pos(other)]
        their_hp, my_hp = max(1.0, float(theirs.hp)), max(1.0, float(mine.hp))
        avgs = []
        for i, move in enumerate(list(mine.moves)[:N_MOVES]):
            rolls = poke_engine.calculate_damage(state, str(move.id).lower(), "none", True)[0]
            avg, low = _avg_and_min(rolls)
            avgs.append(avg)
            out[i] = min(avg / their_hp, 3.0)
            out[N_MOVES + i] = float(low >= their_hp)
        best = max(avgs, default=0.0)
        if best > 0:
            out[2 * N_MOVES : 2 * N_MOVES + len(avgs)] = np.asarray(avgs) / best
        for j, move in enumerate(list(theirs.moves)[:N_MOVES]):
            rolls = poke_engine.calculate_damage(state, "none", str(move.id).lower(), True)[1]
            avg, low = _avg_and_min(rolls)
            out[3 * N_MOVES + j] = min(avg / my_hp, 3.0)
            out[4 * N_MOVES + j] = float(low >= my_hp)
    except BaseException as exc:  # incl. pyo3 panics: an odd position must not cost a prediction
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    return out


# ---------------------------------------------------------------------------
# The model (numpy forward pass)
# ---------------------------------------------------------------------------


class HumanPolicy:
    """MLP ``[features | legal mask] -> hidden -> hidden -> 13`` with an elo embedding.

    Weights come from an ``.npz`` written by ``human_policy_train`` (input standardisation is
    already folded into the first layer).
    """

    def __init__(self, arrays: dict[str, np.ndarray], meta: dict[str, Any] | None = None):
        self.w1 = np.ascontiguousarray(arrays["w1"], dtype=np.float32)  # (N_FEATURES, h1)
        self.wm = np.ascontiguousarray(arrays["wm"], dtype=np.float32)  # (N_SLOTS, h1)
        self.elo = np.ascontiguousarray(arrays["elo"], dtype=np.float32)  # (N_ELO + 1, h1)
        self.b1 = np.ascontiguousarray(arrays["b1"], dtype=np.float32)
        self.w2 = np.ascontiguousarray(arrays["w2"], dtype=np.float32)
        self.b2 = np.ascontiguousarray(arrays["b2"], dtype=np.float32)
        self.w3 = np.ascontiguousarray(arrays["w3"], dtype=np.float32)
        self.b3 = np.ascontiguousarray(arrays["b3"], dtype=np.float32)
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str | Path) -> HumanPolicy:
        with np.load(str(path), allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "meta"}
            meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
        return cls(arrays, meta)

    def save(self, path: str | Path) -> None:
        np.savez(
            str(path), w1=self.w1, wm=self.wm, elo=self.elo, b1=self.b1, w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3, meta=json.dumps(self.meta),
        )

    def logits(self, feats: np.ndarray, mask: np.ndarray, elo_ids: np.ndarray | int) -> np.ndarray:
        feats = np.atleast_2d(np.asarray(feats, dtype=np.float32))
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float32))
        ids = np.broadcast_to(np.asarray(elo_ids, dtype=np.int64), (feats.shape[0],))
        if 1 < feats.shape[0] <= 16:
            # Small batches run row by row: a multithreaded GEMM on a handful of rows stalls badly
            # when the cores are busy with search workers (300 ms vs 0.6 ms per row measured).
            return np.concatenate([self.logits(feats[i : i + 1], mask[i : i + 1], ids[i : i + 1])
                                   for i in range(feats.shape[0])])
        h = feats @ self.w1 + mask @ self.wm + self.elo[ids] + self.b1
        np.maximum(h, 0.0, out=h)
        h = h @ self.w2 + self.b2
        np.maximum(h, 0.0, out=h)
        return h @ self.w3 + self.b3

    def probs(self, feats: np.ndarray, mask: np.ndarray, elo_ids: np.ndarray | int = N_ELO) -> np.ndarray:
        """Softmax over the legal slots (rows with an empty mask: all zeros)."""
        mask = np.atleast_2d(np.asarray(mask, dtype=np.float32))
        z = self.logits(feats, mask, elo_ids)
        z = np.where(mask > 0, z, -np.inf)
        top = np.max(z, axis=1, keepdims=True)
        top = np.where(np.isfinite(top), top, 0.0)
        e = np.exp(z - top)
        total = e.sum(axis=1, keepdims=True)
        return np.where(total > 0, e / np.maximum(total, 1e-30), 0.0)

    def predict(
        self, state_string: str, choices: Sequence[str], elo: str | int | None = None,
        *, side: int = 1,
    ) -> dict[str, float]:
        """``{choice: probability}`` over ``choices`` for the acting ``side`` (1 or 2) of the state.

        Choices that are not one of the actor's slots get probability 0; if the model puts no mass
        on any usable choice the distribution is uniform over the usable ones.
        """
        return self.predict_many([state_string], [choices], elo, side=side)[0]

    def predict_many(
        self, states: Sequence[str], choices: Sequence[Sequence[str]], elo: str | int | None = None,
        *, side: int = 1,
    ) -> list[dict[str, float]]:
        encoded = [encode(swap_sides(s) if side == 2 else s, c) for s, c in zip(states, choices)]
        if not encoded:
            return []
        feats = np.stack([e[0] for e in encoded])
        mask = np.stack([e[1] for e in encoded])
        probs = self.probs(feats, mask, elo_id(elo))
        out: list[dict[str, float]] = []
        for (_f, m, moves, bench), row, legal in zip(encoded, probs, choices):
            dist: dict[str, float] = {}
            for choice in legal:
                slot = slot_of(choice, moves, bench)
                dist[choice] = float(row[slot]) if slot is not None and m[slot] > 0 else 0.0
            total = sum(dist.values())
            if total <= 0:
                usable = [c for c in dist if slot_of(c, moves, bench) is not None] or list(dist)
                dist = {c: (1.0 / len(usable) if c in usable else 0.0) for c in dist}
            else:
                dist = {c: p / total for c, p in dist.items()}
            out.append(dist)
        return out
