"""Tokenizer: turn a battle state into 14 "Pokemon tokens".

In text, a tokenizer chops a sentence into pieces ("un", "believ", "able") and maps each
piece to an integer id.  Here a *token* is a bigger thing: one whole Pokemon (or the
field).  A battle state becomes a sequence of 14 tokens:

    index:   0        1 ........ 6      7 ........ 12     13
    token: SUMMARY | own mons (6)   | opp mons (6)    | FIELD
                     [active first]   [active first]

Every token is described by two kinds of information, because the two kinds are fed to
the network differently:

  * CATEGORICAL fields (names): species, item, ability, status, 4 moves, tera type,
    2 types.  These become integer ids via `Vocab`, and later an embedding lookup.
  * NUMERIC fields (quantities): hp fraction, stats, boosts, ...  These are already
    numbers (scaled to roughly 0..1) and go through a Linear layer.

Output shapes of `tokenize`:
    cats : (14, N_CAT) int64     -- vocab ids; 0 (= <pad>) means "nothing here"
    nums : (14, N_NUM) float32   -- numeric features; zeros where not applicable
"""
from __future__ import annotations

import re
import sys
from collections import Counter

import numpy as np

N_TOKENS = 14
SUMMARY, OWN0, OPP0, FIELD = 0, 1, 7, 13  # token positions
N_CAT = 11   # categorical fields per token
N_NUM = 17   # numeric features per token (padded with zeros when unused)

# Meaning of each categorical column (same for every token):
#   0 species  1 item  2 ability  3 status  4-7 the four moves  8 tera type  9-10 types
# On the FIELD token only columns 0 and 1 are used, for weather and terrain.
CAT_NAMES = ["species", "item", "ability", "status", "move", "move", "move", "move",
             "tera", "type", "type"]


def norm(name: str) -> str:
    """'Swords Dance' / 'SWORDSDANCE' / 'swords-dance' -> 'swordsdance'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# --------------------------------------------------------------------------- Vocab
class Vocab:
    """String -> integer id.  id 0 = <pad> (empty slot), id 1 = <unk> (never seen in training).

    Strings carry a field prefix ("item=leftovers", "move=surf") so an item and a move
    with the same name could never collide, and one shared table serves every field.
    """

    def __init__(self, itos: list[str] | None = None):
        self.itos = itos or ["<pad>", "<unk>"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, cat_strings, min_count: int = 3) -> "Vocab":
        """cat_strings: iterable of arrays/lists of strings. Keep strings seen >= min_count times.
        Rare strings fall back to <unk>, so the model learns a sensible 'rare thing' vector."""
        counts = Counter()
        for row in cat_strings:
            counts.update(s for s in row if s)
        itos = ["<pad>", "<unk>"] + sorted(s for s, c in counts.items() if c >= min_count)
        return cls(itos)

    def encode(self, strings) -> np.ndarray:
        """Array of strings (any shape) -> int array of ids; '' (empty) -> 0 (pad)."""
        flat = np.asarray(strings, dtype=object).ravel()
        ids = [0 if not s else self.stoi.get(s, 1) for s in flat]
        return np.asarray(ids, dtype=np.int64).reshape(np.shape(strings))

    def __len__(self):
        return len(self.itos)


# ------------------------------------------------------------------ one Pokemon -> token
def _mon_token(p, boosts, is_active: bool):
    """Describe one Pokemon `p` (poke_engine Pokemon). `boosts` = 5 stat stages (only for the active mon)."""
    types = list(p.types) + ["", ""]
    cats = [
        "species=" + norm(p.id),
        "item=" + norm(p.item),
        "ability=" + norm(p.ability),
        "status=" + norm(p.status),
        *["move=" + norm(m.id) for m in p.moves[:4]],
        *[""] * (4 - len(p.moves[:4])),           # fewer than 4 moves -> pad
        "tera=" + norm(p.tera_type),
        "type=" + norm(types[0]),
        "type=" + norm(types[1]),
    ]
    hp = p.hp / max(p.maxhp, 1)
    nums = [
        hp,                                         # 0 hp fraction (0..1)
        float(p.hp <= 0),                           # 1 fainted flag
        float(is_active),                           # 2 is this the mon currently on the field?
        p.level / 100.0,                            # 3 level (random battles: ~70-100)
        p.attack / 500.0, p.defense / 500.0,        # 4,5  stats scaled to ~0..1
        p.special_attack / 500.0, p.special_defense / 500.0,  # 6,7
        p.speed / 500.0,                            # 8
        *(b / 6.0 for b in boosts),                 # 9-13 stat stages -6..+6 -> -1..1 (active only)
        float(p.terastallized),                     # 14 already terastallized?
        0.0, 0.0,                                   # 15,16 unused here (used by FIELD token)
    ]
    return cats, nums


def _side_boosts(side):
    return [side.attack_boost, side.defense_boost, side.special_attack_boost,
            side.special_defense_boost, side.speed_boost]


def _side_field_nums(side):
    """8 numbers describing hazards / screens on one side."""
    sc = side.side_conditions
    return [
        float(sc.stealth_rock > 0),     # rocks up?
        sc.spikes / 3.0,                # 0..3 layers
        sc.toxic_spikes / 2.0,          # 0..2 layers
        float(sc.sticky_web > 0),
        float(sc.reflect > 0),          # screens (value = turns left)
        float(sc.light_screen > 0),
        float(sc.aurora_veil > 0),
        float(sc.tailwind > 0),
    ]


def ordered_mons(side):
    """Active mon first, then the rest in team order.  Returns list of (pokemon, is_active)."""
    a = int(side.active_index)    # the engine exposes the index as a string
    mons = [(side.pokemon[a], True)] + [(p, False) for i, p in enumerate(side.pokemon) if i != a]
    return mons


def tokenize_raw(state, perspective: str = "one"):
    """state -> (cat_strings (14, 11) object array, nums (14, 17) float32).
    perspective 'one' means side_one is "us" (our own team goes first); 'two' swaps."""
    me, opp = (state.side_one, state.side_two) if perspective == "one" else (state.side_two, state.side_one)
    cats = [[""] * N_CAT for _ in range(N_TOKENS)]   # SUMMARY row stays all-empty (pad)
    nums = np.zeros((N_TOKENS, N_NUM), dtype=np.float32)

    for base, side in ((OWN0, me), (OPP0, opp)):
        boosts = _side_boosts(side)
        for k, (p, active) in enumerate(ordered_mons(side)[:6]):
            c, n = _mon_token(p, boosts if active else [0] * 5, active)
            cats[base + k] = c
            nums[base + k] = n

    # FIELD token: weather & terrain as categorical, everything else numeric.
    cats[FIELD][0] = "weather=" + norm(state.weather)
    cats[FIELD][1] = "terrain=" + norm(state.terrain)
    nums[FIELD, :8] = _side_field_nums(me)
    nums[FIELD, 8:16] = _side_field_nums(opp)
    nums[FIELD, 16] = float(state.trick_room)
    # intern the strings: millions of rows share a few thousand distinct names, so store each once
    cats = [[sys.intern(c) for c in row] for row in cats]
    return np.asarray(cats, dtype=object), nums


def tokenize(state, perspective: str, vocab: Vocab) -> dict:
    """The friendly one-call version: returns tensors ready for the model."""
    import torch
    cats, nums = tokenize_raw(state, perspective)
    return {"cats": torch.from_numpy(vocab.encode(cats)), "nums": torch.from_numpy(nums)}


# ---------------------------------------------------------- action targets (policy side)
def action_slots(state, perspective: str = "one"):
    """Names of the 9 action slots: our active's 4 moves, then 5 bench Pokemon (in token order).
    Returns (move_names[4], bench_names[5]) normalised; missing -> ''."""
    me = state.side_one if perspective == "one" else state.side_two
    mons = ordered_mons(me)
    moves = [norm(m.id) for m in mons[0][0].moves[:4]] + [""] * 4
    bench = [norm(p.id) for p, _ in mons[1:6]] + [""] * 5
    return moves[:4], bench[:5]


def policy_target(state, policy: dict, perspective: str = "one"):
    """Search policy (action name -> visit share) -> (target[9], legal[9]).

    * '<move>' and '<move>-tera' both fold into the base move's slot (we ignore the tera choice).
    * 'switch <species>' goes to the bench slot holding that species.
    * legal[i] = 1 if the search considered that slot at all (the engine knows what is legal).
    """
    moves, bench = action_slots(state, perspective)
    target = np.zeros(9, dtype=np.float32)
    for name, share in policy.items():
        if name.startswith("switch "):
            key = norm(name[7:])
            if key and key in bench:
                target[4 + bench.index(key)] += share
        else:
            key = norm(name[:-5] if name.endswith("-tera") else name)
            if key and key in moves:
                target[moves.index(key)] += share
    legal = (target > 0).astype(np.float32)
    total = target.sum()
    return (target / total if total > 0 else target), legal
