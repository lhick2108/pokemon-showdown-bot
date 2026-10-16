# Feature version 4: dense v2 + v3 features and learned identity embeddings (589.65)

The v1-v3 value nets cannot see *which* Pokemon, item, ability or moves are on the field beyond
a few hand-picked one-hots (v2: 27 items, 48 abilities, move-property flags). Version 4 keeps
everything v2 and v3 compute and adds the raw identity of every slot as embedding-table indices,
so a wide net can learn "this is Great Tusk with Leftovers and Rapid Spin" directly.
`train.py --features v4`, `poke_engine.state_features(state, side, 4)`,
`poke_engine.state_identity(state, side)`; v1/v2/v3 and every existing `.pev` are unchanged.

## Layout

Two parts, both from `engine/src/features_v4.rs` (`FEATURE_VERSION_4`):

**Dense part, `FEATURE_LENGTH_V4` = 2379** = v1 (738) ++ v2 extras (1453, `features_v2.rs`) ++
v3 matchup block (188, `features_v3.rs`). No code is duplicated: the v2 extras are
`features_v2::extras_into`, the matchup block is `features_v3::matchup_blocks` emitted through a
`ShiftSink` that moves its indices to 2191.. (v3 on its own writes them at 738..). Feature names:
`poke_engine.state_feature_names(4)`.

**Identity part, `IDENTITY_LENGTH` = 84 indices**: for each of the 12 Pokemon slots (own slots
0..5, then opp slots 0..5; slot 0 is the active Pokemon, the rest in team order, exactly like v1)
seven indices `[species, item, ability, move0, move1, move2, move3]`, slot-major
(`ids[slot * 7 + k]`). Fainted Pokemon (hp <= 0) are all `0`.

## Vocabularies

One table per kind, `rows = enum variants + 2`:

| table | engine enum | rows (this build) |
| --- | --- | --- |
| species | `PokemonName` | 1471 |
| item | `Items` | 260 |
| ability | `Abilities` | 324 |
| move | `Choices` | 887 |

Row meaning: `0` = **pad** (a fainted Pokemon: the dense part already carries `alive`),
`1` = **unknown** (an unrevealed opponent item `UNKNOWNITEM`; an unknown ability, species or
empty/unrevealed move slot = the enum's `NONE`), `v + 2` for the enum variant with discriminant
`v`. `Items::NONE` ("no item", e.g. consumed) is a real entry (row 2), distinct from unknown.

The vocabularies are the engine's enums in declaration order (`define_enum_with_from_str!` now
exposes `NAMES` / `COUNT`), so they are deterministic and versioned with the engine data.
`vocab_hash()` (FNV-1a 64 over the table names and all variant names) fingerprints them; it is
stored in every v4 `.pev` and in `<net>.pev.vocab.json` (hash, sizes, dims and the full name
lists, written by `train.py`), and `Evaluator.load` refuses a file whose hash differs from the
loading engine's. **Append new items / abilities / moves / species only at the END of their
enums**; inserting in the middle renumbers everything behind it and invalidates every v4 net
(the loader then fails loudly instead of silently misreading). `tests/test_features_v4.py` pins
the sizes and the hash (`0x41bb936119eb13f2`); update them together with this file when an enum
changes on purpose.

Python: `identity_vocab_sizes()`, `identity_vocab(table)` (names in row order; 0 species,
1 item, 2 ability, 3 move), `identity_vocab_hash()`.

## Model and embedding format

`Evaluator` (Rust `learned.rs`) / `train.EmbedMLP` (PyTorch):

```
slot vector s = [E_species[id0] | E_item[id1] | E_ability[id2] | mean(E_move[id3..id6])]   (dims d_s, d_i, d_a, d_m)
input         = [dense (2379) | s(own0) | ... | s(own5) | s(opp0) | ... | s(opp5)]       (2379 + 12 * (d_s+d_i+d_a+d_m))
logit         = w2 . relu(W1 input + b1) + b2   (+ stock_coef * evaluate(state) when residual)
```

Moves are mean-pooled (order-free: the engine's move order is arbitrary), so the four move
rows are weighted 1/4 each. Dims are configurable per table (`train --embed-dim 8` or
`--embed-dim 16,8,8,8` = species,item,ability,move; default 8 each, so 12 x 32 = 384 embedding
inputs). `W1` is `[hidden][2379 + 12 * sum(dims)]`, torch `nn.Linear` layout.

### `.pev` format 3

Written only for embedding nets (formats 1 and 2 are written exactly as before and still load).
The 24-byte header keeps `magic, format = 3, feature_version = 4, kind = 2, n_in = 2379 (dense),
n_hidden`, then little endian:

```
flags       u32      bit0 = residual
stock_coef  f32      (0 when not residual)
vocab_hash  u64
dims        4 x u32  species, item, ability, move
W1[n_hidden][n_in + 12 * sum(dims)] f32   (dense columns, then per slot s|i|a|pooled-move)
b1[n_hidden], w2[n_hidden], b2
tables      species, item, ability, move: f32 [vocab_size][dim] row-major
```

Python writer: `poke_engine.write_mlp_embed(path, w1, b1, w2, b2, tables, dims, stock_coef)`
(`train.export_pev` calls it); parser: `train.read_pev` (returns `embed_dims`, `tables`,
`vocab_hash`). Loading rejects a wrong hash, zero or huge dims, a size mismatch and any
non-v4 / non-MLP format-3 file.

## Inference cost

The first layer is evaluated sparsely, as for v2: per-Pokemon contributions that change rarely
(types, tera, item flag, stats, **and now the identity embeddings**) are cached per search in
the `Scratch` cache lines (the cache key gained the species; item, ability and the four move ids
were already in the v2 key), the fainted line uses the pad identity. Per leaf only the dynamic
v1 blocks, the v2 side blocks and the v3 matchup block (recomputed per leaf, about 0.7 us) are
accumulated. Hidden sizes 8, 16, 32, 64 and 128 take the fast path (the AVX2 build when the CPU
has it); other sizes fall back to the dense reference path. Measured leaf rates are in the
589.65 bead notes / `docs/training.md`.

Not supported with v4: `--augment-bench` / `--augment-sides` (they permute v1 columns) and
`--aux on`. Policy nets (`PolicyNet`) accept only v1 / v2.
