# engine/ - our fork of poke-engine (learned leaf evaluation)

A fork of [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine)
(MIT, see `LICENSE`; upstream README kept as `README.upstream.md`) that lets
MCTS score leaf positions with a learned value function instead of the
hand-written `evaluate()`.

- **Origin:** upstream commit `c2dbbda7820137f938361106c1d1e7e1922fcb64` (v0.0.48).
  The tracked files of that commit were copied here (CI config under `.github/`
  was dropped).
- **Version:** `0.0.48+sdb.2` (Cargo crates and the Python wheel). The package
  name stays `poke-engine` / `import poke_engine`, so it is a drop-in for the
  upstream wheel. Check which one you have with `poke_engine.FEATURE_VERSION`
  (exists only in the fork).
- **Build:** `scripts/build_engine.ps1` (runs `cargo test --locked`, then builds
  with `--features poke-engine/terastallization --no-default-features --locked`
  and installs into the project `.venv` only). Foul Play's venv keeps its
  unmodified wheel.
- **Scope:** the hook supports the "genx" engine (gens 4-9; we build gen 9).
  The gen1-gen3 engines are not wired up in this fork and will not compile.

## Our changes (everything else is upstream, untouched)

| File | Change |
| --- | --- |
| `src/features.rs` (new) | `features(state, side) -> Vec<f32>`: fixed-length state encoding. Layout in `FEATURES.md`. `FEATURE_VERSION = 1`, `FEATURE_LEN = 738`, `feature_names()`. |
| `src/learned.rs` (new) | `Evaluator` (linear and 1-hidden-layer ReLU MLP, pure Rust, no new dependencies), weights-file load/save, `LeafEval` (stock-or-learned switch used by MCTS). |
| `src/mcts.rs`, `src/mcts_threaded.rs` | Leaf evaluation goes through `LeafEval`. New `perform_mcts_with` / `perform_mcts_shared_tree_with` take `Option<&Evaluator>`; the old `perform_mcts` / `perform_mcts_shared_tree` are unchanged wrappers passing `None`. `Node::rollout` takes an extra `&mut LeafEval`. |
| `src/lib.rs` | `pub mod features; pub mod learned;` |
| `poke-engine-py/src/lib.rs` | `Evaluator` class, `state_features`, `state_feature_names`, `FEATURE_VERSION`, `FEATURE_LENGTH`; `mcts(..., evaluator=None)`. |
| `poke-engine-py/python/poke_engine/__init__.py`, `poke_engine.pyi` | `monte_carlo_tree_search(..., evaluator=None)`, `write_linear`, `write_mlp`. |
| `Cargo.toml` | `default = ["terastallization"]` (was `[]`) so a bare `cargo test` builds the Gen 9 engine; our builds still pass `--no-default-features` explicitly. |
| `Cargo.toml`, `poke-engine-py/Cargo.toml`, `poke-engine-py/pyproject.toml`, `Cargo.lock` | Version `0.0.48+sdb.2` (lockfile updated accordingly; dependencies unchanged). |

### Robustness and item fixes (`0.0.48+sdb.2`)

Found by fuzzing about 3,000 converted live states (beads `showdown-bot-3m0`).
All three are in the genx engine.

| File | Change |
| --- | --- |
| `src/genx/generate_instructions.rs` | **Sleep:** `chance_to_wake_up` is now total over every `i8`. `sleep_turns <= 0` never wakes, `1` wakes with 1/3, `2` with 1/2 and `>= MAX_SLEEP_TURNS` (3) always wakes, which matches Showdown's gen9 `slp` (duration rolled 2-4, wake on the attempt where the counter hits 0). Before, `sleep_turns > 3` gave 1/0 = inf and MCTS panicked with `NonFinite`. A `debug_assert!` guards that the chance is finite and in [0, 1]. |
| `src/genx/generate_instructions.rs` | **Encore with no known last move** (`last_used_move` is `None` or a switch, which a state converted from a live battle can have) no longer panics. Encore is treated as inactive for that turn: the chosen move is not rewritten and the Encore counter does not advance. Move generation already offers every move in this case (see `add_available_moves`). With a known last move, behaviour is unchanged (locked into that move, three turns). |
| `src/genx/items.rs`, `src/genx/generate_instructions.rs` | **New items** `LIGHTCLAY`, `LEPPABERRY`, `SCOPELENS` (ids `lightclay`, `leppaberry`, `scopelens`). The `Items` enum had reached the 256-variant limit of `repr(u8)`, so it is now `repr(u16)`. |
| | *Light Clay:* Reflect, Light Screen and Aurora Veil last 8 turns when the user holds it (`LIGHT_CLAY_DURATION`); Safeguard and Mist stay at 5. |
| | *Scope Lens:* +1 crit stage in the crit-branch logic (`crit_rate`): 1/24 becomes 1/8 and the high-crit-ratio stage (1/8) becomes 1/2. Guaranteed crits and Battle/Shell Armor are unchanged. |
| | *Leppa Berry:* at end of turn, if a move is at 0 PP, restore 10 PP to the first such move and consume the berry (`DecrementPP` with `amount: -10` plus `ChangeItem`, both reversible). Differences from Showdown: the engine only tracks PP below 10 and has no max PP, so the restore is not capped at max PP, and Showdown's fallback to "first move below max PP" is not modelled. Showdown eats the berry right after the move that empties the slot; we eat at end of turn, which is the same unless the Pokemon would have to act again in between. |

Side effect of `Items` becoming `repr(u16)`: `Instruction` (which embeds two
`Items` in `ChangeItem`) grew from 6 to 8 bytes, so `instruction::test::test_instruction_size`
now asserts 8. Stock-search throughput measured the same within noise
(`scripts/engine_bench.py`, interleaved runs against the `sdb.1` wheel).
`scripts/engine_fuzz.py` fuzzes random valid states (sleep 0-10, Encore with
and without a last move, the new items) with short MCTS runs; 0 panics (the
`sdb.1` wheel panics on about half of them).

Scan of `.cache/randbats/full/gen9randombattle.json` (5,802 sets) against the
engine's item, ability and move ids: the only unknown ids were `lightclay`
(19 sets, summed weight 3267, 4 species), `leppaberry` (1 set, weight 344,
1 species) and `scopelens` (9 sets, weight 173, 1 species). No unknown
abilities or moves.

### Mechanics fixes (`0.0.48+sdb.22`, 589.68.4)

Found by `scripts/engine_diff.py` (engine vs the real Showdown simulator, 200 seeds, see
`docs/engine-diff-report.md`): turns where no engine branch reproduces the simulator went from 8.3% to 5.5% (PP
excluded), and from 85% to 5.8% with PP included. Tests: `src/genx/sdb22_tests*.inc.rs` (included in the
`generate_instructions` test module) plus `tests/test_battle_mechanics.rs`.

| Area | Change |
| --- | --- |
| End of turn (`add_end_of_turn_instructions`) | Reordered to Showdown's `onResidualOrder`: weather damage + weather abilities (Solar Power, Dry Skin, Rain Dish, Ice Body), Future Sight, Wish, Grassy Terrain heal (not on the terrain's last turn), Hydration / Shed Skin, **Leftovers / Black Sludge (before** Leech Seed, poison / toxic / Poison Heal and burn), Salt Cure / bind, Yawn, Perish, orbs / Speed Boost / forme checks. New `item_residual_heal`, `ability_weather_residual`, `ability_status_cure_residual`. Black Sludge (non-Poison) and Solar Power damage were clamped to the missing HP instead of the current HP. |
| Heal Block, Throat Chop | New `VolatileStatusDurations.healblock` / `.throatchop` (turns remaining; Heal Block 5, Psychic Noise 2, Persistent 7; Throat Chop 2). They tick in the residual step and end the volatile. Heal Block blocks Leftovers, Black Sludge, Grassy Terrain, Wish, Leech Seed's heal, Poison Heal, Ice Body / Rain Dish / Dry Skin and removes heal moves from the choices; Throat Chop blocks sound moves. Psychic Noise and Throat Chop are now 100% secondaries (Protect, Substitute, Covert Cloak and Sheer Force stop them). |
| PP | `Move.max_pp` (0 = unknown). With `max_pp > 0` every move decrements PP (clamped at 0, Pressure = 2), not when the user is asleep / fully paralysed / flinched or on the 2nd turn of a charge / locked move, but also on a failed Protect and by a sleeping Sleep Talk user. With `max_pp == 0` the old "only below 10" behaviour is kept (this is what hand-built states and old records get). Leppa Berry restores `min(10, max_pp - pp)`. Cost: about 6% MCTS iterations/s with tracking on (`scripts/engine_bench_pp.py`). |
| Moves / abilities / items | Take Heart (+1 SpA/SpD, cures status); Sheer Force suppresses Life Orb recoil (`Choice.sheer_force_applied`); a KO with Giga Impact / Hyper Beam keeps `mustrecharge` (the faint follow-up `(switch, none)` no longer consumes MUSTRECHARGE / TRUANT of the waiting side); Protean / Libero compare the whole type list (Water/Dark + Water move -> pure Water, once per switch-in via TYPECHANGE); Protosynthesis / Quark Drive activate when sun / Electric Terrain starts with the mon already out (`ability_field_change_activation`, after each half-turn); Disguise is not busted by an immune hit; Electromorphosis / Wind Power give Charge and Charge is used up by the next Electric move; Seismic Toss / Night Shade / Super Fang / Endeavor / Final Gambit no longer go through Protect. |

**State format.** Everything is backward compatible: `Move` serializes as `ID;disabled;pp` and appends `;max_pp` only
when it is known; `VolatileStatusDurations` stays `confusion;encore;lockedmove;slowstart;taunt;yawn` and appends
`;healblock;throatchop` only when one is non-zero. Old strings (including the state strings in `data/selfplay`
records) still parse; a `HEALBLOCK` / `THROATCHOP` volatile with duration 0 (old strings, the live bot) is read as a
fresh 5 / 2 turns. A new wheel's strings are readable by older wheels as long as no max PP / Heal Block timer is
set. Python: `Move(id, pp, disabled, max_pp=0)`, `VolatileStatusDurations(..., healblock=0, throatchop=0)`;
`src/showdown_bot/engine_state.py` passes `max_pp` only when `engine_state.TRACK_PP` /
`SHOWDOWN_ENGINE_TRACK_PP=1` is set (self-play, the diff harness).

Not done yet (plans in `docs/engine-diff-report.md`): Harvest, forme changes (Terapagos, Mimikyu-Busted, Eiscue,
Morpeko), Slow Start, Cursed Body, Unburden, Synchronize, a Struggle choice.

With `evaluator=None` (the default) the search is the stock algorithm: the same
`evaluate()` and the same `sigmoid(0.0125 * (eval - root_eval))` formula (the
only difference is one `Option` branch per leaf).

## Using it from Python

```python
import poke_engine as pe

pe.write_linear("lin.pev", weights, bias)            # weights: FEATURE_LENGTH floats
pe.write_mlp("mlp.pev", w1, b1, w2, b2)              # w1: [hidden][FEATURE_LENGTH]
ev = pe.Evaluator.load("mlp.pev")                    # or Evaluator.linear(...) / Evaluator.mlp(...)
result = pe.monte_carlo_tree_search(state, duration_ms=1000, evaluator=ev)
pe.state_features(state)                             # list[float], len FEATURE_LENGTH, side-one view
pe.state_features(state, side=2)                     # side-two view
```

### Weights file format

Little endian: `"PEVN"`, `u32 format`, `u32 feature_version`, `u32 kind`
(1 linear, 2 mlp), `u32 n_in`, `u32 n_hidden`, then the f32 weights (linear:
`w[n_in], b`; mlp: `W1[n_hidden][n_in], b1, w2, b2`).

**Format 2 (residual evaluators)** inserts two fields after `n_hidden`:
`u32 flags` (bit0 = residual) and `f32 stock_coef`. A residual evaluator's logit
is `stock_coef * evaluate(state) + net(features)` (the net's bias included);
`stock_coef = a * 0.0125`, the multiplier on the raw stock evaluation points.
**Format 3 (feature version 4, embedding tables)** is written only for nets with
learned identity embeddings; see `docs/features-v4.md` (`write_mlp_embed`,
`Evaluator.embed_dims`, `state_identity`, `identity_vocab*`). Formats 1 and 2 are
unchanged and still load.
Format 1 files load unchanged (not residual). Plain evaluators are still
written as format 1, so older tooling keeps working; only a residual file needs
a v2-aware reader. `write_linear` / `write_mlp` take `stock_coef` (default 0.0 =
plain format 1); `Evaluator.residual` and `Evaluator.stock_coef` report it.

`w1` follows the `torch.nn.Linear(FEATURE_LENGTH, hidden).weight` layout, so a
trained model exports with `write_mlp(path, l1.weight, l1.bias, l2.weight, l2.bias)`
(torch tensors and numpy arrays are accepted).

## Value semantics (how the model output maps into the search)

The model outputs a single number `z`, the **logit of P(side one wins)**,
computed from `features(state, SideOne)`. It is evaluated at the root once
(`z_root`) and at each non-terminal leaf (`z_leaf`), with the same function.

- **Relative mode (default):** leaf score `= sigmoid(z_leaf - z_root)`.
  This is exactly the shape of the stock scheme `sigmoid(eval - root_eval)`;
  only the unit changes: stock scales hand-made points by 0.0125 so that ~200
  points saturate, whereas `z` is already in log-odds, so no scale factor is
  applied. Terminal states keep their exact 1.0 / 0.0 values (as in stock).
- **Absolute mode:** `Evaluator.load(path, relative_to_root=False)` uses
  `sigmoid(z_leaf)`, the AlphaZero convention (score is the predicted win
  probability).

Why relative is the default: (1) it keeps the existing UCB1 constants and the
rest of the search behaving as tuned, because rewards stay centred on 0.5 at
the root; (2) evaluating the root with the *same* evaluator cancels the net's
position-dependent bias (a net that is 0.8 confident everywhere in this game
would otherwise squash all leaf scores toward 1 and flatten the UCB signal);
(3) it makes a model's output differences, not its calibration, what steers
search, which is robust for the early, weak networks of the self-play loop.
Absolute mode is there for the case where calibrated probabilities matter
(e.g. comparing against a resign threshold); for training targets, take
`sigmoid(z)` of the final model, not the search score. Because the sigmoid is
monotone, both modes order leaves identically for a fixed root.

## Weights file format (`.pev`, little endian)

```
"PEVN"            4 bytes magic
format            u32 = 1
feature_version   u32 (must equal the engine's FEATURE_VERSION on load)
kind              u32: 1 = linear, 2 = mlp
n_in              u32 (must equal FEATURE_LEN)
n_hidden          u32 (0 for linear)
linear: w[n_in] f32, b f32
mlp:    W1[n_hidden][n_in] f32 (row-major), b1[n_hidden], w2[n_hidden], b2   (f32)
```

Loading rejects a wrong magic, format, feature version, input size or file
length. The MLP is `z = w2 . relu(W1 x + b1) + b2`.

## Performance notes

The feature vector is sparse (mostly 0/1 indicators), and both evaluators skip
zero inputs; the MLP stores W1 transposed so each active input adds one
contiguous column. Evaluation allocates nothing (per-worker scratch buffers).
Measured numbers: `python -m uv run python scripts/engine_bench.py`
(see `docs/engine-value-hook.md`).

## Tests

- Rust: `cd engine && cargo test` (same as `cargo test --features terastallization --no-default-features`)
  (feature length and known values, linear/MLP outputs for hand-set weights,
  weights round-trip, score mapping).
- Python: `tests/test_engine.py` (needs the fork installed via
  `scripts/build_engine.ps1`).
