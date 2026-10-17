# Token representation and shards for the transformer policy+value net (bead 589.68.1)

A Jaxcalibur-style token view of an engine `State`, produced by one Rust function and written to
memory-mappable shards by `src/showdown_bot/tokens.py`. The trainer (589.68.2) builds on the shard
format below. `TOKEN_VERSION` = 1, `SHARD_FORMAT` = 1.

Nothing is re-implemented: `engine/src/tokens.rs` writes the **dense v4 vector** once
(`features_v4`: v1 ++ v2 extras ++ v3 matchup block) and slices it into tokens; the categorical ids
are `features_v4::identity_into` (same vocabularies and **vocab hash `0x41bb936119eb13f2`** as v4,
`identity_vocab_hash()`); the action tokens are `prior::ActionCtx` features addressed by
`prior::action_slot`, the same 13 slots and the same `root_options` legality as
`poke_engine.policy_actions` / `policy_action_features`. Training and inference therefore cannot
disagree. Existing nets, `.pev` (formats 1-3) and `.ppol` files are untouched (only visibility of a few
constants changed in `features_v2/v3`).

## Engine API (`poke_engine`, wheel 0.0.48+sdb.20)

| function | returns |
| --- | --- |
| `state_tokens_str(state_string, side=1, opp_reveal=None, with_choices=False)` | `(cat: bytes, cont: bytes, choices: list[str])` |
| `state_tokens(State, side=1, opp_reveal=None, with_choices=False)` | same, from a `State` object (slower: object round trip) |
| `state_hidden_targets(state_string, truth_string, side=1)` | int16 bytes, see "Hidden-set targets" |
| `token_layout()` / `token_constants()` / `tokens.token_spec()` | field table (name, buffer, offset, shape, dtype), constants, vocab hash and sizes |
| `tokens_bench(state_string, side, n, scrub)` | mean seconds of the core computation |

`cat` is little-endian `int16`, `cont` little-endian `float32`; `tokens.split_buffers(cat, cont)`
returns named numpy views. `side` is the viewpoint (1 or 2): "own" is `side`, "opp" the other.
`choices` (only with `with_choices`) are the 26 slot names, own 13 then opp 13, `""` for an empty slot,
exactly the strings of `policy_actions` / the recorder's `policy` keys.

### `opp_reveal`: what the opponent has shown

`None`: the state is tokenised as it is; the `known` flags follow the state's own convention (item
`UNKNOWNITEM`, ability/move/species `NONE` = unknown) and `meta[2]` = 0.

A list of `(species, revealed_moves, item_revealed, ability_revealed, tera_revealed)` rows (the
recorder's `opp_facts`, names matched case/punctuation-insensitively): the **opponent side is scrubbed
before anything is computed** (also the matchup block and the opponent's action tokens): unrevealed
items become `UNKNOWNITEM`, abilities `NONE`, unrevealed moves `NONE` (revealed moves are compacted to
the front in their original order, so move position leaks nothing), unrevealed Tera types `TYPELESS`.
An opponent Pokemon with no row (the active one is always treated as revealed) is a fully hidden slot:
all its ids are `1` (unknown), its continuous features are zero except `is_opp`, and its matchup pairs
are zero. `meta[2]` = 1. This is the view a live bot has; sampled-world fillers do not leak. Known
leaks: the 8 matchup aggregates and `alive_fraction` of the opponent side are computed from the
unscrubbed team *membership* (not its sets).

## Layout

Slot order inside a side is the v4 order: slot 0 = active Pokemon, then the others in team order.
Own Pokemon are tokens 0..5, opponent tokens 6..11.

### `cat` buffer (`int16`, `CAT_LEN` = 262)

| field | shape | meaning |
| --- | --- | --- |
| `mon_cat` | 12 x 7 | `[species, item, ability, move0..move3]` embedding rows (v4 tables: 0 pad = fainted, 1 unknown, v+2 = enum variant). Sizes 1471 / 260 / 324 / 887. Tables 0,1,2 for columns 0,1,2; table 3 (move) for columns 3..6 |
| `mon_known` | 12 x 8 | 1 if known: `[species, item, ability, move0..move3, tera]`. Own side: 1 whenever the engine has a value. Opp: from `opp_reveal` or the convention |
| `act_cat` | 2 x 13 x 2 | action tokens (row 0 own, row 1 opp) `[move row, species row]`: slots 0-7 carry the active's move row (move table), slots 8-12 the switch target's species row (0 if fainted); the other column is 0 |
| `act_legal` | 2 x 13 | 1 if the slot is a legal action now (engine `root_get_all_options`) |
| `meta` | 4 | `[own active team index, opp active team index, opp scrubbed (0/1), 0]` (team indices index the state string's `pokemon[]`, needed to align per-team-index aux targets) |

Action slots (13): 0-3 the active's moves in state order, 4-7 the same with Terastallization, 8-12
switch to the bench Pokemon at slot positions 1-5 (slot order, active excluded). `choices[slot]` names
them.

### `cont` buffer (`float32`, `CONT_LEN` = 3911)

| field | shape | meaning |
| --- | --- | --- |
| `mon_cont` | 12 x 173 | `[0:53]` the v1 slot block (hp fraction, alive, status, types, tera one-hot, `terastallized`, has item, 6 stats/500; names `state_feature_names(4)[side*363 + slot*53 ...]`), `[53:171]` the v2 per-mon block (item/ability one-hots, move-property flags and scalars; v4 feature index `mon_base(side, slot)` = 738 + (side*6+slot)*118), `[171]` is_opp, `[172]` is_active. Hidden opponent slots are zero except `is_opp` |
| `side_cont` | 2 x 62 | per side (own, opp): `[0:45]` v1 side extras (7 boosts/6, 20 volatiles, 14 side conditions/hazards/screens, wish/future sight/force switch/alive fraction), `[45:62]` v2 side block (screen/tailwind turns and the active attacker-vs-defender block: speed order, best effectiveness, damage fraction, KO flags, ...) |
| `glob` | 15 | v1 field block (7 weather, 4 terrain, trick room) ++ v2 weather/terrain/trick room turns |
| `match_pair` | 36 x 5 | the v3 matchup channels per (own slot i, opp slot j) pair, pair index `i*6+j`; channels `speed, dmg_own, dmg_opp, prio_ko_own, prio_ko_opp` (v3 definitions, `docs/features-v3.md`). Pairs with a hidden opp slot are zero |
| `match_agg` | 8 | v3 aggregates (sweepers, answers_min, answers_mean, unanswered) for own then opp attackers |
| `act_cont` | 2 x 13 x 58 | `ActionCtx` features (`N_POLICY_AFEAT`) per slot, own then opp; zeros for an illegal slot. Equal to `policy_action_features(state, side)` rows |

The battle turn is not part of an engine `State`; the shards store it as the `turn` target/meta field.

Suggested token sequence for the model: 12 Pokemon tokens (embed `mon_cat` + project `mon_cont`, add
`mon_known`), 2 side tokens, 1 global token, 36 pair tokens or a pooled matchup token, 2 x 13 action
tokens (embed `act_cat`, project `act_cont`, mask with `act_legal`), 16 event tokens.

## Shards

`python -m showdown_bot.tokens export --source NAME=KIND:PATH ... --out DIR`

```
DIR/manifest.json
DIR/shards/<source>-NNNNN/<field>.npy        one .npy per field, shape [N, ...], np.load(..., mmap_mode="r")
```

`TokenDataset(DIR)` (`tokens.py`) loads `manifest["shards"]` entries as dicts of memory maps. One
sample = one decision (own side = engine side one of the recording). Fields (dtype, trailing shape):

**Inputs.** `mon_cat` i16 (12,7); `mon_known` i8 (12,8); `mon_cont` f16 (12,173); `side_cont` f16 (2,62);
`glob` f16 (15); `match_pair` f16 (36,5); `match_agg` f16 (8); `act_cat` i16 (2,13,2); `act_legal` i8
(2,13); `act_cont` f16 (2,13,58); `meta` i16 (4); `hist_cat` i16 (16,4); `hist_cont` f16 (16,5).

**Targets** (each with a validity field; train only on valid rows, never on the zero fill):

| target | fields | definition |
| --- | --- | --- |
| policy | `pol_target` f32 (13), `pol_valid` | self-play: the decision's aggregated search policy (mean of the worlds' visit shares) mapped to slots, renormalised, valid if >= 95% of the mass maps to a slot. Human: one-hot of the action played, valid if it is a legal slot. Mask logits with `act_legal[:,0]` |
| value, outcome | `z` f32 | 1 win / 0.5 tie / 0 loss from the own side's view (always present) |
| value, search | `v_search` f32, `v_search_valid` (0 none, 1 exported `search_value`, 2 reconstructed) | mean over the decision's worlds of the absolute P(own side wins). Missing in the recordings of gen5/6/8/9 (bead 589.62), so by default it is **reconstructed** from `avg_score` with the recording's own evaluator exactly as `dataset.py` does (`--no-reconstruct-q` leaves it missing). Human: never |
| opponent next action | `opp_act_kind` i8 (-1 unknown, 0 move, 1 tera move, 2 switch), `opp_act_id` i16 (move row or species row), `opp_act_valid`, `opp_act_slot` i8 (-1 if not found) | the action the opponent chose at the same turn, from the peer's record of the same battle (self-play: the other player's file; human: the other seat's file). `opp_act_id` is vocabulary-based and always usable when valid. `opp_act_slot` is its index among the 13 opponent action tokens of the (scrubbed) input, found by matching the id against `act_cat[1]`; -1 when the opponent's move is not revealed in the input |
| opponent hidden sets | `hid_ids` i16 (6,8), `hid_mask` i8 (6), `truth_team` i16 (6), `hid_valid` | self-play only (needs the recorded truth state). Row s = opponent slot s (token 6+s): true `[species, item, ability, move x4 (ascending rows), tera type]`, matched to the input's slot by species; `hid_mask[s]` = 0 when that species is not on the true team (a sampled filler). Tera type is the raw `PokemonType` discriminant (0-17 types, 18 Stellar, 19 none). `truth_team`: true opponent species rows. Combine with `mon_known` to weight the loss on what is still hidden |
| aux | `aux_turns_left` i16, `aux_final_alive` i8 (2), `aux_kos` i8 (12), `aux_alive_end` i8 (12; -1 unknown), `aux_valid` i8 (3) | `aux_valid` = `[turns_left, final_alive, kos/alive_end]`. Human (`docs/human-replays.md` "Aux targets"): all three; arrays re-ordered to token order (own 0..5 then opp 0..5, active first). Self-play: only `turns_left` (= result turns - turn) |

**Meta.** `source` i8 (index into `manifest["sources"]`), `split` i8 (0 train, 1 val, 2 test), `battle_hash`
i64 (blake2b of the battle tag; both sides of a battle share it: **split and group by it**), `turn` i16,
`decision_idx` i16, `rating` i8 (human elo bucket 0-4 = `human_policy.ELO_BUCKETS`, 5 unknown / self-play),
`state_kind` i8 (0 truth state, 1 sampled world), `reveal_src` i8 (1 = opponent scrubbed with `opp_facts`).

`split`: battles whose `battle_hash % 1000 < --val-permille` (default 50) are val, human `test` games are
test, everything else train. Float inputs are stored as float16 (values are within +-65000; clipped).

### Which state is tokenised

`--state auto` (default): the recorded truth state when there is one, else the first searched world;
`truth`/`world` force one. With the default `--scrub` the opponent is reduced to `opp_facts`, so a
truth state and a world of the same decision give (almost) the same tokens, and one sample per decision
is enough. Recordings without `opp_facts` are tokenised unscrubbed (`reveal_src` 0; the `known` flags
are then all 1 and carry no information). Human data: the converter state as is (`reveal_src` 0; the
first-person converter's opponent is revealed-so-far plus sampled fill, but the revealed/sampled split is
not stored). `--human-view species` hides everything but the revealed species (from `aux`).

## Event history (16 tokens)

Our records and the human conversions contain **no per-event log**, only per-decision state snapshots
plus `chosen` actions (self-play: both players' files, human: both seats). From those the writer
reconstructs, for decision `i`, the previous 8 decisions as 16 tokens, most recent first: row `2*(k-1)` is
our action of decision `i-k`, row `2*(k-1)+1` the opponent's action of the same turn (all-zero rows pad
the start of a game).

`hist_cat` = `[side (0 pad, 1 own, 2 opp), kind + 1 (1 move, 2 tera, 3 switch, 0 unknown), action id (move
or species row, 0 unknown), species row of the actor's active Pokemon]`; `hist_cont` = `[turns_ago/16,
hp change of the actor's active Pokemon, hp change of the other active Pokemon, fainted on the actor's
side/6, fainted on the other side/6]` where the changes are the state delta from decision `i-k` to
`i-k+1` (hp fraction of the Pokemon active at `i-k`, matched by species, 0 if it fainted).

Limits (what is missing for true events): the delta is per joint turn, not per action (damage cannot be
attributed to a move vs. residual damage); the executed action may differ from `chosen` (flinch, sleep,
full paralysis, a faster KO); forced switches after a faint, U-turn pivots and no-search forced moves are
not decisions in the data (self-play skips unsearched forced moves, `turns_ago` shows the gap); crits,
misses, status inflicted and items revealed are absent. A real event log would need the replay protocol
(`data/external/pokechamp`) tokenised alongside the state; not done here.

## Running it

```
python -m showdown_bot.tokens export \
    --source gen9=selfplay:data/selfplay/gen9 --source gen8=selfplay:data/selfplay/gen8 \
    --source hp=human:data/human/pc1-fp-aux/train --source hpt=human:data/human/pc1-fp-aux/test \
    --out data/tokens/v1 --workers 2
python -m showdown_bot.tokens info data/tokens/v1
python -m showdown_bot.tokens spec
```

Flags: `--workers N` (default 2; a task = `--games-per-shard` battles, so each worker holds about one
shard of samples, roughly 100-200 MB), `--max-records N` and `--max-games N` per source (testing),
`--max-per-game K` (random K decisions per game side), `--state`, `--no-scrub`, `--human-view`,
`--val-permille`, `--no-reconstruct-q`. A self-play source directory holds `<battle>__<player>.jsonl.gz`
files; the two files of a battle are paired by the name prefix to supply the opponent's actions.

Needs a wheel built from `engine/` of this branch (`scripts/build_engine_wheel.ps1`, which leaves
`.venv` alone and unpacks into `.cache/pe/site`; put that on `PYTHONPATH`).

## Measured (desktop busy with a ladder run, so pessimistic)

| | |
| --- | --- |
| core `state_tokens` (Rust, 12 tokens + matchup + 26 action tokens) | about 5 us, 6 us with `opp_reveal` |
| `state_tokens_str` from Python | about 37 us (30 us of it is `State::deserialize`) |
| `state_tokens(State)` from Python | about 42 us; `state_features(State, 1, 4)` alone is 66 us for comparison |
| writer, self-play gen8/gen9, 1 worker | about 1,200-1,500 samples/s (includes q reconstruction) |
| writer, human fp-aux, 1 worker | about 3,000 samples/s |
| disk | about 8.7 KB per sample |

Target coverage in the small samples (200 samples per source, first games):

| source | policy | v_search (exported / reconstructed) | opp action id (slot found) | hidden sets |
| --- | --- | --- | --- | --- |
| selfplay gen5 | 100% | 0 / 100% | 100% (61%) | 95.5% |
| selfplay gen6 | 100% | 0 / 100% | 29.5% (18.5%) | 82.5% |
| selfplay gen7 | 100% | 100% / 0 | 100% (52%) | 97.5% |
| selfplay gen8 | 100% | 0 / 100% | 100% (59.5%) | 98% |
| selfplay gen9 | 100% | 0 / 100% | 100% (52.5%) | 98% |
| human fp-aux train/test | 86% | none | 86.5% (70.5%) | none |

Gen6's low opponent-action coverage is missing peer files, not a bug. Human policy is missing exactly
where the converter could not recover the action (`chosen: null`: asleep, flinch, locked moves) plus
about 0.3% illegal under the hindsight Choice-lock guess.

### Full export estimate

Counts (files): gen5 12,000, gen6 17,981, gen7 6,000, gen8 6,000, gen9 4,000 recordings (gen1-4 not
included); human pc1-fp-aux about 91k train games and 33k test games (2 seats each). At the measured
samples per recording (gen5 37, gen6 8, gen7 35, gen8 37, gen9 33; human about 22): roughly 1.15 M
self-play and 5.4 M human samples, about **6.5 M samples, 56 GB** at 8.7 KB each. One worker: about 15 min
self-play + 30 min human plus about 10 s per 100k files of directory listing; two workers roughly halve it.
Use `--max-per-game` or restrict the human source to shrink the disk need (the human shards dominate).
