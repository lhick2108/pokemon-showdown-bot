# Self-play data (schema version 1)

Training data for expert iteration: what `SearchPlayer`'s MCTS did at every
searched decision, plus the game result. Code: `src/showdown_bot/recorder.py`
(write/read), `src/showdown_bot/dataset.py` (loader/CLI), `scripts/selfplay.py`
(generator). Output lives in `data/selfplay/<run-id>/` (git-ignored).

## Producing it

```
python -m uv run python scripts/selfplay.py --games 20 --pairs 4        # search vs search
python -m uv run python scripts/selfplay.py --vs foulplay --games 20 --pairs 2   # ours only
python -m uv run python -m showdown_bot.dataset data/selfplay/<run-id> [--check-features]
```

Recording is **off by default**. Turn it on with `SearchPlayer(recorder=GameRecorder(dir))`
or the env var `SHOWDOWN_RECORD_DIR=<dir>` (read once, in the constructor). When off, the
search path adds only `is None` checks.

## Files

One gzip JSONL file per (game, recording player):
`<battle-tag>__<username>.jsonl.gz`. In self-play both players record, so each
battle yields two files (the same battle seen from each side). A file is written
atomically when the game ends; games that never finish leave nothing. Forced
moves (single legal order) are not searched and not recorded.

Each line is one JSON object with a `type`: `game` (first line), `decision`
(one per searched decision, in order), `result` (last line).

### `game`
| field | meaning |
| --- | --- |
| `schema_version` | 1. Readers reject other values. |
| `feature_version` | `poke_engine.FEATURE_VERSION` at recording time (informational: features are recomputed on load). |
| `battle_tag`, `username`, `format`, `player` | identification |
| `role` | `"p1"` / `"p2"`, the server seat of the recording player |
| `config` | the `SearchConfig` (time per world, parallelism, worlds, threads, mode) |
| `created` | timestamp |

### `decision`
| field | meaning |
| --- | --- |
| `turn` | battle turn |
| `side` | always `"one"`: in every stored State, **our** side is engine side one (`side=1` for `state_features`) |
| `worlds` | list, one entry per searched world (see below) |
| `policy` | aggregated policy `{engine_choice: probability}` (mean of each world's visit share; sums to 1) |
| `chosen` | engine choice string actually played (`"closecombat"`, `"switch kingambit"`, `"x-tera"`); null if none |
| `order` | the poke-env order message sent (`/choose move ...`) |
| `compute` | `search_time_ms` per world, `worlds_requested`, `worlds_searched`, `parallelism`, `threads`, total `iterations`, `wall_ms` |
| `truth` | `{"state": "<State string>"}` or null (see below) |
| `truth_status` | `ok`, `no-peer`, `turn-mismatch`, `error`, or `disabled` |
| `opp_facts` | (added by 589.34; absent in older recordings) one row per revealed opposing Pokemon, active first (same order as the worlds): `species`, tracker `moves` / `item` / `ability` / `tera` revealed so far, `n_constraints`, `n_sets` (full randbats sets still consistent), `entropy` (bits, count-weighted), `n_sets_nt` / `entropy_nt` (same with the Tera type ignored), `vary` (how many distinct items / abilities / teras / move sets remain) and `groups` (up to 40 `[item, ability, "move/move/..", probability]` remaining candidates). `scripts/world_accuracy.py --by` turns these into accuracy-by-turn / by-reveals tables. |

World entry: `state` (the engine `State.to_string()` that was searched, sampled
opponent team included), `total_visits`, `avg_score` (root total score / total
visits, in the engine's [0,1] value scale from side one's view), and `moves`
(`choice`, `visits`, `score` for each of our options).

Optional world fields: `search_value` (the search's *absolute* P(side one wins) at the root;
unlike `avg_score` it needs no root evaluation to be added back; written for every searched
world of a recording made after the q-anchor change, single-threaded searches only, earlier
only with `export_nodes` > 0) and `nodes` (only with `selfplay.py --export-nodes K`, engine
0.0.48+sdb.9, bead showdown-bot-rnm; see "Interior nodes" below).

When `search_value` is absent the loader reconstructs it exactly from `avg_score` m with the
evaluator the recording used (header `config.evaluator`, `config.evaluator_scale` = k):
`q = sigmoid(logit(m) / k + z_root)`, `z_root = Evaluator.logit(world state)` (a residual net's
logit already contains its stock term; absolute, non-relative evaluators have `z_root = 0`).
Stock-evaluator recordings use `sigmoid(0.0125 * evaluate(state) + logit(m))`. `train.py
--q-evaluator PATH|stock --q-leaf-scale S` overrides the header; an evaluator ensemble
(`evaluator2`) or an unloadable evaluator file raises instead of falling back to the stock anchor.
`material` > 0 recordings follow the engine's own `root_search_value`, which ignores the blend.

States are stored as strings, not features, so features can be regenerated with
`poke_engine.state_features(State.from_string(s), side)` after any
`FEATURE_VERSION` change. The string format itself is tied to the engine build;
the loader counts strings it cannot parse (`LoadStats.bad_states`).

### `result`
`result`: `"win"`, `"loss"` or `"tie"` from the **recording player's** view
(= side one's), and `turns`.

## Interior nodes (`nodes`, bead showdown-bot-rnm)

Root decisions are one training position per search; the tree below the root has searched
millions of positions much more deeply. With `--export-nodes K` every searched world also
records up to K interior nodes of its MCTS tree: `[state, visits, value, static_value, depth]`.

* Sampling: after the search, the engine walks the tree and draws K nodes **without replacement
  with probability proportional to visits** (Efraimidis-Spirakis weighted reservoir), among
  non-terminal nodes with at least `--export-min-visits` (default 16) visits. The root is not a
  node. One entry per distinct position (a chain of no-op turns or a transposition keeps its
  best-visited copy). Cost: one tree walk plus K state serialisations, negligible next to the
  search. Off by default; single-threaded searches only (`--threads 1`).
* `state`: the position after `depth` joint turns of the tree, i.e. the root world with the
  chosen joint action (and one chance outcome, damage roll / crit / secondary effect) applied
  `depth` times. Side one is still the root's side one. The hidden opponent set is the root
  world's; training on it is as valid as on the root world.
* `value`: the node's mean backed-up leaf score over its `visits`, converted to an absolute
  P(side one wins). Scores in the tree are `sigmoid(s * (leaf - root))`, relative to the root's
  evaluation; the engine inverts that with the root's own value (stock evaluation x 0.0125, or the
  learned logit with its leaf scale), so `value` is anchored to the evaluator that ran the search
  and is on the same scale as the root's `search_value`. The mean is under the search's own
  exploration policy (UCB), so it estimates the position's value under good play by both sides
  from there, not the best-response value.
* `static_value`: the evaluator's own absolute value at the position (no search). `value -
  static_value` is what search added; a node whose `visits` are few is mostly its own leaf
  evaluation (a node's first visit *is* its evaluation), which is why training down-weights low
  visit counts and drops nodes below a threshold.
* `depth`: joint turns below the root (1 = a child of the root's joint-action matrix).

Load them with `dataset.iter_samples(..., nodes="on" | "only")` (kind `"node"`, with
`node_visits`, `node_depth`, `node_static`) and train with `train.py --nodes on|only` (see
the module docstring: weight = `node_weight * min(1, visits / node_visit_ref) *
node_depth_discount ** (depth - 1)`). Disk: about 3 KB of State string per node before gzip
(consecutive nodes share most of it); budget roughly 1.5-2x the size of a recording without
nodes at K = 8 per world.

## Ground-truth ("truth") states

When both players are `SearchPlayer`s in the **same process** (as in
`scripts/selfplay.py`), each registers itself by username. At a decision the
recording player finds its opponent's battle object for the same battle tag and
builds the State with our side as usual and the opponent's real team (species,
level, four moves, ability, item, Tera type) in place of a sample. HP, status,
boosts, side conditions and field come from the recording player's own
(public) view. This is recorded as `truth`, at the decision moment, before
searching.

Known gaps:
- Skipped (`truth_status: "turn-mismatch"`) when the two players' battle objects are
  not on the same turn yet (event-loop timing); typically a few percent of decisions.
- `no-peer` against Foul Play or any out-of-process opponent.
- Opponent stats are derived from the randbats EV/IV rules (as for sampled worlds),
  not read from the opponent's request; exact for random battles unless the engine's
  stat formula differs. Opponent exact HP comes from the public HP fraction.

## Loading

`dataset.iter_samples(root, sides=(1,), worlds=True, truth=True)` yields `Sample`
(`features`, `value`, `policy`, `kind` = `world`/`truth`, `side`, `battle_tag`, `turn`);
`dataset.iter_triples(...)` yields plain `(features, value_target, policy_target)`.

- `value_target`: 1.0 win, 0.5 tie, 0.0 loss from the viewpoint of the side the
  features describe. `sides=(1, 2)` also yields side two's view of each state
  (`state_features(state, 2)`, value `1 - v`, policy `None`).
- `policy_target`: the decision's aggregated `{choice: probability}`. Every world of a
  decision shares it. Indexing choices into a fixed action space is the trainer's job.
- All worlds and truth states of one game share its outcome: they are correlated; split
  train/validation by `battle_tag`, never by position.

Self-play with the same net on both sides makes the two files of a battle opposite
labels for nearly identical positions; keep both (or one) deliberately.

## Human records (`source: "human"`)

`scripts/human_replays.py` converts human ladder logs into this schema (truth states with
hindsight teams, no search, `policy: null`) plus separate policy rows; see `docs/human-replays.md`.

## Fork-augmented records (`source: "fork"`)

`scripts/fork_augment.py` (code: `src/showdown_bot/fork_augment.py`, bead showdown-bot-sd0) forks
recorded truth states, perturbs them (HP, status, stat stages, hazards, screens, weather,
terrain) and plays them out engine-only with one perfect-information MCTS per turn serving both
sides. Output uses this schema; the header adds `source: "fork"` and
`fork: {seed, origin, edited_state, edits, capped, edit_config}`. Each decision is side one's, with
the searched state both as `truth` and as the single `world` (so the search value exists for
`train.py --value-target search`; with `--states both` the position is therefore counted twice,
use `--states truth` to avoid that). `data/forks/<run>` can be passed to `train.py --data`
next to normal runs; split by `source` in analysis.
