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

World entry: `state` (the engine `State.to_string()` that was searched, sampled
opponent team included), `total_visits`, `avg_score` (root total score / total
visits, in the engine's [0,1] value scale from side one's view), and `moves`
(`choice`, `visits`, `score` for each of our options).

States are stored as strings, not features, so features can be regenerated with
`poke_engine.state_features(State.from_string(s), side)` after any
`FEATURE_VERSION` change. The string format itself is tied to the engine build;
the loader counts strings it cannot parse (`LoadStats.bad_states`).

### `result`
`result`: `"win"`, `"loss"` or `"tie"` from the **recording player's** view
(= side one's), and `turns`.

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
