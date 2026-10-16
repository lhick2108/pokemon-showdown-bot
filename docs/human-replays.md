# Human replay data (bead showdown-bot-zcz)

Human `gen9randombattle` server logs (PokeChamp dataset, 357k games) converted into

1. **value records** in the self-play format (`docs/selfplay-data.md`), so
   `train.py --data <out>/train --value-target outcome --states truth` works unchanged, and
2. **policy rows** (what the human actually did, in our engine's choice strings) for a future
   human-policy model / piKL regulariser.

Code: `src/showdown_bot/human_replays.py`, CLI `scripts/human_replays.py`, tests
`tests/test_human_replays.py`.

```
python scripts/human_replays.py --input data/external/pokechamp/gen9randombattle.jsonl.gz \
    --out data/human/run1 [--limit 2000] [--min-elo 1400] [--split train|test|all] \
    [--workers 6] [--seed 0] [--shard-games 250] [--no-policy] [--resume]
```

Input lines are JSON with `text` (the log), `month_year`, `elo` (bucket, `--min-elo` compares the
bucket's lower bound), `battle_id`, `split`. A process pool converts shards of `--shard-games`
games and prints a progress line per shard. Games that cannot be converted are skipped and
counted by reason in `<out>/manifest.json` (with up to 5 example battle ids per reason).
`--resume` skips shards that have a `.done` marker.

## Layout

```
<out>/<split>/shard-NNNNN/<battle_id>__p1.jsonl.gz     value records (one file per game side)
<out>/<split>/shard-NNNNN/<battle_id>__p2.jsonl.gz
<out>/<split>/shard-NNNNN/.done                        shard stats (for --resume)
<out>/policy/<split>/shard-NNNNN.jsonl.gz              policy rows, one JSON object per line
<out>/manifest.json
```

`<split>` is the dataset's own `train` / `test` field. `recorder.iter_game_files` ignores any
`policy/` directory, so `--data <out>` loads everything (train and test!); point `--data` at
`<out>/train` to keep PokeChamp's test games out of training. Both sides of a game share
`battle_tag` (= the dataset's `battle_id`), so the trainer's split-by-battle keeps them together.

## Value records

Same schema as self-play (`schema_version` 1), one file per (game, side). The recording side is
engine side one in every state (`role` = `p1` / `p2` is the server seat). Header extras:
`source: "human"`, `player: "human"`, `elo`, `month_year`, `split`; `username` is just the seat
(`p1`/`p2`). One `decision` per turn start (`|turn|N`), with `truth: {"state": ...}`
(`truth_status: "ok"`), no worlds, `policy: null`, `chosen` = the human's action (or null when it
is not recoverable), `legal` = the engine-derived options. `result` is the game outcome from that
side (`win` / `loss` / `tie`). There is no search, so `Sample.search_value` is `None`; the trainer's
`--value-target outcome` ignores it and the other targets fall back to the outcome `z`.

## Policy rows

One row per decision with a known action:

| field | meaning |
| --- | --- |
| `battle_tag`, `role`, `turn` | game id, seat (`p1`/`p2`), turn number |
| `elo`, `month_year`, `split` | from the source record |
| `result` | outcome from `role`'s view |
| `state` | the same engine `State` string as the value record's truth state (side one = `role`) |
| `action` | engine choice string: `closecombat`, `closecombat-tera`, `switch kingambit` (same mapping as MCTS output / `BuiltState.choice_to_order`) |
| `legal` | options derived from the state: moves with PP and not disabled, `-tera` variants when Tera is still available, switches to living bench Pokemon |
| `action_legal` | whether `action` is in `legal` (about 0.2% are not, see caveats) |

Only turn-start decisions are covered. Forced replacements after a faint and U-turn style pivots
are not separate decisions here. A decision's action is unknown (no policy row, `chosen: null`)
when the player's choice is hidden: `|cant|` (asleep, flinch, full paralysis, recharge), locked or
called moves (`[from]` moves, Outrage continuation, Dancer), Struggle.

## Pipeline and approximations

1. `sanitize_log` keeps battle events and the winner's seat only; chat, joins, `|player|`
   names, timestamps, `|-message|`, `|-hint|` and the side-condition lines' username
   (`|-sidestart|p2: <username>|...` becomes `p2: side`) never leave it. Lowercase details are
   fixed (`l79` -> `L79`) and `move:`/`ability:`/`item:` tags get their display casing back,
   because poke-env compares them case-sensitively.
2. **Hindsight teams** (Metamon style). The whole log is replayed through poke-env twice, once from
   each seat, with the existing `OpponentModel`; every reveal about the other side (moves,
   `[from] item:` / `[from] ability:` reveals, knock off / trick transfers, Tera type, negative
   evidence, speed evidence) narrows that side's `SetTracker`s. One full randbats set per revealed
   Pokemon is then sampled from the tracker (`SetTracker.sample`, level overridden by the log's
   level) and unrevealed slots are filled by `TeamSampler` under the randbats team rules
   (`es.sample_opponent_team`). Sampling is seeded by `--seed` and the battle id, so a run is
   reproducible.
3. **States.** For each seat the log is replayed a third/fourth time with that seat's hindsight team
   as a synthetic `|request|` (randbats EV/IV stats via `engine_state.randbats_evs_ivs`, exact
   moves, item, ability, Tera type) and the other team as `OpponentSlot`s. At every `|turn|N`
   `engine_state.build_state` makes the truth-like state. HP is exact for both sides (the log
   gives real HP for both), status / boosts / volatiles / side conditions / weather / terrain /
   Trick Room / Tera state / consumed items come from the log through poke-env, PP is poke-env's
   count of moves used (Pressure included), and a Choice-locked active Pokemon gets the other
   moves marked disabled the way the request would.
4. **Actions** are read from the events between `|turn|N` and `|turn|N+1`: a side's first plain
   `switch` or `[from]`-less `move` before `|upkeep|`, plus a Tera flag from a preceding
   `|-terastallize|`. A switch target is mapped to the engine id of that bench Pokemon in the
   state.

## Skips

Skipped and counted: `no_turns`, `no_result` (no `|win|`/`|tie|`), `illusion` (any `|replace|`,
Zoroark), `transform` (Ditto / Imposter), `ability_swap` (Trace, Mummy, Skill Swap, Role Play,
Wandering Spirit, Worry Seed, ...), `percent_hp` (log shows percentages), `unmapped_species`
(a revealed Pokemon with no randbats entry, e.g. CAP species), `unknown_species`, `no_switches`
and `error:<ExceptionType>` for anything unexpected.

## Caveats

- Hindsight sets are consistent with everything revealed, but the hidden parts are *sampled*; the
  same game converted with another `--seed` gives different unrevealed moves / items / bench.
  Randbats sets come from the current set data (`.cache/randbats`), while the games are from
  2024 (sets change between patches), so some revealed combinations conflict with every current
  role and the tracker keeps the least-conflicting roles.
- Damage-roll inference is off in the hindsight pass: the other side's real stats are not known
  yet when the opponent model sees the first replay (placeholder own sets with the right species
  and levels are used there), so only event-based and speed evidence narrows sets.
- Max HP of a hindsight set can differ from the log's (HP EV rules depend on the sampled
  moves / item): about 1% of Pokemon. Current HP is the log's exact value for our side (poke-env)
  and `round(computed_max * hp_fraction)` for the opponent slot, so that side is off in those cases.
- A Choice-item guess can wrongly lock the active Pokemon (the human then plays "illegal"
  moves): about 0.2% of recorded actions have `action_legal: false`. Trapping and Tera
  availability beyond "one Tera per side" are not in `legal`.
- Substitute HP, Wish, Future Sight and other things listed in the `engine_state` docstring are
  approximated exactly as in live search.

## First-person (hindsight-free) mode

`--first-person` (`Converter(first_person=True)`) changes only the opponent side of each state.
The default mode above gives both seats the same whole-game team picture, built from everything
revealed by the end of the log, so a model trained on it sees moves, items, abilities and Tera types
that were revealed *after* the decision. Live, the bot knows only what has been revealed so far and
fills the rest by sampling.

In first-person mode `states_for_role` replays the log as the actor with a dedicated
`OpponentModel`, feeding it every event (`observe_events`) before poke-env parses it. At each
`|turn|N` it calls `model.update(battle)` and `engine_state.sample_opponent_team(battle, trackers,
TeamSampler, rng)`, exactly the calls of `SearchPlayer` (one world per state), then
`engine_state.build_state`. The opponent's revealed Pokemon therefore carry only the moves / item /
ability / Tera type seen so far (the rest sampled from the set tracker's remaining candidates),
unrevealed bench slots come from the team sampler, and the two seats' rows of one turn have
different, independently sampled pictures of the opponent. The actor's own side is still the
whole-game hindsight team (the logs never show a player's unrevealed own sets, so this is the best
available "exact" team). `--fp-damage` also turns on damage-roll inference (the live default), off
by default for speed and because it needs the actor's real stats. Output layout, actions and
legal sets are unchanged; write to a separate `--out` (e.g. `data/human/pc1-fp`) so the modes are
not mixed. Test: `tests/test_human_replays.py::test_first_person_states_do_not_leak_later_events`
re-runs the sampler on a log cut at `|turn|k` and requires the identical state at turn k.
Throughput is about 2.4x slower than the hindsight mode.

## Aux targets (`aux` field of each decision)

Every value-record decision carries `aux`, derived from the protocol log (`game_facts`, `build_aux`):

```
{"turns_left": T_last - turn,            # last |turn|N of the game minus this decision turn
 "final_alive": [own, opp],              # alive count at game end, actor side first
 "own": {"ident": [6 x str|null], "alive_end": [6 x 0|1|-1], "kos": [6 x int], "kos_after": [6 x int]},
 "opp": {same four arrays}}
```

Arrays are aligned with the decision's engine `State` string: index i = `side_one.pokemon[i]` (own) /
`side_two.pokemon[i]` (opp), so slot 0 is the active Pokemon (the feature slot order). `ident` is the log's
ident name (the species id in randbats) and is `null` for an unidentified slot. `alive_end` is 1 if the
Pokemon never fainted, 0 if it did, -1 unknown; `kos` is the whole-game count of KOs the Pokemon scored,
`kos_after` those scored at turn >= this decision turn (the future part). Own side: all six are always
identified (the hindsight team). Opp side, hindsight mode: revealed Pokemon are identified, the never-
revealed slots have `ident` null and `alive_end` 1 (never entered, so never fainted). Opp side,
first-person mode: only Pokemon seen by that turn are identified; filler slots have `ident` null and
`alive_end` -1 (they may stand for a Pokemon revealed, and possibly fainted, later). KO credit: user of the
latest `|move|` for plain fatal damage, `[of]` mon for `[from]` damage (Rocky Helmet, ...); status,
weather, hazards, recoil and self-KOs credit nobody (Future Sight is credited to the latest mover).

Baseline: `scripts/hindsight_gap.py` trains the MLP on one conversion mode and scores it on both.
