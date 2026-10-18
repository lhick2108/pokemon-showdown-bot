# Engine vs Showdown differential test (589.68.2)

`scripts/engine_diff.py` (+ `scripts/engine_diff.js`) answers: *how often does our Rust engine fork disagree with
the real Showdown simulator, and about what?* This matters for the self-play RL work (epic 589.68): a policy trained
inside the engine learns the engine's bugs as facts.

First report (200 games, 6,983 turns): [engine-diff-report.md](engine-diff-report.md).

## Method

1. `scripts/engine_diff.js` plays seeded `gen9randombattle` games **in-process** in the local Showdown checkout
   (`vendor/pokemon-showdown/dist`, via `sim/battle`; no server needed, so it never touches the ladder run). Both
   sides pick random legal actions (12% switch, otherwise a uniform enabled move, 15% Terastallize while
   available), from a seeded PRNG. Everything is deterministic given the battle seed `S` (teams, rolls, choices):
   `node scripts/engine_diff.js <showdown-dir> 1 S` replays battle `S`.
2. Per **turn** (everything between two "move" requests, including pivot / faint-replacement switches) the driver
   emits the full true state before and after (both teams' exact stats, HP, PP, moves, items, abilities, status and
   counters, boosts, volatiles, side conditions, weather / terrain / Trick Room with durations), the two actions
   and the protocol log.
3. `engine_diff.py` builds a `poke_engine.State` from the **pre** snapshot (`build_state`; independent of the bot's
   poke-env based `engine_state.py` because it starts from the simulator's exact state, not from observations),
   runs `generate_instructions` for the same action pair, and for in-turn forced switches follows the engine's
   protocol (faint replacement: `(switch, none)`; pivot: `(switch, <saved move of the other side>)`).
   Every engine branch is applied with `apply_instructions`.
4. Each branch is compared with the simulator's **post** snapshot: HP (tolerance below), faint, status, item,
   ability, tera, types, PP, which Pokemon is active, boosts, volatiles, side conditions (+ remaining turns), weather,
   terrain, Trick Room. The turn **matches** if some branch has no mismatch; otherwise it **diverges** and the
   branch with the fewest mismatches (ties: highest probability) is reported.
5. Mismatches are attributed to a mechanic from the simulator log (`[from]` source, else the move being executed at
   the event). When the simulator logged nothing but the engine changed something, the cause is
   `engine-only[item=..,ability=..]` (the mon's item / ability are the suspects).

**HP tolerance.** The engine deals `0.925 x max roll` (the mean of Showdown's 16 rolls, 0.85-1.00), with KO
branches. A sim damage `d_s` and engine damage `d_e` agree when `|d_s - d_e| <= max(3, 0.11 x max(d_e, d_s), 3% of max HP)`.
Real errors below ~11% of the damage are invisible per turn (the aggregate ratio tables resolve ~1-2%).

**PP is reported separately.** Up to sdb.21 the engine only decremented PP below 10 (`pp < 10`), so PP differed in ~84%
of turns. sdb.22 tracks PP at any value for moves built with a known `Move.max_pp` (the harness passes the simulator's
`maxpp`; with a pre-sdb.22 wheel the old behaviour and the 84% remain). A "diverge" status ignores PP; PP-only turns are
`pp_only`.

## Running

Python is the project venv (never `uv sync`); the engine wheel on `PYTHONPATH` must be a build with
`Pokemon(times_attacked=)` for Rage Fist / Last Respects turns to be exact (older wheels still run; those turns
then show up as damage mismatches).

    python scripts/engine_diff.py run --games 100 --seed 1000 --out scratch/ed.json   # prints the report
    python scripts/engine_diff.py report scratch/ed.json --md docs/engine-diff-report.md --top 40
    python scripts/engine_diff.py repro 1005:7      # one turn (battle seed:turn): both states, every engine branch, log
    python scripts/engine_diff.py suspects          # controlled single-mechanic probes (Tera Shell, Booster Energy, ...)

Needs `node` and a built Showdown checkout (`SHOWDOWN_BOT_SERVER_DIR`, else `vendor/pokemon-showdown`, else the main
checkout's when run from a worktree). A game is ~35 turns; 200 games take ~30 s on one core.

Tests: `pytest tests/test_engine_diff.py` (a few seeded turns, ~2 s; skipped without node / Showdown).

## Reading the report

* **overall divergence** is the share of comparable turns where no engine branch reproduces the simulator, PP
  excluded. *Clean turns* exclude turns whose pre-state contains something the engine cannot represent at all
  (`unrepresentable`, e.g. Healing Wish slot conditions, Future Sight) or ids it does not know (`unknown ids`).
* **Divergence categories** are `(field, mechanic)` pairs, one count per turn. `converter risk = guess` marks
  categories where most turns carried a counter mapping that is an inference rather than a verified semantic
  (sleep / Rest counters, confusion / Encore / Taunt / locked-move / Yawn / Slow Start counters, weather / terrain /
  screen turns). Treat those as "maybe my conversion"; the rest are engine behaviour unless noted in the report.
* **By mechanic** merges fields; **Mechanics most associated** is a lift table over everything involved in a turn
  (both actives' species / ability / item, both moves, tera, weather): a high rate with many turns means that
  context is where the engine is least reliable.
* **Damage model check** uses clean single-target hits (one mon lost HP, in both engine and simulator, nothing
  healed, no crit): the median engine/sim ratio should be ~1.0 with +-8% spread. This is the number to watch for
  damage-formula bugs (Tera Shell, ability / item modifiers).
* **Known suspects**: natural-game aggregates (tag based, small n) plus `suspects` probes: fixed teams, 30 seeds,
  engine expected damage (probability-weighted over branches) vs the simulator's mean damage.

## Separating engine errors from conversion errors

* The builder re-reads what it fed in (`conv_errors`: HP, stats, PP, active index) - non-empty means a builder bug.
* Unknown species / ability / item / move ids decay to `none` in the engine; they are listed per turn (`unknown ids`)
  and the affected fields are not compared.
* Pre-state features without an engine counterpart are listed (`unrepresentable`) and counted separately.
* Counter semantics that were inferred: the `guess` flags above. Verified by hand while building the tool:
  toxic counter (engine `count` = Showdown `stage`), Wish (`startingTurn + 3 - turn`), Rest (`rest_turns` = remaining
  sleep attempts), pivot follow-ups.
* `engine_choice` strips the `switch ` prefix: `generate_instructions` takes bare Pokemon names, MCTS results use
  `switch <name>`.

## Limits

* Random play reaches odd states rarely (few late-game 1v1s, little Trick Room / terrain); the `suspects` probes cover
  single mechanics, not interactions.
* Turns with in-turn pivots are compared at the end of the turn only.
* The simulator stops at the winning faint; end-of-turn effects after it are not compared.
* Struggle turns are skipped (the engine has no Struggle choice); Recharge is sent as `none`.
* Crits / misses / secondaries are covered by "any branch", so a mechanic can be wrong yet match by luck on a turn;
  mismatch counts are lower bounds.
