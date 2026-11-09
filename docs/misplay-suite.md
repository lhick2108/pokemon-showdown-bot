# Misplay regression suite

A set of positions that probe the bot's recurring ladder misplays, with a runner that
reports the pass rate per category for any bot spec (`scripts/misplay_suite.py`, suite in
`tests/puzzles/misplays/misplays.json`). It is a **diagnostic**, not a training target:
never train on these positions, and do not tune the evaluator until they pass.

## Categories

| category | what the bot does wrong | intended answer |
| --- | --- | --- |
| `ko_value` | a KO on a weakened Pokemon is available and it plays a slightly weaker attack, sets up or switches | any KO move |
| `heal_loop` | heals (Slack Off, Recover, Strength Sap, Rest ...) while a KO is on the board | an attack that KOs |
| `sub_loop` | repeats Substitute / Protect instead of attacking | the attack |
| `toxic` | Toxic is not valued on a healthy bulky target | Toxic |
| `taunt` | Taunt (0 of 26 possible uses) | Taunt vs recovery / setup / hazard users |
| `encore` | Encore not used on a setup or recovery move just used | Encore |
| `tera` | Terastallizes when it does not need to | the same move without Tera |
| `sleep_clause` | lets the sleeping Pokemon (our Sleep Clause shield) faint | switch it out to a healthy Pokemon |
| `role` | spends the role player (Blissey, Corviknight) on the wrong attacker | the wall that matches the attacker |
| `pivot` | Palafin never pivots out to become Hero | Flip Turn |

Not covered: misplay 2 (damage rolls averaged below depth 1). The engine's own playouts use the
same averaged damage, so no played-out check can confirm a roll puzzle; see the note at the end.

## Evidence

A candidate is kept only when at least one of two independent checks supports the intended answer
(both are recorded in the puzzle's `evidence` field):

* **long search**: a 2 s x 4-world search with the gen2 net *and* with the stock evaluation both
  put the intended answer on top;
* **played-out check**: force the first move, then let MCTS (stock evaluation, 15 ms per move) choose
  every later move for both sides, 20 games per first move. The intended move must beat the best
  other evaluated move by at least 0.25, and a second independent 30-game sample must still show a
  gap of 0.15 (guards against picking noise out of many candidates).

Positions come from the 35 ladder games in `data/ladder/ab2` (the `worlds` the live bot searched,
so the opponent sets are the sampled worlds, up to four per position) and from mechanical
generators (`scripts/misplay_synth.py`) for categories the logs barely contain. The generators
only propose; the checks above decide.

The **review** tier holds a few positions from the human game review (Slowking and Porygon2 healing
into a KO, the Scovillain Substitute loop, Taunt, Palafin) for which neither check confirms the
intended move: the bot's own long search and a played-out check with the same evaluator on both
sides reproduce the misplay, because those games are not decided by it (a KO now versus a KO
six turns later is a tempo and PP loss, not a lost game). They are reported separately and are
not part of the headline numbers.

## The suite today

60 validated puzzles (52 positives and 8 controls) and 11 review puzzles; 21 of the validated
positions come from the ladder games, 39 from the generators. Evidence: 25 long-search only, 15 played-out
only, 20 both.

| category | validated | controls | review |
| --- | --- | --- | --- |
| `encore` | 12 | | |
| `heal_loop` | 4 | 3 (healing is right) | 4 |
| `ko_value` | 7 | | 2 |
| `pivot` | 6 | | 1 |
| `role` | 4 | | |
| `sleep_clause` | 4 | | |
| `taunt` | 6 | | 3 |
| `tera` | 5 | 5 (Tera is right) | |
| `toxic` | 4 | | |
| `sub_loop` | | | 1 |

Controls guard against overcorrection: a net that "always attacks" or "never Terastallizes" must
still heal when healing is right and Tera when Tera is right.

Baseline (argmax, 4 worlds, 4 repetitions; fraction of samples passing):

| spec | time/world | validated | heal_loop | ko_value | tera | tera ctl | taunt | encore | pivot | role | sleep | toxic | review |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen2-res-searchabs-mlp16, scale 1.5 | 100 ms | 0.66 | 0.75 | 0.68 | 0.20 | 1.00 | 0.00 | 0.92 | 0.50 | 0.75 | 0.50 | 1.00 | 0.00 |
| gen2-res-searchabs-mlp16, scale 1.5 | 1000 ms | 0.75 | 1.00 | 0.68 | 0.40 | 1.00 | 0.58 | 0.83 | 0.42 | 0.75 | 0.75 | 1.00 | 0.02 |
| stock evaluation | 100 ms | 0.62 | 0.75 | 0.82 | 0.45 | 0.40 | 0.33 | 0.92 | 0.50 | 0.75 | 0.25 | 0.25 | 0.25 |
| stock evaluation | 1000 ms | 0.73 | 1.00 | 0.71 | 0.60 | 0.80 | 0.62 | 0.67 | 0.67 | 0.75 | 0.75 | 0.81 | 0.00 |

(`tests/puzzles/misplays/baseline_gen2.json` has the per-puzzle numbers for the gen2 rows.) The sampling error of a
category row with 4-6 puzzles is large; read the overall column and the large gaps (Tera, Taunt), not 0.1 differences.

## Running it

```powershell
# pass rate per category: evaluator + leaf scale, time per world, worlds per sample
python scripts/misplay_suite.py run --evaluator models/gen2-res-searchabs-mlp16.pev --scale 1.5 `
    --time-ms 100,1000 --worlds 4 --reps 2 --mode argmax [--tier all] [--cats toxic,tera] [--failures]
```

A sample is `--worlds` searches (the puzzle's sampled worlds, cycled), their visit shares summed, and
the argmax move compared with `accept` (or, with no `accept`, not in `reject`). `--reps` samples
are averaged, so one puzzle contributes a fraction. Tera puzzles compare the `-tera` variant
(`fold_tera` false): "play this move without terastallizing" passes. Without `--evaluator` the
stock evaluation is used.

Rebuilding the suite from new ladder games:

```powershell
python scripts/misplay_suite.py mine --records data/ladder/<run>/records --out c.json
python scripts/misplay_synth.py --out s.json --encore 40 --pivot 16 --sleep 14 --role 14 --ko 40 --heal 40 --taunt 18 --toxic 18
python scripts/misplay_suite.py verify --cands c.json --out c_v.json --evaluator models/x.pev --scale 1.5 --seconds 2 --worlds 4
python scripts/misplay_suite.py playout --cands c_v.json --out c_p.json --n 20 --ms 15
python scripts/misplay_suite.py playout --cands c_v.json --out c_conf.json --n 30 --ms 15 --seed 777 --only <ids that passed>
python scripts/misplay_suite.py build --cands c_p.json --confirm c_conf.json --out tests/puzzles/misplays/misplays.json
```

## Damage rolls (misplay 2)

`generate_instructions_from_damage` (engine/src/genx/generate_instructions.rs) branches on the
damage roll only when `branch_on_damage` is true, i.e. at the root and at the root's children, and
then only when the roll range straddles the target's HP (kill rolls versus non-kill average, crits
included). Deeper nodes use `0.925 x max`. A cheap improvement that keeps the search shape: keep the
kill-straddle branch (not the crit branch) at every depth, which adds branches only when a roll
actually decides a KO. That is a one-condition change in the engine (drop `branch_on_damage &&` from
the straddle test), but it changes tree width and the damage distribution of every caller, so it needs
a duel before it ships; it is not implemented here.
