# Feature version 3: pairwise matchup features (589.57)

`features_v3 = features_v1 (738) ++ 188 matchup features` = 926 values (`engine/src/features_v3.rs`).
v1 and v2 are unchanged and stay selectable (`train.py --features v1|v2|v3`, also `1|2|3`;
`poke_engine.state_features(state, side, 3)`; the `.pev` header records the version).
v3 does not include the v2 extras. Policy nets (`PolicyNet`) accept only v1 / v2.

Motivation: the value net saw only per-slot raw stats, so it could not express "their Cinderace
outspeeds and OHKOs all three of our remaining Pokemon" or "our Iron Valiant is the only
revenge-killer for their sweeper".

## Layout

Slots are ordered like v1 (slot 0 = active, then team order); only alive Pokemon take part.
Five 6x6 blocks indexed `match.<channel>.own<i>.opp<j>`:

| channel | value |
| --- | --- |
| `speed` | +1 if own i moves before opp j, -1 if after, 0 on a tie. Effective speed: the active's speed stage, paralysis, Choice Scarf, Tailwind, Swift Swim / Chlorophyll / Sand Rush / Slush Rush / Surge Surfer, Trick Room inverts. |
| `dmg_own` | best of i's four moves vs j, max roll, as a fraction of j's CURRENT hp, capped at 1 |
| `dmg_opp` | same, j attacking i |
| `prio_ko_own` / `prio_ko_opp` | 1 if i (j) has a damaging priority move that KOs the other (max roll) |

Eight aggregates `match.agg.{own,opp}.{sweepers,answers_min,answers_mean,unanswered}` (counts / 6).
`can_rk(a, b)` = a KOs b (max roll) and (a moves first or a has a priority KO).
`sweepers` = attackers with `can_rk` against EVERY alive opposing Pokemon; `answers_min` /
`answers_mean` = min / mean over the opposing alive Pokemon of the number of attackers that can
revenge-kill it; `unanswered` = opposing Pokemon nobody of the attacking side can revenge-kill.
`own.*` = our attackers vs their team, `opp.*` = theirs vs ours.

## Damage approximation (cost trade-off)

The evaluator runs at every MCTS leaf, so `calculate_damage` (up to 288 calls per state) is too slow.
The damage is the `features_v2::assess` approximation made per move and per pair: level formula x base
power x stat ratio x STAB x type effectiveness x burn / Life Orb / Choice item / sun-rain, defender
ability immunities (Levitate, absorb abilities) and Assault Vest, max roll, no accuracy weighting.
Not modelled: screens, most ability modifiers, multi-hit and variable-power moves, crits, hazards
chip, Tera that has not happened yet. Boosts apply only to each side's active Pokemon (the engine
stores them per side).

## Cost

Everything is recomputed per evaluation (no cache): about 0.8-1.5 us per state on the dev machine,
versus ~0.15-0.3 us for the whole v1 sparse-MLP leaf evaluation (see `examples/bench_net.rs`,
`v3 matchup` row and `zero16v3`). MCTS iterations/s with a 16-hidden v3 net were ~0.70x stock
(v1 net: ~0.92x), i.e. roughly 24% fewer iterations/s than the v1 net at equal time.
The MLP fast path adds the matchup rows straight into the first-layer accumulator (`AccSink`),
no dense vector is built; it matches the dense reference (`fast_path_matches_reference`).
