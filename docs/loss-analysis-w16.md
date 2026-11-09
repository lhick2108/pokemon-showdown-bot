# Loss analysis of the w16 ladder run

Run `data/ladder/w16` (gitignored): 229 gen9randombattle ladder games, 131 W / 98 L (57.2%), opponents
almost all rated >= 2100 (221 of 229; 93 of 98 losses). Bot: DUCT MCTS, 16 sampled worlds x 1 s,
gen7-res-root-mlp16 evaluator, argmax, `prune_noop=false`, `endgame=off`, `tera_margin=0`.

Tooling: `scripts/loss_analysis.py` (pure python, no engine). It joins `games.jsonl`, the protocol logs
and the per-decision records, and prints the numbers below.

    python scripts/loss_analysis.py --run <dir>/data/ladder/w16 --summary
    python scripts/loss_analysis.py --run ... --game 2700183068 [--window 4 | --window -1]   # annotated timeline
    python scripts/loss_analysis.py --run ... --dump features.json

Judgements below are mine, not a strong player's: every class carries a confidence (H / M / L).

## Method and its limits

* **Value trace.** For every decision the record has 16 worlds, each with `avg_score` (root mean value for
  us, 0..1). The game value `v_t` is the mean over worlds at the first decision of turn t. The value is
  usable but noisy turn to turn (+-0.1) and well calibrated late (see Calibration), so "the turn it fell"
  is only a pointer.
* **Flip turn (point of no return).** Last turn with `v >= 0.45`; everything after is below. The events of that
  turn (and the next) are read from the log: crit / miss / full-para / flinch, status, opponent Tera,
  opponent boosts, our switch-ins, whether we lost a Pokemon. Rules in `primary_class()`; 4 games are
  hand-labelled in `MANUAL` (`futile_moves` x3, `tactical_choice_unlock` x1). I read the annotated timelines of
  roughly 55 of the 98 losses (all with a big drop, the rest sampled) to check the rules.
* **Wrong-world check.** For every fact the opponent revealed (move, item, ability, tera type) I counted in how
  many of the 16 worlds of the last decision before the reveal the opponent Pokemon had that fact.
* **Not covered.** Engine-vs-Showdown damage mismatches are not re-measured here (`docs/ko-check.md` and
  `docs/engine-diff-report.md` already do that: 99.35% KO-class agreement). I cannot say what a top player
  would have played; "no single event" below is "I could not find a clear mistake", not "no mistake".

## Summary table

Primary class of the flip turn, 98 losses (one class per loss):

| class | losses | % of losses | confidence | notes |
| --- | --- | --- | --- | --- |
| no single event (gradual loss / matchup / lost trades) | 48 | 49% | M that nothing simple is wrong, L on what is | value drifts down over 10+ turns; 36 of 98 losses last >= 35 turns |
| luck at the flip turn (crit on us, our miss, our full-para/flinch) | 21 | 21% | M; net excess over chance is about 7-11 | 2-turn window base rate is ~14% so ~7 are chance; see Luck |
| lost exchange (we lose a Pokemon at the flip turn, nothing unusual) | 11 | 11% | L | mostly an attacker we sent in is out-damaged |
| opponent set-up sweeper at the flip turn | 6 | 6% | L | not above base rate (14.4% of loss flips vs 13.3% of all turns) |
| futile / no-op moves (Glare on a paralysed mon, Headbutt into Ghost, Sub with a Sub up ...) | 4 | 4% | H that it is a bug class, L-M that it decided the game | all four are 45-113 turn games |
| world-model surprise (fact present in <= 2/16 worlds) | 3 | 3% | M | several more are parser artefacts (Trace, Embody Aspect) |
| opponent Tera flip | 2 | 2% | M | |
| status on us at the flip | 1 | 1% | L | |
| switch-in punished at the flip | 1 | 1% | L | but see Switching below |
| Knock Off un-locked a Choice mon | 1 | 1% | M | 2700092169 |

Cross-cutting tendencies (overlap the classes above, so not additive):

| tendency | evidence | confidence |
| --- | --- | --- |
| long games are lost | win rate by length: <25 turns 0.69, 25-34 0.56, 35-49 0.39, >= 50 0.30; 36 of 98 losses are >= 35 turns | H that it holds, L on cause |
| punished switch-ins | 32 of 98 losses have >= 2 switch-ins that fainted or lost >= 50% in the arrival turn, vs 11 of 131 wins; voluntary switches/turn 0.35 (L) vs 0.27 (W) | M (partly an effect of being behind: forced replacements into hazards count) |
| futile moves | non-protect "wasted" moves (target immune / move failed) per own move: wins 2.7%, losses 8.1% (5.3% without the one 113-turn game) | H |
| bad luck | luck index (crits, misses, full-para/flinch both ways) <= -2: win rate 0.36 (n=42) vs 0.58-0.69 elsewhere | M |
| status exposure | burn on us 0.30/game in losses vs 0.12 in wins, poison 0.26 vs 0.15, toxic 0.15 vs 0.06 (losses are 1.35x longer) | L |

## Base rates: wins vs losses (per game)

| | wins (131) | losses (98) |
| --- | --- | --- |
| turns | 26.3 | 35.0 |
| crits on us / for us | 0.63 / 0.75 | 0.95 / 0.58 |
| our misses / their misses | 0.33 / 0.37 | 0.66 / 0.43 |
| our lost turns (par/slp/frz/flinch) | 0.86 | 1.18 |
| our switches | 7.0 | 12.4 |
| punished switch-ins (any) | 0.44 | 1.22 |
| we Terastallize | 0.80 | 0.93 |
| world surprises (<= 2/16 worlds) | 0.31 | 0.51 |
| opponent positive offensive boosts | 6.1 | 6.5 |
| value at turn 1 | 0.51 | 0.51 |

Win rate by opponent rating: 2100-2250 0.63 (n=71), 2250-2400 0.64 (n=96), >= 2400 **0.41** (n=54); the
5 games below 2100 do not matter. Material is decisive: with equal Pokemon alive at turn 20 we win 46%, one
down 26%, one up 72%.

## Findings by class

### Luck (21 flagged; about 7-11 net; confidence M)

The point-of-no-return window contains a crit on us, a miss of ours or a full-para / flinch of ours in 21 of 98
losses (21% vs a ~14% chance baseline per two-turn window). Games with a luck index <= -2 are won 36% of
the time vs 58-69% in the other buckets: about 11 extra losses out of 42, i.e. roughly 11% of all losses. The
index also counts opponent full-paras we caused, which is partly skill, so I take 7-11, not 21.
Our own misses are twice as frequent in losses (0.66 vs 0.33 per game) while theirs are not (0.43 vs 0.37):
that part is probably the low-accuracy moves we pick (Focus Blast, Hurricane, Stone Edge, Fire Blast,
Blizzard) rather than a mechanic bug, but I did not verify the accuracy model.

Examples (all `battle-gen9randombattle-<id>`):
* **2700183068** (vs aceguy123, 2272), T26-T31: value 0.998-1.0, we have 5 Pokemon, the opponent has only a Shell
  Smash Smeargle. Two crits from Population Bomb kill Scrafty, later a third crit kills Arceus-Steel. The bot
  never dropped below 0.95 before T32. Lost from ~1.0; luck, confidence H. Also the single clearest
  "too confident" game.
* **2700198843** (2234), T20: Staraptor Brave Bird vs Iron Valiant at v = 0.96, crit on us.
* **2700105409** (2429), T19-T20: Gurdurr at v = 0.92 loses two turns to full paralysis (the paralysis came from
  a Tera-Electric Cresselia Thunderbolt); lost 0.92 -> 0.
* **2700078834** (2445): Mamoswine Icicle Crash crit (T16) and Maushold Population Bomb crit (T19).
* Misses at the flip: 2700054984 (Stone Edge + Gunk Shot misses), 2700079981 (Salazzle Fire Blast miss).

What should have happened: nothing fixable, except that high-variance lines (Population Bomb, 10 hits, each can
crit) are poorly represented by the engine's expected-value damage; I mark this L.

### Futile / no-op moves (4 decisive, ~125 moves in other losses; confidence H as a bug class)

`waste` events (our move answered by `-immune` / `-fail`): 225 in losses (8.1% of own moves) vs 100 in wins
(2.7%). Most frequent in losses: Glare fails x46, Headbutt immune x20, Protect fail x12, Substitute fail x12,
Earthquake immune x12 (some `-immune` are unknowable ability reveals such as Levitate / Earth Eater and are fine).

* **2700189152** (vs Antumbral, 2354), 113 turns: Dudunsparce uses Glare 46 times into an already-paralysed
  Giratina-Origin (T6-T64), then Headbutt x20 into the same Ghost/Dragon (T80-T98), then Earthquake into
  Corviknight. Value sat at 0.30 (then 0.03); the root visit shares were Glare 0.40 / Roost 0.15 / switches 0.10 each
  turn, i.e. the search sees no difference and the prior picks the no-op. Giratina meanwhile wins by
  Will-O-Wisp / Shadow Sneak chip. Should have switched to a real attacker or at least not repeated a provable no-op.
* **hs230knwpw** (battle-gen9randombattle-2700171683-...hs230knwpw, vs gyuuoy (2220), 81 turns): Venusaur
  Substitute fails x11 in a Sub / Leech Seed / Sleep Powder loop against Chansey + Vespiquen recovery.
* **2700102093** (88 turns): Thunder Wave fails x6.

`prune_noop` (`SearchConfig.prune_noop`, `players/search.py` ~L277, applied ~L1144-1190, engine proof in
`decision_quality.noop_choices`) exists but is **off** in this run. Its default `noop_vs_switches=True`
requires the move to be a no-op against every opponent switch too, which Glare-on-a-paralysed-mon is not (a
fresh switch-in could be paralysed), so even enabled it would not prune the Glare loop.

### Lost exchange / no single event (59 combined; confidence L)

No crit, no new information, no set-up; the value falls over many turns while trades go against us. Typical
shape (2700094369, 2525, 52 turns): Mandibuzz mirror in which the opponent's Roost / Defog / Foul Play outlasts
our Roost / Knock Off while Jolteon / Mandibuzz / Landorus take Entei Extreme Speed chip; value 0.5 -> 0.1 in
the last 15 turns. 2700098775 (2515), 2700088658 (2471, 74 turns), 2700243860 (2448, 58 turns, Latias Calm Mind
/ Recover vs Deoxys-Defense Cosmic Power) are similar. In these I could not point to one decision; the
evidence is statistical (long games and many punished switch-ins, below), not a named error. This is also where
a stronger evaluator or deeper search would help, which is out of scope here.

### Switching (suspected; confidence M-L)

Losses switch 0.35 times per turn vs 0.27 in wins. Voluntary switch-ins (switch before any faint in that turn
block, so pivots included) that arrive with < 40% HP: 12.1% (losses) vs 6.2% (wins); that faint on the arrival
turn: 7.6% vs 3.5%; that lose >= 50%: 9.0% vs 6.5%. 32 of 98 losses vs 11 of 131 wins have >= 2 punished
switch-ins. Examples: 2700090729 (2420) T11 Zacian enters at 19% HP and dies; 2700097963 (2432) T5 Lokix enters
into Wave Crash and drops to 16%. Because losing sides also switch more (forced replacements into
Stealth Rock + a hit), I cannot separate cause from effect.

### Wrong world / set inference (3 decisive; confidence M that this is small)

Mean share of worlds containing the fact at the moment it was revealed: moves 0.82 (wins) / 0.82 (losses),
items 0.82 / 0.79, abilities 0.83 / 0.85, tera type 0.61 / 0.61. Facts present in <= 2 of 16 worlds: moves 1.9% /
2.7%, items 1.5% / 4.5%. Wins and losses are the same, so set inference is not what separates them. The few
real surprises: Weavile
Leftovers (2700080601), Cobalion Iron Defense (2700002803, T27), Ogerpon-Hearthflame ability
(name mismatch `embodyaspecthearthflame`), Hawlucha Stone Edge (2700100915, T20: Whiscash Liquidation vs a Hawlucha that
then crit-killed Gouging Fire). Tera type is the loosest part (61% mean presence; the prior over 2-3 types), and
the only decisive Tera hits were 2700064973 (Serperior Tera Rock T44) and 2700106274 (below). The 8 "ability" surprises with
presence 0.00 are mostly Trace copies, not sampler errors.

### Tera (confidence M that it is not a major class)

We Terastallize in 196 / 229 games (win rate 0.54; 0.79 in the 33 games where we did not, but those are short
wins). Mean value change over the Tera turn +0.02; Tera at HP < 50%: 33 times, win 0.48, fainted within a turn
24% (vs 15% at HP >= 80%). First Tera turn <= 2: 18 games, win 0.56. Seven losses never used Tera
(2700044050, eptlfr6ypw, 2700078834, 2700125650, 2700135377, 2700162158, 2700247075); in 5 of them Tera
could not obviously have been decisive, so I do not call it an error class. The opponent
Terastallized in 154 games and our win rate is the same with or without (0.57 vs 0.57).
* **2700106274** (2402), T14-T16: v = 0.98 / 0.97 / 0.97 against a last Feraligatr that has just Dragon-Danced,
  with our Rayquaza at 16% and Gholdengo. The opponent Terastallizes Water, Shadow Ball leaves it at 6%, Feraligatr
  KOs both; 0.97 -> 0.18 -> loss. Confidence M: an over-optimistic line against an unused opponent Tera in a
  one-Pokemon endgame (the worlds assign Water to a minority of worlds).
* **2700048217**: Ninetales Solar Beam leaves Alcremie at 16% (we expected a KO), see Bugs.

### Set-up sweepers (6 decisive; confidence L)

Opponent Reshiram Dragon Dance (2700080601, T20-T21, then Tera Fire Flare Blitz), Scizor Swords Dance +
Tera Steel (2700198843), Ogerpon Swords Dance + Tera Water (2700266732, v 0.82 -> 0.01 in two turns), Feraligatr
DD (2700106274). Frequency of an opponent mon at >= +2 on the flip turn is 14/97, equal to the base rate (13.3% of all
turns), so this is not over-represented.

### Overconfidence (confidence H that it is rare)

Games that reached v >= 0.9 at some turn: 108, of which 7 were lost (6.5%); v >= 0.8: 127, 16 lost. The value
is close to calibrated at the top and, if anything, pessimistic when ahead (3 v 2 alive: value 0.57, win 0.65; 4 v 3:
0.56 vs 0.67). Seven losses from >= 0.9 are 2700183068 and 2700198843 (crit), 2700105409 (para), 2700106274 (Tera),
ytw25owgpw, 2700221645, 2700247075.

### Time / forced moves

1 decision of 7738 took > 1.5 s; no decision searched fewer than the requested 16 worlds. Not a factor.

### Engine mechanics / state tracking (2 decisions in 2 losses; confidence H that it is a bug)

* Instant Solar Beam. In Sun (or with Power Herb) Showdown logs `|move|..|Solar Beam||[still]`, `|-prepare|` and then the
  damage in the same turn. poke-env sets `preparing_move` on `-prepare` and our state builder
  (`src/showdown_bot/engine_state.py` L735-745 `_CHARGING_VOLATILES`, applied L963-966) adds the `solarbeam`
  volatile, so the next decision is forced to Solar Beam. In **2700048217** T19 (Ninetales, 67/263 HP vs Alcremie 224/263) the
  record shows a single option `solarbeam` (1000 visits, `search_value` 0.94 vs `avg_score` 0.25) although Fire
  Blast, Nasty Plot and Scorching Sands were legal; Solar Beam left Alcremie at 16%, Alluring Voice killed us (v 0.88 -> 0.29). Same
  bug in **2700022584** T12 (Torkoal at 10/266 forced into a second Solar Beam). Only 2 decisions in 229 games, so
  low impact, but it is a one-line fix.
* Knock Off on a Choice-locked mon (**2700092169** T9): Heracross was locked into Earthquake (resisted by Meganium) for
  three turns; Meganium's Knock Off removed the Band, Heracross then Close Combat KO'd Meganium (v 0.73 -> 0.31).
  Knock Off removed a Choice item from the opponent in 12 games; this is the only decisive case. Worth a check that
  the engine releases the lock when the item is removed (confidence L that the engine is wrong; the search gave
  Knock Off 68% of visits, as a normal good move).
* No evidence in this run of damage-model errors big enough to matter on their own (the one miss-by-16% is
  most likely a low roll, and `docs/ko-check.md` measures 99.35% KO-class agreement).

## Ranked fixes

1. **Make futile moves un-pickable** (H confidence it is a bug class; modest absolute gain, cheap).
   Files: `src/showdown_bot/players/search.py` (`SearchConfig.prune_noop`, `noop_vs_switches`, root filtering ~L1144-1190),
   `src/showdown_bot/decision_quality.py` `noop_choices`. (a) Enable `prune_noop` in the ladder spec; (b) relax
   the strict "no-op against every opponent switch" rule to "no-op against all opponent moves of the current
   target and not better than the best alternative", or hard-code the cheap cases from the log tracker (status move on a mon
   already carrying that status or immune to it, Substitute with a Sub up, Rest at full HP, Protect twice, attack into a type
   immunity of a revealed type/ability). 4 losses are decided by this and 8.1% vs 2.7% of moves are wasted. Test with the
   `heal_loop` / `sub_loop` categories of `docs/misplay-suite.md` plus a duel; the misplay suite does not contain the
   Glare-on-paralysed case, add it.
2. **Fix the Solar Beam / charge-state bug** (H bug, 2 decisions). `engine_state.py` `_volatiles` ~L963: only add a
   charging volatile if the move cannot have completed this turn: for our own Pokemon trust the request (more than one
   non-disabled move => not locked), for the opponent clear `preparing_move` when a damage/boost event follows the
   `-prepare` within the same turn or the weather is Sun for Solar Beam / Solar Blade. Add a regression test with
   the 2700048217 log (turn 19) in `tests/`.
3. **Audit switch evaluation** (M-L). Over-switching and punished switch-ins (32/98 vs 11/131) are the largest measurable
   behavioural difference. Look at how the search scores `switch X` against the opponent's predicted move at depth 1-2
   (`players/search.py` aggregation; `nash_world`), how often the chosen switch has < 40% HP, and test a small switch
   tempo penalty / Tera-aware prior (`prior_opp`) in a duel. Needs an experiment, not a patch.
4. **Long-game / stall plan** (L). 37% of losses are >= 35 turns with win rate 0.30-0.39. Candidates: A/B `endgame` (currently `off`) when <= 5 are
   alive (`endgame.py`), and a progress term / PP-or-loop breaker so recoveries are not mirrored. The material-conversion
   numbers are fine (4 v 1 0.97, 3 v 1 0.85, 2 v 1 0.75), so the endgame solver may not help; the damage is done in the
   middlegame.
5. **Opponent Tera in one-Pokemon endgames** (L). 2700106274 and 2700198843: when the opponent still has an unused Tera
   the worlds should weight each plausible Tera type, and the last-Pokemon value should not exceed what the worst
   likely Tera allows; look at the sampler's tera distribution (`src/showdown_bot/sets.py`, `OpponentModel`;
   `players/search.py` `_sample_worlds` ~L1208). Evidence is two games.
6. **Low-priority / not worth changing:** world sampler (calibrated, wins and losses identical), time budget (no overruns),
   Tera policy (no measurable effect), value calibration (good).

## Reproducing the numbers

    python scripts/loss_analysis.py --run C:/Users/Lewis/pokemon-showdown-bot/data/ladder/w16 --summary

prints every table above (the Tera, luck, switching and base-rate sections included). Known false positives of
the surprise detector: Trace-copied abilities, Ogerpon `Embody Aspect (X)` naming, moves revealed by Metronome-like
effects. The point-of-no-return rule is crude: games whose value oscillates around 0.45 get an arbitrary flip turn.
