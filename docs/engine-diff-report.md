# Engine vs Showdown: first differential report (589.68.2)

Engine: poke-engine fork wheel `0.0.48+sdb.19` (gen 9, terastallization). Simulator: local Showdown checkout
(`vendor/pokemon-showdown`). 200 seeded `gen9randombattle` games (battle seeds 5000-5199), random legal actions,
6,992 turns. Method and caveats: [engine-diff.md](engine-diff.md). Reproduce a turn: `python scripts/engine_diff.py repro 5005:6`
(battle seed : turn). The auto-generated tables follow the summary.

## Summary

* **8.5% of turns (591 / 6,983) diverge** (no engine branch reproduces the simulator), PP excluded. With PP included
  84% of turns differ, because the engine does not decrement PP at 10 or more.
* The base damage model is good: on 1,112 clean single-target hits the median engine/sim damage ratio is 1.000 and
  only 1.2% are outside +-15% (the expected roll spread is +-8%). Almost all divergence is in **effects around the hit**
  (end-of-turn ordering, volatiles and their timers, items, forme changes, a few unimplemented moves / abilities),
  not in the damage formula. Tera turns are no worse than average (34 / 374 = 9.1%).
* No unknown species / ability / item / move ids in 6,992 turns (every id the sim produced exists in the engine);
  the only thing the engine cannot represent in the pre-state was Healing Wish slot conditions (49 turns) and
  Future Sight (6).
* Converter self-check (what goes into the engine comes back out): 0 failures.

## Verdicts on the known suspects

| suspect | verdict | evidence |
|---|---|---|
| Terapagos-Terastal takes ~4x damage | **Not reproduced. Tera Shell damage is right.** Engine/sim = 1.01-1.02 at full HP for neutral, super-effective and special hits (fixed by sdb.16). **But Terapagos formes are not modelled**: Tera Shift on switch-in (Terapagos -> Terastal, `engine active terapagos vs sim terapagosterastal`, 6 turns, e.g. 5190:3) and Terastallizing into Stellar (`terapagosterastal vs terapagosstellar`, 2 turns, e.g. 5127:9); Stellar Terapagos damage taken/dealt is off by 0.4-1.6x in 6 natural turns (5127:9, 5190:7, 5190:8, 5190:24). | `suspects` table below; natural-game rows above |
| Protosynthesis / Quark Drive up to 2x | **Not reproduced: the boost itself is right.** Booster Energy attacker: engine/sim 1.01, the same as its no-item control (1.01), with the sim's x1.3 present in both. Direct engine probes: Booster Energy on switch-in activates and consumes the item; switching into existing sun activates. **Gap: activation when the weather starts while the mon is already out** (Sunny Day / Drought partner: engine sets no boost, 5111:22 reproduced by probe). 4 more natural turns have a sim boost volatile the engine lacks with no obvious trigger (5013:5, 5013:26, 5078:25, 5068:4). Natural-game hp mismatches on these abilities are mostly other effects (Scream Tail + Rage Fist = missing `times_attacked`; Leftovers order). | `suspects`; 5 of 242 + 280 natural turns |
| Supreme Overlord | **Correct**: engine/sim 1.01-1.02 with 0, 2, 5 fainted allies (fainted bench counted from the team). | `suspects` |
| Flash Fire | **Correct**: probe damage after absorbing a Fire move is engine/sim 1.02 (control 1.00); a direct engine probe shows the absorb sets `FLASHFIRE` with no damage and the next Fire move does 1.45x. 4 of 59 natural Flash Fire turns diverged, none about the boost. | `suspects` |
| PP tracking | **Confirmed**: the engine decrements PP only below 10 (`generate_instructions.rs:2335`). 96.8% of slots under 10 PP match the simulator, 0% of slots at 10+ (0 of 8,643 decrements). No max PP, so no PP Ups / Pressure / Leppa cap. In engine self-play stall loops (Recover, Protect, Roost, Wish) never run out. | PP table below |

## Largest engine-side findings (ranked by share of turns; each reproducible with `repro <seed>:<turn>`)

| # | finding | turns (% of 6,983) | example | note |
|---|---|---|---|---|
| 1 | **Heal Block**: Psychic Noise's Heal Block never expires (no timer) and does not stop the same-turn Leftovers heal | 67 (1.0%) | 5031:8, 5036:10 | engine keeps `HEALBLOCK` forever; also mismatches Recover / Roost under block |
| 2 | **End-of-turn order**: Leftovers (and Black Sludge) heal *after* poison / burn damage in the engine; Showdown heals first. At full HP burn / poison then "does nothing" in the engine; near death the engine lets Leftovers save a mon that Showdown's poison tick kills | 38 (0.5%) | 5005:6 (psn), 5034:7 (brn), 5056:26 (faint) | sim order: Leftovers, then psn / brn |
| 3 | **Harvest** not implemented (berry never regrows, Arboliva / Tropius / Trevenant); Sitrus-berry turns also affected | 26 (0.4%) | 5010:18, 5032:24 | berry id of the consumed item is not stored |
| 4 | **Forme changes**: Terapagos (Tera Shift, Stellar), Mimikyu-Busted, Eiscue-Noice, Morpeko-Hangry, Ogerpon tera, Ursaluna-Bloodmoon... the engine active keeps the base forme | 19 (0.3%) | 5007:1, 5190:3, 5127:9 | stats / types / ability of the new forme missing |
| 5 | **Protean / Libero**: engine retypes on every move; gen 9 does it once per switch-in (types mismatch for Greninja, Meowscarada, Cinderace) | 18 (0.3%) | 5014:17 | |
| 6 | **Take Heart** has no effect (+1 SpA/SpD and status cure missing) | 14 (0.2%) | 5121:25 | move data |
| 7 | **Sheer Force + Life Orb**: engine still applies Life Orb recoil to boosted (no-secondary) moves | 13 (0.2%) | 5057:8 | |
| 8 | **Outrage / Thrash lock length**: Showdown locks for a random 2-3 turns, the engine has a fixed counter, so the confusion at the end arrives on a different turn | 13 (0.2%) | 5015:21 | counter semantics are partly guessed (`guess` column) |
| 9 | Throat Chop (applied through Protect / Substitute, no 2-turn timer), Slow Start (kept too long), Cursed Body (`disable` missing), Unburden (`unburden` volatile missing), Imposter (extra `transformed`), Truant | 5-12 each | 1009:20, 5135:5, 5036:6, 5011:12 | |
| 10 | **Charge / Electromorphosis**: ability not implemented, Charge not consumed by the next Electric move | 7 | 3048:44 | |
| - | Giga Impact that KOs sets no `mustrecharge` (Slaking) | 3 | 1000:1 | |
| - | Disguise bust triggers on an *immune* hit (Dragon move into Mimikyu) | 2 | 2002:14 | |
| - | Non-Ghost Curse leaves a `CURSE` volatile (harmless: no damage in 3 turns) | 25 | 5000:3 | cosmetic |
| - | A fresh Struggle choice does not exist (8 turns skipped) | 8 | | |

Not yet checked in depth (appear in the generated tables, small counts): Psychic Noise / Recover damage rows (21 + 17 turns;
mostly Heal Block / Leftovers ordering above), Knock Off item loss rows, `Substitute` HP rows, Wish rows,
`engine_error` x1 (a follow-up switch the engine considered illegal, 5107:25).

Converter caveats: older wheels have no `Pokemon.times_attacked`, so Rage Fist / Last Respects turns mismatch there
(2 turns here); weather / screen turn counts, sleep counters and volatile durations are mapped from the simulator
by inference (see `guess` in the table).

## Generated tables (`engine_diff.py report`)

- games: 200 (seed0 5000), turns analysed: 6992, wall time 25s
- turn status: pp_only=5375, match=1017, diverge=591, skipped=8, engine_error=1
- **overall divergence (all comparable turns): 591/6983 = 8.5%** (PP-only differences are not counted here, see the PP section)
- clean turns (no unrepresentable features, no unknown ids): 6928; divergence 581/6928 = 8.4%
- lossy turns (engine cannot represent something in the pre-state or lacks an id): 55 (0.8%); divergence there 10/55 = 18.2%
- engine branches per turn: median 2, p95 9, max 55
- converter self-check failures: none

## Divergence categories (best-matching branch's mismatches; one count per turn per category)

| # | field | mechanic (sim cause) | turns | % of turns | converter risk | example (seed:turn) |
|---|---|---|---|---|---|---|
| 1 | hp | psychicnoise | 21 | 0.3% |  | 5031:7, 5031:8, 5032:24 |
| 2 | item | abilityharvest | 18 | 0.3% |  | 5010:18, 5027:3, 5032:13 |
| 3 | hp | recover | 17 | 0.2% |  | 5025:3, 5028:33, 5028:35 |
| 4 | boost.spa | takeheart | 14 | 0.2% |  | 5121:25, 5121:26, 5121:27 |
| 5 | boost.spd | takeheart | 14 | 0.2% |  | 5121:25, 5121:26, 5121:27 |
| 6 | hp | psn | 12 | 0.2% |  | 5005:6, 5010:20, 5028:9 |
| 7 | hp | engine-only[item=lifeorb,ability=sheerforce] | 12 | 0.2% |  | 5057:8, 5060:32, 5087:2 |
| 8 | active | (no sim event) | 10 | 0.1% |  | 5007:1, 5045:31, 5057:12 |
| 9 | hp | brn | 10 | 0.1% |  | 5034:7, 5046:6, 5051:1 |
| 10 | types | engine-only[item=lifeorb,ability=protean] | 9 | 0.1% |  | 5014:17, 5014:18, 5014:19 |
| 11 | hp | bodyslam | 9 | 0.1% |  | 5043:18, 5067:13, 5108:2 |
| 12 | hp | itemsitrusberry | 9 | 0.1% |  | 5067:60, 5067:65, 5074:35 |
| 13 | volatile.extra_in_engine:slowstart | residual | 9 | 0.1% |  | 5135:5, 5135:38, 5135:47 |
| 14 | faint | itemleftovers | 8 | 0.1% |  | 5010:10, 5056:26, 5060:33 |
| 15 | hp | roost | 8 | 0.1% |  | 5022:42, 5039:7, 5060:23 |
| 16 | hp | knockoff | 8 | 0.1% |  | 5028:6, 5131:28, 5135:41 |
| 17 | volatile.missing_in_engine:disable | abilitycursedbody | 8 | 0.1% |  | 5036:6, 5043:37, 5069:13 |
| 18 | volatile.missing_in_engine:unburden | engine-only[item=whiteherb,ability=unburden] | 7 | 0.1% |  | 5011:12, 5024:14, 5100:1 |
| 19 | active | residual | 7 | 0.1% |  | 5011:14, 5022:16, 5058:16 |
| 20 | volatile.extra_in_engine:healblock | psychicnoise | 7 | 0.1% |  | 5031:8, 5036:10, 5108:7 |
| 21 | item | engine-only[item=sitrusberry,ability=harvest] | 7 | 0.1% |  | 5032:24, 5032:28, 5032:29 |
| 22 | hp | itemleftovers | 7 | 0.1% |  | 5047:6, 5047:7, 5085:33 |
| 23 | volatile.extra_in_engine:curse | engine-only[item=none,ability=unaware] | 6 | 0.1% |  | 5000:3, 5000:5, 5000:10 |
| 24 | volatile.extra_in_engine:confusion | residual | 6 | 0.1% | guess | 5011:26, 5015:17, 5015:28 |
| 25 | types | abilityprotean | 5 | 0.1% |  | 5014:16, 5023:1, 5050:6 |
| 26 | hp | darkpulse | 5 | 0.1% |  | 5014:18, 5043:22, 5064:7 |
| 27 | hp | engine-only[item=sitrusberry,ability=harvest] | 5 | 0.1% |  | 5032:28, 5032:29, 5032:30 |
| 28 | boost.spe | engine-only[item=heavydutyboots,ability=speedboost] | 5 | 0.1% |  | 5034:27, 5034:36, 5034:41 |
| 29 | volatile.extra_in_engine:curse | engine-only[item=leftovers,ability=unaware] | 5 | 0.1% |  | 5038:4, 5043:20, 5043:23 |
| 30 | hp | itemlifeorb | 5 | 0.1% |  | 5045:25, 5045:31, 5057:12 |

## By mechanic (cause attributed from the simulator log, merged across fields)

| mechanic | turns | % of turns |
|---|---|---|
| residual | 34 | 0.5% |
| psychicnoise | 25 | 0.4% |
| ability:harvest | 18 | 0.3% |
| recover | 18 | 0.3% |
| item:leftovers | 15 | 0.2% |
| takeheart | 14 | 0.2% |
| knockoff | 13 | 0.2% |
| engine-only[item=lifeorb,ability=sheerforce] | 13 | 0.2% |
| psn | 12 | 0.2% |
| substitute | 11 | 0.2% |
| brn | 11 | 0.2% |
| (no sim event) | 10 | 0.1% |
| engine-only[item=lifeorb,ability=protean] | 9 | 0.1% |
| roost | 9 | 0.1% |
| bodyslam | 9 | 0.1% |
| item:sitrusberry | 9 | 0.1% |
| ability:cursedbody | 8 | 0.1% |
| engine-only[item=whiteherb,ability=unburden] | 7 | 0.1% |
| uturn | 7 | 0.1% |
| engine-only[item=sitrusberry,ability=harvest] | 7 | 0.1% |
| engine-only[item=none,ability=unaware] | 6 | 0.1% |
| bodypress | 6 | 0.1% |
| darkpulse | 6 | 0.1% |
| outrage | 6 | 0.1% |
| icebeam | 6 | 0.1% |
| item:lifeorb | 6 | 0.1% |
| earthquake | 6 | 0.1% |
| engine-only[item=choicescarf,ability=imposter] | 6 | 0.1% |
| hypervoice | 5 | 0.1% |
| ability:protean | 5 | 0.1% |

## By field

| field | turns | % of turns |
|---|---|---|
| hp | 281 | 4.0% |
| volatile.extra_in_engine | 125 | 1.8% |
| boost | 66 | 0.9% |
| item | 57 | 0.8% |
| faint | 46 | 0.7% |
| volatile.missing_in_engine | 34 | 0.5% |
| types | 22 | 0.3% |
| active | 19 | 0.3% |
| status | 11 | 0.2% |
| ability | 1 | 0.0% |
| terrain | 1 | 0.0% |

volatile mismatches by name (mismatch count): extra_in_engine:healblock 49, extra_in_engine:curse 25, extra_in_engine:confusion 13, extra_in_engine:throatchop 12, extra_in_engine:slowstart 9, missing_in_engine:unburden 8, missing_in_engine:disable 8, extra_in_engine:transformed 7, missing_in_engine:truant 4, missing_in_engine:charge 3, missing_in_engine:taunt 2, missing_in_engine:substitute 2, missing_in_engine:protosynthesisdef 2, missing_in_engine:yawn 2, extra_in_engine:disable 2, extra_in_engine:substitute 2, missing_in_engine:encore 2, extra_in_engine:charge 2, missing_in_engine:mustrecharge 2, extra_in_engine:leechseed 1

## Mechanics most associated with divergence (tag involved in the turn; base rate 8.5%)

| tag | turns involved | diverged | rate | excess diverged turns |
|---|---|---|---|---|
| ability:harvest | 350 | 86 | 24.6% | 56.4 |
| item:sitrusberry | 343 | 78 | 22.7% | 49.0 |
| move:curse | 34 | 27 | 79.4% | 24.1 |
| move:psychicnoise | 108 | 31 | 28.7% | 21.9 |
| item:leftovers | 3749 | 338 | 9.0% | 21.1 |
| species:trevenant | 117 | 30 | 25.6% | 20.1 |
| ability:sheerforce | 84 | 24 | 28.6% | 16.9 |
| species:tropius | 118 | 26 | 22.0% | 16.0 |
| ability:protean | 56 | 20 | 35.7% | 15.3 |
| move:substitute | 113 | 24 | 21.2% | 14.4 |
| species:mesprit | 46 | 17 | 37.0% | 13.1 |
| move:takeheart | 21 | 14 | 66.7% | 12.2 |
| species:dondozo | 48 | 16 | 33.3% | 11.9 |
| ability:hydration | 147 | 24 | 16.3% | 11.6 |
| ability:shieldsdown | 30 | 14 | 46.7% | 11.5 |
| species:phione | 74 | 17 | 23.0% | 10.7 |
| species:miniormeteor | 16 | 12 | 75.0% | 10.6 |
| ability:unaware | 196 | 27 | 13.8% | 10.4 |
| species:greninja | 24 | 12 | 50.0% | 10.0 |
| ability:intimidate | 358 | 39 | 10.9% | 8.7 |
| ability:slowstart | 27 | 11 | 40.7% | 8.7 |
| species:regigigas | 27 | 11 | 40.7% | 8.7 |
| species:mimikyu | 17 | 10 | 58.8% | 8.6 |
| species:farigiraf | 53 | 13 | 24.5% | 8.5 |
| event:crit | 330 | 36 | 10.9% | 8.1 |
| move:recover | 237 | 28 | 11.8% | 8.0 |
| move:bodyslam | 108 | 17 | 15.7% | 7.9 |
| species:wyrdeer | 61 | 13 | 21.3% | 7.8 |
| species:tauros | 26 | 10 | 38.5% | 7.8 |
| species:hatterene | 54 | 12 | 22.2% | 7.4 |

## Engine cannot represent / does not know (converter limits that are ENGINE gaps)

unrepresentable features (turns): slotcond:healingwish 49, slotcond:futuremove 6

unknown ids (turns): none

counter mappings that are GUESSES (present in pre-state, turns): toxic 553, weather_turns 377, rest 314, terrain_turns 206, sleep 162, substitute 148, wish 103, encore 82, confusion 69, taunt 62, lockedmove 62, screen_turns 38, yawn 12, trickroom_turns 11, tailwind_turns 3

## Engine panics / errors / converter crashes

- 1x followup ValueError: Invalid move for s1: leafeon [choices ['leafeon', 'none']]

## Damage model check (clean single-target damage turns, no crit)

- samples: 1112; median engine/sim ratio 1.000; outside [0.85, 1.15]: 13 (1.2%); outside [0.7, 1.4]: 7; ratio >= 1.8 or <= 0.55: 3
- the engine uses 0.925 x max roll (mean of the simulator's 0.85..1.00 rolls), so a correct model gives ratio ~1.0 with spread about +-8%

Contexts (n >= 8) with the most out-of-band samples (ranked by outliers beyond the 4% base rate):

| context | n | median engine/sim | outside [0.85,1.15] |
|---|---|---|---|
| move=psychicnoise | 13 | 1.00 | 4 (30.8%) |
| atk_ab=sapsipper | 26 | 1.00 | 3 (11.5%) |
| def_ab=unaware | 10 | 1.00 | 2 (20.0%) |
| atk=volcarona | 8 | 1.00 | 1 (12.5%) |
| def_ab=contrary | 9 | 0.98 | 1 (11.1%) |
| move=rapidspin | 12 | 1.01 | 1 (8.3%) |
| atk_item=leftovers | 141 | 1.00 | 6 (4.3%) |
| def_ab=multitype | 16 | 1.01 | 1 (6.2%) |
| move=bodyslam | 17 | 1.00 | 1 (5.9%) |
| def_item=choicescarf | 19 | 1.00 | 1 (5.3%) |
| atk_ab=flamebody | 20 | 1.00 | 1 (5.0%) |
| atk_ab=levitate | 70 | 1.00 | 3 (4.3%) |
| def_ab=quarkdrive | 23 | 1.00 | 1 (4.3%) |
| atk_ab=intimidate | 24 | 1.00 | 1 (4.2%) |
| def_ab=intimidate | 30 | 1.00 | 1 (3.3%) |
| def_ab=regenerator | 32 | 1.00 | 1 (3.1%) |
| move=heavyslam | 8 | 1.00 | 0 (0.0%) |
| move=gigadrain | 8 | 0.98 | 0 (0.0%) |
| def_item=whiteherb | 8 | 1.00 | 0 (0.0%) |
| def_item=icicleplate | 8 | 1.04 | 0 (0.0%) |

## Known suspects

- **Terapagos-Terastal/Stellar on field**: involved turns 57, diverged 8 (14.0%); hp/faint mismatches 9, median engine/sim damage ratio 0.80 (n=6)
- **ability protosynthesis on an active**: involved turns 242, diverged 23 (9.5%); hp/faint mismatches 11, median engine/sim damage ratio 0.47 (n=7)
- **ability quarkdrive on an active**: involved turns 280, diverged 11 (3.9%); hp/faint mismatches 7, median engine/sim damage ratio 1.14 (n=4)
- **ability supremeoverlord on an active**: involved turns 24, diverged 0 (0.0%); hp/faint mismatches 0
- **ability flashfire on an active**: involved turns 59, diverged 4 (6.8%); hp/faint mismatches 2, median engine/sim damage ratio 1.04 (n=2)
- **PP tracking**: PP differs from the simulator in 5882 of 6983 turns (84.2%). Per pre-turn PP bucket, for slots whose PP changed in either engine or simulator (engine decrement vs simulator decrement):

| pre-turn PP | slots | sim decremented | engine decremented | engine matches sim |
|---|---|---|---|---|
| 1-9 | 2603 | 2526 | 2599 | 2521 (96.8%) |
| 10-15 | 2061 | 2061 | 0 | 0 (0.0%) |
| 16-31 | 5037 | 5037 | 0 | 0 (0.0%) |
| 32+ | 1545 | 1545 | 0 | 0 (0.0%) |

## Targeted suspect probes (30 seeds each, fixed teams, last scripted turn measured)

Engine damage = probability-weighted over all engine branches (includes crit / miss branches); simulator = mean over seeds.

| scenario | n | sim mean dmg | sim min-max | engine expected dmg | engine/sim |
|---|---|---|---|---|---|
| tera_shell: Karate Chop vs Terapagos-Terastal (full HP) | 30 | 35.5 | 31-51 | 36.1 | 1.02 |
| control: Karate Chop vs Snorlax | 30 | 208.9 | 186-302 | 214.6 | 1.03 |
| tera_shell: Earthquake vs Terapagos-Terastal (full HP) | 30 | 67.6 | 62-73 | 68.4 | 1.01 |
| control: Earthquake vs Snorlax | 30 | 199.1 | 183-216 | 203.2 | 1.02 |
| tera_shell: Hyper Voice vs Terapagos-Terastal (full HP) | 30 | 35.2 | 32-38 | 35.7 | 1.02 |
| control: Hyper Voice vs Snorlax | 30 | 70.8 | 65-77 | 72.5 | 1.02 |
| protosynthesis: Roaring Moon Booster Energy Earthquake vs Snorlax | 30 | 183.6 | 167-274 | 185.8 | 1.01 |
| protosynthesis control: Roaring Moon (no item) Earthquake vs Snorlax | 30 | 141.2 | 129-212 | 142.9 | 1.01 |
| quark_drive: Iron Valiant Booster Energy Knock Off vs Snorlax | 30 | 113.4 | 103-170 | 114.4 | 1.01 |
| quark_drive control: Iron Valiant (no item) Knock Off vs Snorlax | 30 | 87.2 | 79-131 | 87.8 | 1.01 |
| protosynthesis (sun, defender Iron Hands-like Scream Tail): Earthquake vs Scream Tail | 30 | 147.3 | 135-220 | 150.0 | 1.02 |
| supreme_overlord: Kingambit with 0 fainted allies, Kowtow Cleave vs Snorlax | 30 | 175.3 | 160-262 | 177.6 | 1.01 |
| supreme_overlord: Kingambit with 2 fainted allies, Kowtow Cleave vs Snorlax | 30 | 210.1 | 192-315 | 213.3 | 1.02 |
| supreme_overlord: Kingambit with 5 fainted allies, Kowtow Cleave vs Snorlax | 30 | 262.2 | 238-393 | 266.5 | 1.02 |
| flash_fire: Heatran after absorbing Flamethrower, Flamethrower vs Blissey | 30 | 159.9 | 141-231 | 162.5 | 1.02 |
| flash_fire control: Heatran (not activated) Flamethrower vs Blissey | 30 | 109.7 | 94-156 | 109.4 | 1.00 |
