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

---

# After sdb.22 (589.68.4): same 200 seeds, engine `0.0.48+sdb.22`

`python scripts/engine_diff.py run --games 200 --seed 5000` with the sdb.22 wheel (6,992 turns, 6,983 comparable, one
process, 23 s). The harness also changed slightly: it now passes the simulator's `maxpp` (`Move.max_pp`), the
Heal Block / Throat Chop durations, and marks Protean / Libero as used from the simulator's ability state. To separate
engine from harness effects the sdb.21 wheel was re-run with the *same* harness.

| | sdb.19 report (old harness) | sdb.21 wheel, new harness | **sdb.22** |
|---|---|---|---|
| diverging turns, PP excluded | 591 / 6,983 = 8.5% | 577 = 8.3% | **383 = 5.5%** |
| clean turns only | 581 / 6,928 = 8.4% | 567 = 8.2% | 376 / 6,928 = 5.4% |
| turns differing in anything, PP included | 5,882 = 84.2% (PP only) | 5,962 = 85.4% | **403 = 5.8%** (383 diverge + 20 PP-only) |
| `healblock` / `throatchop` volatile mismatches | 49 / 12 | | 0 / 0 |

PP now agrees with the simulator on 99.5-99.8% of changed slots in every PP bucket (was 0% at 10 and above); PP
differs in 25 of 6,983 turns. What is left: a move used against a target that fainted earlier in the turn (the
simulator still spends the PP, the engine skips the move before the deduction; 5000:34) and a few Pressure /
Sleep Talk interactions.

## What changed in the engine

* **Residual order** (`add_end_of_turn_instructions`) now follows Showdown's `onResidualOrder`: weather damage and
  weather abilities (Solar Power, Dry Skin, Rain Dish, Ice Body; order 1), Future Sight (3), Wish (4), Grassy Terrain heal
  (5/2, and not on the terrain's last turn), Hydration / Shed Skin (5/3), **Leftovers / Black Sludge (5/4)**, Leech Seed (8),
  poison / toxic / Poison Heal (9), burn (10), then Salt Cure and bind (13), Yawn, Perish Song, Roost, and finally the
  orbs / Speed Boost / form checks. Fixed on the way: Black Sludge damage and Solar Power damage were clamped to the
  *missing* HP instead of the current HP; Hydration / Shed Skin cured after the poison tick.
* **Heal Block** (5 turns, Psychic Noise 2, Persistent 7) is a real timer; it blocks Leftovers, Black Sludge, Grassy
  Terrain, Wish, Leech Seed's heal, Poison Heal, Ice Body / Rain Dish / Dry Skin and heal moves (which are no longer
  offered as choices). **Throat Chop** lasts 2 turns, blocks sound moves, and (like Psychic Noise's Heal Block) is now a
  100% *secondary*, so Protect / Substitute / Covert Cloak / Sheer Force stop it.
* **PP** is tracked at any value for moves with `max_pp > 0`, clamped at 0 (Pressure), not spent when the user is
  asleep / fully paralysed / flinched or on the continuation turn of a charge / locked move, still spent by a failed
  Protect and by a Sleep Talk used while asleep; Leppa Berry restores at most up to `max_pp`. Moves with unknown max PP
  (`max_pp == 0`, hand-built states and old records) keep the old "track below 10" behaviour.
  MCTS cost: see `scripts/engine_bench_pp.py`, about 6% fewer iterations/s with full tracking (0.92-0.99x over three
  states and four runs, single thread, noisy because the ladder was running). The live bot therefore does not turn
  it on (`engine_state.TRACK_PP`, env `SHOWDOWN_ENGINE_TRACK_PP=1`); self-play and this harness do.
* Take Heart (+1 SpA/SpD and status cure); Sheer Force removes Life Orb recoil; a KO with Giga Impact / Hyper Beam keeps
  its recharge (the follow-up `(switch, none)` call used to consume it); Protean / Libero retype Water/Dark Greninja to
  pure Water on a Water move (whole type list compared, once per switch-in); Protosynthesis / Quark Drive activate when
  sun / Electric Terrain starts with the mon already out; Disguise survives an immune hit; Electromorphosis / Wind Power
  give Charge and Charge is used up by the next Electric move; Seismic Toss, Night Shade, Super Fang, Endeavor, Final
  Gambit no longer go through Protect.

## New top-10 (turns, % of 6,983)

| # | field / mechanic | turns | note |
|---|---|---|---|
| 1 | item, Harvest | 18 (0.3%) | Harvest not implemented (needs the consumed berry; plan below) |
| 2 | active, no sim event | 10 | Mimikyu-Busted (5045:31), Terapagos shift, forme changes (plan below) |
| 3 | hp, Sitrus Berry | 9 | mostly Harvest-regrown Sitrus (5067:60) |
| 4 | volatile `slowstart` still on in the engine | 9 | Slow Start timer (Regigigas) |
| 5 | hp, Recover | 8 | harness artifact: heal + damage in one turn is compared as a net change with a tolerance sized to the net |
| 6 | hp, Knock Off | 8 | Knock Off + item interactions (Harvest, Unburden, Sitrus) |
| 7 | volatile `disable` missing | 8 | Cursed Body not implemented |
| 8 | volatile `unburden` missing | 7 | Unburden not implemented |
| 9 | active, residual | 7 | forme changes at end of turn (Shields Down, Hunger Switch) |
| 10 | hp, Roost | 7 | same harness artifact as Recover |

Then: `curse` volatile left on non-Ghost Curse (cosmetic, 25), confusion length of Outrage / Thrash (random 2-3 turns
in Showdown), Imposter's `transformed` (5), Speed Boost (9, engine ticks on a turn the sim does not), Synchronize
(5028:9), Life Orb after a Disguise-absorbed hit (5045:31).

## Not done (plans)

* **Harvest** (about 0.6% of turns incl. Sitrus follow-ups). Needs (1) `Pokemon.last_item: Items` (serialized as an
  optional trailing field, py binding `last_item`), (2) set when a berry is *eaten* (all `ChangeItem berry -> none` sites
  in `items.rs`, but not Knock Off / Thief / Trick), cleared when it regrows, via a new reversible
  `Instruction::SetLastItem{side, new, previous}` (fits the 8-byte `Instruction`), (3) a last-in-turn Harvest step with
  probability 1 in sun else 0.5: `add_end_of_turn_instructions` would return the "did not regrow" branch (a clone of the
  instruction list before the Harvest step with probability x (1-p)) for the 4 call sites in
  `generate_instructions_from_move_pair` to push, then re-run `item_end_of_turn` for the regrown Sitrus.
* **Forme changes** (about 0.3%): Terapagos -> Terastal on switch-in (and Stellar on Tera: HP/stat/type/ability
  recalculation via `recalculate_stats`, the pattern `Abilities::SHIELDSDOWN`/`SCHOOLING` already use), Mimikyu-Busted
  on Disguise bust (already switches id but the harness shows the *active* mismatch; check `FormeChange` in the
  best-matching-branch path), Eiscue -> Noice (exists for physical hits; Hail restores Ice Face, missing), Morpeko-Hangry
  (`Abilities::HUNGERSWITCH` exists; the stats / move Aura Wheel type are not recalculated), Ursaluna-Bloodmoon.
* Slow Start (9 turns), Cursed Body (8), Unburden (7), Synchronize, a real Struggle choice when all PP are spent
  (currently the side gets `none`), Outrage / Thrash random length.

---

# After sdb.28 (72d, 589.68.4, 589.61): same 200 seeds, engine `0.0.48+sdb.28`

`python scripts/engine_diff.py run --games 200 --seed 5000` (6,992 turns, 6,983 comparable, 22 s). The harness changed
again, so sdb.27 was re-run with the *same* harness:

* **Stable identity** (`uid` = the set species instead of `baseSpecies`): a permanent forme change (Disguise ->
  Mimikyu-Busted, Tera Shift -> Terapagos-Terastal, Tera -> Terapagos-Stellar) rewrites `baseSpecies`, so the old harness
  reported the same mon as a different active Pokemon (the 10 + 7 `active` rows of the sdb.22 top-10) and silently
  skipped its HP / item / status comparison.
* **Slow Start** is an ability counter in gen 9 (`abilityState.counter`), not a volatile; the harness now maps it to the
  engine's `slowstart` volatile both ways (the 9 `slowstart` rows were a harness artifact; the engine timer was right).
* With a sdb.28 wheel the harness also passes `Pokemon.last_item` (Harvest holders) and the Disable duration/slot.

| | sdb.27, sdb.22-era harness | sdb.27, new harness | **sdb.28, new harness** |
|---|---|---|---|
| diverging turns, PP excluded | 383 / 6,983 = 5.5% | 362 = 5.2% | **284 = 4.1%** |
| clean turns only | 376 / 6,928 = 5.4% | 355 = 5.1% | 279 / 6,928 = 4.0% |
| PP-only turns | 20 | 20 | **8** |

## What changed in the engine

* **Choice lock release (72d)**: Knock Off, Thief, Covet, Trick / Switcheroo, Corrosive Gas and Magician re-enable the
  moves a Choice lock disabled once the holder no longer holds a Choice item (`items::on_item_taken`; Choice-for-Choice
  swaps and Gorilla Tactics keep the lock). Covet now steals like Thief; Corrosive Gas removes the target's item. A mon
  that receives a Choice item is locked after its next move (unchanged `item_before_move`).
* **Harvest**: `Pokemon.last_item` (optional trailing serialized field, py `Pokemon(last_item=)`), set by the new
  reversible `SetLastItem` instruction when a Harvest holder *eats* a berry (not Knock Off / Thief / Trick). Last
  residual: regrow with probability 1 in sun, else 0.5; a regrown berry whose condition holds is eaten at once (Sitrus
  at <= 1/2). `add_end_of_turn_instructions` returns the "did not regrow" branches (up to 3 with two Harvest sides).
  The live bot sets `last_item` for an opponent Harvest holder whose berry was eaten (`engine_state._opponent_pokemon`).
* **Unnerve / As One** stop the foe's berries (end of turn, before-move pinch / resist berries, Lum / Chesto on status).
  This, not Harvest, was most of the remaining Sitrus/Harvest rows (Pyroar vs Trevenant).
* **Formes**: `abilities::permanent_forme_change` (Showdown `formeChange(..., isPermanent)`): each stat moves by the
  base-stat difference, max HP via the new reversible `ChangeMaxHp` keeping the damage taken (`updateMaxHp`), ability and
  base ability replaced. Used for **Tera Shift** on switch-in (Terapagos -> Terastal / Tera Shell, cannot be
  suppressed) and **Terastal -> Stellar on Tera** (Teraform Zero, which also clears weather and terrain). Base stats
  for the three Terapagos formes added. Mimikyu-Busted, Eiscue-Noice (incl. snow restore), Morpeko Hunger Switch,
  Minior Shields Down and Ogerpon Tera formes were already handled; their `active` rows were the harness uid artifact.
* **Unburden**: the `unburden` volatile is applied when the holder loses its item (any site: detected from the branch's
  `ChangeItem x -> none` after each move and at end of turn); Speed x2 only while itemless.
* **Cursed Body**: 30% Disable (secondary on the attacker) of the move that hit it; `VolatileStatusDurations.disable`
  = `turns * 4 + slot` (optional trailing field, py `disable`), the move is disabled and re-enabled after 4 residuals.
* **PP on a fainted target**: a move aimed at a target that fainted earlier in the turn spends its PP (Showdown
  `[notarget]`) unless the user was asleep / frozen / flinched.

Validation: `cargo test --release` 1,114 passed (1 ignored), 19 new `test_sdb28_*` (each also checks that every
branch reverses to the exact starting state); `scripts/engine_fuzz.py --states 1000`: 1000 ok, 0 panics; pytest
1,015 passed + `tests/test_engine_sdb28.py` (6). MCTS throughput (`scripts/engine_bench_pp.py --seconds 2 --reps 3`,
4 alternating rounds per wheel, single thread): sdb.28 is 0-6% slower than sdb.27 (mean over rounds: legacy PP
opening 301k vs 301k it/s, midgame 297k vs 304k, late 3v2 346k vs 363k; tracked PP 272k vs 289k, 275k vs 287k,
321k vs 337k); round-to-round noise on this box is about +-4%.

## New top-10 (turns, % of 6,983)

| # | field / mechanic | turns | note |
|---|---|---|---|
| 1 | `curse` volatile left on a non-Ghost Curse user | 23 (5 rows) | cosmetic: the engine keeps the volatile, no effect |
| 2 | hp, Recover | 8 | harness artifact (heal + damage compared as a net change) |
| 3 | hp, Roost | 7 | same harness artifact |
| 4 | Speed Boost | 9 | engine boosts on the switch-in turn (Showdown needs `activeTurns`) |
| 5 | confusion after Outrage / Thrash | 10 | lock length is random 2-3 in Showdown, engine always 3 |
| 6 | hp, Knock Off | 5 | net-HP tolerance (drain in the same turn) and Ice Face / Disguise + Knock Off (item still removed in Showdown) |
| 7 | hp, Life Orb | 5 | no recoil after a Disguise-absorbed hit (Showdown applies it) |
| 8 | `transformed` (Imposter) | 5 | Imposter / Choice Scarf interactions |
| 9 | hp, Leftovers | 5 | mixed |
| 10 | hp, Acrobatics | 4 | item-loss timing within the turn |

Remaining plans: Speed Boost on the switch-in turn, random Outrage length, Life Orb / Knock Off through Disguise and
Ice Face, Synchronize and a real Struggle choice.
