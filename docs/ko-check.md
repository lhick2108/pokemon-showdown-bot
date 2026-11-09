# KO-call accuracy of the engine's roll model (589.58)

`scripts/ko_check.py` (+ `scripts/ko_calc.mjs`, uses `@smogon/calc` installed under the gitignored
`scratch/tools`) compares, for sampled (attacker, defender, move) situations from recorded games, the
number of KO rolls out of 16 the search uses (`generate_instructions.rs`: `max*0.85 + i*max*0.01`, strict
`> hp`, from `calculate_damage`'s max roll) with the exact 16 Showdown rolls vs the defender's current HP.

    cd scratch/tools && npm install @smogon/calc
    python scripts/ko_check.py validate
    python scripts/ko_check.py run --n 3000 --n-near 3000 --out scratch/ko.jsonl
    python scripts/ko_check.py analyze scratch/ko.jsonl

Result (seed 1, 4,001 usable situations: 2,510 uniform + 1,491 "near-KO" where engine max is 0.6-1.6x HP):

- Engine max roll equals calc's max exactly in 82% of cases, within 5% in 99.7%; the rest are in-battle
  ability states calc cannot see (Protosynthesis/Quark Drive/Slow Start/...; excluded) .
- KO class (never / partial / guaranteed) differs in 0.65% of all cases (0.3% uniform, 1.2% near-KO).
- Among KO-relevant cases (~10% of cases) mean |KO-probability error| is 2.3 percentage points
  (1.8 pp from the roll model alone, i.e. applying the engine's rule to calc's own max); the engine
  slightly OVER-estimates KOs (+0.6 pp signed; 91 over vs 51 under).
- No dominant source; errors are spread across Life Orb / Leftovers matchups and common moves.
- Non-crit only; multi-hit and fixed-damage moves excluded. Cross-check: the replicated rule matches the
  real `generate_instructions` KO branch in 165/182 partial-KO cases; the other 17 are burn-secondary
  moves (end-of-turn burn damage adds KOs).

## In-battle ability state (589.61)

The harness now passes the ability state that the engine state string exposes to `@smogon/calc`
(`boostedStat` from `PROTOSYNTHESIS*/QUARKDRIVE*` volatiles, `abilityOn` from `FLASHFIRE` / `SLOWSTART` /
`CHARGE` (Electromorphosis), `alliesFainted` from the fainted bench count), so only Stakeout (needs "just
switched in") stays excluded. `Download` and `Analytic` are neutralised in calc (it re-applies the Download boost
and assumes Analytic moves last), and Meteor Beam / Electro Shot / always-crit moves / Payback-style moves are
excluded (`ASSUMPTION_MOVES`: calc adds the charge-turn boost, forces the crit or assumes move order).

    python scripts/ko_check.py run --n 20000 --n-near 0 --focus --per-game 12 --max-games 6000 --out f.jsonl
    python scripts/ko_check.py run ... --no-ability-state --out f0.jsonl      # old behaviour, for comparison
    python scripts/ko_check.py abilities f.jsonl --worst 3                     # per-ability engine/calc max table

Share of focus-sample cases (any stateful ability or Terapagos involved; 19,587 cases) where engine max differs
from calc max by >5%: 6.63% without ability state -> 0.56% with state fed to calc -> 0.19% after the engine fixes
below. Engine bugs found and fixed (engine sdb.16):

- **Tera Shell was not implemented** (Terapagos-Terastal defender at full HP: every neutral / super-effective hit is
  x0.5, resisted / immune unchanged, ignored by Mold Breaker): 31% of Terapagos-defender cases were off, engine up to
  4x too high. Fixed in `damage_calc.rs` (`attacker_ignores_target_ability` in `abilities.rs`).
- Acrobatics with no item after Tera applied the 60-BP floor before doubling (120 instead of 110).
- Sacred Sword / Darkest Lariat / Chip Away did not ignore the target's Defense boosts.

Remaining >5% cases are rounding on tiny damages (Slow Start halving), Lum Berry holders with a status (engine
cures first), Greninja-Bond arriving with only Water typing from the bot's species data (not an engine issue) and
Grassy Glide under Tera (calc and engine disagree on whether the 60-BP floor applies).
