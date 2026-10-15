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
