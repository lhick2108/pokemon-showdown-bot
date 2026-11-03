# Evaluator speed: ft-v5-h128 vs gen7, line-cache fix (sdb.29), distilled v5 nets (589.70)

## Summary

* The biggest single cost of `ft-v5-h128` in a real search was **not** the hidden layer but
  **misses of the per-Pokemon line cache** (`learned.rs`, `Scratch::cache_*`): with 2 ways per
  set, 18% of leaf evaluations in a 1 s search recomputed at least one line (40 ladder states,
  ft-v5-h128, sdb.28). A v5 miss re-adds about 150 first-layer rows (types, stats, v2 static
  block, both identity-embedding variants). Engine **sdb.29** uses 4 ways: 0.7% (8 ways: 0.4%).
  Evaluations are bit-identical (see Equivalence). The churn is worse on sdb.28 than sdb.19
  (more item changes inside the tree: Harvest, Choice-lock release, Unburden), which is part of
  why ft-v5 lost more speed on sdb.28.
* With the cache fixed, the hidden-size-dependent part is about half of a v5 h128 leaf; the
  other half is v5 feature code that does not depend on H (v2 dynamic block ~0.1 us, 3-word
  keys, ~34 queued rows vs ~20 for v1, stock `evaluate` for the residual).
* **Distillation** (`train.py --distill ft-v5-h128.pev`) gives v5 nets at hidden 64 / 32 / 16
  with the teacher's offline accuracy (h64 is practically the teacher, logit RMS 0.025; h32 RMS
  0.11; h16 RMS 0.18). With the cache fixed, h16 / h32 / h64 search at about the same speed
  (the H-independent part dominates), so h32 is the sweet spot.

## Speed (scripts/iter_bench.py, 40 recorded w16 ladder states, 1 s, 1 thread)

All specs use the `best` preset (prior-gen789-h32, scale 1.5, c_puct 2) except `best-gen7`
(prior-gen56-h32; both priors are v2 h32, same cost). Measured on 2026-11-03 while a 22-worker
SPRT loaded the machine, so absolute numbers are ~45% below an idle box; ratios are paired per
state (geometric mean of per-state ratios vs `best-gen7` on the same engine, +- 1 s.e.,
2 repeats each, `--json` + per-state pairing).

| evaluator | engine | median it | vs gen7 (same engine) |
| --- | --- | --- | --- |
| gen7-res-root-mlp16 (`best-gen7`) | sdb.19 | 117.8k | 1.000 |
| ft-v5-h128 (`best`) | sdb.19 | 96.8k | 0.865 +- 0.014 |
| dist-v5-h32 | sdb.19 | 104.5k | 0.906 +- 0.012 |
| gen7 | sdb.28 | 111.3k | 1.000 |
| ft-v5-h128 | sdb.28 | 86.5k | 0.825 +- 0.015 |
| dist-v5-h32 | sdb.28 | 96.0k | 0.912 +- 0.015 |
| gen7 | sdb.29 | 112.5-118.8k | 1.000 (sdb.28: 0.969 +- 0.009 of sdb.29) |
| ft-v5-h128 | sdb.29 | 98.0-102.0k | 0.875-0.895 (three batches) |
| dist-v5-h64 | sdb.29 | 102.5k | 0.921 +- 0.015 (one batch) |
| dist-v5-h32 | sdb.29 | 103.8-104.8k | 0.923-0.956 (three batches) |
| dist-v5-h16 | sdb.29 | 105.3k | 0.932 +- 0.018 |

sdb.28 -> sdb.29 on the same states: ft-v5 +13% iterations, dist-v5-h32 +8%, gen7 +3%.
`dist-v5-h32` on sdb.29 searches about as many iterations as gen7 did on sdb.28 (0.987 +-
0.013). gen7 itself is equally fast on sdb.19 and sdb.29 in these runs.

`engine/examples/bench_net.rs` (no prior, 40 states, MCTS 10k iterations, loaded box): MCTS
it/s ratio vs stock evaluate, sdb.28 code -> sdb.29 code: ft-v5-h128 0.70 -> 0.78, gen7 0.885 ->
0.91; dist-v5-h32 on sdb.29 0.83. Hot single-state `Evaluator.logit` (cache always hits):
gen7 ~0.47 us, dist-v5-h32 ~0.91 us, ft-v5-h128 ~1.27 us (loaded box, ~2x idle).

Where the v5 h128 hot leaf goes (instrumented build, loaded box, ns): total 1437; queued rows
(~33.5 per leaf) ~260, the 12 cached lines ~150, output layer (128 serial adds, kept serial
for bit-exactness) ~120; everything H-independent ~770 (v2 dynamic block ~180-200, stock
evaluate ~70-140, keys / lookups / queueing). gen7 total ~420.

## Engine changes (sdb.29, `engine/src/learned.rs`)

1. `CACHE_WAYS` 2 -> 4, round-robin eviction (the dominant win, above).
2. The 12 cached lines are no longer added to the accumulator one by one in memory; their
   offsets are queued and `Acc::flush` adds them in register-resident chunks before the queued
   rows (same per-lane order: bias, lines, rows -> bit-identical).
3. `Acc::flush` chunk 32 -> 64 floats (8 independent add chains; same order).

Not done (would change bits, so only worth it as an opt-in file flag): a tree reduction of
the output layer (~0.1 us at h128, ~0.02 us at h32), FMA, f16/int8 weights. The hot loop is
latency- not bandwidth-bound once the line cache hits, so quantisation would buy little.

### Equivalence

* `cargo test --lib learned` (23) and `prior` (28) pass; new test
  `learned::embed_tests::line_cache_is_bit_exact_under_eviction` cycles 9 key variants per
  Pokemon through one `Scratch` (hits, misses, evictions; v5 h16 / h128 and v1 h16) and asserts
  bit-equality with a fresh `Scratch`.
* `scripts/eval_equivalence.py --engine sdb.28 --engine sdb.29 --n 2000`: identical float32
  bits for gen7-res-root-mlp16, ft-v5-h128, ft-v4-h128, human-v4-h128 (2000 ladder states each)
  and `PolicyPrior.option_logits` of prior-gen789-h32 / prior-gen56-h32 (36,234 logits each).
* `bench_net` prints a logit fingerprint; equal for the sdb.28 code and sdb.29 code on 1000
  ladder states for gen7, ft-v5-h128 and truncated v5 h32 / h64 nets.

Wheel: `engine/wheels/poke_engine-0.0.48+sdb.29-cp312-cp312-win_amd64.whl` (= sdb.28 + the
above; build with `scripts/build_engine_wheel.ps1`). Use `engine=sdb.29` in a league spec or
`--engine sdb.29` in iter_bench / human_eval.

## Distillation (`train.py --distill`)

`--distill TEACHER.pev` loads the teacher (same `--features`, any hidden size, residual) as a
torch model, evaluates it on the GPU on the loaded feature rows and replaces the train and val
targets by `A * sigmoid(teacher) + (1 - A) * label` (`--distill-weight A`, default 1).
`--distill-init trunc` (default) starts the student from the teacher's embedding tables and
its `--hidden` hidden units of largest `|w2|` (and the teacher's stock coefficient `a`);
`emb` copies only the tables; `none` is a random start. The log adds the teacher's val BCE,
the student's BCE vs the original labels and the student-teacher logit gap.

Commands (PowerShell, repo root; same data / split / val-data as ft-v5-h128, so the val split
is identical; single-process loading, ~14 min load + ~1 min GPU per run):

```powershell
python -m showdown_bot.train --data data\selfplay\gen8 --data data\selfplay\gen9 `
    --distill models\ft-v5-h128.pev --states prefer-truth --value-target search --q-kind absolute `
    --model mlp --residual --features v5 --embed-dim 8 --hidden 32 --x-dtype float16 `
    --load-workers 1 --batch-size 1024 --lr 1e-3 --weight-decay 1e-4 --epochs 30 --patience 3 `
    --out models\dist-v5-h32.pev --val-data data\human\pc1-fp-aux\test@4
```

(`--hidden 16` / `64` for the others; logs: `models/log-dist-v5-h{16,32,64}.txt`.)

## Offline accuracy

train.py val split of gen8+gen9 self-play (500 held-out battles, 41,596 samples, weighted, as
in `log-ft-v5-h128.txt`; "search label" = the `--value-target search` target ft-v5 was trained
on):

| net | val BCE vs outcome (w / unw) | decisive acc | BCE vs search label | human test@4 BCE / acc (264k) |
| --- | --- | --- | --- | --- |
| ft-v5-h128 (teacher) | 0.5658 / 0.5932 | 0.698 | 0.4599 | 0.6802 / 0.663 |
| dist-v5-h64 | 0.5657 / 0.5935 | 0.698 | 0.4599 | 0.6801 / 0.663 |
| dist-v5-h32 | 0.5655 / 0.5934 | 0.697 | 0.4602 | 0.6811 / 0.663 |
| dist-v5-h16 | 0.5669 / 0.5955 | 0.698 | 0.4605 | 0.6826 / 0.663 |

`scripts/human_eval.py` (engine `Evaluator.logit`, so gen7's v1 features can be compared too;
logit gap vs ft-v5-h128 as RMS):

| net | self-play val battles, all truth decisions (38,794): BCE / acc / RMS vs ft-v5 | human test, first 20,000 games x 4 (79,410): BCE / acc / RMS vs ft-v5 |
| --- | --- | --- |
| ft-v5-h128 | 0.5915 / 0.697 / 0 | 0.6875 / 0.664 / 0 |
| gen7-res-root-mlp16 | 0.5838 / 0.699 / 0.598 | 0.6728 / 0.660 / 0.623 |
| dist-v5-h64 | 0.5916 / 0.697 / 0.025 | 0.6873 / 0.664 / 0.026 |
| dist-v5-h32 | 0.5917 / 0.696 / 0.109 | 0.6887 / 0.663 / 0.107 |
| dist-v5-h16 | 0.5939 / 0.698 / 0.176 | 0.6901 / 0.663 / 0.179 |

(gen7's lower outcome BCE on these sets is the known calibration difference: ft-v5 was fit to
search values and is over-confident vs the raw outcome; outcome BCE alone did not predict
playing strength for these nets, see training.md.) The distilled nets differ from ft-v5 by a
fraction of how much gen7 differs from it.

Tools: `scripts/human_eval.py` now takes repeated `--data`, `--val-split F [--split-seed S]`
(the val battles of train.py's split over those dirs) and `--engine TAG`, and prints each
model's logit gap to the first model.
