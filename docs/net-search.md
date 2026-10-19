# Net-guided search (Jaxcalibur-style), bead showdown-bot-589.68.5

`net-search` plays by a PUCT search in which **every expansion is evaluated by the token transformer**
(`models/tok-d256.pt`, docs/token-net.md): policy priors for BOTH sides plus a value, batched on the GPU. Our
existing MCTS does ~115k cheap iterations per decision with a tiny value net, so a better root prior cannot change
its move (docs/token-prior-diag.md). This search instead spends a few thousand expansions per decision, each one
priced by the big net.

Engine: wheel `0.0.48+sdb.24` (`poke_engine.NetSearch`); Python: `showdown_bot/net_search.py` (driver),
`showdown_bot/players/netsearch.py` (bot), `scripts/net_search_offline.py` (offline check).

## Design

### Rust (`engine/src/netsearch.rs`, pyclass in `engine/poke-engine-py/src/lib.rs`)

* **One tree per sampled world** (the search bots' world sampler; default 32). A tree is an arena of decision
  nodes; a joint action `(a1, a2)` leads to a chance node whose children are the engine's instruction branches.
  One outcome is sampled per descent, in proportion to its probability, exactly as `mcts.rs` does (damage rolls
  branch at depth <= 1 only, as there). Every descent starts from a clone of the world's root state, applies the
  sampled instructions on the way down and drops the clone: nothing is undone.
* **Decoupled PUCT per side.** Each side has its own prior, visit count and value sum per option.
  `score = Q + c_puct * P * sqrt(N) / (1 + n)`. Values are P(side one wins) in [0, 1]; side two stores `1 - v`.
  An unvisited option starts at `node mean - fpu` (the node's own mean from the side's view).
* **Expansion.** A descent that reaches an unevaluated node computes the legal options of both sides
  (`root_get_all_options`, the legality the tokens use) and writes the node's token buffers from **side one's view
  and (default) side two's view** (`tokens::state_tokens_into`; both sides are fully known inside a world, history
  rows are padding as in live play). The node becomes `Pending`.
* **Virtual loss.** Along every pending path each chosen option gets `vloss += 1`, which counts as a visit that
  scored 0, so the next descents choose other options. A descent that reaches a `Pending` node (chance sampling
  sent it there again) releases its virtual loss and counts a collision; a world with 8 free/colliding descents
  in a row is skipped for the rest of that `collect`.
* **Feed.** `feed` completes the OLDEST in-flight batch (several batches may be in flight): it sets both sides'
  priors from the 13-slot policies (read at each option's `action_slot`, renormalised over the legal options,
  mixed with `eps` = 0.01 uniform), stores the node's value, backs the value up the path and releases the virtual
  loss.
* **Depth limit** (default 4 turns): a node `depth` turns below the root is evaluated once for its value only
  (`Leaf`), and that value is backed up on every later visit without a net call. **Terminal** states (a side has no
  Pokemon left) back up their exact result, 1 or 0, without a net call. Both-`None` joint actions back up the
  node's own value.
* **GIL**: `collect` and `feed` run under `py.detach`, so the tree work of several games in one process overlaps.
  The tree itself is single threaded.

### Python

* `NetEvaluator`: the raw engine buffers go to the GPU as two tensors and are sliced into the model's input fields
  on the device (no per-state numpy work); one fp16 forward over `2n` rows (leaf i side one = row `2i`, side two =
  row `2i+1`); policy softmax per row (masked to the legal own slots by the net), value = sigmoid of the side-one
  row. Flags: `vavg=1` averages with `1 - value(side two's view)`; `views=1` runs only side one's view and takes
  side two's prior from the net's opponent-action head (half the GPU work, see below).
* `drive`: `collect(batch)` -> launch forward -> (with `pipe` >= 2 collect the next batch while the GPU works) ->
  `feed`, until `time_ms` or `max_evals`, or until the trees have nothing new to ask for (tiny trees: forced moves).
  The first batch is the roots, so every world is evaluated before it is expanded.
* `aggregate`: summed root visits of our options over the worlds. `mode=argmax` (default) plays the most visited
  option, `mode=mixed` samples from the visit shares.
* `NetSearchPlayer` (subclass of the token-policy bot): samples `worlds` worlds, scrubs the opponent for the side-one
  view with the set tracker's revealed facts (`scrub=1`, the training view), runs the search in a thread pool and
  plays the result. If the search raises, it plays the **token-policy argmax** (its own failure falls back to a
  random move) and counts `net_errors`; `net_searches`, `net_evals`, `net_batches`, `net_ms` are kept for logging.

## Spec

```
net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:max_evals=20000:batch=128:c_puct=2:depth=4:fpu=0
```

| key | default | meaning |
| --- | ---: | --- |
| `model` | required | token net `.pt` |
| `device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |
| `worlds` | 32 | sampled opposing teams (one tree each) |
| `time_ms` | 1000 | search budget per decision (wall clock, all included) |
| `max_evals` | 20000 | cap on net evaluations (leaves) per decision |
| `batch` | 128 | leaves per `collect` (per GPU forward); 256 is a little better (below) |
| `c_puct` | 2 | PUCT constant |
| `depth` | 4 | turns below the root; deeper nodes are value-only leaves |
| `fpu` | 0 | first-play reduction (`node mean - fpu`) |
| `vavg` | 0 | 1: value = mean of side one's value and `1 -` side two's value (needs `views=2`) |
| `scrub` | 1 | 1: scrub the opponent in side one's view to the revealed facts (training view); 0 is much worse |
| `pipe` | 1 | batches in flight (2: collect while the GPU runs); no measurable gain, the GPU is the bottleneck |
| `views` | 2 | 2: run both sides' views; 1: side one only, opponent prior from the opponent head (faster, weaker) |
| `mode` | `argmax` | `argmax` or `mixed` over the summed visits |
| `fast` | 0 | 1: `fast_forward.FastForward` (fp16 weights, drops the masked history tokens when `hist=0`, CUDA graphs per padded shape, pinned buffers); same policy up to fp16 rounding |
| `fp16res` | 0 | 1 (with `fast`): also compaction, last-layer pruning and an fp16 residual stream; fastest, tiny numeric drift |
| `hist` | 0 | 0: masked history tokens (training-time padding); 1: live `BattleHistory` of the real battle (`history_tokens.py`) shared by all worlds; 2: 1 plus the per-world action rows the engine's `take_hist()` provides |
| `scrub2` | 0 | 1: also scrub side two's view to what the OPPONENT has seen of us (`RevealTracker`) instead of showing it our whole team |
| `tokthreads` | 1 | 1..4 threads for the Rust tokeniser (the GIL is released) |

`showdown_bot.league.parse_spec` validates all of it (duplicate keys, unknown keys, ranges); `c_puct` and `fpu`
do not require a `.ppol` for this type. The bot is registered as `net-search` in `bench.py` / `league.py`.

## Measured

Machine under load (a league with ~18 workers on the CPU), GPU shared with nothing else. 300 held-out self-play
decisions (gen7/8/9: 100 each, `scripts/net_search_offline.py`, the loader of `scripts/diag_tok_prior.py`: val-split
battles, <= 2 decisions per recording, >= 2 legal actions), target = the recorded search's aggregated visit argmax.
The recordings hold 4 worlds per decision; they are repeated to 32 trees (each tree samples its own chance
outcomes). Search = `time_ms=1000`, `depth=4`, `c_puct=2`, `batch=256`, `pipe=2`, both views, `scrub=1`.

### Throughput and latency (1000 ms per decision, mean over the 300 decisions)

| | |
| --- | ---: |
| net evaluations per decision | 7,350 |
| evaluations per second | 7,300 |
| GPU batches per decision | 33 (batch 256) / 53 (batch 128) |
| max tree depth reached | 4.0 |
| Rust tree (select, expand, backprop) | 43 ms |
| Rust tokenise (2 views x 7,350 leaves) | 209 ms |
| Python `collect` call total (tree + tokenise + byte copy) | 312 ms |
| GPU forward incl. host copy (blocking part) | 690 ms |
| `feed` | 4 ms |

The forward is the bottleneck: ~14 ms per 256-leaf batch (512 rows), about 9 ms of it fixed launch cost under
the current CPU load. `batch` 128 / 256 / 512 give 6.3k / 7.2k / 7.4k evaluations/s; `pipe` 2-3 changes nothing
because the GPU is saturated. `views=1` (half the rows) reaches 11.1k evaluations/s. Ideas not done: drop the 16
masked history tokens at inference (-17% tokens), CUDA graphs / `torch.compile` for the fixed launch cost, tokenise
once per leaf and share the matchup block between the two views.

### Agreement with the recorded search (top-1 of the visit argmax, n = 300)

| | top-1 | CE of the visits (after 3% mix) | agrees with raw net |
| --- | ---: | ---: | ---: |
| raw net, live path (this subset; the 2,920-decision figure is 0.579) | 0.607 | 1.408 | |
| **net-search base** (depth 4, c 2, both views, scrub) | **0.643** | **1.351** | 0.877 |
| depth 2 | 0.607 | 1.369 | 0.810 |
| depth 6 | 0.647 | 1.353 | 0.873 |
| c_puct 1 | 0.640 | 1.355 | 0.867 |
| c_puct 4 | 0.627 | 1.368 | 0.920 |
| `vavg=1` | 0.637 | 1.355 | 0.880 |
| `views=1` (opponent head, 11k evals/s) | 0.593 | 1.424 | 0.760 |
| `scrub=0` (side one's view unscrubbed) | 0.497 | 1.699 | 0.547 |
| `time_ms=300` (2.3k evals, 11 batches) | 0.613 | 1.409 | 0.960 |
| `time_ms=3000` (21.5k evals, 98 batches) | 0.633 | 1.312 | 0.807 |

By kind of the recorded argmax, base: move 0.67 (raw 0.66), switch 0.61 (raw 0.53), tera 0.43 (raw 0.14);
per generation raw / search: gen7 0.63 / 0.65, gen8 0.62 / 0.69, gen9 0.57 / 0.59.

Reading: the search moves the raw net's choice on 12% of decisions and the changes are net-positive against the
recorded search (+3.6 points top-1 over the raw net, CE -0.06), mostly on switches and tera. n = 300 means one
standard error of ~2.8 points, so the +3.6 is suggestive, not significant by itself; the per-gen and per-kind
splits point the same way. More time sharpens the visit distribution (CE 1.409 / 1.351 / 1.312 at 0.3 / 1 / 3 s) while top-1 is flat within noise. Agreement with a *search* that used a
different evaluator is a weak proxy for strength: the SPRTs below decide. `scrub=0` and `views=1` are clearly
worse on this metric, so the defaults stay `scrub=1`, `views=2`.

Live smoke (local server, `worlds=32`, `time_ms=1000`, `batch=256`, real net on CUDA, 2 games vs random): 2/2 won, 38 searches,
0 errors, 7,080 evaluations and 30 batches per decision, 989 ms per search.

### Speedups, history and side-two scrub (bead 589.68.7, engine sdb.25)

Measured while a league SPRT used ~18 CPU cores and another process held the GPU at ~90% utilisation, so absolute
numbers are lower than the table above (baseline 3.7k evals/s here vs 7.3k there); compare rows within this table
only. Offline: 300 decisions, batch 256, time_ms 1000. Top-1 is against the recorded search argmax; n = 300 so the
standard error is about 2.8 points (differences under ~5 points are noise); CE is the visit distribution's cross
entropy to the target (lower is better).

| config | evals/s | evals/dec | top-1 | CE |
| --- | ---: | ---: | ---: | ---: |
| baseline (pipe 1) | 3,702 | 3,778 | 0.643 | 1.370 |
| `pipe=2` | 3,722 | 3,811 | 0.627 | 1.386 |
| `fast=1 pipe=2` | 10,742 | 11,027 | 0.643 | 1.340 |
| `fast=1 fp16res=1 pipe=2` | 16,745 | 16,871 | 0.637 | 1.317 |
| `tokthreads=2` | 7,326 | 7,368 | 0.653 | 1.341 |
| `hist=1` | 6,816 | 6,844 | 0.680 | 1.301 |
| `hist=2` | 6,759 | 6,782 | 0.677 | 1.295 |
| `scrub2=1` | 6,458 | 6,512 | 0.653 | 1.353 |
| all (fast, pipe 2, hist 2, scrub2, tokthreads 2) | 13,825 | 13,959 | 0.657 | 1.280 |
| all + `fp16res=1` | 15,195 | 15,308 | 0.663 | 1.277 |

Raw net (no search), same 300 rows: top-1 0.607 / CE 1.408 with masked history, 0.620 / 1.357 with the real history
(`--raw-hist 1`). Search lifts both; real history helps the raw net and the search by a similar amount (CE -0.03 to
-0.07), top-1 gains are within noise except `hist` (+3 to +4 points). Throughput rows change evals per decision, not
tree depth (about 3.97 everywhere), so more evals mostly mean better-converged visits (CE drops 1.37 -> 1.28).

Forward microbench (`.cache/ns25/bench2.py`, interleaved, min of 12 rounds, ms per 256-leaf batch; GPU contended so
the medians are 30-50% above the minima): eager autocast 41.8; fp16 weights 37.5; + drop history tokens 32.4; + CUDA
graphs 24.2; compaction 29.6-29.9 (worse than graphs alone under contention); + last-layer pruning 28.0; + fp16
residual 26.8. End to end `fast`+`fp16res` is about 4.5x the baseline offline throughput.

**Live history gap / exporter leak.** The live `BattleHistory` is built from protocol events only. When the
opponent's chosen move did not execute (asleep, flinch, full paralysis, KO'd before moving, etc.) the live row has
no action (kind 0), while the training exporter wrote the RECORDED chosen action, which is information the
live bot cannot see. Live is the correct behaviour (no hidden information) and must not guess. This is a leak in
the exporter's history rows for future exports: they should only contain actions that were observable in the log.
In the live-vs-exporter smoke (239 decisions, 3,824 rows) 96.9% of rows were identical; the differences were 95
"action unknown on live side" (this leak), 9 different action, 8 own hp/faint numbers and 8 opponent actor species
(0.2% of rows each; not investigated further, the smoke harness was not kept).

## How to SPRT

Do not run these on a busy machine: each net-search decision uses a Rust thread for ~0.3 s and the GPU for ~0.7 s,
and `--concurrency` lanes of the candidate share one GPU. Candidate first, baseline second; `--teams swap` is implied.

1. **net-search vs the raw token policy** (gate: >= +80 Elo, i.e. the search must add at least that much to its own
   net):

```
python -m showdown_bot.league --bots "cand=net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4,base=token-policy:model=models\tok-d256.pt:device=cuda" --sprt 0,21 --max-games 2000 --search-workers 18 --game-timeout 3600 --concurrency 4 --teams swap --out data/league/sprt-netsearch-vs-tokpol --seed 1
```

   (To stop at the +80 gate instead of the default +21 alternative use `--sprt 0,80`.)

2. **net-search vs the current best search bot**:

```
python -m showdown_bot.league --bots "cand=net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4,best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev" --sprt 0,21 --max-games 2000 --search-workers 18 --game-timeout 3600 --concurrency 4 --teams swap --out data/league/sprt-netsearch-vs-best --seed 1
```

   `--strict-workers` (implied) caps the lanes at `search_workers / worlds of the baseline` = 4 here.
   Check `net_errors` (fallbacks to the token policy) in the logs: a non-zero count means the run measured a mix.

## Tests

* Rust (`cargo test --lib netsearch`, 13 tests): PUCT picks follow the priors of both sides, a stronger value beats
  a larger prior, virtual loss diverts descents and is fully released, a flat prior with virtual loss opens many
  children in one batch, priors are renormalised over the legal options, value perspective (side one `v`, side
  two `1 - v`), the depth limit reuses leaf values without a net call, terminal roots and terminal children back up
  exact 0/1, collisions at a pending root, option slots agree with the token `act_legal` flags of each view in
  8 states, one-view mode, seed determinism, a fake evaluator throughout.
* Python (`tests/test_net_search.py`): spec parsing/rejection, the documented SPRT specs, the 13 slot names of both
  sides' views against the option names `NetSearch` reports, one-hot policies steering each side's root prior,
  `feed` shape validation, a smoke search with a dummy evaluator (counts, depth, legality), the evaluator against
  `TokenNet.predict` for both views, a tiny-net `NetSearcher`, the real net on a real state, and (when the local
  Showdown server is up on :8000) a one-game smoke vs random.

## Limitations

* History rows are padding (as for the token-policy bot); the net was trained with 16 event rows.
* Worlds from the sampler fill the unrevealed opponent sets; the side-one view is scrubbed back to the revealed
  facts (`scrub=1`), the side-two view is not (it is what the opponent sees of its own team, and of ours).
* The opponent in every world is assumed to play the net's policy-guided PUCT; there is no exploitation or
  opponent modelling, and no endgame solver.
* Interior positions repeat the root's scrub rows: a Pokemon revealed in the tree but not at the root stays
  hidden for the side-one view.
* Free descents (terminal or already-evaluated leaves) stop after 8 in a row per world per `collect`; very small
  trees (forced moves, last-Pokemon endgames) end early with fewer evaluations than the budget.

### SPRT commands for the sped-up spec (589.68.7)

NEW = `net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4:pipe=2:fast=1:fp16res=1:hist=2:scrub2=1:tokthreads=2`
(wheel sdb.25 required; validated with `parse_spec`).

```
python -m showdown_bot.league --bots "cand=NEW,base=net-search:model=models\tok-d256.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4" --sprt 0,21 --search-workers 18 --game-timeout 3600 --concurrency 1 --out data/league/sprt-ns25-vs-ns --seed 1
python -m showdown_bot.league --bots "cand=NEW,best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev" --sprt 0,21 --search-workers 18 --game-timeout 3600 --concurrency 1 --out data/league/sprt-ns25-vs-best --seed 1
```
(substitute the NEW spec text for `NEW`.)
