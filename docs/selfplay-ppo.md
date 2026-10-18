# Self-play PPO for the token transformer (bead showdown-bot-589.68.6)

Infrastructure for a Jaxcalibur-style reinforcement-learning loop: random-battle games simulated entirely in Rust
(`poke_engine.SelfPlayPool`, wheel `0.0.48+sdb.26`), a PPO trainer (`src/showdown_bot/ppo.py`) and checkpoints in
the existing `TokenNet.export` format. The supervised net (`models/tok-d256.pt`) is the starting point and the frozen
anchor. Files:

| file | what |
| --- | --- |
| `engine/src/selfplay.rs` | team generator, game state machine, reveal tracking, history rows, the pool |
| `engine/poke-engine-py/src/lib.rs` | `SelfPlayPool`, `selfplay_teams`, `selfplay_data_info` bindings |
| `engine/data/randbats_gen9.txt` | baked team data (`scripts/gen_selfplay_data.py`, 760 KB, force-added: `data/` is ignored) |
| `src/showdown_bot/ppo.py` | PPO trainer, opponent pool, distillation, `train` / `bench` / `eval` CLI |
| `tests/test_ppo.py`, `selfplay::tests` | Python / Rust tests |

## The pool (`SelfPlayPool`)

```python
pool = poke_engine.SelfPlayPool(n_games=512, seed=0, threads=4, turn_cap=200)
obs = pool.observe()      # dict of byte buffers for every (game, side) that has a real decision
rew, done, finished = pool.step(actions)   # actions: one byte (slot 0..12) per observed row, in row order
pool.stats(); pool.game_state(i); pool.set_threads(n)
```

`observe()` keys (little endian): `n`; `game` int32[n]; `side` uint8[n] (0 = side one); `turn` int32[n];
`cat` int16[n, 262] and `cont` float32[n, 3911] (the token buffers, `tokens.layout()` offsets, see `docs/tokens.md`);
`hist_cat` int16[n,16,4], `hist_cont` float32[n,16,5]; `legal` uint8[n,13] (also `act_legal[0]` inside `cat`).
`step()` returns float32 `reward[n]` / uint8 `done[n]` per observed row (+1 / -1 / 0 at the end of the game, else 0) and
`finished = [(game, winner 0|1|2, turns)]` for every game that ended in this step. **Use `finished` for the terminal
reward of a seat that had no row in the final step** (a side without a real decision is played automatically); `ppo.py`
does. Finished games are reset immediately with a new team pair (a fresh `episode` of the same slot), so the next
`observe()` is already the next game. `observe` and `step` release the GIL; the work is split over `threads` scoped Rust
threads (std threads with dynamic chunking; `rayon` is not in the offline crate cache). Results do not depend on the
thread count: every game owns an RNG seeded from `(seed, slot, episode)`.

A row exists for a side only if it has **more than one legal action**. Forced switches after a faint, the passive side
of a forced switch, Struggle, a locked-in move without alternatives are played automatically (they are not decisions
in the supervised data either). Action slots are the token net's 13 slots (4 moves, 4 Tera moves, 5 switches in slot
order). An illegal slot is replaced by the first legal action and counted in `stats()["illegal"]` (always 0 in the
tests and runs). The engine step samples the chance node of `generate_instructions_from_move_pair(..., branch_on_damage=true)`
in proportion to its probability (crits and kill rolls are branches, other damage is the average roll, like
`mcts.rs`/`netsearch.rs`). A panic inside the engine for one game is contained (`catch_unwind`): the game is scored
a draw and replaced (`stats()["panics"]`, 0 in the fuzz: 256 games x 3000 steps of random play, ~12k games). Turn cap 200
battle turns (or 1200 engine steps) = draw.

### Teams (`selfplay.rs`, `scripts/gen_selfplay_data.py`)

Source: pkmn/randbats **full** joint-set data (`.cache/randbats/full/gen9randombattle.json`, the file
`showdown_bot.sets` already samples live opponents from): exact `level,item,ability,moves,tera` sets with the number of
times each appeared in 100k generated teams, plus the dex and max PP. A team is drawn like `engine_state.TeamSampler`:
species proportional to the summed count of their sets, rejected when they break a randbats team rule (one per base
species, at most two of a type, at most three weak to a type and one 4x weak, at most four weak to Freeze-Dry, at most
one level-100, the incompatible-pair list, no Zoroark in the last slot; after 40 rejections the soft rules are dropped
as in the live sampler), then a set proportional to its count. EVs/IVs are what `teams.ts` picks
(`engine_state.randbats_evs_ivs`: HP EVs for Stealth Rock parity, 0 Atk for no-Attack sets, 0 Spe for Gyro Ball/Trick
Room), neutral nature, stats by the Showdown formulas, full PP tracking (`max_pp` set). 508 species, 5800 sets, no set
dropped. Test: 300 teams are legal engine states (serialise round trip, options, all slots filled, species clause).

Simplifications (documented, not hidden):

* **Not Showdown's role-based generator**: the frequencies are the empirical ones of that generator (full-set counts), the
  team rules are the ones the live opponent model already uses (a port of `getPokemonCompatibility`/`randomTeam`), but
  other generator details are not reproduced (e.g. Tera-type/role dedupe across the team, per-role move-count
  rounding). Sets that never occurred in the 100k teams cannot occur.
* **Ditto is excluded** (Imposter needs the live bot's pre-transform bookkeeping and rewrites the move list that reveal
  tracking keys on). Zoroark's Illusion is not modelled (the engine treats it as a normal ability).
* Opponent EV/IV spreads are exactly those of the set; there is no hidden-information mismatch between a side's "true" team
  and what the other side's sampler would guess (live play has that mismatch; here each side sees the scrubbed truth).
* Lead = first drawn Pokemon (random), team order random; no team preview in gen9 random battles.

### Per-side views and reveal tracking

Every side's tokens come from `tokens::state_tokens_into(state, side, Some(reveal_rows))`: the opponent is scrubbed
exactly as in live play (`docs/tokens.md`, TOKEN_VERSION 2: unrevealed item/ability/moves/Tera -> unknown, a
never-seen bench Pokemon is canonicalised away, including from the matchup aggregates). The reveal sets are updated
after every engine step from state differences, per side, per team index:

| revealed | when |
| --- | --- |
| Pokemon (species) | it is the active one at the start, switched in (also if it left or fainted within the turn), dragged in, or any active Pokemon after the step |
| move | the engine spent PP on it (the move was executed; a chosen move that never executed stays hidden) |
| item | the item changed (consumed, knocked off, swapped); Leftovers/Black Sludge when the holder healed without a healing move; Life Orb when the holder used a damaging move and lost HP; Rocky Helmet when the holder lost HP to a contact move |
| ability | it changed, or a switch-in with an announcing ability (Intimidate, weather/terrain setters, Pressure, Unnerve, Trace, Download, Frisk, Anticipation, Forewarn, ...) |
| Tera type | the Pokemon terastallized |

These are heuristics for what the Showdown protocol shows (a live tracker also learns from damage ranges, speed, ...,
so the training view can be a little *less* informed than a live one, never more). Own legality comes from the truth
state (the scrubbed copy hides a trapping ability such as Shadow Tag, which the request would show a live bot).

**Test** (`selfplay::tests::unrevealed_opponent_info_never_appears_in_tokens`, 12 games x 40 steps, every observation):
the token buffers of a row are compared with the buffers of a *tampered* copy of the game in which every unrevealed
thing of the opponent (unseen bench Pokemon replaced by other teams' Pokemon with other HP, unrevealed items, abilities,
moves and Tera types) is changed: all of `cat` and `cont` must be bit-identical. It also checks that every opponent
entry of the history (switch target, executed move, actor) belongs to a seen Pokemon / revealed move. (The test found
and the code fixes: the trapping-ability legality case above.)

### History rows

`hist_cat` / `hist_cont` follow the exporter's format (`docs/tokens.md`, "Event history"): the previous 8 engine steps,
most recent first, row `2k` = this viewer's action of step `k`, row `2k+1` the opponent's. `hist_cat` =
`[side (1 own, 2 opp), kind+1 (1 move, 2 Tera move, 3 switch, 0 unknown), move/species row, species row of the actor's
active Pokemon]`; `hist_cont` = `[turns_ago/16, hp change of the actor's active, hp change of the other active,
fainted on the actor's side/6, fainted on the other side/6]`, the deltas computed on the viewer's own visible
summary (hidden opponent Pokemon do not count), exactly `tokens.build_history`. Differences from the supervised
export: every engine step is an entry (forced switches included, with an all-zero row for the side that did not
act); the opponent's move is its *executed* move (kind 0, id 0 if it never executed), the viewer's own action is what it
chose; a fainted actor has species row 0, like the exporter. The live bots still feed padding history rows (docs/net-search.md),
so a net trained with real history needs the live history tracking to profit from it.

## The trainer (`ppo.py`)

* **Actor-critic on the token net**: policy head = masked softmax over the 13 slots, value head = logit of P(win)
  (value in reward units `v = 2 sigmoid - 1`; reward +1 / -1 / 0 only at the end of the game). Other heads (opponent action,
  hidden sets, aux) are not trained and not used by `token-policy` / `net-search` (except `views=1`, which uses the
  opponent-slot head; PPO leaves it frozen but its trunk moves).
* **Rollout**: `rollout_steps` pool steps; per step `observe`, one batched fp16 forward per acting model, multinomial
  sampling at temperature 1 (training and opponents), `step`. Learner transitions are stored per (game, seat) trajectory
  (float16 `cont`, as the net was trained on float16 shard inputs).
* **GAE** (`compute_gae`, gamma 0.997, lambda 0.95) per seat trajectory; trajectories cut at the end of a rollout are
  bootstrapped with the value of their pending next observation.
* **Update**: clipped surrogate (clip 0.2), value loss = BCE of the value logit against `(return+1)/2`, entropy bonus,
  `beta * KL(pi || anchor)` over the legal actions (anchor = the frozen supervised net; `beta` decays linearly
  `kl_beta0 -> kl_beta1` over `kl_decay_updates`), AdamW, grad clip 0.5, fp16 autocast + GradScaler, early stop of the epochs when
  the sampled KL(old || new) exceeds 1.5 x `target_kl`. **Warm-up**: for the first `value_warmup` updates there is no
  policy gradient (the critic is fitted first) but the KL-to-anchor term keeps the policy where it is (without it the shared
  trunk drifted with the value loss alone: after 4 updates KL 0.045 and the win rate against the anchor fell from 0.50 to 0.37).
* **Opponent pool** (per game slot, re-drawn when its game ends): `p_self` the latest learner on both seats (both seats
  train), `p_anchor` the frozen supervised net, the rest a random past checkpoint (snapshot every `snapshot_every`
  updates, at most `max_past` kept on the GPU; anchor while none exists). Against an opponent the learner takes a random seat. The
  rolling win rate (`window` games) against the anchor, past checkpoints and (seat one in) self-play is logged; against the anchor it is
  the headline number.
* **Memory**: `--gpu-mem-gb` (default 6) sets a hard CUDA cap for the process. At 512 games and minibatch 512 the peak is about
  3.4 GB of GPU memory (1024 peaks at 6.5 GB) and ~2.3 GB of host RAM (rollout buffers: ~33k transitions x 8.3 KB = 270 MB, the pool, torch).
* **Distillation** (`--arch d192L4 --distill-updates N`): a fresh d192/4-layer/8-head net (the teacher's token spec) is trained
  to match the teacher's policy (KL) and value (soft BCE) on states from the teacher playing itself in the pool, then PPO runs on
  the student with the teacher still as anchor and as the win-rate yardstick (`distilled.pt` is written to the run directory).
* **Checkpoints**: `ckpt-NNNNN.pt` and `latest.pt` every `ckpt_every` updates (and at exit) with `TokenNet.export` (state dict +
  config + token-spec hash + a `.json`), plus `state.pt` (optimizer, counters; `--resume`), `metrics.jsonl` (one JSON line per update:
  games/s, decisions/s rollout and end to end, kl_anchor, approx_kl, entropy, value loss, explained variance, clip fraction, win rates,
  GPU MB, pool panics/illegal counts) and `config.json`. **Format = the supervised nets' format**: any PPO checkpoint works as
  `token-policy:model=<ckpt>`, `net-search:model=<ckpt>` and `tok_prior=<ckpt>` (`tests/test_ppo.py` loads one through
  `TokenPolicy.load`). The `distill`/`small` nets load the same way (config carries the width).

## Measured

Machine: RTX 3080 (shared), 24 cores of which a league SPRT used ~18 the whole time, so everything below is pessimistic and
noisy (the same bench gave 16.9k decisions/s at 512 games when the GPU was quiet and 7.6k when another job shared it).

**Throughput** (`ppo bench`, tok-d256 fp16 forward on the GPU included, 30 steps, 4 Rust threads, random-battle pool):

| games in pool | rows/step | decisions/s (quiet GPU) | decisions/s (GPU shared) | observe / forward / step ms (quiet) |
| ---: | ---: | ---: | ---: | --- |
| 64 | 113 | 6.6k | | 2.4 / 13.9 / 0.9 |
| 128 | 226 | 11.5k | 5.5k | 4.0 / 14.6 / 1.0 |
| 256 | 454 | 14.4k | | 7.3 / 22.7 / 1.5 |
| 512 | 904 | **16.9k** | 7.6k | 12.6 / 39.2 / 1.9 |
| 1024 | 1811 | | 7.9k | 26.6 / 200 / 3.4 (shared) |

Rust threads at 512 games (shared GPU): 1 thread 6.7k, 2 threads 7.2k, 4 threads 7.6k decisions/s (the forward dominates; the
engine step is ~2-4 ms for ~450 games). Pure pool speed with random actions: ~39k decisions/s, ~1000 games/s at 256 games / 4 threads.
Target (>= 1000 decisions/s end to end) is met by 7-17x for sampling alone, and **including the PPO update** the whole loop
(rollout + 2 epochs over ~33k transitions) runs at 1.0-2.4k decisions/s (60-150 games/s, the spread is the CPU/GPU contention);
the update (about 15 s per 33k samples) is the bottleneck, not the pool.

**Smoke run** (`.cache/runs/smoke1`, 26.6 min, 48 updates, 2.09 M decisions, 35k games, tok-d256 -> tok-d256, 512 games, 4 threads,
`--lr 2e-5 --kl-beta0 0.05 --kl-beta1 0.01 --kl-decay-updates 100`, value warm-up 8 updates, peak GPU 3.45 GB, host RSS 2.3 GB). Rolling
win rate of the learner against the frozen supervised net (sampling at T=1 on both sides, last 400 games):

| update | min | wr vs anchor | wr vs past ckpts | KL(pi, anchor) | entropy | value loss | explained var |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7 | 0.59 (n=240) | | 0.000 | 1.46 | 0.478 | 0.60 |
| 5 | 3.2 | 0.53 | 0.49 | 0.001 | 1.39 | 0.464 | 0.63 |
| 9 | 6.1 | 0.51 | 0.53 | 0.008 | 1.32 | 0.455 | 0.60 |
| 13 | 8.2 | 0.71 | 0.62 | 0.078 | 1.06 | 0.479 | 0.59 |
| 21 | 11.8 | 0.78 | 0.71 | 0.180 | 0.85 | 0.491 | 0.60 |
| 29 | 14.8 | 0.81 | 0.71 | 0.242 | 0.76 | 0.513 | 0.56 |
| 37 | 19.5 | 0.87 | 0.71 | 0.266 | 0.73 | 0.499 | 0.58 |
| 45 | 24.3 | 0.80 | 0.62 | 0.297 | 0.71 | 0.521 | 0.56 |
| 48 | 26.6 | 0.83 | 0.64 | 0.310 | 0.68 | 0.503 | 0.59 |

(updates 1-8 are the value warm-up: no policy gradient.) Sanity: no NaN (value/policy losses finite throughout, no skipped minibatch), games end
normally (mean 26 battle turns, 0.4% draws, 0 turn-cap draws), `pool.stats()` shows 0 illegal actions and 0 engine panics over
35k games; the Rust fuzz (256 games x 3000 steps random play) too.

**Honest strength check** of the final checkpoint against the supervised net, in the same pool (`ppo eval`, 600 games each, seats alternate):
sampled (T=1) both sides **0.827 (+271 Elo)**; **greedy (argmax) both sides 0.587 +- 0.020 (+61 +- 14 Elo)**. Most of the sampled-play gain is
entropy reduction (the policy got sharper: entropy 1.46 -> 0.68), the argmax gain is the real policy improvement and it is what token-policy
plays. Not yet measured: league games against gen7 search (CPU was busy with another SPRT); do that with the commands below. The
value head did not improve (explained variance flat at 0.56-0.63): the critic is the weak point of this smoke configuration.

## Commands

Benchmark: `python -m showdown_bot.ppo bench --init models\tok-d256.pt --games 64 128 256 512 --threads 4`

Head-to-head in the pool (stochastic or `--greedy`, seats alternate): `python -m showdown_bot.ppo eval --a runs\ppo1\latest.pt --b models\tok-d256.pt --games 1000`

### Recommended long run

The smoke run was 26 minutes at lr 2e-5. For a multi-hour run (about 110 updates per hour at the observed speed on a busy machine):

```
python -m showdown_bot.ppo train --init models\tok-d256.pt --out runs\ppo1 --minutes 600 ^
  --games 512 --threads 4 --rollout-steps 48 --epochs 2 --minibatch 512 --lr 2e-5 ^
  --ent-coef 0.01 --kl-beta0 0.1 --kl-beta1 0.02 --kl-decay-updates 400 --value-warmup 12 ^
  --snapshot-every 20 --max-past 6 --ckpt-every 20 --gpu-mem-gb 6
```

Why these values: `ent-coef 0.01` and a stronger, longer KL leash than the smoke run because the smoke run collapsed entropy
from 1.46 to 0.68 in 25 minutes (a sampled-play win rate of 0.83 but only +61 Elo with argmax); `value-warmup 12` because the critic
(explained variance ~0.6) limits GAE quality; `epochs 2`, clip 0.2, `target_kl 0.03` (early stop) were never triggered in the smoke run
(approx KL per update 0.003-0.008). Resume with `--resume` (same `--out`). Gate checkpoints with `ppo eval --greedy` against
`models\tok-d256.pt` before spending league time, and watch `wr_past`: if it stays near 0.5 while `wr_anchor` rises the net is
only getting sharper. A smaller net: add `--arch d192L4 --distill-updates 60` (distils from `--init` first, 2.70 M params).
Larger games (1024) buy nothing: the GPU forward dominates. `--zero-history true` trains with padding history rows (what the live bots feed).

## Evaluating a checkpoint against the gen7 search bot (league)

Do this only when the machine is idle enough (`--search-workers 18`). Candidate first, baseline second; `--teams swap` is implied.
The raw policy, then net-search with the same net (the interesting number; the supervised net scores -218 Elo raw and -82 Elo in
net-search against this baseline):

```
python -m showdown_bot.league --bots "cand=token-policy:model=runs\ppo1\latest.pt:device=cuda,best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev" --sprt 0,21 --search-workers 18 --game-timeout 3600 --max-games 2000 --teams swap --out data/league/sprt-ppo1-raw --seed 1

python -m showdown_bot.league --bots "cand=net-search:model=runs\ppo1\latest.pt:device=cuda:worlds=32:time_ms=1000:batch=256:c_puct=2:depth=4,best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev" --sprt 0,21 --search-workers 18 --game-timeout 3600 --max-games 2000 --concurrency 4 --teams swap --out data/league/sprt-ppo1-netsearch --seed 1
```

All three specs are validated by `league.parse_spec` in `tests/test_ppo.py::test_documented_evaluation_specs_parse`. The
cheap in-pool check (`ppo eval`, no server) is a good gate before spending league time: a checkpoint that is not clearly
above 50% against the supervised net in the pool will not beat gen7 search.

## Known gaps

* The pool plays *random battles only against itself*: no human-like or search-bot opponents, so exploitation of a policy that
  only the pool plays is possible (the opponent pool with past checkpoints and the anchor mitigates it).
* Live play still supplies padding history rows (docs/net-search.md); a PPO net that learned to use real history
  needs the live bots to track events (the other agent's work on `token_policy.py`/`net_search.py`); until then evaluate with
  history padded or expect a gap. Train with `--zero-history true` to match the live bots (padding rows).
* Reveal tracking is heuristic (see the table); the aux heads (opponent action, hidden sets) are not trained in PPO.
* Ditto excluded, Illusion/Imposter not modelled, team generator is the empirical one (see above).
* Policy is sampled at temperature 1 in training; live bots use argmax or search on top.
