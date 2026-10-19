# Transformer hidden-set heads vs the live set sampler (offline go/no-go)

Script: `scripts/hidden_head_vs_sampler.py`. Question: would `models/tok-d256.pt`'s `hid_*` heads predict an
opponent's unrevealed item / ability / Tera type / next move better than the current world sampler, so that we could
use them to reweight or resample the 16 MCTS worlds?

**Verdict: NO-GO.** The head does not beat the sampler by the 5-point top-1 bar on any attribute; it is level on
item and ability, worse on Tera and moves, and clearly worse on log-loss everywhere.

## Why the training log prints `hidden item acc nan`

The hidden heads are supervised **only on self-play** shards (`hid_ids` need the recorded truth state;
`docs/tokens.md`, "hidden sets" coverage table: human fp-aux = none). On the human sources `hid_valid` is 0 for
every row, the metric meters are empty (`n item 0`) and the mean prints `nan`. The nan sources are the human blocks
(and the aggregate "human" line); the self-play blocks report item ~0.86, ability ~0.96, tera ~0.60, move recall@4
~0.92. So the head is usable but was trained and validated only on self-play opponents (sets from the generator,
the bot's own teams), never on a human game. Human shards cannot provide a label either: their hidden sets are
hindsight-sampled (`docs/human-replays.md`), not real.

## Method

* Data: PokeChamp `gen9randombattle` `test`-split games (the split the token trainer holds out), every 25th test
  game, 463 games used (500 requested; skips: ability swap 20, transform 8, Illusion 9), both seats. All Elo buckets.
* Live-style replay: `human_replays.states_for_role` in first-person mode. An `OpponentModel` consumes the log event
  by event, so at each `|turn|` the `SetTracker`s know only what that seat had seen. Damage-roll inference is off
  (the viewer's real stats are not in a replay), so the sampler here is slightly weaker than the live bot's.
* Ground truth: an attribute that is unrevealed in the tracker at turn T and revealed (per tracker) at game end.
  Item / ability / tera: the revealed value. Move: the first move newly revealed after T (the "next-revealed move");
  top-1 for moves counts a hit if the top prediction is any move revealed later.
  36,945 (position, mon, attribute) examples, 1,229 more dropped because the mon's token slot could not be matched
  by species name (forme naming).
* Sampler distribution (a): the exact marginal of `SetTracker.sample` (count-weighted full sets consistent with all
  binding evidence and constraints; the role-based marginal when no full set matches). Belief reweighting is off
  by default, so it is off here.
* Head distribution (b): the token net on the live token view (state with the opponent scrubbed to revealed facts via
  `tokens.reveal_rows`, 16 real history rows rebuilt like the shard exporter, CPU). Item/ability: softmax over the
  vocabulary; Tera: 20-way softmax, "none" removed; moves: softmax with revealed moves masked and renormalised.
* Also scored: `head_in_support` (head renormalised over the sampler's support: what reweighting the sampler's own
  worlds by the head would give) and `product` (sampler x head, renormalised).
* Log-loss floors every probability at 1e-3 (nats), identically for all methods. Game-clustered bootstrap CIs.

## Results (463 games, 36,945 examples)

Top-1 accuracy

| attr | n | sampler | head | head in support | product |
| --- | ---: | ---: | ---: | ---: | ---: |
| item | 6,911 | **0.807** | 0.802 | 0.820 | 0.827 |
| ability | 1,356 | 0.982 | 0.980 | 0.982 | 0.982 |
| tera | 3,564 | **0.616** | 0.569 | 0.573 | 0.577 |
| next move | 25,114 | **0.569** | 0.525 | 0.530 | 0.555 |
| item+ability+tera | 11,831 | 0.770 | 0.752 | 0.764 | 0.769 |

Log-loss (nats, lower is better)

| attr | sampler | head | head in support | product |
| --- | ---: | ---: | ---: | ---: |
| item | **0.371** | 0.737 | 0.698 | 0.690 |
| ability | **0.035** | 0.072 | 0.042 | 0.043 |
| tera | **0.911** | 1.752 | 1.749 | 1.742 |
| next move | **1.358** | 1.499 | 1.490 | 1.526 |

Differences in top-1 vs sampler (95% CI, clustered by game): head item -0.005 [-0.033, +0.023]; ability -0.002;
tera -0.047 [-0.113, +0.016]; move -0.043 [-0.061, -0.023]. Best case (product, item): +0.020 [-0.007, +0.046],
far below +5 points.

Impossible worlds (sampler support excludes the truth): item 0.9%, ability 0.0%, tera 4.2%, move 1.3%. The
role-based fallback is used for 1-1.6% of examples. The head puts < 1e-3 on the truth 4% (item), 10% (tera) of the time.

By sampler uncertainty (number of consistent full sets), item top-1 sampler / head: 1 set 1.00/0.995; 2-4 sets
0.83/0.85; 5-20 sets 0.78/0.75; 21+ sets 0.59/0.59. Tera: 5-20 sets 0.50/0.37, 21+ sets 0.36/0.48 (small n=231).

Most of the head's information is already in the sampler's evidence (reveals, speed, negative evidence); the
extra signal in the history rows (hp deltas) does not add enough to matter. A 6-game smoke run (n=359) showed the
head +15 points on item; that was noise, and is the reason to read only the 463-game result.

## Caveats

* Ground truth is "revealed later in the log", a biased subset (items that get consumed or triggered, Tera types
  that are actually used). It applies equally to both methods.
* Sampler has no damage-roll inference here (it has in the live bot), so it is if anything understated.
* The head was trained only on self-play sets; the human metagame/play differs (the item head is also not
  restricted to the species' legal items, which is why `head_in_support` helps a little).
* Only the max-HP stat of the token view comes from the sampled set; in self-play training it came from the true
  set (a slight leak the head could use), so in this evaluation and live the head is a little handicapped. I checked
  that no other opponent token changed between sampling seeds.
* Tera head: the final test accuracy (~0.60) on self-play includes easy cases; on revealed-later positions it
  loses to the sampler marginal, with overconfident misses.
* History rows are rebuilt from per-turn snapshots exactly as in the exporter; the live bot's history would differ slightly.
* Positions are correlated within a game (hence game-clustered CIs); same-mon repeats at successive turns are all counted.

## If it were a go (not pursued): sketch

World sampling lives in `players/search.py` (`SearchPlayer`, trackers via `OpponentModel.update`, worlds from
`es.sample_opponent_team`); `_token_root_prior` already runs one batched token forward per decision (about 7-9 ms
on GPU, `docs/token-net.md`). Reweighting would be one extra forward on the scrubbed state (1 state, not 16),
then per revealed mon multiply each candidate full set's weight by head(item) x head(ability) x head(tera) (the
`head_in_support`/`product` variant) via `SetTracker.update_belief`-style factors before `sample(u=...)`. Given the
results above this would not be worth the latency.
