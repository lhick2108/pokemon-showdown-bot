# Policy priors for MCTS selection (PUCT)

Engine `0.0.48+sdb.10`. Off by default; with no prior the search is bit-for-bit the old decoupled
UCB1 (`SelectionConfig::default()`).

## Why

Decoupled UCB1 gives every unvisited option `+inf`, so a node with 9-13 options per side first tries
all of them, for both sides, before it goes deeper: at 1 s per world the tree is ~4.5 plies deep.
A policy prior `P` lets selection spend its visits on plausible moves (AlphaZero PUCT):

    score_i = Q_i + c_puct * P_i * sqrt(N) / (1 + n_i)

per side, independently (decoupled, simultaneous moves). `Q_i` is the option's mean score from that
side's view; an unvisited option starts from the node's own mean (`- fpu_reduction`). The prior is mixed
with uniform (`prior_eps`, default 0.03) so no option is unreachable. The root's reported policy is
still the visit share, so `mixed` / `argmax` / `opp_model` in `players/search.py` are unchanged.
`ucb1`/`exp3`/`rm` code is untouched; a side without a prior keeps UCB1.

## The prior net (`engine/src/prior.rs`, `.ppol`)

Input = the value evaluator's v2 state features seen from the acting side (same sparse first layer,
`learned::hidden_acc`; side two is evaluated from its own perspective, so one net serves both sides).
Output = one logit per legal option. The v2 move features are *aggregated over the four moves*, so the
state alone cannot say which move slot is which; each option therefore also carries its own
`N_AFEAT = 58` **action features** (`ActionCtx::features`): damage fraction, KO, KO-and-faster,
effectiveness, STAB, accuracy, priority and the 30 move-property flags of that very move (Tera variants
set the Tera flag and Tera STAB), the matchup of a switch target (hp, status, resist / threat,
speed, damage both ways) and 3 context scalars.

    hr = relu(W1 x + b1)
    logit(option) = b2[slot] + W2[slot] . hr + v . relu(P hr + c + A af(option))

`slot` = 4 moves, 4 Tera moves, 5 switch targets in feature-slot order (`prior::action_slot`; the
Python helpers `policy_actions` / `policy_action_features` expose the engine's own mapping and features
so trainers cannot disagree with it). Hidden 16/32/64/128, embedding 16. Cost: one extra sparse
first-layer pass per side per *populated* node (a node is populated on its second visit), no
allocation. File format: see the module docs of `prior.rs`.

Python: `poke_engine.PolicyPrior.load(path)`, `.option_logits(state, side)`, `.priors(state, side)`;
`monte_carlo_tree_search(..., prior=, prior_opp=, c_puct=1.0, prior_eps=0.03, fpu_reduction=0.0,
root_prior_s1=, root_prior_s2=)`; the result has `depth_max` / `depth_mean` (plies from the root of the
node each iteration expanded). Priors need `selection="ucb1"` and `threads=1`.

## Opponent side

`prior_opp`: `self` (default; our net mirrored: the opponent is assumed to play like a search-distilled
player), `none` (opponent keeps UCB1), `human`, or a path to another `.ppol`.

`human` uses `models/human-policy.npz` **at the root only** (`root_prior_s2`, computed in the worker per
world with `HumanPolicy.predict(..., side=2)`, ~1 ms) and the mirrored net below the root. Reasons: the
human model is a numpy MLP over v2 + damage-calc features (too slow per node), the root prior is what
decides where the root visits (and therefore the opponent column of the joint matrix that
`opp_model` / Nash read) go, and below the root a cheap search-style prior is what keeps the tree deep.
Distilling the human model into the Rust format is possible with the same trainer if the root-only
version proves worthwhile in duels (its inputs would be human ladder records instead of visits).

## Plumbing

- `SearchConfig`: `prior` (path), `prior_opp`, `c_puct`, `prior_eps`, `prior_fpu`, `prior_human`;
  env `SHOWDOWN_SEARCH_PRIOR`, `SHOWDOWN_SEARCH_PRIOR_OPP`, `SHOWDOWN_SEARCH_C_PUCT`.
  `SearchPlayer` validates (file exists, ucb1, threads == 1).
- `players/_worker.py`: `search_world(..., prior_path=, prior_opp=, c_puct=, prior_eps=, prior_fpu=,
  human_path=, human_elo=)`; nets are cached per process (like evaluators). Only passed when a prior is set,
  so the default call shape is unchanged.
- League / selfplay specs (`search`, `search-net`): `prior=<.ppol>`, `prior_opp=self|none|human|<path>`,
  `c_puct=`, `prior_eps=`, `fpu=`, `prior_human=`. The latter keys need `prior=`.

## Training (`python -m showdown_bot.prior_train`)

Expert-iteration distillation of the self-play records (`docs/selfplay-data.md`): every searched world
of a decision is a sample (input: its state from side one; target: the world's root visit share over our
options), loss = cross-entropy over the legal options, train/val split by battle. Held-out reports: top-1
vs the visit argmax, cross-entropy vs the uniform baseline and a max-damage heuristic.

    python -m showdown_bot.prior_train --data data/selfplay/gen6 --data data/selfplay/gen5 \
        --hidden 32 --out models/prior-gen56-h32.ppol

`scripts/prior_bench.py` measures iterations/s, depth and agreement with a long reference search.

## Results (first net: `models/prior-gen56-h32.ppol`)

Trained on `data/selfplay/gen5` + `gen6` (29,981 records; 27,075 train / 2,906 held-out records split
by battle; 266k / 66k samples, hidden 32, ~1 min on the 3080). Held-out, one sample per searched world:

| | top-1 vs visit argmax | cross-entropy vs visit share |
| --- | ---: | ---: |
| uniform over legal options | 15.2 % | 2.004 |
| max-estimated-damage heuristic | 35.7 % | n/a |
| **prior net (hidden 32)** | **47.2 %** (train 50.1 %) | **1.653** |
| (floor: entropy of the targets) | | 1.160 |

Where the visit share is sharp (best option >= 60 %, 45 % of samples) top-1 is 56.8 %. gen6 alone
(266k samples) gave 44.5 %, hidden 64 did not beat hidden 32 (44.3 %): the data, not the capacity, is
the limit. A first version without the action features reached only 24.7 %.

`scripts/prior_bench.py`, 36 realistic states (turns 3-20 of gen6 games), one thread, 1000 ms,
`gen2-res-searchabs-mlp16.pev` scale 1.5, prior mirrored for the opponent, `prior_eps` 0.03:

| setting | iter/s | vs ucb1 | depth mean | depth max | root argmax = 6 s UCB1 reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| ucb1 | 233,000 | 1.00 | 4.7 | 8.7 | 0.89 |
| puct c=0.5 | 175,000 | 0.75 | 7.3 | 16.1 | 0.75 |
| puct c=1 | 176,000 | 0.75 | 6.7 | 14.6 | 0.78 |
| puct c=2 | 170,000 | 0.73 | 6.1 | 12.8 | 0.81 |
| puct c=4 | 164,000 | 0.73 | 5.4 | 11.2 | 0.78 |

The prior costs ~25 % of the iterations and buys ~1.5-3 plies of depth. The agreement column is not
evidence of strength: the reference is itself plain UCB1 (so UCB1 agrees with it by construction) and a
36-state sample has a standard error of ~0.07. Whether the deeper tree plays better is a duel question.
