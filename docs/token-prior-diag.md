# Why is the token net a weak search root prior? Offline diagnosis

`scripts/diag_tok_prior.py` (single process, GPU for the net). Data: 2,920 held-out decisions (gen7 1000, gen8 1000, gen9 920;
p1 1460 / p2 1460) from `data/selfplay`, battles in the trainer's **val split** (`tokens.split_of(battle_hash(tag), 50)`),
at most 5 random decisions per recording, >= 2 legal actions, valid `pol_target`. The target is the decision's aggregated
search visit share (mean over the 4 worlds, 1 s each, search run with the `.ppol` at c_puct 2) on the 13 action slots. Its mean
entropy is 0.90, mean max-prob 0.66, and 31% of decisions have a visit argmax with > 0.8 of the visits. The argmax is a move
for 68% of decisions, tera for 3%, switch for 29%.

Metrics: top-1 vs the visit argmax; CE and KL of the visit share under the prior **after the engine's 3% uniform mix**
(`prior_eps`); entropy / max-prob of the raw prior; fraction of decisions with top prob > 0.8; top-1 split by the kind of the
search argmax (n = 1987 move, 90 tera, 843 switch). Engine wheel whl23 unless stated.

## Variants

| | variant |
| --- | --- |
| a | `prior-gen56-h32.ppol` via `PolicyPrior.priors` on every recorded world, mean (the evaluation the engine does at the root) |
| b | token net, LIVE path: `TokenPolicy.predict_states` over the recorded worlds, opponent scrubbed with the recorded `opp_facts`, history padding, mean over worlds |
| bt | live tokenisation on the recorded truth state (padding history) |
| c | TRAINING export (`tokens.game_samples`, truth state, real 16-row event history) |
| d | (c) with the history rows zeroed |

## Results (whl23, n = 2920)

| | top-1 | top-1 move | top-1 tera | top-1 switch | CE | KL | entropy | max-prob | top > 0.8 | mass on search argmax | best T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| target (visits) | | | | | | | 0.90 | 0.66 | 31.5% | | |
| a .ppol | 0.525 | 0.637 | 0.200 | 0.298 | 1.522 | 0.621 | 1.65 | 0.378 | 1.7% | 0.303 | 0.7 |
| b live | 0.579 | 0.665 | 0.333 | 0.403 | 1.415 | 0.514 | 1.51 | 0.449 | 3.7% | 0.362 | 1.0 |
| bt live, truth | 0.581 | 0.666 | 0.344 | 0.405 | 1.412 | 0.511 | 1.50 | 0.454 | 4.2% | 0.366 | 1.0 |
| c train export | 0.597 | 0.686 | 0.322 | 0.416 | 1.371 | 0.470 | 1.41 | 0.486 | 6.0% | 0.395 | 1.0 |
| d export, no history | 0.581 | 0.666 | 0.344 | 0.405 | 1.412 | 0.511 | 1.50 | 0.454 | 4.2% | 0.366 | 1.0 |

Per generation top-1 (a / b / c): gen7 0.521 / 0.578 / 0.584, gen8 0.517 / 0.583 / 0.596, gen9 0.539 / 0.577 / 0.613.
Per role (a / b / c): p1 0.531 / 0.580 / 0.594, p2 0.520 / 0.579 / 0.601.

Reading:

* **The net is a better prior than the `.ppol` on every offline measure** (top-1 +5.4 points, CE -0.11 nats, higher mass on the
  search argmax), including on gen7-9 where neither was trained on these battles' positions as targets of its own search. It is not
  sharper in the harmful sense: its mean max-prob is 0.45 (the `.ppol` 0.38; search visits 0.66) and only 3.7% of decisions
  exceed 0.8, so it cannot dominate the search through over-confidence. Its best CE temperature is 1.0 (calibrated); the
  `.ppol` would like T = 0.7 (too flat).
* **History mismatch**: removing the 16 history rows costs top-1 0.597 -> 0.581 (-1.6 points) and CE +0.04. Real but small. (The
  `59.6%` figure of the trainer is variant c.) The trainer's history is also not an exact event log (reconstructed from
  per-decision snapshots), so live could not match it precisely even if it tracked events.
* Live tokenisation on the worlds (b) vs on the truth state (bt) is the same to within noise: opponent scrubbing makes worlds
  nearly identical (mean TV distance of one world's policy to the world mean: 0.0009). Averaging over worlds is harmless.
* d == bt up to float16 batching noise (max abs diff 1.3e-3): the live tokeniser and the exporter agree.
* Switch decisions are the weak spot of both priors (net 0.40, `.ppol` 0.30); tera is rarely the search argmax (3%) and rarely
  predicted by either.

## (e) Engine-version token drift (whl21 export wheel vs whl23 live wheel)

Same decisions, live tokenisation (world 0) and truth state: **1.95% of rows have any differing token field**, `mon_cont` 1.88%,
`act_cont` 0.41%, `act_legal` 0.07%; all categorical fields, `mon_known`, `match_pair`, `side_cont`, `glob`, `meta` identical. Effect on the
live policy: mean TV 2.3e-4, argmax changed in 0.10% of decisions. Top-1 of b under whl21 = 0.580 vs whl23 0.579. Negligible.

## Live-code checks

* **Mapping.** `tokens.Vocab`/`slot_names` and `root_prior_s1` use the same `movechoice_to_string(root_options)` as
  `policy_actions`. Over all 2,920 decisions: 0 prior keys that are not engine root options, 0 root options without a prior key,
  0 keys not among the 13 slot names, 0 unnormalised dicts, and 0 recorded search-policy keys that do not map to a slot (moves in state
  order, `-tera` variants, `switch <species>` in bench order are consistent). End to end: a one-hot `root_prior_s1` on a random root
  option (c_puct 30, eps 0) made that option the most visited in 30/30 real searches.
* **Perspective.** Every recorded state has us as side one (`side` is `one` in all records, p1 and p2 alike), so the live path
  tokenises side 1 for both seats. Top-1 is identical for p1 and p2 games (0.580 / 0.579), so a p2 perspective error is ruled out.
* **Averaging over worlds** is by construction the mean of near-identical vectors (see above).
* **Config note** (`docs/token-net.md`): `tok_prior` with `prior=` set keeps the `.ppol` for interior nodes and the opponent; without
  `prior=` the interior and the opponent are plain UCB1. The latter is a different, weaker search, not a prior change.

## Does a better-looking root prior change the search?

Real searches, 200 decisions, one world, 1 thread, 300 ms (about 115k iterations), live evaluator (`gen7-res-root-mlp16.pev`,
scale 1.5), c_puct 2, eps 0.03. Reference = the recorded 4-world 1 s visit argmax (produced WITH the `.ppol`, so biased in its
favour).

| root prior | interior | top-1 vs reference | mass on reference argmax | search entropy | argmax agreement with `.ppol`-root search |
| --- | --- | ---: | ---: | ---: | ---: |
| `.ppol` (live baseline) | `.ppol` | 0.730 | 0.591 | 0.78 | 1.00 |
| token net | `.ppol` | 0.725 | 0.604 | 0.76 | 0.905 |
| token net, T = 0.7 | `.ppol` | 0.730 | 0.613 | 0.71 | 0.90 |
| token net | UCB1 | 0.700 | 0.580 | 0.75 | 0.715 |
| uniform | `.ppol` | 0.715 | 0.572 | 0.84 | 0.88 |
| none (plain UCB1) | none | 0.700 | 0.515 | 1.10 | 0.74 |

The root prior is a small perturbation of a 100k-iteration search: replacing the `.ppol` by *uniform* at the root moves the
decision's argmax in 12% of decisions and the agreement with the reference by 1.5 points (n = 200, standard error about 3 points). The
token root prior and the `.ppol` root prior give indistinguishable search quality here (0.725 vs 0.730); the token net's
extra offline accuracy (+5 points of top-1 on the prior itself) does not survive the search.

## Live evidence in context

`data/league/sprt-tokprior.log` at 393 games: 186-207, Elo -19.5 [-51.0, +11.7], LLR -2.49 vs bounds +-2.94 (H0 = 0, H1 = +21).
That interval contains 0, and the test was designed to detect +21; it shows the net adds nothing, not that it is worse. The raw
`token-policy` bot (no search) at about -218 Elo against search is what a 58-60% top-1 policy should score against a 100k-iteration
search whose own argmax it matches 58% of the time; it says nothing about the prior.

## Conclusion

| hypothesis | verdict | evidence |
| --- | --- | --- |
| history mismatch | real but small | -1.6 points top-1, +0.04 nats CE (0.597 -> 0.581); net still beats the `.ppol` by 5 points without history |
| sharpness vs c_puct | no | max-prob 0.45 (`.ppol` 0.38, visits 0.66), 3.7% of decisions above 0.8, best temperature 1.0 (calibrated); sharpening to T 0.7 changes nothing in search |
| engine-version token drift (whl21 vs whl23) | no | 1.95% of rows differ in `mon_cont`/`act_cont`/`act_legal`; policy TV 2e-4, argmax changed 0.1% |
| mapping / perspective / world-averaging bug | no bug found | 0 unmatched keys in 2,920 decisions, 30/30 end-to-end steering, p1 = p2 top-1 |

The offline premise does not hold: the net is a **better** distribution than the `.ppol` against the search visits, yet as a root
prior it cannot help because the search already moves far away from any root prior within 100k iterations; the measured
difference between `.ppol`, token and even uniform root priors is of the order of the SPRT noise (the observed -20 Elo has a
confidence interval of [-51, +12]). Part of what is seen could also be pure self-consistency: the training targets are the visits of a
search that used the `.ppol`, and the evaluation reference is biased the same way. No definite code bug, so no fix commit.

Recommended next step: stop trying to extract Elo from the root prior (an upper bound of a few Elo at 1 s / 4 worlds). Use the net where
the search cannot reproduce its information: (1) as the interior-node prior and the opponent-side mirror (needs the net inside
the engine or a batched leaf service; the `.ppol` there is the weaker 0.525 prior), or (2) at much smaller budgets (`fast_time_ms`
searches, where the prior matters: uniform vs `.ppol` root differs more at 50-100 ms), or (3) use its value / opponent heads for
belief weighting of worlds. If an SPRT of the root prior is still wanted, it needs about 4x more games to resolve +-10 Elo.
