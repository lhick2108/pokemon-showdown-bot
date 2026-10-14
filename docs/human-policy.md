# Human action-policy model and exploiting predictable humans

The search treats the opponent as a worst-case adversary. Real ladder opponents are not: they spam
the obvious attack, rarely switch, and almost never find the refutation the search is afraid of.
This feature learns how humans choose and lets the bot take the risks a human will not punish.

## Pieces

| file | what |
| --- | --- |
| `src/showdown_bot/human_policy.py` | action slots, featuriser, numpy `HumanPolicy` (inference) |
| `src/showdown_bot/human_policy_train.py`, `scripts/train_human_policy.py` | streaming PyTorch trainer, evaluation, baseline report |
| `src/showdown_bot/human_exploit.py` | `safe_best_response`, restricted-Nash candidates, per-game `ClassAdapter` |
| `src/showdown_bot/players/search.py` | `opp_model=` integration (`_exploit_policy`) |
| `src/showdown_bot/players/human.py` | `HumanPolicyPlayer`: league bot type `human`, plays by sampling the model, no search |

## The model

Training rows are the policy rows of `docs/human-replays.md` (`data/human/pc1/policy`, 332k games).
`state` is the actor's view with the actor as engine side one (hindsight-completed: both teams
full, exact HP). The action space is 13 slots relative to the actor: the active Pokemon's 4 moves,
the same 4 with Tera, and the 5 bench Pokemon (list positions 1-5, the order the engine features
use). Input: the engine's v2 state features (2191), 20 damage-calculator features (average damage
of each own move as a fraction of the target's HP, "guaranteed KO", damage relative to the best
move, and the same two numbers for the opposing moves), the 13-bit legal mask and an elo-bucket
embedding (bucket 5 = unknown, used with 15% dropout so the model also works without a rating).
Output: logits for the 13 slots, softmax over the legal ones. MLP 2211 -> 1024 -> 512 -> 13
(2.8M parameters); the numpy forward pass costs about 0.6 ms per state (small batches are run row
by row on purpose, a multithreaded GEMM stalls when search workers occupy the cores).

```
python scripts/train_human_policy.py --data data/human/pc1/policy --out models/human-policy.npz \
    --epochs 1.5 --workers 8          # streams the shards; ~1 GB of RAM for 8 loader workers
python scripts/train_human_policy.py --report models/human-policy.npz   # baseline table
```

## Using it in the search

Off by default. Spec keys (league / selfplay / `SearchConfig` fields):

| key | field | meaning |
| --- | --- | --- |
| `opp_model=<path.npz>` | `opp_model` | enable; needs mode mixed or argmax, `threads=1` |
| `opp_elo=<bucket or rating>` | `opp_elo` | condition the model on an opponent elo (default unknown) |
| `exploit_delta` | `exploit_delta` | max loss of worst-case win probability vs the base policy (default 0.05) |
| `exploit_weight` | `exploit_weight` | max fraction of the way from the base policy to the best response (default 1) |
| `exploit_kappa` | `exploit_kappa` | prior strength of the in-game class adaptation, 0 = off (default 4) |
| `exploit_gain` | `exploit_gain` | minimum expected gain under the belief to deviate at all (default 0.005) |

Environment equivalents: `SHOWDOWN_SEARCH_OPP_MODEL`, `..._OPP_ELO`, `..._EXPLOIT_DELTA`, `..._WEIGHT`,
`..._KAPPA`, `..._GAIN`.

Per decision, for every searched world:

1. The world's state string is swapped (`human_policy.swap_sides`) so the opponent is the actor,
   giving exactly the input the model was trained on, and the model predicts P(reply) over the
   opponent's engine root columns (`result[4]`, the joint matrix columns).
2. The belief is the model's distribution mixed with 2% uniform, rescaled class-wise by the
   `ClassAdapter` (attack / priority / status / protect / switch): the adapter compares what this
   opponent actually did earlier in the game, read from the `TendencyTracker`, with what the model
   predicted for those turns and shifts the belief toward the opponent's own habits (Dirichlet
   posterior over the mean predicted class mix, `exploit_kappa` pseudo-turns).
3. The base strategy is the world's own visit share, i.e. exactly what mixed/argmax would use.
   `safe_best_response` walks from the base toward each candidate (every pure row, plus
   restricted-Nash responses at three trust levels) as far as
   `min_j (x A)_j >= min_j (base A)_j - exploit_delta` allows, and keeps the candidate with the
   best expected payoff under the belief (the base unchanged if the gain is below `exploit_gain`).
   A wrong belief therefore costs at most `exploit_delta` of win probability against any reply.
4. The per-world strategies are averaged like `aggregate_policy` and the usual selection
   (argmax / mixed) runs on the result.

Diagnostics: `SearchStats` prints `exploit=<decisions> shifted=<lambda>0> changed=<top choice
changed> exploit_ms`; with a recorder each decision record carries an `exploit` block (turn, top
belief entries, class belief and multipliers, lambda / target / gain / security per world,
`base_top` vs `top`).

## Test opponent

League bot type `human` (`HumanPolicyPlayer`): samples one world of the opposing team, asks the
model for P(action) over its legal actions and samples from it. Keys: `model=<npz>`,
`pick=sample|argmax`, `temp` (below 1 sharpens: a more predictable human), `elo`.

## Accuracy (test split, 1.53M decisions with >= 2 legal actions)

Trained for 1.5 epochs (17M decisions) of the train split, about 28 minutes with 8 loader workers.
Metrics are on the 143 held-out test shards; the model sees the elo bucket (the "unknown elo" row
is the same model with the unknown embedding, which is how the live bot uses it unless `opp_elo`
is given).

| elo bucket | decisions | top-1 | top-3 | NLL | uniform top-1 / NLL | slot prior top-1 / NLL |
| --- | --- | --- | --- | --- | --- | --- |
| all | 1,530,263 | 0.504 | 0.850 | 1.337 | 0.118 / 2.219 | 0.251 / 1.824 |
| 1000-1199 | 565,769 | 0.489 | 0.842 | 1.375 | 0.120 / 2.203 | 0.262 / 1.774 |
| 1200-1399 | 276,592 | 0.518 | 0.862 | 1.294 | 0.120 / 2.211 | 0.254 / 1.803 |
| 1400-1599 | 277,970 | 0.520 | 0.862 | 1.296 | 0.117 / 2.226 | 0.248 / 1.839 |
| 1600-1799 | 179,731 | 0.512 | 0.854 | 1.318 | 0.117 / 2.234 | 0.242 / 1.873 |
| 1800+ | 230,201 | 0.497 | 0.841 | 1.362 | 0.114 / 2.249 | 0.233 / 1.918 |
| all, elo unknown | 1,530,263 | 0.503 | 0.849 | 1.344 | | |

On a 40k-decision sample ("max damage" = the legal non-Tera move with the largest average damage
to the target's current HP, never a switch):

| decisions | n | model top-1 | max-damage top-1 | uniform top-1 | model NLL |
| --- | --- | --- | --- | --- | --- |
| all | 39,897 | 0.509 | 0.369 | 0.118 | 1.334 |
| human played a move | 34,006 | 0.557 | 0.432 | 0.119 | 1.188 |
| human switched | 5,891 | 0.228 | 0.000 | 0.114 | 2.174 |

The damage-calculator inputs matter: a first version without them (same trunk, 0.2 epoch) scored
0.28-0.30 top-1, below the max-damage heuristic.

Live-path check (`opponent_view_report`): for each (game, turn) seat B's real action is predicted
from seat A's row with the state swapped and side two as the actor, which is exactly what
`SearchPlayer` feeds the model. Top-1 0.486 / NLL 1.387 on 23.6k covered decisions versus 0.486 /
1.386 from B's own row; 99.6% of real actions are expressible in the other seat's picture of B's
team (a move a sampled team lacks accounts for the rest). Caveat: both seats' rows share the same
hindsight-sampled picture of each team, so this checks the swap/slot plumbing, not the effect of a
different sampled opponent set.

Cost: encode 0.65 ms + forward 0.41 ms per state, 4.3 ms for the four worlds of a decision; with
the matrix solves the whole exploit step measured about 8 ms per decision in a smoke league run.
