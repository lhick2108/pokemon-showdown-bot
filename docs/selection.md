# Simultaneous-move selection in engine MCTS (Exp3 / regret matching)

`engine/src/mcts.rs` chooses each side's move at a node independently with
UCB1. In simultaneous-move games that "decoupled UCB1" is known to converge to
exploitable strategies. A runtime mode selects the tree policy:

| mode   | policy at a node (per side) |
|--------|-----------------------------|
| `ucb1` | default, decoupled UCB1, unchanged |
| `exp3` | sigma_i = (1-g) * softmax(eta * G)_i + g/K, eta = g/K; on reward r for the chosen i: G_i += r / sigma_i |
| `rm`   | sigma_i = (1-g) * R_i^+ / sum R^+ + g/K (uniform if sum R^+ = 0); on reward r for chosen a: xhat_a = r / sigma_a, R_j += xhat_j - r |

K is the side's option count at that node, rewards are in [0, 1] from that
side's view (side two uses `1 - r`), the action is *sampled* from sigma, and
the average strategy S_i += sigma_i is accumulated per visit. The softmax uses
max-subtraction; G, R and S are `f64`. Per-node stats are allocated lazily and
only when the mode needs them (`Node.s1_stats/s2_stats` are `None` under
ucb1), so the ucb1 path only pays one enum compare per node.

## Final move

For exp3/rm the Nash-converging quantity is the normalised average strategy of
the root, not the visit counts. The engine returns it as `avg_strategy` on each
`MctsSideResult` (for ucb1 it equals the visit share). `visits` and
`total_score` are unchanged. `SearchPlayer` then averages `avg_strategy` over
the sampled worlds (instead of visit share) and applies the usual mixed/argmax
rule; `SearchStats`/recordings also store the per-move `strategy`.

## Usage

* Engine: `monte_carlo_tree_search(state, ..., selection="exp3", gamma=0.1)`.
* Python player: `SearchConfig.selection` / `selection_gamma`, constructor args
  `selection=` / `selection_gamma=`, env `SHOWDOWN_SEARCH_SELECTION`
  (`ucb1|exp3|rm`) and `SHOWDOWN_SEARCH_SELECTION_GAMMA` (default 0.1).
* Duel: `scripts/duel.py --evaluator m.pev --stock-uses-evaluator --net-selection rm --stock-selection ucb1`
  (same evaluator on both sides, only the selection differs). `--selection-gamma` sets g.
* League: `label=search:selection=rm:gamma=0.1` (also on `search-net`).

## Threads

The shared-tree (threaded) search, `mcts_threaded.rs`, is ucb1 only. Asking for
exp3/rm with `threads > 1` raises `ValueError` (Python engine call,
`SearchPlayer` constructor, `duel.py`); its result reports the visit share as
`avg_strategy`. Use one engine thread per world (the default).

## Notes

* gamma defaults to 0.1 for both; Tak et al. tuned it per game, so sweep it.
* Importance-weighted updates are high-variance with a small gamma/K floor
  (an update is up to r*K/gamma); at short horizons (a few thousand iterations)
  RM in particular can commit to one option early.
* exp3/rm are slower per iteration than ucb1 (a softmax / regret pass and an
  allocation per expanded node).

## References

* Lisy, Kovarik, Lanctot, Bosansky (2013). *Convergence of Monte Carlo Tree
  Search in Simultaneous Move Games.* NeurIPS.
* Tak, Lanctot, Winands (2014). *Monte Carlo Tree Search Variants for
  Simultaneous Move Games.* IEEE CIG.
