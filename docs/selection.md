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

## Root Nash move selection (`mode="nash"`)

Exp3/RM change how the tree is searched; Nash selection changes only the final
move. The engine can return the root's joint-action matrix and the bot plays the
equilibrium of that matrix game instead of the visit share (the 75% rule).

**Engine.** Every node now also accumulates `total_score` (side one's view, in
[0, 1]). With `monte_carlo_tree_search(..., return_joint=True)` the result carries
`joint = [(i, j, visits, mean), ...]` for each expanded root cell: `i`/`j` index
`side_one`/`side_two`, `visits` sums the cell's chance-outcome children and `mean`
is their total score over `visits`. Works with ucb1, exp3 and rm; single-threaded
only (`threads > 1` raises). Off by default, so the normal path pays nothing.

**Per world.** The opponent's option set differs between sampled worlds (different
sampled movesets), so matrices are never averaged across worlds. For each world:

1. Build the K1 x K2 payoff matrix (`nash.shrunk_matrix`). Each cell is shrunk
   toward its row prior with a pseudo-count c (default 5):
   `A_ij = (n_ij * m_ij + c * prior_i) / (n_ij + c)`, `prior_i` = the row's
   visit-weighted mean (the root mean for a row never visited, 0.5 if nothing was).
   MCTS deliberately leaves bad joint actions under-visited, so raw cell means rest
   on a few rollouts and unvisited cells have no value at all; unshrunk, that noise
   (a lucky 1-visit 1.0, a zero-filled hole) produces fake equilibria. A cell with
   few visits now says "about as good as the rest of its row".
2. Solve the zero-sum game (`nash.solve_matrix_game`): exact simplex on
   `max sum(u) s.t. B u <= 1` with `B = A - min(A) + 1`, Bland's rule so degenerate
   games terminate; regret matching+ (alternating, linear averaging) is only a
   fallback if the simplex fails. 13 x 13 takes about 1.5 ms. The result has the
   equilibrium strategies, the game value and the exploitability.
3. Keep our row strategy over our engine choices.

**Across worlds.** Our strategies are averaged by choice string, each weighted by
the world's total root visits. Actions below `nash_floor` (default 0.03) are
dropped and the rest renormalised (the best is always kept), then the move is
sampled from the result (illegal choices are skipped first, as for mixed). If no
world carries a joint matrix the player falls back to the mixed rule.

**Usage.**

* Python: `SearchConfig.mode="nash"`, `nash_pseudocount`, `nash_floor`; constructor
  `mode=` / `nash_pseudocount=` / `nash_floor=`; env `SHOWDOWN_SEARCH_MODE=nash`,
  `SHOWDOWN_SEARCH_NASH_PSEUDOCOUNT`, `SHOWDOWN_SEARCH_NASH_FLOOR`. Needs
  `threads == 1`.
* Duel: `scripts/duel.py --net-mode nash --stock-mode mixed` with no `--evaluator`
  duels two stock searches. With `--evaluator`, `--net-selection/--stock-selection`
  still apply, so any combination of tree policy and final rule can be compared.
* League: `label=search:mode=nash`.
* Recordings: each world record gets `nash = {strategy, value, exploitability}`
  (the game value is a candidate value target) and each decision `nash` = the
  aggregated strategy; `policy` is the strategy that was sampled from.

## Notes on the matrix

* The cells are estimated under the tree policy that generated them, so with ucb1
  the weakest joint actions are the least reliable; that is what the pseudo-count
  protects against. exp3/rm sample every cell at least `gamma/K`-fold and give a
  better-covered matrix, at the cost of slower iterations.
* The game value of the root matrix is not the search's root mean (which is the
  value of the search's own mixed play); the two differ when the search is
  exploitable.

## References

* Lisy, Kovarik, Lanctot, Bosansky (2013). *Convergence of Monte Carlo Tree
  Search in Simultaneous Move Games.* NeurIPS.
* Tak, Lanctot, Winands (2014). *Monte Carlo Tree Search Variants for
  Simultaneous Move Games.* IEEE CIG.
* O'Donoghue, Lattimore, Osband (2020). *Matrix games with bandit feedback.* UAI.
* Oak (lab-oak/oak) solves the root matrix game; GPL, ideas only, no code used.
