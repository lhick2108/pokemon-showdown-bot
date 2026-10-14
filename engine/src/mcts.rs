use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::learned::{Evaluator, LeafEval};
use crate::prior::{fill_priors, PriorConfig};
use crate::selection::{SelectionConfig, NodeStats, SelectionMode, SideStats};
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u32,
    // sum of rollout scores (side one's view, [0, 1]) over every visit of this node;
    // for a chance child of the root this is the joint-action cell's score
    pub total_score: f32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,

    // Exp3 / regret-matching state; always None under ucb1
    pub stats: Option<Box<NodeStats>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            total_score: 0.0,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            stats: None,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: -1.0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: -1.0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode]) -> usize {
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    /// PUCT for one side: `Q + c * P * sqrt(N) / (1 + n)` with `Q` the option's mean score
    /// (side's view) and, for an unvisited option, the node's own mean minus
    /// `fpu_reduction`. `is_side_one` selects which view the node's mean is read from.
    /// Falls back to UCB1 for a side without priors (`prior < 0`).
    pub fn maximize_puct_for_side(&self, side_map: &[MoveNode], pc: &PriorConfig, is_side_one: bool) -> usize {
        if side_map[0].prior < 0.0 {
            return self.maximize_ucb_for_side(side_map);
        }
        let n = self.times_visited as f32;
        let sqrt_n = n.max(1.0).sqrt();
        let mean = if self.times_visited > 0 {
            let m = self.total_score / n;
            if is_side_one { m } else { 1.0 - m }
        } else {
            0.5
        };
        let fpu = mean - pc.fpu_reduction;
        let mut choice = 0;
        let mut best = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let q = if node.visits == 0 { fpu } else { node.total_score / node.visits as f32 };
            let score = q + pc.c_puct * node.prior * sqrt_n / (1.0 + node.visits as f32);
            if score > best {
                best = score;
                choice = index;
            }
        }
        choice
    }

    /// Walks from `self` down the tree to the first node whose chosen joint action has no
    /// child yet. Iterative: in a stalling position the tree is a chain tens of thousands of
    /// nodes deep, and recursing there overflowed the stack (bead sdb.8).
    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
        sel: &SelectionConfig,
        leaf_eval: &mut LeafEval,
    ) -> (*mut Node, usize, usize) {
        let mut node: *mut Node = self;
        let mut depth = 0u32;
        loop {
            if (*node).s1_options.is_none() {
                let (s1_options, s2_options) = state.get_all_options();
                (*node).populate(s1_options, s2_options);
                if let Some(pc) = sel.prior.as_ref() {
                    fill_priors(
                        pc,
                        state,
                        (*node).s1_options.as_mut().unwrap(),
                        (*node).s2_options.as_mut().unwrap(),
                        false,
                        &mut leaf_eval.prior_scratch,
                    );
                }
            }

            let (s1_mc_index, s2_mc_index) = if let (SelectionMode::Ucb1, Some(pc)) = (sel.mode, sel.prior.as_ref()) {
                (
                    (*node).maximize_puct_for_side((*node).s1_options.as_ref().unwrap(), pc, true),
                    (*node).maximize_puct_for_side((*node).s2_options.as_ref().unwrap(), pc, false),
                )
            } else if sel.mode == SelectionMode::Ucb1 {
                (
                    (*node).maximize_ucb_for_side((*node).s1_options.as_ref().unwrap()),
                    (*node).maximize_ucb_for_side((*node).s2_options.as_ref().unwrap()),
                )
            } else {
                if (*node).stats.is_none() {
                    let k1 = (*node).s1_options.as_ref().unwrap().len();
                    let k2 = (*node).s2_options.as_ref().unwrap().len();
                    (*node).stats = Some(Box::new(NodeStats::new(sel, k1, k2)));
                }
                let st = (*node).stats.as_mut().unwrap();
                (st.s1.select(rng), st.s2.select(rng))
            };
            let key = (node as usize, s1_mc_index, s2_mc_index);
            match children.get_mut(&key) {
                Some(child_vector) => {
                    let child_vec_ptr = child_vector as *mut Box<[Node]>;
                    let chosen_child = (*node).sample_node(child_vec_ptr, rng);
                    state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                    node = chosen_child;
                    depth += 1;
                }
                None => {
                    // the node this iteration expands sits `depth + 1` plies below the root
                    leaf_eval.note_depth(depth + 1);
                    return (node, s1_mc_index, s2_mc_index);
                }
            }
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>, rng: &mut impl Rng) -> *mut Node {
        let nodes = &mut **move_vector;

        let total_weight: f32 = nodes
            .iter()
            .map(|n| n.instructions.percentage.max(0.0))
            .sum();

        let mut threshold = rng.random_range(0.0..total_weight);

        for node in nodes.iter_mut() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node as *mut Node;
            }
        }

        // fallback: return last node (handles float rounding issues that can come up)
        &mut nodes[nodes.len() - 1] as *mut Node
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on.
        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized, which ensures the node
        // addresses are stable for the children map keys
        let mut boxed = this_pair_vec.into_boxed_slice();
        let new_node_ptr = self.sample_node(&mut boxed, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);

        let key = (self as *mut Node as usize, s1_move_index, s2_move_index);
        children.insert(key, boxed);
        new_node_ptr
    }

    /// Adds `score` to this node and every ancestor, undoing the instructions on the way up.
    /// Iterative (see `selection`).
    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        let mut node: *mut Node = self;
        loop {
            (*node).times_visited += 1;
            (*node).total_score += score;
            if (*node).root {
                return;
            }
            let parent = (*node).parent;
            let s1_choice = (*node).s1_choice as usize;
            let s2_choice = (*node).s2_choice as usize;

            let parent_s1_movenode = &mut (*parent).s1_options.as_mut().unwrap()[s1_choice];
            parent_s1_movenode.total_score += score;
            parent_s1_movenode.visits += 1;
            if let Some(stats) = (*parent).stats.as_mut() {
                stats.s1.update(s1_choice, score);
            }

            let parent_s2_movenode = &mut (*parent).s2_options.as_mut().unwrap()[s2_choice];
            parent_s2_movenode.total_score += 1.0 - score;
            parent_s2_movenode.visits += 1;
            if let Some(stats) = (*parent).stats.as_mut() {
                stats.s2.update(s2_choice, 1.0 - score);
            }

            state.reverse_instructions(&(*node).instructions.instruction_list);
            node = parent;
        }
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32, leaf_eval: &mut LeafEval) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = leaf_eval.value(state);
            leaf_eval.score(eval, *root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    /// policy prior of this option (PUCT); -1 when the side has no prior
    pub prior: f32,
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    /// share of the root's average strategy (Exp3 / RM); the visit share under ucb1
    pub avg_strategy: f32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

/// One cell of the root's joint-action matrix: option indices into `MctsResult::s1` /
/// `s2`, the number of iterations that went through the cell (summed over its chance
/// outcomes) and the mean rollout score there, from side one's view in [0, 1].
#[derive(Clone, Debug)]
pub struct JointCell {
    pub s1: usize,
    pub s2: usize,
    pub visits: u32,
    pub mean: f32,
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
    /// Deepest / average (over iterations) ply of the node an iteration expanded
    /// (1 = a child of the root). Single-threaded search only; 0 otherwise.
    pub depth_max: u32,
    pub depth_mean: f32,
    /// Root joint matrix (only cells that were expanded); `Some` only when requested.
    pub joint: Option<Vec<JointCell>>,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    rng: &mut impl Rng,
    leaf_eval: &mut LeafEval,
    sel: &SelectionConfig,
) {
    let (mut new_node, s1_move, s2_move) =
        unsafe { root_node.selection(state, children, rng, sel, leaf_eval) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move, children, rng) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval, leaf_eval) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

enum SearchLimit {
    Time(Duration),
    Iterations(u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    limit: SearchLimit,
    leaf_eval: &mut LeafEval,
    sel: &SelectionConfig,
) {
    let mut rng = rng();
    let start_time = std::time::Instant::now();
    loop {
        for _ in 0..1000 {
            mcts_iteration(root_node, state, root_eval, children, &mut rng, leaf_eval, sel);
        }
        if root_node.times_visited >= 10_000_000 {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
) -> MctsResult {
    perform_mcts_with(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        None,
    )
}

/// Like `perform_mcts`, but leaves are scored by `evaluator` when it is `Some`
/// (see `learned.rs`). With `None` this is exactly the stock search.
pub fn perform_mcts_with(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    evaluator: Option<&Evaluator>,
) -> MctsResult {
    perform_mcts_selecting(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        evaluator,
        &SelectionConfig::default(),
    )
}

/// Like `perform_mcts_with`, with a runtime selection mode (ucb1 / exp3 / rm).
/// `SelectionConfig::default()` is exactly the stock decoupled-UCB1 search.
pub fn perform_mcts_selecting(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    evaluator: Option<&Evaluator>,
    sel: &SelectionConfig,
) -> MctsResult {
    perform_mcts_joint(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        evaluator,
        sel,
        false,
    )
}

/// Like `perform_mcts_selecting`; with `want_joint` the result also carries the root's
/// joint-action matrix (`MctsResult::joint`), read off the root's chance children.
pub fn perform_mcts_joint(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    evaluator: Option<&Evaluator>,
    sel: &SelectionConfig,
    want_joint: bool,
) -> MctsResult {
    let mut leaf_eval = LeafEval::new(evaluator);
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;
    if let Some(pc) = sel.prior.as_ref() {
        fill_priors(
            pc,
            state,
            root_node.s1_options.as_mut().unwrap(),
            root_node.s2_options.as_mut().unwrap(),
            true,
            &mut leaf_eval.prior_scratch,
        );
    }
    let mut children: HashMap<(usize, usize, usize), Box<[Node]>> = HashMap::new();

    let root_eval = leaf_eval.value(state);
    let search_limit = if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &root_eval,
        &mut children,
        search_limit,
        &mut leaf_eval,
        sel,
    );

    let side_results = |options: &[MoveNode], stats: Option<&SideStats>| -> Vec<MctsSideResult> {
        let total_visits: f32 = options.iter().map(|v| v.visits as f32).sum();
        let strategy: Vec<f32> = match stats {
            Some(st) => {
                let a = st.avg_strategy();
                let t: f64 = a.iter().sum();
                a.iter()
                    .map(|x| if t > 0.0 { (x / t) as f32 } else { 0.0 })
                    .collect()
            }
            None => options
                .iter()
                .map(|v| {
                    if total_visits > 0.0 {
                        v.visits as f32 / total_visits
                    } else {
                        0.0
                    }
                })
                .collect(),
        };
        options
            .iter()
            .zip(strategy)
            .map(|(v, avg_strategy)| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                avg_strategy,
            })
            .collect()
    };

    let joint = if want_joint {
        let root_ptr = &root_node as *const Node as usize;
        let mut cells: Vec<JointCell> = children
            .iter()
            .filter(|((ptr, _, _), _)| *ptr == root_ptr)
            .map(|((_, i, j), nodes)| {
                let visits: u32 = nodes.iter().map(|n| n.times_visited).sum();
                let total: f32 = nodes.iter().map(|n| n.total_score).sum();
                JointCell {
                    s1: *i,
                    s2: *j,
                    visits,
                    mean: if visits > 0 { total / visits as f32 } else { 0.0 },
                }
            })
            .collect();
        cells.sort_by_key(|c| (c.s1, c.s2));
        Some(cells)
    } else {
        None
    };

    MctsResult {
        joint,
        s1: side_results(root_node.s1_options.as_ref().unwrap(), root_node.stats.as_ref().map(|n| &n.s1)),
        s2: side_results(root_node.s2_options.as_ref().unwrap(), root_node.stats.as_ref().map(|n| &n.s2)),
        iteration_count: root_node.times_visited,
        depth_max: leaf_eval.depth_max,
        depth_mean: if root_node.times_visited > 0 {
            (leaf_eval.depth_sum as f64 / root_node.times_visited as f64) as f32
        } else {
            0.0
        },
    }
}

#[cfg(test)]
#[path = "mcts_prior_tests.rs"]
mod prior_tests;
