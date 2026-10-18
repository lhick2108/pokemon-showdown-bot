//! Net-guided search (bead showdown-bot-589.68.5): a Jaxcalibur-style search where every
//! expansion is evaluated by a big policy+value network instead of a cheap leaf evaluator.
//!
//! One tree per sampled world. Decision nodes are simultaneous-move nodes with decoupled PUCT
//! per side (each side has its own priors, visit counts and values; values are stored from
//! side one's view in [0, 1] and side two reads `1 - v`). A joint action leads to a chance
//! node whose outcomes are the engine's instruction branches; one outcome is sampled per
//! descent (exactly what `mcts.rs` does).
//!
//! The search is driven from outside in batches (no evaluator is called from Rust):
//!
//! * [`NetSearch::collect`] descends the trees, applying a virtual loss along every pending
//!   path, until `max_batch` unevaluated nodes have been found. For each it writes the token
//!   buffers of the node's state from side one's AND side two's view (`tokens.rs`).
//! * the caller evaluates the batch (policy over the 13 action slots of each side, value =
//!   P(side one wins)) and calls [`NetSearch::feed`], which sets the priors of the node,
//!   removes the virtual loss and backs the value up.
//!
//! Terminal states back up their exact result without a net call. A node `depth` turns below
//! the root is never expanded: it is evaluated once for its value and that value is reused.
//! Every descent starts from a clone of the world's root state and applies the sampled
//! outcome instructions on the way down (the clone is dropped, so nothing has to be undone).

use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::prior::{action_slot, N_ACTIONS};
use crate::state::{SideReference, State};
use crate::tokens::{self, Revealed, CAT_LEN, CONT_LEN};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::VecDeque;
use std::time::Instant;

const NONE: u32 = u32::MAX;
/// free descents (known value / collision) in a row after which a world is skipped for the rest of a collect call
const STUCK: u32 = 8;

#[derive(Clone, Debug)]
pub struct Params {
    pub c_puct: f32,
    /// first-play urgency: an unvisited option starts at `node mean - fpu`
    pub fpu: f32,
    /// turns below the root at which a node is only evaluated for its value
    pub depth: u16,
    /// priors are mixed with the uniform distribution: `(1 - eps) p + eps / K`
    pub eps: f32,
    /// token views written per leaf: 2 = side one's and side two's, 1 = side one's only (the
    /// caller then takes side two's policy from the net's opponent head)
    pub views: u8,
}

impl Default for Params {
    fn default() -> Params {
        Params { c_puct: 2.0, fpu: 0.0, depth: 4, eps: 0.01, views: 2 }
    }
}

#[derive(Clone, Debug)]
pub struct Opt {
    pub choice: MoveChoice,
    /// action slot of the option for the net (None for `MoveChoice::None`)
    pub slot: Option<u8>,
    pub prior: f32,
    pub visits: u32,
    pub vloss: u32,
    /// sum of backed-up values from this side's view
    pub total: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Fresh,
    Pending,
    /// options and priors are set; descents go through it
    Ready,
    /// depth limit: the stored value is backed up on every visit
    Leaf,
    Terminal,
}

struct Outcome {
    instr: StateInstructions,
    child: u32,
}

struct Node {
    #[allow(dead_code)]
    parent: u32,
    depth: u16,
    status: Status,
    visits: u32,
    /// sum of backed-up values, side one's view
    total: f32,
    /// value of the node's own evaluation (Leaf / Terminal / Ready), side one's view
    value: f32,
    s1: Vec<Opt>,
    s2: Vec<Opt>,
    /// `chance[a1 * s2.len() + a2]`
    chance: Vec<Option<Box<[Outcome]>>>,
}

impl Node {
    fn new(parent: u32, depth: u16) -> Node {
        Node {
            parent,
            depth,
            status: Status::Fresh,
            visits: 0,
            total: 0.0,
            value: 0.5,
            s1: Vec::new(),
            s2: Vec::new(),
            chance: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Step {
    node: u32,
    a1: u16,
    a2: u16,
}

struct World {
    root_state: State,
    nodes: Vec<Node>,
    /// consecutive collisions in this collect call
    stuck: u32,
}

/// A leaf waiting for its evaluation.
struct PendingLeaf {
    world: u32,
    node: u32,
    path: Vec<Step>,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub evals: u64,
    pub batches: u64,
    pub collisions: u64,
    pub terminal: u64,
    pub leaf_reuse: u64,
    pub both_none: u64,
    pub max_depth: u16,
    pub nodes: u64,
    pub tree_ns: u64,
    pub tokenise_ns: u64,
    pub feed_ns: u64,
}

/// What one collect call hands out: token buffers (row 0 side one's view, row 1 side two's)
/// for `n` leaves, in the order [`NetSearch::feed`] expects the evaluations.
pub struct Collected {
    pub n: usize,
    /// `[n][views][CAT_LEN]`
    pub cat: Vec<i16>,
    /// `[n][views][CONT_LEN]`
    pub cont: Vec<f32>,
}

/// Evaluation of one leaf: policies over the 13 slots of each side (any non-negative
/// weights; only the legal slots of the node are read) and P(side one wins).
pub struct Eval {
    pub policy1: [f32; N_ACTIONS],
    pub policy2: [f32; N_ACTIONS],
    pub value: f32,
}

pub struct NetSearch {
    worlds: Vec<World>,
    params: Params,
    reveal1: Option<Vec<Revealed>>,
    rng: StdRng,
    inflight: VecDeque<Vec<PendingLeaf>>,
    cursor: usize,
    pub stats: Stats,
}

/// Decoupled PUCT for one side. `node_mean` is the node's mean value from this side's view
/// (the first-play value of an unvisited option is `node_mean - fpu`). A virtual loss counts
/// as a visit that scored 0.
pub fn puct_pick(opts: &[Opt], node_visits: u32, node_mean: f32, p: &Params) -> usize {
    if opts.len() == 1 {
        return 0;
    }
    let pending: u32 = opts.iter().map(|o| o.vloss).sum();
    let sqrt_n = ((node_visits + pending).max(1) as f32).sqrt();
    let fpu_q = node_mean - p.fpu;
    let mut best = f32::MIN;
    let mut choice = 0;
    for (i, o) in opts.iter().enumerate() {
        let n_eff = o.visits + o.vloss;
        let q = if n_eff == 0 { fpu_q } else { o.total / n_eff as f32 };
        let score = q + p.c_puct * o.prior * sqrt_n / (1.0 + n_eff as f32);
        if score > best {
            best = score;
            choice = i;
        }
    }
    choice
}

fn make_opts(side: &crate::state::Side, choices: Vec<MoveChoice>) -> Vec<Opt> {
    let k = choices.len().max(1) as f32;
    choices
        .into_iter()
        .map(|c| {
            let slot = action_slot(side, &c).map(|s| s as u8);
            Opt { choice: c, slot, prior: 1.0 / k, visits: 0, vloss: 0, total: 0.0 }
        })
        .collect()
}

/// Priors of the options from a 13-slot policy: renormalised over the legal options, mixed
/// with the uniform distribution by `eps`.
pub fn set_priors(opts: &mut [Opt], policy: &[f32; N_ACTIONS], eps: f32) {
    let k = opts.len();
    let raw: Vec<f32> = opts
        .iter()
        .map(|o| match o.slot {
            Some(s) => {
                let p = policy[s as usize];
                if p.is_finite() && p > 0.0 {
                    p
                } else {
                    0.0
                }
            }
            None => 0.0,
        })
        .collect();
    let sum: f32 = raw.iter().sum();
    for (o, r) in opts.iter_mut().zip(raw) {
        let p = if sum > 0.0 { r / sum } else { 1.0 / k as f32 };
        o.prior = (1.0 - eps) * p + eps / k as f32;
    }
}

impl NetSearch {
    pub fn new(states: Vec<State>, params: Params, reveal1: Option<Vec<Revealed>>, seed: u64) -> NetSearch {
        let worlds = states
            .into_iter()
            .map(|s| World { root_state: s, nodes: vec![Node::new(NONE, 0)], stuck: 0 })
            .collect();
        NetSearch {
            worlds,
            params,
            reveal1,
            rng: StdRng::seed_from_u64(seed),
            inflight: VecDeque::new(),
            cursor: 0,
            stats: Stats::default(),
        }
    }

    pub fn n_worlds(&self) -> usize {
        self.worlds.len()
    }

    /// Leaves handed out by `collect` and not yet fed.
    pub fn pending(&self) -> usize {
        self.inflight.iter().map(|b| b.len()).sum()
    }

    /// Descends the trees (round-robin over worlds) and returns the tokens of up to `max_batch`
    /// unevaluated leaves. An empty batch means nothing more can be asked for right now
    /// (everything left is pending in an in-flight batch, or every descent ended without a
    /// net call).
    pub fn collect(&mut self, max_batch: usize) -> Collected {
        let t0 = Instant::now();
        let tok0 = self.stats.tokenise_ns;
        let mut leaves: Vec<PendingLeaf> = Vec::new();
        let mut cat: Vec<i16> = Vec::new();
        let mut cont: Vec<f32> = Vec::new();
        let nw = self.worlds.len();
        for w in &mut self.worlds {
            w.stuck = 0;
        }
        let mut attempts = 0usize;
        let max_attempts = max_batch.saturating_mul(6).max(64);
        loop {
            let mut progressed = false;
            for _ in 0..nw {
                if leaves.len() >= max_batch || attempts >= max_attempts {
                    break;
                }
                let wi = self.cursor % nw;
                self.cursor += 1;
                if self.worlds[wi].stuck >= STUCK {
                    continue;
                }
                attempts += 1;
                progressed = true;
                if let Some(l) = self.descend(wi, &mut cat, &mut cont) {
                    self.worlds[wi].stuck = 0;
                    leaves.push(l);
                } else if self.worlds[wi].stuck < STUCK {
                    // a free backup (known value) or a collision: after STUCK in a row this
                    // world has nothing new to ask for in this call
                    self.worlds[wi].stuck += 1;
                }
            }
            if !progressed || leaves.len() >= max_batch || attempts >= max_attempts {
                break;
            }
        }
        let n = leaves.len();
        if n > 0 {
            self.stats.batches += 1;
            self.inflight.push_back(leaves);
        }
        // tree time excludes tokenisation (counted separately)
        let total = t0.elapsed().as_nanos() as u64;
        self.stats.tree_ns += total.saturating_sub(self.stats.tokenise_ns - tok0);
        Collected { n, cat, cont }
    }

    /// Completes the OLDEST in-flight batch. `evals.len()` must equal its size.
    pub fn feed(&mut self, evals: &[Eval]) -> Result<(), String> {
        let t0 = Instant::now();
        let batch = self.inflight.pop_front().ok_or("feed without a collected batch")?;
        if batch.len() != evals.len() {
            let n = batch.len();
            self.inflight.push_front(batch);
            return Err(format!("feed got {} evaluations for a batch of {}", evals.len(), n));
        }
        let eps = self.params.eps;
        let depth_limit = self.params.depth;
        for (leaf, ev) in batch.into_iter().zip(evals) {
            let value = if ev.value.is_finite() { ev.value.clamp(0.0, 1.0) } else { 0.5 };
            let node = &mut self.worlds[leaf.world as usize].nodes[leaf.node as usize];
            node.value = value;
            if node.depth >= depth_limit {
                node.status = Status::Leaf;
            } else {
                set_priors(&mut node.s1, &ev.policy1, eps);
                set_priors(&mut node.s2, &ev.policy2, eps);
                node.status = Status::Ready;
            }
            self.stats.evals += 1;
            self.backprop(leaf.world as usize, leaf.node, &leaf.path, value);
        }
        self.stats.feed_ns += t0.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Backs `value` up the path: visits and totals of every node and chosen option, and
    /// removes the virtual loss the descent put on the chosen options.
    fn backprop(&mut self, wi: usize, leaf: u32, path: &[Step], value: f32) {
        let nodes = &mut self.worlds[wi].nodes;
        nodes[leaf as usize].visits += 1;
        nodes[leaf as usize].total += value;
        for st in path.iter().rev() {
            let n = &mut nodes[st.node as usize];
            n.visits += 1;
            n.total += value;
            let o1 = &mut n.s1[st.a1 as usize];
            o1.visits += 1;
            o1.total += value;
            o1.vloss -= 1;
            let o2 = &mut n.s2[st.a2 as usize];
            o2.visits += 1;
            o2.total += 1.0 - value;
            o2.vloss -= 1;
        }
    }

    fn release(&mut self, wi: usize, path: &[Step]) {
        let nodes = &mut self.worlds[wi].nodes;
        for st in path {
            let n = &mut nodes[st.node as usize];
            n.s1[st.a1 as usize].vloss -= 1;
            n.s2[st.a2 as usize].vloss -= 1;
        }
    }

    /// One descent in world `wi`. Returns a pending leaf (its tokens appended to `cat` /
    /// `cont`), or None when the descent backed up a known value or collided with a pending
    /// node.
    fn descend(&mut self, wi: usize, cat: &mut Vec<i16>, cont: &mut Vec<f32>) -> Option<PendingLeaf> {
        let params = self.params.clone();
        let mut state = self.worlds[wi].root_state.clone();
        let mut path: Vec<Step> = Vec::new();
        let mut cur: u32 = 0;
        loop {
            let status = self.worlds[wi].nodes[cur as usize].status;
            match status {
                Status::Pending => {
                    self.stats.collisions += 1;
                    self.release(wi, &path);
                    return None;
                }
                Status::Terminal | Status::Leaf => {
                    let v = self.worlds[wi].nodes[cur as usize].value;
                    if status == Status::Terminal {
                        self.stats.terminal += 1;
                    } else {
                        self.stats.leaf_reuse += 1;
                    }
                    self.backprop(wi, cur, &path, v);
                    return None;
                }
                Status::Fresh => {
                    let over = state.battle_is_over();
                    if over != 0.0 {
                        let v = if over > 0.0 { 1.0 } else { 0.0 };
                        let n = &mut self.worlds[wi].nodes[cur as usize];
                        n.status = Status::Terminal;
                        n.value = v;
                        self.stats.terminal += 1;
                        self.backprop(wi, cur, &path, v);
                        return None;
                    }
                    // an unevaluated node: queue it for the net
                    let t1 = Instant::now();
                    {
                        let node = &mut self.worlds[wi].nodes[cur as usize];
                        node.status = Status::Pending;
                        if node.depth < params.depth {
                            let (o1, o2) = state.root_get_all_options();
                            node.s1 = make_opts(&state.side_one, o1);
                            node.s2 = make_opts(&state.side_two, o2);
                            node.chance = (0..node.s1.len() * node.s2.len()).map(|_| None).collect();
                        }
                    }
                    let base_c = cat.len();
                    let base_f = cont.len();
                    let views = params.views.clamp(1, 2) as usize;
                    cat.resize(base_c + views * CAT_LEN, 0);
                    cont.resize(base_f + views * CONT_LEN, 0.0);
                    tokens::state_tokens_into(
                        &state,
                        SideReference::SideOne,
                        self.reveal1.as_deref(),
                        &mut cat[base_c..base_c + CAT_LEN],
                        &mut cont[base_f..base_f + CONT_LEN],
                    );
                    if views == 2 {
                        tokens::state_tokens_into(
                            &state,
                            SideReference::SideTwo,
                            None,
                            &mut cat[base_c + CAT_LEN..base_c + 2 * CAT_LEN],
                            &mut cont[base_f + CONT_LEN..base_f + 2 * CONT_LEN],
                        );
                    }
                    self.stats.tokenise_ns += t1.elapsed().as_nanos() as u64;
                    return Some(PendingLeaf { world: wi as u32, node: cur, path });
                }
                Status::Ready => {
                    let (a1, a2, depth, n2, both_none) = {
                        let node = &self.worlds[wi].nodes[cur as usize];
                        let mean = if node.visits > 0 { node.total / node.visits as f32 } else { node.value };
                        let a1 = puct_pick(&node.s1, node.visits, mean, &params);
                        let a2 = puct_pick(&node.s2, node.visits, 1.0 - mean, &params);
                        let bn = node.s1[a1].choice == MoveChoice::None && node.s2[a2].choice == MoveChoice::None;
                        (a1, a2, node.depth, node.s2.len(), bn)
                    };
                    if both_none {
                        // nothing can happen: the node's own value stands in for the visit
                        self.stats.both_none += 1;
                        let v = self.worlds[wi].nodes[cur as usize].value;
                        self.backprop(wi, cur, &path, v);
                        return None;
                    }
                    {
                        let node = &mut self.worlds[wi].nodes[cur as usize];
                        node.s1[a1].vloss += 1;
                        node.s2[a2].vloss += 1;
                    }
                    path.push(Step { node: cur, a1: a1 as u16, a2: a2 as u16 });
                    let key = a1 * n2 + a2;
                    if self.worlds[wi].nodes[cur as usize].chance[key].is_none() {
                        let (c1, c2) = {
                            let node = &self.worlds[wi].nodes[cur as usize];
                            (node.s1[a1].choice.clone(), node.s2[a2].choice.clone())
                        };
                        let mut list = generate_instructions_from_move_pair(&mut state, &c1, &c2, depth <= 1);
                        if list.is_empty() {
                            list.push(StateInstructions::default());
                        }
                        let w = &mut self.worlds[wi];
                        let mut outs = Vec::with_capacity(list.len());
                        for instr in list.drain(..) {
                            let child = w.nodes.len() as u32;
                            w.nodes.push(Node::new(cur, depth + 1));
                            self.stats.nodes += 1;
                            outs.push(Outcome { instr, child });
                        }
                        w.nodes[cur as usize].chance[key] = Some(outs.into_boxed_slice());
                    }
                    let outs = self.worlds[wi].nodes[cur as usize].chance[key].as_ref().unwrap();
                    let total: f32 = outs.iter().map(|o| o.instr.percentage.max(0.0)).sum();
                    let mut idx = outs.len() - 1;
                    if total > 0.0 {
                        let mut th = self.rng.random_range(0.0..total);
                        for (i, o) in outs.iter().enumerate() {
                            th -= o.instr.percentage.max(0.0);
                            if th <= 0.0 {
                                idx = i;
                                break;
                            }
                        }
                    } else {
                        idx = 0;
                    }
                    state.apply_instructions(&outs[idx].instr.instruction_list);
                    cur = outs[idx].child;
                    let d = self.worlds[wi].nodes[cur as usize].depth;
                    if d > self.stats.max_depth {
                        self.stats.max_depth = d;
                    }
                }
            }
        }
    }

    /// Root option statistics of world `wi`: `(side one options, side two options)` as
    /// `(choice, visits, mean value from that side's view, prior)`.
    #[allow(clippy::type_complexity)]
    pub fn root(&self, wi: usize) -> (Vec<(MoveChoice, u32, f32, f32)>, Vec<(MoveChoice, u32, f32, f32)>) {
        let n = &self.worlds[wi].nodes[0];
        let f = |o: &Opt| (o.choice.clone(), o.visits, if o.visits > 0 { o.total / o.visits as f32 } else { 0.0 }, o.prior);
        (n.s1.iter().map(f).collect(), n.s2.iter().map(f).collect())
    }

    /// Mean backed-up value of the root, side one's view.
    pub fn root_value(&self, wi: usize) -> f32 {
        let n = &self.worlds[wi].nodes[0];
        if n.visits > 0 {
            n.total / n.visits as f32
        } else {
            0.5
        }
    }

    pub fn root_visits(&self, wi: usize) -> u32 {
        self.worlds[wi].nodes[0].visits
    }

    pub fn root_state(&self, wi: usize) -> &State {
        &self.worlds[wi].root_state
    }

    /// Slots of the options of the nodes pending in flight, in feed order (tests / debugging).
    #[allow(clippy::type_complexity)]
    pub fn pending_slots(&self) -> Vec<(Vec<Option<u8>>, Vec<Option<u8>>)> {
        let mut out = Vec::new();
        for b in &self.inflight {
            for l in b {
                let n = &self.worlds[l.world as usize].nodes[l.node as usize];
                out.push((n.s1.iter().map(|o| o.slot).collect(), n.s2.iter().map(|o| o.slot).collect()));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn states() -> Vec<State> {
        include_str!("../data/gen9randombattle.txt")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(State::deserialize)
            .collect()
    }

    fn opt(prior: f32, visits: u32, vloss: u32, total: f32) -> Opt {
        Opt { choice: MoveChoice::None, slot: None, prior, visits, vloss, total }
    }

    fn fake(n: usize, p1: [f32; N_ACTIONS], p2: [f32; N_ACTIONS], v: f32) -> Vec<Eval> {
        (0..n).map(|_| Eval { policy1: p1, policy2: p2, value: v }).collect()
    }

    fn run(ns: &mut NetSearch, batch: usize, rounds: usize, p1: [f32; N_ACTIONS], p2: [f32; N_ACTIONS], v: f32) {
        for _ in 0..rounds {
            let c = ns.collect(batch);
            if c.n == 0 {
                break;
            }
            ns.feed(&fake(c.n, p1, p2, v)).unwrap();
        }
    }

    #[test]
    fn puct_follows_prior_for_both_sides() {
        let p = Params { c_puct: 1.0, ..Params::default() };
        let opts = vec![opt(0.1, 0, 0, 0.0), opt(0.7, 0, 0, 0.0), opt(0.2, 0, 0, 0.0)];
        assert_eq!(puct_pick(&opts, 0, 0.5, &p), 1);
        // a strong value beats a high prior with many visits
        let opts = vec![opt(0.8, 500, 0, 150.0), opt(0.1, 100, 0, 90.0), opt(0.1, 400, 0, 260.0)];
        assert_eq!(puct_pick(&opts, 1000, 0.5, &p), 1);
        // an unvisited option starts at the node mean of the side: side two (mean 0.1) keeps
        // its visited 0.3 option, side two with mean 0.9 explores the unvisited one
        let opts = vec![opt(0.01, 50, 0, 15.0), opt(0.01, 0, 0, 0.0)];
        assert_eq!(puct_pick(&opts, 100, 0.1, &p), 0);
        assert_eq!(puct_pick(&opts, 100, 0.9, &p), 1);
    }

    #[test]
    fn virtual_loss_diverts_descents() {
        let p = Params { c_puct: 1.0, ..Params::default() };
        let mut opts = vec![opt(0.6, 10, 0, 6.0), opt(0.4, 10, 0, 7.0)];
        assert_eq!(puct_pick(&opts, 20, 0.5, &p), 1);
        opts[1].vloss = 3;
        assert_eq!(puct_pick(&opts, 20, 0.5, &p), 0);
        let mut fresh = vec![opt(0.5, 0, 0, 0.0), opt(0.3, 0, 0, 0.0), opt(0.2, 0, 0, 0.0)];
        assert_eq!(puct_pick(&fresh, 0, 0.5, &p), 0);
        fresh[0].vloss = 1;
        assert_ne!(puct_pick(&fresh, 0, 0.5, &p), 0);
    }

    #[test]
    fn priors_normalise_over_legal_options() {
        let mut o = vec![
            Opt { choice: MoveChoice::None, slot: Some(0), prior: 0.0, visits: 0, vloss: 0, total: 0.0 },
            Opt { choice: MoveChoice::None, slot: Some(9), prior: 0.0, visits: 0, vloss: 0, total: 0.0 },
        ];
        let mut pol = [0f32; N_ACTIONS];
        pol[0] = 0.2;
        pol[9] = 0.6;
        pol[4] = 0.9; // not a legal option: ignored
        set_priors(&mut o, &pol, 0.0);
        assert!((o[0].prior - 0.25).abs() < 1e-6 && (o[1].prior - 0.75).abs() < 1e-6);
        set_priors(&mut o, &[0.0; N_ACTIONS], 0.0); // degenerate -> uniform
        assert!((o[0].prior - 0.5).abs() < 1e-6);
    }

    #[test]
    fn root_priors_are_used_for_both_sides() {
        let st = states();
        let mut ns = NetSearch::new(vec![st[0].clone()], Params::default(), None, 1);
        let c = ns.collect(8);
        assert_eq!(c.n, 1, "the first batch is the root alone");
        assert_eq!(c.cat.len(), 2 * CAT_LEN);
        assert_eq!(c.cont.len(), 2 * CONT_LEN);
        let slots = ns.pending_slots();
        let (s1, s2) = &slots[0];
        let last1 = s1.iter().rev().flatten().next().unwrap();
        let last2 = s2.iter().rev().flatten().next().unwrap();
        let mut p1 = [0f32; N_ACTIONS];
        let mut p2 = [0f32; N_ACTIONS];
        p1[*last1 as usize] = 1.0;
        p2[*last2 as usize] = 1.0;
        ns.feed(&fake(1, p1, p2, 0.5)).unwrap();
        let (r1, r2) = ns.root(0);
        let best1 = r1.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
        let best2 = r2.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
        assert!(best1.3 > 0.9 && best2.3 > 0.9);
        let c = ns.collect(4);
        assert!(c.n >= 1);
        // the first descent follows both priors
        let (r1, r2) = ns.root(0);
        assert!(r1.iter().any(|o| o.3 > 0.9));
        assert!(r2.iter().any(|o| o.3 > 0.9));
    }

    #[test]
    fn collisions_at_pending_root() {
        let st = states();
        let mut ns = NetSearch::new(vec![st[0].clone()], Params::default(), None, 2);
        assert_eq!(ns.collect(8).n, 1);
        assert_eq!(ns.collect(8).n, 0, "root pending: every further descent collides");
        assert!(ns.stats.collisions >= 1);
    }

    #[test]
    fn virtual_loss_spreads_a_batch_and_is_released() {
        let st = states();
        let mut ns = NetSearch::new(vec![st[0].clone()], Params::default(), None, 3);
        run(&mut ns, 1, 1, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5); // evaluate the root
        let c = ns.collect(16);
        assert!(c.n > 4, "a flat prior with virtual loss must open several children, got {}", c.n);
        let mut firsts = std::collections::HashSet::new();
        for b in &ns.inflight {
            for l in b {
                firsts.insert((l.path[0].a1, l.path[0].a2));
            }
        }
        assert!(firsts.len() >= 3);
        {
            let n = &ns.worlds[0].nodes[0];
            let pend: u32 = n.s1.iter().map(|o| o.vloss).sum();
            assert_eq!(pend as usize, c.n, "one virtual loss per pending path");
        }
        ns.feed(&fake(c.n, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
        let n = &ns.worlds[0].nodes[0];
        assert!(n.s1.iter().all(|o| o.vloss == 0) && n.s2.iter().all(|o| o.vloss == 0));
        assert_eq!(n.visits as usize, 1 + c.n);
    }

    #[test]
    fn value_backs_up_with_the_right_perspective() {
        let st = states();
        let mut ns = NetSearch::new(vec![st[0].clone()], Params::default(), None, 4);
        run(&mut ns, 8, 40, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.8);
        let (r1, r2) = ns.root(0);
        for o in r1.iter().filter(|o| o.1 > 0) {
            assert!(o.2 > 0.65, "{:?}", o);
        }
        for o in r2.iter().filter(|o| o.1 > 0) {
            assert!(o.2 < 0.35, "{:?}", o);
        }
        assert!(ns.root_value(0) > 0.65);
        let total: u32 = r1.iter().map(|o| o.1).sum();
        assert_eq!(total, ns.root_visits(0) - 1);
    }

    #[test]
    fn depth_limit_reuses_leaf_values() {
        let st = states();
        let p = Params { depth: 1, ..Params::default() };
        let mut ns = NetSearch::new(vec![st[0].clone()], p, None, 5);
        run(&mut ns, 4, 60, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.6);
        assert!(ns.stats.max_depth <= 1);
        assert!(ns.stats.leaf_reuse > 0, "depth-1 leaves are revisited without a net call");
        assert!(ns.stats.evals < ns.root_visits(0) as u64);
    }

    #[test]
    fn terminal_states_use_the_exact_result() {
        let mut s = states()[0].clone();
        for p in s.side_two.pokemon.pkmn.iter_mut() {
            p.hp = 0;
        }
        let mut ns = NetSearch::new(vec![s], Params::default(), None, 6);
        assert_eq!(ns.collect(4).n, 0, "a finished battle needs no evaluation");
        assert!(ns.stats.terminal >= 1 && ns.stats.terminal <= STUCK as u64);
        assert_eq!(ns.worlds[0].nodes[0].value, 1.0);
        let mut s = states()[0].clone();
        for p in s.side_one.pokemon.pkmn.iter_mut() {
            p.hp = 0;
        }
        let mut ns = NetSearch::new(vec![s], Params::default(), None, 6);
        ns.collect(4);
        assert_eq!(ns.worlds[0].nodes[0].status, Status::Terminal);
        assert_eq!(ns.worlds[0].nodes[0].value, 0.0);
    }

    #[test]
    fn terminal_children_back_up_exactly() {
        // every one of side two's Pokemon but the active one fainted and the active at 1 hp:
        // most joint actions end the game within the search
        let mut s = states()[0].clone();
        let active = s.side_two.active_index as usize;
        for (i, p) in s.side_two.pokemon.pkmn.iter_mut().enumerate() {
            if i != active {
                p.hp = 0;
            } else {
                p.hp = 1;
            }
        }
        let mut ns = NetSearch::new(vec![s], Params::default(), None, 8);
        run(&mut ns, 8, 40, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5);
        assert!(ns.stats.terminal > 0);
        // wins are exact 1.0 backups, so some root option has a mean above the 0.5 net value
        let (r1, _) = ns.root(0);
        assert!(r1.iter().any(|o| o.1 > 0 && o.2 > 0.55), "{:?}", r1);
    }

    #[test]
    fn many_worlds_and_slot_mapping_both_sides() {
        let st = states();
        let worlds: Vec<State> = st.iter().take(8).cloned().collect();
        let mut ns = NetSearch::new(worlds, Params::default(), None, 7);
        let c = ns.collect(64);
        assert_eq!(c.n, 8);
        let slots = ns.pending_slots();
        for (i, (s1, s2)) in slots.iter().enumerate() {
            assert!(!s1.is_empty() && !s2.is_empty());
            // row 0 of the token pair is side one's view, row 1 side two's; in each the first
            // `act_legal` row is the viewer's own legal slots
            for (row, ss) in [(0usize, s1), (1usize, s2)] {
                let base = i * 2 * CAT_LEN + row * CAT_LEN + tokens::O_ACT_LEGAL;
                let own: Vec<usize> = (0..N_ACTIONS).filter(|&k| c.cat[base + k] > 0).collect();
                let mut want: Vec<usize> = ss.iter().flatten().map(|&s| s as usize).collect();
                want.sort();
                assert_eq!(own, want, "world {i} row {row}");
            }
        }
    }

    #[test]
    fn one_view_writes_half_the_tokens() {
        let st = states();
        let p = Params { views: 1, ..Params::default() };
        let mut ns = NetSearch::new(vec![st[0].clone(); 3], p, None, 11);
        let c = ns.collect(8);
        assert_eq!(c.n, 3);
        assert_eq!(c.cat.len(), 3 * CAT_LEN);
        assert_eq!(c.cont.len(), 3 * CONT_LEN);
        ns.feed(&fake(3, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
        let c = ns.collect(8);
        assert!(c.n > 3 && c.cat.len() == c.n * CAT_LEN);
    }

    #[test]
    fn deterministic_for_a_seed() {
        let st = states();
        let mut a = NetSearch::new(vec![st[1].clone()], Params::default(), None, 9);
        let mut b = NetSearch::new(vec![st[1].clone()], Params::default(), None, 9);
        run(&mut a, 8, 30, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5);
        run(&mut b, 8, 30, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5);
        let va: Vec<u32> = a.root(0).0.iter().map(|o| o.1).collect();
        let vb: Vec<u32> = b.root(0).0.iter().map(|o| o.1).collect();
        assert_eq!(va, vb);
    }
}
