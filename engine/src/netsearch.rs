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
use crate::features_v2::moves_of;
use crate::features_v4::{move_idx, species_idx, IDS_PER_SLOT, IDX_UNKNOWN};
use crate::instruction::StateInstructions;
use crate::pokemon::PokemonName;
use crate::prior::{action_slot, N_ACTIONS};
use crate::state::{Pokemon, Side, SideReference, State};
use crate::tokens::{self, Revealed, CAT_LEN, CONT_LEN, C_MON, O_MON_CAT, O_MON_CONT};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::{HashMap, VecDeque};
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
    /// event-history rows handed out with every leaf (`Collected::hist_*`): 0 = none (the net sees
    /// padding), 1 = the root's 16 rows for every leaf, 2 = the root's rows pushed down by the
    /// turns simulated on the way to the leaf (two new rows per turn, most recent first)
    pub hist: u8,
    /// worker threads for tokenising the leaves of a batch (1 = in the calling thread)
    pub threads: u8,
    /// statistics side one selects its options with (sdb.31): 0 = the node's own (one tree per world),
    /// 1 = shared by every node, of any world, at the same depth whose side-one view is the same
    /// information set ([`view_key`]: the scrubbed tokens, continuous values in tenths)
    pub stats: u8,
    /// selection rule: 0 = PUCT (argmax of `Q + c P sqrt(N) / (1 + n)`), 1 = regret matching (an option is
    /// SAMPLED with weight `max(Q - V, 0) + c P / sqrt(N)`)
    pub select: u8,
    /// > 0: below an action of side one that side two could not expect (a switch to a Pokemon it has not
    /// seen, or an action its view predicted with probability < `cut`), side two no longer searches: it
    /// samples its prior
    pub cut: f32,
}

impl Default for Params {
    fn default() -> Params {
        Params { c_puct: 2.0, fpu: 0.0, depth: 4, eps: 0.01, views: 2, hist: 0, threads: 1, stats: 0, select: 0, cut: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Event history (docs/tokens.md "Event history", `tokens.build_history` in Python)
// ---------------------------------------------------------------------------

pub const HIST_ROWS: usize = 16;
pub const HIST_CAT_W: usize = 4;
pub const HIST_CONT_W: usize = 5;
pub const HIST_CAT_LEN: usize = HIST_ROWS * HIST_CAT_W;
pub const HIST_CONT_LEN: usize = HIST_ROWS * HIST_CONT_W;
/// Turns between two consecutive history decisions are 1 in the tree: `turns_ago / 16`.
const AGO_STEP: f32 = 1.0 / 16.0;

/// The 16 event rows of the token net, side one's view: row `2 (k - 1)` is our action `k` turns ago,
/// row `2 (k - 1) + 1` the opponent's. `cat` rows: `[side (0 pad, 1 own, 2 opp), kind + 1, action id,
/// species row of the actor's active Pokemon]`; `cont` rows: `[turns_ago / 16, hp change of the
/// actor's active, hp change of the other active, fainted on the actor's side / 6, fainted on the other
/// side / 6]`.
#[derive(Clone, Debug)]
pub struct Hist {
    pub cat: [i16; HIST_CAT_LEN],
    pub cont: [f32; HIST_CONT_LEN],
}

impl Default for Hist {
    fn default() -> Hist {
        Hist { cat: [0; HIST_CAT_LEN], cont: [0.0; HIST_CONT_LEN] }
    }
}

impl Hist {
    /// The same history from the other side's view: the two rows of every turn swap and the side
    /// ids flip (the changes are relative to the actor already).
    pub fn mirrored(&self) -> Hist {
        let mut m = Hist::default();
        for k in 0..HIST_ROWS / 2 {
            for (dst, src) in [(2 * k, 2 * k + 1), (2 * k + 1, 2 * k)] {
                m.cat[dst * HIST_CAT_W..(dst + 1) * HIST_CAT_W]
                    .copy_from_slice(&self.cat[src * HIST_CAT_W..(src + 1) * HIST_CAT_W]);
                m.cont[dst * HIST_CONT_W..(dst + 1) * HIST_CONT_W]
                    .copy_from_slice(&self.cont[src * HIST_CONT_W..(src + 1) * HIST_CONT_W]);
                let side = &mut m.cat[dst * HIST_CAT_W];
                *side = match *side {
                    1 => 2,
                    2 => 1,
                    x => x,
                };
            }
        }
        m
    }

    /// `self` one turn later with `rows` (our action, the opponent's) in front: older rows get
    /// `turns_ago + 1` (padding stays padding) and the two oldest rows fall off.
    fn pushed(&self, rows: &[([i16; HIST_CAT_W], [f32; HIST_CONT_W]); 2]) -> Hist {
        let mut h = Hist::default();
        for (r, (c, f)) in rows.iter().enumerate() {
            h.cat[r * HIST_CAT_W..(r + 1) * HIST_CAT_W].copy_from_slice(c);
            h.cont[r * HIST_CONT_W..(r + 1) * HIST_CONT_W].copy_from_slice(f);
        }
        h.cat[2 * HIST_CAT_W..].copy_from_slice(&self.cat[..HIST_CAT_LEN - 2 * HIST_CAT_W]);
        h.cont[2 * HIST_CONT_W..].copy_from_slice(&self.cont[..HIST_CONT_LEN - 2 * HIST_CONT_W]);
        for r in 2..HIST_ROWS {
            if h.cat[r * HIST_CAT_W] > 0 {
                let a = &mut h.cont[r * HIST_CONT_W];
                *a = (*a + AGO_STEP).min(1.0);
            }
        }
        h
    }
}

/// Public per-decision quantities of side one's (scrubbed) token view that the history deltas are
/// built from (`tokens._summary`): species of both active Pokemon and the hp fraction of every
/// alive, revealed Pokemon by species row.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    active: [i16; 2],
    n: [u8; 2],
    sp: [[i16; 6]; 2],
    hp: [[f32; 6]; 2],
}

impl Summary {
    pub fn of(cat: &[i16], cont: &[f32]) -> Summary {
        let mut s = Summary { active: [cat[O_MON_CAT], cat[O_MON_CAT + 6 * IDS_PER_SLOT]], ..Summary::default() };
        for m in 0..12 {
            let id = cat[O_MON_CAT + m * IDS_PER_SLOT];
            let alive = cont[O_MON_CONT + m * C_MON + 1] > 0.5;
            if alive && id > 1 {
                let side = m / 6;
                let k = s.n[side] as usize;
                // a repeated species keeps the last hp, as the Python dict does
                if let Some(j) = s.sp[side][..k].iter().position(|&x| x == id) {
                    s.hp[side][j] = cont[O_MON_CONT + m * C_MON];
                } else {
                    s.sp[side][k] = id;
                    s.hp[side][k] = cont[O_MON_CONT + m * C_MON];
                    s.n[side] += 1;
                }
            }
        }
        s
    }

    fn hp_of(&self, side: usize, species: i16) -> Option<f32> {
        self.sp[side][..self.n[side] as usize].iter().position(|&x| x == species).map(|j| self.hp[side][j])
    }

    /// hp change of the Pokemon active before, from `self` to `after` (0 if fainted or unknown).
    fn hp_delta(&self, after: &Summary, side: usize) -> f32 {
        let x = self.active[side];
        if x <= 1 {
            return 0.0;
        }
        match self.hp_of(side, x) {
            Some(b) => after.hp_of(side, x).unwrap_or(0.0) - b,
            None => 0.0,
        }
    }

    fn fainted(&self, after: &Summary, side: usize) -> u32 {
        self.sp[side][..self.n[side] as usize].iter().filter(|&&x| after.hp_of(side, x).is_none()).count() as u32
    }
}

/// Action row ingredients: `(kind, id)` with kind 0 move, 1 tera move, 2 switch, -1 none; id the
/// move or species vocabulary row (1 = unknown for a switch target the viewer has never seen).
type ActRow = (i8, i16);

/// The history rows of one simulated turn from `before` to `after` (`tokens.build_history`).
fn turn_rows(before: &Summary, after: &Summary, own: ActRow, opp: ActRow) -> [([i16; HIST_CAT_W], [f32; HIST_CONT_W]); 2] {
    let (d_own, d_opp) = (before.hp_delta(after, 0), before.hp_delta(after, 1));
    let (f_own, f_opp) = (before.fainted(after, 0) as f32 / 6.0, before.fainted(after, 1) as f32 / 6.0);
    [
        ([1, own.0 as i16 + 1, own.1, before.active[0]], [AGO_STEP, d_own, d_opp, f_own, f_opp]),
        ([2, opp.0 as i16 + 1, opp.1, before.active[1]], [AGO_STEP, d_opp, d_own, f_opp, f_own]),
    ]
}

fn norm_name(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// `(kind, id)` of `choice` taken by `side`. A switch to a Pokemon that `hidden` says the viewer
/// has not seen has the unknown species row.
fn act_row(side: &Side, choice: &MoveChoice, hidden: &dyn Fn(&Pokemon) -> bool) -> ActRow {
    let active = side.get_active_immutable();
    match choice {
        MoveChoice::Move(m) | MoveChoice::MoveMega(m) => (0, move_idx(moves_of(active)[*m as usize].id) as i16),
        MoveChoice::MoveTera(m) => (1, move_idx(moves_of(active)[*m as usize].id) as i16),
        MoveChoice::Switch(p) => {
            let target = &side.pokemon.pkmn[*p as usize];
            (2, if hidden(target) { IDX_UNKNOWN as i16 } else { species_idx(target) as i16 })
        }
        _ => (-1, 0),
    }
}

/// What a node keeps for the history of its children (`hist` mode 2).
#[derive(Clone, Debug)]
struct NodeInfo {
    summary: Summary,
    hist: Hist,
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
    /// side one's options: the probability the OTHER side's view gives this action (1 when not fed)
    pub pred: f32,
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
    /// summary and history of the node's state (`hist` mode 2, set when its tokens are written)
    info: Option<Box<NodeInfo>>,
    /// index of the information set whose statistics side one selects with (`stats` = 1), or NONE
    shared: u32,
    /// side two samples its prior here (`cut`)
    cut: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct SharedOpt {
    visits: u32,
    vloss: u32,
    total: f32,
}

/// Side one's statistics of one information set (`stats` = 1): what the per-node counters of every
/// member node add up to.
struct Shared {
    visits: u32,
    total: f32,
    opts: Vec<SharedOpt>,
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
            info: None,
            shared: NONE,
            cut: false,
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
    /// the leaf's state until its tokens are written (they are written after the descents, in parallel)
    state: Option<State>,
    /// hist mode 2: the parent's summary / history and the joint action taken to get here
    parent: Option<(Summary, Hist, ActRow, ActRow)>,
    /// hist mode 2: filled by the token writer
    info: Option<Box<NodeInfo>>,
    /// `stats` = 1: [`view_key`] of side one's tokens, filled by the token writer
    key: u64,
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
    /// information sets with shared statistics / nodes where side two only samples its prior
    pub info_sets: u64,
    pub cut_nodes: u64,
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
    /// `[n][views][HIST_CAT_LEN]` / `[n][views][HIST_CONT_LEN]` (empty when `hist` = 0); view 1
    /// (side two) holds the mirrored history
    pub hist_cat: Vec<i16>,
    pub hist_cont: Vec<f32>,
}

/// Evaluation of one leaf: policies over the 13 slots of each side (any non-negative
/// weights; only the legal slots of the node are read) and P(side one wins).
pub struct Eval {
    pub policy1: [f32; N_ACTIONS],
    pub policy2: [f32; N_ACTIONS],
    pub value: f32,
    /// side two's prediction of side one's action from ITS view, over side one's slots (`cut`)
    pub pred1: Option<[f32; N_ACTIONS]>,
}

/// Token context of ONE world (sdb.30, self-play search): every world is its own game, with its own reveal
/// rows for both views and its own root history per view.
#[derive(Clone, Debug)]
pub struct WorldCtx {
    pub reveal1: Vec<Revealed>,
    pub reveal2: Vec<Revealed>,
    pub hist1: Hist,
    pub hist2: Hist,
    names1: Vec<String>,
    names2: Vec<String>,
}

impl WorldCtx {
    pub fn new(reveal1: Vec<Revealed>, reveal2: Vec<Revealed>, hist1: Hist, hist2: Hist) -> WorldCtx {
        let names1 = reveal1.iter().map(|x| norm_name(&x.species)).collect();
        let names2 = reveal2.iter().map(|x| norm_name(&x.species)).collect();
        WorldCtx { reveal1, reveal2, hist1, hist2, names1, names2 }
    }
}

pub struct NetSearch {
    worlds: Vec<World>,
    /// per-world reveals / root history (overrides `reveal1`, `reveal2`, `root_hist`)
    per_world: Option<Vec<WorldCtx>>,
    params: Params,
    reveal1: Option<Vec<Revealed>>,
    /// what the opponent has seen of our team, applied to side two's view (None = side two sees all)
    reveal2: Option<Vec<Revealed>>,
    /// normalised species names listed in `reveal1`
    names1: Option<Vec<String>>,
    /// normalised species names listed in `reveal2`
    names2: Option<Vec<String>>,
    /// `stats` = 1: side one's statistics per information set and the index of a key
    shared: Vec<Shared>,
    shared_ix: HashMap<u64, u32>,
    /// the root's event rows from side one's view (`hist` > 0)
    root_hist: Hist,
    rng: StdRng,
    inflight: VecDeque<Vec<PendingLeaf>>,
    cursor: usize,
    pub stats: Stats,
}

/// Decoupled PUCT for one side. `node_mean` is the node's mean value from this side's view
/// (the first-play value of an unvisited option is `node_mean - fpu`). A virtual loss counts
/// as a visit that scored 0.
pub fn puct_pick(opts: &[Opt], node_visits: u32, node_mean: f32, p: &Params) -> usize {
    puct_by(opts.len(), |i| (opts[i].prior, opts[i].visits, opts[i].vloss, opts[i].total), node_visits, node_mean, p)
}

/// `(prior, visits, virtual losses, total)` of an option.
type Stat = (f32, u32, u32, f32);

fn puct_by(n: usize, stat: impl Fn(usize) -> Stat, node_visits: u32, node_mean: f32, p: &Params) -> usize {
    if n == 1 {
        return 0;
    }
    let pending: u32 = (0..n).map(|i| stat(i).2).sum();
    let sqrt_n = ((node_visits + pending).max(1) as f32).sqrt();
    let fpu_q = node_mean - p.fpu;
    let mut best = f32::MIN;
    let mut choice = 0;
    for i in 0..n {
        let (prior, visits, vloss, total) = stat(i);
        let n_eff = visits + vloss;
        let q = if n_eff == 0 { fpu_q } else { total / n_eff as f32 };
        let score = q + p.c_puct * prior * sqrt_n / (1.0 + n_eff as f32);
        if score > best {
            best = score;
            choice = i;
        }
    }
    choice
}

/// Regret-matching weights (`select` = 1): `max(Q - V, 0) + c_puct P / sqrt(N)`. An option nobody tried has
/// no regret; a virtual loss counts as a visit that scored 0.
pub fn rm_weights(n: usize, stat: impl Fn(usize) -> Stat, node_visits: u32, node_mean: f32, p: &Params) -> Vec<f32> {
    let pending: u32 = (0..n).map(|i| stat(i).2).sum();
    let explore = p.c_puct / ((node_visits + pending).max(1) as f32).sqrt();
    (0..n)
        .map(|i| {
            let (prior, visits, vloss, total) = stat(i);
            let n_eff = visits + vloss;
            let regret = if n_eff == 0 { 0.0 } else { (total / n_eff as f32 - node_mean).max(0.0) };
            regret + explore * prior
        })
        .collect()
}

/// An index drawn in proportion to the weights (the last positive one on rounding; 0 if none is positive).
fn sample(w: &[f32], rng: &mut StdRng) -> usize {
    let total: f32 = w.iter().map(|x| x.max(0.0)).sum();
    if w.len() <= 1 || !(total > 0.0) || !total.is_finite() {
        return 0;
    }
    let mut th = rng.random_range(0.0..total);
    let mut last = 0;
    for (i, x) in w.iter().enumerate() {
        if *x > 0.0 {
            last = i;
            th -= *x;
            if th <= 0.0 {
                return i;
            }
        }
    }
    last
}

fn select_by(n: usize, stat: impl Fn(usize) -> Stat, node_visits: u32, node_mean: f32, p: &Params, rng: &mut StdRng) -> usize {
    if n == 1 {
        0
    } else if p.select == 1 {
        sample(&rm_weights(n, stat, node_visits, node_mean, p), rng)
    } else {
        puct_by(n, stat, node_visits, node_mean, p)
    }
}

/// Key of an information set: the viewer's scrubbed tokens with every continuous value in tenths (hp in
/// ten buckets), so states that differ only in what the viewer cannot see, or by a damage roll, coincide.
pub fn view_key(cat: &[i16], cont: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in cat {
        h = (h ^ c as u16 as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &f in cont {
        h = (h ^ ((f * 10.0).ceil() as i32 as u32 as u64)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^= h >> 31;
    h = h.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^ (h >> 29)
}

fn make_opts(side: &crate::state::Side, choices: Vec<MoveChoice>) -> Vec<Opt> {
    let k = choices.len().max(1) as f32;
    choices
        .into_iter()
        .map(|c| {
            let slot = action_slot(side, &c).map(|s| s as u8);
            Opt { choice: c, slot, prior: 1.0 / k, visits: 0, vloss: 0, total: 0.0, pred: 1.0 }
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
            per_world: None,
            params,
            names1: reveal1.as_ref().map(|r| r.iter().map(|x| norm_name(&x.species)).collect()),
            reveal1,
            reveal2: None,
            names2: None,
            shared: Vec::new(),
            shared_ix: HashMap::new(),
            root_hist: Hist::default(),
            rng: StdRng::seed_from_u64(seed),
            inflight: VecDeque::new(),
            cursor: 0,
            stats: Stats::default(),
        }
    }

    /// Side two's token view is scrubbed to `rows` (what the opponent knows about our team).
    pub fn set_reveal2(&mut self, rows: Option<Vec<Revealed>>) {
        self.names2 = rows.as_ref().map(|r| r.iter().map(|x| norm_name(&x.species)).collect());
        self.reveal2 = rows;
    }

    /// One context per world (self-play: the worlds are different games).
    pub fn set_world_ctx(&mut self, ctx: Vec<WorldCtx>) {
        assert_eq!(ctx.len(), self.worlds.len(), "one WorldCtx per world");
        self.per_world = Some(ctx);
    }

    /// The root options of world `wi` (both sides).
    pub fn root_opts(&self, wi: usize) -> (&[Opt], &[Opt]) {
        let n = &self.worlds[wi].nodes[0];
        (&n.s1, &n.s2)
    }

    /// The root's event history from side one's view (used when `Params::hist` > 0).
    pub fn set_root_hist(&mut self, h: Hist) {
        self.root_hist = h;
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
        let mut leaves: Vec<PendingLeaf> = Vec::new();
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
                if let Some(l) = self.descend(wi) {
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
        let tree_ns = t0.elapsed().as_nanos() as u64;
        let (cat, cont, hist_cat, hist_cont) = self.write_tokens(&mut leaves);
        if n > 0 {
            self.stats.batches += 1;
            self.inflight.push_back(leaves);
        }
        // tree time excludes tokenisation (counted separately)
        self.stats.tree_ns += tree_ns;
        Collected { n, cat, cont, hist_cat, hist_cont }
    }

    /// Writes the token buffers (and event rows) of the leaves, on up to `params.threads` threads.
    #[allow(clippy::type_complexity)]
    fn write_tokens(&mut self, leaves: &mut [PendingLeaf]) -> (Vec<i16>, Vec<f32>, Vec<i16>, Vec<f32>) {
        let t1 = Instant::now();
        let n = leaves.len();
        let views = self.params.views.clamp(1, 2) as usize;
        let hist_mode = self.params.hist.min(2);
        let hv = if hist_mode > 0 { n * views } else { 0 };
        let mut cat = vec![0i16; n * views * CAT_LEN];
        let mut cont = vec![0f32; n * views * CONT_LEN];
        let mut hcat = vec![0i16; hv * HIST_CAT_LEN];
        let mut hcont = vec![0f32; hv * HIST_CONT_LEN];
        {
            let ctx = TokCtx {
                views,
                hist_mode,
                reveal1: self.reveal1.as_deref(),
                reveal2: self.reveal2.as_deref(),
                root_hist: &self.root_hist,
                per_world: self.per_world.as_deref(),
                keyed: self.params.stats == 1,
            };
            let threads = (self.params.threads.clamp(1, 4) as usize).min(n.max(1));
            if threads <= 1 || n < 8 {
                fill_chunk(&ctx, leaves, &mut cat, &mut cont, &mut hcat, &mut hcont);
            } else {
                let per = n.div_ceil(threads);
                let ctx = &ctx;
                std::thread::scope(|sc| {
                    let (mut lv, mut c, mut f, mut hc, mut hf) =
                        (&mut *leaves, &mut cat[..], &mut cont[..], &mut hcat[..], &mut hcont[..]);
                    while !lv.is_empty() {
                        let take = per.min(lv.len());
                        let (l, lr) = lv.split_at_mut(take);
                        let (cc, cr) = c.split_at_mut(take * views * CAT_LEN);
                        let (ff, fr) = f.split_at_mut(take * views * CONT_LEN);
                        let hn = if hv > 0 { take * views } else { 0 };
                        let (hcc, hcr) = hc.split_at_mut(hn * HIST_CAT_LEN);
                        let (hff, hfr) = hf.split_at_mut(hn * HIST_CONT_LEN);
                        lv = lr;
                        c = cr;
                        f = fr;
                        hc = hcr;
                        hf = hfr;
                        sc.spawn(move || fill_chunk(ctx, l, cc, ff, hcc, hff));
                    }
                });
            }
        }
        if hist_mode == 2 {
            for l in leaves.iter_mut() {
                if let Some(info) = l.info.take() {
                    let node = &mut self.worlds[l.world as usize].nodes[l.node as usize];
                    if node.depth < self.params.depth {
                        node.info = Some(info);
                    }
                }
            }
        }
        if self.params.stats == 1 {
            // a node that will be expanded joins the information set of its side-one view (same depth, same options)
            for l in leaves.iter() {
                let node = &mut self.worlds[l.world as usize].nodes[l.node as usize];
                if node.depth >= self.params.depth || node.s1.is_empty() {
                    continue;
                }
                let mut k = l.key ^ (node.depth as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                for o in &node.s1 {
                    k = (k ^ o.slot.map(|s| s as u64 + 1).unwrap_or(0)).wrapping_mul(0x0000_0100_0000_01b3);
                }
                let next = self.shared.len() as u32;
                let ix = *self.shared_ix.entry(k).or_insert(next);
                if ix == next {
                    self.shared.push(Shared { visits: 0, total: 0.0, opts: vec![SharedOpt::default(); node.s1.len()] });
                    self.stats.info_sets += 1;
                }
                if self.shared[ix as usize].opts.len() == node.s1.len() {
                    node.shared = ix;
                }
            }
        }
        self.stats.tokenise_ns += t1.elapsed().as_nanos() as u64;
        (cat, cont, hcat, hcont)
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
                if let Some(pr) = &ev.pred1 {
                    let at = |o: &Opt| o.slot.map(|s| pr[s as usize]).filter(|x| x.is_finite() && *x > 0.0).unwrap_or(0.0);
                    let sum: f32 = node.s1.iter().map(at).sum();
                    if sum > 0.0 {
                        for o in node.s1.iter_mut() {
                            o.pred = at(o) / sum;
                        }
                    }
                }
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
        let shared = &mut self.shared;
        nodes[leaf as usize].visits += 1;
        nodes[leaf as usize].total += value;
        if nodes[leaf as usize].shared != NONE {
            let sh = &mut shared[nodes[leaf as usize].shared as usize];
            sh.visits += 1;
            sh.total += value;
        }
        for st in path.iter().rev() {
            let n = &mut nodes[st.node as usize];
            n.visits += 1;
            n.total += value;
            if n.shared != NONE {
                let sh = &mut shared[n.shared as usize];
                sh.visits += 1;
                sh.total += value;
                let o = &mut sh.opts[st.a1 as usize];
                o.visits += 1;
                o.total += value;
                o.vloss -= 1;
            }
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
            if n.shared != NONE {
                self.shared[n.shared as usize].opts[st.a1 as usize].vloss -= 1;
            }
            n.s1[st.a1 as usize].vloss -= 1;
            n.s2[st.a2 as usize].vloss -= 1;
        }
    }

    /// One descent in world `wi`. Returns a pending leaf (its tokens appended to `cat` /
    /// `cont`), or None when the descent backed up a known value or collided with a pending
    /// node.
    fn descend(&mut self, wi: usize) -> Option<PendingLeaf> {
        let params = self.params.clone();
        let mut state = self.worlds[wi].root_state.clone();
        let mut path: Vec<Step> = Vec::new();
        let mut cur: u32 = 0;
        // hist mode 2: (node, our action, the opponent's) of the joint action last applied
        let mut last_act: Option<(u32, ActRow, ActRow)> = None;
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
                    // an unevaluated node: queue it for the net (its tokens are written after the descents)
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
                    let parent = match last_act {
                        Some((pn, own, opp)) if params.hist == 2 => self.worlds[wi].nodes[pn as usize]
                            .info
                            .as_ref()
                            .map(|i| (i.summary, i.hist.clone(), own, opp)),
                        _ => None,
                    };
                    return Some(PendingLeaf { world: wi as u32, node: cur, path, state: Some(state), parent, info: None, key: 0 });
                }
                Status::Ready => {
                    let (a1, a2, depth, n2, both_none) = {
                        let node = &self.worlds[wi].nodes[cur as usize];
                        let mean = if node.visits > 0 { node.total / node.visits as f32 } else { node.value };
                        let (k1, k2) = (node.s1.len(), node.s2.len());
                        let a1 = if node.shared != NONE {
                            let sh = &self.shared[node.shared as usize];
                            let m = if sh.visits > 0 { sh.total / sh.visits as f32 } else { node.value };
                            let st = |i: usize| (node.s1[i].prior, sh.opts[i].visits, sh.opts[i].vloss, sh.opts[i].total);
                            select_by(k1, st, sh.visits, m, &params, &mut self.rng)
                        } else {
                            let st = |i: usize| (node.s1[i].prior, node.s1[i].visits, node.s1[i].vloss, node.s1[i].total);
                            select_by(k1, st, node.visits, mean, &params, &mut self.rng)
                        };
                        let a2 = if node.cut {
                            let w: Vec<f32> = node.s2.iter().map(|o| o.prior).collect();
                            sample(&w, &mut self.rng)
                        } else {
                            let st = |i: usize| (node.s2[i].prior, node.s2[i].visits, node.s2[i].vloss, node.s2[i].total);
                            select_by(k2, st, node.visits, 1.0 - mean, &params, &mut self.rng)
                        };
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
                        if node.shared != NONE {
                            self.shared[node.shared as usize].opts[a1].vloss += 1;
                        }
                    }
                    path.push(Step { node: cur, a1: a1 as u16, a2: a2 as u16 });
                    let key = a1 * n2 + a2;
                    if params.hist == 2 {
                        let node = &self.worlds[wi].nodes[cur as usize];
                        let names = match &self.per_world {
                            Some(v) => Some(v[wi].names1.as_slice()),
                            None => self.names1.as_deref(),
                        };
                        let hidden_opp = |p: &Pokemon| match names {
                            Some(v) => !v.contains(&norm_name(PokemonName::NAMES[p.id as usize])),
                            None => false,
                        };
                        let own = act_row(&state.side_one, &node.s1[a1].choice, &|_| false);
                        let opp = act_row(&state.side_two, &node.s2[a2].choice, &hidden_opp);
                        last_act = Some((cur, own, opp));
                    }
                    if self.worlds[wi].nodes[cur as usize].chance[key].is_none() {
                        let (c1, c2) = {
                            let node = &self.worlds[wi].nodes[cur as usize];
                            (node.s1[a1].choice.clone(), node.s2[a2].choice.clone())
                        };
                        // side two stops searching below an action of ours it could not expect
                        let child_cut = params.cut > 0.0 && {
                            let node = &self.worlds[wi].nodes[cur as usize];
                            let names = match &self.per_world {
                                Some(v) => Some(v[wi].names2.as_slice()),
                                None => self.names2.as_deref(),
                            };
                            let unseen = match (&c1, names) {
                                (MoveChoice::Switch(p), Some(v)) => {
                                    let target = &state.side_one.pokemon.pkmn[*p as usize];
                                    !v.contains(&norm_name(PokemonName::NAMES[target.id as usize]))
                                }
                                _ => false,
                            };
                            node.cut || unseen || node.s1[a1].pred < params.cut
                        };
                        let mut list = generate_instructions_from_move_pair(&mut state, &c1, &c2, depth <= 1);
                        if list.is_empty() {
                            list.push(StateInstructions::default());
                        }
                        let w = &mut self.worlds[wi];
                        let mut outs = Vec::with_capacity(list.len());
                        for instr in list.drain(..) {
                            let child = w.nodes.len() as u32;
                            let mut child_node = Node::new(cur, depth + 1);
                            child_node.cut = child_cut;
                            self.stats.cut_nodes += child_cut as u64;
                            w.nodes.push(child_node);
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

/// Read-only inputs of the token writer, shared by its threads.
struct TokCtx<'a> {
    views: usize,
    hist_mode: u8,
    reveal1: Option<&'a [Revealed]>,
    reveal2: Option<&'a [Revealed]>,
    root_hist: &'a Hist,
    per_world: Option<&'a [WorldCtx]>,
    keyed: bool,
}

/// Writes the tokens (and event rows) of a chunk of leaves into their slices of the batch buffers.
fn fill_chunk(ctx: &TokCtx, leaves: &mut [PendingLeaf], cat: &mut [i16], cont: &mut [f32], hcat: &mut [i16], hcont: &mut [f32]) {
    let v = ctx.views;
    for (i, leaf) in leaves.iter_mut().enumerate() {
        let state = leaf.state.take().expect("pending leaf without a state");
        let (c, f) = (&mut cat[i * v * CAT_LEN..(i + 1) * v * CAT_LEN], &mut cont[i * v * CONT_LEN..(i + 1) * v * CONT_LEN]);
        let pw = ctx.per_world.map(|w| &w[leaf.world as usize]);
        let (r1, r2) = match pw {
            Some(w) => (Some(w.reveal1.as_slice()), Some(w.reveal2.as_slice())),
            None => (ctx.reveal1, ctx.reveal2),
        };
        let root1 = pw.map(|w| &w.hist1).unwrap_or(ctx.root_hist);
        tokens::state_tokens_into(&state, SideReference::SideOne, r1, &mut c[..CAT_LEN], &mut f[..CONT_LEN]);
        if v == 2 {
            tokens::state_tokens_into(&state, SideReference::SideTwo, r2, &mut c[CAT_LEN..], &mut f[CONT_LEN..]);
        }
        if ctx.keyed {
            leaf.key = view_key(&c[..CAT_LEN], &f[..CONT_LEN]);
        }
        if ctx.hist_mode == 0 {
            continue;
        }
        // side two's rows: its own root history when the world has one and the leaf still shows the root's rows
        let own2 = match pw {
            Some(w) if ctx.hist_mode == 1 || leaf.parent.is_none() => Some(&w.hist2),
            _ => None,
        };
        let (hist, info) = if ctx.hist_mode == 2 {
            let summary = Summary::of(&c[..CAT_LEN], &f[..CONT_LEN]);
            let h = match &leaf.parent {
                Some((ps, ph, own, opp)) => ph.pushed(&turn_rows(ps, &summary, *own, *opp)),
                None => root1.clone(),
            };
            (h.clone(), Some(Box::new(NodeInfo { summary, hist: h })))
        } else {
            (root1.clone(), None)
        };
        leaf.info = info;
        let (hc, hf) = (
            &mut hcat[i * v * HIST_CAT_LEN..(i + 1) * v * HIST_CAT_LEN],
            &mut hcont[i * v * HIST_CONT_LEN..(i + 1) * v * HIST_CONT_LEN],
        );
        hc[..HIST_CAT_LEN].copy_from_slice(&hist.cat);
        hf[..HIST_CONT_LEN].copy_from_slice(&hist.cont);
        if v == 2 {
            let m = match own2 {
                Some(h) => h.clone(),
                None => hist.mirrored(),
            };
            hc[HIST_CAT_LEN..].copy_from_slice(&m.cat);
            hf[HIST_CONT_LEN..].copy_from_slice(&m.cont);
        }
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
        Opt { choice: MoveChoice::None, slot: None, prior, visits, vloss, total, pred: 1.0 }
    }

    fn fake(n: usize, p1: [f32; N_ACTIONS], p2: [f32; N_ACTIONS], v: f32) -> Vec<Eval> {
        (0..n).map(|_| Eval { policy1: p1, policy2: p2, value: v, pred1: None }).collect()
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
            Opt { choice: MoveChoice::None, slot: Some(0), prior: 0.0, visits: 0, vloss: 0, total: 0.0, pred: 1.0 },
            Opt { choice: MoveChoice::None, slot: Some(9), prior: 0.0, visits: 0, vloss: 0, total: 0.0, pred: 1.0 },
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

    // ---------------------------------------------------------------- sdb.25: threads, history, reveal2

    fn collect_all(ns: &mut NetSearch, batch: usize, rounds: usize) -> Vec<(usize, Vec<i16>, Vec<f32>, Vec<i16>, Vec<f32>)> {
        let mut out = Vec::new();
        for _ in 0..rounds {
            let c = ns.collect(batch);
            if c.n == 0 {
                break;
            }
            ns.feed(&fake(c.n, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
            out.push((c.n, c.cat, c.cont, c.hist_cat, c.hist_cont));
        }
        out
    }

    #[test]
    fn threaded_tokenisation_is_identical() {
        let st = states();
        let worlds: Vec<State> = st.iter().take(8).cloned().collect();
        let rev = vec![Revealed { species: "x".into(), ..Default::default() }];
        let mut a = NetSearch::new(worlds.clone(), Params::default(), Some(rev.clone()), 21);
        let mut b = NetSearch::new(worlds, Params { threads: 3, hist: 2, ..Params::default() }, Some(rev), 21);
        let ra = collect_all(&mut a, 40, 6);
        let rb = collect_all(&mut b, 40, 6);
        assert_eq!(ra.len(), rb.len());
        for (x, y) in ra.iter().zip(&rb) {
            assert_eq!(x.0, y.0);
            assert!(x.1 == y.1 && x.2 == y.2, "tokens must not depend on the number of threads");
        }
        assert!(ra.iter().map(|r| r.0).sum::<usize>() > 100);
    }

    #[test]
    fn root_history_is_given_to_every_leaf_and_mirrored() {
        let st = states();
        let mut root = Hist::default();
        for r in 0..4 {
            root.cat[r * HIST_CAT_W..(r + 1) * HIST_CAT_W].copy_from_slice(&[1 + (r % 2) as i16, 2, 100 + r as i16, 7]);
            root.cont[r * HIST_CONT_W..(r + 1) * HIST_CONT_W].copy_from_slice(&[0.0625 * (1 + r / 2) as f32, -0.25, 0.5, 0.0, 0.1667]);
        }
        let mut ns = NetSearch::new(vec![st[0].clone(); 2], Params { hist: 1, ..Params::default() }, None, 22);
        ns.set_root_hist(root.clone());
        let rounds = collect_all(&mut ns, 16, 4);
        let m = root.mirrored();
        for (n, _, _, hc, hf) in &rounds {
            assert_eq!(hc.len(), n * 2 * HIST_CAT_LEN);
            for i in 0..*n {
                assert_eq!(hc[i * 2 * HIST_CAT_LEN..i * 2 * HIST_CAT_LEN + HIST_CAT_LEN], root.cat);
                assert_eq!(hc[(i * 2 + 1) * HIST_CAT_LEN..(i * 2 + 2) * HIST_CAT_LEN], m.cat);
                assert_eq!(hf[(i * 2 + 1) * HIST_CONT_LEN..(i * 2 + 2) * HIST_CONT_LEN], m.cont);
            }
        }
        // mirrored: the two rows of a turn swap, side ids flip, the actor-relative numbers stay with the row
        assert_eq!(m.cat[0..4], [1, 2, 101, 7]);
        assert_eq!(m.cat[4..8], [2, 2, 100, 7]);
        assert_eq!(m.mirrored().cat, root.cat);
    }

    #[test]
    fn tree_history_pushes_simulated_turns() {
        let st = states();
        let mut root = Hist::default();
        root.cat[0..4].copy_from_slice(&[1, 1, 55, 9]);
        root.cont[0..5].copy_from_slice(&[0.0625, -0.1, -0.2, 0.0, 0.0]);
        root.cat[60..64].copy_from_slice(&[2, 3, 44, 8]); // the oldest row
        root.cont[75..80].copy_from_slice(&[0.9375, 0.0, 0.0, 0.0, 0.0]);
        let mut ns = NetSearch::new(vec![st[0].clone()], Params { hist: 2, ..Params::default() }, None, 23);
        ns.set_root_hist(root.clone());
        let r0 = ns.collect(4); // the root alone: its own rows are the root's
        assert_eq!(r0.n, 1);
        assert_eq!(r0.hist_cat[..HIST_CAT_LEN], root.cat);
        ns.feed(&fake(1, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
        let c = ns.collect(16); // depth-1 leaves
        assert!(c.n > 3);
        for i in 0..c.n {
            let hc = &c.hist_cat[i * 2 * HIST_CAT_LEN..i * 2 * HIST_CAT_LEN + HIST_CAT_LEN];
            let hf = &c.hist_cont[i * 2 * HIST_CONT_LEN..i * 2 * HIST_CONT_LEN + HIST_CONT_LEN];
            // two new rows in front: our action (side 1) and the opponent's (side 2), one turn ago
            assert_eq!(hc[0], 1);
            assert_eq!(hc[4], 2);
            assert!(hc[1] >= 0 && hc[1] <= 3 && hc[5] >= 0 && hc[5] <= 3);
            assert_eq!(hf[0], 0.0625);
            assert_eq!(hf[5], 0.0625);
            // the active Pokemon of the root, as the summary of the root's tokens says
            assert!(hc[3] > 1 && hc[7] > 1, "{:?}", &hc[..12]);
            // the root's rows moved down by two rows and got one turn older; the oldest pair fell off
            assert_eq!(hc[8..12], [1, 1, 55, 9]);
            assert!((hf[10] - 0.125).abs() < 1e-6 && (hf[11] + 0.1).abs() < 1e-6);
            assert_eq!(hc[60..64], [0, 0, 0, 0]);
        }
        // a switch is kind 3 with the species row, a move kind 1 / 2 with the move row
        assert!((0..c.n).any(|i| c.hist_cat[i * 2 * HIST_CAT_LEN + 1] == 3 || c.hist_cat[i * 2 * HIST_CAT_LEN + 1] == 1));
        ns.feed(&fake(c.n, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
        // depth 2: four new-ish rows, the depth-1 turn is one turn older
        let mut seen2 = false;
        for _ in 0..30 {
            let c2 = ns.collect(64);
            for i in 0..c2.n {
                let hf = &c2.hist_cont[i * 2 * HIST_CONT_LEN..i * 2 * HIST_CONT_LEN + HIST_CONT_LEN];
                let hc = &c2.hist_cat[i * 2 * HIST_CAT_LEN..i * 2 * HIST_CAT_LEN + HIST_CAT_LEN];
                if hc[16] > 0 && hc[24] == 0 && (hf[5] - 0.0625).abs() < 1e-6 && (hf[10] - 0.125).abs() < 1e-6 {
                    seen2 = true;
                    assert_eq!(hc[16..20], [1, 1, 55, 9]);
                    assert!((hf[20] - 0.1875).abs() < 1e-6);
                }
            }
            if c2.n == 0 {
                break;
            }
            ns.feed(&fake(c2.n, [1.0; N_ACTIONS], [1.0; N_ACTIONS], 0.5)).unwrap();
        }
        assert!(seen2, "depth-2 leaves must carry two simulated turns plus the root rows");
    }

    #[test]
    fn history_summary_deltas() {
        let mut a = Summary { active: [10, 20], ..Summary::default() };
        a.n = [2, 2];
        a.sp = [[10, 11, 0, 0, 0, 0], [20, 21, 0, 0, 0, 0]];
        a.hp = [[1.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.8, 1.0, 0.0, 0.0, 0.0, 0.0]];
        let mut b = a;
        b.hp[0][0] = 0.75; // our active lost 0.25
        b.n[1] = 1; // the opponent's second Pokemon fainted
        b.sp[1][1] = 0;
        let rows = turn_rows(&a, &b, (0, 300), (-1, 0));
        assert_eq!(rows[0].0, [1, 1, 300, 10]);
        assert_eq!(rows[1].0, [2, 0, 0, 20]);
        assert!((rows[0].1[1] + 0.25).abs() < 1e-6 && rows[0].1[2].abs() < 1e-6);
        assert!(rows[1].1[1].abs() < 1e-6 && (rows[1].1[2] + 0.25).abs() < 1e-6);
        assert!((rows[0].1[4] - 1.0 / 6.0).abs() < 1e-6 && (rows[1].1[3] - 1.0 / 6.0).abs() < 1e-6);
        assert_eq!(rows[0].1[0], 0.0625);
    }

    #[test]
    fn reveal2_scrubs_side_twos_view_of_our_team() {
        let st = states();
        let mut all = NetSearch::new(vec![st[3].clone()], Params::default(), None, 24);
        let mut scr = NetSearch::new(vec![st[3].clone()], Params::default(), None, 24);
        scr.set_reveal2(Some(Vec::new())); // the opponent has seen nothing of our team
        let a = all.collect(4);
        let b = scr.collect(4);
        // side one's view is identical, side two's is not
        assert_eq!(a.cat[..CAT_LEN], b.cat[..CAT_LEN]);
        assert_ne!(a.cont[CONT_LEN..], b.cont[CONT_LEN..]);
        assert_eq!(b.cat[CAT_LEN + tokens::O_META + 2], 1, "meta[2] marks a scrubbed view");
        assert_eq!(a.cat[CAT_LEN + tokens::O_META + 2], 0);
        // every non-active member of our team is hidden from side two's view: all 'known' flags 0
        let known = &b.cat[CAT_LEN + tokens::O_MON_KNOWN..CAT_LEN + tokens::O_MON_KNOWN + 12 * tokens::KNOWN_W];
        let opp_half = &known[6 * tokens::KNOWN_W..];
        assert!(opp_half[tokens::KNOWN_W..].iter().all(|&k| k == 0), "bench of ours unknown to the opponent");
    }

    fn uniform() -> [f32; N_ACTIONS] {
        [1.0; N_ACTIONS]
    }

    /// every information set's counters are the sums of its member nodes' counters, and no virtual loss is left
    fn check_shared(ns: &NetSearch) {
        let mut sums: Vec<Vec<(u32, f32)>> = ns.shared.iter().map(|s| vec![(0, 0.0); s.opts.len()]).collect();
        let mut visits = vec![0u32; ns.shared.len()];
        for w in &ns.worlds {
            for n in &w.nodes {
                if n.shared == NONE {
                    continue;
                }
                visits[n.shared as usize] += n.visits;
                for (i, o) in n.s1.iter().enumerate() {
                    assert_eq!(o.vloss, 0);
                    sums[n.shared as usize][i].0 += o.visits;
                    sums[n.shared as usize][i].1 += o.total;
                }
            }
        }
        for (k, sh) in ns.shared.iter().enumerate() {
            assert_eq!(sh.visits, visits[k]);
            for (i, o) in sh.opts.iter().enumerate() {
                assert_eq!(o.vloss, 0);
                assert_eq!(o.visits, sums[k][i].0);
                assert!((o.total - sums[k][i].1).abs() < 1e-2 * (1.0 + o.total.abs()));
            }
        }
    }

    #[test]
    fn worlds_that_look_the_same_share_side_one_statistics() {
        let st = states();
        // two worlds of one battle: they differ only in what side one cannot see (the active's item)
        let a = st[0].clone();
        let mut b = st[0].clone();
        let act = b.side_two.active_index as usize;
        let other = (act + 1) % 6;
        let item = b.side_two.pokemon.pkmn[other].item;
        b.side_two.pokemon.pkmn[act].item = item;
        let rev = vec![Revealed { species: "x".into(), ..Default::default() }];
        let p = Params { stats: 1, ..Params::default() };
        let mut ns = NetSearch::new(vec![a.clone(), b.clone(), a, b], p.clone(), Some(rev.clone()), 31);
        let c = ns.collect(8);
        assert_eq!(c.n, 4);
        assert_eq!(ns.stats.info_sets, 1, "the four roots are one information set");
        ns.feed(&fake(4, uniform(), uniform(), 0.5)).unwrap();
        run(&mut ns, 16, 40, uniform(), uniform(), 0.5);
        check_shared(&ns);
        let root = &ns.shared[ns.worlds[0].nodes[0].shared as usize];
        let total: u32 = (0..4).map(|w| ns.root_visits(w)).sum();
        assert_eq!(root.visits, total);
        assert!(ns.stats.info_sets > 1 && ns.stats.info_sets < ns.stats.nodes, "interior nodes share too");
        // different battles do not share a root
        let mut two = NetSearch::new(vec![st[0].clone(), st[1].clone()], p, Some(rev), 32);
        two.collect(8);
        assert_eq!(two.stats.info_sets, 2);
        // stats = 0 keeps no shared statistics
        let mut off = NetSearch::new(vec![st[0].clone(); 2], Params::default(), None, 33);
        run(&mut off, 8, 10, uniform(), uniform(), 0.5);
        assert!(off.shared.is_empty() && off.stats.info_sets == 0);
    }

    #[test]
    fn shared_statistics_spread_a_batch_over_our_options() {
        // the virtual loss of one world's descent diverts the next world's
        let st = states();
        let p = Params { stats: 1, ..Params::default() };
        let mut ns = NetSearch::new(vec![st[0].clone(); 6], p, Some(Vec::new()), 34);
        let c = ns.collect(8);
        ns.feed(&fake(c.n, uniform(), uniform(), 0.5)).unwrap();
        let c = ns.collect(6);
        assert_eq!(c.n, 6);
        let picked: Vec<usize> =
            (0..6).map(|w| ns.worlds[w].nodes[0].s1.iter().position(|o| o.vloss > 0).unwrap()).collect();
        let mut distinct = picked.clone();
        distinct.sort();
        distinct.dedup();
        assert!(distinct.len() > 1, "six worlds, one information set: {picked:?}");
        ns.feed(&fake(6, uniform(), uniform(), 0.5)).unwrap();
        check_shared(&ns);
    }

    #[test]
    fn regret_matching_weights() {
        let p = Params { c_puct: 2.0, select: 1, ..Params::default() };
        let opts = [opt(0.5, 10, 0, 7.0), opt(0.3, 10, 0, 4.0), opt(0.2, 0, 0, 0.0)];
        let st = |i: usize| (opts[i].prior, opts[i].visits, opts[i].vloss, opts[i].total);
        let w = rm_weights(3, st, 100, 0.5, &p);
        // explore = 2 / sqrt(100) = 0.2: regret 0.2 + 0.1, 0 + 0.06, 0 + 0.04
        assert!((w[0] - 0.3).abs() < 1e-6 && (w[1] - 0.06).abs() < 1e-6 && (w[2] - 0.04).abs() < 1e-6, "{w:?}");
        let mut rng = StdRng::seed_from_u64(1);
        let mut n = [0u32; 3];
        for _ in 0..4000 {
            n[sample(&w, &mut rng)] += 1;
        }
        assert!((n[0] as f32 / 4000.0 - 0.75).abs() < 0.03 && n[2] > 0, "{n:?}");
        assert_eq!(sample(&[0.0, 0.0], &mut rng), 0);
    }

    #[test]
    fn regret_matching_search_runs_and_is_deterministic() {
        let st = states();
        let p = Params { stats: 1, select: 1, ..Params::default() };
        let go = |seed: u64| {
            let mut ns = NetSearch::new(vec![st[2].clone(); 4], p.clone(), Some(Vec::new()), seed);
            run(&mut ns, 16, 40, uniform(), uniform(), 0.4);
            check_shared(&ns);
            assert!(ns.stats.evals > 200);
            (0..4).map(|w| ns.root(w).0.iter().map(|r| r.1).collect::<Vec<u32>>()).collect::<Vec<_>>()
        };
        assert_eq!(go(5), go(5));
        assert_ne!(go(5), go(6));
    }

    #[test]
    fn opponent_search_is_cut_after_an_unexpected_action() {
        let st = states();
        let p = Params { cut: 0.65, ..Params::default() };
        let feed_all = |ns: &mut NetSearch, pred: Option<[f32; N_ACTIONS]>| {
            for _ in 0..30 {
                let c = ns.collect(16);
                if c.n == 0 {
                    break;
                }
                let ev: Vec<Eval> =
                    (0..c.n).map(|_| Eval { policy1: uniform(), policy2: uniform(), value: 0.5, pred1: pred }).collect();
                ns.feed(&ev).unwrap();
            }
        };
        // nothing fed, the opponent sees our whole team: nothing is unexpected
        let mut none = NetSearch::new(vec![st[0].clone()], p.clone(), None, 41);
        feed_all(&mut none, None);
        assert_eq!(none.stats.cut_nodes, 0);
        // a uniform prediction over several options is below 0.65 for each: everything below the root is cut
        let mut all = NetSearch::new(vec![st[0].clone()], p.clone(), None, 41);
        feed_all(&mut all, Some(uniform()));
        assert!(all.stats.cut_nodes > 0);
        let w = &all.worlds[0];
        assert!(!w.nodes[0].cut);
        if w.nodes[0].s1.len() > 1 {
            assert!(w.nodes.iter().skip(1).all(|n| n.cut));
        }
        // the opponent has seen none of our bench: only switches are unexpected
        let mut sw = NetSearch::new(vec![st[0].clone()], p, None, 41);
        sw.set_reveal2(Some(Vec::new()));
        feed_all(&mut sw, None);
        let w = &sw.worlds[0];
        let root = &w.nodes[0];
        for (i, o) in root.s1.iter().enumerate() {
            for a2 in 0..root.s2.len() {
                if let Some(outs) = &root.chance[i * root.s2.len() + a2] {
                    let is_switch = matches!(o.choice, MoveChoice::Switch(_));
                    assert!(outs.iter().all(|x| w.nodes[x.child as usize].cut == is_switch));
                }
            }
        }
        assert!(sw.stats.cut_nodes > 0);
    }

    /// Self-play of the selection rule on a one-shot matrix game (side one's payoff, in [0, 1]): each side's
    /// visit shares and how much a best response gains against them (0 at an equilibrium).
    fn matrix_selfplay(pay: &[&[f32]], p: &Params, iters: u32, seed: u64) -> (Vec<f32>, Vec<f32>, f32) {
        let (k1, k2) = (pay.len(), pay[0].len());
        let mut s1 = vec![(0u32, 0.0f64); k1];
        let mut s2 = vec![(0u32, 0.0f64); k2];
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = 0.0f64;
        for n in 0..iters {
            let mean = if n > 0 { (total / n as f64) as f32 } else { 0.5 };
            let a1 = select_by(k1, |i| (1.0 / k1 as f32, s1[i].0, 0, s1[i].1 as f32), n, mean, p, &mut rng);
            let a2 = select_by(k2, |i| (1.0 / k2 as f32, s2[i].0, 0, s2[i].1 as f32), n, 1.0 - mean, p, &mut rng);
            let v = pay[a1][a2] as f64;
            total += v;
            s1[a1].0 += 1;
            s1[a1].1 += v;
            s2[a2].0 += 1;
            s2[a2].1 += 1.0 - v;
        }
        let x1: Vec<f32> = s1.iter().map(|s| s.0 as f32 / iters as f32).collect();
        let x2: Vec<f32> = s2.iter().map(|s| s.0 as f32 / iters as f32).collect();
        let best1 = (0..k1).map(|a| (0..k2).map(|b| pay[a][b] * x2[b]).sum::<f32>()).fold(f32::MIN, f32::max);
        let best2 = (0..k2).map(|b| (0..k1).map(|a| pay[a][b] * x1[a]).sum::<f32>()).fold(f32::MAX, f32::min);
        (x1, x2, best1 - best2)
    }

    /// Not an assertion: prints how each selection rule behaves on two matrix games with a known mixed answer.
    /// `cargo test --lib matrix_game_report -- --nocapture --ignored`
    #[test]
    #[ignore]
    fn matrix_game_report() {
        // asymmetric matching pennies: equilibrium (1/3, 2/3) for both sides, value 1/3
        let pennies: [&[f32]; 2] = [&[1.0, 0.0], &[0.0, 0.5]];
        // rock-paper-scissors where rock beating scissors pays double (0.5 = draw): equilibrium (1/4, 1/2, 1/4)
        let rps: [&[f32]; 3] = [&[0.5, 0.25, 1.0], &[0.75, 0.5, 0.25], &[0.0, 0.75, 0.5]];
        // one option dominates: the answer is pure (option 0 for side one)
        let pure: [&[f32]; 3] = [&[0.7, 0.6, 0.65], &[0.5, 0.4, 0.6], &[0.3, 0.55, 0.2]];
        for (name, g) in [("pennies", &pennies[..]), ("rps2", &rps[..]), ("pure", &pure[..])] {
            for select in [0u8, 1] {
                for c in [0.5f32, 1.0, 2.0, 4.0] {
                    for iters in [100u32, 1000, 10000, 100000] {
                        let p = Params { c_puct: c, select, ..Params::default() };
                        let mut gap = 0.0;
                        let mut x = (Vec::new(), Vec::new());
                        for seed in 0..8 {
                            let (x1, x2, g1) = matrix_selfplay(g, &p, iters, seed);
                            gap += g1 / 8.0;
                            x = (x1, x2);
                        }
                        println!("{name} select={select} c={c} n={iters} gap={gap:.3} x1={:.2?} x2={:.2?}", x.0, x.1);
                    }
                }
            }
        }
    }
}
