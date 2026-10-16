//! Learned leaf evaluators for MCTS (linear and 1-hidden-layer ReLU MLP).
//!
//! An `Evaluator` maps `features(state, SideOne)` to a single logit
//! z = logit(P(side one wins)). The MCTS leaf score is then
//!
//! * relative (default): `sigmoid(z_leaf - z_root)` (same shape as the stock
//!   `sigmoid(0.0125 * (evaluate(leaf) - evaluate(root)))`, but z is already in
//!   log-odds so no 0.0125 scale), or
//! * absolute: `sigmoid(z_leaf)` (AlphaZero style).
//!
//! Both are multiplied by a runtime `leaf_scale` k (default 1.0, not stored in
//! the weights file): `sigmoid(k * (z_leaf - z_root))` / `sigmoid(k * z_leaf)`.
//! A net trained on noisy outcomes has logits that barely move between root
//! and leaf, so k > 1 spreads the scores the way the stock 0.0125 * dEval does.
//!
//! A **residual** evaluator adds the stock evaluation: `z = stock_coef *
//! evaluate(state) + net(features)` (`stock_coef` = a * 0.0125, learned). The
//! net then only has to learn a correction to the stock evaluation.
//!
//! Terminal states keep their exact 0 / 1 value, as in stock.
//!
//! Weights file format (little endian, see `README.md`):
//! ```text
//! magic    4 bytes  "PEVN"
//! format   u32      1
//! feat_ver u32      FEATURE_VERSION used to train
//! kind     u32      1 = linear, 2 = mlp
//! n_in     u32      FEATURE_LEN
//! n_hidden u32      0 for linear
//! linear:  w[n_in] f32, b f32
//! mlp:     W1[n_hidden][n_in] f32 (row-major, PyTorch nn.Linear weight layout),
//!          b1[n_hidden], w2[n_hidden], b2
//! ```
//! Format 2 (written only for residual evaluators; format 1 is still read and
//! written for plain ones) inserts two fields after `n_hidden`:
//! ```text
//! flags      u32  bit0 = residual
//! stock_coef f32  multiplier applied to the raw stock evaluate()
//! ```
//! Format 3 (feature version 4 only: dense features + learned identity embeddings, see
//! `features_v4.rs` / `docs/features-v4.md`; MLP only). The 24-byte header has `feat_ver` = 4,
//! `kind` = 2 and `n_in` = the DENSE length (2379); then:
//! ```text
//! flags      u32      bit0 = residual
//! stock_coef f32      (0 when not residual)
//! vocab_hash u64      features_v4::vocab_hash() of the engine that trained it
//! dims       4 x u32  embedding dims: species, item, ability, move
//! W1[n_hidden][n_in + 12 * sum(dims)] f32 row-major: dense columns, then per slot (own 0..5,
//!            opp 0..5) species | item | ability | mean-pooled move embedding
//! b1[n_hidden], w2[n_hidden], b2
//! tables     species, item, ability, move: f32 row-major [vocab_size(t)][dims[t]]
//! ```

use crate::engine::evaluate::evaluate;
use crate::engine::items::Items;
use crate::engine::state::{PokemonVolatileStatus as V, Terrain, Weather};
use crate::features::{
    feature_len, features_versioned_into, FEATURE_LEN, FEATURE_LEN_V4, FEATURE_VERSION_1,
    FEATURE_VERSION_2, FEATURE_VERSION_3, FEATURE_VERSION_4, PER_POKEMON, PER_SIDE, TEAM_SLOTS,
};
use crate::features_v2::{self, Sink};
use crate::features_v3;
use crate::features_v4::{self, IDS_PER_SLOT, N_IDS, N_SLOTS, N_TABLES};
use crate::state::{PokemonStatus, PokemonType, Side, SideReference, State};
use std::convert::TryInto;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_MODEL_ID: AtomicU64 = AtomicU64::new(1);

pub const MAGIC: &[u8; 4] = b"PEVN";
/// Format written for non-residual evaluators.
pub const FORMAT_VERSION: u32 = 1;
/// Format written for residual evaluators.
pub const FORMAT_VERSION_RESIDUAL: u32 = 2;
/// Format written for evaluators with embedding tables (feature version 4); see the module doc.
pub const FORMAT_VERSION_EMBED: u32 = 3;
const FLAG_RESIDUAL: u32 = 1;
const KIND_LINEAR: u32 = 1;
const KIND_MLP: u32 = 2;

#[derive(Debug, Clone)]
enum Model {
    Linear {
        w: Vec<f32>,
        b: f32,
    },
    Mlp {
        hidden: usize,
        /// transposed first layer: `w1t[i * hidden + j]` = weight from input i to hidden j
        w1t: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
        /// learned identity embeddings (feature version 4 only); the first layer then reads
        /// `n_dense + emb.input_len()` inputs
        emb: Option<Embed>,
    },
}

/// Learned embedding tables of feature version 4: species, item, ability and move identity of
/// the 12 Pokemon slots (see `features_v4.rs`). The four moves of a slot are mean-pooled; the
/// per-slot vectors are concatenated (slot-major: own 0..5, opp 0..5; per slot species, item,
/// ability, pooled move) and appended to the dense features as the input of the first layer.
#[derive(Debug, Clone)]
pub struct Embed {
    dims: [usize; N_TABLES],
    /// `tables[t]` is row-major `[vocab_size(t)][dims[t]]`
    tables: [Vec<f32>; N_TABLES],
}

impl Embed {
    pub fn new(dims: [usize; N_TABLES], tables: [Vec<f32>; N_TABLES]) -> Result<Embed, String> {
        for t in 0..N_TABLES {
            if dims[t] == 0 || dims[t] > 1024 {
                return Err(format!("embedding dim of {} must be in 1..=1024, got {}", features_v4::TABLE_NAMES[t], dims[t]));
            }
            let want = features_v4::vocab_size(t) * dims[t];
            if tables[t].len() != want {
                return Err(format!(
                    "{} table has {} floats, expected vocab {} x dim {} = {}",
                    features_v4::TABLE_NAMES[t],
                    tables[t].len(),
                    features_v4::vocab_size(t),
                    dims[t],
                    want
                ));
            }
        }
        Ok(Embed { dims, tables })
    }

    pub fn dims(&self) -> [usize; N_TABLES] {
        self.dims
    }

    pub fn table(&self, t: usize) -> &[f32] {
        &self.tables[t]
    }

    /// Embedding inputs per Pokemon slot (species + item + ability + pooled move dims).
    pub fn slot_width(&self) -> usize {
        self.dims.iter().sum()
    }

    /// Total embedding inputs of the first layer (`12 * slot_width`).
    pub fn input_len(&self) -> usize {
        N_SLOTS * self.slot_width()
    }

    #[inline(always)]
    fn row(&self, t: usize, idx: u16) -> &[f32] {
        let d = self.dims[t];
        &self.tables[t][idx as usize * d..(idx as usize + 1) * d]
    }

    /// Dense reference input (`input_len()` long) for the identity indices `ids` (`N_IDS` long).
    pub fn fill_inputs(&self, ids: &[u16], out: &mut [f32]) {
        let w = self.slot_width();
        for s in 0..N_SLOTS {
            let ids = &ids[s * IDS_PER_SLOT..(s + 1) * IDS_PER_SLOT];
            let o = &mut out[s * w..(s + 1) * w];
            let mut off = 0;
            for t in 0..3 {
                let d = self.dims[t];
                o[off..off + d].copy_from_slice(self.row(t, ids[t]));
                off += d;
            }
            let d = self.dims[features_v4::MOVE];
            o[off..off + d].fill(0.0);
            for m in 0..4 {
                for (a, b) in o[off..off + d].iter_mut().zip(self.row(features_v4::MOVE, ids[3 + m])) {
                    *a += 0.25 * b;
                }
            }
        }
    }

    /// Adds the first-layer contribution of one slot's identity (`ids`: 7 indices) to `acc`.
    /// `n_dense` is the dense input length (embedding inputs follow it), `pos` the slot
    /// (0..11, own first).
    #[inline(always)]
    fn add_slot<const H: usize>(&self, acc: &mut [f32; H], w1t: &[f32], n_dense: usize, pos: usize, ids: &[u16; IDS_PER_SLOT]) {
        let mut feat = n_dense + pos * self.slot_width();
        for t in 0..3 {
            let row = self.row(t, ids[t]);
            for (c, v) in row.iter().enumerate() {
                axpy_row::<H>(acc, *v, w1t, feat + c);
            }
            feat += self.dims[t];
        }
        for m in 0..4 {
            let row = self.row(features_v4::MOVE, ids[3 + m]);
            for (c, v) in row.iter().enumerate() {
                axpy_row::<H>(acc, 0.25 * *v, w1t, feat + c);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Evaluator {
    model: Model,
    /// Identifies the weights; keys the per-`Scratch` static-feature cache.
    id: u64,
    /// Feature version the weights were trained with (1, 2 or 3); the input length follows from it.
    feat_ver: u32,
    /// CPU supports AVX2 (runtime-detected; enables a wider build of the fast MLP path).
    avx2: bool,
    /// If true the leaf score is `sigmoid(z_leaf - z_root)`, else `sigmoid(z_leaf)`.
    pub relative_to_root: bool,
    /// Multiplies the logit (gap) before the sigmoid. Default 1.0.
    pub leaf_scale: f32,
    /// If `Some(c)` the logit is `c * evaluate(state) + net(features)`.
    pub stock_coef: Option<f32>,
    /// Material blend lambda in [0, 1): leaf score = (1-l)*score + l*material(state). 0 = off.
    /// Runtime-only (not serialized); applied to non-terminal leaves only.
    pub material: f32,
}

/// Per-worker scratch buffers so evaluation does not allocate.
pub struct Scratch {
    feats: Vec<f32>,
    hidden: Vec<f32>,
    /// Fast MLP path: model id the cache below belongs to (0 = none).
    cache_model: u64,
    /// Cache lines, `CACHE_WAYS` per `[side][team index][slot]` set: the
    /// first-layer contribution of a Pokemon's slow-changing features (types,
    /// tera type, terastallized, item flag, stats) for variant 0 = fainted and
    /// 1 = alive (adds the `alive` row). `hidden` floats each.
    cache_vals: Vec<f32>,
    /// Key (types, tera, terastallized, item flag) each line was computed from;
    /// `u32::MAX` = empty. The stats are NOT part of the key: they are fixed for
    /// the life of a search, so a `Scratch` must not be reused for states whose
    /// team stats differ without calling `Scratch::reset()` (every search and
    /// `Evaluator::logit` wrapper in this crate uses a fresh one).
    cache_keys: Vec<[u64; 3]>,
    /// 1 / max(maxhp, 1) per line (hp fraction = hp * recip, no division per leaf).
    cache_recip: Vec<f32>,
    /// Next way to evict, per set.
    cache_next: Vec<u8>,
}

const CACHE_SETS: usize = 2 * TEAM_SLOTS * TEAM_SLOTS;
/// 2 ways: a tera / item-loss toggle on the active Pokemon flips its key mid-search.
const CACHE_WAYS: usize = 2;

impl Scratch {
    /// Forget cached per-Pokemon contributions (required if team stats changed).
    pub fn reset(&mut self) {
        self.cache_model = 0;
    }

    pub fn new() -> Scratch {
        Scratch {
            feats: vec![0.0; FEATURE_LEN_V4],
            hidden: Vec::new(),
            cache_model: 0,
            cache_vals: Vec::new(),
            cache_keys: Vec::new(),
            cache_next: Vec::new(),
            cache_recip: Vec::new(),
        }
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Scratch::new()
    }
}

fn detect_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn mlp_avx2<const H: usize, const V2: bool>(
    ev: &Evaluator,
    w1t: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: f32,
    emb: Option<&Embed>,
    state: &State,
    scratch: &mut Scratch,
) -> f32 {
    ev.mlp_fast::<H, V2>(w1t, b1, w2, b2, emb, state, scratch)
}

/// Adds first-layer weight rows for the (absolute-index) features it receives.
struct AccSink<'a, const H: usize> {
    acc: &'a mut [f32; H],
    w1t: &'a [f32],
}
impl<'a, const H: usize> Sink for AccSink<'a, H> {
    #[inline(always)]
    fn one(&mut self, idx: usize) {
        add_row::<H>(self.acc, self.w1t, idx);
    }
    #[inline(always)]
    fn val(&mut self, idx: usize, v: f32) {
        axpy_row::<H>(self.acc, v, self.w1t, idx);
    }
}

#[inline(always)]
fn add_row<const H: usize>(acc: &mut [f32; H], w1t: &[f32], feat: usize) {
    let row: &[f32; H] = w1t[feat * H..feat * H + H].try_into().unwrap();
    for j in 0..H {
        acc[j] += row[j];
    }
}

#[inline(always)]
fn axpy_row<const H: usize>(acc: &mut [f32; H], x: f32, w1t: &[f32], feat: usize) {
    let row: &[f32; H] = w1t[feat * H..feat * H + H].try_into().unwrap();
    for j in 0..H {
        acc[j] += x * row[j];
    }
}

#[inline(always)]
fn type_idx(t: PokemonType) -> Option<usize> {
    let i = t as usize;
    if i < 18 {
        Some(i)
    } else {
        None
    }
}

/// Slow-changing features of one Pokemon (types, tera, terastallized, item flag, stats),
/// accumulated into `out`.
fn static_contrib<const H: usize>(
    out: &mut [f32; H],
    w1t: &[f32],
    base: usize,
    p: &crate::state::Pokemon,
) {
    let t0 = type_idx(p.types.0);
    let t1 = type_idx(p.types.1);
    if let Some(t) = t0 {
        add_row::<H>(out, w1t, base + 8 + t);
    }
    if let Some(t) = t1 {
        if t0 != Some(t) {
            add_row::<H>(out, w1t, base + 8 + t);
        }
    }
    if let Some(t) = type_idx(p.tera_type) {
        add_row::<H>(out, w1t, base + 26 + t);
    }
    if p.tera_type == PokemonType::STELLAR {
        add_row::<H>(out, w1t, base + 44);
    }
    if p.terastallized {
        add_row::<H>(out, w1t, base + 45);
    }
    if p.item != Items::NONE {
        add_row::<H>(out, w1t, base + 46);
    }
    const STAT_SCALE: f32 = crate::features::STAT_SCALE;
    axpy_row::<H>(out, p.maxhp as f32 / STAT_SCALE, w1t, base + 47);
    axpy_row::<H>(out, p.attack as f32 / STAT_SCALE, w1t, base + 48);
    axpy_row::<H>(out, p.defense as f32 / STAT_SCALE, w1t, base + 49);
    axpy_row::<H>(out, p.special_attack as f32 / STAT_SCALE, w1t, base + 50);
    axpy_row::<H>(out, p.special_defense as f32 / STAT_SCALE, w1t, base + 51);
    axpy_row::<H>(out, p.speed as f32 / STAT_SCALE, w1t, base + 52);
}

#[inline(always)]
fn mlp_side<const H: usize, const V2: bool>(
    acc: &mut [f32; H],
    w1t: &[f32],
    side: &Side,
    side_no: usize,
    emb: Option<&Embed>,
    sc: &mut Scratch,
) {
    let side_base = side_no * PER_SIDE;
    let active = side.active_index as usize;
    let mut alive_count = 0i8;
    // slot order: the active Pokemon first, then the rest in team order
    let mut order = [0usize; TEAM_SLOTS];
    order[0] = active;
    let mut n = 1;
    for k in 0..TEAM_SLOTS {
        if k != active {
            order[n] = k;
            n += 1;
        }
    }
    {
        for slot in 0..TEAM_SLOTS {
            let k = order[slot];
            let p = &side.pokemon.pkmn[k];
            let base = side_base + slot * PER_POKEMON;
            // slow-changing part, cached per (side, team index, slot) set
            let key: [u64; 3] = if V2 {
                let k = features_v2::static_key(p);
                // v4: the species (item, ability and moves are already in the v2 key)
                [k[0], k[1], if emb.is_some() { p.id as u64 } else { 0 }]
            } else {
                [
                    (p.types.0 as u64)
                        | (p.types.1 as u64) << 8
                        | (p.tera_type as u64) << 16
                        | (p.terastallized as u64) << 24
                        | ((p.item != Items::NONE) as u64) << 25,
                    0,
                    0,
                ]
            };
            let set = (side_no * TEAM_SLOTS + k) * TEAM_SLOTS + slot;
            let line0 = set * CACHE_WAYS;
            let line = if sc.cache_keys[line0] == key {
                line0
            } else if sc.cache_keys[line0 + 1] == key {
                line0 + 1
            } else {
                let way = sc.cache_next[set] as usize;
                sc.cache_next[set] = (way ^ 1) as u8;
                let line = line0 + way;
                let mut v = [0.0f32; H];
                static_contrib::<H>(&mut v, w1t, base, p);
                if V2 {
                    features_v2::static_block(
                        &mut AccSink::<H> { acc: &mut v, w1t },
                        features_v2::mon_base(side_no, slot),
                        p,
                    );
                }
                let o = line * 2 * H;
                if let Some(e) = emb {
                    // fainted variant sees the pad identity, the alive one the real identity
                    let pos = side_no * TEAM_SLOTS + slot;
                    let mut v0 = v;
                    e.add_slot::<H>(&mut v0, w1t, FEATURE_LEN_V4, pos, &[features_v4::IDX_PAD; IDS_PER_SLOT]);
                    sc.cache_vals[o..o + H].copy_from_slice(&v0);
                    e.add_slot::<H>(&mut v, w1t, FEATURE_LEN_V4, pos, &features_v4::slot_ids_alive(p));
                } else {
                    sc.cache_vals[o..o + H].copy_from_slice(&v);
                }
                add_row::<H>(&mut v, w1t, base + 1);
                sc.cache_vals[o + H..o + 2 * H].copy_from_slice(&v);
                sc.cache_keys[line] = key;
                sc.cache_recip[line] = 1.0 / p.maxhp.max(1) as f32;
                line
            };
            let o = (line * 2 + (p.hp > 0) as usize) * H;
            let vals: &[f32; H] = sc.cache_vals[o..o + H].try_into().unwrap();
            for j in 0..H {
                acc[j] += vals[j];
            }
            alive_count += (p.hp > 0) as i8;
            // hp fraction (0 when fainted)
            let hpf = (p.hp.max(0) as f32 * sc.cache_recip[line]).min(1.0);
            axpy_row::<H>(acc, hpf, w1t, base);
            if p.status != PokemonStatus::NONE {
                add_row::<H>(acc, w1t, base + 1 + p.status as usize);
            }
        }
    }

    // side extras
    let ex = side_base + TEAM_SLOTS * PER_POKEMON;
    let boosts = [
        side.attack_boost,
        side.defense_boost,
        side.special_attack_boost,
        side.special_defense_boost,
        side.speed_boost,
        side.accuracy_boost,
        side.evasion_boost,
    ];
    if boosts.iter().any(|b| *b != 0) {
        for (i, b) in boosts.iter().enumerate() {
            axpy_row::<H>(acc, *b as f32 / 6.0, w1t, ex + i);
        }
    }
    let vs = &side.volatile_statuses;
    if !vs.is_empty() {
        const VOLS: [V; 16] = [
            V::LEECHSEED,
            V::SUBSTITUTE,
            V::CONFUSION,
            V::TAUNT,
            V::ENCORE,
            V::YAWN,
            V::TORMENT,
            V::DISABLE,
            V::AQUARING,
            V::INGRAIN,
            V::CURSE,
            V::NIGHTMARE,
            V::SALTCURE,
            V::SYRUPBOMB,
            V::HEALBLOCK,
            V::PARTIALLYTRAPPED,
        ];
        for (i, v) in VOLS.iter().enumerate() {
            if vs.contains(v) {
                add_row::<H>(acc, w1t, ex + 7 + i);
            }
        }
        if vs.contains(&V::PERISH4)
            || vs.contains(&V::PERISH3)
            || vs.contains(&V::PERISH2)
            || vs.contains(&V::PERISH1)
        {
            add_row::<H>(acc, w1t, ex + 7 + 16);
        }
        for (i, v) in [V::FOCUSENERGY, V::MUSTRECHARGE, V::LOCKEDMOVE].iter().enumerate() {
            if vs.contains(v) {
                add_row::<H>(acc, w1t, ex + 7 + 17 + i);
            }
        }
    }
    let c = &side.side_conditions;
    let sc0 = ex + 27;
    if c.stealth_rock > 0 {
        add_row::<H>(acc, w1t, sc0);
    }
    if c.spikes != 0 {
        axpy_row::<H>(acc, c.spikes as f32 / 3.0, w1t, sc0 + 1);
    }
    if c.toxic_spikes != 0 {
        axpy_row::<H>(acc, c.toxic_spikes as f32 / 2.0, w1t, sc0 + 2);
    }
    for (i, v) in [
        c.sticky_web,
        c.reflect,
        c.light_screen,
        c.aurora_veil,
        c.safeguard,
        c.tailwind,
        c.mist,
        c.lucky_chant,
        c.healing_wish,
        c.lunar_dance,
        c.protect,
    ].iter().copied().enumerate()
    {
        if v > 0 {
            add_row::<H>(acc, w1t, sc0 + 3 + i);
        }
    }
    let misc = ex + 41;
    if side.wish.0 > 0 {
        add_row::<H>(acc, w1t, misc);
    }
    if side.future_sight.0 > 0 {
        add_row::<H>(acc, w1t, misc + 1);
    }
    if side.force_switch {
        add_row::<H>(acc, w1t, misc + 2);
    }
    axpy_row::<H>(acc, alive_count as f32 / TEAM_SLOTS as f32, w1t, misc + 3);
}

/// First-layer pre-activations `b1 + W1 x` of the sparse fast path for `state` seen from
/// `own` (the own side fills the "own" feature blocks, the other side the "opp" blocks).
/// Shared by the value MLP (`own` = side one) and the policy prior (`prior.rs`).
/// `model_id` keys the static-feature cache in `scratch`; a `Scratch` must only be used
/// with one `own` perspective per model id.
#[inline(always)]
pub(crate) fn hidden_acc<const H: usize, const V2: bool>(
    model_id: u64,
    w1t: &[f32],
    b1: &[f32],
    state: &State,
    own: SideReference,
    scratch: &mut Scratch,
) -> [f32; H] {
    hidden_acc_emb::<H, V2>(model_id, w1t, b1, state, own, scratch, None)
}

/// `hidden_acc` with optional identity embeddings (feature version 4; `V2` must be true).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn hidden_acc_emb<const H: usize, const V2: bool>(
    model_id: u64,
    w1t: &[f32],
    b1: &[f32],
    state: &State,
    own: SideReference,
    scratch: &mut Scratch,
    emb: Option<&Embed>,
) -> [f32; H] {
    if scratch.cache_model != model_id || scratch.cache_vals.len() != CACHE_SETS * CACHE_WAYS * 2 * H {
        scratch.cache_model = model_id;
        scratch.cache_vals.clear();
        scratch.cache_vals.resize(CACHE_SETS * CACHE_WAYS * 2 * H, 0.0);
        scratch.cache_keys.clear();
        scratch.cache_keys.resize(CACHE_SETS * CACHE_WAYS, [u64::MAX; 3]);
        scratch.cache_recip.clear();
        scratch.cache_recip.resize(CACHE_SETS * CACHE_WAYS, 1.0);
        scratch.cache_next.clear();
        scratch.cache_next.resize(CACHE_SETS, 0);
    }
    let mut acc = [0.0f32; H];
    acc.copy_from_slice(&b1[..H]);
    let (own_side, opp_side) = match own {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    mlp_side::<H, V2>(&mut acc, w1t, own_side, 0, emb, scratch);
    mlp_side::<H, V2>(&mut acc, w1t, opp_side, 1, emb, scratch);

    let field = 2 * PER_SIDE;
    let weather = state.get_weather();
    for (k, w) in [
        Weather::SUN,
        Weather::RAIN,
        Weather::SAND,
        Weather::HAIL,
        Weather::SNOW,
        Weather::HARSHSUN,
        Weather::HEAVYRAIN,
    ].iter().copied().enumerate()
    {
        if weather == w {
            add_row::<H>(&mut acc, w1t, field + k);
        }
    }
    let terrain = state.get_terrain();
    for (k, t) in [
        Terrain::ELECTRICTERRAIN,
        Terrain::PSYCHICTERRAIN,
        Terrain::MISTYTERRAIN,
        Terrain::GRASSYTERRAIN,
    ].iter().copied().enumerate()
    {
        if terrain == t {
            add_row::<H>(&mut acc, w1t, field + 7 + k);
        }
    }
    if state.trick_room.active {
        add_row::<H>(&mut acc, w1t, field + 11);
    }
    if V2 {
        features_v2::dynamic_blocks(&mut AccSink::<H> { acc: &mut acc, w1t }, state, own);
    }
    acc
}

#[inline]
pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Evaluator {
    /// value = w . features + b (feature version 1)
    pub fn linear(w: Vec<f32>, b: f32) -> Result<Evaluator, String> {
        Evaluator::linear_v(FEATURE_VERSION_1, w, b)
    }

    /// value = w . features + b over the features of `feat_ver` (1, 2 or 3).
    pub fn linear_v(feat_ver: u32, w: Vec<f32>, b: f32) -> Result<Evaluator, String> {
        let n_in = feature_len(feat_ver).ok_or_else(|| format!("unknown feature version {feat_ver}"))?;
        if feat_ver == FEATURE_VERSION_4 {
            return Err("feature version 4 needs embedding tables (use a MLP with Embed)".to_string());
        }
        if w.len() != n_in {
            return Err(format!(
                "linear weights have length {}, expected {} (feature version {feat_ver})",
                w.len(),
                n_in
            ));
        }
        Ok(Evaluator {
            model: Model::Linear { w, b },
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            feat_ver,
            avx2: detect_avx2(),
            relative_to_root: true,
            leaf_scale: 1.0,
            stock_coef: None,
            material: 0.0,
        })
    }

    /// value = w2 . relu(W1 x + b1) + b2, with `w1` row-major `[hidden][FEATURE_LEN]` (version 1).
    pub fn mlp(
        hidden: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    ) -> Result<Evaluator, String> {
        Evaluator::mlp_v(FEATURE_VERSION_1, hidden, w1, b1, w2, b2)
    }

    /// MLP over the features of `feat_ver`; `w1` is row-major `[hidden][feature_len(feat_ver)]`.
    pub fn mlp_v(
        feat_ver: u32,
        hidden: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    ) -> Result<Evaluator, String> {
        let n_in = feature_len(feat_ver).ok_or_else(|| format!("unknown feature version {feat_ver}"))?;
        if feat_ver == FEATURE_VERSION_4 {
            return Err("feature version 4 needs embedding tables (use Evaluator::mlp_embed)".to_string());
        }
        Evaluator::mlp_inner(feat_ver, n_in, hidden, w1, b1, w2, b2, None)
    }

    /// Feature version 4 MLP: `w1` is row-major `[hidden][FEATURE_LEN_V4 + emb.input_len()]`
    /// (dense features, then the concatenated slot embeddings, see `Embed`).
    pub fn mlp_embed(
        hidden: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
        emb: Embed,
    ) -> Result<Evaluator, String> {
        let n_in = FEATURE_LEN_V4 + emb.input_len();
        Evaluator::mlp_inner(FEATURE_VERSION_4, n_in, hidden, w1, b1, w2, b2, Some(emb))
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp_inner(
        feat_ver: u32,
        n_in: usize,
        hidden: usize,
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
        emb: Option<Embed>,
    ) -> Result<Evaluator, String> {
        if hidden == 0 {
            return Err("hidden size must be > 0".to_string());
        }
        if w1.len() != hidden * n_in {
            return Err(format!(
                "w1 has length {}, expected hidden * n_in = {} (feature version {feat_ver})",
                w1.len(),
                hidden * n_in
            ));
        }
        if b1.len() != hidden || w2.len() != hidden {
            return Err(format!(
                "b1 / w2 must have length hidden = {} (got {} / {})",
                hidden,
                b1.len(),
                w2.len()
            ));
        }
        let mut w1t = vec![0.0f32; w1.len()];
        for j in 0..hidden {
            for i in 0..n_in {
                w1t[i * hidden + j] = w1[j * n_in + i];
            }
        }
        Ok(Evaluator {
            model: Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
                emb,
            },
            id: NEXT_MODEL_ID.fetch_add(1, Ordering::Relaxed),
            feat_ver,
            avx2: detect_avx2(),
            relative_to_root: true,
            leaf_scale: 1.0,
            stock_coef: None,
            material: 0.0,
        })
    }

    /// The identity embedding tables (feature version 4 only).
    pub fn embed(&self) -> Option<&Embed> {
        match &self.model {
            Model::Mlp { emb, .. } => emb.as_ref(),
            _ => None,
        }
    }

    /// Feature version of the weights (1, 2, 3 or 4).
    pub fn feature_version(&self) -> u32 {
        self.feat_ver
    }

    /// Length of the feature vector the weights consume.
    pub fn n_in(&self) -> usize {
        feature_len(self.feat_ver).unwrap_or(FEATURE_LEN)
    }

    /// Inputs of the first layer: `n_in()` dense features plus the embedding inputs (v4).
    pub fn n_total(&self) -> usize {
        self.n_in() + self.embed().map_or(0, |e| e.input_len())
    }

    pub fn with_relative_to_root(mut self, relative: bool) -> Evaluator {
        self.relative_to_root = relative;
        self
    }

    /// Make this a residual evaluator: logit = `stock_coef * evaluate(state) + net`.
    pub fn with_stock_coef(mut self, stock_coef: f32) -> Result<Evaluator, String> {
        if !stock_coef.is_finite() {
            return Err(format!("stock_coef must be finite, got {stock_coef}"));
        }
        self.stock_coef = Some(stock_coef);
        Ok(self)
    }

    pub fn is_residual(&self) -> bool {
        self.stock_coef.is_some()
    }

    /// Set the material blend lambda; must be finite and in [0, 1).
    pub fn with_material(mut self, lambda: f32) -> Result<Evaluator, String> {
        if !lambda.is_finite() || !(0.0..1.0).contains(&lambda) {
            return Err(format!("material must be finite and in [0, 1), got {lambda}"));
        }
        self.material = lambda;
        Ok(self)
    }

    /// Set the leaf scale; must be finite and > 0.
    pub fn with_leaf_scale(mut self, scale: f32) -> Result<Evaluator, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("leaf_scale must be finite and > 0, got {scale}"));
        }
        self.leaf_scale = scale;
        Ok(self)
    }

    pub fn kind_name(&self) -> &'static str {
        match self.model {
            Model::Linear { .. } => "linear",
            Model::Mlp { .. } => "mlp",
        }
    }

    /// Hidden layer size (0 for linear).
    pub fn hidden_size(&self) -> usize {
        match &self.model {
            Model::Linear { .. } => 0,
            Model::Mlp { hidden, .. } => *hidden,
        }
    }

    /// Net output (without the stock term of a residual evaluator) from an
    /// already computed input vector (length `n_total()`: the dense features, then for
    /// feature version 4 the concatenated slot embeddings, see `Embed::fill_inputs`).
    /// Inputs that are exactly 0.0 are skipped (the encoding is sparse).
    pub fn logit_from_features(&self, x: &[f32], hidden_buf: &mut Vec<f32>) -> f32 {
        match &self.model {
            Model::Linear { w, b } => {
                let mut z = *b;
                for (xi, wi) in x.iter().zip(w.iter()) {
                    if *xi != 0.0 {
                        z += xi * wi;
                    }
                }
                z
            }
            Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
                ..
            } => {
                let h = *hidden;
                hidden_buf.clear();
                hidden_buf.extend_from_slice(b1);
                for (i, xi) in x.iter().enumerate() {
                    if *xi != 0.0 {
                        let row = &w1t[i * h..(i + 1) * h];
                        for (hj, wj) in hidden_buf.iter_mut().zip(row.iter()) {
                            *hj += xi * wj;
                        }
                    }
                }
                let mut z = *b2;
                for (hj, wj) in hidden_buf.iter().zip(w2.iter()) {
                    z += hj.max(0.0) * wj;
                }
                z
            }
        }
    }

    /// z = logit of P(side one wins) at `state`.
    ///
    /// MLPs with hidden size 8 / 16 / 32 / 64 take a sparse fast path that
    /// accumulates first-layer weight rows straight from the state (no dense
    /// feature vector); results match `logit_reference` to f32 rounding.
    pub fn logit(&self, state: &State, scratch: &mut Scratch) -> f32 {
        let net = match &self.model {
            Model::Mlp { hidden, w1t, b1, w2, b2, emb } => {
                let v2 = self.feat_ver == FEATURE_VERSION_2 || self.feat_ver == FEATURE_VERSION_4;
                let emb = emb.as_ref();
                #[cfg(target_arch = "x86_64")]
                {
                    if self.avx2 {
                        // SAFETY: `avx2` is only set when the CPU reports AVX2.
                        unsafe {
                            match (*hidden, v2) {
                                (8, false) => mlp_avx2::<8, false>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (16, false) => mlp_avx2::<16, false>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (32, false) => mlp_avx2::<32, false>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (64, false) => mlp_avx2::<64, false>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (128, false) => mlp_avx2::<128, false>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (8, true) => mlp_avx2::<8, true>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (16, true) => mlp_avx2::<16, true>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (32, true) => mlp_avx2::<32, true>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (64, true) => mlp_avx2::<64, true>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                (128, true) => mlp_avx2::<128, true>(self, w1t, b1, w2, *b2, emb, state, scratch),
                                _ => return self.logit_reference(state, scratch),
                            }
                        }
                    } else {
                        if !matches!(*hidden, 8 | 16 | 32 | 64 | 128) {
                            return self.logit_reference(state, scratch);
                        }
                        self.mlp_dispatch(*hidden, v2, w1t, b1, w2, *b2, emb, state, scratch)
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    if !matches!(*hidden, 8 | 16 | 32 | 64 | 128) {
                        return self.logit_reference(state, scratch);
                    }
                    self.mlp_dispatch(*hidden, v2, w1t, b1, w2, *b2, emb, state, scratch)
                }
            }
            Model::Linear { .. } => return self.logit_reference(state, scratch),
        };
        match self.stock_coef {
            Some(c) => c * evaluate(state) + net,
            None => net,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mlp_dispatch(
        &self,
        hidden: usize,
        v2: bool,
        w1t: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: f32,
        emb: Option<&Embed>,
        state: &State,
        scratch: &mut Scratch,
    ) -> f32 {
        match (hidden, v2) {
            (8, false) => self.mlp_fast::<8, false>(w1t, b1, w2, b2, emb, state, scratch),
            (16, false) => self.mlp_fast::<16, false>(w1t, b1, w2, b2, emb, state, scratch),
            (32, false) => self.mlp_fast::<32, false>(w1t, b1, w2, b2, emb, state, scratch),
            (64, false) => self.mlp_fast::<64, false>(w1t, b1, w2, b2, emb, state, scratch),
            (128, false) => self.mlp_fast::<128, false>(w1t, b1, w2, b2, emb, state, scratch),
            (8, true) => self.mlp_fast::<8, true>(w1t, b1, w2, b2, emb, state, scratch),
            (16, true) => self.mlp_fast::<16, true>(w1t, b1, w2, b2, emb, state, scratch),
            (32, true) => self.mlp_fast::<32, true>(w1t, b1, w2, b2, emb, state, scratch),
            (64, true) => self.mlp_fast::<64, true>(w1t, b1, w2, b2, emb, state, scratch),
            (128, true) => self.mlp_fast::<128, true>(w1t, b1, w2, b2, emb, state, scratch),
            _ => f32::NAN, // unreachable: callers only pass supported sizes
        }
    }

    /// Reference implementation: dense feature vector + `logit_from_features`.
    pub fn logit_reference(&self, state: &State, scratch: &mut Scratch) -> f32 {
        features_versioned_into(state, SideReference::SideOne, self.feat_ver, &mut scratch.feats);
        let n_in = self.n_in();
        let net = if let Some(e) = self.embed() {
            let total = n_in + e.input_len();
            if scratch.feats.len() < total {
                scratch.feats.resize(total, 0.0);
            }
            let mut ids = [0u16; N_IDS];
            features_v4::identity_into(state, SideReference::SideOne, &mut ids);
            e.fill_inputs(&ids, &mut scratch.feats[n_in..total]);
            self.logit_from_features(&scratch.feats[..total], &mut scratch.hidden)
        } else {
            self.logit_from_features(&scratch.feats[..n_in], &mut scratch.hidden)
        };
        match self.stock_coef {
            Some(c) => c * evaluate(state) + net,
            None => net,
        }
    }

    #[inline(always)]
    fn mlp_fast<const H: usize, const V2: bool>(
        &self,
        w1t: &[f32],
        b1: &[f32],
        w2: &[f32],
        b2: f32,
        emb: Option<&Embed>,
        state: &State,
        scratch: &mut Scratch,
    ) -> f32 {
        let mut acc = hidden_acc_emb::<H, V2>(self.id, w1t, b1, state, SideReference::SideOne, scratch, emb);
        if self.feat_ver == FEATURE_VERSION_3 {
            features_v3::matchup_blocks(&mut AccSink::<H> { acc: &mut acc, w1t }, state, SideReference::SideOne);
        } else if self.feat_ver == FEATURE_VERSION_4 {
            features_v4::matchup_blocks_v4(&mut AccSink::<H> { acc: &mut acc, w1t }, state, SideReference::SideOne);
        }
        let mut z = b2;
        for j in 0..H {
            z += acc[j].max(0.0) * w2[j];
        }
        z
    }

    /// Leaf score in [0, 1] from the leaf and root logits.
    #[inline]
    pub fn score(&self, leaf_logit: f32, root_logit: f32) -> f32 {
        if self.relative_to_root {
            logistic(self.leaf_scale * (leaf_logit - root_logit))
        } else {
            logistic(self.leaf_scale * leaf_logit)
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        if let Model::Mlp { hidden, w1t, b1, w2, b2, emb: Some(e) } = &self.model {
            return self.to_bytes_embed(*hidden, w1t, b1, w2, *b2, e);
        }
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let push = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        push(
            &mut out,
            if self.is_residual() { FORMAT_VERSION_RESIDUAL } else { FORMAT_VERSION },
        );
        push(&mut out, self.feat_ver);
        let pushf = |out: &mut Vec<u8>, xs: &[f32]| {
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        match &self.model {
            Model::Linear { w, b } => {
                push(&mut out, KIND_LINEAR);
                push(&mut out, self.n_in() as u32);
                push(&mut out, 0);
                pushf(&mut out, w);
                pushf(&mut out, &[*b]);
            }
            Model::Mlp {
                hidden,
                w1t,
                b1,
                w2,
                b2,
                ..
            } => {
                push(&mut out, KIND_MLP);
                push(&mut out, self.n_in() as u32);
                push(&mut out, *hidden as u32);
                // back to row-major [hidden][n_in]
                for j in 0..*hidden {
                    for i in 0..self.n_in() {
                        pushf(&mut out, &[w1t[i * hidden + j]]);
                    }
                }
                pushf(&mut out, b1);
                pushf(&mut out, w2);
                pushf(&mut out, &[*b2]);
            }
        }
        if let Some(c) = self.stock_coef {
            // splice flags + stock_coef in after the 24-byte header
            let mut extra = Vec::with_capacity(8);
            extra.extend_from_slice(&FLAG_RESIDUAL.to_le_bytes());
            extra.extend_from_slice(&c.to_le_bytes());
            out.splice(24..24, extra);
        }
        out
    }

    /// Format 3 (embedding tables), see the module doc.
    fn to_bytes_embed(&self, hidden: usize, w1t: &[f32], b1: &[f32], w2: &[f32], b2: f32, e: &Embed) -> Vec<u8> {
        let n_total = self.n_total();
        let mut out = Vec::with_capacity(64 + (hidden * n_total + 2 * hidden + 1) * 4);
        out.extend_from_slice(MAGIC);
        let push = |out: &mut Vec<u8>, v: u32| out.extend_from_slice(&v.to_le_bytes());
        push(&mut out, FORMAT_VERSION_EMBED);
        push(&mut out, self.feat_ver);
        push(&mut out, KIND_MLP);
        push(&mut out, self.n_in() as u32);
        push(&mut out, hidden as u32);
        push(&mut out, if self.is_residual() { FLAG_RESIDUAL } else { 0 });
        out.extend_from_slice(&self.stock_coef.unwrap_or(0.0).to_le_bytes());
        out.extend_from_slice(&features_v4::vocab_hash().to_le_bytes());
        for d in e.dims() {
            push(&mut out, d as u32);
        }
        let mut pushf = |xs: &[f32]| {
            for x in xs {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        for j in 0..hidden {
            for i in 0..n_total {
                pushf(&[w1t[i * hidden + j]]);
            }
        }
        pushf(b1);
        pushf(w2);
        pushf(&[b2]);
        for t in 0..N_TABLES {
            pushf(e.table(t));
        }
        out
    }

    fn from_bytes_embed(bytes: &[u8]) -> Result<Evaluator, String> {
        const HEADER: usize = 24 + 4 + 4 + 8 + 4 * N_TABLES;
        if bytes.len() < HEADER {
            return Err("truncated format-3 weights header".to_string());
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let feat_ver = u32_at(8);
        if feat_ver != FEATURE_VERSION_4 {
            return Err(format!("format-3 weights need FEATURE_VERSION 4, got {feat_ver}"));
        }
        if u32_at(12) != KIND_MLP {
            return Err(format!("format-3 weights must be an MLP, got kind {}", u32_at(12)));
        }
        let n_dense = u32_at(16) as usize;
        if n_dense != FEATURE_LEN_V4 {
            return Err(format!("weights expect {n_dense} dense inputs, FEATURE_VERSION 4 has {FEATURE_LEN_V4}"));
        }
        let hidden = u32_at(20) as usize;
        let flags = u32_at(24);
        if flags & !FLAG_RESIDUAL != 0 {
            return Err(format!("unknown weights flags {flags:#x}"));
        }
        let coef = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
        let stock_coef = if flags & FLAG_RESIDUAL != 0 {
            if !coef.is_finite() {
                return Err(format!("non-finite stock_coef {coef}"));
            }
            Some(coef)
        } else {
            None
        };
        let hash = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        if hash != features_v4::vocab_hash() {
            return Err(format!(
                "weights were trained with embedding vocabulary {hash:#018x}, this engine has {:#018x} \
                 (the species / item / ability / move enums changed; retrain or use the matching engine)",
                features_v4::vocab_hash()
            ));
        }
        let mut dims = [0usize; N_TABLES];
        for (t, d) in dims.iter_mut().enumerate() {
            *d = u32_at(40 + 4 * t) as usize;
            if *d == 0 || *d > 1024 {
                return Err(format!("bad {} embedding dim {}", features_v4::TABLE_NAMES[t], d));
            }
        }
        let slot_w: usize = dims.iter().sum();
        let n_total = n_dense + N_SLOTS * slot_w;
        let n_tab: usize = (0..N_TABLES).map(|t| features_v4::vocab_size(t) * dims[t]).sum();
        let floats_needed = hidden * n_total + 2 * hidden + 1 + n_tab;
        let body = &bytes[HEADER..];
        if body.len() != floats_needed * 4 {
            return Err(format!("weights body is {} bytes, expected {}", body.len(), floats_needed * 4));
        }
        let floats: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let a = hidden * n_total;
        let mut tables: [Vec<f32>; N_TABLES] = Default::default();
        let mut off = a + 2 * hidden + 1;
        for t in 0..N_TABLES {
            let n = features_v4::vocab_size(t) * dims[t];
            tables[t] = floats[off..off + n].to_vec();
            off += n;
        }
        let ev = Evaluator::mlp_embed(
            hidden,
            floats[..a].to_vec(),
            floats[a..a + hidden].to_vec(),
            floats[a + hidden..a + 2 * hidden].to_vec(),
            floats[a + 2 * hidden],
            Embed::new(dims, tables)?,
        )?;
        match stock_coef {
            Some(c) => ev.with_stock_coef(c),
            None => Ok(ev),
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Evaluator, String> {
        if bytes.len() < 24 || &bytes[0..4] != MAGIC {
            return Err("not a poke-engine evaluator file (bad magic)".to_string());
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let format = u32_at(4);
        if format == FORMAT_VERSION_EMBED {
            return Evaluator::from_bytes_embed(bytes);
        }
        if format != FORMAT_VERSION && format != FORMAT_VERSION_RESIDUAL {
            return Err(format!("unsupported weights format version {format}"));
        }
        let feat_ver = u32_at(8);
        let expected_in = feature_len(feat_ver).ok_or_else(|| {
            format!("weights were trained with unknown FEATURE_VERSION {feat_ver} (this engine knows 1, 2, 3 and 4)")
        })?;
        let kind = u32_at(12);
        let n_in = u32_at(16) as usize;
        let hidden = u32_at(20) as usize;
        if n_in != expected_in {
            return Err(format!(
                "weights expect {n_in} inputs, FEATURE_VERSION {feat_ver} has {expected_in}"
            ));
        }
        let mut stock_coef = None;
        let mut body_start = 24;
        if format == FORMAT_VERSION_RESIDUAL {
            if bytes.len() < 32 {
                return Err("truncated v2 weights header".to_string());
            }
            let flags = u32_at(24);
            if flags & !FLAG_RESIDUAL != 0 {
                return Err(format!("unknown weights flags {flags:#x}"));
            }
            if flags & FLAG_RESIDUAL != 0 {
                let c = f32::from_le_bytes(bytes[28..32].try_into().unwrap());
                if !c.is_finite() {
                    return Err(format!("non-finite stock_coef {c}"));
                }
                stock_coef = Some(c);
            }
            body_start = 32;
        }
        let body = &bytes[body_start..];
        let floats_needed = match kind {
            KIND_LINEAR => n_in + 1,
            KIND_MLP => hidden * n_in + hidden + hidden + 1,
            _ => return Err(format!("unknown model kind {kind}")),
        };
        if body.len() != floats_needed * 4 {
            return Err(format!(
                "weights body is {} bytes, expected {}",
                body.len(),
                floats_needed * 4
            ));
        }
        let floats: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let ev = if kind == KIND_LINEAR {
            Evaluator::linear_v(feat_ver, floats[..n_in].to_vec(), floats[n_in])
        } else {
            let a = hidden * n_in;
            Evaluator::mlp_v(
                feat_ver,
                hidden,
                floats[..a].to_vec(),
                floats[a..a + hidden].to_vec(),
                floats[a + hidden..a + 2 * hidden].to_vec(),
                floats[a + 2 * hidden],
            )
        }?;
        match stock_coef {
            Some(c) => ev.with_stock_coef(c),
            None => Ok(ev),
        }
    }

    pub fn load(path: &str) -> Result<Evaluator, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        Evaluator::from_bytes(&bytes)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("cannot write {path}: {e}"))
    }
}

/// Material term m in (0, 1) for side one: `sigmoid(K * (hp_frac_sum_1 - hp_frac_sum_2
/// + ALIVE_W * (alive_1 - alive_2)))`. HP fractions are per Pokemon (hp / maxhp).
pub fn material_value(state: &State) -> f32 {
    const K: f32 = 1.5;
    const ALIVE_W: f32 = 0.25;
    fn side(s: &crate::state::Side) -> (f32, f32) {
        let (mut frac, mut alive) = (0.0f32, 0.0f32);
        for p in &s.pokemon {
            if p.hp > 0 {
                frac += (p.hp as f32 / p.maxhp.max(1) as f32).min(1.0);
                alive += 1.0;
            }
        }
        (frac, alive)
    }
    let (f1, a1) = side(&state.side_one);
    let (f2, a2) = side(&state.side_two);
    logistic(K * ((f1 - f2) + ALIVE_W * (a1 - a2)))
}

/// Leaf evaluation strategy used by the MCTS loops: stock `evaluate()` or a
/// learned `Evaluator`. Each search worker owns one (it carries scratch buffers).
pub struct LeafEval<'a> {
    learned: Option<&'a Evaluator>,
    scratch: Scratch,
    /// scratch of the policy prior (`prior.rs`), one per side
    pub prior_scratch: crate::prior::PriorScratch,
    /// tree depth statistics of the search this worker runs (see `MctsResult::depth_max`)
    pub depth_sum: u64,
    pub depth_max: u32,
}

impl<'a> LeafEval<'a> {
    pub fn new(learned: Option<&'a Evaluator>) -> LeafEval<'a> {
        LeafEval {
            learned,
            scratch: Scratch::new(),
            prior_scratch: Default::default(),
            depth_sum: 0,
            depth_max: 0,
        }
    }

    #[inline]
    pub fn note_depth(&mut self, depth: u32) {
        self.depth_sum += depth as u64;
        self.depth_max = self.depth_max.max(depth);
    }

    /// Raw value of a non-terminal position: stock evaluation points, or the
    /// learned logit.
    #[inline]
    pub fn value(&mut self, state: &State) -> f32 {
        match self.learned {
            None => evaluate(state),
            Some(ev) => ev.logit(state, &mut self.scratch),
        }
    }

    /// Absolute logit of P(side one wins) for a raw `value()`: the stock evaluation is scaled
    /// by 0.0125 (as in `score`), a learned logit is already one.
    #[inline]
    pub fn abs_logit_of_raw(&self, raw: f32) -> f32 {
        match self.learned {
            None => 0.0125 * raw,
            Some(_) => raw,
        }
    }

    /// Inverts `score` for a mean MCTS score `mean` (relative to the root value `root_raw`):
    /// the absolute logit of P(side one wins) the search assigns to the averaged positions.
    pub fn absolute_logit(&self, root_raw: f32, mean: f32) -> f32 {
        let p = mean.clamp(1e-4, 1.0 - 1e-4);
        let rel = (p / (1.0 - p)).ln();
        match self.learned {
            None => 0.0125 * root_raw + rel,
            Some(ev) if ev.relative_to_root => root_raw + rel / ev.leaf_scale,
            Some(ev) => rel / ev.leaf_scale,
        }
    }

    /// `score` for a non-terminal `state`, with the optional material blend of the learned
    /// evaluator (`Evaluator::material`). With material 0 this is exactly `score`.
    #[inline]
    pub fn score_state(&self, state: &State, leaf: f32, root: f32) -> f32 {
        let s = self.score(leaf, root);
        match self.learned {
            Some(ev) if ev.material > 0.0 => (1.0 - ev.material) * s + ev.material * material_value(state),
            _ => s,
        }
    }

    /// Maps (leaf value, root value) to an MCTS score in [0, 1].
    #[inline]
    pub fn score(&self, leaf: f32, root: f32) -> f32 {
        match self.learned {
            // identical to the stock `sigmoid(eval - root_eval)`
            None => 1.0 / (1.0 + (-0.0125 * (leaf - root)).exp()),
            Some(ev) => ev.score(leaf, root),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_names, features};

    fn idx(name: &str) -> usize {
        feature_names().iter().position(|n| n == name).unwrap()
    }

    fn state_with_hp(own_hp: i16, opp_hp: i16) -> State {
        let mut s = State::default();
        s.side_one.get_active().hp = own_hp;
        s.side_two.get_active().hp = opp_hp;
        s
    }

    #[test]
    fn linear_expected_output() {
        // z = 2 * own.p0.hp - 3 * opp.p0.hp + 0.5
        let mut w = vec![0.0; FEATURE_LEN];
        w[idx("own.p0.hp")] = 2.0;
        w[idx("opp.p0.hp")] = -3.0;
        let ev = Evaluator::linear(w, 0.5).unwrap();
        let mut sc = Scratch::new();
        let z = ev.logit(&state_with_hp(100, 40), &mut sc);
        assert!((z - (2.0 * 1.0 - 3.0 * 0.4 + 0.5)).abs() < 1e-5, "z = {z}");
        let z2 = ev.logit(&state_with_hp(50, 100), &mut sc);
        assert!((z2 - (2.0 * 0.5 - 3.0 + 0.5)).abs() < 1e-5, "z2 = {z2}");
    }

    #[test]
    fn mlp_expected_output() {
        // hidden0 = relu(own.hp - opp.hp), hidden1 = relu(-own.hp + opp.hp) , hidden2 = relu(-1)
        // z = 4 * h0 + 1 * h1 - 5 * h2 + 0.25
        let hidden = 3;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN];
        let (own, opp) = (idx("own.p0.hp"), idx("opp.p0.hp"));
        w1[0 * FEATURE_LEN + own] = 1.0;
        w1[0 * FEATURE_LEN + opp] = -1.0;
        w1[1 * FEATURE_LEN + own] = -1.0;
        w1[1 * FEATURE_LEN + opp] = 1.0;
        let b1 = vec![0.0, 0.0, -1.0];
        let w2 = vec![4.0, 1.0, -5.0];
        let ev = Evaluator::mlp(hidden, w1, b1, w2, 0.25).unwrap();
        let mut sc = Scratch::new();
        // own 100%, opp 40%: h0 = 0.6, h1 = relu(-0.6) = 0, h2 = relu(-1) = 0
        let z = ev.logit(&state_with_hp(100, 40), &mut sc);
        assert!((z - (4.0 * 0.6 + 0.25)).abs() < 1e-5, "z = {z}");
        // own 20%, opp 100%: h0 = 0, h1 = 0.8
        let z = ev.logit(&state_with_hp(20, 100), &mut sc);
        assert!((z - (0.8 + 0.25)).abs() < 1e-5, "z = {z}");
    }

    #[test]
    fn mlp_matches_dense_reference_on_random_weights() {
        // deterministic pseudo-random weights, compare against a dense computation
        let hidden = 7;
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        };
        let w1: Vec<f32> = (0..hidden * FEATURE_LEN).map(|_| next()).collect();
        let b1: Vec<f32> = (0..hidden).map(|_| next()).collect();
        let w2: Vec<f32> = (0..hidden).map(|_| next()).collect();
        let ev = Evaluator::mlp(hidden, w1.clone(), b1.clone(), w2.clone(), 0.1).unwrap();
        let mut s = state_with_hp(60, 30);
        s.side_one.attack_boost = 2;
        s.weather.weather_type = crate::engine::state::Weather::SUN;
        s.weather.turns_remaining = 4;
        let x = features(&s, SideReference::SideOne);
        let mut expected = 0.1f32;
        for j in 0..hidden {
            let mut a = b1[j];
            for i in 0..FEATURE_LEN {
                a += w1[j * FEATURE_LEN + i] * x[i];
            }
            expected += a.max(0.0) * w2[j];
        }
        let mut buf = Vec::new();
        let z = ev.logit_from_features(&x, &mut buf);
        assert!((z - expected).abs() < 1e-3, "z = {z}, expected = {expected}");
    }

    #[test]
    fn score_mapping() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0).unwrap();
        assert!((ev.score(1.5, 1.5) - 0.5).abs() < 1e-6);
        assert!((ev.score(2.0, 0.0) - logistic(2.0)).abs() < 1e-6);
        let abs = ev.clone().with_relative_to_root(false);
        assert!((abs.score(2.0, 7.0) - logistic(2.0)).abs() < 1e-6);
        // stock path is unchanged
        let le = LeafEval::new(None);
        let stock = 1.0 / (1.0 + (-0.0125f32 * (30.0 - 10.0)).exp());
        assert_eq!(le.score(30.0, 10.0), stock);
    }

    #[test]
    fn bytes_roundtrip_and_validation() {
        let mut w = vec![0.0; FEATURE_LEN];
        w[3] = 1.25;
        let lin = Evaluator::linear(w, -2.0).unwrap();
        let back = Evaluator::from_bytes(&lin.to_bytes()).unwrap();
        let mut b = Vec::new();
        let x = {
            let mut x = vec![0.0; FEATURE_LEN];
            x[3] = 2.0;
            x
        };
        assert_eq!(back.logit_from_features(&x, &mut b), 0.5);

        let hidden = 2;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN];
        w1[FEATURE_LEN + 5] = 3.0;
        let mlp = Evaluator::mlp(hidden, w1, vec![0.0, 1.0], vec![1.0, 2.0], 0.5).unwrap();
        let bytes = mlp.to_bytes();
        let back = Evaluator::from_bytes(&bytes).unwrap();
        let mut x = vec![0.0; FEATURE_LEN];
        x[5] = 1.0;
        // h = [0, 3 + 1] -> 0 + 2*4 + 0.5
        assert_eq!(back.logit_from_features(&x, &mut b), 8.5);
        assert_eq!(back.hidden_size(), 2);

        assert!(Evaluator::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        let mut bad = bytes.clone();
        bad[8] = 99; // feature version
        assert!(Evaluator::from_bytes(&bad).is_err());
    }

    #[test]
    fn v1_bytes_unchanged_for_non_residual() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 1.0).unwrap();
        let bytes = ev.to_bytes();
        assert_eq!(&bytes[4..8], &1u32.to_le_bytes());
        assert_eq!(bytes.len(), 24 + (FEATURE_LEN + 1) * 4);
        assert!(!Evaluator::from_bytes(&bytes).unwrap().is_residual());
    }

    #[test]
    fn residual_roundtrip_v2() {
        let mut w = vec![0.0; FEATURE_LEN];
        w[idx("own.p0.hp")] = 1.5;
        let lin = Evaluator::linear(w, 0.25).unwrap().with_stock_coef(0.0063).unwrap();
        let bytes = lin.to_bytes();
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        assert_eq!(bytes.len(), 32 + (FEATURE_LEN + 1) * 4);
        let back = Evaluator::from_bytes(&bytes).unwrap();
        assert_eq!(back.stock_coef, Some(0.0063));
        let s = state_with_hp(80, 30);
        let mut sc = Scratch::new();
        assert_eq!(lin.logit(&s, &mut sc), back.logit(&s, &mut sc));

        let hidden = 2;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN];
        w1[FEATURE_LEN + 5] = 3.0;
        let mlp = Evaluator::mlp(hidden, w1, vec![0.0, 1.0], vec![1.0, 2.0], 0.5)
            .unwrap()
            .with_stock_coef(-0.01)
            .unwrap();
        let back = Evaluator::from_bytes(&mlp.to_bytes()).unwrap();
        assert_eq!(back.stock_coef, Some(-0.01));
        assert_eq!(back.hidden_size(), 2);
        assert_eq!(mlp.logit(&s, &mut sc), back.logit(&s, &mut sc));
    }

    #[test]
    fn residual_with_zero_net_is_scaled_stock() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0)
            .unwrap()
            .with_stock_coef(0.00625)
            .unwrap();
        let mut sc = Scratch::new();
        for s in [state_with_hp(100, 40), state_with_hp(20, 90), State::default()] {
            let z = ev.logit(&s, &mut sc);
            let expected = 0.00625 * evaluate(&s);
            assert!((z - expected).abs() < 1e-5, "z = {z}, expected = {expected}");
        }
    }

    #[test]
    fn feature_version_2_roundtrip_and_isolation() {
        use crate::features::FEATURE_LEN_V2;
        let hidden = 3;
        let mut w1 = vec![0.0; hidden * FEATURE_LEN_V2];
        w1[FEATURE_LEN_V2 + 700] = 0.5;
        w1[FEATURE_LEN_V2 * 2 + FEATURE_LEN_V2 - 1] = 1.5;
        let ev = Evaluator::mlp_v(2, hidden, w1.clone(), vec![0.1; hidden], vec![1.0, 2.0, 3.0], 0.2)
            .unwrap()
            .with_stock_coef(0.01)
            .unwrap();
        assert_eq!(ev.feature_version(), 2);
        let bytes = ev.to_bytes();
        assert_eq!(&bytes[8..12], &2u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &(FEATURE_LEN_V2 as u32).to_le_bytes());
        let back = Evaluator::from_bytes(&bytes).unwrap();
        assert_eq!(back.feature_version(), 2);
        assert_eq!(back.stock_coef, Some(0.01));
        let s = state_with_hp(70, 30);
        let mut sc = Scratch::new();
        assert_eq!(ev.logit(&s, &mut sc), back.logit(&s, &mut Scratch::new()));
        // wrong input length for the declared version is rejected
        assert!(Evaluator::mlp_v(2, hidden, vec![0.0; hidden * FEATURE_LEN], vec![0.0; hidden], vec![0.0; hidden], 0.0).is_err());
        assert!(Evaluator::linear_v(3, vec![], 0.0).is_err());
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&3u32.to_le_bytes());
        assert!(Evaluator::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[16..20].copy_from_slice(&(FEATURE_LEN as u32).to_le_bytes()); // v2 header with v1 n_in
        assert!(Evaluator::from_bytes(&bad).is_err());
    }

    #[test]
    fn corrupt_v2_rejected() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0)
            .unwrap()
            .with_stock_coef(0.0125)
            .unwrap();
        let bytes = ev.to_bytes();
        assert!(Evaluator::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        assert!(Evaluator::from_bytes(&bytes[..28]).is_err());
        let mut bad = bytes.clone();
        bad[24] = 0x80; // unknown flag bit
        assert!(Evaluator::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[28..32].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(Evaluator::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[4] = 3; // unknown format
        assert!(Evaluator::from_bytes(&bad).is_err());
        assert!(ev.clone().with_stock_coef(f32::INFINITY).is_err());
    }

    #[test]
    fn material_zero_is_bit_identical_and_positive_prefers_material() {
        let w = vec![0.0f32; crate::features::feature_len(1).unwrap()];
        let base = Evaluator::linear(w, 0.0).unwrap();
        let blended = base.clone().with_material(0.2).unwrap();
        // two states the (all-zero) net cannot tell apart: equal score 0.5
        let rich = state_with_hp(100, 100);
        let mut poor = state_with_hp(100, 100);
        poor.side_one.pokemon.pkmn[1].hp = 0;
        poor.side_one.get_active().hp = 20;
        let (a, b) = (LeafEval::new(Some(&base)), LeafEval::new(Some(&blended)));
        let zero = base.clone().with_material(0.0).unwrap();
        let off = LeafEval::new(Some(&zero));
        for st in [&rich, &poor] {
            let raw = a.score(0.3, 0.1);
            assert_eq!(off.score_state(st, 0.3, 0.1).to_bits(), raw.to_bits());
            assert_eq!(a.score_state(st, 0.3, 0.1).to_bits(), raw.to_bits());
        }
        assert!(material_value(&rich) > material_value(&poor));
        assert!(b.score_state(&rich, 0.0, 0.0) > b.score_state(&poor, 0.0, 0.0));
        let m = b.score_state(&rich, 0.0, 0.0);
        assert!((m - (0.8 * 0.5 + 0.2 * material_value(&rich))).abs() < 1e-6);
        assert!(base.clone().with_material(1.0).is_err());
        assert!(base.clone().with_material(-0.1).is_err());
    }

    #[test]
    fn leaf_scale_scales_the_logit_gap() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0).unwrap();
        assert_eq!(ev.leaf_scale, 1.0);
        let base = ev.score(0.7, 0.2);
        assert!((base - logistic(0.5)).abs() < 1e-6);
        let ev2 = ev.clone().with_leaf_scale(2.0).unwrap();
        assert!((ev2.score(0.7, 0.2) - logistic(1.0)).abs() < 1e-6);
        let abs = ev.clone().with_relative_to_root(false).with_leaf_scale(3.0).unwrap();
        assert!((abs.score(0.5, 99.0) - logistic(1.5)).abs() < 1e-6);
    }

    #[test]
    fn leaf_scale_rejects_bad_values() {
        let ev = Evaluator::linear(vec![0.0; FEATURE_LEN], 0.0).unwrap();
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(ev.clone().with_leaf_scale(bad).is_err(), "{bad}");
        }
    }
}

#[cfg(test)]
mod fast_path_tests {
    use super::*;
    use crate::engine::state::PokemonVolatileStatus as V;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    fn random_mlp(ver: u32, h: usize, seed: u64) -> Evaluator {
        let mut r = Lcg(seed);
        let n_in = feature_len(ver).unwrap();
        let w1: Vec<f32> = (0..h * n_in).map(|_| r.unit() * 0.2).collect();
        let b1: Vec<f32> = (0..h).map(|_| r.unit() * 0.3).collect();
        let w2: Vec<f32> = (0..h).map(|_| r.unit()).collect();
        Evaluator::mlp_v(ver, h, w1, b1, w2, 0.1)
            .unwrap()
            .with_stock_coef(0.0125)
            .unwrap()
    }

    #[test]
    fn fast_path_matches_reference() {
        let text = include_str!("../data/gen9randombattle.txt");
        let states: Vec<State> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(State::deserialize)
            .collect();
        assert!(states.len() > 50);
        let mut rng = Lcg(7);
        let combos: Vec<(u32, usize)> = [1u32, 2]
            .iter()
            .flat_map(|&v| [8usize, 16, 32, 64].iter().map(move |&h| (v, h)))
            .collect();
        for &(ver, h) in &combos {
            let ev = random_mlp(ver, h, 100 + h as u64 + 1000 * ver as u64);
            let mut maxdiff = 0.0f32;
            for s0 in &states {
                // one Scratch per team (stats fixed), reused over mutated variants so the cache is exercised
                let mut scratch = Scratch::new();
                for round in 0..4 {
                    let mut s = s0.clone();
                    if round > 0 {
                        // mutate dynamic and static fields
                        for side in [&mut s.side_one, &mut s.side_two] {
                            for p in side.pokemon.pkmn.iter_mut() {
                                if rng.next() % 3 == 0 {
                                    p.hp = (rng.next() % (p.maxhp.max(1) as u32 + 1)) as i16;
                                }
                                if rng.next() % 5 == 0 {
                                    p.status = unsafe { std::mem::transmute::<u8, PokemonStatus>((rng.next() % 7) as u8) };
                                }
                                if rng.next() % 7 == 0 {
                                    p.terastallized = !p.terastallized;
                                    p.types.1 = p.types.0;
                                }
                                if rng.next() % 9 == 0 {
                                    p.item = Items::NONE;
                                }
                                // v2 inputs: item / ability identity, move set, pp
                                if rng.next() % 4 == 0 {
                                    p.item = features_v2::ITEM_LIST[rng.next() as usize % features_v2::ITEM_LIST.len()];
                                }
                                if rng.next() % 4 == 0 {
                                    p.ability = features_v2::ABILITY_LIST[rng.next() as usize % features_v2::ABILITY_LIST.len()];
                                }
                                if rng.next() % 4 == 0 {
                                    std::mem::swap(&mut p.moves.m0, &mut p.moves.m2);
                                }
                                if rng.next() % 4 == 0 {
                                    p.moves.m1.pp = (rng.next() % 20) as i8;
                                    p.moves.m3.pp = 0;
                                }
                            }
                            side.side_conditions.light_screen = (rng.next() % 6) as i8;
                            side.side_conditions.tailwind = (rng.next() % 5) as i8;
                            side.attack_boost = (rng.next() % 13) as i8 - 6;
                            side.speed_boost = (rng.next() % 13) as i8 - 6;
                            side.side_conditions.spikes = (rng.next() % 4) as i8;
                            side.side_conditions.reflect = (rng.next() % 2) as i8;
                            side.wish.0 = (rng.next() % 2) as i8;
                            side.force_switch = rng.next() % 4 == 0;
                            if rng.next() % 2 == 0 {
                                side.volatile_statuses.insert(V::SUBSTITUTE);
                                side.volatile_statuses.insert(V::PERISH2);
                                side.volatile_statuses.insert(V::LOCKEDMOVE);
                            }
                            if rng.next() % 3 == 0 {
                                side.active_index = unsafe { std::mem::transmute::<u8, crate::state::PokemonIndex>((rng.next() % 6) as u8) };
                            }
                        }
                        s.trick_room.active = rng.next() % 4 == 0;
                        s.trick_room.turns_remaining = (rng.next() % 6) as i8;
                        s.terrain.turns_remaining = (rng.next() % 6) as i8;
                    }
                    let a = ev.logit(&s, &mut scratch);
                    let b = ev.logit_reference(&s, &mut Scratch::new());
                    maxdiff = maxdiff.max((a - b).abs());
                }
            }
            assert!(maxdiff <= 1e-5, "version {} hidden {}: max abs diff {}", ver, h, maxdiff);
        }
    }
}

#[cfg(test)]
mod embed_tests {
    use super::*;
    use crate::engine::abilities::Abilities;
    use crate::choices::Choices;
    use crate::state::PokemonStatus;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn unit(&mut self) -> f32 {
            (self.next() % 20001) as f32 / 10000.0 - 1.0
        }
    }

    fn random_embed(dims: [usize; N_TABLES], seed: u64) -> Embed {
        let mut r = Lcg(seed);
        let tables: [Vec<f32>; N_TABLES] = std::array::from_fn(|t| {
            (0..features_v4::vocab_size(t) * dims[t]).map(|_| r.unit() * 0.5).collect()
        });
        Embed::new(dims, tables).unwrap()
    }

    fn random_v4(h: usize, dims: [usize; N_TABLES], seed: u64) -> Evaluator {
        let mut r = Lcg(seed);
        let emb = random_embed(dims, seed + 1);
        let n_in = FEATURE_LEN_V4 + emb.input_len();
        let w1: Vec<f32> = (0..h * n_in).map(|_| r.unit() * 0.2).collect();
        let b1: Vec<f32> = (0..h).map(|_| r.unit() * 0.3).collect();
        let w2: Vec<f32> = (0..h).map(|_| r.unit()).collect();
        Evaluator::mlp_embed(h, w1, b1, w2, 0.1, emb).unwrap()
    }

    fn states() -> Vec<State> {
        include_str!("../data/gen9randombattle.txt")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(State::deserialize)
            .collect()
    }

    #[test]
    fn v4_fast_path_matches_reference() {
        let states = states();
        let mut rng = Lcg(11);
        for &h in &[8usize, 16, 64, 128] {
            let ev = random_v4(h, [8, 4, 6, 5], 500 + h as u64).with_stock_coef(0.0125).unwrap();
            assert_eq!(ev.feature_version(), 4);
            let mut maxdiff = 0.0f32;
            for s0 in states.iter().take(40) {
                let mut scratch = Scratch::new();
                for round in 0..4 {
                    let mut s = s0.clone();
                    if round > 0 {
                        for side in [&mut s.side_one, &mut s.side_two] {
                            for p in side.pokemon.pkmn.iter_mut() {
                                if rng.next() % 3 == 0 {
                                    p.hp = (rng.next() % (p.maxhp.max(1) as u32 + 1)) as i16;
                                }
                                if rng.next() % 4 == 0 {
                                    p.item = features_v2::ITEM_LIST[rng.next() as usize % features_v2::ITEM_LIST.len()];
                                }
                                if rng.next() % 6 == 0 {
                                    p.item = if rng.next() % 2 == 0 { Items::UNKNOWNITEM } else { Items::NONE };
                                }
                                if rng.next() % 4 == 0 {
                                    p.ability = features_v2::ABILITY_LIST[rng.next() as usize % features_v2::ABILITY_LIST.len()];
                                }
                                if rng.next() % 6 == 0 {
                                    p.ability = Abilities::NONE;
                                }
                                if rng.next() % 4 == 0 {
                                    std::mem::swap(&mut p.moves.m0, &mut p.moves.m2);
                                }
                                if rng.next() % 5 == 0 {
                                    p.moves.m3.id = Choices::NONE;
                                }
                                if rng.next() % 9 == 0 {
                                    p.id = crate::pokemon::PokemonName::NONE;
                                }
                                if rng.next() % 5 == 0 {
                                    p.status = unsafe { std::mem::transmute::<u8, PokemonStatus>((rng.next() % 7) as u8) };
                                }
                            }
                            side.speed_boost = (rng.next() % 13) as i8 - 6;
                            if rng.next() % 3 == 0 {
                                side.active_index = unsafe { std::mem::transmute::<u8, crate::state::PokemonIndex>((rng.next() % 6) as u8) };
                            }
                        }
                    }
                    let a = ev.logit(&s, &mut scratch);
                    let b = ev.logit_reference(&s, &mut Scratch::new());
                    maxdiff = maxdiff.max((a - b).abs());
                }
            }
            assert!(maxdiff <= 2e-4, "hidden {}: max abs diff {}", h, maxdiff);
        }
    }

    #[test]
    fn v4_identity_changes_the_logit() {
        let ev = random_v4(16, [8, 8, 8, 8], 3);
        let mut s = states().remove(0);
        let mut sc = Scratch::new();
        let base = ev.logit(&s, &mut Scratch::new());
        s.side_two.get_active().item = Items::UNKNOWNITEM;
        let unk = ev.logit(&s, &mut sc);
        s.side_two.get_active().item = Items::NONE;
        let none = ev.logit(&s, &mut Scratch::new());
        assert!((base - unk).abs() > 1e-6 || (base - none).abs() > 1e-6);
        assert!((unk - none).abs() > 1e-6, "unknown and none items must be distinct inputs");
        // a fainted Pokemon sees only the pad identity: its species does not matter
        let mut a = states().remove(1);
        a.side_one.pokemon.pkmn[1].hp = 0;
        let mut b = a.clone();
        b.side_one.pokemon.pkmn[1].id = a.side_one.pokemon.pkmn[2].id;
        assert_eq!(ev.logit_reference(&a, &mut Scratch::new()), ev.logit_reference(&b, &mut Scratch::new()));
    }

    #[test]
    fn v4_bytes_roundtrip() {
        let ev = random_v4(16, [8, 5, 4, 6], 21).with_stock_coef(0.0125).unwrap();
        let bytes = ev.to_bytes();
        assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &(FEATURE_LEN_V4 as u32).to_le_bytes());
        let back = Evaluator::from_bytes(&bytes).unwrap();
        assert_eq!(back.to_bytes(), bytes);
        assert_eq!(back.stock_coef, Some(0.0125));
        assert_eq!(back.embed().unwrap().dims(), [8, 5, 4, 6]);
        for s in states().iter().take(10) {
            assert_eq!(ev.logit(s, &mut Scratch::new()), back.logit(s, &mut Scratch::new()));
        }
        // plain (non residual) too
        let plain = random_v4(8, [2, 2, 2, 2], 4);
        let back = Evaluator::from_bytes(&plain.to_bytes()).unwrap();
        assert!(!back.is_residual());
        // corruption is rejected
        assert!(Evaluator::from_bytes(&bytes[..bytes.len() - 4]).is_err());
        let mut bad = bytes.clone();
        bad[32] ^= 1; // vocabulary hash
        let err = Evaluator::from_bytes(&bad).unwrap_err();
        assert!(err.contains("vocabulary"), "{err}");
        let mut bad = bytes.clone();
        bad[8..12].copy_from_slice(&2u32.to_le_bytes()); // format 3 needs version 4
        assert!(Evaluator::from_bytes(&bad).is_err());
        let mut bad = bytes.clone();
        bad[40..44].copy_from_slice(&0u32.to_le_bytes()); // zero dim
        assert!(Evaluator::from_bytes(&bad).is_err());
        // v4 without embeddings cannot be built from the plain constructors
        assert!(Evaluator::mlp_v(4, 2, vec![0.0; 2 * FEATURE_LEN_V4], vec![0.0; 2], vec![0.0; 2], 0.0).is_err());
        assert!(Evaluator::linear_v(4, vec![0.0; FEATURE_LEN_V4], 0.0).is_err());
    }

    #[test]
    fn older_formats_still_load_and_are_unchanged() {
        // formats 1 / 2 are written exactly as before for every non-embedding evaluator
        let lin = Evaluator::linear(vec![0.0; FEATURE_LEN], 1.0).unwrap();
        assert_eq!(&lin.to_bytes()[4..8], &1u32.to_le_bytes());
        let v3 = Evaluator::mlp_v(3, 2, vec![0.01; 2 * crate::features::FEATURE_LEN_V3], vec![0.0; 2], vec![1.0; 2], 0.0)
            .unwrap()
            .with_stock_coef(0.0125)
            .unwrap();
        let bytes = v3.to_bytes();
        assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
        let back = Evaluator::from_bytes(&bytes).unwrap();
        assert_eq!(back.feature_version(), 3);
        assert!(back.embed().is_none());
        let s = &states()[0];
        assert_eq!(v3.logit(s, &mut Scratch::new()), back.logit(s, &mut Scratch::new()));
    }

    #[test]
    fn reference_input_layout() {
        // embedding input = per slot [species | item | ability | mean of the 4 moves]
        let dims = [2, 3, 1, 2];
        let emb = random_embed(dims, 9);
        let s = &states()[0];
        let ids = features_v4::identity(s, SideReference::SideOne);
        let mut out = vec![0.0f32; emb.input_len()];
        emb.fill_inputs(&ids, &mut out);
        let w = emb.slot_width();
        assert_eq!(w, 8);
        // slot 3 (own bench 2)
        let ids3 = &ids[3 * IDS_PER_SLOT..4 * IDS_PER_SLOT];
        let o = &out[3 * w..4 * w];
        assert_eq!(&o[0..2], &emb.table(0)[ids3[0] as usize * 2..ids3[0] as usize * 2 + 2]);
        assert_eq!(&o[2..5], &emb.table(1)[ids3[1] as usize * 3..ids3[1] as usize * 3 + 3]);
        assert_eq!(o[5], emb.table(2)[ids3[2] as usize]);
        for c in 0..2 {
            let mean: f32 = (0..4).map(|m| emb.table(3)[ids3[3 + m] as usize * 2 + c]).sum::<f32>() / 4.0;
            assert!((o[6 + c] - mean).abs() < 1e-6);
        }
    }

    #[test]
    fn embed_new_validates_shapes() {
        let dims = [2, 2, 2, 2];
        let good: [Vec<f32>; N_TABLES] = std::array::from_fn(|t| vec![0.0; features_v4::vocab_size(t) * 2]);
        assert!(Embed::new(dims, good.clone()).is_ok());
        let mut bad = good.clone();
        bad[1].pop();
        assert!(Embed::new(dims, bad).is_err());
        assert!(Embed::new([0, 2, 2, 2], good).is_err());
    }
}
