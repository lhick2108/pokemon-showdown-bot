//! Policy priors for MCTS selection (PUCT, decoupled per side).
//!
//! A `PolicyNet` scores the legal options of one side at a node. The legal options'
//! logits are softmaxed into the prior `P_i` of that side's `MoveNode`s, and selection
//! uses `Q_i + c * P_i * sqrt(N) / (1 + n_i)` (see `Node::maximize_puct_for_side` in
//! `mcts.rs`).
//!
//! # Architecture
//!
//! The state is encoded exactly like the value evaluator's input (feature version 2,
//! sparse first-layer fast path, `learned::hidden_acc`), seen from the acting side:
//! `hr = relu(W1 x + b1)`. The v2 move features are *aggregated over the four moves*, so
//! the hidden layer alone cannot tell the moves apart; every option therefore also gets
//! its own `N_AFEAT` **action features** (`action_features`: damage / KO / type
//! effectiveness / property flags of that very move, matchup of the switch target, ...):
//!
//! ```text
//! logit(option s) = b2[slot] + W2[slot] . hr + v . relu(P hr + c + A af(option))
//! ```
//!
//! with `slot` the option's action slot (4 moves, 4 Tera moves, 5 switch targets in
//! feature-slot order) and `af` sparse, so `A af` is a handful of column adds. The same net
//! serves both sides: side two is evaluated from its own perspective (own / opp blocks
//! swapped), which is how the training data (always "our side = side one") is mirrored.
//!
//! # Weights file `.ppol` (little endian)
//! ```text
//! magic    4 bytes  "PPOL"
//! format   u32      1
//! feat_ver u32      FEATURE_VERSION used to train (1 or 2)
//! n_in     u32      feature length
//! n_hidden u32      16 / 32 / 64 / 128
//! n_act    u32      N_ACTIONS (13)
//! n_afeat  u32      N_AFEAT
//! n_emb    u32      N_EMB
//! W1[n_hidden][n_in], b1[n_hidden], W2[n_act][n_hidden], b2[n_act],
//! P[n_emb][n_hidden], c[n_emb], A[n_emb][n_afeat], v[n_emb]        (all f32)
//! ```

use crate::choices::{Choice, MoveCategory};
use crate::engine::abilities::Abilities;
use crate::engine::damage_calc::type_effectiveness_modifier;
use crate::engine::items::Items;
use crate::engine::state::MoveChoice;
use crate::features::{feature_len, FEATURE_VERSION_2};
use crate::features_v2::{
    ability_immune, accumulate_move, effective_speed, has_stab, is_damaging, moves_of, stage_mult,
    MoveAcc, N_FLAGS,
};
use crate::learned::{hidden_acc, Scratch};
use crate::mcts::MoveNode;
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, SideReference, State};
use std::convert::TryInto;
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAGIC: &[u8; 4] = b"PPOL";
pub const FORMAT_VERSION: u32 = 1;
/// 4 moves + 4 Tera moves + 5 switch targets.
pub const N_ACTIONS: usize = 13;
/// Action features per option.
pub const N_AFEAT: usize = 58;
/// Width of the per-option embedding.
pub const N_EMB: usize = 16;

// action feature layout (see `action_features`)
const F_MOVE: usize = 0;
const F_TERA: usize = 1;
const F_SWITCH: usize = 2;
const F_DAMAGING: usize = 3; // 3 damaging, 4 physical, 5 special
const F_BP: usize = 6;
const F_EFF: usize = 7; // 7 log2 effectiveness, 8 immune, 9 stab, 10 accuracy, 11 prio+, 12 prio-
const F_FRAC: usize = 13; // 13 damage fraction, 14 ko, 15 ko and faster
const F_FLAGS: usize = 16; // N_FLAGS (30) move flags
const F_RECOVERY: usize = F_FLAGS + N_FLAGS; // 46, then setup stages
const F_SWITCH_BLOCK: usize = 48; // 48 hp, 49 status, 50 def mult, 51 off mult, 52 speed, 53 own dmg, 54 opp dmg
const F_CTX: usize = 55; // 55 own active fainted, 56 opp damage on own active, 57 own active hp

static NEXT_NET_ID: AtomicU64 = AtomicU64::new(1 << 40);

/// Action slot of `choice` for `side` (see module docs); `None` for choices that carry no
/// learned logit (`None`, team preview).
pub fn action_slot(side: &Side, choice: &MoveChoice) -> Option<usize> {
    match choice {
        MoveChoice::Move(m) | MoveChoice::MoveMega(m) => Some(*m as usize),
        MoveChoice::MoveTera(m) => Some(4 + *m as usize),
        MoveChoice::Switch(p) => {
            let a = side.active_index as usize;
            let k = *p as usize;
            if k == a {
                return None;
            }
            let slot = if k < a { k + 1 } else { k };
            Some(8 + slot - 1)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Action features
// ---------------------------------------------------------------------------

fn log_eff(eff: f32) -> f32 {
    if eff <= 0.0 {
        -1.0
    } else {
        (eff.log2() / 2.0).clamp(-1.0, 1.0)
    }
}

fn eff_vs(c: &Choice, def: &Pokemon) -> f32 {
    if ability_immune(def, c.move_type) {
        0.0
    } else {
        type_effectiveness_modifier(&c.move_type, def)
    }
}

/// Rough expected damage of `c` as a fraction of `def`'s max hp (average roll, accuracy-weighted,
/// STAB / burn / Life Orb / Choice item; no weather, screens or ability modifiers). Same model as
/// the `assess` of `features_v2`, per move and with explicit stat stages.
fn move_frac(
    att: &Pokemon,
    att_boosts: (i8, i8),
    def: &Pokemon,
    def_boosts: (i8, i8),
    c: &Choice,
    stab: bool,
) -> f32 {
    let eff = eff_vs(c, def);
    if eff == 0.0 {
        return 0.0;
    }
    let physical = c.category == MoveCategory::Physical;
    let (a, d) = if physical {
        (att.attack as f32 * stage_mult(att_boosts.0), def.defense as f32 * stage_mult(def_boosts.0))
    } else {
        (
            att.special_attack as f32 * stage_mult(att_boosts.1),
            def.special_defense as f32 * stage_mult(def_boosts.1),
        )
    };
    let lvl_term = 2.0 * att.level as f32 / 5.0 + 2.0;
    let mut dmg = (lvl_term * c.base_power * a / d.max(1.0) / 50.0 + 2.0) * 0.925 * eff;
    if stab {
        dmg *= if att.ability == Abilities::ADAPTABILITY { 2.0 } else { 1.5 };
    }
    if physical && att.status == PokemonStatus::BURN && att.ability != Abilities::GUTS {
        dmg *= 0.5;
    }
    match att.item {
        Items::LIFEORB => dmg *= 1.3,
        Items::CHOICEBAND if physical => dmg *= 1.5,
        Items::CHOICESPECS if !physical => dmg *= 1.5,
        _ => {}
    }
    let acc = if c.accuracy <= 0.0 { 1.0 } else { (c.accuracy / 100.0).min(1.0) };
    dmg * acc / def.maxhp.max(1) as f32
}

fn best_frac(att: &Pokemon, att_boosts: (i8, i8), def: &Pokemon, def_boosts: (i8, i8)) -> f32 {
    let mut best = 0.0f32;
    for m in moves_of(att) {
        let c = &m.choice;
        if m.id == crate::choices::Choices::NONE || !is_damaging(c) {
            continue;
        }
        best = best.max(move_frac(att, att_boosts, def, def_boosts, c, has_stab(att, c.move_type)));
    }
    best
}

fn boosts_of(side: &Side) -> (i8, i8) {
    (side.attack_boost, side.special_attack_boost)
}

fn def_boosts_of(side: &Side) -> (i8, i8) {
    (side.defense_boost, side.special_defense_boost)
}

/// Per-node, per-side quantities shared by all options of that side.
pub struct ActionCtx<'a> {
    own: &'a Side,
    opp: &'a Side,
    me: &'a Pokemon,
    them: &'a Pokemon,
    my_atk: (i8, i8),
    their_atk: (i8, i8),
    their_def: (i8, i8),
    /// context features shared by every option (own active fainted, damage threat, own hp)
    ctx: [f32; 3],
    faster: bool,
}

impl<'a> ActionCtx<'a> {
    pub fn new(state: &'a State, own_ref: SideReference) -> ActionCtx<'a> {
        let (own, opp) = match own_ref {
            SideReference::SideOne => (&state.side_one, &state.side_two),
            SideReference::SideTwo => (&state.side_two, &state.side_one),
        };
        let me = own.get_active_immutable();
        let them = opp.get_active_immutable();
        let (my_atk, my_def) = (boosts_of(own), def_boosts_of(own));
        let (their_atk, their_def) = (boosts_of(opp), def_boosts_of(opp));
        let ctx = [
            (me.hp <= 0) as u8 as f32,
            best_frac(them, their_atk, me, my_def).min(1.5) / 1.5,
            (me.hp.max(0) as f32 / me.maxhp.max(1) as f32).min(1.0),
        ];
        let (s_me, s_them) = (effective_speed(me, own), effective_speed(them, opp));
        let faster = (s_me > s_them) != state.trick_room.active && s_me != s_them;
        ActionCtx { own, opp, me, them, my_atk, their_atk, their_def, ctx, faster }
    }

    /// The `N_AFEAT` features describing `choice` for the acting side.
    pub fn features(&self, choice: &MoveChoice, out: &mut [f32; N_AFEAT]) {
        *out = [0.0; N_AFEAT];
        let (me, them) = (self.me, self.them);
        out[F_CTX..F_CTX + 3].copy_from_slice(&self.ctx);
        match choice {
            MoveChoice::Move(mi) | MoveChoice::MoveTera(mi) | MoveChoice::MoveMega(mi) => {
                let tera = matches!(choice, MoveChoice::MoveTera(_));
                out[F_MOVE] = 1.0;
                if tera {
                    out[F_TERA] = 1.0;
                }
                let mv = moves_of(me)[*mi as usize];
                if mv.id == crate::choices::Choices::NONE {
                    return;
                }
                let c = &mv.choice;
                let stab = has_stab(me, c.move_type) || (tera && me.tera_type == c.move_type);
                if is_damaging(c) {
                    out[F_DAMAGING] = 1.0;
                    out[F_DAMAGING + 1 + (c.category != MoveCategory::Physical) as usize] = 1.0;
                    out[F_BP] = (c.base_power / 150.0).min(1.5);
                    let eff = eff_vs(c, them);
                    out[F_EFF] = log_eff(eff);
                    out[F_EFF + 1] = (eff == 0.0) as u8 as f32;
                    out[F_EFF + 2] = stab as u8 as f32;
                    let frac = move_frac(me, self.my_atk, them, self.their_def, c, stab);
                    out[F_FRAC] = frac.min(1.5) / 1.5;
                    let ko = them.hp > 0 && frac * them.maxhp.max(1) as f32 >= them.hp as f32;
                    out[F_FRAC + 1] = ko as u8 as f32;
                    out[F_FRAC + 2] = (ko && self.faster) as u8 as f32;
                }
                out[F_EFF + 3] = if c.accuracy <= 0.0 { 1.0 } else { (c.accuracy / 100.0).min(1.0) };
                out[F_EFF + 4] = (c.priority > 0) as u8 as f32;
                out[F_EFF + 5] = (c.priority < 0) as u8 as f32;
                let mut acc = MoveAcc::default();
                accumulate_move(&mut acc, me, mv);
                for (i, f) in acc.flags.iter().enumerate() {
                    if *f {
                        out[F_FLAGS + i] = 1.0;
                    }
                }
                out[F_RECOVERY] = acc.recovery_frac.min(1.0);
                out[F_RECOVERY + 1] = acc.setup_stages.clamp(0, 6) as f32 / 6.0;
            }
            MoveChoice::Switch(pi) => {
                out[F_SWITCH] = 1.0;
                let t = &self.own.pokemon.pkmn[*pi as usize];
                let b = F_SWITCH_BLOCK;
                out[b] = (t.hp.max(0) as f32 / t.maxhp.max(1) as f32).min(1.0);
                out[b + 1] = (t.status != PokemonStatus::NONE) as u8 as f32;
                let def_mult = type_effectiveness_modifier(&them.types.0, t)
                    .max(type_effectiveness_modifier(&them.types.1, t));
                out[b + 2] = log_eff(def_mult);
                let off_mult = type_effectiveness_modifier(&t.types.0, them)
                    .max(type_effectiveness_modifier(&t.types.1, them));
                out[b + 3] = log_eff(off_mult);
                let ratio = t.speed as f32 / effective_speed(them, self.opp);
                out[b + 4] = (ratio.max(1e-3).log2() / 2.0).clamp(-1.0, 1.0);
                out[b + 5] = best_frac(t, (0, 0), them, self.their_def).min(1.5) / 1.5;
                out[b + 6] = best_frac(them, self.their_atk, t, (0, 0)).min(1.5) / 1.5;
            }
            _ => {}
        }
    }
}

/// The `N_AFEAT` features describing `choice` for the acting side `own_ref` of `state`.
pub fn action_features(
    state: &State,
    own_ref: SideReference,
    choice: &MoveChoice,
    out: &mut [f32; N_AFEAT],
) {
    ActionCtx::new(state, own_ref).features(choice, out)
}

// ---------------------------------------------------------------------------
// The net
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PolicyNet {
    id: u64,
    feat_ver: u32,
    hidden: usize,
    /// transposed first layer: `w1t[i * hidden + j]`
    w1t: Vec<f32>,
    b1: Vec<f32>,
    /// `[action][hidden]`
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// `[emb][hidden]`
    p: Vec<f32>,
    c: Vec<f32>,
    /// transposed action-feature layer: `at[d * N_EMB + e]`
    at: Vec<f32>,
    v: Vec<f32>,
    avx2: bool,
}

/// Raw (file-order) weights, for construction from Python / training code.
pub struct PolicyWeights {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub p: Vec<f32>,
    pub c: Vec<f32>,
    /// `[emb][afeat]`
    pub a: Vec<f32>,
    pub v: Vec<f32>,
}

#[inline(always)]
fn eval_generic<const H: usize>(
    net: &PolicyNet,
    state: &State,
    own: SideReference,
    choices: &[MoveChoice],
    scratch: &mut Scratch,
    out: &mut [f32],
) {
    let acc = if net.feat_ver == FEATURE_VERSION_2 {
        hidden_acc::<H, true>(net.id, &net.w1t, &net.b1, state, own, scratch)
    } else {
        hidden_acc::<H, false>(net.id, &net.w1t, &net.b1, state, own, scratch)
    };
    let mut hr = [0.0f32; H];
    for j in 0..H {
        hr[j] = acc[j].max(0.0);
    }
    // P hr + c, once per node
    let mut ph = [0.0f32; N_EMB];
    for e in 0..N_EMB {
        let row: &[f32; H] = net.p[e * H..e * H + H].try_into().unwrap();
        let mut z = net.c[e];
        for j in 0..H {
            z += row[j] * hr[j];
        }
        ph[e] = z;
    }
    let side = match own {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    let mut af = [0.0f32; N_AFEAT];
    let actx = ActionCtx::new(state, own);
    for (o, choice) in out.iter_mut().zip(choices) {
        let slot = match action_slot(side, choice) {
            Some(s) => s,
            None => {
                *o = f32::NAN;
                continue;
            }
        };
        actx.features(choice, &mut af);
        let row: &[f32; H] = net.w2[slot * H..slot * H + H].try_into().unwrap();
        let mut z = net.b2[slot];
        for j in 0..H {
            z += row[j] * hr[j];
        }
        let mut e = ph;
        for d in 0..N_AFEAT {
            let x = af[d];
            if x != 0.0 {
                let col: &[f32; N_EMB] = net.at[d * N_EMB..d * N_EMB + N_EMB].try_into().unwrap();
                for k in 0..N_EMB {
                    e[k] += x * col[k];
                }
            }
        }
        for k in 0..N_EMB {
            z += net.v[k] * e[k].max(0.0);
        }
        *o = z;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn eval_avx2<const H: usize>(
    net: &PolicyNet,
    state: &State,
    own: SideReference,
    choices: &[MoveChoice],
    scratch: &mut Scratch,
    out: &mut [f32],
) {
    eval_generic::<H>(net, state, own, choices, scratch, out)
}

impl PolicyNet {
    pub fn new(feat_ver: u32, hidden: usize, w: PolicyWeights) -> Result<PolicyNet, String> {
        if feat_ver == crate::features::FEATURE_VERSION_3 || crate::features_v4::has_identity(feat_ver) {
            return Err("policy nets support feature versions 1 and 2 only".to_string());
        }
        let n_in = feature_len(feat_ver).ok_or_else(|| format!("unknown feature version {feat_ver}"))?;
        if !matches!(hidden, 16 | 32 | 64 | 128) {
            return Err(format!("policy hidden size must be 16, 32, 64 or 128 (got {hidden})"));
        }
        let want = [
            ("w1", w.w1.len(), hidden * n_in),
            ("b1", w.b1.len(), hidden),
            ("w2", w.w2.len(), N_ACTIONS * hidden),
            ("b2", w.b2.len(), N_ACTIONS),
            ("p", w.p.len(), N_EMB * hidden),
            ("c", w.c.len(), N_EMB),
            ("a", w.a.len(), N_EMB * N_AFEAT),
            ("v", w.v.len(), N_EMB),
        ];
        for (name, got, expected) in want {
            if got != expected {
                return Err(format!("policy weights {name} have length {got}, expected {expected}"));
            }
        }
        let mut w1t = vec![0.0f32; w.w1.len()];
        for j in 0..hidden {
            for i in 0..n_in {
                w1t[i * hidden + j] = w.w1[j * n_in + i];
            }
        }
        let mut at = vec![0.0f32; w.a.len()];
        for e in 0..N_EMB {
            for d in 0..N_AFEAT {
                at[d * N_EMB + e] = w.a[e * N_AFEAT + d];
            }
        }
        #[cfg(target_arch = "x86_64")]
        let avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        Ok(PolicyNet {
            id: NEXT_NET_ID.fetch_add(1, Ordering::Relaxed),
            feat_ver,
            hidden,
            w1t,
            b1: w.b1,
            w2: w.w2,
            b2: w.b2,
            p: w.p,
            c: w.c,
            at,
            v: w.v,
            avx2,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden
    }

    pub fn feature_version(&self) -> u32 {
        self.feat_ver
    }

    /// Logits of `choices` (the legal options of side `own`) at `state`; NaN for a choice that
    /// has no action slot.
    pub fn option_logits(
        &self,
        state: &State,
        own: SideReference,
        choices: &[MoveChoice],
        scratch: &mut Scratch,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; choices.len()];
        self.option_logits_into(state, own, choices, scratch, &mut out);
        out
    }

    /// `option_logits` into a caller-provided slice of the same length as `choices`.
    pub fn option_logits_into(
        &self,
        state: &State,
        own: SideReference,
        choices: &[MoveChoice],
        scratch: &mut Scratch,
        out: &mut [f32],
    ) {
        debug_assert_eq!(choices.len(), out.len());
        #[cfg(target_arch = "x86_64")]
        if self.avx2 {
            // SAFETY: `avx2` is only set when the CPU reports AVX2.
            unsafe {
                match self.hidden {
                    16 => eval_avx2::<16>(self, state, own, choices, scratch, out),
                    32 => eval_avx2::<32>(self, state, own, choices, scratch, out),
                    64 => eval_avx2::<64>(self, state, own, choices, scratch, out),
                    _ => eval_avx2::<128>(self, state, own, choices, scratch, out),
                }
            }
            return;
        }
        match self.hidden {
            16 => eval_generic::<16>(self, state, own, choices, scratch, out),
            32 => eval_generic::<32>(self, state, own, choices, scratch, out),
            64 => eval_generic::<64>(self, state, own, choices, scratch, out),
            _ => eval_generic::<128>(self, state, own, choices, scratch, out),
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        let n_in = feature_len(self.feat_ver).unwrap();
        for v in [
            FORMAT_VERSION,
            self.feat_ver,
            n_in as u32,
            self.hidden as u32,
            N_ACTIONS as u32,
            N_AFEAT as u32,
            N_EMB as u32,
        ] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let h = self.hidden;
        let mut push = |x: f32| out.extend_from_slice(&x.to_le_bytes());
        for j in 0..h {
            for i in 0..n_in {
                push(self.w1t[i * h + j]);
            }
        }
        for x in self.b1.iter().chain(&self.w2).chain(&self.b2).chain(&self.p).chain(&self.c) {
            push(*x);
        }
        for e in 0..N_EMB {
            for d in 0..N_AFEAT {
                push(self.at[d * N_EMB + e]);
            }
        }
        for x in &self.v {
            push(*x);
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<PolicyNet, String> {
        const HEADER: usize = 32;
        if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
            return Err("not a policy prior file (bad magic)".to_string());
        }
        let u = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        if u(4) != FORMAT_VERSION {
            return Err(format!("unsupported policy prior format {}", u(4)));
        }
        let (feat_ver, n_in, hidden) = (u(8), u(12) as usize, u(16) as usize);
        let expected_in = feature_len(feat_ver).ok_or_else(|| format!("unknown feature version {feat_ver}"))?;
        if n_in != expected_in || u(20) as usize != N_ACTIONS || u(24) as usize != N_AFEAT || u(28) as usize != N_EMB {
            return Err(format!(
                "policy prior shape mismatch: n_in {n_in} (want {expected_in}), n_act {} (want {N_ACTIONS}), \
                 n_afeat {} (want {N_AFEAT}), n_emb {} (want {N_EMB})",
                u(20),
                u(24),
                u(28)
            ));
        }
        let need = hidden * n_in + hidden + N_ACTIONS * hidden + N_ACTIONS + N_EMB * hidden + N_EMB
            + N_EMB * N_AFEAT + N_EMB;
        let body = &bytes[HEADER..];
        if body.len() != need * 4 {
            return Err(format!("policy prior body is {} bytes, expected {}", body.len(), need * 4));
        }
        let f: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut at = 0;
        let mut take = |n: usize| {
            let s = f[at..at + n].to_vec();
            at += n;
            s
        };
        let w = PolicyWeights {
            w1: take(hidden * n_in),
            b1: take(hidden),
            w2: take(N_ACTIONS * hidden),
            b2: take(N_ACTIONS),
            p: take(N_EMB * hidden),
            c: take(N_EMB),
            a: take(N_EMB * N_AFEAT),
            v: take(N_EMB),
        };
        PolicyNet::new(feat_ver, hidden, w)
    }

    pub fn load(path: &str) -> Result<PolicyNet, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
        PolicyNet::from_bytes(&bytes)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("cannot write {path}: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Search integration
// ---------------------------------------------------------------------------

/// Prior settings of a search. `PriorConfig` is `Copy`: it only borrows the nets.
///
/// A side with neither a net nor a root override keeps decoupled UCB1 selection.
#[derive(Clone, Copy, Debug)]
pub struct PriorConfig<'a> {
    /// exploration constant `c` of PUCT
    pub c_puct: f32,
    /// net for side one (from side one's perspective)
    pub net_s1: Option<&'a PolicyNet>,
    /// net for side two (from side two's perspective)
    pub net_s2: Option<&'a PolicyNet>,
    /// root-only overrides (probabilities aligned with the root option order)
    pub root_s1: Option<&'a [f32]>,
    pub root_s2: Option<&'a [f32]>,
    /// the prior is mixed with the uniform distribution: `(1 - eps) p + eps / K`
    pub eps: f32,
    /// first-play urgency: unvisited options start from `parent mean - fpu_reduction`
    pub fpu_reduction: f32,
}

impl<'a> PriorConfig<'a> {
    pub fn new(c_puct: f32) -> PriorConfig<'a> {
        PriorConfig {
            c_puct,
            net_s1: None,
            net_s2: None,
            root_s1: None,
            root_s2: None,
            eps: 0.03,
            fpu_reduction: 0.0,
        }
    }
}

/// Per-search scratch (one `Scratch` per side: the static-feature cache depends on the
/// perspective).
pub struct PriorScratch {
    s: [Scratch; 2],
}

impl Default for PriorScratch {
    fn default() -> Self {
        PriorScratch {
            s: [Scratch::new(), Scratch::new()],
        }
    }
}

const MAX_OPTIONS: usize = 16;

/// Writes the prior of the legal options of `side_ref` at `state` into `options[i].prior`.
/// Returns false (leaving the priors untouched) if no prior applies to this side.
fn side_prior(
    cfg: &PriorConfig,
    state: &State,
    side_ref: SideReference,
    options: &mut [MoveNode],
    root: bool,
    scratch: &mut Scratch,
) -> bool {
    let (net, over) = match side_ref {
        SideReference::SideOne => (cfg.net_s1, cfg.root_s1),
        SideReference::SideTwo => (cfg.net_s2, cfg.root_s2),
    };
    let k = options.len();
    if k == 0 || k > MAX_OPTIONS {
        return false;
    }
    let kf = k as f32;
    let eps = cfg.eps;
    if root {
        if let Some(p) = over {
            if p.len() == k {
                let sum: f32 = p.iter().map(|x| x.max(0.0)).sum();
                for (o, x) in options.iter_mut().zip(p) {
                    let q = if sum > 0.0 { x.max(0.0) / sum } else { 1.0 / kf };
                    o.prior = (1.0 - eps) * q + eps / kf;
                }
                return true;
            }
        }
    }
    let net = match net {
        Some(n) => n,
        None => return false,
    };
    if k == 1 {
        options[0].prior = 1.0;
        return true;
    }
    let mut choices = [MoveChoice::None; MAX_OPTIONS];
    for (c, o) in choices.iter_mut().zip(options.iter()) {
        *c = o.move_choice;
    }
    let mut logits = [0.0f32; MAX_OPTIONS];
    net.option_logits_into(state, side_ref, &choices[..k], scratch, &mut logits[..k]);
    if logits[..k].iter().any(|z| !z.is_finite()) {
        // an option without a learned slot (e.g. a forced `None`): fall back to uniform
        options.iter_mut().for_each(|o| o.prior = 1.0 / kf);
        return true;
    }
    let max = logits[..k].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for z in logits[..k].iter_mut() {
        *z = (*z - max).exp();
        sum += *z;
    }
    for (o, z) in options.iter_mut().zip(&logits[..k]) {
        o.prior = (1.0 - eps) * (z / sum) + eps / kf;
    }
    true
}

/// Fills `MoveNode::prior` of a freshly populated node. Sides without a prior get -1
/// (selection then falls back to UCB1 for that side).
pub fn fill_priors(
    cfg: &PriorConfig,
    state: &State,
    s1: &mut [MoveNode],
    s2: &mut [MoveNode],
    root: bool,
    scratch: &mut PriorScratch,
) {
    for (side_ref, options, idx) in [
        (SideReference::SideOne, s1, 0usize),
        (SideReference::SideTwo, s2, 1usize),
    ] {
        if !side_prior(cfg, state, side_ref, options, root, &mut scratch.s[idx]) {
            for o in options.iter_mut() {
                o.prior = -1.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_len, features_versioned, FEATURE_VERSION_2};
    use crate::state::{PokemonIndex, PokemonMoveIndex};

    pub(crate) fn rand_weights(hidden: usize, seed: u64) -> PolicyWeights {
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let n_in = feature_len(FEATURE_VERSION_2).unwrap();
        PolicyWeights {
            w1: (0..hidden * n_in).map(|_| next() * 0.2).collect(),
            b1: (0..hidden).map(|_| next() * 0.1).collect(),
            w2: (0..N_ACTIONS * hidden).map(|_| next()).collect(),
            b2: (0..N_ACTIONS).map(|_| next()).collect(),
            p: (0..N_EMB * hidden).map(|_| next() * 0.3).collect(),
            c: (0..N_EMB).map(|_| next() * 0.1).collect(),
            a: (0..N_EMB * N_AFEAT).map(|_| next()).collect(),
            v: (0..N_EMB).map(|_| next()).collect(),
        }
    }

    fn all_choices(side: &Side) -> Vec<MoveChoice> {
        let mut v = Vec::new();
        for m in [PokemonMoveIndex::M0, PokemonMoveIndex::M1, PokemonMoveIndex::M2, PokemonMoveIndex::M3] {
            v.push(MoveChoice::Move(m));
            v.push(MoveChoice::MoveTera(m));
        }
        let a = side.active_index as usize;
        for (k, p) in [
            PokemonIndex::P0,
            PokemonIndex::P1,
            PokemonIndex::P2,
            PokemonIndex::P3,
            PokemonIndex::P4,
            PokemonIndex::P5,
        ]
        .iter()
        .enumerate()
        {
            if k != a {
                v.push(MoveChoice::Switch(*p));
            }
        }
        v
    }

    #[test]
    fn bytes_roundtrip() {
        let net = PolicyNet::new(FEATURE_VERSION_2, 16, rand_weights(16, 3)).unwrap();
        let back = PolicyNet::from_bytes(&net.to_bytes()).unwrap();
        let st = crate::features::tests::constructed();
        let ch = all_choices(&st.side_one);
        let (mut a, mut b) = (Scratch::new(), Scratch::new());
        assert_eq!(
            net.option_logits(&st, SideReference::SideOne, &ch, &mut a),
            back.option_logits(&st, SideReference::SideOne, &ch, &mut b)
        );
        assert!(PolicyNet::from_bytes(b"nope").is_err());
        let mut bad = net.to_bytes();
        bad.pop();
        assert!(PolicyNet::from_bytes(&bad).is_err());
    }

    /// The sparse fast path must agree with a dense reference for both perspectives.
    #[test]
    fn fast_path_matches_dense_reference() {
        for hidden in [16usize, 32, 64, 128] {
            let w = rand_weights(hidden, hidden as u64);
            let (w1, b1, w2, b2) = (w.w1.clone(), w.b1.clone(), w.w2.clone(), w.b2.clone());
            let (p, c, a, v) = (w.p.clone(), w.c.clone(), w.a.clone(), w.v.clone());
            let net = PolicyNet::new(FEATURE_VERSION_2, hidden, w).unwrap();
            let n_in = feature_len(FEATURE_VERSION_2).unwrap();
            let st = crate::features::tests::constructed();
            for own in [SideReference::SideOne, SideReference::SideTwo] {
                let x = features_versioned(&st, own, FEATURE_VERSION_2);
                let mut hr = b1.clone();
                for i in 0..n_in {
                    if x[i] != 0.0 {
                        for j in 0..hidden {
                            hr[j] += x[i] * w1[j * n_in + i];
                        }
                    }
                }
                hr.iter_mut().for_each(|z| *z = z.max(0.0));
                let side = if own == SideReference::SideOne { &st.side_one } else { &st.side_two };
                let ch = all_choices(side);
                let got = net.option_logits(&st, own, &ch, &mut Scratch::new());
                for (choice, g) in ch.iter().zip(got) {
                    let slot = action_slot(side, choice).unwrap();
                    let mut af = [0.0f32; N_AFEAT];
                    action_features(&st, own, choice, &mut af);
                    let mut z = b2[slot] + (0..hidden).map(|j| w2[slot * hidden + j] * hr[j]).sum::<f32>();
                    for e in 0..N_EMB {
                        let pre = c[e]
                            + (0..hidden).map(|j| p[e * hidden + j] * hr[j]).sum::<f32>()
                            + (0..N_AFEAT).map(|d| a[e * N_AFEAT + d] * af[d]).sum::<f32>();
                        z += v[e] * pre.max(0.0);
                    }
                    assert!((g - z).abs() < 2e-3, "h{hidden} {own:?} {choice:?}: {g} vs {z}");
                }
            }
        }
    }

    #[test]
    fn action_slots() {
        let mut st = State::default();
        st.side_one.active_index = PokemonIndex::P2;
        let s = &st.side_one;
        assert_eq!(action_slot(s, &MoveChoice::Move(PokemonMoveIndex::M3)), Some(3));
        assert_eq!(action_slot(s, &MoveChoice::MoveTera(PokemonMoveIndex::M1)), Some(5));
        // bench in feature-slot order: team 0, 1 -> slots 1, 2; team 3, 4, 5 -> slots 3, 4, 5
        assert_eq!(action_slot(s, &MoveChoice::Switch(PokemonIndex::P0)), Some(8));
        assert_eq!(action_slot(s, &MoveChoice::Switch(PokemonIndex::P1)), Some(9));
        assert_eq!(action_slot(s, &MoveChoice::Switch(PokemonIndex::P3)), Some(10));
        assert_eq!(action_slot(s, &MoveChoice::Switch(PokemonIndex::P5)), Some(12));
        assert_eq!(action_slot(s, &MoveChoice::Switch(PokemonIndex::P2)), None);
        assert_eq!(action_slot(s, &MoveChoice::None), None);
    }

    #[test]
    fn action_features_describe_the_move() {
        let mut st = State::default();
        {
            let a = st.side_one.get_active();
            a.types = (PokemonType::FIRE, PokemonType::TYPELESS);
            a.special_attack = 300;
            a.moves.m0 = crate::state::Move::deserialize("FLAMETHROWER;false;16");
            a.moves.m1 = crate::state::Move::deserialize("RECOVER;false;8");
        }
        {
            let d = st.side_two.get_active();
            d.types = (PokemonType::GRASS, PokemonType::TYPELESS);
            d.maxhp = 100;
            d.hp = 100;
            d.special_defense = 100;
        }
        let mut f = [0.0f32; N_AFEAT];
        action_features(&st, SideReference::SideOne, &MoveChoice::Move(PokemonMoveIndex::M0), &mut f);
        assert_eq!(f[F_MOVE], 1.0);
        assert_eq!(f[F_DAMAGING], 1.0);
        assert_eq!(f[F_DAMAGING + 2], 1.0); // special
        assert!(f[F_EFF] > 0.0, "fire vs grass is super effective");
        assert_eq!(f[F_EFF + 2], 1.0); // stab
        assert_eq!(f[F_FRAC + 1], 1.0, "a 300 spa flamethrower KOs a 100 hp 100 spd grass type");
        let mut g = [0.0f32; N_AFEAT];
        action_features(&st, SideReference::SideOne, &MoveChoice::Move(PokemonMoveIndex::M1), &mut g);
        assert_eq!(g[F_DAMAGING], 0.0);
        assert_eq!(g[F_FLAGS], 1.0, "recovery flag");
        assert_eq!(g[F_RECOVERY], 0.5);
        let mut t = [0.0f32; N_AFEAT];
        action_features(&st, SideReference::SideOne, &MoveChoice::MoveTera(PokemonMoveIndex::M0), &mut t);
        assert_eq!((t[F_MOVE], t[F_TERA]), (1.0, 1.0));
        action_features(&st, SideReference::SideOne, &MoveChoice::Switch(PokemonIndex::P1), &mut t);
        assert_eq!((t[F_SWITCH], t[F_MOVE]), (1.0, 0.0));
        assert!(t[F_SWITCH_BLOCK] > 0.0);
    }
}
