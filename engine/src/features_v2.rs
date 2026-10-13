//! FEATURE_VERSION 2: identity and move-property features, appended to the v1 layout.
//!
//! `features_v2 = features_v1 (738) ++ extras`. The extras are produced through a
//! `Sink` so the dense encoder (`features_into`) and the sparse MLP fast path
//! (`learned.rs`, which adds first-layer weight rows) share exactly one definition.
//!
//! Layout of the extras (starting at `V2_BASE`):
//!
//! * 12 Pokemon blocks (own slots 0..5, then opp slots 0..5; slot 0 = active, the
//!   rest in team order, like v1) of `PM2` features each: item one-hot, ability
//!   one-hot, move-property flags / scalars over the four moves. These depend only
//!   on item, ability, types, tera and the four move ids (not on hp, pp or stats),
//!   so the fast path caches them with the v1 slow-changing features.
//! * 2 side blocks (own, opp) of `PS2` features: screen/tailwind turns and the
//!   attacker-vs-defender block of the side's active Pokemon (speed order, best
//!   type effectiveness, rough damage fraction, KO flags, heal-vs-incoming-damage,
//!   PP of the best recovery move, exhausted moves, tera availability). These
//!   depend on the opposing active, so they are recomputed per evaluation.
//! * field: weather / terrain / trick room turns remaining.
//!
//! See `docs/features-v2.md`.

use crate::choices::{Choice, Choices, MoveCategory, MoveTarget};
use crate::engine::abilities::Abilities;
use crate::engine::damage_calc::type_effectiveness_modifier;
use crate::engine::items::Items;
use crate::engine::state::Weather;
use crate::state::PokemonSideCondition as SC;
use crate::features::{FEATURE_LEN_V1, TEAM_SLOTS};
use crate::state::{Move, Pokemon, PokemonStatus, PokemonType, Side, SideReference, State};

/// Receives the non-zero v2 features, by absolute index into the feature vector.
pub trait Sink {
    /// feature `idx` = 1
    fn one(&mut self, idx: usize);
    /// feature `idx` = `v`
    fn val(&mut self, idx: usize, v: f32);
}

/// Writes into a dense buffer (the caller zeroes the v2 region first).
pub struct DenseSink<'a> {
    pub buf: &'a mut [f32],
}
impl<'a> Sink for DenseSink<'a> {
    #[inline(always)]
    fn one(&mut self, idx: usize) {
        self.buf[idx] = 1.0;
    }
    #[inline(always)]
    fn val(&mut self, idx: usize, v: f32) {
        self.buf[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// Vocabularies
// ---------------------------------------------------------------------------

/// Named items (one-hot slot = position).
pub const ITEM_LIST: [Items; 27] = [
    Items::LEFTOVERS,
    Items::HEAVYDUTYBOOTS,
    Items::LIFEORB,
    Items::CHOICEBAND,
    Items::ASSAULTVEST,
    Items::CHOICESPECS,
    Items::EVIOLITE,
    Items::CHOICESCARF,
    Items::SITRUSBERRY,
    Items::LOADEDDICE,
    Items::WHITEHERB,
    Items::CHESTOBERRY,
    Items::ROCKYHELMET,
    Items::WEAKNESSPOLICY,
    Items::TOXICORB,
    Items::FLAMEORB,
    Items::WIDELENS,
    Items::LUMBERRY,
    Items::BOOSTERENERGY,
    Items::LIGHTCLAY,
    Items::POWERHERB,
    Items::THROATSPRAY,
    Items::SOULDEW,
    Items::FOCUSSASH,
    Items::AIRBALLOON,
    Items::LIGHTBALL,
    Items::LUSTROUSGLOBE,
];
/// After the named items: type-boosting item (plates, Silk Scarf, ...), other, none, unknown.
pub const N_ITEM: usize = ITEM_LIST.len() + 4;
const ITEM_TYPEBOOST: usize = ITEM_LIST.len();
const ITEM_OTHER: usize = ITEM_LIST.len() + 1;
const ITEM_NONE: usize = ITEM_LIST.len() + 2;
const ITEM_UNKNOWN: usize = ITEM_LIST.len() + 3;

pub fn item_slot(item: Items) -> usize {
    match item {
        Items::NONE => ITEM_NONE,
        Items::UNKNOWNITEM => ITEM_UNKNOWN,
        Items::SILKSCARF
        | Items::MAGNET
        | Items::SILVERPOWDER
        | Items::LUSTROUSORB
        | Items::FISTPLATE
        | Items::DREADPLATE
        | Items::FLAMEPLATE
        | Items::ICICLEPLATE
        | Items::EARTHPLATE
        | Items::SKYPLATE
        | Items::ZAPPLATE
        | Items::PIXIEPLATE
        | Items::IRONPLATE
        | Items::TOXICPLATE
        | Items::SPLASHPLATE
        | Items::DRACOPLATE
        | Items::SPOOKYPLATE
        | Items::MINDPLATE
        | Items::MEADOWPLATE
        | Items::INSECTPLATE
        | Items::STONEPLATE => ITEM_TYPEBOOST,
        other => ITEM_LIST
            .iter()
            .position(|i| *i == other)
            .unwrap_or(ITEM_OTHER),
    }
}

/// Named abilities (one-hot slot = position).
pub const ABILITY_LIST: [Abilities; 48] = [
    Abilities::LEVITATE,
    Abilities::INTIMIDATE,
    Abilities::PROTOSYNTHESIS,
    Abilities::QUARKDRIVE,
    Abilities::TECHNICIAN,
    Abilities::REGENERATOR,
    Abilities::PRANKSTER,
    Abilities::THICKFAT,
    Abilities::CHLOROPHYLL,
    Abilities::GUTS,
    Abilities::SAPSIPPER,
    Abilities::DEFIANT,
    Abilities::VOLTABSORB,
    Abilities::FLAMEBODY,
    Abilities::ADAPTABILITY,
    Abilities::STURDY,
    Abilities::WATERABSORB,
    Abilities::SERENEGRACE,
    Abilities::UNAWARE,
    Abilities::SHEERFORCE,
    Abilities::COMPETITIVE,
    Abilities::MOXIE,
    Abilities::SWIFTSWIM,
    Abilities::MAGICBOUNCE,
    Abilities::SPEEDBOOST,
    Abilities::CONTRARY,
    Abilities::NATURALCURE,
    Abilities::MULTISCALE,
    Abilities::DRIZZLE,
    Abilities::DROUGHT,
    Abilities::SANDSTREAM,
    Abilities::SNOWWARNING,
    Abilities::UNBURDEN,
    Abilities::MAGICGUARD,
    Abilities::HUGEPOWER,
    Abilities::PROTEAN,
    Abilities::FLASHFIRE,
    Abilities::LIGHTNINGROD,
    Abilities::STORMDRAIN,
    Abilities::MOTORDRIVE,
    Abilities::SOLIDROCK,
    Abilities::POISONHEAL,
    Abilities::MAGNETPULL,
    Abilities::TOXICCHAIN,
    Abilities::WEAKARMOR,
    Abilities::ROUGHSKIN,
    Abilities::DISGUISE,
    Abilities::SHARPNESS,
];
/// After the named abilities: other, none.
pub const N_ABILITY: usize = ABILITY_LIST.len() + 2;
const ABILITY_OTHER: usize = ABILITY_LIST.len();
const ABILITY_NONE: usize = ABILITY_LIST.len() + 1;

pub fn ability_slot(a: Abilities) -> usize {
    if a == Abilities::NONE {
        return ABILITY_NONE;
    }
    // Libero behaves like Protean, Filter like Solid Rock
    let a = match a {
        Abilities::LIBERO => Abilities::PROTEAN,
        Abilities::FILTER => Abilities::SOLIDROCK,
        a => a,
    };
    ABILITY_LIST.iter().position(|x| *x == a).unwrap_or(ABILITY_OTHER)
}

pub const MOVE_FLAG_NAMES: [&str; 30] = [
    "recovery",
    "rest",
    "wish",
    "drain",
    "setup.attack",
    "setup.defense",
    "setup.special_attack",
    "setup.special_defense",
    "setup.speed",
    "priority_attack",
    "fake_out",
    "pivot",
    "hazard_setter",
    "hazard_removal",
    "inflict.sleep",
    "inflict.paralyze",
    "inflict.burn",
    "inflict.poison",
    "phaze",
    "protect",
    "substitute",
    "screens",
    "leechseed",
    "disruption",
    "explosion",
    "recoil",
    "cure_status",
    "field_setter",
    "trick",
    "sleep_talk",
];
pub const MOVE_SCALAR_NAMES: [&str; 7] = [
    "recovery_frac",
    "setup_stages",
    "n_damaging",
    "n_status",
    "best_stab_bp",
    "best_other_bp",
    "n_attack_types",
];
const N_FLAGS: usize = MOVE_FLAG_NAMES.len();
pub const N_MOVE: usize = N_FLAGS + MOVE_SCALAR_NAMES.len();

const ITEM_OFF: usize = 0;
const ABILITY_OFF: usize = N_ITEM;
const MOVE_OFF: usize = N_ITEM + N_ABILITY;
/// Features per Pokemon block.
pub const PM2: usize = N_ITEM + N_ABILITY + N_MOVE;

pub const ACTIVE_NAMES: [&str; 13] = [
    "moves_first",
    "speed_log_ratio",
    "best_effectiveness",
    "has_super_effective",
    "best_damage_frac",
    "can_ko",
    "can_2hko",
    "ko_and_faster",
    "immune_all",
    "outheal",
    "recovery_pp",
    "exhausted_moves",
    "tera_available",
];
pub const SIDE_SCALAR_NAMES: [&str; 4] = ["reflect_turns", "light_screen_turns", "aurora_veil_turns", "tailwind_turns"];
/// Features per side block.
pub const PS2: usize = SIDE_SCALAR_NAMES.len() + ACTIVE_NAMES.len();
pub const FIELD2_NAMES: [&str; 3] = ["weather_turns", "terrain_turns", "trick_room_turns"];

pub const V2_BASE: usize = FEATURE_LEN_V1;
const SIDE2_BASE: usize = V2_BASE + 2 * TEAM_SLOTS * PM2;
const FIELD2_BASE: usize = SIDE2_BASE + 2 * PS2;
/// Total length of the v2 feature vector.
pub const FEATURE_LEN_V2: usize = FIELD2_BASE + FIELD2_NAMES.len();

#[inline(always)]
pub fn mon_base(side_no: usize, slot: usize) -> usize {
    V2_BASE + (side_no * TEAM_SLOTS + slot) * PM2
}

// ---------------------------------------------------------------------------
// Per-Pokemon static block
// ---------------------------------------------------------------------------

fn moves_of(p: &Pokemon) -> [&Move; 4] {
    [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
}

fn is_damaging(c: &Choice) -> bool {
    c.category != MoveCategory::Status && c.base_power > 0.0
}

fn has_stab(p: &Pokemon, t: PokemonType) -> bool {
    p.types.0 == t || p.types.1 == t || (p.terastallized && p.tera_type == t)
}

fn type_idx(t: PokemonType) -> Option<usize> {
    let i = t as usize;
    if i < 18 {
        Some(i)
    } else {
        None
    }
}

/// Emits the item / ability / move-property features of one Pokemon block at `base`.
/// Depends only on item, ability, types, tera type, terastallized and the four move ids.
pub fn static_block<S: Sink>(sink: &mut S, base: usize, p: &Pokemon) {
    sink.one(base + ITEM_OFF + item_slot(p.item));
    sink.one(base + ABILITY_OFF + ability_slot(p.ability));

    let mut flags = [false; N_FLAGS];
    let mut recovery_frac = 0.0f32;
    let mut setup_stages = 0i32;
    let mut n_damaging = 0u32;
    let mut n_status = 0u32;
    let mut best_stab = 0.0f32;
    let mut best_other = 0.0f32;
    let mut attack_types = 0u32; // bitmask of the types of damaging moves
    for m in moves_of(p) {
        if m.id == Choices::NONE {
            continue;
        }
        let c = &m.choice;
        let dmg = is_damaging(c);
        if dmg {
            n_damaging += 1;
            if let Some(t) = type_idx(c.move_type) {
                attack_types |= 1 << t;
            }
            if has_stab(p, c.move_type) {
                best_stab = best_stab.max(c.base_power);
            } else {
                best_other = best_other.max(c.base_power);
            }
            if c.priority > 0 && !matches!(m.id, Choices::FAKEOUT | Choices::FIRSTIMPRESSION) {
                flags[9] = true;
            }
            if c.drain.is_some() {
                flags[3] = true;
            }
        } else if c.category == MoveCategory::Status {
            n_status += 1;
        }
        if c.recoil.is_some() || c.crash.is_some() {
            flags[25] = true;
        }
        if let Some(h) = &c.heal {
            if h.target == MoveTarget::User && h.amount > 0.0 {
                flags[0] = true;
                recovery_frac = recovery_frac.max(h.amount);
            }
        }
        match m.id {
            Choices::REST => {
                flags[0] = true;
                flags[1] = true;
                recovery_frac = recovery_frac.max(1.0);
            }
            Choices::WISH => {
                flags[2] = true;
                recovery_frac = recovery_frac.max(0.5);
            }
            Choices::FAKEOUT | Choices::FIRSTIMPRESSION => flags[10] = true,
            Choices::RAPIDSPIN
            | Choices::DEFOG
            | Choices::MORTALSPIN
            | Choices::COURTCHANGE
            | Choices::TIDYUP => flags[13] = true,
            Choices::PROTECT
            | Choices::DETECT
            | Choices::SPIKYSHIELD
            | Choices::BANEFULBUNKER
            | Choices::KINGSSHIELD
            | Choices::SILKTRAP
            | Choices::BURNINGBULWARK
            | Choices::OBSTRUCT => flags[19] = true,
            Choices::SUBSTITUTE | Choices::SHEDTAIL => flags[20] = true,
            Choices::REFLECT | Choices::LIGHTSCREEN | Choices::AURORAVEIL => flags[21] = true,
            Choices::LEECHSEED => flags[22] = true,
            Choices::TAUNT | Choices::ENCORE | Choices::TORMENT | Choices::DISABLE | Choices::YAWN => {
                flags[23] = true
            }
            Choices::EXPLOSION | Choices::SELFDESTRUCT | Choices::MISTYEXPLOSION => flags[24] = true,
            Choices::HEALBELL | Choices::AROMATHERAPY => flags[26] = true,
            Choices::SUNNYDAY
            | Choices::RAINDANCE
            | Choices::SANDSTORM
            | Choices::SNOWSCAPE
            | Choices::CHILLYRECEPTION
            | Choices::TRICKROOM
            | Choices::TAILWIND => flags[27] = true,
            Choices::TRICK | Choices::SWITCHEROO => flags[28] = true,
            Choices::SLEEPTALK => flags[29] = true,
            Choices::BELLYDRUM => {
                flags[4] = true;
                setup_stages = setup_stages.max(6);
            }
            _ => {}
        }
        if c.flags.pivot {
            flags[11] = true;
        }
        if c.flags.drag {
            flags[18] = true;
        }
        if let Some(sc) = &c.side_condition {
            if sc.target == MoveTarget::Opponent
                && matches!(sc.condition, SC::Stealthrock | SC::Spikes | SC::ToxicSpikes | SC::StickyWeb)
            {
                flags[12] = true;
            }
        }
        if c.category == MoveCategory::Status {
            if let Some(st) = &c.status {
                if st.target == MoveTarget::Opponent {
                    match st.status {
                        PokemonStatus::SLEEP => flags[14] = true,
                        PokemonStatus::PARALYZE => flags[15] = true,
                        PokemonStatus::BURN => flags[16] = true,
                        PokemonStatus::POISON | PokemonStatus::TOXIC => flags[17] = true,
                        _ => {}
                    }
                }
            }
            if let Some(b) = &c.boost {
                if b.target == MoveTarget::User {
                    let bs = &b.boosts;
                    let stats = [bs.attack, bs.defense, bs.special_attack, bs.special_defense, bs.speed];
                    let mut total = 0i32;
                    for (i, s) in stats.iter().enumerate() {
                        if *s > 0 {
                            flags[4 + i] = true;
                            total += *s as i32;
                        }
                    }
                    setup_stages = setup_stages.max(total);
                }
            }
        }
    }
    for (i, f) in flags.iter().enumerate() {
        if *f {
            sink.one(base + MOVE_OFF + i);
        }
    }
    let s = base + MOVE_OFF + N_FLAGS;
    if recovery_frac > 0.0 {
        sink.val(s, recovery_frac.min(1.0));
    }
    if setup_stages > 0 {
        sink.val(s + 1, setup_stages.min(6) as f32 / 6.0);
    }
    if n_damaging > 0 {
        sink.val(s + 2, n_damaging as f32 / 4.0);
    }
    if n_status > 0 {
        sink.val(s + 3, n_status as f32 / 4.0);
    }
    if best_stab > 0.0 {
        sink.val(s + 4, (best_stab / 150.0).min(1.5));
    }
    if best_other > 0.0 {
        sink.val(s + 5, (best_other / 150.0).min(1.5));
    }
    if attack_types != 0 {
        sink.val(s + 6, attack_types.count_ones().min(4) as f32 / 4.0);
    }
}

/// Hash of everything `static_block` reads: two words, used as the fast-path cache key.
#[inline(always)]
pub fn static_key(p: &Pokemon) -> [u64; 2] {
    let k0 = (p.types.0 as u64)
        | (p.types.1 as u64) << 8
        | (p.tera_type as u64) << 16
        | (p.terastallized as u64) << 24
        | (p.item as u64) << 32
        | (p.ability as u64) << 48;
    let k1 = (p.moves.m0.id as u64)
        | (p.moves.m1.id as u64) << 16
        | (p.moves.m2.id as u64) << 32
        | (p.moves.m3.id as u64) << 48;
    [k0, k1]
}

// ---------------------------------------------------------------------------
// Active-vs-active block (recomputed per evaluation)
// ---------------------------------------------------------------------------

fn stage_mult(b: i8) -> f32 {
    if b >= 0 {
        (2 + b as i32) as f32 / 2.0
    } else {
        2.0 / (2 - b as i32) as f32
    }
}

fn effective_speed(p: &Pokemon, side: &Side) -> f32 {
    let mut s = p.speed as f32 * stage_mult(side.speed_boost);
    if p.status == PokemonStatus::PARALYZE && p.ability != Abilities::QUICKFEET {
        s *= 0.5;
    }
    if p.item == Items::CHOICESCARF {
        s *= 1.5;
    }
    if side.side_conditions.tailwind > 0 {
        s *= 2.0;
    }
    s.max(1.0)
}

/// Ability immunity to a move type (a small, common subset).
fn ability_immune(def: &Pokemon, t: PokemonType) -> bool {
    match def.ability {
        Abilities::LEVITATE => t == PokemonType::GROUND,
        Abilities::VOLTABSORB | Abilities::LIGHTNINGROD | Abilities::MOTORDRIVE => t == PokemonType::ELECTRIC,
        Abilities::WATERABSORB | Abilities::STORMDRAIN => t == PokemonType::WATER,
        Abilities::FLASHFIRE => t == PokemonType::FIRE,
        Abilities::SAPSIPPER => t == PokemonType::GRASS,
        _ => false,
    }
}

struct Assessment {
    /// max over damaging moves of the type effectiveness (0 if none)
    best_eff: f32,
    any_damaging: bool,
    /// expected damage of the best move as a fraction of the defender's max hp
    best_frac: f32,
}

/// Rough expected damage of `att`'s best move on `def` (average roll, accuracy-weighted,
/// STAB / burn / Life Orb / Choice item, no weather, screens or ability modifiers).
fn assess(att: &Pokemon, att_side: &Side, def: &Pokemon, def_side: &Side) -> Assessment {
    let mut best_eff = 0.0f32;
    let mut best_frac = 0.0f32;
    let mut any = false;
    let level = att.level as f32;
    let lvl_term = 2.0 * level / 5.0 + 2.0;
    let def_maxhp = def.maxhp.max(1) as f32;
    for m in moves_of(att) {
        if m.id == Choices::NONE {
            continue;
        }
        let c = &m.choice;
        if !is_damaging(c) {
            continue;
        }
        any = true;
        let mut eff = type_effectiveness_modifier(&c.move_type, def);
        if ability_immune(def, c.move_type) {
            eff = 0.0;
        }
        best_eff = best_eff.max(eff);
        if eff == 0.0 {
            continue;
        }
        let physical = c.category == MoveCategory::Physical;
        let (a, d) = if physical {
            (
                att.attack as f32 * stage_mult(att_side.attack_boost),
                def.defense as f32 * stage_mult(def_side.defense_boost),
            )
        } else {
            (
                att.special_attack as f32 * stage_mult(att_side.special_attack_boost),
                def.special_defense as f32 * stage_mult(def_side.special_defense_boost),
            )
        };
        let mut dmg = (lvl_term * c.base_power * a / d.max(1.0) / 50.0 + 2.0) * 0.925 * eff;
        if has_stab(att, c.move_type) {
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
        let frac = dmg * acc / def_maxhp;
        if frac > best_frac {
            best_frac = frac;
        }
    }
    Assessment { best_eff, any_damaging: any, best_frac }
}

fn best_heal(p: &Pokemon) -> (f32, f32) {
    // (largest self-heal fraction among the moves, PP fraction of that move)
    let mut best = 0.0f32;
    let mut pp = 0.0f32;
    for m in moves_of(p) {
        let amount = match (&m.choice.heal, m.id) {
            (_, Choices::REST) => 1.0,
            (Some(h), _) if h.target == MoveTarget::User => h.amount,
            _ => 0.0,
        };
        if amount > best || (amount == best && amount > 0.0 && (m.pp as f32) > pp) {
            best = amount;
            pp = (m.pp.max(0) as f32).min(16.0) / 16.0;
        }
    }
    (best, pp)
}

fn put_side_block<S: Sink>(
    sink: &mut S,
    base: usize,
    own: &Side,
    opp: &Side,
    own_to_opp: &Assessment,
    opp_to_own: &Assessment,
    trick_room: bool,
) {
    let sc = &own.side_conditions;
    for (i, (turns, scale)) in [
        (sc.reflect, 5.0f32),
        (sc.light_screen, 5.0),
        (sc.aurora_veil, 5.0),
        (sc.tailwind, 4.0),
    ]
    .iter()
    .enumerate()
    {
        if *turns > 0 {
            sink.val(base + i, (*turns as f32 / scale).min(1.0));
        }
    }
    let a = base + SIDE_SCALAR_NAMES.len();
    let me = own.get_active_immutable();
    let them = opp.get_active_immutable();
    let (s_me, s_them) = (effective_speed(me, own), effective_speed(them, opp));
    let first = if s_me == s_them {
        0.5
    } else if (s_me > s_them) != trick_room {
        1.0
    } else {
        0.0
    };
    if first > 0.0 {
        sink.val(a, first);
    }
    let lr = (s_me / s_them).log2().clamp(-2.0, 2.0) / 2.0;
    if lr != 0.0 {
        sink.val(a + 1, lr);
    }
    if own_to_opp.best_eff > 0.0 {
        sink.val(a + 2, own_to_opp.best_eff / 4.0);
    }
    if own_to_opp.best_eff >= 2.0 {
        sink.one(a + 3);
    }
    if own_to_opp.best_frac > 0.0 {
        sink.val(a + 4, own_to_opp.best_frac.min(1.5) / 1.5);
    }
    let dmg_hp = own_to_opp.best_frac * them.maxhp.max(1) as f32;
    let alive = them.hp > 0;
    let ko = alive && dmg_hp >= them.hp as f32;
    if ko {
        sink.one(a + 5);
    }
    if alive && 2.0 * dmg_hp >= them.hp as f32 {
        sink.one(a + 6);
    }
    if ko && first == 1.0 {
        sink.one(a + 7);
    }
    if own_to_opp.any_damaging && own_to_opp.best_eff == 0.0 {
        sink.one(a + 8);
    }
    let (heal, heal_pp) = best_heal(me);
    let outheal = (heal - opp_to_own.best_frac).clamp(-1.0, 1.0);
    if outheal != 0.0 {
        sink.val(a + 9, outheal);
    }
    if heal_pp > 0.0 {
        sink.val(a + 10, heal_pp);
    }
    let exhausted = moves_of(me).iter().filter(|m| m.id != Choices::NONE && m.pp <= 0).count();
    if exhausted > 0 {
        sink.val(a + 11, exhausted as f32 / 4.0);
    }
    if !own.pokemon.into_iter().any(|p| p.terastallized) {
        sink.one(a + 12);
    }
}

/// Emits the side blocks and field block (everything not cached per Pokemon).
pub fn dynamic_blocks<S: Sink>(sink: &mut S, state: &State, own_ref: SideReference) {
    let (own, opp) = match own_ref {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    let a_me = own.get_active_immutable();
    let a_opp = opp.get_active_immutable();
    let own_to_opp = assess(a_me, own, a_opp, opp);
    let opp_to_own = assess(a_opp, opp, a_me, own);
    let tr = state.trick_room.active;
    put_side_block(sink, SIDE2_BASE, own, opp, &own_to_opp, &opp_to_own, tr);
    put_side_block(sink, SIDE2_BASE + PS2, opp, own, &opp_to_own, &own_to_opp, tr);

    if state.get_weather() != Weather::NONE && state.weather.turns_remaining != 0 {
        // permanent weather (negative turns) counts as full
        let t = state.weather.turns_remaining;
        sink.val(FIELD2_BASE, if t < 0 { 1.0 } else { (t as f32 / 8.0).min(1.0) });
    }
    if state.terrain.turns_remaining > 0 {
        sink.val(FIELD2_BASE + 1, (state.terrain.turns_remaining as f32 / 5.0).min(1.0));
    }
    if state.trick_room.active && state.trick_room.turns_remaining > 0 {
        sink.val(FIELD2_BASE + 2, (state.trick_room.turns_remaining as f32 / 5.0).min(1.0));
    }
}

/// Dense v2 extras for `state` from `side`'s perspective; `out` is `FEATURE_LEN_V2` long
/// and the v1 part (`..V2_BASE`) is written by the caller.
pub fn extras_into(state: &State, side: SideReference, out: &mut [f32]) {
    out[V2_BASE..FEATURE_LEN_V2].fill(0.0);
    let (own, opp) = match side {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    let mut sink = DenseSink { buf: out };
    for (side_no, s) in [own, opp].iter().enumerate() {
        let active = s.active_index as usize;
        let mut order = [0usize; TEAM_SLOTS];
        order[0] = active;
        let mut n = 1;
        for k in 0..TEAM_SLOTS {
            if k != active {
                order[n] = k;
                n += 1;
            }
        }
        for (slot, k) in order.iter().enumerate() {
            static_block(&mut sink, mon_base(side_no, slot), &s.pokemon.pkmn[*k]);
        }
    }
    dynamic_blocks(&mut sink, state, side);
}

/// Names of the v2 extras, same order as `extras_into`.
pub fn extra_names() -> Vec<String> {
    let mut names = Vec::with_capacity(FEATURE_LEN_V2 - V2_BASE);
    for who in ["own", "opp"] {
        for slot in 0..TEAM_SLOTS {
            for i in ITEM_LIST {
                names.push(format!("{who}.p{slot}.item.{}", format!("{i:?}").to_lowercase()));
            }
            for n in ["typeboost", "other", "none", "unknown"] {
                names.push(format!("{who}.p{slot}.item.{n}"));
            }
            for a in ABILITY_LIST {
                names.push(format!("{who}.p{slot}.ability.{}", format!("{a:?}").to_lowercase()));
            }
            for n in ["other", "none"] {
                names.push(format!("{who}.p{slot}.ability.{n}"));
            }
            for n in MOVE_FLAG_NAMES {
                names.push(format!("{who}.p{slot}.mv.{n}"));
            }
            for n in MOVE_SCALAR_NAMES {
                names.push(format!("{who}.p{slot}.mv.{n}"));
            }
        }
    }
    for who in ["own", "opp"] {
        for n in SIDE_SCALAR_NAMES {
            names.push(format!("{who}.side2.{n}"));
        }
        for n in ACTIVE_NAMES {
            names.push(format!("{who}.active.{n}"));
        }
    }
    for n in FIELD2_NAMES {
        names.push(format!("field.{n}"));
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_names_versioned, features_versioned, FEATURE_LEN};
    use crate::state::PokemonIndex;

    fn idx(name: &str) -> usize {
        feature_names_versioned(2)
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no feature {name}"))
    }

    fn mv(id: &str, pp: i8) -> Move {
        Move::deserialize(&format!("{id};false;{pp}"))
    }

    fn constructed() -> State {
        let mut s = State::default();
        {
            let p = s.side_one.get_active();
            p.item = Items::LIFEORB;
            p.ability = Abilities::REGENERATOR;
            p.types = (PokemonType::FLYING, PokemonType::STEEL);
            p.moves.m0 = mv("ROOST", 8);
            p.moves.m1 = mv("SWORDSDANCE", 16);
            p.moves.m2 = mv("BRAVEBIRD", 24);
            p.moves.m3 = mv("DEFOG", 0);
            p.speed = 300;
            p.attack = 300;
        }
        {
            let p = s.side_two.get_active();
            p.item = Items::NONE;
            p.types = (PokemonType::GRASS, PokemonType::TYPELESS);
            p.moves.m0 = mv("STEALTHROCK", 32);
            p.moves.m1 = mv("UTURN", 16);
            p.moves.m2 = mv("SPORE", 8);
            p.moves.m3 = mv("PROTECT", 16);
            p.speed = 100;
            p.maxhp = 200;
            p.hp = 200;
        }
        s.side_one.side_conditions.tailwind = 2;
        s.side_one.get_active().hp = 80;
        s.side_one.get_active().maxhp = 100;
        s
    }

    #[test]
    fn length_and_names_agree() {
        assert_eq!(FEATURE_LEN_V2, 738 + 12 * PM2 + 2 * PS2 + 3);
        assert_eq!(feature_names_versioned(2).len(), FEATURE_LEN_V2);
        assert_eq!(extra_names().len(), FEATURE_LEN_V2 - V2_BASE);
        let f = features_versioned(&State::default(), SideReference::SideOne, 2);
        assert_eq!(f.len(), FEATURE_LEN_V2);
        assert!(f.iter().all(|x| x.is_finite()));
        assert!(FEATURE_LEN_V2 > 1500 && FEATURE_LEN_V2 < 3000);
    }

    #[test]
    fn v1_is_a_prefix_of_v2() {
        let s = constructed();
        for side in [SideReference::SideOne, SideReference::SideTwo] {
            let a = features_versioned(&s, side, 1);
            let b = features_versioned(&s, side, 2);
            assert_eq!(&a[..], &b[..FEATURE_LEN]);
        }
    }

    #[test]
    fn known_values() {
        let s = constructed();
        let f = features_versioned(&s, SideReference::SideOne, 2);
        assert_eq!(f[idx("own.p0.item.lifeorb")], 1.0);
        assert_eq!(f[idx("own.p0.item.leftovers")], 0.0);
        assert_eq!(f[idx("opp.p0.item.none")], 1.0);
        assert_eq!(f[idx("own.p0.ability.regenerator")], 1.0);
        assert_eq!(f[idx("own.p0.mv.recovery")], 1.0);
        assert_eq!(f[idx("own.p0.mv.recovery_frac")], 0.5);
        assert_eq!(f[idx("own.p0.mv.setup.attack")], 1.0);
        assert_eq!(f[idx("own.p0.mv.setup.speed")], 0.0);
        assert_eq!(f[idx("own.p0.mv.hazard_removal")], 1.0);
        assert_eq!(f[idx("own.p0.mv.hazard_setter")], 0.0);
        assert_eq!(f[idx("own.p0.mv.n_damaging")], 0.25);
        assert_eq!(f[idx("own.p0.mv.n_status")], 0.75);
        assert_eq!(f[idx("opp.p0.mv.hazard_setter")], 1.0);
        assert_eq!(f[idx("opp.p0.mv.pivot")], 1.0);
        assert_eq!(f[idx("opp.p0.mv.inflict.sleep")], 1.0);
        assert_eq!(f[idx("opp.p0.mv.protect")], 1.0);
        assert_eq!(f[idx("own.side2.tailwind_turns")], 0.5);
        // Bravebird (flying) vs grass: super effective; flying/steel out-speeds nothing but is faster
        assert_eq!(f[idx("own.active.has_super_effective")], 1.0);
        assert_eq!(f[idx("own.active.best_effectiveness")], 0.5);
        assert_eq!(f[idx("own.active.moves_first")], 1.0);
        assert_eq!(f[idx("opp.active.moves_first")], 0.0);
        // Defog's pp is 0 -> one exhausted move; Roost pp 8 -> 0.5
        assert_eq!(f[idx("own.active.exhausted_moves")], 0.25);
        assert_eq!(f[idx("own.active.recovery_pp")], 0.5);
        assert_eq!(f[idx("own.active.tera_available")], 1.0);
    }

    #[test]
    fn side_two_perspective_swaps_blocks() {
        let s = constructed();
        let a = features_versioned(&s, SideReference::SideOne, 2);
        let b = features_versioned(&s, SideReference::SideTwo, 2);
        let n = 6 * PM2;
        assert_eq!(&a[V2_BASE..V2_BASE + n], &b[V2_BASE + n..V2_BASE + 2 * n]);
        assert_eq!(&a[V2_BASE + n..V2_BASE + 2 * n], &b[V2_BASE..V2_BASE + n]);
        let sb = V2_BASE + 2 * n;
        assert_eq!(&a[sb..sb + PS2], &b[sb + PS2..sb + 2 * PS2]);
        assert_eq!(&a[sb + PS2..sb + 2 * PS2], &b[sb..sb + PS2]);
    }

    #[test]
    fn bench_slots_follow_the_active() {
        let mut s = constructed();
        s.side_one.active_index = PokemonIndex::P3;
        s.side_one.pokemon.pkmn[3].item = Items::CHOICESCARF;
        let f = features_versioned(&s, SideReference::SideOne, 2);
        assert_eq!(f[idx("own.p0.item.choicescarf")], 1.0);
        // the previously active Pokemon (team index 0) is now bench slot 1
        assert_eq!(f[idx("own.p1.item.lifeorb")], 1.0);
    }
}
