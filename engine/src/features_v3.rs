//! FEATURE_VERSION 3: v1 features plus pairwise matchup features.
//!
//! `features_v3 = features_v1 (738) ++ matchup extras (V3_EXTRA)`. The extras are emitted
//! through the same `Sink` as v2, so the dense encoder and the sparse MLP fast path share
//! one definition. For every pair (own alive Pokemon `i`, opposing alive Pokemon `j`),
//! with slots ordered like v1 (slot 0 = active, then team order), five 6x6 blocks:
//!
//! * `speed`: +1 if `i` moves before `j` (effective speed incl. the active's boosts,
//!   paralysis, Choice Scarf, Tailwind, weather-speed abilities, Trick Room), -1 if after,
//!   0 on a tie (not emitted),
//! * `dmg_own`: best damage `i` does to `j` with its four moves (max roll) as a fraction of
//!   `j`'s CURRENT hp, capped at 1; `dmg_opp`: the same for `j` to `i`,
//! * `prio_ko_own` / `prio_ko_opp`: 1 if `i` (resp. `j`) has a damaging priority move that
//!   KOs the other (max roll vs current hp).
//!
//! Then 8 aggregates, for the two directions (own attackers vs their team, their attackers
//! vs ours). With `can_rk(a, b) = KO(a, b) && (a moves first || a has a priority KO)`:
//! `sweepers` = attackers with `can_rk` against EVERY alive opposing Pokemon; `answers_min`
//! / `answers_mean` = min / mean over the opposing alive Pokemon `b` of the number of
//! attackers with `can_rk(a, b)`; `unanswered` = opposing Pokemon `b` with no such attacker.
//! All counts are divided by 6.
//!
//! The damage is the cheap approximation of `features_v2::assess` made per move and per pair
//! (level formula x base power x stat ratio x STAB x type effectiveness x burn / item /
//! weather; defender ability immunities; Assault Vest), not `calculate_damage`; screens,
//! most ability modifiers, multi-hit moves and variable-power moves are not modelled.
//! Boosts are only applied to the active Pokemon (the engine keeps them per side).

use crate::choices::{Choices, MoveCategory};
use crate::engine::abilities::Abilities;
use crate::engine::damage_calc::type_effectiveness_modifier;
use crate::engine::items::Items;
use crate::engine::state::{Terrain, Weather};
use crate::features::{FEATURE_LEN_V1, TEAM_SLOTS};
use crate::features_v2::{
    ability_immune, effective_speed_with, has_stab, is_damaging, moves_of, stage_mult, DenseSink, Sink,
};
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, SideReference, State};

pub const V3_BASE: usize = FEATURE_LEN_V1;
pub const N_PAIR: usize = TEAM_SLOTS * TEAM_SLOTS;
pub const PAIR_CHANNELS: [&str; 5] = ["speed", "dmg_own", "dmg_opp", "prio_ko_own", "prio_ko_opp"];
pub const AGG_NAMES: [&str; 4] = ["sweepers", "answers_min", "answers_mean", "unanswered"];
const AGG_BASE: usize = V3_BASE + PAIR_CHANNELS.len() * N_PAIR;
pub const V3_EXTRA: usize = PAIR_CHANNELS.len() * N_PAIR + 2 * AGG_NAMES.len();
/// Total length of the v3 feature vector.
pub const FEATURE_LEN_V3: usize = V3_BASE + V3_EXTRA;

/// One attacking move, with everything that does not depend on the defender folded in.
#[derive(Clone, Copy)]
struct Atk {
    /// level term x base power x attacker stat x modifiers / 50 (the `+ 2` is added later)
    num: f32,
    physical: bool,
    mtype: PokemonType,
    prio: bool,
}

#[derive(Clone, Copy)]
struct Mon<'a> {
    p: &'a Pokemon,
    speed: f32,
    /// 1 / (physical, special) defense with the active's boosts and Assault Vest
    inv_def: f32,
    inv_spd: f32,
    inv_hp: f32,
    atk: [Atk; 4],
    n_atk: usize,
}

fn weather_speed_mult(p: &Pokemon, w: Weather, t: Terrain) -> f32 {
    let hit = match p.ability {
        Abilities::SWIFTSWIM => matches!(w, Weather::RAIN | Weather::HEAVYRAIN),
        Abilities::CHLOROPHYLL => matches!(w, Weather::SUN | Weather::HARSHSUN),
        Abilities::SANDRUSH => w == Weather::SAND,
        Abilities::SLUSHRUSH => matches!(w, Weather::HAIL | Weather::SNOW),
        Abilities::SURGESURFER => t == Terrain::ELECTRICTERRAIN,
        _ => false,
    };
    if hit {
        2.0
    } else {
        1.0
    }
}

fn collect<'a>(side: &'a Side, w: Weather, t: Terrain, out: &mut [Option<Mon<'a>>; TEAM_SLOTS]) {
    let active = side.active_index as usize;
    let mut slot = 0;
    for k in std::iter::once(active).chain((0..TEAM_SLOTS).filter(|k| *k != active)) {
        let p = &side.pokemon.pkmn[k];
        if p.hp > 0 {
            let is_active = k == active;
            let boost = |b: i8| if is_active { stage_mult(b) } else { 1.0 };
            let mut speed = effective_speed_with(p, if is_active { side.speed_boost } else { 0 }, side);
            speed = (speed * weather_speed_mult(p, w, t)).max(1.0);
            let lvl_term = 2.0 * p.level as f32 / 5.0 + 2.0;
            let mut atk = [Atk { num: 0.0, physical: true, mtype: PokemonType::NORMAL, prio: false }; 4];
            let mut n = 0;
            for m in moves_of(p) {
                if m.id == Choices::NONE || !is_damaging(&m.choice) {
                    continue;
                }
                let c = &m.choice;
                let physical = c.category == MoveCategory::Physical;
                let stat = if physical {
                    p.attack as f32 * boost(side.attack_boost)
                } else {
                    p.special_attack as f32 * boost(side.special_attack_boost)
                };
                let mut mult = 1.0f32;
                if has_stab(p, c.move_type) {
                    mult *= if p.ability == Abilities::ADAPTABILITY { 2.0 } else { 1.5 };
                }
                if physical && p.status == PokemonStatus::BURN && p.ability != Abilities::GUTS {
                    mult *= 0.5;
                }
                match p.item {
                    Items::LIFEORB => mult *= 1.3,
                    Items::CHOICEBAND if physical => mult *= 1.5,
                    Items::CHOICESPECS if !physical => mult *= 1.5,
                    _ => {}
                }
                match (w, c.move_type) {
                    (Weather::SUN | Weather::HARSHSUN, PokemonType::FIRE)
                    | (Weather::RAIN | Weather::HEAVYRAIN, PokemonType::WATER) => mult *= 1.5,
                    (Weather::SUN | Weather::HARSHSUN, PokemonType::WATER)
                    | (Weather::RAIN | Weather::HEAVYRAIN, PokemonType::FIRE) => mult *= 0.5,
                    _ => {}
                }
                atk[n] = Atk {
                    num: lvl_term * c.base_power * stat * mult / 50.0,
                    physical,
                    mtype: c.move_type,
                    prio: c.priority > 0,
                };
                n += 1;
            }
            let (mut def, mut spd) = (p.defense as f32, p.special_defense as f32);
            def *= boost(side.defense_boost);
            spd *= boost(side.special_defense_boost);
            if p.item == Items::ASSAULTVEST {
                spd *= 1.5;
            }
            out[slot] = Some(Mon {
                p,
                speed,
                inv_def: 1.0 / def.max(1.0),
                inv_spd: 1.0 / spd.max(1.0),
                inv_hp: 1.0 / p.hp.max(1) as f32,
                atk,
                n_atk: n,
            });
        }
        slot += 1;
    }
}

/// (best, best priority) damage of `a` on `d` (max roll) as a fraction of `d`'s current hp,
/// both capped at 1.
#[inline]
fn damage(a: &Mon, d: &Mon) -> (f32, f32) {
    let dp = d.p;
    let (mut best, mut prio) = (0.0f32, 0.0f32);
    for m in &a.atk[..a.n_atk] {
        let eff = type_effectiveness_modifier(&m.mtype, dp);
        if eff == 0.0 || ability_immune(dp, m.mtype) {
            continue;
        }
        let inv = if m.physical { d.inv_def } else { d.inv_spd };
        let f = ((m.num * inv + 2.0) * eff * d.inv_hp).min(1.0);
        if f > best {
            best = f;
        }
        if m.prio && f > prio {
            prio = f;
        }
    }
    (best, prio)
}

/// Emits the matchup features (everything is recomputed per evaluation).
pub fn matchup_blocks<S: Sink>(sink: &mut S, state: &State, own_ref: SideReference) {
    let (own, opp) = match own_ref {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    let (w, t) = (state.get_weather(), state.get_terrain());
    let mut a: [Option<Mon>; TEAM_SLOTS] = [None; TEAM_SLOTS];
    let mut b: [Option<Mon>; TEAM_SLOTS] = [None; TEAM_SLOTS];
    collect(own, w, t, &mut a);
    collect(opp, w, t, &mut b);
    let tr = state.trick_room.active;

    // rk[dir][attacker][defender]: dir 0 = own attacks, dir 1 = opp attacks
    let mut rk = [[[false; TEAM_SLOTS]; TEAM_SLOTS]; 2];
    for i in 0..TEAM_SLOTS {
        let Some(ma) = &a[i] else { continue };
        for j in 0..TEAM_SLOTS {
            let Some(mb) = &b[j] else { continue };
            let pair = i * TEAM_SLOTS + j;
            let (d_ab, p_ab) = damage(ma, mb);
            let (d_ba, p_ba) = damage(mb, ma);
            let first = if ma.speed == mb.speed {
                0.0
            } else if (ma.speed > mb.speed) != tr {
                1.0
            } else {
                -1.0
            };
            if first != 0.0 {
                sink.val(V3_BASE + pair, first);
            }
            if d_ab > 0.0 {
                sink.val(V3_BASE + N_PAIR + pair, d_ab);
            }
            if d_ba > 0.0 {
                sink.val(V3_BASE + 2 * N_PAIR + pair, d_ba);
            }
            let pko_ab = p_ab >= 1.0;
            let pko_ba = p_ba >= 1.0;
            if pko_ab {
                sink.one(V3_BASE + 3 * N_PAIR + pair);
            }
            if pko_ba {
                sink.one(V3_BASE + 4 * N_PAIR + pair);
            }
            rk[0][i][j] = d_ab >= 1.0 && (first > 0.0 || pko_ab);
            rk[1][j][i] = d_ba >= 1.0 && (first < 0.0 || pko_ba);
        }
    }
    let alive_a: [bool; TEAM_SLOTS] = std::array::from_fn(|k| a[k].is_some());
    let alive_b: [bool; TEAM_SLOTS] = std::array::from_fn(|k| b[k].is_some());
    for dir in 0..2 {
        let (att, def) = if dir == 0 { (&alive_a, &alive_b) } else { (&alive_b, &alive_a) };
        let n_def = def.iter().filter(|x| **x).count();
        if n_def == 0 || !att.iter().any(|x| *x) {
            continue;
        }
        let m = &rk[dir];
        let sweepers = (0..TEAM_SLOTS)
            .filter(|&x| att[x] && (0..TEAM_SLOTS).all(|y| !def[y] || m[x][y]))
            .count();
        let (mut min, mut sum, mut zero) = (TEAM_SLOTS, 0usize, 0usize);
        for y in (0..TEAM_SLOTS).filter(|&y| def[y]) {
            let n = (0..TEAM_SLOTS).filter(|&x| att[x] && m[x][y]).count();
            min = min.min(n);
            sum += n;
            zero += (n == 0) as usize;
        }
        let base = AGG_BASE + dir * AGG_NAMES.len();
        let s = 1.0 / TEAM_SLOTS as f32;
        if sweepers > 0 {
            sink.val(base, sweepers as f32 * s);
        }
        if min > 0 {
            sink.val(base + 1, min as f32 * s);
        }
        if sum > 0 {
            sink.val(base + 2, sum as f32 / n_def as f32 * s);
        }
        if zero > 0 {
            sink.val(base + 3, zero as f32 * s);
        }
    }
}

/// Dense v3 extras for `state` from `side`'s perspective; `out` is `FEATURE_LEN_V3` long and
/// the v1 part (`..V3_BASE`) is written by the caller.
pub fn extras_into(state: &State, side: SideReference, out: &mut [f32]) {
    out[V3_BASE..FEATURE_LEN_V3].fill(0.0);
    matchup_blocks(&mut DenseSink { buf: out }, state, side);
}

/// Names of the v3 extras, same order as `extras_into`.
pub fn extra_names() -> Vec<String> {
    let mut names = Vec::with_capacity(V3_EXTRA);
    for ch in PAIR_CHANNELS {
        for i in 0..TEAM_SLOTS {
            for j in 0..TEAM_SLOTS {
                names.push(format!("match.{ch}.own{i}.opp{j}"));
            }
        }
    }
    for dir in ["own", "opp"] {
        for n in AGG_NAMES {
            names.push(format!("match.agg.{dir}.{n}"));
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_names_versioned, features_versioned, FEATURE_LEN};
    use crate::state::Move;

    fn idx(name: &str) -> usize {
        feature_names_versioned(3)
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no feature {name}"))
    }

    fn mv(id: &str) -> Move {
        Move::deserialize(&format!("{id};false;16"))
    }

    /// Two actives only (every other Pokemon fainted): ours fast and strong with a priority
    /// move, theirs slow and frail.
    fn duel() -> State {
        let mut s = State::default();
        for side in [&mut s.side_one, &mut s.side_two] {
            for p in side.pokemon.pkmn.iter_mut() {
                p.hp = 0;
            }
        }
        {
            let p = s.side_one.get_active();
            p.hp = 300;
            p.maxhp = 300;
            p.types = (PokemonType::WATER, PokemonType::TYPELESS);
            p.moves.m0 = mv("SURF");
            p.moves.m1 = mv("AQUAJET");
            p.moves.m2 = mv("PROTECT");
            p.moves.m3 = mv("SPLASH");
            p.special_attack = 400;
            p.attack = 400;
            p.speed = 300;
        }
        {
            let p = s.side_two.get_active();
            p.hp = 40;
            p.maxhp = 300;
            p.types = (PokemonType::FIRE, PokemonType::TYPELESS);
            p.moves.m0 = mv("FLAMETHROWER");
            p.moves.m1 = mv("SPLASH");
            p.moves.m2 = mv("SPLASH");
            p.moves.m3 = mv("SPLASH");
            p.special_attack = 100;
            p.special_defense = 100;
            p.defense = 100;
            p.speed = 100;
        }
        s
    }

    #[test]
    fn length_and_names_agree() {
        assert_eq!(FEATURE_LEN_V3, 738 + 5 * 36 + 8);
        assert_eq!(feature_names_versioned(3).len(), FEATURE_LEN_V3);
        assert_eq!(extra_names().len(), V3_EXTRA);
        let f = features_versioned(&State::default(), SideReference::SideOne, 3);
        assert_eq!(f.len(), FEATURE_LEN_V3);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn v1_is_a_prefix_of_v3() {
        let s = duel();
        for side in [SideReference::SideOne, SideReference::SideTwo] {
            let a = features_versioned(&s, side, 1);
            let b = features_versioned(&s, side, 3);
            assert_eq!(&a[..], &b[..FEATURE_LEN]);
        }
    }

    #[test]
    fn known_values() {
        let s = duel();
        let f = features_versioned(&s, SideReference::SideOne, 3);
        assert_eq!(f[idx("match.speed.own0.opp0")], 1.0);
        assert_eq!(f[idx("match.dmg_own.own0.opp0")], 1.0); // Surf OHKOs the 40 hp fire type
        assert!(f[idx("match.dmg_opp.own0.opp0")] > 0.0 && f[idx("match.dmg_opp.own0.opp0")] < 0.3);
        assert_eq!(f[idx("match.prio_ko_own.own0.opp0")], 1.0); // Aqua Jet (super effective) KOs
        assert_eq!(f[idx("match.prio_ko_opp.own0.opp0")], 0.0);
        assert_eq!(f[idx("match.speed.own1.opp0")], 0.0); // fainted slots stay empty
        assert_eq!(f[idx("match.agg.own.sweepers")], 1.0 / 6.0);
        assert_eq!(f[idx("match.agg.own.answers_min")], 1.0 / 6.0);
        assert_eq!(f[idx("match.agg.own.unanswered")], 0.0);
        assert_eq!(f[idx("match.agg.opp.sweepers")], 0.0);
        assert_eq!(f[idx("match.agg.opp.unanswered")], 1.0 / 6.0);
    }

    #[test]
    fn speed_modifiers_and_trick_room() {
        let mut s = duel();
        s.side_two.get_active().item = Items::CHOICESCARF;
        s.side_two.get_active().speed = 250; // 375 with the scarf > 300
        let i = idx("match.speed.own0.opp0");
        assert_eq!(features_versioned(&s, SideReference::SideOne, 3)[i], -1.0);
        s.side_two.get_active().status = PokemonStatus::PARALYZE; // halved to 187.5
        assert_eq!(features_versioned(&s, SideReference::SideOne, 3)[i], 1.0);
        s.side_two.get_active().speed = 400; // 400 * 0.5 * 1.5 = 300 vs 300: tie
        assert_eq!(features_versioned(&s, SideReference::SideOne, 3)[i], 0.0);
        s.side_two.get_active().speed = 100;
        s.side_two.get_active().status = PokemonStatus::NONE;
        s.side_two.get_active().item = Items::NONE;
        s.trick_room.active = true;
        assert_eq!(features_versioned(&s, SideReference::SideOne, 3)[i], -1.0);
        s.side_one.speed_boost = -6; // boosts only affect the active: we are now slowest, so first
        assert_eq!(features_versioned(&s, SideReference::SideOne, 3)[i], 1.0);
    }

    #[test]
    fn side_two_perspective_mirrors() {
        let s = duel();
        let a = features_versioned(&s, SideReference::SideOne, 3);
        let b = features_versioned(&s, SideReference::SideTwo, 3);
        // side two's view: its own0 is their active; speed and the damage channels swap
        assert_eq!(b[idx("match.speed.own0.opp0")], -a[idx("match.speed.own0.opp0")]);
        assert_eq!(b[idx("match.dmg_own.own0.opp0")], a[idx("match.dmg_opp.own0.opp0")]);
        assert_eq!(b[idx("match.dmg_opp.own0.opp0")], a[idx("match.dmg_own.own0.opp0")]);
        assert_eq!(b[idx("match.agg.opp.sweepers")], a[idx("match.agg.own.sweepers")]);
    }
}
