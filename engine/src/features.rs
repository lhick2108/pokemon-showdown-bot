//! Fixed-length feature encoding of a `State` for learned value functions.
//!
//! Training (Python) and inference (the MCTS leaf evaluator) MUST both use
//! this function. The layout is documented in `FEATURES.md`; any change to the
//! layout or to the meaning of a value requires bumping `FEATURE_VERSION`.
//!
//! Only the "genx" engine (gens 4-9) is supported.

use crate::engine::items::Items;
use crate::engine::state::{PokemonVolatileStatus as V, Terrain, Weather};
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, SideReference, State};

pub const FEATURE_VERSION: u32 = 1;

/// Pokemon slots encoded per side (slot 0 is the active Pokemon).
pub const TEAM_SLOTS: usize = 6;

/// Scale applied to raw stats (maxhp, attack, ...).
pub const STAT_SCALE: f32 = 500.0;

/// Names of the per-Pokemon features, in order.
pub const POKEMON_NAMES: [&str; 53] = [
    "hp",
    "alive",
    "status.burn",
    "status.sleep",
    "status.freeze",
    "status.paralyze",
    "status.poison",
    "status.toxic",
    "type.normal",
    "type.fire",
    "type.water",
    "type.electric",
    "type.grass",
    "type.ice",
    "type.fighting",
    "type.poison",
    "type.ground",
    "type.flying",
    "type.psychic",
    "type.bug",
    "type.rock",
    "type.ghost",
    "type.dragon",
    "type.dark",
    "type.steel",
    "type.fairy",
    "tera.normal",
    "tera.fire",
    "tera.water",
    "tera.electric",
    "tera.grass",
    "tera.ice",
    "tera.fighting",
    "tera.poison",
    "tera.ground",
    "tera.flying",
    "tera.psychic",
    "tera.bug",
    "tera.rock",
    "tera.ghost",
    "tera.dragon",
    "tera.dark",
    "tera.steel",
    "tera.fairy",
    "tera.stellar",
    "terastallized",
    "has_item",
    "maxhp",
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
];
pub const PER_POKEMON: usize = POKEMON_NAMES.len();

pub const BOOST_NAMES: [&str; 7] = [
    "attack",
    "defense",
    "special_attack",
    "special_defense",
    "speed",
    "accuracy",
    "evasion",
];

/// Volatile statuses of the side's active Pokemon, encoded as 0/1 flags.
/// `perish` is set if any of PERISH4..PERISH1 is set.
pub const VOLATILE_NAMES: [&str; 20] = [
    "leechseed",
    "substitute",
    "confusion",
    "taunt",
    "encore",
    "yawn",
    "torment",
    "disable",
    "aquaring",
    "ingrain",
    "curse",
    "nightmare",
    "saltcure",
    "syrupbomb",
    "healblock",
    "partiallytrapped",
    "perish",
    "focusenergy",
    "mustrecharge",
    "lockedmove",
];

pub const SIDE_CONDITION_NAMES: [&str; 14] = [
    "stealth_rock",
    "spikes",
    "toxic_spikes",
    "sticky_web",
    "reflect",
    "light_screen",
    "aurora_veil",
    "safeguard",
    "tailwind",
    "mist",
    "lucky_chant",
    "healing_wish",
    "lunar_dance",
    "protect",
];

pub const MISC_NAMES: [&str; 4] = [
    "wish_pending",
    "future_sight_pending",
    "force_switch",
    "alive_fraction",
];

pub const WEATHER_NAMES: [&str; 7] = [
    "sun", "rain", "sand", "hail", "snow", "harshsun", "heavyrain",
];
pub const TERRAIN_NAMES: [&str; 4] = ["electric", "psychic", "misty", "grassy"];

/// Side extras: boosts, volatiles, side conditions, misc.
pub const PER_SIDE_EXTRAS: usize =
    BOOST_NAMES.len() + VOLATILE_NAMES.len() + SIDE_CONDITION_NAMES.len() + MISC_NAMES.len();
pub const PER_SIDE: usize = TEAM_SLOTS * PER_POKEMON + PER_SIDE_EXTRAS;
/// Field block: weather one-hot (7), terrain one-hot (4), trick room (1).
pub const FIELD_LEN: usize = WEATHER_NAMES.len() + TERRAIN_NAMES.len() + 1;
/// Total length of the feature vector.
pub const FEATURE_LEN: usize = 2 * PER_SIDE + FIELD_LEN;

struct Out<'a> {
    buf: &'a mut [f32],
    i: usize,
}
impl<'a> Out<'a> {
    #[inline(always)]
    fn put(&mut self, v: f32) {
        self.buf[self.i] = v;
        self.i += 1;
    }
    #[inline(always)]
    fn flag(&mut self, b: bool) {
        self.put(if b { 1.0 } else { 0.0 });
    }
}

/// Index 0..18 for the 18 real types, None for STELLAR / TYPELESS.
fn type_index(t: PokemonType) -> Option<usize> {
    let i = t as usize;
    if i < 18 {
        Some(i)
    } else {
        None
    }
}

fn put_pokemon(o: &mut Out, p: &Pokemon) {
    let maxhp = p.maxhp.max(1) as f32;
    o.put((p.hp.max(0) as f32 / maxhp).min(1.0));
    o.flag(p.hp > 0);
    for s in [
        PokemonStatus::BURN,
        PokemonStatus::SLEEP,
        PokemonStatus::FREEZE,
        PokemonStatus::PARALYZE,
        PokemonStatus::POISON,
        PokemonStatus::TOXIC,
    ] {
        o.flag(p.status == s);
    }
    let t0 = type_index(p.types.0);
    let t1 = type_index(p.types.1);
    for t in 0..18 {
        o.flag(t0 == Some(t) || t1 == Some(t));
    }
    let tera = type_index(p.tera_type);
    for t in 0..18 {
        o.flag(tera == Some(t));
    }
    o.flag(p.tera_type == PokemonType::STELLAR);
    o.flag(p.terastallized);
    o.flag(p.item != Items::NONE);
    o.put(p.maxhp as f32 / STAT_SCALE);
    o.put(p.attack as f32 / STAT_SCALE);
    o.put(p.defense as f32 / STAT_SCALE);
    o.put(p.special_attack as f32 / STAT_SCALE);
    o.put(p.special_defense as f32 / STAT_SCALE);
    o.put(p.speed as f32 / STAT_SCALE);
}

fn put_side(o: &mut Out, side: &Side) {
    // slot 0 = active Pokemon, then the rest in team order
    let active = side.active_index as usize;
    let mut order = [0usize; TEAM_SLOTS];
    order[0] = active;
    let mut n = 1;
    for k in 0..TEAM_SLOTS {
        if k != active {
            order[n] = k;
            n += 1;
        }
    }
    for k in order {
        put_pokemon(o, &side.pokemon.pkmn[k]);
    }

    for b in [
        side.attack_boost,
        side.defense_boost,
        side.special_attack_boost,
        side.special_defense_boost,
        side.speed_boost,
        side.accuracy_boost,
        side.evasion_boost,
    ] {
        o.put(b as f32 / 6.0);
    }

    let vs = &side.volatile_statuses;
    for v in [
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
    ] {
        o.flag(vs.contains(&v));
    }
    o.flag(
        vs.contains(&V::PERISH4)
            || vs.contains(&V::PERISH3)
            || vs.contains(&V::PERISH2)
            || vs.contains(&V::PERISH1),
    );
    for v in [V::FOCUSENERGY, V::MUSTRECHARGE, V::LOCKEDMOVE] {
        o.flag(vs.contains(&v));
    }

    let sc = &side.side_conditions;
    o.flag(sc.stealth_rock > 0);
    o.put(sc.spikes as f32 / 3.0);
    o.put(sc.toxic_spikes as f32 / 2.0);
    o.flag(sc.sticky_web > 0);
    o.flag(sc.reflect > 0);
    o.flag(sc.light_screen > 0);
    o.flag(sc.aurora_veil > 0);
    o.flag(sc.safeguard > 0);
    o.flag(sc.tailwind > 0);
    o.flag(sc.mist > 0);
    o.flag(sc.lucky_chant > 0);
    o.flag(sc.healing_wish > 0);
    o.flag(sc.lunar_dance > 0);
    o.flag(sc.protect > 0);

    o.flag(side.wish.0 > 0);
    o.flag(side.future_sight.0 > 0);
    o.flag(side.force_switch);
    o.put(side.visible_alive_pkmn() as f32 / TEAM_SLOTS as f32);
}

/// Writes the feature vector of `state` from `side`'s perspective into the
/// first `FEATURE_LEN` entries of `out` (every one of them is written).
/// Panics if `out` is shorter than `FEATURE_LEN`.
pub fn features_into(state: &State, side: SideReference, out: &mut [f32]) {
    let (own, opp) = match side {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    let mut o = Out {
        buf: &mut out[..FEATURE_LEN],
        i: 0,
    };
    put_side(&mut o, own);
    put_side(&mut o, opp);

    let weather = state.get_weather();
    for w in [
        Weather::SUN,
        Weather::RAIN,
        Weather::SAND,
        Weather::HAIL,
        Weather::SNOW,
        Weather::HARSHSUN,
        Weather::HEAVYRAIN,
    ] {
        o.flag(weather == w);
    }
    let terrain = state.get_terrain();
    for t in [
        Terrain::ELECTRICTERRAIN,
        Terrain::PSYCHICTERRAIN,
        Terrain::MISTYTERRAIN,
        Terrain::GRASSYTERRAIN,
    ] {
        o.flag(terrain == t);
    }
    o.flag(state.trick_room.active);
    debug_assert_eq!(o.i, FEATURE_LEN);
}

/// The feature vector of `state` from `side`'s perspective.
pub fn features(state: &State, side: SideReference) -> Vec<f32> {
    let mut v = vec![0.0; FEATURE_LEN];
    features_into(state, side, &mut v);
    v
}

/// Human-readable name of each feature index, e.g. `own.p0.hp`,
/// `opp.boost.attack`, `field.weather.sun`. Same order as `features`.
pub fn feature_names() -> Vec<String> {
    let mut names = Vec::with_capacity(FEATURE_LEN);
    for who in ["own", "opp"] {
        for slot in 0..TEAM_SLOTS {
            for n in POKEMON_NAMES {
                names.push(format!("{who}.p{slot}.{n}"));
            }
        }
        for n in BOOST_NAMES {
            names.push(format!("{who}.boost.{n}"));
        }
        for n in VOLATILE_NAMES {
            names.push(format!("{who}.vol.{n}"));
        }
        for n in SIDE_CONDITION_NAMES {
            names.push(format!("{who}.sc.{n}"));
        }
        for n in MISC_NAMES {
            names.push(format!("{who}.misc.{n}"));
        }
    }
    for n in WEATHER_NAMES {
        names.push(format!("field.weather.{n}"));
    }
    for n in TERRAIN_NAMES {
        names.push(format!("field.terrain.{n}"));
    }
    names.push("field.trick_room".to_string());
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::PokemonIndex;

    pub fn idx(name: &str) -> usize {
        feature_names()
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no feature {name}"))
    }

    pub fn constructed() -> State {
        let mut s = State::default();
        // side one: active is team index 2 (slot 0), half hp, burned, fire/flying, tera water
        s.side_one.active_index = PokemonIndex::P2;
        {
            let p = s.side_one.get_active();
            p.hp = 50;
            p.maxhp = 100;
            p.status = PokemonStatus::BURN;
            p.types = (PokemonType::FIRE, PokemonType::FLYING);
            p.tera_type = PokemonType::WATER;
            p.item = Items::LEFTOVERS;
            p.attack = 250;
        }
        s.side_one.pokemon.pkmn[0].hp = 0; // fainted
        s.side_one.attack_boost = 3;
        s.side_one.speed_boost = -6;
        s.side_one.side_conditions.stealth_rock = 1;
        s.side_one.side_conditions.spikes = 2;
        s.side_one.volatile_statuses.insert(V::SUBSTITUTE);
        s.side_one.volatile_statuses.insert(V::PERISH2);
        // side two: terastallized active
        s.side_two.get_active().terastallized = true;
        s.side_two.side_conditions.reflect = 4;
        s.weather.weather_type = Weather::RAIN;
        s.weather.turns_remaining = 3;
        s.trick_room.active = true;
        s
    }

    #[test]
    fn length_matches_documented_constants() {
        assert_eq!(PER_POKEMON, 53);
        assert_eq!(PER_SIDE_EXTRAS, 45);
        assert_eq!(PER_SIDE, 6 * 53 + 45);
        assert_eq!(PER_SIDE, 363);
        assert_eq!(FEATURE_LEN, 738);
        assert_eq!(feature_names().len(), FEATURE_LEN);
        let f = features(&State::default(), SideReference::SideOne);
        assert_eq!(f.len(), FEATURE_LEN);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn known_values_on_constructed_state() {
        let s = constructed();
        let f = features(&s, SideReference::SideOne);
        assert_eq!(f[idx("own.p0.hp")], 0.5);
        assert_eq!(f[idx("own.p0.alive")], 1.0);
        assert_eq!(f[idx("own.p0.status.burn")], 1.0);
        assert_eq!(f[idx("own.p0.status.sleep")], 0.0);
        assert_eq!(f[idx("own.p0.type.fire")], 1.0);
        assert_eq!(f[idx("own.p0.type.flying")], 1.0);
        assert_eq!(f[idx("own.p0.type.water")], 0.0);
        assert_eq!(f[idx("own.p0.tera.water")], 1.0);
        assert_eq!(f[idx("own.p0.terastallized")], 0.0);
        assert_eq!(f[idx("own.p0.has_item")], 1.0);
        assert_eq!(f[idx("own.p0.attack")], 0.5);
        // team index 0 (fainted) becomes slot 1 once the active (index 2) moves to the front
        assert_eq!(f[idx("own.p1.hp")], 0.0);
        assert_eq!(f[idx("own.p1.alive")], 0.0);
        assert_eq!(f[idx("own.p2.alive")], 1.0);
        assert_eq!(f[idx("own.boost.attack")], 0.5);
        assert_eq!(f[idx("own.boost.speed")], -1.0);
        assert_eq!(f[idx("own.vol.substitute")], 1.0);
        assert_eq!(f[idx("own.vol.perish")], 1.0);
        assert_eq!(f[idx("own.vol.leechseed")], 0.0);
        assert_eq!(f[idx("own.sc.stealth_rock")], 1.0);
        assert!((f[idx("own.sc.spikes")] - 2.0 / 3.0).abs() < 1e-6);
        assert!((f[idx("own.misc.alive_fraction")] - 5.0 / 6.0).abs() < 1e-6);
        assert_eq!(f[idx("opp.p0.terastallized")], 1.0);
        assert_eq!(f[idx("opp.sc.reflect")], 1.0);
        assert_eq!(f[idx("opp.p0.hp")], 1.0);
        assert_eq!(f[idx("field.weather.rain")], 1.0);
        assert_eq!(f[idx("field.weather.sun")], 0.0);
        assert_eq!(f[idx("field.trick_room")], 1.0);
    }

    #[test]
    fn side_two_perspective_swaps_own_and_opp() {
        let s = constructed();
        let a = features(&s, SideReference::SideOne);
        let b = features(&s, SideReference::SideTwo);
        assert_eq!(&a[..PER_SIDE], &b[PER_SIDE..2 * PER_SIDE]);
        assert_eq!(&a[PER_SIDE..2 * PER_SIDE], &b[..PER_SIDE]);
        assert_eq!(&a[2 * PER_SIDE..], &b[2 * PER_SIDE..]);
    }
}
