//! FEATURE_VERSION 4: v2 ++ v3 dense features plus per-slot identity indices for learned
//! embeddings (bead showdown-bot-589.65). See `docs/features-v4.md`.
//!
//! **Dense part** (`FEATURE_LEN_V4` = 738 + 1453 + 188 = 2379): v1 ++ the v2 extras ++ the v3
//! matchup block. Nothing is duplicated: the v2 extras are `features_v2::extras_into`, the
//! matchup block is `features_v3::matchup_blocks` emitted through a [`ShiftSink`] that moves
//! its indices behind the v2 extras (v3 on its own writes them directly after v1).
//!
//! **Identity part** (`N_IDS` = 84 `u16`): for each of the 12 Pokemon slots (own slots 0..5,
//! then opp slots 0..5; slot 0 = active, the rest in team order, exactly like v1) the 7 indices
//! `[species, item, ability, move0, move1, move2, move3]`, see [`identity_into`].
//!
//! Index space of every table (vocabulary): `0` = pad (fainted Pokemon: all of its identity is
//! hidden, the dense part already carries `alive`), `1` = unknown (unrevealed item
//! `UNKNOWNITEM`; ability / move / species `NONE`), `v + 2` for the engine enum variant with
//! discriminant `v` (so `Items::NONE`, "no item", is a real entry, `2`). The vocabularies ARE
//! the engine's enums in declaration order (`NAMES`); [`vocab_hash`] fingerprints their names
//! and is stored in the `.pev` so a net can only be loaded by an engine with identical
//! vocabularies. **Append new variants only at the END of the enums** (inserting in the middle
//! re-numbers everything after it and invalidates every v4 net).

use crate::choices::Choices;
use crate::engine::abilities::Abilities;
use crate::engine::items::Items;
use crate::features::{FEATURE_LEN_V1, TEAM_SLOTS};
use crate::features_v2::{self, DenseSink, Sink, FEATURE_LEN_V2};
use crate::features_v3;
use crate::pokemon::PokemonName;
use crate::state::{Pokemon, Side, SideReference, State};
use std::sync::OnceLock;

/// Where the v3 matchup block starts inside the v4 dense vector (right after the v2 extras).
pub const V3_SHIFT: usize = FEATURE_LEN_V2 - FEATURE_LEN_V1;
/// Length of the dense part.
pub const FEATURE_LEN_V4: usize = FEATURE_LEN_V2 + features_v3::V3_EXTRA;

/// Pokemon slots with identity indices (own 0..5, opp 0..5).
pub const N_SLOTS: usize = 2 * TEAM_SLOTS;
/// Indices per slot: species, item, ability, 4 moves.
pub const IDS_PER_SLOT: usize = 7;
/// Total number of identity indices.
pub const N_IDS: usize = N_SLOTS * IDS_PER_SLOT;

/// Embedding tables.
pub const N_TABLES: usize = 4;
pub const TABLE_NAMES: [&str; N_TABLES] = ["species", "item", "ability", "move"];
pub const SPECIES: usize = 0;
pub const ITEM: usize = 1;
pub const ABILITY: usize = 2;
pub const MOVE: usize = 3;

pub const IDX_PAD: u16 = 0;
pub const IDX_UNKNOWN: u16 = 1;
/// Rows reserved before the engine enum entries (pad, unknown).
pub const VOCAB_OFFSET: usize = 2;

/// Rows of table `table` (`COUNT + 2`).
pub fn vocab_size(table: usize) -> usize {
    VOCAB_OFFSET
        + match table {
            SPECIES => PokemonName::COUNT,
            ITEM => Items::COUNT,
            ABILITY => Abilities::COUNT,
            MOVE => Choices::COUNT,
            _ => panic!("unknown embedding table {}", table),
        }
}

/// Entry names of table `table`, in row order (`<pad>`, `<unknown>`, then the enum variants).
pub fn vocab_names(table: usize) -> Vec<String> {
    let names: &[&str] = match table {
        SPECIES => PokemonName::NAMES,
        ITEM => Items::NAMES,
        ABILITY => Abilities::NAMES,
        MOVE => Choices::NAMES,
        _ => panic!("unknown embedding table {}", table),
    };
    let mut out = vec!["<pad>".to_string(), "<unknown>".to_string()];
    out.extend(names.iter().map(|n| n.to_string()));
    out
}

/// FNV-1a 64 over `table:name\0` of every table in order: the vocabulary fingerprint stored in
/// `.pev` files.
pub fn vocab_hash() -> u64 {
    static H: OnceLock<u64> = OnceLock::new();
    *H.get_or_init(|| {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut eat = |b: u8| {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        };
        for (t, name) in TABLE_NAMES.iter().enumerate() {
            for b in name.bytes() {
                eat(b);
            }
            eat(b':');
            let names: &[&str] = match t {
                SPECIES => PokemonName::NAMES,
                ITEM => Items::NAMES,
                ABILITY => Abilities::NAMES,
                _ => Choices::NAMES,
            };
            for n in names {
                for b in n.bytes() {
                    eat(b);
                }
                eat(0);
            }
        }
        h
    })
}

#[inline(always)]
fn entry(disc: usize, unknown_disc: usize) -> u16 {
    if disc == unknown_disc {
        IDX_UNKNOWN
    } else {
        (disc + VOCAB_OFFSET) as u16
    }
}

#[inline(always)]
pub fn species_idx(p: &Pokemon) -> u16 {
    entry(p.id as usize, PokemonName::NONE as usize)
}
#[inline(always)]
pub fn item_idx(i: Items) -> u16 {
    entry(i as usize, Items::UNKNOWNITEM as usize)
}
#[inline(always)]
pub fn ability_idx(a: Abilities) -> u16 {
    entry(a as usize, Abilities::NONE as usize)
}
#[inline(always)]
pub fn move_idx(m: Choices) -> u16 {
    entry(m as usize, Choices::NONE as usize)
}

/// The 7 identity indices of one Pokemon slot; all `IDX_PAD` for a fainted Pokemon.
#[inline]
pub fn slot_ids(p: &Pokemon) -> [u16; IDS_PER_SLOT] {
    if p.hp <= 0 {
        return [IDX_PAD; IDS_PER_SLOT];
    }
    slot_ids_alive(p)
}

/// The identity indices of `p` as if it were alive (the fainted case is `slot_ids`).
#[inline]
pub fn slot_ids_alive(p: &Pokemon) -> [u16; IDS_PER_SLOT] {
    [
        species_idx(p),
        item_idx(p.item),
        ability_idx(p.ability),
        move_idx(p.moves.m0.id),
        move_idx(p.moves.m1.id),
        move_idx(p.moves.m2.id),
        move_idx(p.moves.m3.id),
    ]
}

/// Team indices in slot order: the active Pokemon first, then the rest in team order.
#[inline(always)]
pub fn slot_order(side: &Side) -> [usize; TEAM_SLOTS] {
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
    order
}

/// Identity indices of `state` from `side`'s perspective, `N_IDS` long (see the module doc).
pub fn identity_into(state: &State, side: SideReference, out: &mut [u16]) {
    let (own, opp) = match side {
        SideReference::SideOne => (&state.side_one, &state.side_two),
        SideReference::SideTwo => (&state.side_two, &state.side_one),
    };
    for (side_no, s) in [own, opp].iter().enumerate() {
        for (slot, k) in slot_order(s).iter().enumerate() {
            let o = (side_no * TEAM_SLOTS + slot) * IDS_PER_SLOT;
            out[o..o + IDS_PER_SLOT].copy_from_slice(&slot_ids(&s.pokemon.pkmn[*k]));
        }
    }
}

pub fn identity(state: &State, side: SideReference) -> Vec<u16> {
    let mut v = vec![0u16; N_IDS];
    identity_into(state, side, &mut v);
    v
}

/// Forwards a sink with every index moved up by `off`.
pub struct ShiftSink<'a, S: Sink> {
    pub inner: &'a mut S,
    pub off: usize,
}
impl<'a, S: Sink> Sink for ShiftSink<'a, S> {
    #[inline(always)]
    fn one(&mut self, idx: usize) {
        self.inner.one(idx + self.off);
    }
    #[inline(always)]
    fn val(&mut self, idx: usize, v: f32) {
        self.inner.val(idx + self.off, v);
    }
}

/// The v3 matchup block placed after the v2 extras (v4 dense indices).
#[inline(always)]
pub fn matchup_blocks_v4<S: Sink>(sink: &mut S, state: &State, own: SideReference) {
    features_v3::matchup_blocks(&mut ShiftSink { inner: sink, off: V3_SHIFT }, state, own);
}

/// Dense v4 extras for `state` from `side`'s perspective; `out` is `FEATURE_LEN_V4` long and the
/// v1 part is written by the caller. Writes the v2 extras, then the shifted v3 matchup block.
pub fn extras_into(state: &State, side: SideReference, out: &mut [f32]) {
    features_v2::extras_into(state, side, &mut out[..FEATURE_LEN_V2]);
    out[FEATURE_LEN_V2..FEATURE_LEN_V4].fill(0.0);
    matchup_blocks_v4(&mut DenseSink { buf: out }, state, side);
}

/// Names of the v4 extras (v2 extras ++ v3 matchup names), same order as `extras_into`.
pub fn extra_names() -> Vec<String> {
    let mut names = features_v2::extra_names();
    names.extend(features_v3::extra_names());
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{feature_names_versioned, features_versioned};

    fn states() -> Vec<State> {
        include_str!("../data/gen9randombattle.txt")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(State::deserialize)
            .collect()
    }

    #[test]
    fn dense_is_v2_then_shifted_v3() {
        assert_eq!(FEATURE_LEN_V4, 738 + 1453 + 188);
        assert_eq!(feature_names_versioned(4).len(), FEATURE_LEN_V4);
        for s in states().iter().take(20) {
            let v1 = features_versioned(s, SideReference::SideOne, 1);
            let v2 = features_versioned(s, SideReference::SideOne, 2);
            let v3 = features_versioned(s, SideReference::SideOne, 3);
            let v4 = features_versioned(s, SideReference::SideOne, 4);
            assert_eq!(v4.len(), FEATURE_LEN_V4);
            assert_eq!(&v4[..FEATURE_LEN_V2], &v2[..]);
            assert_eq!(&v4[..FEATURE_LEN_V1], &v1[..]);
            assert_eq!(&v4[FEATURE_LEN_V2..], &v3[FEATURE_LEN_V1..]);
        }
        let n = feature_names_versioned(4);
        assert_eq!(n[FEATURE_LEN_V2], "match.speed.own0.opp0");
    }

    #[test]
    fn vocab_is_stable() {
        // sizes follow the enums; pad + unknown come first
        assert_eq!(vocab_size(SPECIES), PokemonName::COUNT + 2);
        assert_eq!(vocab_size(ITEM), Items::COUNT + 2);
        assert_eq!(vocab_size(ABILITY), Abilities::COUNT + 2);
        assert_eq!(vocab_size(MOVE), Choices::COUNT + 2);
        for t in 0..N_TABLES {
            let names = vocab_names(t);
            assert_eq!(names.len(), vocab_size(t));
            assert_eq!(&names[..2], &["<pad>".to_string(), "<unknown>".to_string()]);
        }
        // the first real entries are the first enum variants (NONE is declared first)
        assert_eq!(vocab_names(MOVE)[2], "NONE");
        assert_eq!(vocab_names(ITEM)[2], "NONE");
        assert_eq!(vocab_names(ITEM)[3], "UNKNOWNITEM");
        // deterministic fingerprint
        assert_eq!(vocab_hash(), vocab_hash());
        assert_ne!(vocab_hash(), 0xcbf29ce484222325);
        // every id of a real state is inside its table
        for s in states() {
            for side in [SideReference::SideOne, SideReference::SideTwo] {
                let ids = identity(&s, side);
                for (i, id) in ids.iter().enumerate() {
                    let t = match i % IDS_PER_SLOT {
                        0 => SPECIES,
                        1 => ITEM,
                        2 => ABILITY,
                        _ => MOVE,
                    };
                    assert!((*id as usize) < vocab_size(t));
                }
            }
        }
    }

    #[test]
    fn unknown_none_and_fainted() {
        let mut s = states().remove(0);
        // own active, opp bench slot 1 (team index of the 2nd Pokemon in slot order)
        s.side_one.get_active().item = Items::UNKNOWNITEM;
        s.side_one.get_active().ability = Abilities::NONE;
        s.side_one.get_active().moves.m3.id = Choices::NONE;
        let ids = identity(&s, SideReference::SideOne);
        assert_eq!(ids[1], IDX_UNKNOWN, "unknown item");
        assert_eq!(ids[2], IDX_UNKNOWN, "no ability = unknown");
        assert_eq!(ids[6], IDX_UNKNOWN, "empty move = unknown");
        assert_ne!(ids[0], IDX_UNKNOWN);
        assert!(ids[0] > 1);
        // "no item" is a real entry, not unknown / pad
        s.side_one.get_active().item = Items::NONE;
        let ids = identity(&s, SideReference::SideOne);
        assert_eq!(ids[1] as usize, Items::NONE as usize + VOCAB_OFFSET);
        // fainted: everything pad
        s.side_one.get_active().hp = 0;
        let ids = identity(&s, SideReference::SideOne);
        assert!(ids[..IDS_PER_SLOT].iter().all(|i| *i == IDX_PAD));
        // side two's view swaps the halves
        let a = identity(&s, SideReference::SideOne);
        let b = identity(&s, SideReference::SideTwo);
        assert_eq!(a[..N_IDS / 2], b[N_IDS / 2..]);
        assert_eq!(a[N_IDS / 2..], b[..N_IDS / 2]);
    }

    #[test]
    fn slot_order_matches_v1() {
        // active first: switching the active changes slot 0's species
        let mut s = states().remove(0);
        let before = identity(&s, SideReference::SideOne);
        s.side_one.active_index = unsafe { std::mem::transmute::<u8, crate::state::PokemonIndex>(2) };
        let after = identity(&s, SideReference::SideOne);
        assert_eq!(after[0], species_idx(&s.side_one.pokemon.pkmn[2]));
        assert_ne!(before[0], after[0]);
        // the opposing half is untouched
        assert_eq!(before[N_IDS / 2..], after[N_IDS / 2..]);
    }
}
