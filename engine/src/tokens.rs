//! Token representation of a battle state for a transformer policy+value net (bead
//! showdown-bot-589.68.1). See `docs/tokens.md` for the full layout.
//!
//! Nothing here computes a feature of its own: the dense v4 vector (v1 ++ v2 extras ++ v3
//! matchup block, `features_v4`) is written once and sliced into tokens, the identity ids are
//! `features_v4::identity_into` (same vocabularies and vocab hash as v4) and the action tokens
//! are `prior::ActionCtx` features addressed by `prior::action_slot` (the same 13 slots the
//! policy prior and `poke_engine.policy_actions` use). Training and inference therefore cannot
//! disagree.
//!
//! The output is two flat buffers, `cat` (`i16`) and `cont` (`f32`); [`layout`] lists every
//! named field with its buffer, offset and shape.

use crate::choices::Choices;
use crate::engine::abilities::Abilities;
use crate::engine::items::Items;
use crate::features::{
    features_versioned_into, FEATURE_LEN_V4, FEATURE_VERSION_4, FIELD_LEN, PER_POKEMON, PER_SIDE,
    PER_SIDE_EXTRAS, TEAM_SLOTS,
};
use crate::features_v2::{
    mon_base, moves_of, FEATURE_LEN_V2, FIELD2_BASE, FIELD2_NAMES, PM2, PS2, SIDE2_BASE,
};
use crate::features_v3::{N_PAIR, PAIR_CHANNELS};
use crate::features_v4::{
    identity_into, move_idx, slot_ids, slot_order, IDS_PER_SLOT, IDX_UNKNOWN, N_IDS, N_SLOTS,
};
use crate::pokemon::PokemonName;
use crate::prior::{action_slot, ActionCtx, N_ACTIONS, N_AFEAT};
use crate::state::{Move, PokemonStatus, PokemonType, Side, SideReference, State};

/// 2: the opponent scrub also hides bench membership from the matchup aggregates, the alive
/// fraction and the opponent's switch action tokens (v1 leaked them).
pub const TOKEN_VERSION: u32 = 2;

/// Pokemon tokens (own 0..5, opp 6..11; slot 0 of each side is the active Pokemon).
pub const N_MON: usize = N_SLOTS;
/// Width of the `known` flags per Pokemon: species, item, ability, move0..3, tera type.
pub const KNOWN_W: usize = 8;
/// Continuous width of a Pokemon token: v1 slot block ++ v2 per-mon block ++ [is_opp, is_active].
pub const C_MON: usize = PER_POKEMON + PM2 + 2;
/// Continuous width of a side token: v1 side extras ++ v2 side block (screen turns + active
/// attacker-vs-defender block).
pub const C_SIDE: usize = PER_SIDE_EXTRAS + PS2;
/// Continuous width of the global (field) token: v1 field block ++ v2 field turns.
pub const C_GLOB: usize = FIELD_LEN + FIELD2_NAMES.len();
/// Pair channels / aggregates of the matchup block.
pub const N_PAIR_CH: usize = PAIR_CHANNELS.len();
pub const N_AGG: usize = 8;
/// Candidate action tokens per side.
pub const N_ACT: usize = N_ACTIONS;
/// Categorical columns of an action token: [move id (move table), species id (species table)].
pub const ACT_CAT_W: usize = 2;
/// Hidden-set target width per opponent slot: species, item, ability, 4 moves (sorted), tera type.
pub const HID_W: usize = 8;
/// Width of the `meta` field: [own active team index, opp active team index, opp scrubbed, 0].
pub const META_W: usize = 4;

// cat buffer (i16) offsets
pub const O_MON_CAT: usize = 0;
pub const O_MON_KNOWN: usize = O_MON_CAT + N_MON * IDS_PER_SLOT;
pub const O_ACT_CAT: usize = O_MON_KNOWN + N_MON * KNOWN_W;
pub const O_ACT_LEGAL: usize = O_ACT_CAT + 2 * N_ACT * ACT_CAT_W;
pub const O_META: usize = O_ACT_LEGAL + 2 * N_ACT;
pub const CAT_LEN: usize = O_META + META_W;

// cont buffer (f32) offsets
pub const O_MON_CONT: usize = 0;
pub const O_SIDE_CONT: usize = O_MON_CONT + N_MON * C_MON;
pub const O_GLOB: usize = O_SIDE_CONT + 2 * C_SIDE;
pub const O_PAIR: usize = O_GLOB + C_GLOB;
pub const O_AGG: usize = O_PAIR + N_PAIR * N_PAIR_CH;
pub const O_ACT_CONT: usize = O_AGG + N_AGG;
pub const CONT_LEN: usize = O_ACT_CONT + 2 * N_ACT * N_AFEAT;

/// What an opponent revealed so far (a row of the recorder's `opp_facts`).
#[derive(Clone, Debug, Default)]
pub struct Revealed {
    pub species: String,
    pub moves: Vec<String>,
    pub item: bool,
    pub ability: bool,
    pub tera: bool,
}

fn norm(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// Hides everything about `opp` that `rev` does not list as revealed: item -> UNKNOWNITEM,
/// ability -> NONE, unrevealed moves -> NONE (revealed moves are compacted to the front in
/// their original order), unrevealed Tera type -> TYPELESS. A Pokemon that `rev` does not list
/// (and that is not the active one) is fully unrevealed. Returns the `known` flags per team index.
fn scrub_opp(opp: &mut Side, rev: &[Revealed]) -> [[i16; KNOWN_W]; TEAM_SLOTS] {
    let active = opp.active_index as usize;
    let mut known = [[0i16; KNOWN_W]; TEAM_SLOTS];
    let empty = Revealed::default();
    let rev_norm: Vec<String> = rev.iter().map(|r| norm(&r.species)).collect();
    for k in 0..TEAM_SLOTS {
        let p = &mut opp.pokemon.pkmn[k];
        let name = norm(PokemonName::NAMES[p.id as usize]);
        let r = match rev_norm.iter().position(|n| *n == name) {
            Some(i) => Some(&rev[i]),
            None if k == active => Some(&empty),
            None => None,
        };
        let Some(r) = r else {
            p.item = Items::UNKNOWNITEM;
            p.ability = Abilities::NONE;
            p.moves.m0 = Move::default();
            p.moves.m1 = Move::default();
            p.moves.m2 = Move::default();
            p.moves.m3 = Move::default();
            p.tera_type = PokemonType::TYPELESS;
            // a never-revealed Pokemon is canonically alive and untouched: its hp cannot leak
            p.hp = p.maxhp.max(1);
            p.status = PokemonStatus::NONE;
            p.rest_turns = 0;
            p.sleep_turns = 0;
            p.terastallized = false;
            p.mega_evolved = false;
            continue;
        };
        known[k][0] = 1;
        if r.item {
            known[k][1] = 1;
        } else {
            p.item = Items::UNKNOWNITEM;
        }
        if r.ability {
            known[k][2] = 1;
        } else {
            p.ability = Abilities::NONE;
        }
        let rm: Vec<String> = r.moves.iter().map(|m| norm(m)).collect();
        let all = [p.moves.m0.clone(), p.moves.m1.clone(), p.moves.m2.clone(), p.moves.m3.clone()];
        let mut kept: Vec<Move> = Vec::with_capacity(4);
        for m in all {
            if m.id != Choices::NONE && rm.contains(&norm(Choices::NAMES[m.id as usize])) {
                kept.push(m);
            }
        }
        for i in 0..kept.len() {
            known[k][3 + i] = 1;
        }
        while kept.len() < 4 {
            kept.push(Move::default());
        }
        let mut it = kept.into_iter();
        p.moves.m0 = it.next().unwrap();
        p.moves.m1 = it.next().unwrap();
        p.moves.m2 = it.next().unwrap();
        p.moves.m3 = it.next().unwrap();
        if r.tera {
            known[k][7] = 1;
        } else {
            p.tera_type = PokemonType::TYPELESS;
        }
    }
    known
}

/// Known flags derived from the state's own convention (nothing scrubbed).
fn convention_known(side: &Side) -> [[i16; KNOWN_W]; TEAM_SLOTS] {
    let mut known = [[0i16; KNOWN_W]; TEAM_SLOTS];
    for k in 0..TEAM_SLOTS {
        let p = &side.pokemon.pkmn[k];
        known[k][0] = (p.id != PokemonName::NONE) as i16;
        known[k][1] = (p.item != Items::UNKNOWNITEM) as i16;
        known[k][2] = (p.ability != Abilities::NONE) as i16;
        for (i, m) in moves_of(p).iter().enumerate() {
            known[k][3 + i] = (m.id != Choices::NONE) as i16;
        }
        known[k][7] = (p.tera_type != PokemonType::TYPELESS) as i16;
    }
    known
}

#[inline]
fn pair_idx(ch: usize, pair: usize) -> usize {
    FEATURE_LEN_V2 + ch * N_PAIR + pair
}

/// Writes the token buffers of `state` from `side`'s view. `reveal` (optional) restricts the
/// opponent to what it lists as revealed (see [`scrub_opp`]); without it the state is tokenised
/// as it is and the `known` flags follow the state's unknown convention.
/// `cat` must hold `CAT_LEN`, `cont` `CONT_LEN` entries; every one is written.
pub fn state_tokens_into(
    state: &State,
    side: SideReference,
    reveal: Option<&[Revealed]>,
    cat: &mut [i16],
    cont: &mut [f32],
) {
    cat[..CAT_LEN].fill(0);
    cont[..CONT_LEN].fill(0.0);
    let scrubbed;
    let (st, opp_known_team) = match reveal {
        Some(rev) => {
            let mut c = state.clone();
            let known = match side {
                SideReference::SideOne => scrub_opp(&mut c.side_two, rev),
                SideReference::SideTwo => scrub_opp(&mut c.side_one, rev),
            };
            scrubbed = c;
            (&scrubbed, Some(known))
        }
        None => (state, None),
    };
    let (own, opp) = match side {
        SideReference::SideOne => (&st.side_one, &st.side_two),
        SideReference::SideTwo => (&st.side_two, &st.side_one),
    };
    let sides = [own, opp];

    // dense v4 once
    let mut dense = [0f32; FEATURE_LEN_V4];
    features_versioned_into(st, side, FEATURE_VERSION_4, &mut dense);

    // matchup block source: hidden opponent mons are removed (hp 0) so that neither the pair
    // values nor the aggregates see them; the other opponent features keep them alive.
    let hidden_team: Vec<usize> = match &opp_known_team {
        Some(k) => (0..TEAM_SLOTS).filter(|&i| k[i][0] == 0).collect(),
        None => Vec::new(),
    };
    let mut dense_buf = [0f32; FEATURE_LEN_V4];
    let dense_m: &[f32; FEATURE_LEN_V4] = if hidden_team.is_empty() {
        &dense
    } else {
        let mut c = st.clone();
        let o = match side {
            SideReference::SideOne => &mut c.side_two,
            SideReference::SideTwo => &mut c.side_one,
        };
        for &i in &hidden_team {
            o.pokemon.pkmn[i].hp = 0;
        }
        features_versioned_into(&c, side, FEATURE_VERSION_4, &mut dense_buf);
        &dense_buf
    };

    // identity ids
    let mut ids = [0u16; N_IDS];
    identity_into(st, side, &mut ids);

    let orders = [slot_order(own), slot_order(opp)];
    let own_known = convention_known(own);
    let opp_known = opp_known_team.unwrap_or_else(|| convention_known(opp));
    let mut opp_hidden = [false; TEAM_SLOTS]; // per opp slot: species unrevealed (scrubbed)

    for side_no in 0..2 {
        for slot in 0..TEAM_SLOTS {
            let m = side_no * TEAM_SLOTS + slot;
            let k = orders[side_no][slot];
            let known = if side_no == 0 { &own_known[k] } else { &opp_known[k] };
            let hidden = side_no == 1 && opp_known_team.is_some() && known[0] == 0;
            if side_no == 1 {
                opp_hidden[slot] = hidden;
            }
            // categorical ids + known flags
            for c in 0..IDS_PER_SLOT {
                cat[O_MON_CAT + m * IDS_PER_SLOT + c] = if hidden { IDX_UNKNOWN as i16 } else { ids[m * IDS_PER_SLOT + c] as i16 };
            }
            for c in 0..KNOWN_W {
                cat[O_MON_KNOWN + m * KNOWN_W + c] = known[c];
            }
            // continuous slot features
            let o = O_MON_CONT + m * C_MON;
            if !hidden {
                let v1 = side_no * PER_SIDE + slot * PER_POKEMON;
                cont[o..o + PER_POKEMON].copy_from_slice(&dense[v1..v1 + PER_POKEMON]);
                let v2 = mon_base(side_no, slot);
                cont[o + PER_POKEMON..o + PER_POKEMON + PM2].copy_from_slice(&dense[v2..v2 + PM2]);
            }
            cont[o + PER_POKEMON + PM2] = side_no as f32;
            cont[o + PER_POKEMON + PM2 + 1] = (slot == 0) as u8 as f32;
        }
    }

    // side tokens
    for side_no in 0..2 {
        let o = O_SIDE_CONT + side_no * C_SIDE;
        let v1 = side_no * PER_SIDE + TEAM_SLOTS * PER_POKEMON;
        cont[o..o + PER_SIDE_EXTRAS].copy_from_slice(&dense[v1..v1 + PER_SIDE_EXTRAS]);
        let v2 = SIDE2_BASE + side_no * PS2;
        cont[o + PER_SIDE_EXTRAS..o + C_SIDE].copy_from_slice(&dense[v2..v2 + PS2]);
    }
    // global token
    cont[O_GLOB..O_GLOB + FIELD_LEN].copy_from_slice(&dense[2 * PER_SIDE..2 * PER_SIDE + FIELD_LEN]);
    cont[O_GLOB + FIELD_LEN..O_GLOB + C_GLOB].copy_from_slice(&dense[FIELD2_BASE..FIELD2_BASE + FIELD2_NAMES.len()]);

    // matchup block: pair-major [36][5], then 8 aggregates
    for pair in 0..N_PAIR {
        let j = pair % TEAM_SLOTS;
        if opp_hidden[j] {
            continue;
        }
        for ch in 0..N_PAIR_CH {
            cont[O_PAIR + pair * N_PAIR_CH + ch] = dense_m[pair_idx(ch, pair)];
        }
    }
    cont[O_AGG..O_AGG + N_AGG].copy_from_slice(&dense_m[pair_idx(N_PAIR_CH, 0)..pair_idx(N_PAIR_CH, 0) + N_AGG]);

    // action tokens, own then opp
    let (o1, o2) = st.root_get_all_options();
    let (own_opts, opp_opts) = match side {
        SideReference::SideOne => (o1, o2),
        SideReference::SideTwo => (o2, o1),
    };
    let who = [side, side.get_other_side()];
    let opts = [own_opts, opp_opts];
    for side_no in 0..2 {
        let sd = sides[side_no];
        let order = &orders[side_no];
        let active = sd.get_active_immutable();
        for s in 0..8 {
            cat[O_ACT_CAT + (side_no * N_ACT + s) * ACT_CAT_W] = move_idx(moves_of(active)[s % 4].id) as i16;
        }
        for s in 8..N_ACT {
            let k = order[s - 8 + 1];
            cat[O_ACT_CAT + (side_no * N_ACT + s) * ACT_CAT_W + 1] = if side_no == 1 && opp_hidden[s - 8 + 1] {
                IDX_UNKNOWN as i16
            } else {
                slot_ids(&sd.pokemon.pkmn[k])[0] as i16
            };
        }
        let ctx = ActionCtx::new(st, who[side_no]);
        let mut f = [0f32; N_AFEAT];
        for opt in &opts[side_no] {
            if let Some(s) = action_slot(sd, opt) {
                cat[O_ACT_LEGAL + side_no * N_ACT + s] = 1;
                if side_no == 1 && s >= 8 && opp_hidden[s - 8 + 1] {
                    continue; // switch to a never-revealed Pokemon: legal, but nothing about it is known
                }
                ctx.features(opt, &mut f);
                let o = O_ACT_CONT + (side_no * N_ACT + s) * N_AFEAT;
                cont[o..o + N_AFEAT].copy_from_slice(&f);
            }
        }
    }

    cat[O_META] = own.active_index as i16;
    cat[O_META + 1] = opp.active_index as i16;
    cat[O_META + 2] = reveal.is_some() as i16;
}

/// Hidden-set targets for the opponent slots of `input` (from `side`'s view): for every
/// opponent slot, the true set of the Pokemon of the same species in `truth`.
/// Returns `(targets [6][HID_W], mask [6], truth_team [6])` flattened: `6 * HID_W` ids, then 6
/// mask flags, then the true opponent species ids in `truth` slot order. Targets are vocab ids
/// (see `features_v4`) with the 4 moves sorted ascending and the Tera type as the raw
/// `PokemonType` discriminant (19 = TYPELESS = none).
pub fn hidden_targets(input: &State, truth: &State, side: SideReference) -> Vec<i16> {
    use crate::features_v4::{ability_idx, item_idx, species_idx};
    let opp_of = |s: &State| match side {
        SideReference::SideOne => s.side_two.clone(),
        SideReference::SideTwo => s.side_one.clone(),
    };
    let (io, to) = (opp_of(input), opp_of(truth));
    let (iord, tord) = (slot_order(&io), slot_order(&to));
    let mut out = vec![0i16; TEAM_SLOTS * HID_W + 2 * TEAM_SLOTS];
    for slot in 0..TEAM_SLOTS {
        let ip = &io.pokemon.pkmn[iord[slot]];
        let tp = tord.iter().map(|&k| &to.pokemon.pkmn[k]).find(|t| t.id == ip.id && ip.id != PokemonName::NONE);
        if let Some(t) = tp {
            let o = slot * HID_W;
            out[o] = species_idx(t) as i16;
            out[o + 1] = item_idx(t.item) as i16;
            out[o + 2] = ability_idx(t.ability) as i16;
            let mut mv: Vec<i16> = moves_of(t).iter().map(|m| move_idx(m.id) as i16).collect();
            mv.sort_unstable();
            out[o + 3..o + 7].copy_from_slice(&mv);
            out[o + 7] = t.tera_type as i16;
            out[TEAM_SLOTS * HID_W + slot] = 1;
        }
    }
    for slot in 0..TEAM_SLOTS {
        let t = &to.pokemon.pkmn[tord[slot]];
        out[TEAM_SLOTS * HID_W + TEAM_SLOTS + slot] = species_idx(t) as i16;
    }
    out
}

/// (name, buffer, offset, shape, dtype) of every field, in buffer order.
pub fn layout() -> Vec<(&'static str, &'static str, usize, Vec<usize>, &'static str)> {
    vec![
        ("mon_cat", "cat", O_MON_CAT, vec![N_MON, IDS_PER_SLOT], "int16"),
        ("mon_known", "cat", O_MON_KNOWN, vec![N_MON, KNOWN_W], "int16"),
        ("act_cat", "cat", O_ACT_CAT, vec![2, N_ACT, ACT_CAT_W], "int16"),
        ("act_legal", "cat", O_ACT_LEGAL, vec![2, N_ACT], "int16"),
        ("meta", "cat", O_META, vec![META_W], "int16"),
        ("mon_cont", "cont", O_MON_CONT, vec![N_MON, C_MON], "float32"),
        ("side_cont", "cont", O_SIDE_CONT, vec![2, C_SIDE], "float32"),
        ("glob", "cont", O_GLOB, vec![C_GLOB], "float32"),
        ("match_pair", "cont", O_PAIR, vec![N_PAIR, N_PAIR_CH], "float32"),
        ("match_agg", "cont", O_AGG, vec![N_AGG], "float32"),
        ("act_cont", "cont", O_ACT_CONT, vec![2, N_ACT, N_AFEAT], "float32"),
    ]
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

    fn tok(s: &State, side: SideReference, rev: Option<&[Revealed]>) -> (Vec<i16>, Vec<f32>) {
        let mut cat = vec![0i16; CAT_LEN];
        let mut cont = vec![0f32; CONT_LEN];
        state_tokens_into(s, side, rev, &mut cat, &mut cont);
        (cat, cont)
    }

    #[test]
    fn ids_match_v4_identity() {
        for s in states().iter().take(20) {
            for side in [SideReference::SideOne, SideReference::SideTwo] {
                let (cat, cont) = tok(s, side, None);
                let ids = crate::features_v4::identity(s, side);
                for i in 0..N_IDS {
                    assert_eq!(cat[O_MON_CAT + i], ids[i] as i16);
                }
                assert!(cont.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn own_opp_symmetry() {
        for s in states().iter().take(20) {
            let (c1, f1) = tok(s, SideReference::SideOne, None);
            let (c2, f2) = tok(s, SideReference::SideTwo, None);
            let half = N_MON / 2 * IDS_PER_SLOT;
            assert_eq!(c1[O_MON_CAT..O_MON_CAT + half], c2[O_MON_CAT + half..O_MON_CAT + 2 * half]);
            assert_eq!(c1[O_MON_CAT + half..O_MON_CAT + 2 * half], c2[O_MON_CAT..O_MON_CAT + half]);
            // action tokens swap too
            let n = N_ACT * ACT_CAT_W;
            assert_eq!(c1[O_ACT_CAT..O_ACT_CAT + n], c2[O_ACT_CAT + n..O_ACT_CAT + 2 * n]);
            // pokemon features of own half match the other side's opp half (flags aside)
            for slot in 0..TEAM_SLOTS {
                let a = O_MON_CONT + slot * C_MON;
                let b = O_MON_CONT + (TEAM_SLOTS + slot) * C_MON;
                assert_eq!(f1[a..a + PER_POKEMON + PM2], f2[b..b + PER_POKEMON + PM2]);
            }
        }
    }

    #[test]
    fn scrub_hides_unrevealed() {
        let s = &states()[0];
        let opp = &s.side_two;
        let active = &opp.pokemon.pkmn[opp.active_index as usize];
        let name = PokemonName::NAMES[active.id as usize].to_string();
        let mv = Choices::NAMES[active.moves.m1.id as usize].to_string();
        let rev = vec![Revealed { species: name, moves: vec![mv], item: false, ability: true, tera: false }];
        let (cat, _) = tok(s, SideReference::SideOne, Some(&rev));
        let k = O_MON_KNOWN + TEAM_SLOTS * KNOWN_W;
        assert_eq!(&cat[k..k + KNOWN_W], &[1, 0, 1, 1, 0, 0, 0, 0]);
        let c = O_MON_CAT + TEAM_SLOTS * IDS_PER_SLOT;
        assert_eq!(cat[c + 1], IDX_UNKNOWN as i16, "item hidden");
        assert_eq!(cat[c + 4], IDX_UNKNOWN as i16, "unrevealed move hidden");
        assert_ne!(cat[c + 3], IDX_UNKNOWN as i16, "revealed move kept");
        // a bench Pokemon that was never revealed is fully unknown
        let b = O_MON_CAT + (TEAM_SLOTS + 1) * IDS_PER_SLOT;
        assert!(cat[b..b + IDS_PER_SLOT].iter().all(|&i| i == IDX_UNKNOWN as i16 || i == 0));
    }

    #[test]
    fn hidden_targets_of_truth_are_full() {
        let s = &states()[0];
        let t = hidden_targets(s, s, SideReference::SideOne);
        assert!(t[TEAM_SLOTS * HID_W..TEAM_SLOTS * HID_W + TEAM_SLOTS].iter().all(|&m| m == 1));
    }

    #[test]
    fn unrevealed_bench_does_not_leak() {
        let all = states();
        let s = &all[0];
        let donor = &all[1];
        let opp = &s.side_two;
        let active = &opp.pokemon.pkmn[opp.active_index as usize];
        let rev = vec![Revealed {
            species: PokemonName::NAMES[active.id as usize].to_string(),
            moves: vec![Choices::NAMES[active.moves.m0.id as usize].to_string()],
            item: false,
            ability: false,
            tera: false,
        }];
        let base = tok(s, SideReference::SideOne, Some(&rev));
        for k in 0..TEAM_SLOTS {
            if k == opp.active_index as usize {
                continue;
            }
            let mut t = s.clone();
            // replace the never-revealed bench Pokemon by an unrelated one (species, set, hp)
            let mut p = donor.side_two.pokemon.pkmn[k].clone();
            if rev.iter().any(|r| norm(&r.species) == norm(PokemonName::NAMES[p.id as usize])) {
                continue;
            }
            p.hp = 1;
            t.side_two.pokemon.pkmn[k] = p;
            let other = tok(&t, SideReference::SideOne, Some(&rev));
            assert_eq!(base.0, other.0, "cat differs for bench {k}");
            assert_eq!(base.1, other.1, "cont differs for bench {k}");
        }
    }
}
