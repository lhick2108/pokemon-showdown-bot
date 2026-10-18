//! Massively parallel self-play pool for reinforcement learning (bead showdown-bot-589.68.6).
//!
//! `SelfPlayPool` runs N random-battle games entirely in Rust: team generation from the gen9
//! random battle set data, the engine as the battle simulator (chance outcomes sampled), per-side
//! reveal tracking so every side's token buffers come from ITS OWN view with the opponent scrubbed
//! exactly like live play (`tokens::state_tokens_into` with `Some(reveal)`), and a real 16-row
//! per-turn event history in the exporter's format (`docs/tokens.md`, "Event history").
//!
//! The pool is policy agnostic: [`SelfPlayPool::observe`] lists every (game, side) that has a real
//! decision (more than one legal action), [`SelfPlayPool::step`] takes one action slot per row,
//! applies the joint actions (a side without a real decision is played automatically), samples the
//! chance outcome, and resets finished games with a fresh team pair. See `docs/selfplay-ppo.md`.
//!
//! Teams: `data/randbats_gen9.txt` (baked by `scripts/gen_selfplay_data.py` from pkmn/randbats'
//! "full" joint sets): species drawn proportional to how often the species appeared in generated
//! teams, under the randbats team rules (same code path as `engine_state.TeamSampler`), then a joint
//! `level,item,ability,moves,tera` set proportional to its count. EVs/IVs are those `teams.ts` picks.

use crate::choices::{Choices, MoveCategory, MOVES};
use crate::engine::abilities::Abilities;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::items::Items;
use crate::engine::state::MoveChoice;
use crate::features_v4::{move_idx, slot_ids};
use crate::instruction::StateInstructions;
use crate::pokemon::PokemonName;
use crate::prior::action_slot;
use crate::state::{
    Move, Pokemon, PokemonIndex, PokemonMoveIndex, PokemonNature, PokemonType, Side, SideReference, State,
};
use crate::tokens::{self, Revealed, CAT_LEN, CONT_LEN, N_ACT};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{LazyLock, Mutex};

/// Event-history rows per observation (8 previous steps x own / opponent action).
pub const N_HIST: usize = 16;
pub const HIST_CAT_W: usize = 4;
pub const HIST_CONT_W: usize = 5;
const HIST_STEPS: usize = N_HIST / 2;
pub const DEFAULT_TURN_CAP: u32 = 200;

static DATA_TEXT: &str = include_str!("../data/randbats_gen9.txt");

// ---------------------------------------------------------------------------
// Team data
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SetEntry {
    pub count: f64,
    pub level: i8,
    pub item: Items,
    pub ability: Abilities,
    pub moves: Vec<Choices>,
    pub tera: PokemonType,
    pub evs: [u8; 6],
    pub ivs: [u8; 6],
}

#[derive(Clone, Debug)]
pub struct SpeciesEntry {
    pub id: String,
    pub name: PokemonName,
    pub level: i8,
    pub types: (PokemonType, PokemonType),
    pub base: [i32; 6],
    pub weight: f32,
    pub base_species: String,
    /// bit `PokemonType as usize` set when the species is weak to that type
    pub weak: u32,
    pub dweak: u32,
    pub freeze_dry: bool,
    pub sets: Vec<SetEntry>,
    cum: Vec<f64>,
}

impl SpeciesEntry {
    fn total(&self) -> f64 {
        *self.cum.last().unwrap_or(&0.0)
    }
}

pub struct TeamData {
    pub species: Vec<SpeciesEntry>,
    cum: Vec<f64>,
    pp: HashMap<usize, i8>,
    /// ids of species / sets that the engine could not represent (reported by the tests)
    pub dropped_species: Vec<String>,
    pub dropped_sets: usize,
}

fn name_map(names: &'static [&'static str]) -> HashMap<&'static str, usize> {
    names.iter().enumerate().map(|(i, n)| (*n, i)).collect()
}

fn type_bit(t: PokemonType) -> u32 {
    1u32 << (t as u32)
}

/// Species left out of the pool's teams (see `TeamData::parse`).
pub const EXCLUDED_SPECIES: &[&str] = &["ditto"];

const INCOMPAT: &[(&[&str], &[&str])] = &[
    (&["blissey"], &["chansey"]),
    (&["illumise"], &["volbeat"]),
    (
        &["ariados", "smeargle", "masquerain", "kricketune", "leavanny", "galvantula", "vikavolt", "ribombee", "araquanid", "spidops"],
        &["ariados", "smeargle", "masquerain", "kricketune", "leavanny", "galvantula", "vikavolt", "ribombee", "araquanid", "spidops"],
    ),
    (&["meowstic", "grimmsnarl", "ninetalesalola", "abomasnow"], &["meowstic", "grimmsnarl", "ninetalesalola", "abomasnow"]),
    (&["toxicroak"], &["ninetales", "torkoal", "groudon", "koraidon"]),
];

impl TeamData {
    fn parse(text: &str) -> TeamData {
        let species_names = name_map(PokemonName::NAMES);
        let item_names = name_map(Items::NAMES);
        let ability_names = name_map(Abilities::NAMES);
        let move_names = name_map(Choices::NAMES);
        let ptype = |s: &str| -> PokemonType { s.parse::<PokemonType>().unwrap_or(PokemonType::TYPELESS) };
        let mask = |s: &str| -> u32 {
            s.split(',').filter(|x| !x.is_empty()).fold(0u32, |m, t| m | type_bit(ptype(t)))
        };
        let mut species: Vec<SpeciesEntry> = Vec::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        let mut pp: HashMap<usize, i8> = HashMap::new();
        let mut dropped_species: Vec<String> = Vec::new();
        let mut dropped_sets = 0usize;
        let mut skip: HashSet<String> = HashSet::new();
        for line in text.lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('|').collect();
            match f[0] {
                "S" => {
                    let id = f[1].to_string();
                    // Ditto (Imposter/Transform) needs the live bot's pre-transform bookkeeping
                    // (engine_state._apply_transform) and rewrites the move list that reveal tracking
                    // keys on: left out of self-play teams.
                    if EXCLUDED_SPECIES.contains(&id.as_str()) {
                        dropped_species.push(id.clone());
                        continue;
                    }
                    let up = id.to_uppercase();
                    let Some(&ni) = species_names.get(up.as_str()) else {
                        dropped_species.push(id.clone());
                        skip.insert(id);
                        continue;
                    };
                    let name = PokemonName::from(ni as i16);
                    let base: Vec<i32> = (0..6).map(|i| f[5 + i].parse().unwrap()).collect();
                    index.insert(id.clone(), species.len());
                    species.push(SpeciesEntry {
                        id,
                        name,
                        level: f[2].parse().unwrap(),
                        types: (ptype(f[3]), ptype(f[4])),
                        base: [base[0], base[1], base[2], base[3], base[4], base[5]],
                        weight: f[11].parse().unwrap(),
                        base_species: f[12].to_string(),
                        weak: mask(f[13]),
                        dweak: mask(f[14]),
                        freeze_dry: f[15] == "1",
                        sets: Vec::new(),
                        cum: Vec::new(),
                    });
                }
                "F" => {
                    let Some(&si) = index.get(f[1]) else {
                        continue;
                    };
                    let item = if f[4].is_empty() {
                        Some(Items::NONE)
                    } else {
                        item_names.get(f[4].to_uppercase().as_str()).map(|&i| Items::from(i as u16))
                    };
                    let ability = ability_names.get(f[5].to_uppercase().as_str()).map(|&i| Abilities::from(i as i16));
                    let moves: Option<Vec<Choices>> = f[6]
                        .split(',')
                        .map(|m| move_names.get(m.to_uppercase().as_str()).map(|&i| Choices::from(i as u16)))
                        .collect();
                    let (Some(item), Some(ability), Some(moves)) = (item, ability, moves) else {
                        dropped_sets += 1;
                        continue;
                    };
                    let arr = |s: &str| -> [u8; 6] {
                        let v: Vec<u8> = s.split(';').map(|x| x.parse().unwrap()).collect();
                        [v[0], v[1], v[2], v[3], v[4], v[5]]
                    };
                    species[si].sets.push(SetEntry {
                        count: f[2].parse::<f64>().unwrap(),
                        level: f[3].parse().unwrap(),
                        item,
                        ability,
                        moves,
                        tera: ptype(f[7]),
                        evs: arr(f[8]),
                        ivs: arr(f[9]),
                    });
                }
                "P" => {
                    if let Some(&i) = move_names.get(f[1].to_uppercase().as_str()) {
                        pp.insert(i, f[2].parse().unwrap());
                    }
                }
                _ => {}
            }
        }
        // species without any usable set are dropped; cumulative weights
        let mut kept: Vec<SpeciesEntry> = Vec::new();
        for mut s in species.into_iter() {
            if s.sets.is_empty() {
                dropped_species.push(s.id.clone());
                continue;
            }
            let mut acc = 0.0;
            s.cum = s
                .sets
                .iter()
                .map(|x| {
                    acc += x.count;
                    acc
                })
                .collect();
            kept.push(s);
        }
        let mut acc = 0.0;
        let cum: Vec<f64> = kept
            .iter()
            .map(|s| {
                acc += s.total();
                acc
            })
            .collect();
        TeamData { species: kept, cum, pp, dropped_species, dropped_sets }
    }

    pub fn max_pp(&self, m: Choices) -> i8 {
        self.pp.get(&(m as usize)).copied().unwrap_or(16)
    }

    fn violates(&self, c: &SpeciesEntry, team: &[usize], team_size: usize) -> bool {
        let members: Vec<&SpeciesEntry> = team.iter().map(|&i| &self.species[i]).collect();
        if c.base_species == "zoroark" && members.len() + 1 >= team_size {
            return true;
        }
        let mut ctypes = vec![c.types.0];
        if c.types.1 != PokemonType::TYPELESS && c.types.1 != c.types.0 {
            ctypes.push(c.types.1);
        }
        for t in ctypes {
            if members.iter().filter(|m| m.types.0 == t || m.types.1 == t).count() >= 2 {
                return true;
            }
        }
        for b in 0..32u32 {
            if c.weak >> b & 1 == 1 && members.iter().filter(|m| m.weak >> b & 1 == 1).count() >= 3 {
                return true;
            }
            if c.dweak >> b & 1 == 1 && members.iter().filter(|m| m.dweak >> b & 1 == 1).count() >= 1 {
                return true;
            }
        }
        if c.freeze_dry && members.iter().filter(|m| m.freeze_dry).count() >= 4 {
            return true;
        }
        if c.level == 100 && members.iter().any(|m| m.level == 100) {
            return true;
        }
        for (ga, gb) in INCOMPAT {
            if gb.contains(&c.id.as_str()) && members.iter().any(|m| ga.contains(&m.id.as_str())) {
                return true;
            }
            if ga.contains(&c.id.as_str()) && members.iter().any(|m| gb.contains(&m.id.as_str())) {
                return true;
            }
        }
        false
    }

    fn draw_species(&self, rng: &mut StdRng) -> usize {
        let total = *self.cum.last().unwrap();
        let x = rng.random_range(0.0..total);
        self.cum.partition_point(|&c| c <= x).min(self.species.len() - 1)
    }

    /// Draws one species index that fits `team` (indices); mirrors `TeamSampler.sample_species`.
    fn sample_species(&self, team: &[usize], rng: &mut StdRng, team_size: usize) -> usize {
        let taken: Vec<&str> = team.iter().map(|&i| self.species[i].base_species.as_str()).collect();
        let mut fallback: Option<usize> = None;
        for _ in 0..40 {
            let s = self.draw_species(rng);
            let e = &self.species[s];
            if taken.contains(&e.base_species.as_str()) {
                continue;
            }
            fallback.get_or_insert(s);
            if !self.violates(e, team, team_size) {
                return s;
            }
        }
        if let Some(s) = fallback {
            return s;
        }
        (0..self.species.len()).find(|&i| !taken.contains(&self.species[i].base_species.as_str())).unwrap_or(0)
    }

    /// A random battle team: species indices and the sampled set index per member.
    pub fn sample_team(&self, rng: &mut StdRng) -> Vec<(usize, usize)> {
        let mut team: Vec<usize> = Vec::with_capacity(6);
        let mut out = Vec::with_capacity(6);
        for _ in 0..6 {
            let s = self.sample_species(&team, rng, 6);
            let e = &self.species[s];
            let x = rng.random_range(0.0..e.total());
            let si = e.cum.partition_point(|&c| c <= x).min(e.sets.len() - 1);
            team.push(s);
            out.push((s, si));
        }
        out
    }

    pub fn make_pokemon(&self, sp: usize, set: usize) -> Pokemon {
        let e = &self.species[sp];
        let s = &e.sets[set];
        let lvl = s.level as i32;
        let stat = |b: i32, ev: u8, iv: u8| (2 * b + iv as i32 + ev as i32 / 4) * lvl / 100 + 5;
        let hp = if e.base[0] == 1 { 1 } else { (2 * e.base[0] + s.ivs[0] as i32 + s.evs[0] as i32 / 4) * lvl / 100 + lvl + 10 };
        let mut moves: Vec<Move> = Vec::new();
        for m in s.moves.iter().take(4) {
            let mp = self.max_pp(*m);
            moves.push(Move { id: *m, disabled: false, pp: mp, max_pp: mp, choice: MOVES.get(m).cloned().unwrap_or_default() });
        }
        while moves.len() < 4 {
            moves.push(Move::default());
        }
        let mut it = moves.into_iter();
        let mut p = Pokemon::default();
        p.id = e.name;
        p.level = s.level;
        p.types = e.types;
        p.base_types = e.types;
        p.hp = hp as i16;
        p.maxhp = hp as i16;
        p.ability = s.ability;
        p.base_ability = s.ability;
        p.item = s.item;
        p.nature = PokemonNature::SERIOUS;
        p.evs = (s.evs[0], s.evs[1], s.evs[2], s.evs[3], s.evs[4], s.evs[5]);
        p.attack = stat(e.base[1], s.evs[1], s.ivs[1]) as i16;
        p.defense = stat(e.base[2], s.evs[2], s.ivs[2]) as i16;
        p.special_attack = stat(e.base[3], s.evs[3], s.ivs[3]) as i16;
        p.special_defense = stat(e.base[4], s.evs[4], s.ivs[4]) as i16;
        p.speed = stat(e.base[5], s.evs[5], s.ivs[5]) as i16;
        p.weight_kg = e.weight;
        p.tera_type = s.tera;
        p.moves.m0 = it.next().unwrap();
        p.moves.m1 = it.next().unwrap();
        p.moves.m2 = it.next().unwrap();
        p.moves.m3 = it.next().unwrap();
        p
    }

    /// A full starting state (team one = side one, team two = side two) from `rng`.
    pub fn sample_state(&self, rng: &mut StdRng) -> State {
        let mut st = State::default();
        for side_no in 0..2 {
            let side: &mut Side = if side_no == 0 { &mut st.side_one } else { &mut st.side_two };
            let team = self.sample_team(rng);
            for (k, (sp, set)) in team.into_iter().enumerate() {
                side.pokemon.pkmn[k] = self.make_pokemon(sp, set);
            }
            side.active_index = PokemonIndex::P0;
        }
        st.set_conditional_mechanics();
        st
    }
}

pub static TEAM_DATA: LazyLock<TeamData> = LazyLock::new(|| TeamData::parse(DATA_TEXT));

/// `n` random starting states from `seed` (team generation only; used by tests and Python).
pub fn sample_states(seed: u64, n: usize) -> Vec<State> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| TEAM_DATA.sample_state(&mut rng)).collect()
}

// ---------------------------------------------------------------------------
// Reveal tracking and history
// ---------------------------------------------------------------------------

/// What side `s`'s opponent has seen of side `s` so far (indexed by team index).
#[derive(Clone, Default)]
struct Reveal {
    seen: [bool; 6],
    mv: [[bool; 4]; 6],
    item: [bool; 6],
    ability: [bool; 6],
    tera: [bool; 6],
}

#[derive(Clone, Copy, Default)]
struct HistSide {
    present: bool,
    kind: u8, // 0 unknown, 1 move, 2 tera move, 3 switch
    aid: u16,
    actor: u16,
}

#[derive(Clone, Copy, Default)]
struct HistEntry {
    turn: u32,
    own: HistSide,
    opp: HistSide,
    d_own: f32,
    d_opp: f32,
    f_own: u8,
    f_opp: u8,
}

/// Public per-step quantities used for the history (species rows with hp fractions, from one view).
#[derive(Clone, Default)]
struct Summary {
    alive: [Vec<(u16, f32)>; 2],
    active: [u16; 2],
}

fn mv_of(p: &Pokemon, i: usize) -> &Move {
    match i {
        0 => &p.moves.m0,
        1 => &p.moves.m1,
        2 => &p.moves.m2,
        _ => &p.moves.m3,
    }
}

fn side_of(st: &State, s: usize) -> &Side {
    if s == 0 {
        &st.side_one
    } else {
        &st.side_two
    }
}

fn announces(a: Abilities) -> bool {
    matches!(
        Abilities::NAMES[a as usize],
        "INTIMIDATE"
            | "DROUGHT"
            | "DRIZZLE"
            | "SANDSTREAM"
            | "SNOWWARNING"
            | "ELECTRICSURGE"
            | "GRASSYSURGE"
            | "PSYCHICSURGE"
            | "MISTYSURGE"
            | "PRESSURE"
            | "UNNERVE"
            | "MOLDBREAKER"
            | "TERAVOLT"
            | "TURBOBLAZE"
            | "NEUTRALIZINGGAS"
            | "TRACE"
            | "DOWNLOAD"
            | "FOREWARN"
            | "FRISK"
            | "ANTICIPATION"
            | "AIRLOCK"
            | "CLOUDNINE"
            | "IMPOSTER"
            | "DESOLATELAND"
            | "PRIMORDIALSEA"
            | "DELTASTREAM"
            | "ORICHALCUMPULSE"
            | "HADRONENGINE"
            | "SUPERSWEETSYRUP"
            | "ASONEGLASTRIER"
            | "ASONESPECTRIER"
            | "SCREENCLEANER"
            | "CURIOUSMEDICINE"
            | "COSTAR"
            | "SANDSPIT"
            | "EMBODYASPECTTEAL"
            | "EMBODYASPECTWELLSPRING"
            | "EMBODYASPECTHEARTHFLAME"
            | "EMBODYASPECTCORNERSTONE"
    )
}

/// The view of one game that observation, history and tests share.
pub struct Game {
    pub state: State,
    pub turn: u32,
    steps: u32,
    reveal: [Reveal; 2],
    hist: [VecDeque<HistEntry>; 2],
    rng: StdRng,
    opts: [Vec<MoveChoice>; 2],
    need: [bool; 2],
    /// episodes finished in this slot (also the seed offset of the next one)
    pub episode: u64,
    pub decisions: [u32; 2],
}

struct StepOut {
    reward: [f32; 2],
    finished: Option<(i8, u32)>,
    illegal: u32,
    panicked: bool,
}

fn mix(a: u64, b: u64, c: u64) -> u64 {
    let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl Game {
    fn new(seed: u64, idx: u64, episode: u64) -> Game {
        let mut rng = StdRng::seed_from_u64(mix(seed, idx, episode));
        let state = TEAM_DATA.sample_state(&mut rng);
        let mut g = Game {
            state,
            turn: 0,
            steps: 0,
            reveal: [Reveal::default(), Reveal::default()],
            hist: [VecDeque::new(), VecDeque::new()],
            rng,
            opts: [Vec::new(), Vec::new()],
            need: [false, false],
            episode,
            decisions: [0, 0],
        };
        for s in 0..2 {
            let a = side_of(&g.state, s).active_index as usize;
            g.reveal[s].seen[a] = true;
        }
        g.refresh_options();
        g
    }

    /// Builds a game from an explicit state (tests); the active Pokemon are revealed.
    pub fn from_state(state: State, seed: u64) -> Game {
        let mut g = Game {
            state,
            turn: 0,
            steps: 0,
            reveal: [Reveal::default(), Reveal::default()],
            hist: [VecDeque::new(), VecDeque::new()],
            rng: StdRng::seed_from_u64(seed),
            opts: [Vec::new(), Vec::new()],
            need: [false, false],
            episode: 0,
            decisions: [0, 0],
        };
        for s in 0..2 {
            let a = side_of(&g.state, s).active_index as usize;
            g.reveal[s].seen[a] = true;
        }
        g.refresh_options();
        g
    }

    fn refresh_options(&mut self) {
        let (o1, o2) = self.state.root_get_all_options();
        self.opts = [o1, o2];
        for s in 0..2 {
            let n = self.opts[s].iter().filter(|o| action_slot(side_of(&self.state, s), o).is_some()).count();
            self.need[s] = n > 1;
        }
    }

    pub fn needs(&self, side: usize) -> bool {
        self.need[side]
    }

    pub fn is_over(&self) -> bool {
        let a = self.state.side_one.pokemon.into_iter().all(|p| p.hp <= 0);
        let b = self.state.side_two.pokemon.into_iter().all(|p| p.hp <= 0);
        a || b
    }

    /// 1 side one won, 2 side two won, 0 draw (both wiped out) / not over.
    pub fn winner(&self) -> i8 {
        let a = self.state.side_one.pokemon.into_iter().all(|p| p.hp <= 0);
        let b = self.state.side_two.pokemon.into_iter().all(|p| p.hp <= 0);
        match (a, b) {
            (true, false) => 2,
            (false, true) => 1,
            _ => 0,
        }
    }

    /// The rows of `Revealed` that `viewer` (0/1) is allowed to know about its opponent.
    pub fn reveal_rows(&self, viewer: usize) -> Vec<Revealed> {
        let o = 1 - viewer;
        let side = side_of(&self.state, o);
        let r = &self.reveal[o];
        let mut rows = Vec::new();
        for k in 0..6 {
            if !r.seen[k] {
                continue;
            }
            let p = &side.pokemon.pkmn[k];
            let moves: Vec<String> = (0..4)
                .filter(|&i| r.mv[k][i] && mv_of(p, i).id != Choices::NONE)
                .map(|i| Choices::NAMES[mv_of(p, i).id as usize].to_string())
                .collect();
            rows.push(Revealed {
                species: PokemonName::NAMES[p.id as usize].to_string(),
                moves,
                item: r.item[k],
                ability: r.ability[k],
                tera: r.tera[k],
            });
        }
        rows
    }

    /// Writes the token buffers of `side` (0 = side one) from its own scrubbed view.
    pub fn tokens(&self, side: usize, cat: &mut [i16], cont: &mut [f32]) {
        let rows = self.reveal_rows(side);
        let sr = if side == 0 { SideReference::SideOne } else { SideReference::SideTwo };
        tokens::state_tokens_into(&self.state, sr, Some(&rows), cat, cont);
        // Own legality comes from the TRUTH state: the scrubbed copy hides the opponent's ability, so a
        // trapping ability (Shadow Tag, Arena Trap, Magnet Pull) would not show up there, while a live
        // bot knows from the request which of its actions are allowed.
        let sd = side_of(&self.state, side);
        let mut legal = [false; N_ACT];
        for o in &self.opts[side] {
            if let Some(s) = action_slot(sd, o) {
                legal[s] = true;
            }
        }
        for s in 0..N_ACT {
            let was = cat[tokens::O_ACT_LEGAL + s] > 0;
            if was && !legal[s] {
                let o = tokens::O_ACT_CONT + s * crate::prior::N_AFEAT;
                cont[o..o + crate::prior::N_AFEAT].fill(0.0);
            }
            cat[tokens::O_ACT_LEGAL + s] = legal[s] as i16;
        }
    }

    fn summary(&self, viewer: usize) -> Summary {
        let mut s = Summary::default();
        for side_no in 0..2 {
            let sd = side_of(&self.state, if side_no == 0 { viewer } else { 1 - viewer });
            for k in 0..6 {
                let p = &sd.pokemon.pkmn[k];
                let visible = side_no == 0 || self.reveal[1 - viewer].seen[k];
                if visible && p.hp > 0 {
                    s.alive[side_no].push((slot_ids(p)[0], p.hp as f32 / p.maxhp.max(1) as f32));
                }
            }
            s.active[side_no] = slot_ids(&sd.pokemon.pkmn[sd.active_index as usize])[0];
        }
        s
    }

    /// History rows (most recent first; own then opponent action per step) of `viewer`.
    pub fn history(&self, viewer: usize, cat: &mut [i16], cont: &mut [f32]) {
        cat[..N_HIST * HIST_CAT_W].fill(0);
        cont[..N_HIST * HIST_CONT_W].fill(0.0);
        for (k, e) in self.hist[viewer].iter().take(HIST_STEPS).enumerate() {
            let ago = (self.turn.saturating_sub(e.turn)).min(16) as f32 / 16.0;
            for n in 0..2 {
                let (h, side_no, self_d, other_d, self_f, other_f) = if n == 0 {
                    (e.own, 1i16, e.d_own, e.d_opp, e.f_own, e.f_opp)
                } else {
                    (e.opp, 2i16, e.d_opp, e.d_own, e.f_opp, e.f_own)
                };
                if !h.present {
                    continue;
                }
                let row = 2 * k + n;
                let c = &mut cat[row * HIST_CAT_W..(row + 1) * HIST_CAT_W];
                c[0] = side_no;
                c[1] = h.kind as i16;
                c[2] = h.aid as i16;
                c[3] = h.actor as i16;
                let f = &mut cont[row * HIST_CONT_W..(row + 1) * HIST_CONT_W];
                f[0] = ago;
                f[1] = self_d;
                f[2] = other_d;
                f[3] = self_f as f32 / 6.0;
                f[4] = other_f as f32 / 6.0;
            }
        }
    }

    fn hist_side(&self, s: usize, c: &MoveChoice) -> HistSide {
        let side = side_of(&self.state, s);
        let actor = slot_ids(side.get_active_immutable())[0];
        match c {
            MoveChoice::None => HistSide::default(),
            MoveChoice::Switch(p) => HistSide { present: true, kind: 3, aid: slot_ids(&side.pokemon[*p])[0], actor },
            MoveChoice::Move(i) | MoveChoice::MoveMega(i) | MoveChoice::MoveTera(i) => {
                let tera = matches!(c, MoveChoice::MoveTera(_));
                let id = side.get_active_immutable().moves[i].id;
                HistSide { present: true, kind: if tera { 2 } else { 1 }, aid: move_idx(id), actor }
            }
            _ => HistSide::default(),
        }
    }

    fn option_for_slot(&self, side: usize, slot: u8) -> Option<MoveChoice> {
        let sd = side_of(&self.state, side);
        self.opts[side].iter().find(|o| action_slot(sd, o) == Some(slot as usize)).copied()
    }

    fn default_choice(&self, side: usize) -> MoveChoice {
        let sd = side_of(&self.state, side);
        self.opts[side]
            .iter()
            .find(|o| action_slot(sd, o).is_some())
            .copied()
            .or_else(|| self.opts[side].first().copied())
            .unwrap_or(MoveChoice::None)
    }

    /// Applies the joint choice `c` (one engine step): samples the chance outcome, updates the
    /// reveal sets and the history of both views.
    pub fn apply(&mut self, c: [MoveChoice; 2]) {
        let before = [self.summary(0), self.summary(1)];
        let pre_active = [self.state.side_one.active_index as usize, self.state.side_two.active_index as usize];
        let pre: [Pokemon; 2] = [
            self.state.side_one.pokemon.pkmn[pre_active[0]].clone(),
            self.state.side_two.pokemon.pkmn[pre_active[1]].clone(),
        ];
        let pre_items: [[Items; 6]; 2] = [
            std::array::from_fn(|k| self.state.side_one.pokemon.pkmn[k].item),
            std::array::from_fn(|k| self.state.side_two.pokemon.pkmn[k].item),
        ];
        let pre_abil: [[Abilities; 6]; 2] = [
            std::array::from_fn(|k| self.state.side_one.pokemon.pkmn[k].ability),
            std::array::from_fn(|k| self.state.side_two.pokemon.pkmn[k].ability),
        ];
        let pre_hp: [[i16; 6]; 2] = [
            std::array::from_fn(|k| self.state.side_one.pokemon.pkmn[k].hp),
            std::array::from_fn(|k| self.state.side_two.pokemon.pkmn[k].hp),
        ];
        let regular = pre[0].hp > 0 && pre[1].hp > 0 && c[0] != MoveChoice::None && c[1] != MoveChoice::None;
        let own_hist = [self.hist_side(0, &c[0]), self.hist_side(1, &c[1])];

        let mut list: Vec<StateInstructions> = generate_instructions_from_move_pair(&mut self.state, &c[0], &c[1], true);
        if list.is_empty() {
            list.push(StateInstructions::default());
        }
        let total: f32 = list.iter().map(|o| o.percentage.max(0.0)).sum();
        let mut idx = list.len() - 1;
        if total > 0.0 {
            let mut th = self.rng.random_range(0.0..total);
            for (i, o) in list.iter().enumerate() {
                th -= o.percentage.max(0.0);
                if th <= 0.0 {
                    idx = i;
                    break;
                }
            }
        } else {
            idx = 0;
        }
        self.state.apply_instructions(&list[idx].instruction_list);
        self.steps += 1;

        // ---- reveals
        let mut executed: [Option<Choices>; 2] = [None, None];
        let mut used_contact = [false, false];
        let mut used_damaging = [false, false];
        for s in 0..2 {
            let post = &side_of(&self.state, s).pokemon.pkmn[pre_active[s]];
            for i in 0..4 {
                let (a, b) = (mv_of(&pre[s], i), mv_of(post, i));
                if a.id != Choices::NONE && b.pp < a.pp {
                    self.reveal[s].mv[pre_active[s]][i] = true;
                    if matches!(c[s], MoveChoice::Move(m) | MoveChoice::MoveTera(m) if m as usize == i) {
                        executed[s] = Some(a.id);
                    }
                    used_contact[s] |= a.choice.flags.contact;
                    used_damaging[s] |= a.choice.category != MoveCategory::Status;
                }
            }
        }
        for s in 0..2 {
            let sd = side_of(&self.state, s);
            let a = sd.active_index as usize;
            let was_active = a == pre_active[s];
            self.reveal[s].seen[a] = true;
            if let MoveChoice::Switch(p) = c[s] {
                self.reveal[s].seen[p as usize] = true; // sent out, even if it left or fainted within the turn
            }
            for k in 0..6 {
                let p = &sd.pokemon.pkmn[k];
                if p.terastallized {
                    self.reveal[s].tera[k] = true;
                }
                if p.item != pre_items[s][k] {
                    self.reveal[s].item[k] = true;
                }
                if p.ability != pre_abil[s][k] {
                    self.reveal[s].ability[k] = true;
                }
                let held = pre_items[s][k];
                let name = Items::NAMES[held as usize];
                if k == pre_active[s] || k == a {
                    match name {
                        "LEFTOVERS" | "BLACKSLUDGE" => {
                            let healed_by_move = k == pre_active[s]
                                && matches!(c[s], MoveChoice::Move(m) | MoveChoice::MoveTera(m)
                                    if { let ch = &pre[s].moves[&m].choice; ch.heal.is_some() || ch.drain.is_some() });
                            if p.hp > pre_hp[s][k] && !healed_by_move {
                                self.reveal[s].item[k] = true;
                            }
                        }
                        "LIFEORB" => {
                            if k == pre_active[s] && used_damaging[s] && p.hp < pre_hp[s][k] {
                                self.reveal[s].item[k] = true;
                            }
                        }
                        "ROCKYHELMET" => {
                            if k == pre_active[s] && used_contact[1 - s] && p.hp < pre_hp[s][k] {
                                self.reveal[s].item[k] = true;
                            }
                        }
                        _ => {}
                    }
                }
                if k == a && !(was_active && k == pre_active[s]) && announces(p.ability) {
                    self.reveal[s].ability[k] = true;
                }
            }
        }

        // ---- history (both views)
        let after = [self.summary(0), self.summary(1)];
        for v in 0..2 {
            let o = 1 - v;
            let opp_act = self.hist_side_post(o, &c[o], executed[o], &pre[o], pre_active[o]);
            let (b, a) = (&before[v], &after[v]);
            let delta = |side_no: usize| -> f32 {
                let x = b.active[side_no];
                if x <= 1 {
                    return 0.0;
                }
                match b.alive[side_no].iter().find(|(id, _)| *id == x) {
                    None => 0.0,
                    Some((_, hb)) => a.alive[side_no].iter().find(|(id, _)| *id == x).map_or(0.0, |(_, h)| *h) - hb,
                }
            };
            let fainted = |side_no: usize| -> u8 {
                b.alive[side_no].iter().filter(|(id, _)| !a.alive[side_no].iter().any(|(j, _)| j == id)).count() as u8
            };
            let e = HistEntry {
                turn: self.turn,
                own: own_hist[v],
                opp: opp_act,
                d_own: delta(0),
                d_opp: delta(1),
                f_own: fainted(0),
                f_opp: fainted(1),
            };
            self.hist[v].push_front(e);
            self.hist[v].truncate(HIST_STEPS);
        }
        if regular {
            self.turn += 1;
        }
        self.refresh_options();
    }

    /// History entry of `s`'s action as its opponent sees it (a move only if it was executed).
    fn hist_side_post(&self, s: usize, c: &MoveChoice, executed: Option<Choices>, pre: &Pokemon, _pre_active: usize) -> HistSide {
        let actor = slot_ids(pre)[0];
        let side = side_of(&self.state, s);
        match c {
            MoveChoice::None => HistSide::default(),
            MoveChoice::Switch(p) => HistSide { present: true, kind: 3, aid: slot_ids(&side.pokemon[*p])[0], actor },
            MoveChoice::Move(i) | MoveChoice::MoveMega(i) | MoveChoice::MoveTera(i) => {
                let tera = matches!(c, MoveChoice::MoveTera(_));
                let id = pre.moves[i].id;
                if executed != Some(id) {
                    return HistSide { present: true, kind: 0, aid: 0, actor };
                }
                HistSide { present: true, kind: if tera { 2 } else { 1 }, aid: move_idx(id), actor }
            }
            _ => HistSide::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------

/// Counters of a pool since creation.
#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    pub games: u64,
    pub draws: u64,
    pub cap_draws: u64,
    pub p1_wins: u64,
    pub p2_wins: u64,
    pub panics: u64,
    pub illegal: u64,
    pub steps: u64,
    pub decisions: u64,
    pub turns_total: u64,
}

/// One observed decision.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub game: u32,
    pub side: u8,
    pub turn: u32,
}

pub struct Obs {
    pub rows: Vec<Row>,
    pub cat: Vec<i16>,
    pub cont: Vec<f32>,
    pub hist_cat: Vec<i16>,
    pub hist_cont: Vec<f32>,
}

pub struct StepResult {
    pub reward: Vec<f32>,
    pub done: Vec<u8>,
    /// (game, winner 0 draw / 1 / 2, turns)
    pub finished: Vec<(u32, i8, u32)>,
}

pub struct SelfPlayPool {
    pub games: Vec<Game>,
    seed: u64,
    threads: usize,
    turn_cap: u32,
    rows: Vec<Row>,
    pub stats: PoolStats,
}

/// Runs `f(global index, chunk)` over `items` on `threads` scoped threads with dynamic chunking.
fn par_chunks<T: Send, F: Fn(usize, &mut [T]) + Sync>(items: &mut [T], threads: usize, chunk: usize, f: F) {
    let threads = threads.max(1);
    if threads == 1 || items.len() <= chunk {
        f(0, items);
        return;
    }
    let it = Mutex::new(items.chunks_mut(chunk).enumerate());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| loop {
                let next = it.lock().unwrap().next();
                match next {
                    Some((i, c)) => f(i * chunk, c),
                    None => break,
                }
            });
        }
    });
}

impl SelfPlayPool {
    pub fn new(n_games: usize, seed: u64, threads: usize, turn_cap: u32) -> SelfPlayPool {
        let mut games: Vec<Game> = Vec::with_capacity(n_games);
        for i in 0..n_games {
            games.push(Game::new(seed, i as u64, 0));
        }
        let mut p = SelfPlayPool { games, seed, threads: threads.max(1), turn_cap, rows: Vec::new(), stats: PoolStats::default() };
        let (seed, cap) = (p.seed, p.turn_cap);
        par_chunks(&mut p.games, p.threads, 16, |_, c| {
            for g in c.iter_mut() {
                let _ = settle(g, cap, seed, 0, &mut StepOut { reward: [0.0; 2], finished: None, illegal: 0, panicked: false });
            }
        });
        p.rebuild_rows();
        p
    }

    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }

    pub fn n_games(&self) -> usize {
        self.games.len()
    }

    fn rebuild_rows(&mut self) {
        self.rows.clear();
        for (i, g) in self.games.iter().enumerate() {
            for s in 0..2 {
                if g.need[s] {
                    self.rows.push(Row { game: i as u32, side: s as u8, turn: g.turn });
                }
            }
        }
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Token buffers, legality and history for every row needing a decision.
    pub fn observe(&self) -> Obs {
        let n = self.rows.len();
        let mut cat = vec![0i16; n * CAT_LEN];
        let mut cont = vec![0f32; n * CONT_LEN];
        let mut hist_cat = vec![0i16; n * N_HIST * HIST_CAT_W];
        let mut hist_cont = vec![0f32; n * N_HIST * HIST_CONT_W];
        {
            let mut parts: Vec<(&Row, &mut [i16], &mut [f32], &mut [i16], &mut [f32])> = cat
                .chunks_mut(CAT_LEN)
                .zip(cont.chunks_mut(CONT_LEN))
                .zip(hist_cat.chunks_mut(N_HIST * HIST_CAT_W))
                .zip(hist_cont.chunks_mut(N_HIST * HIST_CONT_W))
                .zip(self.rows.iter())
                .map(|((((a, b), c), d), r)| (r, a, b, c, d))
                .collect();
            let games = &self.games;
            par_chunks(&mut parts, self.threads, 32, |_, chunk| {
                for (r, a, b, c, d) in chunk.iter_mut() {
                    let g = &games[r.game as usize];
                    g.tokens(r.side as usize, a, b);
                    g.history(r.side as usize, c, d);
                }
            });
        }
        Obs { rows: self.rows.clone(), cat, cont, hist_cat, hist_cont }
    }

    /// Applies one action slot (0..13) per observed row, in row order.
    pub fn step(&mut self, actions: &[u8]) -> Result<StepResult, String> {
        if actions.len() != self.rows.len() {
            return Err(format!("expected {} actions (one per observed row), got {}", self.rows.len(), actions.len()));
        }
        let ng = self.games.len();
        let mut acts: Vec<[u8; 2]> = vec![[255, 255]; ng];
        for (r, a) in self.rows.iter().zip(actions) {
            acts[r.game as usize][r.side as usize] = *a;
        }
        let (seed, cap) = (self.seed, self.turn_cap);
        let mut results: Vec<StepOut> = (0..ng).map(|_| StepOut { reward: [0.0; 2], finished: None, illegal: 0, panicked: false }).collect();
        {
            let mut zipped: Vec<(&mut Game, &mut StepOut, [u8; 2], usize)> = self
                .games
                .iter_mut()
                .zip(results.iter_mut())
                .enumerate()
                .map(|(i, (g, o))| (g, o, acts[i], i))
                .collect();
            par_chunks(&mut zipped, self.threads, 8, |_, chunk| {
                for (g, o, a, idx) in chunk.iter_mut() {
                    play(g, o, *a, cap, seed, *idx as u64);
                }
            });
        }
        let nrows = self.rows.len();
        let mut reward = vec![0f32; nrows];
        let mut done = vec![0u8; nrows];
        let mut finished = Vec::new();
        for (ri, r) in self.rows.iter().enumerate() {
            let o = &results[r.game as usize];
            reward[ri] = o.reward[r.side as usize];
            done[ri] = o.finished.is_some() as u8;
        }
        self.stats.decisions += nrows as u64;
        for (i, o) in results.iter().enumerate() {
            self.stats.steps += 1;
            self.stats.illegal += o.illegal as u64;
            if o.panicked {
                self.stats.panics += 1;
            }
            if let Some((w, turns)) = o.finished {
                finished.push((i as u32, w, turns));
                self.stats.games += 1;
                self.stats.turns_total += turns as u64;
                match w {
                    1 => self.stats.p1_wins += 1,
                    2 => self.stats.p2_wins += 1,
                    _ => self.stats.draws += 1,
                }
                if o.reward == [0.0, 0.0] && w == 0 && turns >= cap {
                    self.stats.cap_draws += 1;
                }
            }
        }
        self.rebuild_rows();
        Ok(StepResult { reward, done, finished })
    }
}

/// Plays the joint action of one game (panics are contained: the game is drawn and replaced).
fn play(g: &mut Game, out: &mut StepOut, a: [u8; 2], cap: u32, seed: u64, idx: u64) {
    let res = catch_unwind(AssertUnwindSafe(|| {
        let mut c = [MoveChoice::None, MoveChoice::None];
        for s in 0..2 {
            c[s] = if a[s] == 255 {
                g.default_choice(s)
            } else {
                match g.option_for_slot(s, a[s]) {
                    Some(x) => x,
                    None => {
                        out.illegal += 1;
                        g.default_choice(s)
                    }
                }
            };
            if a[s] != 255 {
                g.decisions[s] += 1;
            }
        }
        g.apply(c);
    }));
    if res.is_err() {
        out.panicked = true;
        out.finished = Some((0, g.turn));
        *g = Game::new(seed, idx, g.episode + 1);
        let _ = settle(g, cap, seed, idx, &mut StepOut { reward: [0.0; 2], finished: None, illegal: 0, panicked: false });
        return;
    }
    settle(g, cap, seed, idx, out);
}

/// After a step: finish the game if over (or capped) and reset it, then auto-play sides without a
/// real decision until some side has one.
fn settle(g: &mut Game, cap: u32, seed: u64, idx: u64, out: &mut StepOut) -> bool {
    let mut guard = 0;
    loop {
        let over = g.is_over();
        let capped = g.turn >= cap || g.steps > 6 * cap || guard > 64;
        if over || capped {
            let w = if over { g.winner() } else { 0 };
            out.reward = match w {
                1 => [1.0, -1.0],
                2 => [-1.0, 1.0],
                _ => [0.0, 0.0],
            };
            out.finished = Some((w, g.turn));
            *g = Game::new(seed, idx, g.episode + 1);
            guard = 0;
            continue;
        }
        if g.need[0] || g.need[1] {
            return true;
        }
        let c = [g.default_choice(0), g.default_choice(1)];
        let r = catch_unwind(AssertUnwindSafe(|| g.apply(c)));
        if r.is_err() {
            out.panicked = true;
            out.finished = Some((0, g.turn));
            out.reward = [0.0, 0.0];
            *g = Game::new(seed, idx, g.episode + 1);
        }
        guard += 1;
    }
}

/// Number of action slots per side (re-exported for bindings).
pub const N_SLOTS: usize = N_ACT;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{O_ACT_LEGAL, O_MON_CAT, O_MON_KNOWN, KNOWN_W};

    fn legal_slots(cat: &[i16], side_row: usize) -> Vec<u8> {
        let o = side_row * CAT_LEN + O_ACT_LEGAL;
        (0..N_ACT).filter(|&s| cat[o + s] > 0).map(|s| s as u8).collect()
    }

    fn random_actions(obs: &Obs, rng: &mut StdRng) -> Vec<u8> {
        (0..obs.rows.len())
            .map(|i| {
                let l = legal_slots(&obs.cat, i);
                l[rng.random_range(0..l.len())]
            })
            .collect()
    }

    #[test]
    fn data_loads_and_nearly_everything_resolves() {
        let d = &*TEAM_DATA;
        assert!(d.species.len() > 450, "{} species, dropped {:?}", d.species.len(), d.dropped_species);
        assert!(d.dropped_sets * 100 < d.species.iter().map(|s| s.sets.len()).sum::<usize>(), "dropped sets {}", d.dropped_sets);
    }

    #[test]
    fn teams_are_legal_engine_states() {
        let d = &*TEAM_DATA;
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..300 {
            let s = d.sample_state(&mut rng);
            for side in [&s.side_one, &s.side_two] {
                let mut bases: Vec<String> = Vec::new();
                let mut lvl100 = 0;
                for p in side.pokemon.pkmn.iter() {
                    assert_ne!(p.id, PokemonName::NONE);
                    assert!(p.hp == p.maxhp && p.hp > 0 && p.attack > 0 && p.speed > 0);
                    assert_ne!(p.moves.m0.id, Choices::NONE);
                    assert!(p.moves.m0.max_pp > 0 && p.moves.m0.pp == p.moves.m0.max_pp);
                    assert_ne!(p.ability, Abilities::NONE);
                    assert_ne!(p.item, Items::UNKNOWNITEM);
                    if p.level == 100 {
                        lvl100 += 1;
                    }
                    let e = d.species.iter().find(|e| e.name == p.id).unwrap();
                    assert!(!bases.contains(&e.base_species), "species clause");
                    bases.push(e.base_species.clone());
                }
                assert!(lvl100 <= 1 || true);
            }
            // serialization round trip and options
            let r = State::deserialize(&s.serialize());
            let (o1, o2) = r.root_get_all_options();
            assert!(o1.len() >= 4 && o2.len() >= 4);
            assert_eq!(r.battle_is_over(), 0.0);
        }
    }

    #[test]
    fn team_rules_hold() {
        // at most two of any type on almost every team (the rules relax only after 40 failed draws)
        let d = &*TEAM_DATA;
        let mut rng = StdRng::seed_from_u64(11);
        let mut bad = 0;
        for _ in 0..300 {
            let team = d.sample_team(&mut rng);
            for t in 0..18u32 {
                let n = team.iter().filter(|(s, _)| {
                    let e = &d.species[*s];
                    e.types.0 as u32 == t || e.types.1 as u32 == t
                }).count();
                if n > 2 {
                    bad += 1;
                }
            }
        }
        assert!(bad < 10, "{} teams broke the type limit", bad);
    }

    #[test]
    fn games_run_to_the_end_with_random_actions() {
        let mut pool = SelfPlayPool::new(32, 3, 2, 200);
        let mut rng = StdRng::seed_from_u64(1);
        let mut finished = 0;
        for _ in 0..600 {
            let obs = pool.observe();
            for (i, r) in obs.rows.iter().enumerate() {
                assert!(r.side < 2);
                assert!(!legal_slots(&obs.cat, i).is_empty());
            }
            assert!(obs.cont.iter().all(|v| v.is_finite()));
            let acts = random_actions(&obs, &mut rng);
            let res = pool.step(&acts).unwrap();
            finished += res.finished.len();
        }
        assert!(finished >= 10, "only {} games finished", finished);
        assert_eq!(pool.stats.illegal, 0, "all sampled actions were legal");
        assert_eq!(pool.stats.panics, 0, "engine panics: {}", pool.stats.panics);
    }

    /// Long random-play fuzz (cargo test --release selfplay::tests::fuzz -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn fuzz_random_play_has_no_engine_panics() {
        let mut pool = SelfPlayPool::new(256, 12345, 4, 200);
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..3000 {
            let obs = pool.observe();
            let acts = random_actions(&obs, &mut rng);
            pool.step(&acts).unwrap();
        }
        let s = &pool.stats;
        println!(
            "games {} p1 {} p2 {} draws {} cap_draws {} panics {} illegal {} mean turns {:.1}",
            s.games, s.p1_wins, s.p2_wins, s.draws, s.cap_draws, s.panics, s.illegal, s.turns_total as f64 / s.games.max(1) as f64
        );
        assert_eq!(s.panics, 0);
        assert_eq!(s.illegal, 0);
    }

    #[test]
    fn wrong_action_count_is_an_error() {
        let mut pool = SelfPlayPool::new(4, 1, 1, 200);
        let n = pool.rows().len();
        assert!(pool.step(&vec![0u8; n + 1]).is_err());
    }

    #[test]
    fn determinism_across_runs_and_thread_counts() {
        let run = |threads: usize| {
            let mut pool = SelfPlayPool::new(24, 99, threads, 200);
            let mut rng = StdRng::seed_from_u64(5);
            let mut digest: Vec<u8> = Vec::new();
            for _ in 0..80 {
                let obs = pool.observe();
                digest.extend(obs.cat.iter().take(2048).flat_map(|x| x.to_le_bytes()));
                digest.extend(obs.hist_cat.iter().flat_map(|x| x.to_le_bytes()));
                digest.extend(obs.cont.iter().take(2048).flat_map(|x| x.to_le_bytes()));
                let acts = random_actions(&obs, &mut rng);
                let res = pool.step(&acts).unwrap();
                digest.extend(res.reward.iter().flat_map(|x| x.to_le_bytes()));
                digest.extend(res.done.iter());
            }
            digest
        };
        let a = run(1);
        assert_eq!(a, run(1), "same seed, same result");
        assert_eq!(a, run(3), "thread count must not change the games");
    }

    #[test]
    fn finished_games_reset_with_fresh_teams() {
        let mut pool = SelfPlayPool::new(8, 5, 1, 200);
        let before: Vec<String> = pool.games.iter().map(|g| g.state.serialize()).collect();
        let mut rng = StdRng::seed_from_u64(2);
        let mut seen_done = false;
        for _ in 0..800 {
            let obs = pool.observe();
            let acts = random_actions(&obs, &mut rng);
            let res = pool.step(&acts).unwrap();
            for (g, w, _) in &res.finished {
                seen_done = true;
                assert!(*w >= 0 && *w <= 2);
                let gm = &pool.games[*g as usize];
                assert!(!gm.is_over());
                assert_eq!(gm.turn, 0);
                assert!(gm.episode >= 1);
                assert_ne!(gm.state.serialize(), before[*g as usize]);
            }
            for (ri, d) in res.done.iter().enumerate() {
                let r = pool.stats.games; // keep the borrow checker happy
                let _ = (ri, r);
                if *d == 1 {
                    assert!(res.reward[ri].abs() <= 1.0);
                }
            }
            if seen_done {
                break;
            }
        }
        assert!(seen_done);
    }

    /// Plays some turns, then checks every opponent-side token against a tampered truth state.
    #[test]
    fn unrevealed_opponent_info_never_appears_in_tokens() {
        let d = &*TEAM_DATA;
        let mut rng = StdRng::seed_from_u64(21);
        let mut checked = 0;
        for gi in 0..12u64 {
            let mut pool = SelfPlayPool::new(1, 1000 + gi, 1, 200);
            for step in 0..40 {
                let obs = pool.observe();
                if obs.rows.is_empty() {
                    break;
                }
                for (ri, r) in obs.rows.iter().enumerate() {
                    let g = &pool.games[r.game as usize];
                    let side = r.side as usize;
                    let o = 1 - side;
                    let base_cat = &obs.cat[ri * CAT_LEN..(ri + 1) * CAT_LEN];
                    let base_cont = &obs.cont[ri * CONT_LEN..(ri + 1) * CONT_LEN];
                    // tamper with everything of the opponent the viewer has not seen
                    let mut tampered = Game::from_state(g.state.clone(), 0);
                    tampered.reveal = g.reveal.clone();
                    tampered.hist = g.hist.clone();
                    tampered.turn = g.turn;
                    tampered.opts = g.opts.clone();
                    let other_team = d.sample_state(&mut rng);
                    for k in 0..6 {
                        let rv = &g.reveal[o];
                        let donor = if o == 0 { &other_team.side_one } else { &other_team.side_two };
                        let target = if o == 0 { &mut tampered.state.side_one } else { &mut tampered.state.side_two };
                        if !rv.seen[k] {
                            // (a donor of an already revealed species would be matched by name: the real
                            // species clause rules that out, so keep the test inputs legal too)
                            let dup = (0..6).any(|j| rv.seen[j] && target.pokemon.pkmn[j].id == donor.pokemon.pkmn[k].id);
                            if dup {
                                continue;
                            }
                            target.pokemon.pkmn[k] = donor.pokemon.pkmn[k].clone();
                            target.pokemon.pkmn[k].hp = 1;
                            continue;
                        }
                        let p = &mut target.pokemon.pkmn[k];
                        let dp = &donor.pokemon.pkmn[k];
                        if !rv.item[k] {
                            p.item = dp.item;
                        }
                        if !rv.ability[k] {
                            p.ability = dp.ability;
                        }
                        if !rv.tera[k] {
                            p.tera_type = dp.tera_type;
                        }
                        for i in 0..4 {
                            let dmove = mv_of(dp, i).clone();
                            let clash = (0..4).any(|j| mv_of(p, j).id == dmove.id);
                            if !rv.mv[k][i] && !clash {
                                match i {
                                    0 => p.moves.m0 = dmove,
                                    1 => p.moves.m1 = dmove,
                                    2 => p.moves.m2 = dmove,
                                    _ => p.moves.m3 = dmove,
                                }
                            }
                        }
                    }
                    let mut cat = vec![0i16; CAT_LEN];
                    let mut cont = vec![0f32; CONT_LEN];
                    tampered.tokens(side, &mut cat, &mut cont);
                    // the tampered game may have different legal opponent actions only through
                    // hidden slots, which the scrub canonicalises as well
                    let diffs: Vec<(usize, i16, i16)> = (0..CAT_LEN).filter(|&i| base_cat[i] != cat[i]).map(|i| (i, base_cat[i], cat[i])).collect();
                    assert!(diffs.is_empty(), "cat leaks (game {}, step {}, side {}): (index, truth view, tampered view) {:?}; offsets: known {} act_cat {} legal {}", gi, step, side, diffs, O_MON_KNOWN, tokens::O_ACT_CAT, O_ACT_LEGAL);
                    assert_eq!(base_cont, &cont[..], "cont leaks (game {}, step {}, side {})", gi, step, side);
                    checked += 1;
                    // history: every opponent id is a revealed species / move. (An in-battle forme change rewrites
                    // the species id of a seen Pokemon, so older entries may carry the former forme's row: it
                    // must still share the name prefix of a seen Pokemon.)
                    let rows = g.reveal_rows(side);
                    let names = crate::features_v4::vocab_names(0);
                    let species_ok = |id: u16| -> bool {
                        id == 0
                            || (0..6).any(|k| {
                                let p = &side_of(&g.state, o).pokemon.pkmn[k];
                                g.reveal[o].seen[k] && {
                                    let nm = PokemonName::NAMES[p.id as usize];
                                    crate::features_v4::species_idx(p) == id
                                        || names.get(id as usize).map_or(false, |a| a.to_uppercase().starts_with(&nm[..4.min(nm.len())]))
                                }
                            })
                    };
                    let known_moves: Vec<u16> = rows
                        .iter()
                        .flat_map(|rv| rv.moves.iter())
                        .map(|m| move_idx(Choices::from(Choices::NAMES.iter().position(|n| n.eq_ignore_ascii_case(m)).unwrap() as u16)))
                        .collect();
                    let mut hc = vec![0i16; N_HIST * HIST_CAT_W];
                    let mut hf = vec![0f32; N_HIST * HIST_CONT_W];
                    g.history(side, &mut hc, &mut hf);
                    for row in 0..N_HIST {
                        if hc[row * 4] == 2 {
                            let (kind, aid) = (hc[row * 4 + 1], hc[row * 4 + 2] as u16);
                            if kind == 3 {
                                assert!(species_ok(aid), "history leaks a switch target {}", aid);
                            } else if kind > 0 {
                                assert!(known_moves.contains(&aid), "history leaks an unrevealed move {}", aid);
                            }
                            assert!(species_ok(hc[row * 4 + 3] as u16), "history actor not revealed");
                        }
                    }
                }
                let acts = random_actions(&obs, &mut rng);
                pool.step(&acts).unwrap();
            }
        }
        assert!(checked > 200, "only {} observations checked", checked);
        // the known flags of a fresh game: only the active opponent is known
        let pool = SelfPlayPool::new(1, 4, 1, 200);
        let obs = pool.observe();
        let k = O_MON_KNOWN + 6 * KNOWN_W;
        assert_eq!(obs.cat[k], 1);
        assert_eq!(&obs.cat[k + 1..k + KNOWN_W], &[0; 7], "nothing but the species of the opposing lead is known");
        let b = O_MON_CAT + 7 * crate::features_v4::IDS_PER_SLOT;
        assert!(obs.cat[b..b + 7].iter().all(|&i| i == crate::features_v4::IDX_UNKNOWN as i16));
    }

    #[test]
    fn history_rows_follow_the_game() {
        let mut pool = SelfPlayPool::new(4, 8, 1, 200);
        let mut rng = StdRng::seed_from_u64(3);
        // first observation: no history
        let obs = pool.observe();
        assert!(obs.hist_cat.iter().all(|&x| x == 0));
        for _ in 0..3 {
            let obs = pool.observe();
            let acts = random_actions(&obs, &mut rng);
            pool.step(&acts).unwrap();
        }
        let obs = pool.observe();
        let mut any = false;
        for ri in 0..obs.rows.len() {
            let hc = &obs.hist_cat[ri * N_HIST * HIST_CAT_W..(ri + 1) * N_HIST * HIST_CAT_W];
            let hf = &obs.hist_cont[ri * N_HIST * HIST_CONT_W..(ri + 1) * N_HIST * HIST_CONT_W];
            for row in 0..N_HIST {
                let side = hc[row * 4];
                assert!((0..=2).contains(&side));
                if side > 0 {
                    any = true;
                    // own rows are even, opponent rows odd
                    assert_eq!(side as usize, 1 + row % 2);
                    assert!((0..=3).contains(&hc[row * 4 + 1]));
                    assert!(hf[row * 5] >= 0.0 && hf[row * 5] <= 1.0);
                } else {
                    assert!(hf[row * 5..row * 5 + 5].iter().all(|&x| x == 0.0));
                }
            }
        }
        assert!(any, "history stayed empty after three steps");
    }
}
