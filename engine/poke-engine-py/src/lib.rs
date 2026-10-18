use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyType};
use pyo3::{pyfunction, pymethods, pymodule, wrap_pyfunction, Bound, PyResult};
use std::collections::HashSet;

use poke_engine::choices::{Choices, MoveCategory, MOVES};
use poke_engine::engine::abilities::Abilities;
use poke_engine::engine::generate_instructions::{
    calculate_both_damage_rolls, generate_instructions_from_move_pair,
};
use poke_engine::engine::items::Items;
use poke_engine::engine::state::{MoveChoice, PokemonVolatileStatus, Terrain, Weather};
use poke_engine::instruction::{Instruction, StateInstructions};
use poke_engine::features::{
    feature_len, feature_names_versioned, features_versioned, FEATURE_LEN, FEATURE_LEN_V2,
    FEATURE_LEN_V3, FEATURE_LEN_V4, FEATURE_LEN_V5, FEATURE_VERSION,
};
use poke_engine::features_v4;
use poke_engine::tokens;
use poke_engine::netsearch;
use poke_engine::learned::{Evaluator, Scratch};
use poke_engine::mcts::{perform_mcts_export, ExportConfig, MctsResult, MctsSideResult};
use poke_engine::prior::{
    action_features, action_slot, PolicyNet, PolicyWeights, PriorConfig, N_ACTIONS, N_AFEAT, N_EMB,
};
use poke_engine::selection::{SelectionConfig, SelectionMode};
use poke_engine::mcts_threaded::perform_mcts_shared_tree_with;
use poke_engine::pokemon::PokemonName;
use poke_engine::search::iterative_deepen_expectiminimax;
use poke_engine::state::{
    LastUsedMove, Move, Pokemon, PokemonIndex, PokemonMoves, PokemonNature, PokemonStatus,
    PokemonType, Side, SideConditions, SidePokemon, SideReference, State, StateTerrain, StateTrickRoom,
    StateWeather, VolatileStatusBitset, VolatileStatusDurations,
};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

fn movechoice_to_string(side: &Side, move_choice: &MoveChoice) -> String {
    match move_choice {
        MoveChoice::Switch(_) => {
            format!("switch {}", move_choice.to_string(side))
        }
        _ => move_choice.to_string(side),
    }
}

#[derive(Clone)]
#[pyclass(name = "State", module = "poke_engine", get_all, from_py_object)]
pub struct PyState {
    pub side_one: PySide,
    pub side_two: PySide,
    pub weather: String,
    pub weather_turns_remaining: i8,
    pub terrain: String,
    pub terrain_turns_remaining: i8,
    pub trick_room: bool,
    pub trick_room_turns_remaining: i8,
    pub team_preview: bool,
}

impl From<State> for PyState {
    fn from(other: State) -> Self {
        PyState {
            side_one: PySide::from(other.side_one),
            side_two: PySide::from(other.side_two),
            weather: other.weather.weather_type.to_string(),
            weather_turns_remaining: other.weather.turns_remaining,
            terrain: other.terrain.terrain_type.to_string(),
            terrain_turns_remaining: other.terrain.turns_remaining,
            trick_room: other.trick_room.active,
            trick_room_turns_remaining: other.trick_room.turns_remaining,
            team_preview: other.team_preview,
        }
    }
}

impl Into<State> for PyState {
    fn into(self) -> State {
        let mut state = State {
            side_one: self.side_one.into(),
            side_two: self.side_two.into(),
            weather: StateWeather {
                weather_type: Weather::from_str(&self.weather).unwrap(),
                turns_remaining: self.weather_turns_remaining,
            },
            terrain: StateTerrain {
                terrain_type: Terrain::from_str(&self.terrain).unwrap(),
                turns_remaining: self.terrain_turns_remaining,
            },
            trick_room: StateTrickRoom {
                active: self.trick_room,
                turns_remaining: self.trick_room_turns_remaining,
            },
            team_preview: self.team_preview,
            use_last_used_move: false,
            use_damage_dealt: false,
        };
        state.set_conditional_mechanics();
        state
    }
}

#[pymethods]
impl PyState {
    #[new]
    #[pyo3(signature = (
        side_one=PySide::from(Side::default()),
        side_two=PySide::from(Side::default()),
        weather="none".to_string(),
        weather_turns_remaining=0,
        terrain="none".to_string(),
        terrain_turns_remaining=0,
        trick_room=false,
        trick_room_turns_remaining=0,
        team_preview=false,
    ))]
    fn new(
        side_one: PySide,
        side_two: PySide,
        weather: String,
        weather_turns_remaining: i8,
        terrain: String,
        terrain_turns_remaining: i8,
        trick_room: bool,
        trick_room_turns_remaining: i8,
        team_preview: bool,
    ) -> Self {
        PyState {
            side_one,
            side_two,
            weather,
            weather_turns_remaining,
            terrain,
            terrain_turns_remaining,
            trick_room,
            trick_room_turns_remaining,
            team_preview,
        }
    }
    fn apply_instructions(&self, instructions: PyStateInstructions) -> PyState {
        let mut state: State = self.clone().into();
        let instructions: StateInstructions = instructions.into();
        state.apply_instructions(&instructions.instruction_list);
        state.into()
    }
    fn reverse_instructions(&self, instructions: PyStateInstructions) -> PyState {
        let mut state: State = self.clone().into();
        let instructions: StateInstructions = instructions.into();
        state.reverse_instructions(&instructions.instruction_list);
        state.into()
    }
    #[classmethod]
    fn from_string(_cls: &Bound<'_, PyType>, state_str: String) -> PyResult<Self> {
        let state: State = State::deserialize(&state_str);
        Ok(PyState::from(state))
    }
    fn to_string(&self) -> String {
        let state: State = self.clone().into();
        state.serialize()
    }
}

#[derive(Clone)]
#[pyclass(name = "Side", module = "poke_engine", get_all, from_py_object)]
pub struct PySide {
    pokemon: [PyPokemon; 6],
    side_conditions: PySideConditions,

    active_index: String,
    baton_passing: bool,
    shed_tailing: bool,
    volatile_status_durations: PyVolatileStatusDurations,
    wish: (i8, i16),
    future_sight: (i8, String),
    force_switch: bool,
    force_trapped: bool,
    slow_uturn_move: bool,
    volatile_statuses: HashSet<String>,
    substitute_health: i16,
    attack_boost: i8,
    defense_boost: i8,
    special_attack_boost: i8,
    special_defense_boost: i8,
    speed_boost: i8,
    accuracy_boost: i8,
    evasion_boost: i8,
    last_used_move: String,
    switch_out_move_second_saved_move: String,
}

impl From<Side> for PySide {
    fn from(other: Side) -> Self {
        let pokemon = other.pokemon.pkmn.map(|pokemon| pokemon.into());
        let mut volatile_statuses = HashSet::new();
        let mut remaining = other.volatile_statuses.0;
        while remaining != 0 {
            let bit_index = remaining.trailing_zeros() as u8;
            let vs = PokemonVolatileStatus::from(bit_index);
            volatile_statuses.insert(vs.to_string());
            remaining &= remaining - 1;
        }
        PySide {
            pokemon,
            side_conditions: PySideConditions::from(other.side_conditions),
            active_index: other.active_index.serialize(),
            baton_passing: other.baton_passing,
            shed_tailing: other.shed_tailing,
            volatile_status_durations: PyVolatileStatusDurations::from(
                other.volatile_status_durations,
            ),
            wish: other.wish,
            future_sight: (other.future_sight.0, other.future_sight.1.serialize()),
            force_switch: other.force_switch,
            force_trapped: other.force_trapped,
            slow_uturn_move: other.slow_uturn_move,
            volatile_statuses,
            substitute_health: other.substitute_health,
            attack_boost: other.attack_boost,
            defense_boost: other.defense_boost,
            special_attack_boost: other.special_attack_boost,
            special_defense_boost: other.special_defense_boost,
            speed_boost: other.speed_boost,
            accuracy_boost: other.accuracy_boost,
            evasion_boost: other.evasion_boost,
            last_used_move: other.last_used_move.serialize(),
            switch_out_move_second_saved_move: other.switch_out_move_second_saved_move.to_string(),
        }
    }
}

impl Into<Side> for PySide {
    fn into(self) -> Side {
        let mut volatile_statuses = VolatileStatusBitset::default();
        for s in &self.volatile_statuses {
            if let Ok(vs) = PokemonVolatileStatus::from_str(s) {
                volatile_statuses.insert(vs);
            }
        }
        Side {
            active_index: PokemonIndex::deserialize(&self.active_index),
            baton_passing: self.baton_passing,
            shed_tailing: self.shed_tailing,
            pokemon: SidePokemon {
                pkmn: [
                    self.pokemon[0].clone().into(),
                    self.pokemon[1].clone().into(),
                    self.pokemon[2].clone().into(),
                    self.pokemon[3].clone().into(),
                    self.pokemon[4].clone().into(),
                    self.pokemon[5].clone().into(),
                ],
            },
            side_conditions: self.side_conditions.into(),
            volatile_status_durations: VolatileStatusDurations::from(
                self.volatile_status_durations.into(),
            ),
            wish: self.wish,
            future_sight: (
                self.future_sight.0,
                PokemonIndex::deserialize(&self.future_sight.1),
            ),
            force_switch: self.force_switch,
            force_trapped: self.force_trapped,
            slow_uturn_move: self.slow_uturn_move,
            volatile_statuses,
            substitute_health: self.substitute_health,
            attack_boost: self.attack_boost,
            defense_boost: self.defense_boost,
            special_attack_boost: self.special_attack_boost,
            special_defense_boost: self.special_defense_boost,
            speed_boost: self.speed_boost,
            accuracy_boost: self.accuracy_boost,
            evasion_boost: self.evasion_boost,
            last_used_move: LastUsedMove::deserialize(&self.last_used_move),
            damage_dealt: Default::default(),
            switch_out_move_second_saved_move: Choices::from_str(
                &self.switch_out_move_second_saved_move,
            )
            .unwrap(),
        }
    }
}

#[pymethods]
impl PySide {
    #[new]
    #[pyo3(signature = (
        pokemon=Vec::<PyPokemon>::new(),
        side_conditions=PySideConditions::from(SideConditions::default()),
        active_index="0".to_string(),
        baton_passing=false,
        shed_tailing=false,
        volatile_status_durations=PyVolatileStatusDurations::new(0, 0, 0, 0, 0, 0, 0, 0),
        wish=(0, 0),
        future_sight=(0, "0".to_string()),
        force_switch=false,
        force_trapped=false,
        slow_uturn_move=false,
        volatile_statuses=HashSet::<String>::new(),
        substitute_health=0,
        attack_boost=0,
        defense_boost=0,
        special_attack_boost=0,
        special_defense_boost=0,
        speed_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        last_used_move="move:none".to_string(),
        switch_out_move_second_saved_move="none".to_string(),
    ))]
    fn new(
        mut pokemon: Vec<PyPokemon>,
        side_conditions: PySideConditions,

        active_index: String,
        baton_passing: bool,
        shed_tailing: bool,
        volatile_status_durations: PyVolatileStatusDurations,
        wish: (i8, i16),
        future_sight: (i8, String),
        force_switch: bool,
        force_trapped: bool,
        slow_uturn_move: bool,
        volatile_statuses: HashSet<String>,
        substitute_health: i16,
        attack_boost: i8,
        defense_boost: i8,
        special_attack_boost: i8,
        special_defense_boost: i8,
        speed_boost: i8,
        accuracy_boost: i8,
        evasion_boost: i8,
        last_used_move: String,
        switch_out_move_second_saved_move: String,
    ) -> Self {
        while pokemon.len() < 6 {
            pokemon.push(PyPokemon::create_fainted());
        }
        PySide {
            pokemon: [
                pokemon[0].clone(),
                pokemon[1].clone(),
                pokemon[2].clone(),
                pokemon[3].clone(),
                pokemon[4].clone(),
                pokemon[5].clone(),
            ],
            side_conditions,
            active_index,
            baton_passing,
            shed_tailing,
            volatile_status_durations,
            wish,
            future_sight,
            force_switch,
            force_trapped,
            slow_uturn_move,
            volatile_statuses,
            substitute_health,
            attack_boost,
            defense_boost,
            special_attack_boost,
            special_defense_boost,
            speed_boost,
            accuracy_boost,
            evasion_boost,
            last_used_move,
            switch_out_move_second_saved_move,
        }
    }
}

#[derive(Clone)]
#[pyclass(
    name = "VolatileStatusDurations",
    module = "poke_engine",
    get_all,
    from_py_object
)]
pub struct PyVolatileStatusDurations {
    pub confusion: i8,
    pub encore: i8,
    pub lockedmove: i8,
    pub slowstart: i8,
    pub taunt: i8,
    pub yawn: i8,
    pub healblock: i8,
    pub throatchop: i8,
}

impl From<VolatileStatusDurations> for PyVolatileStatusDurations {
    fn from(other: VolatileStatusDurations) -> Self {
        PyVolatileStatusDurations {
            confusion: other.confusion,
            encore: other.encore,
            lockedmove: other.lockedmove,
            slowstart: other.slowstart,
            taunt: other.taunt,
            yawn: other.yawn,
            healblock: other.healblock,
            throatchop: other.throatchop,
        }
    }
}

impl Into<VolatileStatusDurations> for PyVolatileStatusDurations {
    fn into(self) -> VolatileStatusDurations {
        VolatileStatusDurations {
            confusion: self.confusion,
            encore: self.encore,
            lockedmove: self.lockedmove,
            slowstart: self.slowstart,
            taunt: self.taunt,
            yawn: self.yawn,
            healblock: self.healblock,
            throatchop: self.throatchop,
        }
    }
}

#[pymethods]
impl PyVolatileStatusDurations {
    #[new]
    #[pyo3(signature = (
        confusion=0,
        encore=0,
        lockedmove=0,
        slowstart=0,
        taunt=0,
        yawn=0,
        healblock=0,
        throatchop=0,
    ))]
    fn new(
        confusion: i8,
        encore: i8,
        lockedmove: i8,
        slowstart: i8,
        taunt: i8,
        yawn: i8,
        healblock: i8,
        throatchop: i8,
    ) -> PyVolatileStatusDurations {
        PyVolatileStatusDurations {
            confusion,
            encore,
            lockedmove,
            slowstart,
            taunt,
            yawn,
            healblock,
            throatchop,
        }
    }
}

#[derive(Clone)]
#[pyclass(
    name = "SideConditions",
    module = "poke_engine",
    get_all,
    from_py_object
)]
pub struct PySideConditions {
    pub aurora_veil: i8,
    pub crafty_shield: i8,
    pub healing_wish: i8,
    pub light_screen: i8,
    pub lucky_chant: i8,
    pub lunar_dance: i8,
    pub mat_block: i8,
    pub mist: i8,
    pub protect: i8,
    pub quick_guard: i8,
    pub reflect: i8,
    pub safeguard: i8,
    pub spikes: i8,
    pub stealth_rock: i8,
    pub sticky_web: i8,
    pub tailwind: i8,
    pub toxic_count: i8,
    pub toxic_spikes: i8,
    pub wide_guard: i8,
}

impl From<SideConditions> for PySideConditions {
    fn from(other: SideConditions) -> Self {
        PySideConditions {
            aurora_veil: other.aurora_veil,
            crafty_shield: other.crafty_shield,
            healing_wish: other.healing_wish,
            light_screen: other.light_screen,
            lucky_chant: other.lucky_chant,
            lunar_dance: other.lunar_dance,
            mat_block: other.mat_block,
            mist: other.mist,
            protect: other.protect,
            quick_guard: other.quick_guard,
            reflect: other.reflect,
            safeguard: other.safeguard,
            spikes: other.spikes,
            stealth_rock: other.stealth_rock,
            sticky_web: other.sticky_web,
            tailwind: other.tailwind,
            toxic_count: other.toxic_count,
            toxic_spikes: other.toxic_spikes,
            wide_guard: other.wide_guard,
        }
    }
}

impl Into<SideConditions> for PySideConditions {
    fn into(self) -> SideConditions {
        SideConditions {
            aurora_veil: self.aurora_veil,
            crafty_shield: self.crafty_shield,
            healing_wish: self.healing_wish,
            light_screen: self.light_screen,
            lucky_chant: self.lucky_chant,
            lunar_dance: self.lunar_dance,
            mat_block: self.mat_block,
            mist: self.mist,
            protect: self.protect,
            quick_guard: self.quick_guard,
            reflect: self.reflect,
            safeguard: self.safeguard,
            spikes: self.spikes,
            stealth_rock: self.stealth_rock,
            sticky_web: self.sticky_web,
            tailwind: self.tailwind,
            toxic_count: self.toxic_count,
            toxic_spikes: self.toxic_spikes,
            wide_guard: self.wide_guard,
        }
    }
}

#[pymethods]
impl PySideConditions {
    #[new]
    #[pyo3(signature = (
        spikes=0,
        toxic_spikes=0,
        stealth_rock=0,
        sticky_web=0,
        tailwind=0,
        lucky_chant=0,
        lunar_dance=0,
        reflect=0,
        light_screen=0,
        aurora_veil=0,
        crafty_shield=0,
        safeguard=0,
        mist=0,
        protect=0,
        healing_wish=0,
        mat_block=0,
        quick_guard=0,
        toxic_count=0,
        wide_guard=0,
    ))]
    fn new(
        spikes: i8,
        toxic_spikes: i8,
        stealth_rock: i8,
        sticky_web: i8,
        tailwind: i8,
        lucky_chant: i8,
        lunar_dance: i8,
        reflect: i8,
        light_screen: i8,
        aurora_veil: i8,
        crafty_shield: i8,
        safeguard: i8,
        mist: i8,
        protect: i8,
        healing_wish: i8,
        mat_block: i8,
        quick_guard: i8,
        toxic_count: i8,
        wide_guard: i8,
    ) -> Self {
        PySideConditions {
            spikes,
            toxic_spikes,
            stealth_rock,
            sticky_web,
            tailwind,
            lucky_chant,
            lunar_dance,
            reflect,
            light_screen,
            aurora_veil,
            crafty_shield,
            safeguard,
            mist,
            protect,
            healing_wish,
            mat_block,
            quick_guard,
            toxic_count,
            wide_guard,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "Pokemon", module = "poke_engine", get_all, from_py_object)]
pub struct PyPokemon {
    pub id: String,
    pub level: i8,
    pub types: (String, String),
    pub base_types: (String, String),
    pub hp: i16,
    pub maxhp: i16,
    pub ability: String,
    pub base_ability: String,
    pub item: String,
    pub nature: String,
    pub evs: (u8, u8, u8, u8, u8, u8),
    pub attack: i16,
    pub defense: i16,
    pub special_attack: i16,
    pub special_defense: i16,
    pub speed: i16,
    pub status: String,
    pub rest_turns: i8,
    pub sleep_turns: i8,
    pub weight_kg: f32,
    pub mega_evolved: bool,
    pub terastallized: bool,
    pub tera_type: String,
    pub moves: Vec<PyMove>,
}

impl From<Pokemon> for PyPokemon {
    fn from(other: Pokemon) -> Self {
        Self {
            id: other.id.to_string(),
            level: other.level,
            types: (other.types.0.to_string(), other.types.1.to_string()),
            base_types: (
                other.base_types.0.to_string(),
                other.base_types.1.to_string(),
            ),
            hp: other.hp,
            maxhp: other.maxhp,
            ability: other.ability.to_string(),
            base_ability: other.base_ability.to_string(),
            item: other.item.to_string(),
            nature: other.nature.to_string(),
            evs: (
                other.evs.0,
                other.evs.1,
                other.evs.2,
                other.evs.3,
                other.evs.4,
                other.evs.5,
            ),
            attack: other.attack,
            defense: other.defense,
            special_attack: other.special_attack,
            special_defense: other.special_defense,
            speed: other.speed,
            status: other.status.to_string(),
            rest_turns: other.rest_turns,
            sleep_turns: other.sleep_turns,
            weight_kg: other.weight_kg,
            mega_evolved: other.mega_evolved,
            terastallized: other.terastallized,
            tera_type: other.tera_type.to_string(),
            moves: other
                .moves
                .into_iter()
                .map(|m| PyMove::from(m.clone()))
                .collect(),
        }
    }
}

impl Into<Pokemon> for PyPokemon {
    fn into(self) -> Pokemon {
        let mut moves_vec = self.moves.clone();
        while moves_vec.len() < 4 {
            moves_vec.push(PyMove::create_empty_move());
        }
        Pokemon {
            id: PokemonName::from_str(&self.id).unwrap(),
            level: self.level,
            types: (
                PokemonType::from_str(&self.types.0).unwrap(),
                PokemonType::from_str(&self.types.1).unwrap(),
            ),
            base_types: (
                PokemonType::from_str(&self.base_types.0).unwrap(),
                PokemonType::from_str(&self.base_types.1).unwrap(),
            ),
            hp: self.hp,
            maxhp: self.maxhp,
            ability: Abilities::from_str(&self.ability).unwrap(),
            base_ability: Abilities::from_str(&self.base_ability).unwrap(),
            item: Items::from_str(&self.item).unwrap(),
            nature: PokemonNature::from_str(&self.nature).unwrap(),
            evs: (
                self.evs.0, self.evs.1, self.evs.2, self.evs.3, self.evs.4, self.evs.5,
            ),
            attack: self.attack,
            defense: self.defense,
            special_attack: self.special_attack,
            special_defense: self.special_defense,
            speed: self.speed,
            status: PokemonStatus::from_str(&self.status).unwrap(),
            rest_turns: self.rest_turns,
            sleep_turns: self.sleep_turns,
            weight_kg: self.weight_kg,
            mega_evolved: self.mega_evolved,
            terastallized: self.terastallized,
            tera_type: PokemonType::from_str(&self.tera_type).unwrap(),
            moves: PokemonMoves {
                m0: moves_vec[0].clone().into(),
                m1: moves_vec[1].clone().into(),
                m2: moves_vec[2].clone().into(),
                m3: moves_vec[3].clone().into(),
            },
        }
    }
}

#[pymethods]
impl PyPokemon {
    #[new]
    #[pyo3(signature = (
        id="pikachu".to_string(),
        level=50,
        types=("normal".to_string(), "typeless".to_string()),
        base_types=("normal".to_string(), "typeless".to_string()),
        hp=100,
        maxhp=100,
        ability="none".to_string(),
        base_ability="".to_string(),
        item="none".to_string(),
        nature="serious".to_string(),
        evs=(85, 85, 85, 85, 85, 85),
        attack=100,
        defense=100,
        special_attack=100,
        special_defense=100,
        speed=100,
        status="none".to_string(),
        rest_turns=0,
        sleep_turns=0,
        weight_kg=0.0,
        moves=Vec::<PyMove>::new(),
        mega_evolved=false,
        terastallized=false,
        tera_type="typeless".to_string(),
    ))]
    fn new(
        id: String,
        level: i8,
        types: (String, String),
        base_types: (String, String),
        hp: i16,
        maxhp: i16,
        ability: String,
        mut base_ability: String,
        item: String,
        nature: String,
        evs: (u8, u8, u8, u8, u8, u8),
        attack: i16,
        defense: i16,
        special_attack: i16,
        special_defense: i16,
        speed: i16,
        status: String,
        rest_turns: i8,
        sleep_turns: i8,
        weight_kg: f32,
        moves: Vec<PyMove>,
        mega_evolved: bool,
        terastallized: bool,
        tera_type: String,
    ) -> Self {
        if base_ability == "" {
            base_ability = ability.clone();
        }
        PyPokemon {
            id,
            level,
            types,
            base_types,
            hp,
            maxhp,
            ability,
            base_ability,
            item,
            nature,
            evs,
            attack,
            defense,
            special_attack,
            special_defense,
            speed,
            status,
            rest_turns,
            sleep_turns,
            weight_kg,
            mega_evolved,
            terastallized,
            tera_type,
            moves,
        }
    }
    #[staticmethod]
    pub fn create_fainted() -> PyPokemon {
        let pkmn = Pokemon::default();
        let mut py_pkmn = PyPokemon::from(pkmn);
        py_pkmn.hp = 0; // fainted
        py_pkmn
    }
}

#[derive(Clone)]
#[pyclass(name = "Move", module = "poke_engine", get_all, from_py_object)]
pub struct PyMove {
    pub id: String,
    pub disabled: bool,
    pub pp: i8,
    pub max_pp: i8,
}

impl From<Move> for PyMove {
    fn from(other: Move) -> Self {
        PyMove {
            id: other.id.to_string(),
            disabled: other.disabled,
            pp: other.pp,
            max_pp: other.max_pp,
        }
    }
}

impl Into<Move> for PyMove {
    fn into(self) -> Move {
        Move {
            id: Choices::from_str(&self.id).unwrap(),
            disabled: self.disabled,
            pp: self.pp,
            max_pp: self.max_pp,
            choice: MOVES
                .get(&Choices::from_str(&self.id).unwrap())
                .unwrap()
                .clone(),
        }
    }
}

impl PyMove {
    fn create_empty_move() -> PyMove {
        PyMove {
            id: String::new(),
            disabled: true,
            pp: 0,
            max_pp: 0,
        }
    }
}

#[pymethods]
impl PyMove {
    /// `max_pp` 0 = unknown: the engine then only tracks PP below 10 (legacy behaviour).
    #[new]
    #[pyo3(signature = (
        id="none".to_string(),
        pp=16,
        disabled=false,
        max_pp=0,
    ))]
    fn new(id: String, pp: i8, disabled: bool, max_pp: i8) -> Self {
        PyMove { id, disabled, pp, max_pp }
    }
}

#[derive(Clone)]
#[pyclass(get_all, from_py_object)]
struct PyMctsSideResult {
    pub move_choice: String,
    pub total_score: f32,
    pub visits: u32,
    pub avg_strategy: f32,
}

impl PyMctsSideResult {
    fn from_mcts_side_result(result: MctsSideResult, side: &Side) -> Self {
        PyMctsSideResult {
            move_choice: movechoice_to_string(side, &result.move_choice),
            total_score: result.total_score,
            visits: result.visits,
            avg_strategy: result.avg_strategy,
        }
    }
}

#[derive(Clone)]
#[pyclass(get_all, from_py_object)]
struct PyMctsResult {
    s1: Vec<PyMctsSideResult>,
    s2: Vec<PyMctsSideResult>,
    iteration_count: u32,
    depth_max: u32,
    depth_mean: f32,
    /// (s1 index, s2 index, visits, mean score for side one); None unless return_joint
    joint: Option<Vec<(usize, usize, u32, f32)>>,
    /// sampled interior search nodes (state string, visits, search value, static value,
    /// depth); None unless export_nodes > 0. Values are absolute P(side one wins).
    nodes: Option<Vec<(String, u32, f32, f32, u32)>>,
    /// the evaluator's absolute P(side one wins) at the root (NaN with threads > 1)
    root_value: f32,
    /// the search's absolute P(side one wins) at the root (NaN with threads > 1)
    root_search_value: f32,
}

impl PyMctsResult {
    fn from_mcts_result(result: MctsResult, state: &State) -> Self {
        PyMctsResult {
            s1: result
                .s1
                .iter()
                .map(|r| PyMctsSideResult::from_mcts_side_result(r.clone(), &state.side_one))
                .collect(),
            s2: result
                .s2
                .iter()
                .map(|r| PyMctsSideResult::from_mcts_side_result(r.clone(), &state.side_two))
                .collect(),
            iteration_count: result.iteration_count,
            depth_max: result.depth_max,
            depth_mean: result.depth_mean,
            joint: result
                .joint
                .map(|cells| cells.iter().map(|c| (c.s1, c.s2, c.visits, c.mean)).collect()),
            nodes: result.nodes.map(|nodes| {
                nodes
                    .into_iter()
                    .map(|n| (n.state, n.visits, n.value, n.static_value, n.depth))
                    .collect()
            }),
            root_value: result.root_value,
            root_search_value: result.root_search_value,
        }
    }
}

#[derive(Clone)]
#[pyclass(get_all, from_py_object)]
struct PyIterativeDeepeningResult {
    s1: Vec<String>,
    s2: Vec<String>,
    matrix: Vec<f32>,
    depth_searched: i8,
}

impl PyIterativeDeepeningResult {
    fn from_iterative_deepening_result(
        result: (Vec<MoveChoice>, Vec<MoveChoice>, Vec<f32>, i8),
        state: &State,
    ) -> Self {
        PyIterativeDeepeningResult {
            s1: result
                .0
                .iter()
                .map(|c| movechoice_to_string(&state.side_one, c))
                .collect(),
            s2: result
                .1
                .iter()
                .map(|c| movechoice_to_string(&state.side_two, c))
                .collect(),
            matrix: result.2,
            depth_searched: result.3,
        }
    }
}

/// A learned leaf evaluator (linear or MLP) for MCTS. See engine/README.md.
#[derive(Clone)]
#[pyclass(name = "Evaluator", module = "poke_engine", from_py_object)]
struct PyEvaluator {
    inner: Arc<Evaluator>,
}

fn value_err(e: String) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e)
}

/// `stock_coef == 0` keeps the evaluator plain; anything else makes it residual.
fn with_residual(ev: Evaluator, stock_coef: f32) -> PyResult<Evaluator> {
    if stock_coef == 0.0 {
        Ok(ev)
    } else {
        ev.with_stock_coef(stock_coef).map_err(value_err)
    }
}

#[pymethods]
impl PyEvaluator {
    /// Load a weights file written by `write_linear` / `write_mlp` (or Rust `Evaluator::save`).
    #[staticmethod]
    #[pyo3(signature = (path, relative_to_root=true, leaf_scale=1.0, material=0.0))]
    fn load(path: &str, relative_to_root: bool, leaf_scale: f32, material: f32) -> PyResult<PyEvaluator> {
        let ev = Evaluator::load(path)
            .map_err(value_err)?
            .with_relative_to_root(relative_to_root)
            .with_leaf_scale(leaf_scale)
            .map_err(value_err)?
            .with_material(material)
            .map_err(value_err)?;
        Ok(PyEvaluator { inner: Arc::new(ev) })
    }

    /// value = weights . state_features + bias
    #[staticmethod]
    #[pyo3(signature = (weights, bias=0.0, relative_to_root=true, leaf_scale=1.0, stock_coef=0.0, feature_version=1, material=0.0))]
    fn linear(
        weights: Vec<f32>,
        bias: f32,
        relative_to_root: bool,
        leaf_scale: f32,
        stock_coef: f32,
        feature_version: u32,
        material: f32,
    ) -> PyResult<PyEvaluator> {
        let ev = with_residual(Evaluator::linear_v(feature_version, weights, bias).map_err(value_err)?, stock_coef)?
            .with_relative_to_root(relative_to_root)
            .with_leaf_scale(leaf_scale)
            .map_err(value_err)?
            .with_material(material)
            .map_err(value_err)?;
        Ok(PyEvaluator { inner: Arc::new(ev) })
    }

    /// value = w2 . relu(W1 x + b1) + b2; `w1` is flat row-major [hidden][FEATURE_LENGTH].
    #[staticmethod]
    #[pyo3(signature = (w1, b1, w2, b2=0.0, relative_to_root=true, leaf_scale=1.0, stock_coef=0.0, feature_version=1, material=0.0))]
    fn mlp(
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
        relative_to_root: bool,
        leaf_scale: f32,
        stock_coef: f32,
        feature_version: u32,
        material: f32,
    ) -> PyResult<PyEvaluator> {
        let hidden = b1.len();
        let ev = with_residual(Evaluator::mlp_v(feature_version, hidden, w1, b1, w2, b2).map_err(value_err)?, stock_coef)?
            .with_relative_to_root(relative_to_root)
            .with_leaf_scale(leaf_scale)
            .map_err(value_err)?
            .with_material(material)
            .map_err(value_err)?;
        Ok(PyEvaluator { inner: Arc::new(ev) })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(value_err)
    }

    /// The model output for `state`: logit of P(side one wins).
    fn evaluate(&self, py_state: PyState) -> f32 {
        let state: State = py_state.into();
        let mut scratch = Scratch::new();
        self.inner.logit(&state, &mut scratch)
    }

    /// Alias of `evaluate`: the model logit of P(side one wins) at `state`.
    fn logit(&self, py_state: PyState) -> f32 {
        self.evaluate(py_state)
    }

    #[getter]
    fn leaf_scale(&self) -> f32 {
        self.inner.leaf_scale
    }

    /// Material blend lambda (0 = off).
    #[getter]
    fn material(&self) -> f32 {
        self.inner.material
    }

    /// Feature version the weights consume (1 to 5).
    #[getter]
    fn feature_version(&self) -> u32 {
        self.inner.feature_version()
    }

    /// Embedding dims (species, item, ability, move) of a feature-version-4/5 net, else None.
    #[getter]
    fn embed_dims(&self) -> Option<Vec<usize>> {
        self.inner.embed().map(|e| e.dims().to_vec())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        self.inner.kind_name()
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    #[getter]
    fn relative_to_root(&self) -> bool {
        self.inner.relative_to_root
    }

    /// True if the logit includes `stock_coef * evaluate(state)`.
    #[getter]
    fn residual(&self) -> bool {
        self.inner.is_residual()
    }

    /// Multiplier on the raw stock evaluation (0.0 when not residual).
    #[getter]
    fn stock_coef(&self) -> f32 {
        self.inner.stock_coef.unwrap_or(0.0)
    }
}

/// The stock hand-written evaluation of `state` (eval points, side one's view).
/// A learned policy prior (`.ppol`) for PUCT selection in MCTS. See docs/policy-prior.md.
#[derive(Clone)]
#[pyclass(name = "PolicyPrior", module = "poke_engine", from_py_object)]
struct PyPolicyPrior {
    inner: Arc<PolicyNet>,
}

fn root_options(state: &State, side: u8) -> (&Side, Vec<MoveChoice>) {
    let (o1, o2) = state.root_get_all_options();
    if side == 1 {
        (&state.side_one, o1)
    } else {
        (&state.side_two, o2)
    }
}

#[pymethods]
impl PyPolicyPrior {
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyPolicyPrior> {
        Ok(PyPolicyPrior { inner: Arc::new(PolicyNet::load(path).map_err(value_err)?) })
    }

    /// Flat row-major weights in file order: `w1` [hidden][feature_len], `w2` [13][hidden],
    /// `p` [emb][hidden], `a` [emb][n_afeat] (see engine/src/prior.rs).
    #[staticmethod]
    #[pyo3(signature = (w1, b1, w2, b2, p, c, a, v, feature_version=2))]
    #[allow(clippy::too_many_arguments)]
    fn mlp(
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        p: Vec<f32>,
        c: Vec<f32>,
        a: Vec<f32>,
        v: Vec<f32>,
        feature_version: u32,
    ) -> PyResult<PyPolicyPrior> {
        let hidden = b1.len();
        let net = PolicyNet::new(feature_version, hidden, PolicyWeights { w1, b1, w2, b2, p, c, a, v })
            .map_err(value_err)?;
        Ok(PyPolicyPrior { inner: Arc::new(net) })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(value_err)
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }

    #[getter]
    fn feature_version(&self) -> u32 {
        self.inner.feature_version()
    }

    /// Raw logits of the legal root options of `side` (1 or 2): [(choice, slot, logit)].
    #[pyo3(signature = (py_state, side=1))]
    fn option_logits(&self, py_state: PyState, side: u8) -> PyResult<Vec<(String, i32, f32)>> {
        let r = side_ref(side)?;
        let state: State = py_state.into();
        let (own, opts) = root_options(&state, side);
        let mut scratch = Scratch::new();
        let z = self.inner.option_logits(&state, r, &opts, &mut scratch);
        Ok(opts
            .iter()
            .zip(z)
            .map(|(o, z)| {
                (movechoice_to_string(own, o), action_slot(own, o).map(|s| s as i32).unwrap_or(-1), z)
            })
            .collect())
    }

    /// Softmax prior over the legal root options of `side`: [(choice, probability)].
    #[pyo3(signature = (py_state, side=1))]
    fn priors(&self, py_state: PyState, side: u8) -> PyResult<Vec<(String, f32)>> {
        let rows = self.option_logits(py_state, side)?;
        let max = rows.iter().map(|r| r.2).fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = rows.iter().map(|r| (r.2 - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        Ok(rows.into_iter().zip(e).map(|(r, x)| (r.0, x / sum)).collect())
    }
}

fn side_ref(side: u8) -> PyResult<SideReference> {
    match side {
        1 => Ok(SideReference::SideOne),
        2 => Ok(SideReference::SideTwo),
        _ => Err(value_err(format!("side must be 1 or 2, got {side}"))),
    }
}

/// The legal root options of `side` with their action slot (-1: none) and the `N_POLICY_AFEAT`
/// action features the policy prior sees: [(choice, slot, features)]. Training code must use
/// this (it is the engine's own computation) to build inputs and targets.
#[pyfunction]
#[pyo3(signature = (py_state, side=1))]
fn policy_action_features(py_state: PyState, side: u8) -> PyResult<Vec<(String, i32, Vec<f32>)>> {
    let r = side_ref(side)?;
    let state: State = py_state.into();
    let (own, opts) = root_options(&state, side);
    Ok(opts
        .iter()
        .map(|o| {
            let mut f = [0.0f32; N_AFEAT];
            action_features(&state, r, o, &mut f);
            (
                movechoice_to_string(own, o),
                action_slot(own, o).map(|s| s as i32).unwrap_or(-1),
                f.to_vec(),
            )
        })
        .collect())
}

/// The root options of `side` as `(choice string, action slot)`; slot is -1 for an option with
/// no learned slot.
#[pyfunction]
#[pyo3(signature = (py_state, side=1))]
fn policy_actions(py_state: PyState, side: u8) -> PyResult<Vec<(String, i32)>> {
    side_ref(side)?;
    let state: State = py_state.into();
    let (own, opts) = root_options(&state, side);
    Ok(opts
        .iter()
        .map(|o| (movechoice_to_string(own, o), action_slot(own, o).map(|s| s as i32).unwrap_or(-1)))
        .collect())
}

// ---------------------------------------------------------------------------
// Transformer tokens (bead showdown-bot-589.68.1, docs/tokens.md)
// ---------------------------------------------------------------------------

type RevealArg = Option<Vec<(String, Vec<String>, bool, bool, bool)>>;
type TokenOut<'py> = (Bound<'py, PyBytes>, Bound<'py, PyBytes>, Vec<String>);

fn reveal_of(r: RevealArg) -> Option<Vec<tokens::Revealed>> {
    r.map(|rows| {
        rows.into_iter()
            .map(|(species, moves, item, ability, tera)| tokens::Revealed { species, moves, item, ability, tera })
            .collect()
    })
}

fn i16_bytes(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The 13 action-slot choice strings of `side` ("" for an empty slot), from the same
/// `root_options` + `action_slot` path as `policy_actions`.
fn slot_names(state: &State, side: u8) -> Vec<String> {
    let (own, opts) = root_options(state, side);
    let mut names = vec![String::new(); N_ACTIONS];
    for o in &opts {
        if let Some(s) = action_slot(own, o) {
            names[s] = movechoice_to_string(own, o);
        }
    }
    names
}

fn tokens_core<'py>(
    py: Python<'py>,
    state: &State,
    side: u8,
    reveal: RevealArg,
    with_choices: bool,
) -> PyResult<TokenOut<'py>> {
    let r = side_ref(side)?;
    let rev = reveal_of(reveal);
    let mut cat = vec![0i16; tokens::CAT_LEN];
    let mut cont = vec![0f32; tokens::CONT_LEN];
    tokens::state_tokens_into(state, r, rev.as_deref(), &mut cat, &mut cont);
    let mut names = Vec::new();
    if with_choices {
        names = slot_names(state, side);
        names.extend(slot_names(state, 3 - side));
    }
    Ok((PyBytes::new(py, &i16_bytes(&cat)), PyBytes::new(py, &f32_bytes(&cont)), names))
}

/// Token buffers of a `State` object: `(cat int16 bytes, cont float32 bytes, choices)`; see
/// `token_layout()` / docs/tokens.md. `opp_reveal` is an optional list of
/// `(species, moves, item_revealed, ability_revealed, tera_revealed)` rows; when given, the
/// opponent is reduced to what they list. `choices` is the 26 slot names (own 13, opp 13)
/// when `with_choices`, else empty.
#[pyfunction]
#[pyo3(signature = (py_state, side=1, opp_reveal=None, with_choices=false))]
fn state_tokens<'py>(
    py: Python<'py>,
    py_state: PyState,
    side: u8,
    opp_reveal: RevealArg,
    with_choices: bool,
) -> PyResult<TokenOut<'py>> {
    let state: State = py_state.into();
    tokens_core(py, &state, side, opp_reveal, with_choices)
}

/// `state_tokens` for a serialized state string (`State.to_string()`): skips the slow
/// `State` object round trip.
#[pyfunction]
#[pyo3(signature = (state, side=1, opp_reveal=None, with_choices=false))]
fn state_tokens_str<'py>(
    py: Python<'py>,
    state: &str,
    side: u8,
    opp_reveal: RevealArg,
    with_choices: bool,
) -> PyResult<TokenOut<'py>> {
    let state = State::deserialize(state);
    tokens_core(py, &state, side, opp_reveal, with_choices)
}

/// Hidden-set targets (int16 bytes: `6 * HID_W` ids, 6 mask flags, 6 true species ids) for the
/// opponent slots of `state`, taken from `truth`'s set of the same species.
#[pyfunction]
#[pyo3(signature = (state, truth, side=1))]
fn state_hidden_targets<'py>(py: Python<'py>, state: &str, truth: &str, side: u8) -> PyResult<Bound<'py, PyBytes>> {
    let r = side_ref(side)?;
    let (s, t) = (State::deserialize(state), State::deserialize(truth));
    Ok(PyBytes::new(py, &i16_bytes(&tokens::hidden_targets(&s, &t, r))))
}

/// Mean seconds per core `state_tokens` computation (no Python conversion), `n` repetitions.
#[pyfunction]
#[pyo3(signature = (state, side=1, n=1000, scrub=false))]
fn tokens_bench(state: &str, side: u8, n: usize, scrub: bool) -> PyResult<f64> {
    let r = side_ref(side)?;
    let s = State::deserialize(state);
    let rev = if scrub { Some(vec![tokens::Revealed::default()]) } else { None };
    let mut cat = vec![0i16; tokens::CAT_LEN];
    let mut cont = vec![0f32; tokens::CONT_LEN];
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        tokens::state_tokens_into(&s, r, rev.as_deref(), &mut cat, &mut cont);
        std::hint::black_box(&cont);
    }
    Ok(t0.elapsed().as_secs_f64() / n as f64)
}

/// `(name, buffer, offset, shape, dtype)` of every token field; buffer is "cat" (int16) or
/// "cont" (float32), offset in elements.
#[pyfunction]
fn token_layout() -> Vec<(String, String, usize, Vec<usize>, String)> {
    tokens::layout()
        .into_iter()
        .map(|(n, b, o, s, d)| (n.to_string(), b.to_string(), o, s, d.to_string()))
        .collect()
}

/// Constants of the token layout.
#[pyfunction]
fn token_constants() -> Vec<(String, usize)> {
    vec![
        ("TOKEN_VERSION", tokens::TOKEN_VERSION as usize),
        ("N_MON", tokens::N_MON),
        ("IDS_PER_MON", features_v4::IDS_PER_SLOT),
        ("KNOWN_W", tokens::KNOWN_W),
        ("C_MON", tokens::C_MON),
        ("C_SIDE", tokens::C_SIDE),
        ("C_GLOB", tokens::C_GLOB),
        ("N_PAIR", poke_engine::features_v3::N_PAIR),
        ("N_PAIR_CH", tokens::N_PAIR_CH),
        ("N_AGG", tokens::N_AGG),
        ("N_ACT", tokens::N_ACT),
        ("N_AFEAT", N_AFEAT),
        ("ACT_CAT_W", tokens::ACT_CAT_W),
        ("HID_W", tokens::HID_W),
        ("META_W", tokens::META_W),
        ("CAT_LEN", tokens::CAT_LEN),
        ("CONT_LEN", tokens::CONT_LEN),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect()
}

// ---------------------------------------------------------------------------
// Net-guided search (bead showdown-bot-589.68.5, docs/net-search.md)
// ---------------------------------------------------------------------------

type OptRow = (String, u32, f32, f32);

/// Batched net-guided search over sampled worlds: `collect` returns the token buffers of the
/// leaves to evaluate, `feed` takes the evaluations back. Tree work releases the GIL.
#[pyclass(name = "NetSearch", module = "poke_engine")]
struct PyNetSearch {
    inner: netsearch::NetSearch,
}

fn le_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[pymethods]
impl PyNetSearch {
    /// `states`: serialized world states (side one = us). `reveal` (optional rows as in
    /// `state_tokens_str`) scrubs the opponent in side one's token view only.
    #[new]
    #[pyo3(signature = (states, c_puct=2.0, fpu=0.0, depth=4, eps=0.01, seed=0, reveal=None, views=2))]
    fn new(
        states: Vec<String>,
        c_puct: f32,
        fpu: f32,
        depth: u16,
        eps: f32,
        seed: u64,
        reveal: RevealArg,
        views: u8,
    ) -> PyResult<PyNetSearch> {
        if states.is_empty() {
            return Err(value_err("NetSearch needs at least one world".to_string()));
        }
        if !(c_puct.is_finite() && c_puct > 0.0) || !fpu.is_finite() || !(0.0..1.0).contains(&eps) || depth == 0 || !(1..=2).contains(&views) {
            return Err(value_err(format!(
                "need c_puct > 0, finite fpu, 0 <= eps < 1, depth >= 1, views 1 or 2 (got c_puct={c_puct}, fpu={fpu}, eps={eps}, depth={depth}, views={views})"
            )));
        }
        let worlds: Vec<State> = states.iter().map(|s| State::deserialize(s)).collect();
        let params = netsearch::Params { c_puct, fpu, depth, eps, views };
        Ok(PyNetSearch { inner: netsearch::NetSearch::new(worlds, params, reveal_of(reveal), seed) })
    }

    /// `(n, cat bytes [n,views,CAT_LEN] int16, cont bytes [n,views,CONT_LEN] float32)` of up
    /// to `max_batch` leaves; view 0 is side one's, view 1 (when `views` = 2) side two's.
    fn collect<'py>(&mut self, py: Python<'py>, max_batch: usize) -> (usize, Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let c = py.detach(|| self.inner.collect(max_batch));
        // zero-copy views of the buffers (the wire format is little endian = the host's)
        #[cfg(not(target_endian = "little"))]
        compile_error!("NetSearch::collect hands out host-endian buffers; port the byte view first");
        let cat = unsafe { std::slice::from_raw_parts(c.cat.as_ptr() as *const u8, c.cat.len() * 2) };
        let cont = unsafe { std::slice::from_raw_parts(c.cont.as_ptr() as *const u8, c.cont.len() * 4) };
        (c.n, PyBytes::new(py, cat), PyBytes::new(py, cont))
    }

    /// Completes the oldest collected batch: float32 little-endian bytes `policy1 [n,13]`,
    /// `policy2 [n,13]` (each side's policy over its own action slots) and `value [n]`
    /// (P(side one wins)).
    fn feed(&mut self, py: Python<'_>, policy1: &[u8], policy2: &[u8], value: &[u8]) -> PyResult<()> {
        let (p1, p2, v) = (le_f32(policy1), le_f32(policy2), le_f32(value));
        let n = v.len();
        if p1.len() != n * N_ACTIONS || p2.len() != n * N_ACTIONS {
            return Err(value_err(format!(
                "feed shapes: {} values, {} / {} policy entries (need {} each)",
                n,
                p1.len(),
                p2.len(),
                n * N_ACTIONS
            )));
        }
        let evals: Vec<netsearch::Eval> = (0..n)
            .map(|i| {
                let mut e = netsearch::Eval { policy1: [0.0; N_ACTIONS], policy2: [0.0; N_ACTIONS], value: v[i] };
                e.policy1.copy_from_slice(&p1[i * N_ACTIONS..(i + 1) * N_ACTIONS]);
                e.policy2.copy_from_slice(&p2[i * N_ACTIONS..(i + 1) * N_ACTIONS]);
                e
            })
            .collect();
        py.detach(|| self.inner.feed(&evals)).map_err(value_err)
    }

    /// Leaves handed out and not yet fed.
    fn pending(&self) -> usize {
        self.inner.pending()
    }

    fn n_worlds(&self) -> usize {
        self.inner.n_worlds()
    }

    /// Per world `(side one options, side two options, root value, root visits)`; an option
    /// row is `(choice, visits, mean value from that side's view, prior)`.
    #[allow(clippy::type_complexity)]
    fn result(&self) -> Vec<(Vec<OptRow>, Vec<OptRow>, f32, u32)> {
        (0..self.inner.n_worlds())
            .map(|w| {
                let (r1, r2) = self.inner.root(w);
                let st = self.inner.root_state(w);
                let conv = |rows: Vec<(MoveChoice, u32, f32, f32)>, side: &Side| -> Vec<OptRow> {
                    rows.into_iter().map(|(c, n, q, p)| (movechoice_to_string(side, &c), n, q, p)).collect()
                };
                (conv(r1, &st.side_one), conv(r2, &st.side_two), self.inner.root_value(w), self.inner.root_visits(w))
            })
            .collect()
    }

    fn stats(&self) -> std::collections::HashMap<String, f64> {
        let s = &self.inner.stats;
        [
            ("evals", s.evals as f64),
            ("batches", s.batches as f64),
            ("collisions", s.collisions as f64),
            ("terminal", s.terminal as f64),
            ("leaf_reuse", s.leaf_reuse as f64),
            ("both_none", s.both_none as f64),
            ("max_depth", s.max_depth as f64),
            ("nodes", s.nodes as f64),
            ("tree_ms", s.tree_ns as f64 / 1e6),
            ("tokenise_ms", s.tokenise_ns as f64 / 1e6),
            ("feed_ms", s.feed_ns as f64 / 1e6),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }
}

#[pyfunction(name = "evaluate")]
fn stock_evaluate(py_state: PyState) -> f32 {
    let state: State = py_state.into();
    poke_engine::engine::evaluate::evaluate(&state)
}

/// Experimental (589.32): scale the stock evaluation's alive/item bonuses of a Pokemon by
/// `min(1, hp_fraction / knee)`. `0.0` (the default) restores the stock evaluation.
/// Process-wide; affects `evaluate`, residual learned evaluators and the MCTS leaf.
#[pyfunction]
fn set_eval_low_hp_knee(knee: f32) {
    poke_engine::engine::evaluate::set_low_hp_knee(knee);
}

/// Experimental (589.32): multiply every stat-boost term of the stock evaluation by `scale`.
#[pyfunction]
fn set_eval_boost_scale(scale: f32) {
    poke_engine::engine::evaluate::set_boost_scale(scale);
}

/// The current low-HP knee (see `set_eval_low_hp_knee`).
#[pyfunction]
fn eval_low_hp_knee() -> f32 {
    poke_engine::engine::evaluate::low_hp_knee()
}

/// The fixed-length feature vector of `state` from `side`'s perspective (1 or 2).
#[pyfunction]
#[pyo3(signature = (py_state, side=1, version=None))]
fn state_features(py_state: PyState, side: u8, version: Option<u32>) -> PyResult<Vec<f32>> {
    let side_ref = match side {
        1 => SideReference::SideOne,
        2 => SideReference::SideTwo,
        _ => return Err(value_err(format!("side must be 1 or 2, got {side}"))),
    };
    let version = check_version(version)?;
    let state: State = py_state.into();
    Ok(features_versioned(&state, side_ref, version))
}

fn check_version(version: Option<u32>) -> PyResult<u32> {
    let v = version.unwrap_or(FEATURE_VERSION);
    if feature_len(v).is_none() {
        return Err(value_err(format!("unknown feature version {v} (known: 1 to 5)")));
    }
    Ok(v)
}

/// Identity indices of feature version 4 (`IDENTITY_LENGTH` = 84 ints): per Pokemon slot (own
/// 0..5 then opp 0..5, slot 0 = the active) `[species, item, ability, move0..move3]`; 0 = pad
/// (fainted), 1 = unknown, else enum discriminant + 2. See docs/features-v4.md.
#[pyfunction]
#[pyo3(signature = (py_state, side=1))]
fn state_identity(py_state: PyState, side: u8) -> PyResult<Vec<u16>> {
    let side_ref = match side {
        1 => SideReference::SideOne,
        2 => SideReference::SideTwo,
        _ => return Err(value_err(format!("side must be 1 or 2, got {side}"))),
    };
    let state: State = py_state.into();
    Ok(features_v4::identity(&state, side_ref))
}

/// Rows of the four embedding tables (species, item, ability, move), pad and unknown included.
#[pyfunction]
fn identity_vocab_sizes() -> Vec<usize> {
    (0..features_v4::N_TABLES).map(features_v4::vocab_size).collect()
}

/// Entry names of embedding table `table` (0 species, 1 item, 2 ability, 3 move), in row order.
#[pyfunction]
fn identity_vocab(table: usize) -> PyResult<Vec<String>> {
    if table >= features_v4::N_TABLES {
        return Err(value_err(format!("table must be 0..{}, got {table}", features_v4::N_TABLES - 1)));
    }
    Ok(features_v4::vocab_names(table))
}

/// Fingerprint of the four vocabularies, stored in v4 `.pev` files.
#[pyfunction]
fn identity_vocab_hash() -> u64 {
    features_v4::vocab_hash()
}

/// Names of the features, same order as `state_features`.
#[pyfunction]
#[pyo3(signature = (version=None))]
fn state_feature_names(version: Option<u32>) -> PyResult<Vec<String>> {
    Ok(feature_names_versioned(check_version(version)?))
}

/// Length of the feature vector of `version` (default: `FEATURE_VERSION`).
#[pyfunction]
#[pyo3(signature = (version=None))]
fn feature_length(version: Option<u32>) -> PyResult<usize> {
    Ok(feature_len(check_version(version)?).unwrap())
}

#[pyfunction]
#[pyo3(signature = (py_state, duration_ms, iterations, threads, evaluator=None, selection="ucb1", gamma=0.1, return_joint=false, prior=None, prior_opp=None, c_puct=1.0, prior_eps=0.03, fpu_reduction=0.0, root_prior_s1=None, root_prior_s2=None, export_nodes=0, export_min_visits=16))]
fn mcts(
    py_state: PyState,
    duration_ms: u64,
    mut iterations: u32,
    threads: usize,
    evaluator: Option<PyEvaluator>,
    selection: &str,
    gamma: f64,
    return_joint: bool,
    prior: Option<PyPolicyPrior>,
    prior_opp: Option<PyPolicyPrior>,
    c_puct: f32,
    prior_eps: f32,
    fpu_reduction: f32,
    root_prior_s1: Option<std::collections::HashMap<String, f32>>,
    root_prior_s2: Option<std::collections::HashMap<String, f32>>,
    export_nodes: usize,
    export_min_visits: u32,
) -> PyResult<PyMctsResult> {
    let mode = SelectionMode::parse(selection).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown selection {selection:?} (expected ucb1, exp3 or rm)"
        ))
    })?;
    if !(gamma.is_finite() && gamma > 0.0 && gamma <= 1.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gamma must be in (0, 1], got {gamma}"
        )));
    }
    if mode != SelectionMode::Ucb1 && threads > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "selection exp3/rm is single-threaded only; the shared-tree search is ucb1 (use threads=1)",
        ));
    }
    if return_joint && threads > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "return_joint is single-threaded only (use threads=1)",
        ));
    }
    let use_prior = prior.is_some() || prior_opp.is_some() || root_prior_s1.is_some() || root_prior_s2.is_some();
    if use_prior {
        if mode != SelectionMode::Ucb1 {
            return Err(value_err("policy priors (PUCT) need selection=\"ucb1\"".to_string()));
        }
        if threads > 1 {
            return Err(value_err("policy priors are single-threaded only (use threads=1)".to_string()));
        }
        if !(c_puct.is_finite() && c_puct > 0.0) || !(0.0..1.0).contains(&prior_eps) || !fpu_reduction.is_finite() {
            return Err(value_err(format!(
                "need c_puct > 0 and 0 <= prior_eps < 1 (got c_puct={c_puct}, prior_eps={prior_eps})"
            )));
        }
    }
    if export_nodes > 0 && threads > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "export_nodes is single-threaded only (use threads=1)",
        ));
    }
    let export_config = ExportConfig {
        max_nodes: export_nodes,
        min_visits: export_min_visits.max(1),
    };
    let evaluator_ref: Option<&Evaluator> = evaluator.as_ref().map(|e| e.inner.as_ref());
    let mut state: State = py_state.into();
    let duration = Duration::from_millis(duration_ms);
    let (s1_options, s2_options) = state.root_get_all_options();
    // root-only overrides, aligned with the root option order (unlisted options get 0)
    let root_vec = |m: &Option<std::collections::HashMap<String, f32>>, side: &Side, opts: &[MoveChoice]| {
        m.as_ref().map(|m| {
            opts.iter()
                .map(|o| m.get(&movechoice_to_string(side, o)).copied().unwrap_or(0.0))
                .collect::<Vec<f32>>()
        })
    };
    let root_s1 = root_vec(&root_prior_s1, &state.side_one, &s1_options);
    let root_s2 = root_vec(&root_prior_s2, &state.side_two, &s2_options);
    let prior_config = if use_prior {
        let mut pc = PriorConfig::new(c_puct);
        pc.net_s1 = prior.as_ref().map(|p| p.inner.as_ref());
        pc.net_s2 = prior_opp.as_ref().map(|p| p.inner.as_ref());
        pc.root_s1 = root_s1.as_deref();
        pc.root_s2 = root_s2.as_deref();
        pc.eps = prior_eps;
        pc.fpu_reduction = fpu_reduction;
        Some(pc)
    } else {
        None
    };
    let selection_config = SelectionConfig { mode, gamma, prior: prior_config };
    if s1_options.len() <= 1 {
        iterations = 100; // if there's only one option, force a quick exit
    }
    let mcts_result = if threads > 1 {
        perform_mcts_shared_tree_with(
            &mut state,
            s1_options,
            s2_options,
            duration,
            iterations,
            threads,
            evaluator_ref,
        )
    } else {
        perform_mcts_export(
            &mut state,
            s1_options,
            s2_options,
            duration,
            iterations,
            evaluator_ref,
            &selection_config,
            return_joint,
            if export_nodes > 0 { Some(&export_config) } else { None },
        )
    };

    let py_mcts_result = PyMctsResult::from_mcts_result(mcts_result, &state);
    Ok(py_mcts_result)
}

#[pyfunction]
fn id(py_state: PyState, duration_ms: u64) -> PyResult<PyIterativeDeepeningResult> {
    let mut state: State = py_state.into();
    let duration = Duration::from_millis(duration_ms);
    let (s1_options, s2_options) = state.root_get_all_options();
    let id_result = iterative_deepen_expectiminimax(&mut state, s1_options, s2_options, duration);

    let py_id_result =
        PyIterativeDeepeningResult::from_iterative_deepening_result(id_result, &state);
    Ok(py_id_result)
}

#[derive(Clone)]
#[pyclass(
    name = "StateInstructions",
    module = "poke_engine",
    get_all,
    from_py_object
)]
struct PyStateInstructions {
    pub percentage: f32,
    pub instruction_list: Vec<PyInstruction>,
}

impl From<StateInstructions> for PyStateInstructions {
    fn from(other: StateInstructions) -> Self {
        PyStateInstructions {
            percentage: other.percentage,
            instruction_list: other
                .instruction_list
                .into_iter()
                .map(|i| PyInstruction::from(i))
                .collect(),
        }
    }
}

impl Into<StateInstructions> for PyStateInstructions {
    fn into(self) -> StateInstructions {
        StateInstructions {
            percentage: self.percentage,
            instruction_list: self
                .instruction_list
                .into_iter()
                .map(|i| i.into())
                .collect(),
        }
    }
}

#[pymethods]
impl PyStateInstructions {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "{}: {:?}",
            self.percentage,
            self.instruction_list
                .iter()
                .map(|x| format!("{:?}", x.instruction))
                .collect::<Vec<_>>()
                .join(", ")
        ))
    }
}

#[derive(Clone)]
#[pyclass(module = "poke_engine", from_py_object)]
struct PyInstruction {
    instruction: Instruction,
}

impl From<Instruction> for PyInstruction {
    fn from(instruction: Instruction) -> Self {
        PyInstruction { instruction }
    }
}

impl Into<Instruction> for PyInstruction {
    fn into(self) -> Instruction {
        self.instruction
    }
}

#[pymethods]
impl PyInstruction {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.instruction))
    }
}

#[pyfunction]
fn generate_instructions(
    py_state: PyState,
    side_one_move: String,
    side_two_move: String,
) -> PyResult<Vec<PyStateInstructions>> {
    let (s1_move, s2_move);
    let mut state: State = py_state.into();
    match MoveChoice::from_string(&side_one_move, &state.side_one) {
        Some(m) => s1_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {}",
                side_one_move
            )))
        }
    }
    match MoveChoice::from_string(&side_two_move, &state.side_two) {
        Some(m) => s2_move = m,
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {}",
                side_two_move
            )))
        }
    }
    let instructions = generate_instructions_from_move_pair(&mut state, &s1_move, &s2_move, true);
    let py_instructions = instructions.iter().map(|i| i.clone().into()).collect();

    Ok(py_instructions)
}

#[pyfunction]
fn calculate_damage(
    py_state: PyState,
    side_one_move: String,
    side_two_move: String,
    side_one_moves_first: bool,
) -> PyResult<(Vec<i16>, Vec<i16>)> {
    let state: State = py_state.into();
    let (mut s1_choice, mut s2_choice);
    match MOVES.get(&Choices::from_str(side_one_move.as_str()).unwrap()) {
        Some(m) => s1_choice = m.to_owned(),
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s1: {}",
                side_one_move
            )))
        }
    }
    match MOVES.get(&Choices::from_str(side_two_move.as_str()).unwrap()) {
        Some(m) => s2_choice = m.to_owned(),
        None => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid move for s2: {}",
                side_one_move
            )))
        }
    }
    if side_one_move == "switch" {
        s1_choice.category = MoveCategory::Switch
    }
    if side_two_move == "switch" {
        s2_choice.category = MoveCategory::Switch
    }

    let (s1_damage_rolls, s2_damage_rolls) =
        calculate_both_damage_rolls(&state, s1_choice, s2_choice, side_one_moves_first);

    let (s1_py_rolls, s2_py_rolls);
    match s1_damage_rolls {
        Some(rolls) => s1_py_rolls = rolls,
        None => s1_py_rolls = vec![0, 0],
    }
    match s2_damage_rolls {
        Some(rolls) => s2_py_rolls = rolls,
        None => s2_py_rolls = vec![0, 0],
    }

    Ok((s1_py_rolls, s2_py_rolls))
}

#[pymodule]
#[pyo3(name = "poke_engine")]
fn py_poke_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_damage, m)?)?;
    m.add_function(wrap_pyfunction!(generate_instructions, m)?)?;
    m.add_function(wrap_pyfunction!(id, m)?)?;
    m.add_function(wrap_pyfunction!(mcts, m)?)?;
    m.add_function(wrap_pyfunction!(stock_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(set_eval_low_hp_knee, m)?)?;
    m.add_function(wrap_pyfunction!(eval_low_hp_knee, m)?)?;
    m.add_function(wrap_pyfunction!(set_eval_boost_scale, m)?)?;
    m.add_function(wrap_pyfunction!(state_features, m)?)?;
    m.add_function(wrap_pyfunction!(state_feature_names, m)?)?;
    m.add("FEATURE_VERSION", FEATURE_VERSION)?;
    m.add("FEATURE_LENGTH", FEATURE_LEN)?;
    m.add("FEATURE_LENGTH_V2", FEATURE_LEN_V2)?;
    m.add("FEATURE_LENGTH_V3", FEATURE_LEN_V3)?;
    m.add("FEATURE_LENGTH_V4", FEATURE_LEN_V4)?;
    m.add("FEATURE_LENGTH_V5", FEATURE_LEN_V5)?;
    m.add("FEATURE_VERSIONS", vec![1u32, 2u32, 3u32, 4u32, 5u32])?;
    m.add("IDENTITY_LENGTH", features_v4::N_IDS)?;
    m.add("IDENTITY_IDS_PER_SLOT", features_v4::IDS_PER_SLOT)?;
    m.add_function(wrap_pyfunction!(state_identity, m)?)?;
    m.add_function(wrap_pyfunction!(identity_vocab_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(identity_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(identity_vocab_hash, m)?)?;
    m.add_function(wrap_pyfunction!(feature_length, m)?)?;
    m.add_class::<PyEvaluator>()?;
    m.add_class::<PyPolicyPrior>()?;
    m.add_function(wrap_pyfunction!(policy_actions, m)?)?;
    m.add_function(wrap_pyfunction!(policy_action_features, m)?)?;
    m.add_function(wrap_pyfunction!(state_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(state_tokens_str, m)?)?;
    m.add_function(wrap_pyfunction!(state_hidden_targets, m)?)?;
    m.add_function(wrap_pyfunction!(tokens_bench, m)?)?;
    m.add_function(wrap_pyfunction!(token_layout, m)?)?;
    m.add_function(wrap_pyfunction!(token_constants, m)?)?;
    m.add("N_POLICY_ACTIONS", N_ACTIONS)?;
    m.add("N_POLICY_AFEAT", N_AFEAT)?;
    m.add("N_POLICY_EMB", N_EMB)?;
    m.add_class::<PyNetSearch>()?;
    m.add_class::<PyState>()?;
    m.add_class::<PySide>()?;
    m.add_class::<PySideConditions>()?;
    m.add_class::<PyVolatileStatusDurations>()?;
    m.add_class::<PyPokemon>()?;
    m.add_class::<PyMove>()?;
    m.add_class::<PyStateInstructions>()?;
    m.add_class::<PyInstruction>()?;
    Ok(())
}
