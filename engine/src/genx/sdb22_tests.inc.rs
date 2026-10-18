// sdb.22 mechanics fixes (589.68.4). Included at the end of the `tests` module of
// generate_instructions.rs (shares its helpers: set_move, use_move, ...).

fn eot(state: &mut State) -> StateInstructions {
    let mut instructions = StateInstructions::default();
    add_end_of_turn_instructions(state, &mut instructions, &SideReference::SideOne);
    instructions
}

fn kinds(instr: &StateInstructions) -> Vec<&'static str> {
    instr
        .instruction_list
        .iter()
        .filter_map(|i| match i {
            Instruction::Heal(_) => Some("heal"),
            Instruction::Damage(_) => Some("damage"),
            _ => None,
        })
        .collect()
}

#[test]
fn test_sdb22_leftovers_heals_before_poison_and_burn_damage() {
    for status in [PokemonStatus::POISON, PokemonStatus::BURN, PokemonStatus::TOXIC] {
        let mut state = State::default();
        let active = state.side_one.get_active();
        active.hp = 50;
        active.item = Items::LEFTOVERS;
        active.status = status;
        let instr = eot(&mut state);
        assert_eq!(kinds(&instr), vec!["heal", "damage"], "{status:?}");
    }
}

#[test]
fn test_sdb22_leftovers_does_not_save_a_mon_the_status_tick_still_kills() {
    // 3 HP + 6 Leftovers = 9, poison deals 12 -> faints (same as Showdown: heal first, then tick)
    let mut state = State::default();
    let active = state.side_one.get_active();
    active.hp = 3;
    active.item = Items::LEFTOVERS;
    active.status = PokemonStatus::POISON;
    let instr = eot(&mut state);
    assert_eq!(state.side_one.get_active().hp, 0);
}

#[test]
fn test_sdb22_black_sludge_heals_poison_types_and_damages_others_without_overkill() {
    let mut state = State::default();
    let active = state.side_one.get_active();
    active.hp = 50;
    active.item = Items::BLACKSLUDGE;
    active.types = (PokemonType::POISON, PokemonType::TYPELESS);
    active.status = PokemonStatus::BURN;
    let instr = eot(&mut state);
    assert_eq!(kinds(&instr), vec!["heal", "damage"]);

    let mut state = State::default();
    let active = state.side_one.get_active();
    active.hp = 3;
    active.item = Items::BLACKSLUDGE;
    active.types = (PokemonType::NORMAL, PokemonType::TYPELESS);
    let instr = eot(&mut state);
    // damage is min(1/16, current hp) = 3, never the missing-HP amount
    assert_eq!(
        instr.instruction_list,
        vec![Instruction::Damage(DamageInstruction {
            side_ref: SideReference::SideOne,
            damage_amount: 3
        })]
    );
}

#[test]
fn test_sdb22_residual_order_wish_grassy_leftovers_leechseed_poison() {
    let mut state = State::default();
    state.terrain.terrain_type = Terrain::GRASSYTERRAIN;
    state.terrain.turns_remaining = 3;
    state.side_one.wish = (1, 20);
    let active = state.side_one.get_active();
    active.hp = 10;
    active.item = Items::LEFTOVERS;
    active.status = PokemonStatus::POISON;
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LEECHSEED);
    state.side_two.get_active().hp = 50;
    let instr = eot(&mut state);
    let order: Vec<(&str, i16)> = instr
        .instruction_list
        .iter()
        .filter_map(|i| match i {
            Instruction::Heal(h) if h.side_ref == SideReference::SideOne => {
                Some(("heal", h.heal_amount))
            }
            Instruction::Damage(d) if d.side_ref == SideReference::SideOne => {
                Some(("damage", d.damage_amount))
            }
            _ => None,
        })
        .collect();
    // wish 20, grassy 6, leftovers 6, leech seed 12 (sapped), poison 12
    assert_eq!(
        order,
        vec![("heal", 20), ("heal", 6), ("heal", 6), ("damage", 12), ("damage", 12)]
    );
}

#[test]
fn test_sdb22_grassy_terrain_does_not_heal_on_its_last_turn() {
    let mut state = State::default();
    state.terrain.terrain_type = Terrain::GRASSYTERRAIN;
    state.terrain.turns_remaining = 1;
    state.side_one.get_active().hp = 50;
    let instr = eot(&mut state);
    assert!(kinds(&instr).is_empty());

    state.terrain.terrain_type = Terrain::GRASSYTERRAIN;
    state.terrain.turns_remaining = 2;
    let instr = eot(&mut state);
    assert_eq!(kinds(&instr), vec!["heal"]);
}

#[test]
fn test_sdb22_hydration_cures_before_poison_damage() {
    let mut state = State::default();
    state.weather.weather_type = Weather::RAIN;
    state.weather.turns_remaining = 3;
    let active = state.side_one.get_active();
    active.ability = Abilities::HYDRATION;
    active.status = PokemonStatus::POISON;
    let instr = eot(&mut state);
    assert!(kinds(&instr).is_empty());
    assert_eq!(state.side_one.get_active().status, PokemonStatus::NONE);
}

#[test]
fn test_sdb22_solar_power_damage_is_not_clamped_to_missing_hp() {
    let mut state = State::default();
    state.weather.weather_type = Weather::SUN;
    state.weather.turns_remaining = 3;
    state.side_one.get_active().ability = Abilities::SOLARPOWER;
    let instr = eot(&mut state);
    assert_eq!(
        instr.instruction_list,
        vec![
            Instruction::DecrementWeatherTurnsRemaining,
            Instruction::Damage(DamageInstruction {
                side_ref: SideReference::SideOne,
                damage_amount: 12
            })
        ]
    );
}

fn healblocked_leftovers_state(duration: i8) -> State {
    let mut state = State::default();
    let active = state.side_one.get_active();
    active.hp = 50;
    active.item = Items::LEFTOVERS;
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::HEALBLOCK);
    state.side_one.volatile_status_durations.healblock = duration;
    state
}

#[test]
fn test_sdb22_heal_block_blocks_same_turn_leftovers_and_expires() {
    // the turn Heal Block ends (duration 1 -> 0) Leftovers (order 5) is still blocked
    let mut state = healblocked_leftovers_state(1);
    let instr = eot(&mut state);
    assert!(kinds(&instr).is_empty());
    assert!(!state
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::HEALBLOCK));
    assert_eq!(state.side_one.volatile_status_durations.healblock, 0);
    // next turn heals again
    let instr = eot(&mut state);
    assert_eq!(kinds(&instr), vec!["heal"]);
    // reversing restores the block
    let mut state = healblocked_leftovers_state(1);
    let before = format!("{:?}", state);
    let instr = eot(&mut state);
    state.reverse_instructions(&instr.instruction_list);
    assert_eq!(before, format!("{:?}", state));
}

#[test]
fn test_sdb22_heal_block_move_lasts_five_turns_psychic_noise_two() {
    for (mv, turns) in [(Choices::HEALBLOCK, 5), (Choices::PSYCHICNOISE, 2)] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, mv, 16);
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            mv,
            PokemonMoveIndex::M0,
            Choices::SPLASH,
        );
        // the 100% (secondary) branch carries the volatile
        let o = outcomes
            .iter()
            .find(|o| applies_status(std::slice::from_ref(*o), PokemonVolatileStatus::HEALBLOCK))
            .expect("heal block applied");
        state.apply_instructions(&o.instruction_list);
        assert_eq!(state.side_two.volatile_status_durations.healblock, turns);
        for tick in 1..=turns {
            let instr = eot(&mut state);
            let still = state
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::HEALBLOCK);
            assert_eq!(still, tick < turns, "{mv:?} tick {tick}");
        }
    }
}

#[test]
fn test_sdb22_throat_chop_two_turns_blocks_sound_not_through_protect_or_substitute() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::THROATCHOP, 16);
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::THROATCHOP,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    let o = outcomes
        .iter()
        .find(|o| applies_status(std::slice::from_ref(*o), PokemonVolatileStatus::THROATCHOP))
        .expect("throat chop applied");
    state.apply_instructions(&o.instruction_list);
    assert_eq!(state.side_two.volatile_status_durations.throatchop, 2);
    // sound moves are no longer selectable
    set_move(&mut state.side_two, PokemonMoveIndex::M0, Choices::HYPERVOICE, 16);
    set_move(&mut state.side_two, PokemonMoveIndex::M1, Choices::TACKLE, 16);
    let (_, s2) = state.get_all_options();
    assert!(!s2.contains(&MoveChoice::Move(PokemonMoveIndex::M0)));
    assert!(s2.contains(&MoveChoice::Move(PokemonMoveIndex::M1)));
    // two residuals later it is gone
    for _ in 0..2 {
        let instr = eot(&mut state);
    }
    assert!(!state
        .side_two
        .volatile_statuses
        .contains(&PokemonVolatileStatus::THROATCHOP));

    // Substitute absorbs it
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::THROATCHOP, 16);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    state.side_two.substitute_health = 100;
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::THROATCHOP,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    assert!(!applies_status(&outcomes, PokemonVolatileStatus::THROATCHOP));

    // Protect stops it
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::THROATCHOP, 16);
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::PROTECT);
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::THROATCHOP,
        PokemonMoveIndex::M0,
        Choices::PROTECT,
    );
    assert!(!applies_status(&outcomes, PokemonVolatileStatus::THROATCHOP));
}

#[test]
fn test_sdb22_old_state_strings_still_parse_and_new_fields_roundtrip() {
    use crate::state::VolatileStatusDurations;
    let old = VolatileStatusDurations::deserialize("0;0;1;0;2;0");
    assert_eq!((old.lockedmove, old.taunt, old.healblock, old.throatchop), (1, 2, 0, 0));
    assert_eq!(old.serialize(), "0;0;1;0;2;0");
    let new = VolatileStatusDurations::deserialize("0;0;0;0;0;0;4;1");
    assert_eq!((new.healblock, new.throatchop), (4, 1));
    assert_eq!(new.serialize(), "0;0;0;0;0;0;4;1");
}

#[test]
fn test_sdb22_legacy_healblock_without_duration_is_a_fresh_five_turns() {
    let mut state = State::default();
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::HEALBLOCK);
    let instr = eot(&mut state);
    assert_eq!(state.side_one.volatile_status_durations.healblock, 4);
    assert!(state
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::HEALBLOCK));
}
