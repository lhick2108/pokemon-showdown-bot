// sdb.28 mechanics fixes (72d, 589.68.4, 589.61). Included at the end of the `tests` module of
// generate_instructions.rs (shares its helpers: set_move, use_move, ...).

/// Every branch of a full turn, applied to a copy of `state`. Also checks that every branch
/// reverses back to the exact starting state (the search applies and undoes instructions).
fn sdb28_turn(state: &mut State, s1: MoveChoice, s2: MoveChoice) -> Vec<(f32, State)> {
    let before = state.serialize();
    let outcomes = generate_instructions_from_move_pair(state, &s1, &s2, false);
    assert_eq!(before, state.serialize(), "generation mutated the state");
    let mut out = vec![];
    for o in outcomes {
        let mut s = state.clone();
        s.apply_instructions(&o.instruction_list);
        let after = s.clone();
        s.reverse_instructions(&o.instruction_list);
        assert_eq!(before, s.serialize(), "branch does not reverse: {:?}", o);
        out.push((o.percentage, after));
    }
    out
}

fn sdb28_moves(state: &mut State, s1: Choices, s2: Choices) -> Vec<(f32, State)> {
    set_move(&mut state.side_one, PokemonMoveIndex::M0, s1, 16);
    set_move(&mut state.side_two, PokemonMoveIndex::M0, s2, 16);
    sdb28_turn(
        state,
        MoveChoice::Move(PokemonMoveIndex::M0),
        MoveChoice::Move(PokemonMoveIndex::M0),
    )
}

fn sdb28_choice_locked(side: &mut Side, item: Items) {
    let active = side.get_active();
    active.item = item;
    set_move(side, PokemonMoveIndex::M1, Choices::MOONBLAST, 16);
    set_move(side, PokemonMoveIndex::M2, Choices::PSYCHIC, 16);
    set_move(side, PokemonMoveIndex::M3, Choices::CALMMIND, 16);
    let active = side.get_active();
    active.moves.m1.disabled = true;
    active.moves.m2.disabled = true;
    active.moves.m3.disabled = true;
}

fn sdb28_any_disabled(state: &State, side: SideReference) -> bool {
    let s = match side {
        SideReference::SideOne => &state.side_one,
        SideReference::SideTwo => &state.side_two,
    };
    s.get_active_immutable().moves.into_iter().any(|m| m.disabled)
}

// ---------------------------------------------------------------- 72d: Choice lock release

#[test]
fn test_sdb28_knock_off_releases_choice_lock() {
    let mut state = State::default();
    sdb28_choice_locked(&mut state.side_two, Items::CHOICESPECS);
    let outcomes = sdb28_moves(&mut state, Choices::KNOCKOFF, Choices::TACKLE);
    let knocked: Vec<_> = outcomes
        .iter()
        .filter(|(_, s)| s.side_two.get_active_immutable().item == Items::NONE)
        .collect();
    assert!(!knocked.is_empty());
    for (_, s) in knocked {
        assert!(!sdb28_any_disabled(s, SideReference::SideTwo));
    }
}

#[test]
fn test_sdb28_knock_off_on_non_choice_item_keeps_other_disables() {
    let mut state = State::default();
    sdb28_choice_locked(&mut state.side_two, Items::LEFTOVERS);
    let outcomes = sdb28_moves(&mut state, Choices::KNOCKOFF, Choices::TACKLE);
    for (_, s) in outcomes {
        assert!(sdb28_any_disabled(&s, SideReference::SideTwo));
    }
}

#[test]
fn test_sdb28_gorilla_tactics_keeps_lock_after_knock_off() {
    let mut state = State::default();
    sdb28_choice_locked(&mut state.side_two, Items::CHOICEBAND);
    state.side_two.get_active().ability = Abilities::GORILLATACTICS;
    let outcomes = sdb28_moves(&mut state, Choices::KNOCKOFF, Choices::TACKLE);
    for (_, s) in outcomes {
        assert!(sdb28_any_disabled(&s, SideReference::SideTwo));
    }
}

#[test]
fn test_sdb28_thief_and_covet_release_choice_lock() {
    for mv in [Choices::THIEF, Choices::COVET] {
        let mut state = State::default();
        sdb28_choice_locked(&mut state.side_two, Items::CHOICESCARF);
        let outcomes = sdb28_moves(&mut state, mv, Choices::TACKLE);
        let stolen: Vec<_> = outcomes
            .iter()
            .filter(|(_, s)| s.side_one.get_active_immutable().item == Items::CHOICESCARF)
            .collect();
        assert!(!stolen.is_empty(), "{mv:?}");
        for (_, s) in stolen {
            assert!(!sdb28_any_disabled(s, SideReference::SideTwo), "{mv:?}");
        }
    }
}

#[test]
fn test_sdb28_trick_releases_the_users_lock_and_locks_the_target_on_its_next_move() {
    // side one holds Choice Scarf and Tricks it away: locked into Trick, then released
    let mut state = State::default();
    state.side_one.get_active().item = Items::CHOICESCARF;
    state.side_two.get_active().item = Items::LEFTOVERS;
    set_move(&mut state.side_one, PokemonMoveIndex::M1, Choices::TACKLE, 16);
    set_move(&mut state.side_two, PokemonMoveIndex::M1, Choices::CALMMIND, 16);
    let outcomes = sdb28_moves(&mut state, Choices::TRICK, Choices::SPLASH);
    assert!(!outcomes.is_empty());
    for (_, s) in &outcomes {
        assert_eq!(s.side_one.get_active_immutable().item, Items::LEFTOVERS);
        assert_eq!(s.side_two.get_active_immutable().item, Items::CHOICESCARF);
        assert!(!sdb28_any_disabled(s, SideReference::SideOne));
    }
    // the target is locked after its next move
    let mut s = outcomes[0].1.clone();
    let next = sdb28_turn(
        &mut s,
        MoveChoice::Move(PokemonMoveIndex::M1),
        MoveChoice::Move(PokemonMoveIndex::M0),
    );
    for (_, s) in next {
        let m = &s.side_two.get_active_immutable().moves;
        assert!(!m.m0.disabled && m.m1.disabled);
    }
}

#[test]
fn test_sdb28_choice_for_choice_trick_keeps_lock() {
    let mut state = State::default();
    sdb28_choice_locked(&mut state.side_two, Items::CHOICESPECS);
    state.side_one.get_active().item = Items::CHOICEBAND;
    let outcomes = sdb28_moves(&mut state, Choices::TRICK, Choices::TACKLE);
    for (_, s) in outcomes {
        assert_eq!(s.side_two.get_active_immutable().item, Items::CHOICEBAND);
        assert!(sdb28_any_disabled(&s, SideReference::SideTwo));
    }
}

#[test]
fn test_sdb28_corrosive_gas_removes_item_and_releases_lock() {
    let mut state = State::default();
    sdb28_choice_locked(&mut state.side_two, Items::CHOICESPECS);
    let outcomes = sdb28_moves(&mut state, Choices::CORROSIVEGAS, Choices::TACKLE);
    let removed: Vec<_> = outcomes
        .iter()
        .filter(|(_, s)| s.side_two.get_active_immutable().item == Items::NONE)
        .collect();
    assert!(!removed.is_empty());
    for (_, s) in removed {
        assert!(!sdb28_any_disabled(s, SideReference::SideTwo));
    }
}

// ---------------------------------------------------------------- Harvest

fn sdb28_harvest_mon(state: &mut State) {
    let a = state.side_one.get_active();
    a.ability = Abilities::HARVEST;
    a.base_ability = Abilities::HARVEST;
}

#[test]
fn test_sdb28_eating_sitrus_records_last_item_for_harvest_only() {
    let mut state = State::default();
    sdb28_harvest_mon(&mut state);
    state.side_one.get_active().item = Items::SITRUSBERRY;
    state.side_one.get_active().hp = 20;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    // eaten at end of turn, then regrown (50%) and eaten again at <= 1/2 HP, or not regrown
    let total: f32 = outcomes.iter().map(|(p, _)| p).sum();
    assert!((total - 100.0).abs() < 1e-3, "{total}");
    assert_eq!(outcomes.len(), 2);
    for (p, s) in &outcomes {
        let a = s.side_one.get_active_immutable();
        assert_eq!(a.item, Items::NONE);
        assert_eq!(a.last_item, Items::SITRUSBERRY);
        assert!((p - 50.0).abs() < 1e-3);
    }
    // one branch healed twice
    let hps: Vec<i16> = outcomes
        .iter()
        .map(|(_, s)| s.side_one.get_active_immutable().hp)
        .collect();
    assert!(hps.contains(&45) && hps.contains(&70), "{hps:?}");

    // without Harvest nothing is recorded
    let mut state = State::default();
    state.side_one.get_active().item = Items::SITRUSBERRY;
    state.side_one.get_active().hp = 40;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].1.side_one.get_active_immutable().last_item, Items::NONE);
}

#[test]
fn test_sdb28_harvest_regrows_in_sun_always_and_keeps_berry_above_half() {
    let mut state = State::default();
    sdb28_harvest_mon(&mut state);
    state.side_one.get_active().last_item = Items::SITRUSBERRY;
    state.weather.weather_type = Weather::SUN;
    state.weather.turns_remaining = 3;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    assert_eq!(outcomes.len(), 1);
    let a = outcomes[0].1.side_one.get_active_immutable();
    assert_eq!(a.item, Items::SITRUSBERRY);
    assert_eq!(a.last_item, Items::NONE);
}

#[test]
fn test_sdb28_knocked_off_berry_is_not_harvested() {
    let mut state = State::default();
    sdb28_harvest_mon(&mut state);
    state.side_one.get_active().item = Items::SITRUSBERRY;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::KNOCKOFF);
    for (_, s) in outcomes {
        let a = s.side_one.get_active_immutable();
        if a.item == Items::NONE {
            assert_eq!(a.last_item, Items::NONE);
        }
    }
}

#[test]
fn test_sdb28_two_harvest_sides_make_four_branches() {
    let mut state = State::default();
    sdb28_harvest_mon(&mut state);
    state.side_one.get_active().last_item = Items::LUMBERRY;
    let b = state.side_two.get_active();
    b.ability = Abilities::HARVEST;
    b.last_item = Items::SITRUSBERRY;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    assert_eq!(outcomes.len(), 4);
    for (p, _) in &outcomes {
        assert!((p - 25.0).abs() < 1e-3);
    }
}

#[test]
fn test_sdb28_unnerve_stops_berries() {
    let mut state = State::default();
    state.side_one.get_active().item = Items::SITRUSBERRY;
    state.side_one.get_active().hp = 20;
    state.side_two.get_active().ability = Abilities::UNNERVE;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    for (_, s) in outcomes {
        let a = s.side_one.get_active_immutable();
        assert_eq!(a.item, Items::SITRUSBERRY);
        assert_eq!(a.hp, 20);
    }
    // Lum does not block a status while the foe has Unnerve
    let mut state = State::default();
    state.side_one.get_active().item = Items::LUMBERRY;
    state.side_two.get_active().ability = Abilities::UNNERVE;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::WILLOWISP);
    assert!(outcomes
        .iter()
        .any(|(_, s)| s.side_one.get_active_immutable().status == PokemonStatus::BURN));
}

// ---------------------------------------------------------------- formes

fn sdb28_terapagos(state: &mut State) {
    let a = state.side_one.get_active();
    a.id = PokemonName::TERAPAGOS;
    a.ability = Abilities::TERASHIFT;
    a.base_ability = Abilities::TERASHIFT;
    a.level = 77;
    a.evs = (85, 85, 85, 85, 85, 85);
    let stats = a.calculate_stats_from_base_stats();
    a.maxhp = stats.0;
    a.hp = stats.0 - 30;
    a.attack = stats.1;
    a.defense = stats.2;
    a.special_attack = stats.3;
    a.special_defense = stats.4;
    a.speed = stats.5;
}

#[test]
fn test_sdb28_tera_shift_on_switch_in_keeps_damage_taken() {
    let mut state = State::default();
    sdb28_terapagos(&mut state);
    // put Terapagos on the bench and switch it in
    state.side_one.pokemon.pkmn.swap(0, 1);
    let outcomes = sdb28_turn(
        &mut state,
        MoveChoice::Switch(PokemonIndex::P1),
        MoveChoice::None,
    );
    assert_eq!(outcomes.len(), 1);
    let a = outcomes[0].1.side_one.get_active_immutable();
    assert_eq!(a.id, PokemonName::TERAPAGOSTERASTAL);
    assert_eq!(a.ability, Abilities::TERASHELL);
    assert_eq!(a.base_ability, Abilities::TERASHELL);
    // L77: Terapagos 265 HP, Terastal 273 HP; 30 damage is kept
    assert_eq!(a.maxhp, 273);
    assert_eq!(a.hp, 243);
}

#[cfg(feature = "terastallization")]
#[test]
fn test_sdb28_terastal_terapagos_becomes_stellar_on_tera_and_clears_weather() {
    let mut state = State::default();
    sdb28_terapagos(&mut state);
    let mut tmp = StateInstructions::default();
    permanent_forme_change(
        &mut state.side_one,
        &SideReference::SideOne,
        PokemonName::TERAPAGOSTERASTAL,
        Some(Abilities::TERASHELL),
        &mut tmp,
    );
    state.side_one.get_active().tera_type = PokemonType::STELLAR;
    state.weather.weather_type = Weather::RAIN;
    state.weather.turns_remaining = 3;
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::SPLASH, 16);
    set_move(&mut state.side_two, PokemonMoveIndex::M0, Choices::SPLASH, 16);
    let outcomes = sdb28_turn(
        &mut state,
        MoveChoice::MoveTera(PokemonMoveIndex::M0),
        MoveChoice::Move(PokemonMoveIndex::M0),
    );
    for (_, s) in outcomes {
        let a = s.side_one.get_active_immutable();
        assert_eq!(a.id, PokemonName::TERAPAGOSSTELLAR);
        assert_eq!(a.ability, Abilities::TERAFORMZERO);
        assert!(a.maxhp > 273);
        assert_eq!(a.maxhp - a.hp, 30);
        assert_eq!(s.weather.weather_type, Weather::NONE);
    }
}

// ---------------------------------------------------------------- Unburden

#[test]
fn test_sdb28_unburden_activates_when_the_item_is_lost() {
    let mut state = State::default();
    let a = state.side_one.get_active();
    a.ability = Abilities::UNBURDEN;
    a.item = Items::SITRUSBERRY;
    a.hp = 20;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    for (_, s) in &outcomes {
        assert!(s
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::UNBURDEN));
    }
    // knocked off
    let mut state = State::default();
    let a = state.side_one.get_active();
    a.ability = Abilities::UNBURDEN;
    a.item = Items::LEFTOVERS;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::KNOCKOFF);
    for (_, s) in &outcomes {
        let lost = s.side_one.get_active_immutable().item == Items::NONE;
        assert_eq!(
            lost,
            s.side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::UNBURDEN)
        );
    }
    // no item to begin with: nothing
    let mut state = State::default();
    state.side_one.get_active().ability = Abilities::UNBURDEN;
    let outcomes = sdb28_moves(&mut state, Choices::SPLASH, Choices::SPLASH);
    assert!(!outcomes[0]
        .1
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::UNBURDEN));
}

// ---------------------------------------------------------------- Cursed Body / Disable

#[test]
fn test_sdb28_cursed_body_disables_the_move_for_four_residuals() {
    let mut state = State::default();
    state.side_two.get_active().ability = Abilities::CURSEDBODY;
    state.side_two.get_active().hp = 500;
    state.side_two.get_active().maxhp = 500;
    let outcomes = sdb28_moves(&mut state, Choices::TACKLE, Choices::SPLASH);
    let disabled: Vec<_> = outcomes
        .iter()
        .filter(|(_, s)| {
            s.side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::DISABLE)
        })
        .collect();
    let p: f32 = disabled.iter().map(|(p, _)| p).sum();
    assert!((p - 30.0).abs() < 0.5, "{p}");
    let mut s = disabled[0].1.clone();
    assert!(s.side_one.get_active_immutable().moves.m0.disabled);
    // after this turn's residual 3 turns remain (encoded 3 * 4 + slot 0)
    assert_eq!(s.side_one.volatile_status_durations.disable, 12);
    for turn in 0..3 {
        assert!(s.side_one.get_active_immutable().moves.m0.disabled, "turn {turn}");
        let next = sdb28_turn(
            &mut s,
            MoveChoice::Move(PokemonMoveIndex::M1),
            MoveChoice::Move(PokemonMoveIndex::M0),
        );
        s = next[0].1.clone();
    }
    assert!(!s.side_one.get_active_immutable().moves.m0.disabled);
    assert!(!s
        .side_one
        .volatile_statuses
        .contains(&PokemonVolatileStatus::DISABLE));
    assert_eq!(s.side_one.volatile_status_durations.disable, 0);
}

#[test]
fn test_sdb28_disable_duration_serializes() {
    let mut state = State::default();
    state.side_one.volatile_status_durations.disable = 13;
    let ser = state.serialize();
    let back = State::deserialize(&ser);
    assert_eq!(back.side_one.volatile_status_durations.disable, 13);
    assert_eq!(back.serialize(), ser);
}

// ---------------------------------------------------------------- PP vs a fainted target

#[test]
fn test_sdb28_pp_is_spent_on_a_target_that_fainted_earlier_in_the_turn() {
    // side one uses Healing Wish first (faints), side two's Tackle has no target but costs PP
    let mut state = State::default();
    state.side_one.get_active().speed = 200;
    state.side_one.pokemon.pkmn[1].hp = 100;
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::HEALINGWISH, 16);
    set_move(&mut state.side_two, PokemonMoveIndex::M0, Choices::TACKLE, 35);
    state.side_two.get_active().moves.m0.max_pp = 56;
    let outcomes = sdb28_turn(
        &mut state,
        MoveChoice::Move(PokemonMoveIndex::M0),
        MoveChoice::Move(PokemonMoveIndex::M0),
    );
    for (_, s) in outcomes {
        assert_eq!(s.side_one.get_active_immutable().hp, 0);
        assert_eq!(s.side_two.get_active_immutable().moves.m0.pp, 34);
    }
}

#[test]
fn test_sdb28_last_item_serialization_is_backward_compatible() {
    let mut state = State::default();
    let plain = state.serialize();
    assert_eq!(State::deserialize(&plain).serialize(), plain);
    state.side_one.get_active().last_item = Items::SITRUSBERRY;
    let with = state.serialize();
    assert_ne!(with, plain);
    let back = State::deserialize(&with);
    assert_eq!(back.side_one.get_active_immutable().last_item, Items::SITRUSBERRY);
    assert_eq!(back.serialize(), with);
}
