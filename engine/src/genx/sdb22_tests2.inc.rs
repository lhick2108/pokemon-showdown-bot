// sdb.22 (589.68.4): PP tracking and the "quick" mechanics fixes. Included in the `tests` module.

fn pp_decrements(outcomes: &[StateInstructions]) -> Vec<i8> {
    outcomes
        .iter()
        .flat_map(|o| o.instruction_list.iter())
        .filter_map(|i| match i {
            Instruction::DecrementPP(d) => Some(d.amount),
            _ => None,
        })
        .collect()
}

fn set_move_pp(side: &mut Side, id: Choices, pp: i8, max_pp: i8) {
    set_move(side, PokemonMoveIndex::M0, id, pp);
    side.get_active().moves[&PokemonMoveIndex::M0].max_pp = max_pp;
}

fn tackle_outcomes(state: &mut State) -> Vec<StateInstructions> {
    use_move(
        state,
        SideReference::SideOne,
        Choices::TACKLE,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    )
}

#[test]
fn test_sdb22_pp_is_tracked_at_any_value_when_max_pp_is_known() {
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::TACKLE, 20, 32);
    let outcomes = tackle_outcomes(&mut state);
    assert!(!pp_decrements(&outcomes).is_empty());
    assert!(pp_decrements(&outcomes).iter().all(|a| *a == 1));

    // unknown max PP keeps the legacy "only below 10" behaviour
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::TACKLE, 20, 0);
    assert!(pp_decrements(&tackle_outcomes(&mut state)).is_empty());
    set_move_pp(&mut state.side_one, Choices::TACKLE, 5, 0);
    assert!(!pp_decrements(&tackle_outcomes(&mut state)).is_empty());
}

#[test]
fn test_sdb22_pressure_never_pushes_known_pp_below_zero() {
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::TACKLE, 1, 32);
    state.side_two.get_active().ability = Abilities::PRESSURE;
    assert!(pp_decrements(&tackle_outcomes(&mut state)).iter().all(|a| *a == 1));
    set_move_pp(&mut state.side_one, Choices::TACKLE, 10, 32);
    assert!(pp_decrements(&tackle_outcomes(&mut state)).iter().all(|a| *a == 2));
}

#[test]
fn test_sdb22_no_pp_is_spent_on_the_continuation_of_a_locked_move() {
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::OUTRAGE, 20, 16);
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::OUTRAGE,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    assert!(!pp_decrements(&outcomes).is_empty());
    state
        .side_one
        .volatile_statuses
        .insert(PokemonVolatileStatus::LOCKEDMOVE);
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::OUTRAGE,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    assert!(pp_decrements(&outcomes).is_empty());
}

#[test]
fn test_sdb22_leppa_berry_is_capped_at_max_pp() {
    for (max_pp, expected) in [(24, 10), (8, 8), (0, 10)] {
        let mut state = State::default();
        set_move_pp(&mut state.side_one, Choices::TACKLE, 0, max_pp);
        state.side_one.get_active().item = Items::LEPPABERRY;
        let mut instr = StateInstructions::default();
        item_end_of_turn(&mut state, &SideReference::SideOne, &mut instr);
        assert_eq!(
            state.side_one.get_active().moves[&PokemonMoveIndex::M0].pp,
            expected,
            "max_pp {max_pp}"
        );
    }
}

#[test]
fn test_sdb22_move_serialization_is_backward_compatible() {
    use crate::state::Move as StateMove;
    let old = StateMove::deserialize("TACKLE;false;12");
    assert_eq!((old.pp, old.max_pp), (12, 0));
    assert_eq!(old.serialize(), "TACKLE;false;12");
    let new = StateMove::deserialize("TACKLE;false;12;56");
    assert_eq!((new.pp, new.max_pp), (12, 56));
    assert_eq!(new.serialize(), "TACKLE;false;12;56");
}

#[test]
fn test_sdb22_take_heart_boosts_and_cures_status() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::TAKEHEART, 16);
    state.side_one.get_active().status = PokemonStatus::BURN;
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::TAKEHEART,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    assert_eq!(outcomes.len(), 1);
    let list = &outcomes[0].instruction_list;
    let boosts: Vec<_> = list
        .iter()
        .filter_map(|i| match i {
            Instruction::Boost(b) => Some((b.stat, b.amount)),
            _ => None,
        })
        .collect();
    assert!(boosts.contains(&(PokemonBoostableStat::SpecialAttack, 1)));
    assert!(boosts.contains(&(PokemonBoostableStat::SpecialDefense, 1)));
    assert!(list
        .iter()
        .any(|i| matches!(i, Instruction::ChangeStatus(c) if c.new_status == PokemonStatus::NONE)));
}

fn life_orb_recoil(ability: Abilities, mv: Choices) -> bool {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, mv, 16);
    state.side_one.get_active().ability = ability;
    state.side_one.get_active().item = Items::LIFEORB;
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        mv,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    outcomes.iter().any(|o| {
        o.instruction_list.iter().any(|i| {
            matches!(i, Instruction::Heal(h)
                if h.side_ref == SideReference::SideOne && h.heal_amount < 0)
        })
    })
}

#[test]
fn test_sdb22_sheer_force_suppresses_life_orb_recoil_only_when_it_applies() {
    // Flamethrower has a burn secondary -> Sheer Force applies, no Life Orb recoil
    assert!(!life_orb_recoil(Abilities::SHEERFORCE, Choices::FLAMETHROWER));
    // Earthquake has no secondary -> recoil as usual
    assert!(life_orb_recoil(Abilities::SHEERFORCE, Choices::EARTHQUAKE));
    // control: without Sheer Force Flamethrower recoils
    assert!(life_orb_recoil(Abilities::NONE, Choices::FLAMETHROWER));
}

#[test]
fn test_sdb22_ko_with_giga_impact_keeps_recharge_through_the_replacement_call() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::GIGAIMPACT, 8);
    state.side_two.get_active().hp = 1;
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::GIGAIMPACT,
        PokemonMoveIndex::M0,
        Choices::SPLASH,
    );
    assert!(applies_status(&outcomes, PokemonVolatileStatus::MUSTRECHARGE));

    // the follow-up call `(switch, none)` must not eat the recharge of the side that has to wait
    let mut state = State::default();
    state.side_one.get_active().hp = 0;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::MUSTRECHARGE);
    let outcomes = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Switch(PokemonIndex::P1),
        &MoveChoice::None,
        false,
    );
    assert!(!outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(
        i,
        Instruction::RemoveVolatileStatus(r) if r.volatile_status == PokemonVolatileStatus::MUSTRECHARGE
    ))));
}

#[test]
fn test_sdb22_protosynthesis_activates_when_sun_starts_with_the_mon_already_out() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::SUNNYDAY, 8);
    state.side_two.get_active().ability = Abilities::PROTOSYNTHESIS;
    let outcomes = generate_instructions_from_move_pair(
        &mut state,
        &MoveChoice::Move(PokemonMoveIndex::M0),
        &MoveChoice::None,
        false,
    );
    assert!(outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(
        i,
        Instruction::ApplyVolatileStatus(a)
            if a.side_ref == SideReference::SideTwo
                && matches!(a.volatile_status,
                    PokemonVolatileStatus::PROTOSYNTHESISATK
                    | PokemonVolatileStatus::PROTOSYNTHESISDEF
                    | PokemonVolatileStatus::PROTOSYNTHESISSPA
                    | PokemonVolatileStatus::PROTOSYNTHESISSPD
                    | PokemonVolatileStatus::PROTOSYNTHESISSPE)
    ))));
}

#[test]
fn test_sdb22_disguise_is_not_busted_by_an_immune_hit() {
    let busts = |mv: Choices| -> bool {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, mv, 16);
        let d = state.side_two.get_active();
        d.id = PokemonName::MIMIKYU;
        d.ability = Abilities::DISGUISE;
        d.types = (PokemonType::GHOST, PokemonType::FAIRY);
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            mv,
            PokemonMoveIndex::M0,
            Choices::SPLASH,
        );
        outcomes.iter().any(|o| {
            o.instruction_list
                .iter()
                .any(|i| matches!(i, Instruction::FormeChange(_)))
        })
    };
    assert!(!busts(Choices::DRAGONCLAW)); // Dragon -> Fairy is immune
    assert!(busts(Choices::SHADOWCLAW)); // a damaging hit busts it
}

#[test]
fn test_sdb22_fixed_damage_moves_do_not_get_through_protect() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::SEISMICTOSS, 16);
    state.side_two.volatile_statuses.insert(PokemonVolatileStatus::PROTECT);
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::SEISMICTOSS, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(!damages(&outcomes, SideReference::SideTwo));
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::SEISMICTOSS, 16);
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::SEISMICTOSS, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(damages(&outcomes, SideReference::SideTwo));
}

#[test]
fn test_sdb22_protean_retypes_a_dual_type_mon_to_the_pure_move_type_once() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::WATERGUN, 16);
    let a = state.side_one.get_active();
    a.ability = Abilities::PROTEAN;
    a.types = (PokemonType::WATER, PokemonType::DARK);
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::WATERGUN, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(i, Instruction::ChangeType(_)))));
    // already pure Water, or already used this switch-in: no change
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::WATERGUN, 16);
    let a = state.side_one.get_active();
    a.ability = Abilities::PROTEAN;
    a.types = (PokemonType::WATER, PokemonType::TYPELESS);
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::WATERGUN, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(!outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(i, Instruction::ChangeType(_)))));
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::WATERGUN, 16);
    let a = state.side_one.get_active();
    a.ability = Abilities::PROTEAN;
    a.types = (PokemonType::FIRE, PokemonType::TYPELESS);
    state.side_one.volatile_statuses.insert(PokemonVolatileStatus::TYPECHANGE);
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::WATERGUN, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(!outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(i, Instruction::ChangeType(_)))));
}

#[test]
fn test_sdb22_charge_is_used_up_by_an_electric_move_and_electromorphosis_gives_it() {
    let removes_charge = |mv: Choices| -> bool {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, mv, 16);
        state.side_one.volatile_statuses.insert(PokemonVolatileStatus::CHARGE);
        let outcomes = use_move(&mut state, SideReference::SideOne, mv, PokemonMoveIndex::M0, Choices::SPLASH);
        outcomes.iter().any(|o| o.instruction_list.iter().any(|i| matches!(i, Instruction::RemoveVolatileStatus(r) if r.volatile_status == PokemonVolatileStatus::CHARGE)))
    };
    assert!(removes_charge(Choices::THUNDERBOLT));
    assert!(!removes_charge(Choices::TACKLE));
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::TACKLE, 16);
    state.side_two.get_active().ability = Abilities::ELECTROMORPHOSIS;
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::TACKLE, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(applies_status(&outcomes, PokemonVolatileStatus::CHARGE));
}

#[test]
fn test_sdb22_a_failed_protect_and_a_woken_sleep_talker_still_spend_pp() {
    // second Protect in a row: the failure branch is split off early but still pays the PP
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::PROTECT, 16, 16);
    state.side_one.side_conditions.protect = 1;
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::PROTECT, PokemonMoveIndex::M0, Choices::SPLASH);
    assert!(outcomes.len() >= 2);
    for o in &outcomes {
        assert_eq!(pp_decrements(std::slice::from_ref(o)), vec![1]);
    }
    // a sleeping Sleep Talk user pays once, not twice
    let mut state = State::default();
    set_move_pp(&mut state.side_one, Choices::SLEEPTALK, 16, 16);
    state.side_one.get_active().status = PokemonStatus::SLEEP;
    state.side_one.get_active().sleep_turns = 0;
    let outcomes = use_move(&mut state, SideReference::SideOne, Choices::SLEEPTALK, PokemonMoveIndex::M0, Choices::SPLASH);
    for o in &outcomes {
        assert!(pp_decrements(std::slice::from_ref(o)).len() <= 1);
    }
}
