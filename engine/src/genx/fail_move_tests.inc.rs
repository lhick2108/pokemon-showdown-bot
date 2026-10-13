// 589.39: moves that must fail / be immune. Included at the end of the `tests` module of
// generate_instructions.rs (shares its helpers: set_move, use_move, changes_status, ...).

fn damages(outcomes: &[StateInstructions], side: SideReference) -> bool {
    outcomes.iter().any(|o| {
        o.instruction_list.iter().any(
            |i| matches!(i, Instruction::Damage(d) if d.side_ref == side && d.damage_amount > 0),
        )
    })
}

fn heals(outcomes: &[StateInstructions]) -> bool {
    outcomes.iter().any(|o| {
        o.instruction_list
            .iter()
            .any(|i| matches!(i, Instruction::Heal(_)))
    })
}

fn has_wish_change(outcomes: &[StateInstructions]) -> bool {
    outcomes.iter().any(|o| {
        o.instruction_list
            .iter()
            .any(|i| matches!(i, Instruction::ChangeWish(_)))
    })
}

#[test]
fn test_fake_out_only_works_on_the_first_turn_out() {
    for (last, works) in [
        (LastUsedMove::Switch(PokemonIndex::P0), true),
        (LastUsedMove::None, true),
        (LastUsedMove::Move(PokemonMoveIndex::M1), false),
        (LastUsedMove::Move(PokemonMoveIndex::M0), false),
    ] {
        for id in [Choices::FAKEOUT, Choices::FIRSTIMPRESSION] {
            let mut state = State::default();
            state.use_last_used_move = true;
            set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 16);
            set_move(&mut state.side_one, PokemonMoveIndex::M1, Choices::KNOCKOFF, 16);
            state.side_one.last_used_move = last;
            let outcomes = use_move(
                &mut state,
                SideReference::SideOne,
                id,
                PokemonMoveIndex::M0,
                Choices::TACKLE,
            );
            assert_eq!(
                damages(&outcomes, SideReference::SideTwo),
                works,
                "{id:?} {last:?}"
            );
        }
    }
}

#[test]
fn test_fake_out_is_spent_by_a_move_after_a_switch_in() {
    // Persian switches in (turn 19), uses Knock Off (turn 20): Fake Out on turn 21 fails.
    let mut state = State::default();
    state.use_last_used_move = true;
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::FAKEOUT, 16);
    set_move(&mut state.side_one, PokemonMoveIndex::M1, Choices::KNOCKOFF, 16);
    state.side_one.last_used_move = LastUsedMove::Switch(PokemonIndex::P0);
    let mut knock_state = state.clone();
    let knock = use_move(
        &mut knock_state,
        SideReference::SideOne,
        Choices::KNOCKOFF,
        PokemonMoveIndex::M1,
        Choices::TACKLE,
    );
    assert!(damages(&knock, SideReference::SideTwo));
    state.apply_instructions(&knock[0].instruction_list);
    assert_eq!(
        state.side_one.last_used_move,
        LastUsedMove::Move(PokemonMoveIndex::M1)
    );
    let fake_out = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::FAKEOUT,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!damages(&fake_out, SideReference::SideTwo));
}

#[test]
fn test_powder_moves_fail_into_grass_overcoat_and_safety_goggles() {
    for id in [
        Choices::SPORE,
        Choices::STUNSPORE,
        Choices::SLEEPPOWDER,
        Choices::POISONPOWDER,
    ] {
        let build = |tweak: &dyn Fn(&mut State)| {
            let mut state = State::default();
            set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 16);
            state.side_two.get_active().types = (PokemonType::NORMAL, PokemonType::TYPELESS);
            state.side_two.get_active().ability = Abilities::NONE;
            state.side_two.get_active().item = Items::NONE;
            tweak(&mut state);
            use_move(
                &mut state,
                SideReference::SideOne,
                id,
                PokemonMoveIndex::M0,
                Choices::TACKLE,
            )
        };
        assert!(changes_status(&build(&|_| {})), "{id:?} plain target");
        assert!(
            !changes_status(&build(&|s| s.side_two.get_active().types =
                (PokemonType::GRASS, PokemonType::POISON))),
            "{id:?} grass"
        );
        assert!(
            !changes_status(&build(&|s| s.side_two.get_active().ability = Abilities::OVERCOAT)),
            "{id:?} overcoat"
        );
        assert!(
            !changes_status(&build(&|s| s.side_two.get_active().item = Items::SAFETYGOGGLES)),
            "{id:?} safety goggles"
        );
    }
}

#[test]
fn test_wish_fails_while_a_wish_is_pending() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::WISH, 16);
    let set = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::WISH,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(has_wish_change(&set));
    let mut pending = State::default();
    set_move(&mut pending.side_one, PokemonMoveIndex::M0, Choices::WISH, 16);
    pending.side_one.wish = (1, 50);
    let again = use_move(
        &mut pending,
        SideReference::SideOne,
        Choices::WISH,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!has_wish_change(&again));
    assert_eq!(pending.side_one.wish, (1, 50));
}

#[test]
fn test_status_moves_fail_into_an_already_statused_target() {
    let moves = [
        Choices::THUNDERWAVE,
        Choices::TOXIC,
        Choices::WILLOWISP,
        Choices::SPORE,
        Choices::SLEEPPOWDER,
        Choices::STUNSPORE,
        Choices::GLARE,
        Choices::HYPNOSIS,
        Choices::POISONPOWDER,
    ];
    for id in moves {
        for status in [
            PokemonStatus::BURN,
            PokemonStatus::PARALYZE,
            PokemonStatus::SLEEP,
            PokemonStatus::POISON,
            PokemonStatus::TOXIC,
            PokemonStatus::FREEZE,
        ] {
            let mut state = State::default();
            set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 32);
            state.side_two.get_active().status = status;
            let outcomes = use_move(
                &mut state,
                SideReference::SideOne,
                id,
                PokemonMoveIndex::M0,
                Choices::TACKLE,
            );
            assert!(!changes_status(&outcomes), "{id:?} into {status:?}");
        }
    }
}

#[test]
fn test_yawn_fails_into_a_statused_target_and_a_second_yawn() {
    for status in [
        PokemonStatus::BURN,
        PokemonStatus::PARALYZE,
        PokemonStatus::SLEEP,
        PokemonStatus::POISON,
    ] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::YAWN, 32);
        state.side_two.get_active().status = status;
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            Choices::YAWN,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        assert!(
            !applies_status(&outcomes, PokemonVolatileStatus::YAWN),
            "{status:?}"
        );
    }
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::YAWN, 32);
    let ok = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::YAWN,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(applies_status(&ok, PokemonVolatileStatus::YAWN));
    let mut yawned = State::default();
    set_move(&mut yawned.side_one, PokemonMoveIndex::M0, Choices::YAWN, 32);
    yawned
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::YAWN);
    let second = use_move(
        &mut yawned,
        SideReference::SideOne,
        Choices::YAWN,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!applies_status(&second, PokemonVolatileStatus::YAWN));
}

#[test]
fn test_recovery_moves_do_nothing_at_full_hp() {
    for id in [
        Choices::RECOVER,
        Choices::ROOST,
        Choices::SOFTBOILED,
        Choices::MOONLIGHT,
        Choices::SYNTHESIS,
        Choices::SLACKOFF,
        Choices::MILKDRINK,
        Choices::SHOREUP,
    ] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 16);
        let full = use_move(
            &mut state,
            SideReference::SideOne,
            id,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        assert!(!heals(&full), "{id:?} at full hp");
        assert!(
            !full.iter().any(|o| o
                .instruction_list
                .iter()
                .any(|i| matches!(i, Instruction::ApplyVolatileStatus(_)))),
            "{id:?} at full hp applies a volatile"
        );
        state.side_one.get_active().hp = 10;
        let hurt = use_move(
            &mut state,
            SideReference::SideOne,
            id,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        assert!(heals(&hurt), "{id:?} when hurt");
    }
}

#[test]
fn test_status_moves_are_blocked_by_substitute() {
    for id in [
        Choices::THUNDERWAVE,
        Choices::TOXIC,
        Choices::WILLOWISP,
        Choices::SPORE,
        Choices::GLARE,
        Choices::YAWN,
        Choices::LEECHSEED,
        Choices::CONFUSERAY,
    ] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 32);
        state
            .side_two
            .volatile_statuses
            .insert(PokemonVolatileStatus::SUBSTITUTE);
        state.side_two.substitute_health = 25;
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            id,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        let applied = changes_status(&outcomes)
            || applies_status(&outcomes, PokemonVolatileStatus::YAWN)
            || applies_status(&outcomes, PokemonVolatileStatus::LEECHSEED)
            || applies_status(&outcomes, PokemonVolatileStatus::CONFUSION);
        assert!(!applied, "{id:?} into a substitute");
    }
}

#[test]
fn test_consecutive_protect_success_chance_is_a_third_per_use() {
    for (counter, expected) in [(0i8, 1.0f32), (1, 1.0 / 3.0), (2, 1.0 / 9.0)] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::PROTECT, 16);
        state.side_one.side_conditions.protect = counter;
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            Choices::PROTECT,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        let success: f32 = outcomes
            .iter()
            .filter(|o| {
                o.instruction_list
                    .iter()
                    .any(|i| matches!(i, Instruction::ApplyVolatileStatus(_)))
            })
            .map(|o| o.percentage)
            .sum();
        assert!(
            (success / 100.0 - expected).abs() < 1e-3,
            "counter {counter}: {success}"
        );
    }
}

#[test]
fn test_defog_lowers_evasion_and_triggers_competitive_and_defiant() {
    for (ability, stat) in [
        (Abilities::COMPETITIVE, PokemonBoostableStat::SpecialAttack),
        (Abilities::DEFIANT, PokemonBoostableStat::Attack),
    ] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::DEFOG, 16);
        state.side_two.get_active().ability = ability;
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            Choices::DEFOG,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        let boosts: Vec<_> = outcomes[0]
            .instruction_list
            .iter()
            .filter_map(|i| match i {
                Instruction::Boost(b) => Some((b.side_ref, b.stat, b.amount)),
                _ => None,
            })
            .collect();
        assert!(
            boosts.contains(&(SideReference::SideTwo, PokemonBoostableStat::Evasion, -1)),
            "{ability:?}: {boosts:?}"
        );
        assert!(
            boosts.contains(&(SideReference::SideTwo, stat, 2)),
            "{ability:?}: {boosts:?}"
        );
    }
}

#[test]
fn test_defog_skips_the_evasion_drop_through_a_substitute_but_still_clears_hazards() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::DEFOG, 16);
    state.side_one.side_conditions.stealth_rock = 1;
    state
        .side_two
        .volatile_statuses
        .insert(PokemonVolatileStatus::SUBSTITUTE);
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::DEFOG,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!outcomes[0]
        .instruction_list
        .iter()
        .any(|i| matches!(i, Instruction::Boost(_))));
    state.apply_instructions(&outcomes[0].instruction_list);
    assert_eq!(state.side_one.side_conditions.stealth_rock, 0);
}

#[test]
fn test_defog_keeps_the_users_own_screens_and_clears_the_targets() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::DEFOG, 16);
    state.side_one.side_conditions.reflect = 3;
    state.side_one.side_conditions.light_screen = 3;
    state.side_one.side_conditions.aurora_veil = 3;
    state.side_two.side_conditions.reflect = 3;
    state.side_two.side_conditions.light_screen = 3;
    state.side_two.side_conditions.aurora_veil = 3;
    state.side_two.side_conditions.safeguard = 3;
    state.side_two.side_conditions.mist = 3;
    let outcomes = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::DEFOG,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    state.apply_instructions(&outcomes[0].instruction_list);
    assert_eq!(state.side_one.side_conditions.reflect, 3);
    assert_eq!(state.side_one.side_conditions.light_screen, 3);
    assert_eq!(state.side_one.side_conditions.aurora_veil, 3);
    assert_eq!(state.side_two.side_conditions.reflect, 0);
    assert_eq!(state.side_two.side_conditions.light_screen, 0);
    assert_eq!(state.side_two.side_conditions.aurora_veil, 0);
    assert_eq!(state.side_two.side_conditions.safeguard, 0);
    assert_eq!(state.side_two.side_conditions.mist, 0);
}

#[test]
fn test_normal_and_fighting_moves_do_nothing_to_ghosts() {
    for id in [
        Choices::FACADE,
        Choices::TACKLE,
        Choices::BODYSLAM,
        Choices::CLOSECOMBAT,
    ] {
        let mut state = State::default();
        set_move(&mut state.side_one, PokemonMoveIndex::M0, id, 16);
        state.side_two.get_active().types = (PokemonType::GHOST, PokemonType::TYPELESS);
        let outcomes = use_move(
            &mut state,
            SideReference::SideOne,
            id,
            PokemonMoveIndex::M0,
            Choices::TACKLE,
        );
        assert!(!damages(&outcomes, SideReference::SideTwo), "{id:?}");
    }
}

#[test]
fn test_poltergeist_needs_a_target_item() {
    let mut state = State::default();
    set_move(&mut state.side_one, PokemonMoveIndex::M0, Choices::POLTERGEIST, 8);
    state.side_two.get_active().types = (PokemonType::WATER, PokemonType::TYPELESS);
    state.side_two.get_active().item = Items::NONE;
    let none = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::POLTERGEIST,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!damages(&none, SideReference::SideTwo));
    state.side_two.get_active().item = Items::LEFTOVERS;
    let with_item = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::POLTERGEIST,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(damages(&with_item, SideReference::SideTwo));
    // ladder log 2697375078: Houndstone's Poltergeist into Snorlax (Normal) is a Ghost move
    state.side_two.get_active().types = (PokemonType::NORMAL, PokemonType::TYPELESS);
    let into_normal = use_move(
        &mut state,
        SideReference::SideOne,
        Choices::POLTERGEIST,
        PokemonMoveIndex::M0,
        Choices::TACKLE,
    );
    assert!(!damages(&into_normal, SideReference::SideTwo));
}

#[test]
fn test_sticky_web_counts_as_a_foe_drop_for_defiant_and_clear_body() {
    for (ability, expected) in [
        (Abilities::DEFIANT, Some((PokemonBoostableStat::Attack, 2))),
        (
            Abilities::COMPETITIVE,
            Some((PokemonBoostableStat::SpecialAttack, 2)),
        ),
        (Abilities::CLEARBODY, None),
    ] {
        let mut state = State::default();
        state.side_one.side_conditions.sticky_web = 1;
        state.side_one.pokemon[PokemonIndex::P1].ability = ability;
        state.side_one.pokemon[PokemonIndex::P1].types =
            (PokemonType::NORMAL, PokemonType::TYPELESS);
        let mut ins = StateInstructions::default();
        generate_instructions_from_switch(
            &mut state,
            PokemonIndex::P1,
            SideReference::SideOne,
            &mut ins,
        );
        match expected {
            Some((stat, amount)) => {
                assert_eq!(
                    state.side_one.get_boost_from_boost_enum(&stat),
                    amount,
                    "{ability:?}"
                );
                assert_eq!(state.side_one.speed_boost, -1, "{ability:?}");
            }
            None => assert_eq!(state.side_one.speed_boost, 0, "{ability:?}"),
        }
    }
}

#[test]
fn test_foe_caused_stat_drops_trigger_defiant_and_competitive() {
    for (ability, stat) in [
        (Abilities::DEFIANT, PokemonBoostableStat::Attack),
        (Abilities::COMPETITIVE, PokemonBoostableStat::SpecialAttack),
    ] {
        let mut state = State::default();
        state.side_two.get_active().ability = ability;
        let mut ins = StateInstructions::default();
        apply_boost_instruction(
            &mut state.side_two,
            &PokemonBoostableStat::Speed,
            &-1,
            &SideReference::SideOne,
            &SideReference::SideTwo,
            &mut ins,
        );
        assert_eq!(state.side_two.get_boost_from_boost_enum(&stat), 2);
    }
}
