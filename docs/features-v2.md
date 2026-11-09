# Value-net features, version 2 (bead 589.42)

Why: the v1 vector (738 values, `engine/FEATURES.md`) knows only HP, status, types, boosts, stats and a
has-item flag. A net (and the stock `evaluate()`) cannot tell Life Orb from Leftovers, cannot see recovery,
setup, priority, hazard or status moves, and has no PP or speed-order information. Version 2 adds those.

## Layout

`features_v2 = features_v1 (738) ++ extras (1453)`, total `FEATURE_LENGTH_V2 = 2191`.
`.pev` headers carry the feature version, so v1 nets load and run unchanged; the engine default
(`FEATURE_VERSION`, `state_features(state)`) stays 1. v2 is selected explicitly
(`state_features(state, side, 2)`, `train --features v2`).

Extras (code: `engine/src/features_v2.rs`, names via `state_feature_names(2)`):

1. 12 Pokemon blocks (own slots 0..5, opp slots 0..5, slot 0 = active, rest in team order), 116 values each:
   * item one-hot (31): 27 common randbats items, type-boost items, other, none, unknown
   * ability one-hot (50): 48 impactful abilities, other, none
   * move flags over the four moves (30): recovery, rest, wish, drain, setup per stat (atk/def/spa/spd/spe),
     priority attack, fake out, pivot, hazard setter, hazard removal, inflicts sleep/paralysis/burn/poison,
     phaze, protect, substitute, screens, leech seed, disruption (taunt/encore/...), explosion, recoil,
     cure status, field setter (weather/trick room/tailwind), trick, sleep talk
   * move scalars (7): best recovery fraction, setup stages, damaging / status move counts, best STAB and
     non-STAB base power (/150), distinct attack types
2. 2 side blocks (own, opp), 17 values each: reflect / light screen / aurora veil / tailwind turns left, and the
   active-vs-active block: moves first (speed incl. boosts, paralysis, scarf, tailwind, trick room), speed log
   ratio, best type effectiveness, super effective available, rough best-move damage as a fraction of the
   target's max HP (level, stats, boosts, STAB, burn, Life Orb / Choice item, accuracy; no weather or screens),
   can KO / 2HKO, KO and outspeeds, all moves immune, heal-minus-incoming-damage ("outheal", the stall signal),
   PP of the best recovery move, exhausted moves, tera still available.
3. Field (3): weather, terrain and trick room turns left.

Already in v1 and kept: hazards, screens (flags), wish pending, alive count, tera type / used, boosts, volatiles.

## Fast path

Everything is emitted through one `Sink` trait, so the dense encoder and the sparse MLP path (first-layer weight
rows added straight from the state) share one definition and agree by construction (test:
`fast_path_matches_reference`, <= 1e-5, hidden 8/16/32/64, both versions). The Pokemon blocks depend only on
types, tera, item, ability and the four move ids, so they are cached per (side, team index, slot) in the existing
static cache (key: two u64 words). The side blocks need the opposing active and are recomputed per evaluation
(two small damage assessments of four moves each).

## Training

`python -m showdown_bot.train --features v2 ...` re-featurizes the recorded State strings (no re-recording).
Features are held as float16 (the stock evaluation of a residual net is kept in a separate float32 array), so
3,000 games take about 2.5 GB instead of 5+ GB. `--augment-*` is v1 only.
