# State feature encoding (FEATURE_VERSION = 1)

`features(state, side)` in `src/features.rs` (Python: `poke_engine.state_features(state, side=1)`)
encodes a `State` as a flat `f32` vector of fixed length **738**
(`poke_engine.FEATURE_LENGTH`). Training and inference MUST both use this
function; never re-implement it in Python. Any change to the layout or to the
meaning of a value requires bumping `FEATURE_VERSION`; weights files record the
version they were built for and the engine refuses to load a mismatch.
`poke_engine.state_feature_names()` returns the name of every index (same
order as below).

Everything is from the point of view of `side` ("own"); the other side is
"opp". The learned evaluators always use `side = SideOne` and predict
P(side one wins). For training data from side two's perspective, either use
`side=2` and flip the label, or use `side=1` of the same state.

```
[ own side block (363) ][ opp side block (363) ][ field block (12) ]  = 738
```

## Side block (363 = 6 x 53 + 45)

Six Pokemon slots, then side-level extras.

### Pokemon slots (6 x 53)

Slot 0 is the **active** Pokemon; slots 1..5 are the remaining team members in
team order. Missing members are fainted placeholders (all-zero hp/alive, default
stats). Indices below are within one slot (`own.p<slot>.<name>`):

| idx | name | value |
| --- | --- | --- |
| 0 | `hp` | `hp / maxhp`, clamped to [0, 1] |
| 1 | `alive` | 1 if `hp > 0` |
| 2-7 | `status.burn`, `.sleep`, `.freeze`, `.paralyze`, `.poison`, `.toxic` | one-hot (none = all zero) |
| 8-25 | `type.<t>` | multi-hot of the Pokemon's current types (`Pokemon.types`, as the engine tracks them) over normal, fire, water, electric, grass, ice, fighting, poison, ground, flying, psychic, bug, rock, ghost, dragon, dark, steel, fairy |
| 26-43 | `tera.<t>` | one-hot of its Tera type, same 18 types |
| 44 | `tera.stellar` | 1 if Tera type is Stellar |
| 45 | `terastallized` | 1 if already terastallized |
| 46 | `has_item` | 1 if holding an item |
| 47 | `maxhp` | `maxhp / 500` |
| 48-52 | `attack`, `defense`, `special_attack`, `special_defense`, `speed` | stat `/ 500` (battle stats, not base stats) |

Slot `k` starts at index `53 * k`.

### Side extras (45), starting at index 318 within the side block

| idx (rel.) | names | value |
| --- | --- | --- |
| 0-6 | `boost.attack`, `.defense`, `.special_attack`, `.special_defense`, `.speed`, `.accuracy`, `.evasion` | stage `/ 6`, in [-1, 1] (active Pokemon) |
| 7-26 | `vol.<v>` for leechseed, substitute, confusion, taunt, encore, yawn, torment, disable, aquaring, ingrain, curse, nightmare, saltcure, syrupbomb, healblock, partiallytrapped, perish (any of PERISH4..1), focusenergy, mustrecharge, lockedmove | 0/1 (active Pokemon) |
| 27-40 | `sc.<c>` for stealth_rock, spikes, toxic_spikes, sticky_web, reflect, light_screen, aurora_veil, safeguard, tailwind, mist, lucky_chant, healing_wish, lunar_dance, protect | `stealth_rock`..: 0/1 for presence (`> 0`); `spikes` = layers `/ 3`; `toxic_spikes` = layers `/ 2` |
| 41-44 | `misc.wish_pending`, `.future_sight_pending`, `.force_switch`, `.alive_fraction` | 0/1, 0/1, 0/1, `alive Pokemon / 6` |

## Field block (12), starting at index 726

| idx (rel.) | names | value |
| --- | --- | --- |
| 0-6 | `field.weather.sun`, `.rain`, `.sand`, `.hail`, `.snow`, `.harshsun`, `.heavyrain` | one-hot of the effective weather (`State::get_weather`, accounts for Air Lock / Cloud Nine) |
| 7-10 | `field.terrain.electric`, `.psychic`, `.misty`, `.grassy` | one-hot of the effective terrain |
| 11 | `field.trick_room` | 0/1 |

## Not encoded (yet)

Species / ability / item identity, move sets and PP, last used move, turn
counters (screen/weather duration), substitute HP. Adding any of these is a
`FEATURE_VERSION` bump.
