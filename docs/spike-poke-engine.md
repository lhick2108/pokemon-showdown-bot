# poke-engine Gen 9 Python spike

## Verdict

**Yes for Gen 9 singles search, with battle-rule coverage caveats.** The cloned
`poke-engine` 0.0.48 source declares `gen9` and `terastallization = ["gen9"]`;
the latter is the feature to pass when building. The README still says only
Gens 4–8 are available, so that statement is stale relative to the manifest and
implementation. The engine describes itself as singles-only and explicitly
warns that it is not as complete as Pokémon Showdown.

The Python `Pokemon` binding carries `tera_type` and `terastallized` fields.
With the `terastallization` feature, the Rust state/action and damage code
handles Tera. The Gen 9 source also includes implementations for abilities such
as Protosynthesis, Quark Drive, Hadron Engine, and Orichalcum Pulse, and items
including Booster Energy, Clear Amulet, Covert Cloak, and Loaded Dice. This is
evidence of support for those mechanics, not an exhaustive Gen 9 parity audit.

## Build and install

The checkout was cloned with:

```powershell
git -c http.sslBackend=openssl clone --depth 1 https://github.com/pmariglia/poke-engine.git vendor/poke-engine
```

Reviewed commit: `c2dbbda7820137f938361106c1d1e7e1922fcb64`. Stage 4 must pin
this commit in a build script (with `cargo --locked`) and record the
dependency, because the clone above follows the moving `main` branch.

Foul Play's Makefile and requirements use `--features poke-engine/<generation>
--no-default-features`; its Gen 9 dependency selects
`poke-engine/terastallization`. The local-source equivalent was installed into
the project venv with this command (from the repository root):

```powershell
$env:CARGO_HOME = (Join-Path (Get-Location) '.cache\cargo')
$env:CARGO_TARGET_DIR = (Join-Path (Get-Location) '.cache\cargo-target')
$env:UV_CACHE_DIR = (Join-Path (Get-Location) '.cache\uv')
$env:Path = 'C:\Users\lhick\.cargo\bin;' + $env:Path
cmd.exe /d /c 'call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 && python -m uv pip install --python .venv\Scripts\python.exe --reinstall --no-cache --verbose --config-settings "build-args=--features poke-engine/terastallization --no-default-features" vendor\poke-engine\poke-engine-py'
```

Cargo's default Windows Schannel connection failed with
`SEC_E_NO_CREDENTIALS` while contacting crates.io. To complete the build in
this environment, the exact crates from `Cargo.lock` were downloaded and
checksum-verified into `.cache/cargo/vendor`; `.cache/cargo/config.toml` points
Cargo at that local directory in offline mode. These files are ignored build
artifacts. On a normal connected machine, the same install command should fetch
the locked crates directly.

The successful Maturin build produced and installed
`poke_engine-0.0.48-cp312-cp312-win_amd64.whl`. The required import check passed:

```text
python -m uv run python -c "import poke_engine; print(poke_engine.__file__)"
C:\Users\lhick\code\showdown-bot\.venv\Lib\site-packages\poke_engine\__init__.py
```

## Python API needed by Stage 4

- Build `State(side_one=Side(pokemon=[...]), side_two=Side(pokemon=[...]), ...)`.
  `Side` accepts six Pokémon and the active index; it stores side conditions
  and volatile state. `Pokemon` takes Showdown-style string IDs, current/base
  types, HP and stats, ability/base ability, item, status, up to four `Move`s,
  and Tera type/state. Stats are supplied by the caller; the binding does not
  translate a `poke-env` battle object or calculate the Pokémon's stats.
- Call
  `monte_carlo_tree_search(state, duration_ms=1000, iterations=0, threads=1)`.
  The Python result has `side_one` and `side_two` lists of entries with
  `move_choice`, `total_score`, and `visits`, plus `total_visits` for the
  iteration count. Move choices include switches as well as attacks.
- `calculate_damage(state, side_one_move, side_two_move,
  side_one_moves_first)` returns one two-element list per side. For ordinary
  damaging moves that list is **[maximum normal damage, maximum critical
  damage]** (from `DamageRolls::Max`). It is not a min–max range or a roll
  distribution. Fixed-damage moves return their fixed value, and moves that
  deal no damage return `[0, 0]`. Each side is calculated on its own copy of the
  starting state, so one side's attack doesn't affect the other's. Stage 3's
  accuracy check needs the full 16-roll range, so it must use smogon/damage-calc
  (or reproduce Showdown's modifier order and rounding). Scaling the max by 0.85
  is not exact, because the random roll is applied before STAB, type
  effectiveness and other modifiers, each with its own rounding. The binding also exposes `generate_instructions` and
  iterative-deepening search.

`scripts/spike_engine.py` demonstrates these constructors with two full,
level-100 singles teams and assigned Tera types. It first runs a damage
calculation, then single-threaded MCTS for 1, 2, and 5 seconds, and reports
elapsed wall time, iterations, iterations/sec, and the three highest-visit
side-one choices.
The hand-entered stats and teams form a representative opening position for a
throughput spike; they are not an accuracy fixture.

## Benchmark

Command: `python -m uv run python scripts/spike_engine.py`. Environment: Windows
x64, Python 3.12.15, Rust 1.99.0 MSVC, release build, `threads=1`. Throughput is
`total_visits / measured wall time`. This is one run at each duration, so treat
the results as a feasibility measurement rather than a stable machine score.

| Requested | Elapsed | Iterations | Iterations/sec | Top 3 side-one choices by visits |
| ---: | ---: | ---: | ---: | --- |
| 1 s | 1.088 s | 284,000 | 261,127.5 | knockoff 124,927; switch kingambit 56,642; switch dragapult 30,929 |
| 2 s | 2.213 s | 588,000 | 265,717.8 | knockoff 264,724; switch kingambit 172,200; switch dragapult 52,548 |
| 5 s | 5.528 s | 1,380,000 | 249,653.1 | knockoff 501,504; switch kingambit 359,926; switch corviknight 115,858 |

The damage API returned `[87, 131]` (max normal, max crit) for Great Tusk's
Close Combat and `[82, 123]` for Gliscor's Earthquake in this state.
The figures are end-to-end call throughput. Rust only checks the deadline after
each batch of 1,000 iterations, and the timed call also includes setup, tree
cleanup and converting results to Python, so elapsed time runs past the
requested duration.

## License, gaps, and Stage 4 recommendation

The repository's `LICENSE` is MIT (copyright pmariglia, 2024). The main gaps
are state translation and rule coverage: Stage 4 must map the live
`poke-env` state into this fixed engine model, including stats, active/base
types, PP, status, boosts, side conditions, items, abilities, and Tera state.
The engine targets singles and says it is not a complete Showdown simulator;
this spike did not test every Gen 9 interaction or compare results against
Showdown.

**Recommendation:** proceed with poke-engine as the Stage 4 Gen 9 singles MCTS
backend. The Python interface is usable and the observed single-thread rate is
about 250–266k iterations/sec for this position. Keep the state/action adapter
in Python, preserve Terastallization fields, and validate unsupported or
uncertain interactions against the authoritative battle state before relying
on a search result.

## Source references

- [poke-engine Cargo features](https://github.com/pmariglia/poke-engine/blob/main/Cargo.toml)
- [Python binding state and search functions](https://github.com/pmariglia/poke-engine/blob/main/poke-engine-py/src/lib.rs)
- [Gen 9 ability logic](https://github.com/pmariglia/poke-engine/blob/main/src/genx/abilities.rs)
- [Tera state, move generation, and damage logic](https://github.com/pmariglia/poke-engine/blob/main/src/genx/state.rs), [instruction generation](https://github.com/pmariglia/poke-engine/blob/main/src/genx/generate_instructions.rs), [damage calculation](https://github.com/pmariglia/poke-engine/blob/main/src/genx/damage_calc.rs)
- [Gen 9 item identifiers](https://github.com/pmariglia/poke-engine/blob/main/src/genx/items.rs)
- [Upstream README](https://github.com/pmariglia/poke-engine/blob/main/README.md)
- [MIT license](https://github.com/pmariglia/poke-engine/blob/main/LICENSE)
- [Foul Play Makefile build target](https://github.com/pmariglia/foul-play/blob/main/Makefile)
- [Foul Play Gen 9 requirements](https://github.com/pmariglia/foul-play/blob/main/requirements.txt)
