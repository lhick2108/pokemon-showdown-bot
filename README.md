# pokemon-showdown-bot

Research code for two Pokémon Showdown bots that play **Gen 9 Random Battles**, and the tools used to build and
measure them.

- **System A** puts its strength in the search: a Monte Carlo tree search over sampled guesses of the opponent's
  hidden team, guided by small learned evaluator and prior nets.
- **System B** puts its strength in a learned policy: a 9.2M-parameter transformer trained by PPO self-play, with a
  shallow policy-rollout search at play time (after Sokota et al. 2025, arXiv:2511.07312).

Both share the same battle engine, belief tracker, team sampler and test harness.

## Results

On the public gen9randombattle ladder, System B finished at 166–42 in counted games (GXE 95.0). Its rating peaked at
2412 and averaged 2324 over one day of readings. It passes 2341 regularly but does not hold above it. System A's last
sessions ended at 2208–2238 (peak 2483). Over 700 local games the two systems are level (359–341). These numbers come
from small samples on one account each; treat them as indicative.

## What is and is not in this repository

**Included**

- The battle engine: our fork of [poke-engine](https://github.com/pmariglia/poke-engine) in `engine/`, with a
  self-play pool that runs many games in parallel.
- The belief tracker and team sampler (which sets an opponent can still hold, and full guesses of the hidden team).
- Both bots' code, the PPO trainer, the rollout search, and the training command used for the 9.2M net
  (`runs/ppo8p.sh`).
- The test harness (paired matches with sequential tests, round-robin leagues), the ladder runner and the analysis
  scripts.

**Not included**

- **No trained models.** `models/` and `runs/` are not published, for either system. Without them the stock search
  bot still runs; System A at full strength and System B do not. Reproducing System B means training it: a 2.7M
  teacher net over about 40 million self-play games, then the 9.2M net. That took weeks on one RTX 3080.
- Game data, run logs and ladder records (`data/`).

## Status and requirements

This is research code from a six-week project, not a packaged tool. It has not been tested from a fresh clone.

- Windows, Python 3.12. Dependencies are in `pyproject.toml` ([poke-env](https://github.com/hsahovic/poke-env);
  PyTorch for training, see `docs/training.md`).
- Prebuilt engine wheels for Windows / CPython 3.12 are in `engine/wheels/` up to build `sdb.29`, which is enough for
  the search bots. System B needs build `sdb.34` or later, built from source with
  `scripts/build_engine_wheel.ps1` (needs Rust and the MSVC build tools).
- A local Pokémon Showdown server for testing: `scripts/start_server.ps1` (see `docs/league.md`).

## Layout

| Path | What |
| --- | --- |
| `engine/` | Rust engine fork: simulator, tree search, self-play pool, net-guided search, Python bindings |
| `src/showdown_bot/` | Bots, belief tracking, sampling, trainers, harnesses |
| `src/showdown_bot/players/` | The bot types (`search`, `token-policy`, `net-search`, `rollout-search`, …) |
| `src/showdown_bot/ppo.py` | PPO self-play trainer for the transformer |
| `src/showdown_bot/rollout_search.py`, `rollout_live.py` | Rollout search in the simulator and on the server |
| `src/showdown_bot/league.py` | Matches between bots: paired games, SPRT, leagues |
| `scripts/ladder.py` | Ladder and challenge runner |
| `scripts/` | Engine builds, benchmarks, loss analysis, switch audit and other tools |
| `docs/` | Design notes, one file per component |
| `tests/` | Python tests (`pytest`); Rust tests live in `engine/` |

## Where to start reading

| Topic | File |
| --- | --- |
| Running matches between bots | `docs/league.md` |
| The ladder runner | `docs/ladder.md` |
| The token transformer and its inputs | `docs/token-net.md`, `docs/tokens.md` |
| Self-play PPO | `docs/selfplay-ppo.md` |
| Net-guided tree search | `docs/net-search.md` |
| Training setup (PyTorch, CUDA) | `docs/training.md` |

## Typical commands

```
# a match between two bot specs on the local server, with a sequential test
python -m showdown_bot.league --bots "a=<spec>,b=<spec>" --teams swap --sprt 0,21

# a dry run of the ladder runner against the local server
python scripts/ladder.py --games 3 --dry-run

# PPO self-play training (the full command for the 9.2M net is in runs/ppo8p.sh)
python -m showdown_bot.ppo train --help
```

Bot specs are `type:key=value:…`; the accepted keys for each type are listed at the top of
`src/showdown_bot/league.py`.

## If you run a bot on the public ladder

Pokémon Showdown has a staff policy on bots and a registration scheme for them. Read it first, register the account as
a bot if the policy asks for that, and make it clear from the account name that it is automated. Our own sessions did
not meet that standard, and we recommend that others do better. Credentials go in a git-ignored `.env`
(`docs/ladder.md`); never commit them.

## Credits

Built on [poke-engine](https://github.com/pmariglia/poke-engine) and compared throughout against
[Foul Play](https://github.com/pmariglia/foul-play), both by pmariglia, and on
[poke-env](https://github.com/hsahovic/poke-env). The rollout search follows Sokota et al. (2025); the training recipe
draws on the public description of Jaxcalibur.
