# Bot league

`python -m showdown_bot.league` plays a round-robin between registered bots on the
local Showdown server (never the public ladder) and rates them.

```
python -m showdown_bot.league --bots random,maxpower,heuristic,search,foulplay \
    --games-per-pair 100 --out data/league/run1 --search-time-ms 100 --seed 1
python -m showdown_bot.league --rate-only data/league/run1   # recompute ratings offline
```

Bots are the names in `bench.PLAYER_REGISTRY` (`random`, `maxpower`, `heuristics`
(alias `heuristic`), `search`, `search-net`) plus `foulplay`, which runs as a
separate process exactly as in `bench.py` (set `SHOWDOWN_BOT_FOULPLAY_DIR` from a
worktree). Options: `--search-time-ms`, `--fp-search-time-ms`, `--fp-parallelism`,
`--fp-threads`, `--concurrency` (games in flight per matchup), `--game-timeout`,
`--bootstrap N` (default 1000), `--anchor BOT`, `--seed`, `--pairing roundrobin`
(Swiss is not implemented). Matchups run sequentially; every player gets a unique
username and is disconnected (and Foul Play killed) when its matchup ends.

Seats: poke-env's challenger is p1. Against our own bots the challenger alternates
every `--concurrency` games. Foul Play only accepts challenges, so it is always p2;
its pairs are therefore not seat balanced.

## Bot specs

Each `--bots` entry is `label=type[:key=value[:key=value...]]`; a plain `type` has label = type.
The label is the bot's name in ratings and `games.jsonl`. For `search` / `search-net`
the keys are `evaluator` (path to a `.pev`), `scale` (leaf-value scale, passed as
`evaluator_scale=`; needs a checkout that has it, otherwise any scale other than 1
is an error), `time_ms` and `worlds`. Example:

```
--bots stock=search,net64=search-net:evaluator=models/a.pev:scale=4,foulplay
```

Config is passed per player (the `SHOWDOWN_SEARCH_*` env vars are not used). A colon
inside a value (a Windows drive letter) is kept. The spec text of every bot is stored
in `results.json` (`specs`) and on each `games.jsonl` line.

## Compute

- `--search-workers N`: one MCTS process pool of N workers shared by all search
  players (as in `scripts/duel.py`). Default: each player uses its own default pool.
- `--parallel-pairs K`: play up to K matchups at once (default 1); Foul Play matchups
  each launch their own instance. A cores-used line is logged and a WARNING when the
  peak exceeds `os.cpu_count()`. If any matchup fails the others are cancelled and
  cleaned up.

## Outputs (`--out`)

- `games.jsonl`: one line per finished game: `schema_version`, `battle_tag`, `p1`/`p2`
  (bot names by seat), `p1_user`/`p2_user`, `bots`, `winner` (bot name or `tie`), `turns`,
  `seed`, `format`, `specs` (label to spec text). The ratings are computed only from this file.
- `replays/`: poke-env replay HTML, saved by every player (each side's view).
- `records/`: for `search`/`search-net`, the existing recorder output
  (see `docs/selfplay-data.md`): per-decision state, policy, chosen action, outcome.
- `results.md`, `results.json`: ratings with CIs, head-to-head matrix, leak report.

## Rating method

Bradley-Terry strengths fitted by Hunter's MM algorithm
(`p_i <- W_i / sum_j n_ij / (p_i + p_j)`, geometric mean renormalised to 1). A tie
counts as half a win for each side. A prior of 0.5 virtual tie games between every
pair keeps undefeated or winless bots finite. Ratings are shown as `400*log10(p)`
(mean 0, or `--anchor` bot = 0). 95% CIs are percentile intervals of a bootstrap
that resamples the game list with replacement and refits (seeded by `--seed`).
Head-to-head cells show W-L-T and the Wilson 95% interval of the score. The
sequential Elo column (K=16, game order) is secondary and order dependent.

## Leak check

After every matchup and at the end the league logs any player websocket still open,
Foul Play process still running, or stray asyncio task. Any finding is written to
`results.json` and the run exits with code 3.
