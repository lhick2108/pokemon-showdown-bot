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

## Outputs (`--out`)

- `games.jsonl`: one line per finished game: `schema_version`, `battle_tag`, `p1`/`p2`
  (bot names by seat), `p1_user`/`p2_user`, `bots`, `winner` (bot name or `tie`), `turns`,
  `seed`, `format`. The ratings are computed only from this file.
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
