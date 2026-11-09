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

## Fixed gauntlet (`scripts/gauntlet.py`)

Head-to-head against one bot is intransitive, so candidates are also judged against a fixed
opponent set. Run from the checkout that holds `models/` (specs use relative paths):

```
python scripts/gauntlet.py --candidate "search-net:evaluator=models/x.pev:scale=1.5:mode=argmax:time_ms=1000:worlds=4" \
    --games 100 --out data/gauntlet/x.jsonl --league-args "--search-workers 8 --concurrency 2"
python scripts/gauntlet.py --out data/gauntlet/x.jsonl --report     # summary only
```

Default opponents: `gen2` (`search-net` with `models/gen2-res-searchabs-mlp16.pev`, scale 1.5, argmax,
1000 ms, 4 worlds), `search` (stock, argmax, 1000 ms, 4 worlds), `human`
(`human:model=models/human-policy-heldout.npz`, an opponent model independent of the exploit
model) and, with `--foulplay`, `foulplay` (needs `scripts/setup_foulplay.ps1` and the server on port
8000; `--league-args "--fp-search-time-ms 1000"`). Override the set with `--opponents-file f.json`
(`{"label": "spec", ...}`); `--only a,b` plays a subset.

Each opponent is played in chunks (`--chunk`, default 10 games, seats alternate) by a fresh
`python -m showdown_bot.league` subprocess in `<out stem>/<opp>-<chunk>/`, so the league's
stall handling applies per chunk and a hung chunk is killed after `--chunk-timeout` (default
3600 s) with its finished games kept (an opponent is abandoned after `--max-failures` chunks that
finish nothing). Every game is appended to the jsonl as it is read back; rerunning with the same
`--out` and candidate resumes and plays only the missing games. Output: score per opponent
(win 1, tie 0.5) with a Wilson 95% CI, and the mean of the per-opponent scores with a bootstrap
95% CI (resampling games within each opponent).

## Material blend and KO tie-break (bead 589.55, both default off)

Search-net spec keys for positions the evaluator rates as decided:

- `material=<lambda>` (0 <= lambda < 1, e.g. 0.05; needs `evaluator=`): engine leaf score becomes
  `(1-lambda)*value + lambda*m`, `m = sigmoid(1.5*(sum hp fraction diff + 0.25*alive diff))` of side one
  vs two. Non-terminal leaves only; won/lost terminals stay exactly 1/0. Engine `Evaluator(material=)`, wheel >= 0.0.48+sdb.12.
- `ko_tiebreak=<eps>` (e.g. 0.01): after the root policy is aggregated, if the top option does not KO the
  opponent active and a KO option (>= 50% KO vs a majority of its non-switch replies, in a majority of worlds)
  has mean score within eps, play the KO option. Counted as `ko_tiebreaks` in the search stats.

e.g. `net=search-net:evaluator=m.pev:scale=4:material=0.05:ko_tiebreak=0.01`

## SPRT duel (`--sprt`)

Instead of a fixed number of games, play paired games until a sequential probability ratio test
decides, as Fishtest does for chess engines. Exactly two of our bots: the candidate first, the
baseline second.

```
python -m showdown_bot.league \
    --bots "cand=search-net:evaluator=models/new.pev:scale=1.5:mode=argmax:time_ms=1000:worlds=4,best=search-net:evaluator=models/gen2-res-searchabs-mlp16.pev:scale=1.5:mode=argmax:time_ms=1000:worlds=4" \
    --sprt 0,21 --max-games 2000 --search-workers 32 --concurrency 4 \
    --out data/league/sprt-new-vs-best --seed 1
```

- `--sprt elo0,elo1[,alpha,beta]` (defaults alpha = beta = 0.05; `0,21` is about +3 points).
  Implies `--teams swap` (a pair = two games, same two generated teams, seats swapped, see
  `showdown_bot/teams.py`; `--teams mirror` also works) and `--strict-workers`. `--max-games`
  (even, default 2000) is the budget; `--games-per-pair` is ignored.
- `--concurrency` is the number of pair lanes. Each lane owns a candidate and a baseline player
  and plays its pairs back to back; the lanes share the `--search-workers` pool.
- `--strict-workers` (`--no-strict-workers` to disable): lowers `--concurrency` so that
  `concurrency x (worlds of both players) <= --search-workers`; the chosen value is logged
  (`strict workers: concurrency N (requested M)`). Without it the league only warns
  ("peak demand ... > workers") and games queue for the pool, which inflates decision time.
- Test: each finished pair gives the candidate 0, 0.5, 1, 1.5 or 2 points; counts are kept as
  the pentanomial `[0, 0.5, 1, 1.5, 2]`. LLR = `N (s1-s0)(2s-s0-s1) / (2 var)` (the normal GSPRT
  approximation of Fishtest and fastchess, `src/showdown_bot/sprt.py`), `s` the mean pair
  score in [0, 1], `s0, s1` the logistic scores of elo0 and elo1, zero counts regularised to
  1e-3. Bounds: `ln(beta/(1-alpha))` and `ln((1-beta)/alpha)` (about +-2.94). No verdict is
  given before 20 pairs. The verdict is frozen at the first crossing; pairs in flight are
  cancelled (their games stay in `games.jsonl` but are not counted).
- Every progress line shows `SPRT LLR x [lo, hi] pairs [n0, n1, n2, n3, n4] Elo +e [lo, hi]`.
  The run ends with `SPRT: H1 accepted`, `SPRT: H0 accepted` or `SPRT: inconclusive at max games`
  and the Elo estimate with its 95% CI. The same is stored in `results.json` under
  `compute.sprt`.
