# Public ladder mode

`scripts/ladder.py` plays gen9randombattle on the public Showdown ladder
(play.pokemonshowdown.com) with our search player, one battle at a time.

## Account and credentials

1. Register an account on play.pokemonshowdown.com whose name clearly says it is a
   bot (e.g. `LewisBot-bot`; "bot" in the name). Do not impersonate a human.
2. Create `.env` in the repo root (already gitignored; never commit it):

   ```
   SHOWDOWN_USERNAME=LewisBot-bot
   SHOWDOWN_PASSWORD=your-password
   ```

   Real environment variables take precedence over `.env`. The credentials are
   never printed, logged, or written to `games.jsonl`, logs or summaries.

## Running

```
python scripts/ladder.py --games 1                      # first real game
python scripts/ladder.py --games 20 --bot search-net:evaluator=models/x.pev:scale=1.5
python scripts/ladder.py --games 300 --bot best:worlds=16  # promoted config (league.PRESETS)
python scripts/ladder.py --games 3 --dry-run            # local server 127.0.0.1:8000, throwaway accounts
```

Options: `--bot` (league spec syntax, default `search`; the presets `best` and `best-gen7` expand to the
current and previous promoted configs, keys after them override, e.g. `best:worlds=16`), `--max-concurrent 1`,
`--move-cap 5` (hard seconds per decision; on overrun a base-power x effectiveness
heuristic move is played), `--save-replays` (asks the server for `/savereplay`,
default off), `--game-timeout 1800` (forfeit a battle that long as a last resort,
logged), `--out data/ladder/<run-id>`.

Output in `--out`: `games.jsonl` (battle tag, opponent, result, turns, rating
before/after when the server sends them), `logs/<battle>.log` (protocol log per
battle), `summary.txt` (W-L-T, Wilson 95% CI, rating trajectory). A dropped
connection is retried with backoff (up to `--max-reconnects`); the server re-sends
a running battle to the new connection and it is played out.

## Politeness rules

- One game at a time (`--max-concurrent 1`); no farming loops, no parallel accounts.
- The account name says it is a bot.
- Play a modest number of games per session and keep an eye on the first ones.
- Do not spam the login server: reconnects back off, and failures stop after
  `--max-reconnects`.

## Stopping

`--games N` stops after N games. Ctrl-C stops early: a battle in progress is
forfeited (logged) and the summary is still written. The `/timer` of the server
also applies; a decision never takes longer than `--move-cap`.

## Analysing a game with our real sets

`logs/<battle>.log` interleaves every `|request|{json}` the server sent us (our full team: moves, PP, item, ability, tera type, stats, HP) at the position it arrived. Replay a turn with exact sets, or print what the bot saw and decided live:

    python scripts/replay_log.py data/ladder/<run>/logs/<battle>.log --turn 11
    python scripts/replay_log.py data/ladder/<run>/logs/<battle>.log --turn 11 --from-record data/ladder/<run>/records [--research --time-ms 300 --evaluator X.pev]

`--moves` / `--item` only matter for old logs without requests. poke-env tracks our PP from the protocol (as live), not from the request pp.

## Challenge mode (accept challenges instead of laddering)

The bot stays logged in on the public server and plays whoever challenges it, in
`gen9randombattle` only:

    python scripts/ladder.py --accept-challenges --games 5
    python scripts/ladder.py --accept-challenges --games 5 --from alice,bob --idle-timeout 30

- `--games N`: stop after N finished battles. `--from a,b`: accept only these users
  (compared case/punctuation-insensitively). `--idle-timeout MIN`: stop after MIN
  minutes with no challenge and no battle. Same `--bot`, `--move-cap`, `--game-timeout`,
  credentials and reconnect handling as the ladder. Without `--idle-timeout` it waits
  until N games are played or you press Ctrl-C.
- Output goes to `data/challenges/<run-id>` (override with `--out`): the same
  `games.jsonl`, `logs/`, `records/`, `summary.txt` as ladder runs, plus
  `challenges.jsonl` (every challenge seen, accepted or rejected). Friendly games, so
  no rating is reported.
- Answers: a gen9randombattle challenge while idle is accepted (`/accept`); any other
  format, a user outside `--from`, or a challenge arriving while a battle is running is
  rejected (`/reject USER`). Each challenge is answered once. No chat messages are sent.
  (poke-env's own `accept_challenges` silently drops other formats and never rejects,
  so challenge mode answers challenges itself.)
- Politeness: one battle at a time, the account name says it is a bot, no greeting or
  chat spam.
- Stopping: reach `--games`, hit `--idle-timeout`, or Ctrl-C (a running battle is
  forfeited and logged, the summary is still written).
- Local check (needs the local server): `python scripts/challenge_dryrun.py` starts the
  bot with `--dry-run`, challenges it in gen8randombattle (rejected) then
  gen9randombattle (accepted and played).

## Interleaved A/B testing (one account)

Compare two search configurations on the same ladder account, same hours, same
opponent pool. Repeat `--arm NAME=EVALUATOR.pev[:key=value...]` (league spec syntax: the
usual `scale=`, `time_ms=`, `worlds=` ... keys; the long form
`NAME=search-net:evaluator=X.pev:scale=2` also works). `--arm` replaces `--bot`.

```bash
python scripts/ladder.py --games 200 --ab-seed 1 \
  --arm gen2=models/gen2-res-searchabs-mlp16.pev:scale=1.5 \
  --arm v2s=models/v2-search-h64.pev:scale=2.0
python scripts/ab_report.py data/ladder/ab.jsonl
```

How it works: there is still ONE player and ONE connection (no second login). Before each
`ladder(1)` search the arm of the next battle is drawn from shuffled blocks holding each arm
twice (balanced to within 2 games, seeded by `--ab-seed`). The battle is bound to that arm at its
first decision, and the player swaps its `SearchConfig` (evaluator, scale, ...) to the
battle's arm on every decision, so a battle never mixes arms. Requires `--max-concurrent 1`
(the default) and ladder mode. `--dry-run` works for a local rehearsal.

Records: one JSON line per battle appended to `--ab-log` (default `data/ladder/ab.jsonl`, kept
across runs so it accumulates): battle tag, arm, evaluator path, scale, spec, our rating
before/after, opponent name and rating, result, turns, timestamp. `games.jsonl` gets an
`arm` field and the per-decision `records/` store the arm's config; `summary.txt` lists
games and wins per arm.

`scripts/ab_report.py` prints per arm games / wins / win rate with a Wilson 95% CI / mean
opponent rating; the arm difference with a 95% CI; a logistic regression
`win ~ arm + (opp_rating - our_rating)/400` (IRLS, numpy) giving the arm effect in
win-probability points at equal ratings; and the games per arm needed to resolve the
observed difference (~50% power and 80% power). Ties are excluded; games missing a
rating are left out of the regression only. Ladder noise is large: resolving a 5-point
difference needs on the order of 1500+ games per arm.
