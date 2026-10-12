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
python scripts/ladder.py --games 3 --dry-run            # local server 127.0.0.1:8000, throwaway accounts
```

Options: `--bot` (league spec syntax, default `search`), `--max-concurrent 1`,
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
