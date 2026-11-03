# queue v12b (2026-11-03): time-saving version. 2 pair lanes (no strict workers), pooled SPRTs, game caps.
# A: ft-v5 (best) vs gen7 (best-gen7), 16w, engine sdb.19; pooled with sprt-stack16 (Oct, same matchup and
#    engine) and the first 14 games tonight; cap 800 pooled games.
# B: sdb.28 vs sdb.19 on best-gen7 16w (b28 pins its pool, main process sdb.19); 11 workers per pool; cap 600.
R=C:/Users/Lewis/pokemon-showdown-bot
PY=.venv/Scripts/python.exe
export PYTHONPATH="$R/.cache/engine/sdb.19;$R/src"
L="$PY -u -m showdown_bot.league --sprt 0,21 --no-strict-workers --concurrency 2 --game-timeout 3600"
$L --max-games 800 --search-workers 22 --seed 154 \
  --sprt-prior data/league/sprt-stack16:stack16:best16 --sprt-prior data/league/sprt-v5-gen7-16w:v5:gen7 \
  --bots "v5=best:worlds=16,gen7=best-gen7:worlds=16" \
  --out data/league/sprt-v5-gen7-16w-b > data/league/sprt-v5-gen7-16w-b.log 2>&1
echo "A done $(date)"; grep -E "SPRT: |Elo" data/league/sprt-v5-gen7-16w-b.log | tail -2
$L --max-games 600 --search-workers 11 --seed 153 \
  --bots "b28=best-gen7:worlds=16:engine=sdb.28,b19=best-gen7:worlds=16" \
  --out data/league/sprt-sdb28-sdb19-16w > data/league/sprt-sdb28-sdb19-16w.log 2>&1
echo "B done $(date)"; grep -E "SPRT: |Elo" data/league/sprt-sdb28-sdb19-16w.log | tail -2
echo Q12DONE
