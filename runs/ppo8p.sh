# 2026-11-23 00:35: second style opponent "sweep" (7% of games): the same net biased to use a boosting move when the learner's active threatens little (+50% set-up moves; 0.54 +-0.03 v the plain net in 300 simulator games). "style" (pivot + heal) now 8%. From the all-games ladder review: an opposing boosted Pokemon scoring 2+ knockouts is 23 of our 54 losses v 1900+.
# 2026-11-22 23:58: style opponent added (--extra-opps style=...@0.10@1.0@3@5): 10% of games v the update-11,400 net biased to pivot into switch-ins that take little damage (+45% such pivots) and to heal when hurt (+45% heals); as strong as the plain net in a 300-game simulator check (0.48 +-0.03). From the ladder opponent that beat us 21-25.
# 2026-11-20 10:20: training packs (--pack-share 0.05, engine sdb.35): 5% of games start from saved positions of games the learner lost to the gen7 search, hidden opponent sets redrawn, half replayed v gen7 search and half v the ordinary mix; stop file runs/ppo8p.stop.
# 2026-11-20 00:05: hold on the old net loosened (KL weight 0.19 -> 0.10 at update 3100, 0 at 6000, was 0 at 8000) after 6000-game tests showed only +5 Elo at update 3000; stop file runs/ppo8p.stop.
# 2026-11-19 15:00: same run continued with 5% of games v reduced-budget gen7 search (--g7-share, engine sdb.34); stop file runs/ppo8p.stop.
# ppo8 restart (bead 589.68.8, 2026-11-19): the first self-play stage (runs/ppo8, updates 150-2345) LOST ~30 Elo v the
# teacher (2000-game best-move tests: ckpt-150 50.0%, ckpt-2250 45.5%, its EMA 47.8%). Restart from ckpt-150 in a new
# folder: 300 updates of value-only warm-up (the value adapts to gamma 1 while the policy is held on the teacher),
# then PPO at lr 5e-5 -> 5e-6, teacher KL 0.3 -> 0 over 8000 updates, entropy 0.03 -> 0.01, EMA, short batch dropped.
# runs/ppo8r_eval.sh tests the saves v the teacher. Restarts itself after a crash; stop with runs/ppo8p.stop.
cd C:/Users/Lewis/pokemon-showdown-bot
export PYTHONPATH="C:/Users/Lewis/pokemon-showdown-bot/.cache/engine/sdb.35;C:/Users/Lewis/pokemon-showdown-bot/src"
T=runs/ppo7/ckpt-46000.pt
for k in 1 2 3 4 5 6 7 8 9 10 11 12; do
  [ -f runs/ppo8p.stop ] && break
  R="--resume-from runs/ppo8/ckpt-00150.pt"
  [ -f runs/ppo8r/state.pt ] && R="--resume --resume-from runs/ppo8r/latest.pt"
  echo "== ppo8r+g7 packs start $k $(date) $R" >> runs/ppo8r.log
  .venv/Scripts/python.exe -u -m showdown_bot.ppo train --init $T --out runs/ppo8r $R --seed $((110 + k)) --pack-share 0.05 --pack-size 2000 --pack-g7 0.5 --extra-opps "style=runs/ppo8r/ema-ckpt-11400.pt@0.08@1.0@3@5,sweep=runs/ppo8r/ema-ckpt-11400.pt@0.07@1.0@0@0@3" --g7-share 0.05 --g7-worlds 4 --g7-iters 2000 --g7-procs 10 \
    --games 1024 --threads 8 --rollout-steps 32 --epochs 1 --minibatch 512 --min-batch-frac 0.5 --opp-coef 0.5 \
    --lr 5e-5 --lr-final 5e-6 --lr-decay-start 300 --lr-decay-updates 30000 \
    --ent-coef 0.03 --ent-final 0.01 --ent-decay-start 300 --ent-decay-updates 30000 \
    --gamma 1.0 --ema-decay 0.9995 --value-warmup 300 \
    --kl-beta0 0.207 --kl-beta1 0.0 --kl-decay-updates 6000 --kl-off-after 6000 \
    --p-anchor 0.1 --snapshot-every 200 --max-past 12 --active-past 3 --seed-past 24 --ckpt-every 100 --gpu-mem-gb 7.2 --minutes 43200 >> runs/ppo8r.log 2>&1
  sleep 20
done
echo PPO8RDONE >> runs/ppo8r.log
