# Training a learned value function

The experiment: does MCTS with a learned leaf evaluator beat MCTS with the
hand-written `evaluate()` at equal time per world? Code:
`src/showdown_bot/train.py` (training + `.pev` export),
`scripts/duel.py` (the duel), `scripts/selfplay.py` (data),
`SearchConfig.evaluator` / `SHOWDOWN_SEARCH_EVALUATOR` (using the model).
Engine side: `engine/README.md`, `docs/engine-value-hook.md`.

## Setup

PyTorch is an optional dependency group (`train`), so the base install stays
light. **Never run `uv sync` in this repo**: the project `.venv` holds our
locally built `poke_engine` fork, which is not in `uv.lock`, and an exact sync
deletes it (rebuild with `scripts/build_engine.ps1` if that happens). Install
torch into the venv with `uv pip` instead.

### Laptop (CPU wheel)

```powershell
python -m uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
```

(`pyproject.toml` pins the `train` group's torch to the same CPU index.)

### 3080 machine (Windows, CUDA 12.x)

```powershell
# 1. build / install the engine fork as usual
scripts\build_engine.ps1
# 2. CUDA wheel (cu124 works with any recent driver supporting CUDA 12.x; cu126/cu128 also exist)
python -m uv pip install --python .venv\Scripts\python.exe --reinstall torch --index-url https://download.pytorch.org/whl/cu124
# 3. check
python -m uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and the RTX 3080. Run everything with `uv run --no-sync` (or the
venv's python directly) on that machine so uv does not swap the CUDA wheel for
the CPU one from the lock file. `train.py` picks `cuda` automatically when
available (`--device cpu` forces CPU); the same code runs on both. The whole
dataset is moved to the device once (738 float32 = 3 KB per position, so 1M
positions = 3 GB of VRAM; the 3080 has 10 GB, so use `--states truth` or less
data if it gets tight).

The MCTS processes use the CPU, so keep training and duels from running at the
same time.

## The loop

```powershell
# 1. generate (search vs search, both sides record; needs the server: scripts\start_server.ps1)
python -m uv run python scripts/selfplay.py --games 600 --pairs 4 --run-id gen0

# 2. train + export (prints train/val BCE, constant baseline, accuracy, calibration;
#    writes the .pev and re-checks it against Evaluator.evaluate)
python -m uv run python -m showdown_bot.train --data data/selfplay/gen0 --model mlp --hidden 64 --out models/gen0-mlp64.pev
python -m uv run python -m showdown_bot.train --data data/selfplay/gen0 --model linear --out models/gen0-linear.pev

# 3. duel: learned vs stock at equal time per world
python -m uv run python scripts/duel.py --evaluator models/gen0-mlp64.pev --games 100 --pairs 4

# 4. iterate: generate more data (optionally with the net playing itself), train on all runs
python -m uv run python -m showdown_bot.train --data data/selfplay/gen0 --data data/selfplay/gen1 --out models/gen1.pev
```

Use the model anywhere a `SearchPlayer` is built:

```python
SearchPlayer(..., evaluator="models/gen0-mlp64.pev")      # constructor argument
# or SHOWDOWN_SEARCH_EVALUATOR=models\gen0-mlp64.pev (read by SearchConfig.from_env)
```

Each MCTS pool worker loads the file once (`players/_worker.get_evaluator`) and
reuses it. With no evaluator the player is the stock one.

`bench` has a `search-net` player (needs `SHOWDOWN_SEARCH_EVALUATOR`), e.g. vs
Foul Play. The env var is process-wide, so `--p1 search-net --p2 search` in one
bench process would give both the net; use `scripts/duel.py` for that match-up
(per-player config, shared MCTS pool, symmetric CPU contention).

## Training details

- **Data**: `dataset.iter_samples(sides=(1,))`: side one's features
  (`poke_engine.state_features`, 738 floats, scaled to roughly [-1, 1]) and the
  game outcome (1/0.5/0) as the target. `--states both|truth|worlds` selects
  exact truth states, sampled opponent worlds, or both (see
  `docs/selfplay-data.md` for the difference: worlds contain a guessed
  opponent team, truth is the real state).
- **Split**: by `battle_tag` (`--val-fraction`, default 0.1, seeded). Both
  players' recordings of a battle share a tag and stay on the same side of the
  split; the several sampled worlds of one decision are near-duplicates, so a
  per-position split would leak.
- **Weights**: `--weighting sqrt` (default) gives a position weight
  `1/sqrt(positions in its battle)`, normalised to mean 1, so a long game counts
  more than a short one but not in proportion to its length. `game` = equal
  weight per battle, `none` = per position. Early stopping and the reported
  "weighted" numbers use these weights; "unweighted" ones are also printed.
- **Model**: exactly the engine's: `Linear(738, hidden) -> ReLU -> Linear(hidden, 1)`
  (`--model mlp`) or `Linear(738, 1)` (`--model linear`); output = logit of
  P(side one wins); loss = BCE-with-logits; AdamW (`--lr`, `--weight-decay`),
  early stopping on val loss (`--patience`), seeded (`--seed`). Hidden size is
  constrained by search cost: 1.2x slower than stock for linear, 1.6x for 64,
  2.4x for 256.
- **Export**: `poke_engine.write_mlp` / `write_linear` with torch's weight
  layout (`W1` is `[hidden][738]` row-major, as learned.rs expects). The
  trainer reloads the file with `Evaluator.load` and checks `evaluate` against
  the torch logits on up to 100 val states (tolerance 1e-3, observed ~1e-6).

### Value targets

`--value-target outcome|search|blend` chooses what the net regresses on:

- `outcome` (default, the original behaviour): the game result z (1/0.5/0).
- `search`: q, the search's own value estimate. The recorder stores each
  searched world's `avg_score` (MCTS root total score / visits, in [0, 1] from
  our side). q of a decision is the mean over its worlds that had visits;
  world and truth samples of the decision share it (`--q-source decision`,
  default). `--q-source world` gives each world sample its own `avg_score`
  (truth keeps the decision mean). Side-two views use `1 - q`.
- `blend`: `(1 - L) * z + L * q` with `--blend-lambda L` (default 0.5).

Positions without a usable q (no world with visits) fall back to z; the count
is printed. The loss stays BCE-with-logits, which accepts soft targets. The
report always prints BCE, accuracy and calibration against the pure outcome z
(so models trained on different targets stay comparable with the `ln 2`
baseline), and for `search`/`blend` also the BCE on the training target plus
the val correlation between q and z.

Rationale: a single game result is a very noisy label for a mid-game position;
search-derived (soft) targets have lower variance and were found to beat the
plain outcome target in AlphaZero-style training (Willemsen, Plaat & Preuss
2021, "Value targets in off-policy AlphaZero: a new greedy backup"; Boige,
Boumaza & Scherrer 2026, arXiv:2609.09094). Caveat: q comes from the search
that used the evaluator which generated the data, so with the stock evaluator
it carries that evaluator's biases.

**Absolute vs relative q.** The recorded `avg_score` is relative to the root:
the stock MCTS score is `sigmoid(0.0125 * (e_leaf - e_root))`, so q is centred
on 0.5 and measures search improvement, not win probability (first smoke run
with relative q: search target barely beat the constant). `--q-kind absolute`
(default) adds the root evaluation back:
`q_abs = sigmoid(0.0125 * evaluate(state) + logit(clip(avg_score, 1e-4, 1-1e-4)))`
(`--q-eval-scale` changes 0.0125). Per world it uses that world's state;
the decision value is the mean over valid worlds; truth states use
`evaluate(truth) + logit(decision mean score)`. `poke_engine.evaluate` is side
one's view (checked: the fitted slope of z on it is positive, +0.62).
`--q-kind relative` keeps the old behaviour for comparison.

The report also prints the **stock-eval baseline**: a logistic regression of z
on `0.0125 * evaluate` fitted on train, scored on the same val split. A net
must beat this (not just ln 2).

Full gen0 (604 battles, 123.6k positions, 60 val battles, MLP-64, seed 0,
same split; BCE and accuracy vs the pure outcome z, constant = 0.693):

| target | val outcome BCE | stock-eval logistic | accuracy |
|---|---|---|---|
| outcome | 0.534 | 0.511 | 0.700 |
| blend 0.5 abs | 0.478 | 0.511 | 0.736 |
| blend 0.75 abs | 0.464 | 0.511 | 0.761 |
| search abs | 0.474 | 0.511 | 0.763 |

Val corr(q_abs, z) = 0.607. Single seed; **duel results are pending**.

## Reading the results

- Val BCE must beat the constant predictor (`ln 2 = 0.693` for a balanced set).
  Self-play outcomes are noisy labels (the position at turn 2 barely predicts
  the winner), so large gains are impossible; the useful signal is in
  mid/late-game positions. Check the calibration table too.
- In search the default score is `sigmoid(z_leaf - z_root)`
  (`relative_to_root`), the same shape as the stock
  `sigmoid(0.0125 * (e_leaf - e_root))`: only the *differences* of the net
  between leaf and root steer the search, not its calibration.
- The duel prints the iterations per second of each side. The net costs
  1.5-2.4x per iteration, so at equal time it must be better per iteration by
  more than it is slower to win.

## Leaf scale

Stock leaf scores are `sigmoid(0.0125 * dEval)` with typical dEval of 50-200
points, so scores spread widely. A net trained on noisy outcomes has logits that
barely move between root and leaf, so `sigmoid(z_leaf - z_root)` sits near 0.5
everywhere and MCTS cannot discriminate (hypothesis (a) of showdown-bot-589.16).

The runtime knob `leaf_scale` k (not stored in the `.pev`; default 1.0 = old
behaviour) makes the score `sigmoid(k * (z_leaf - z_root))` (or `sigmoid(k * z_leaf)`
with `relative_to_root=False`).

- Python: `Evaluator.load(path, leaf_scale=k)` (also `linear` / `mlp`; must be
  finite and > 0, else `ValueError`); `ev.leaf_scale`, `ev.logit(state)` and the
  stock `poke_engine.evaluate(state)` are exposed.
- Search player: `SearchConfig.evaluator_scale`, constructor arg
  `evaluator_scale=`, env `SHOWDOWN_SEARCH_EVALUATOR_SCALE`. Pool workers cache
  evaluators per (path, scale).
- Duel: `scripts/duel.py --leaf-scale K` (net side only).
- Calibration: `python scripts/calibrate_leaf_scale.py --data DIR [--data DIR...]
  --model x.pev [--max-games N]` compares the per-decision change of
  `0.0125 * evaluate` with the change of the net logit over consecutive recorded
  states and prints k = std(d_stock) / std(d_net) (and a MAD version) plus a
  k/2, k, 2k sweep.

Status: the duel sweep over k is pending; no result is recorded yet.
