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
