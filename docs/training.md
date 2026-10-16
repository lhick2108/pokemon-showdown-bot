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

### Residual evaluator

`--residual` starts from the stock evaluation and learns only a correction:

    z = a * 0.0125 * evaluate(state) + net(features)

On gen0 truth positions a 2-parameter logistic fit of the outcome on
`0.0125 * evaluate` is already about as good as a plain learned MLP, so the net
should not have to re-learn it. `a` is a learnable scalar, initialised from a
weighted Newton logistic fit (with intercept b) on the train split
(`--stock-coef-init A` overrides, `--freeze-stock-coef` keeps it fixed). The
net's output layer starts at zero weights and bias b, so the untrained model is
exactly the calibrated stock logistic; the trainer prints that stock-only val
BCE next to the net's. The raw `evaluate()` per sample is computed once at load
time (`dataset.iter_samples(..., with_stock_eval=True)`) and travels as the last
column of `Data.x`.

Export writes weights-format 2 with `stock_coef = a * 0.0125` (see
`engine/README.md`; non-residual models still write format 1), and the
post-export check reloads the file and compares `Evaluator.evaluate` (which adds
the stock term) with the torch logit. In search the leaf score is still
`sigmoid(leaf_scale * (z_leaf - z_root))`, so with a zero net this is the stock
formula with the scale `a * 0.0125` instead of `0.0125`.

gen0 truth states (35,381 positions, 604 battles, seed 0, 60 val battles),
weighted val BCE against the outcome, `--lr 1e-4` (the default `1e-3` overfits
after one epoch):

| model | val BCE | stock-only logistic | learned a |
| --- | --- | --- | --- |
| residual MLP-64 | 0.5200 | 0.5336 | 0.509 |
| residual MLP-16 | 0.5189 | 0.5336 | 0.506 |
| residual linear | 0.5179 | 0.5336 | 0.522 |

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

### Bootstrapped (TD) targets

`--value-target td|nstep` replaces the single-step search value by a return
that looks ahead along the same recording (one side's view of one battle; the
file is the unit, so the chain is that player's decisions in order). Per
decision t the chain value is q_t, the absolute decision-level search value
(truth state's `q_abs` if the decision has a truth state, else the mean over
worlds; `--q-kind relative` uses the raw score). World and truth samples of a
decision share it and therefore share the target. With decisions t = 0..T-1
and the outcome z as the step after the last decision:

- `td` (`--td-lambda L`): the lambda-return
  `G_t = (1-L) sum_{n=1}^{T-t-1} L^(n-1) q_{t+n} + L^(T-t-1) z`, computed by the
  backward recursion `G_{T-1} = z`, `G_t = (1-L) q_{t+1} + L G_{t+1}`
  (Sutton & Barto, lambda-returns). L=1 is the outcome z, L=0 the next
  decision's q.
- `nstep` (`--td-n N`): `G_t = q_{t+N}` if `t+N < T` else z.
- `--td-include-self --td-mu M`: `target = (1-M) G_t + M q_t`.

A decision with no usable q is a gap: it takes the next available later q
(or z); the number of gaps is printed. Existing modes are unchanged. Motivation:
Boige, Boumaza & Scherrer 2026 (arXiv:2609.09094) bootstrap AlphaZero value
targets from later search values; the later q is a less noisy and less biased
label than z while still carrying information from beyond the current search
horizon.

Full gen0 (604 battles, 40,359 decisions, none without q; MLP-64 plain, seed 0,
same split as above; BCE/accuracy vs the pure outcome z):

| target | val outcome BCE | accuracy | best epoch | train BCE |
|---|---|---|---|---|
| stock-eval logistic (baseline) | 0.5113 | - | - | - |
| blend 0.75 abs (previous best) | 0.4637 | 0.761 | 3 | 0.4677 |
| td L=0.5 | 0.4653 | 0.761 | 3 | 0.5162 |
| td L=0.7 | 0.4672 | 0.758 | 4 | 0.4896 |
| td L=0.9 | 0.4676 | 0.756 | 3 | 0.4493 |
| nstep n=1 | 0.4597 | 0.767 | 3 | 0.5354 |
| nstep n=3 | 0.4631 | 0.766 | 4 | 0.5033 |
| td L=0.7, mu=0.5 | **0.4588** | 0.769 | 3 | 0.5241 |
| td L=0.7, mu=0.5, residual MLP-16 (lr 1e-4) | 0.4656 | 0.751 | 46 | 0.5274 |

Single seed; differences of ~0.005 are probably within noise. Pure TD without
the own q is about equal to the blend; mixing in the decision's own q (mu=0.5)
or a 1-step target is slightly better. Duels pending.

### Interior-node targets (bead showdown-bot-rnm)

A root decision gives one search-backed value per searched world, although the tree below
it has already valued thousands of deeper positions (a KO two turns out, the end of a heal
loop, a hazard that stuck). `selfplay.py --export-nodes K` has the engine sample K interior
nodes per searched world, visit-weighted, and record `[state, visits, value, static_value,
depth]` (format and sampling: docs/selfplay-data.md, "Interior nodes"; needs engine
0.0.48+sdb.9 and `--threads 1`). `train.py --nodes on` turns each into an extra sample whose
target is the node's absolute search value, exactly like a root `--value-target search`.

* **Leaf-dominated nodes.** A node's first visit is its own leaf evaluation, so a node with a
  handful of visits only repeats what the net already thinks. Nodes under
  `--node-min-visits` (default 16) are dropped and the rest are weighted by
  `--node-weight * min(1, visits / --node-visit-ref)` (default ref 64) times
  `--node-depth-discount ** (depth - 1)`. The weights are relative to a root sample (1.0) and are
  normalised with the per-battle weighting like every other sample.
* **Scale.** The node value is anchored to the evaluator that ran the search (stock: 0.0125 *
  evaluate; learned: its logit and leaf scale), and a world with exported nodes also records the
  exact absolute root value, which the loader then uses instead of reconstructing it with the
  stock evaluation. Recordings without nodes load exactly as before.
* **Size.** K = 8 adds about 1 KB gzipped per decision (measured: 0.8 -> 1.8 KB per
  2-world decision at 300 ms) and costs no search time. In memory every node is a full feature
  row (738 floats), so thin them at load time with `--nodes-per-world N`.
* The existing targets are untouched: `--nodes off` is the default and
  `--value-target outcome|blend|td|nstep` ignore the recorded nodes (nodes need `search|blend`).

Recipe (3000 battles, then a with/without-nodes A/B on the same data):

```powershell
# 1. data: net-guided search, 300 ms x 2 worlds per decision, 8 interior nodes per world
python scripts/selfplay.py --games 3000 --pairs 4 --run-id gen7 --search-time-ms 300 --worlds 2 `
    --evaluator models/gen2-res-searchabs-mlp16.pev --leaf-scale 1.5 --export-nodes 8 --export-min-visits 24
# 2. train with nodes (and the control without them)
python -m showdown_bot.train --data data/selfplay/gen7 --model mlp --hidden 16 --residual `
    --value-target search --nodes on --nodes-per-world 3 --node-min-visits 24 --node-weight 0.5 `
    --node-depth-discount 0.9 --x-dtype float16 --out models/gen7-res-nodes-mlp16.pev
python -m showdown_bot.train --data data/selfplay/gen7 --model mlp --hidden 16 --residual `
    --value-target search --x-dtype float16 --out models/gen7-res-root-mlp16.pev
```

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

## Augmentation and dropout (589.20)

gen0 is 604 battles; every position of a battle shares its label and the
per-slot stats/types fingerprint the battle, so the effective N is closer to
the number of games than of positions (the MLP memorises: best epoch 1-3).
Three training-only options (validation is never touched; code in
`showdown_bot/augment.py`, layout constants asserted against
`poke_engine.state_feature_names()` so a layout change fails loudly):

- `--augment-bench`: per sample and side, randomly permute the bench slots 1..5
  (53-wide blocks; slot 0, side extras and field untouched). Done as a GPU
  gather per batch. Bench order is arbitrary team order, so it is
  label-preserving. In residual mode the stock-eval column is untouched
  (evaluate is permutation-invariant).
- `--augment-sides`: with p = 1/2 swap the own/opp 363-blocks. This equals
  `state_features(state, 2)` exactly (verified on 1500 real states and in a
  test), the label becomes `1 - y` (outcome and q targets), and in residual mode
  the raw stock eval flips sign (`evaluate` is exactly antisymmetric by
  construction). Both views of a battle stay in the same split because the
  swap is applied to training batches only.
- `--dropout P`: dropout between the hidden ReLU and the output layer, training
  only; eval-mode output equals the export (verified by the existing export
  check and a test).

Grid on gen0 (MLP-64, `--value-target blend --blend-lambda 0.75 --q-kind
absolute`, states both, seed 0, same split: 60 val battles; val BCE is against
the pure outcome, weighted; stock-eval logistic baseline 0.5113; train BCE is
the clean (un-augmented) train set at the best epoch):

| model | config | lr | val BCE | acc | best ep | train BCE |
|---|---|---|---|---|---|---|
| plain | none | 1e-3 | **0.4637** | 0.761 | 3/9 | 0.4677 |
| plain | bench | 1e-3 | 0.4675 | 0.760 | 6/12 | 0.5074 |
| plain | bench+sides | 1e-3 | 0.4684 | 0.762 | 6/12 | 0.5163 |
| plain | bench+sides+dropout 0.2 | 1e-3 | 0.4787 | 0.764 | 4/10 | 0.5156 |
| residual | none | 1e-3 | 0.4798 | 0.752 | 1/7 | 0.5110 |
| residual | bench | 1e-3 | 0.4765 | 0.749 | 3/9 | 0.5191 |
| residual | bench+sides | 1e-3 | 0.4809 | 0.755 | 1/7 | 0.5509 |
| residual | bench+sides+dropout 0.2 | 1e-3 | 0.4822 | 0.755 | 1/7 | 0.5523 |
| residual | none | 1e-4 | 0.4775 | 0.751 | 10/16 | 0.5069 |
| residual | bench | 1e-4 | 0.4798 | 0.756 | 7/13 | 0.5460 |
| residual | bench+sides | 1e-4 | 0.4808 | 0.757 | 13/19 | 0.5456 |
| residual | bench+sides+dropout 0.2 | 1e-4 | 0.4809 | 0.755 | 13/19 | 0.5432 |

(Residual runs: stock-only logistic baseline on this split is 0.5007.)

Reading: augmentation lets the net train longer and shrinks the train/val
mismatch but does **not** improve val BCE on gen0; differences of about 0.005
are within the noise of 60 val battles and one seed. Early stopping already
captures most of the benefit; the lever that is left is more games. Models
kept: `models/gen0-aug-best-plain.pev` (bench) and
`models/gen0-aug-best-res.pev` (residual, bench, lr 1e-3); the unaugmented
plain baseline is still the best plain val BCE.

## Auxiliary targets (`--aux on`, bead 589.56)

KataGo-style extra heads on the value net's hidden layer, training only (the `.pev` export is
byte-identical in format; the Rust evaluator is unchanged). `--aux-weight` (default 0.5) scales
the aux loss added to the value BCE. Mlp + v1 features only; incompatible with `--augment-*`
and `--nodes`. Code: `auxtargets.py`.

* Survival: 12 logits (own slots 0..5 then opp slots 0..5, feature slot order: active first, rest
  in team order), BCE, masked for empty / already-fainted slots, ties, and opposing slots that are
  not on the real team (sampled, never-revealed worlds).
* Remaining-Pokemon difference (own alive - opp alive at the end) / 6, tanh output, MSE.
* The recordings store no terminal state, so the end of the game is reconstructed: the loser has
  nobody left; the winner keeps the Pokemon alive in the LAST stored decision state (truth state if
  any). This can overcount the winner's survivors by those that faint in the last unrecorded turns.
* `--aux-metrics` prints the same val baselines for a run with `--aux off`.

## Human pretraining, self-play fine-tuning and the human-test metric

Idea: pretrain the value net on first-person human ladder positions (outcome labels, no search
values), then warm-start (`--init`) a self-play fine-tune on `--value-target search`. The human
data is `data\human\pc1-fp-aux\{train,test}` (100k / 35.7k games, see `docs/human-replays.md`).

`--init MODEL.pev` copies the weights of an existing `.pev` into the model being trained. It
needs the same `--model`, `--hidden`, `--features` and `--residual`-ness (otherwise it exits 2
naming the mismatch). Carried over: all layer weights and biases and, for residual nets, the stock
coefficient `a = stock_coef / 0.0125` (still trainable unless `--freeze-stock-coef`) and the output
bias. Not applied with `--init`: the logistic fit of `a, b` on the new train split
(`--stock-coef-init` is ignored; the fit is still printed as the stock-only reference). The net
has no input normalisation, so there are no other statistics to carry. Learning rate and its
schedule are unchanged (AdamW, early stopping); pass `--lr` (about `3e-4` for a fine-tune).

`scripts/human_eval.py MODEL.pev [MODEL2.pev ...] --data DIR [--max-games N] [--per-game K]
[--seed S] [--json OUT]` scores nets exactly as the engine reads them at a search root
(`Evaluator.logit(state)`: absolute logit for side one, residual stock term included; the leaf
scale does not matter for an absolute probability). It reports BCE and decisive accuracy vs the
game outcome overall and per turn bucket (1-5, 6-10, 11-15, 16-20, 21-30, 31+) next to the
constant predictor, on a fixed subsample (default 4 truth decisions per game, seed 0) so numbers
compare across models. It streams one game file at a time (a few MB RAM).

Feature version: the deployed gen7/gen8 nets (`gen7-res-root-mlp16.pev`, `gen8-res-root-mlp16.pev`)
are **features v1** (738 inputs), not v2, so the commands below use `--features v1` to stay
warm-startable from them and comparable. Swap in `--features v2` for a v2 run (rows are 3x wider:
2,191 columns, 4.4 KB per float16 row, so RAM scales 3x).

Smoke tests (single process, CPU; pretrain on 456 human battles with `--max-per-game 3`, then a
fine-tune on 50 gen7 records with `--init`, `--lr 3e-4`; human_eval on 300 test games = 1,178
positions). BCE: pretrained 0.593 (v1) / 0.593 (v2), after the 50-game fine-tune 0.594 / 0.595,
`gen7-res-root-mlp16` 0.666, `gen2-res-searchabs-mlp16` 0.648, constant 0.693. human_eval costs
about 0.65 s per 1,000 positions per model, i.e. about 2.5 s per 1,000 games for one model at 4
per game (state parsing dominates, so extra models are cheaper).

Full-scale commands (PowerShell, from the repo root; hidden 16, residual, features v1):

```powershell
# (a) pretrain on all of pc1-fp-aux/train: outcome targets, truth states, <= 6 decisions per game
python -m showdown_bot.train --data data\human\pc1-fp-aux\train --states truth `
    --value-target outcome --model mlp --residual --hidden 16 --features v1 --x-dtype float16 `
    --max-per-game 6 --load-workers 8 --out models\human-pre-res16.pev `
    --val-data data\human\pc1-fp-aux\test@4

# (b) fine-tune on gen8 + gen9 self-play, warm-started, search-value targets
python -m showdown_bot.train --data data\selfplay\gen8 --data data\selfplay\gen9 `
    --init models\human-pre-res16.pev --states truth --value-target search --q-kind absolute `
    --model mlp --residual --hidden 16 --features v1 --x-dtype float16 --lr 3e-4 `
    --load-workers 8 --out models\human-ft-gen89-res16.pev `
    --val-data data\human\pc1-fp-aux\test@4

# (c) evaluate on first-person human test positions (the references use the same subsample)
python scripts\human_eval.py models\human-pre-res16.pev models\human-ft-gen89-res16.pev `
    models\gen7-res-root-mlp16.pev models\gen8-res-root-mlp16.pev `
    --data data\human\pc1-fp-aux\test --json human-eval.json
```

For (b) use the same `--states` / `--weighting` flags as the recipe that produced the reference
net if it differs from `truth`. `--init` only takes nets of the same feature version and hidden
size.

Estimates, extrapolated from the smoke tests (about 1,200 samples/s single-process loading,
60 us per sample-epoch for the h16 MLP on one CPU core; the 3080 is much faster per epoch; float16
rows are 1.5 KB for v1, 4.4 KB for v2). Treat as +-50%:

| step | rows | RAM (peak) | time |
| --- | --- | --- | --- |
| (a) pretrain, 185,906 records (100k games), cap 6 | about 1.1M train rows | v1 about 2-3 GB (v2 about 6 GB), plus the `@4` val set (about 0.5 GB v1) | load about 15 min single-process (2-3 min with 8 workers); about 1 min per epoch on CPU, up to 60 epochs with patience 6, so about 30-60 min CPU, a few minutes on the GPU |
| (b) fine-tune, 6,684 gen8 + gen9 records | about 230k truth rows | about 1 GB | load about 3 min single-process; about 15 s per epoch on CPU |
| (c) human_eval, 35.7k games x 4 | about 140k positions per model | below 0.5 GB | about 1.5 min per model |

Uncapped (`--max-per-game 0`) the pretrain set is about 2.6M rows (about 4 GB v1, 12 GB v2).

## Feature version 4: identity embeddings, wide net, human pretrain then self-play fine-tune (589.65)

`--features v4` = dense v2 ++ v3 features (2379) plus the species / item / ability / four move
identity of all 12 slots as learned embeddings (`--embed-dim`, default 8 per table, moves
mean-pooled), feeding `Linear(2379 + 12 * 32, --hidden) -> ReLU -> Linear(hidden, 1)`. Layout,
vocabularies and the format-3 `.pev`: `docs/features-v4.md`. `--residual`, `--init`,
`--value-target search|blend|td|nstep`, `--q-*`, `--nodes`, `--dropout` and `--val-data` work as
for v1-v3; `--augment-*` and `--aux on` are v1 only. A run also writes `<out>.pev.vocab.json`.

Rows are 2463 columns (the 84 identity indices ride in the same matrix; float16 holds them
exactly), 4.9 KB per float16 row: 1M rows = 4.9 GB, 3M rows = 15 GB. Cap the human set with
`--max-per-game` to fit RAM (6 decisions per game over the 100k-game set is about 1.1M train
rows = 5.5 GB); matrices over 2 GB stay in CPU RAM and are moved to the GPU per batch.

Recommended commands (PowerShell, repo root, 3080; `--batch-size 2048` because the net is wide
and the sets are large; set `OMP_NUM_THREADS=4` to cap the CPU side):

```powershell
# (a) human pretrain: outcome targets on first-person truth states, h128, embed 8, residual
python -m showdown_bot.train --data data\human\pc1-fp-aux\train --states truth `
    --value-target outcome --model mlp --residual --features v4 --embed-dim 8 --hidden 128 `
    --x-dtype float16 --max-per-game 8 --load-workers 6 --batch-size 2048 --lr 1e-3 `
    --weight-decay 1e-4 --epochs 12 --patience 3 --out models\human-v4-h128.pev `
    --val-data data\human\pc1-fp-aux\test@4

# (b) self-play fine-tune: warm start, search-value targets (corrected q labels), lower lr
python -m showdown_bot.train --data data\selfplay\gen8 --data data\selfplay\gen9 `
    --init models\human-v4-h128.pev --states prefer-truth --value-target search --q-kind absolute `
    --model mlp --residual --features v4 --embed-dim 8 --hidden 128 --x-dtype float16 `
    --load-workers 6 --batch-size 1024 --lr 2e-4 --weight-decay 1e-4 --dropout 0.1 `
    --epochs 15 --patience 3 --out models\ft-v4-h128.pev `
    --val-data data\human\pc1-fp-aux\test@4

# (c) score both on human test positions next to the references
python scripts\human_eval.py models\human-v4-h128.pev models\ft-v4-h128.pev `
    models\gen7-res-root-mlp16.pev --data data\human\pc1-fp-aux\test --json human-eval-v4.json
```

`--init` needs the same `--features`, `--hidden`, `--embed-dim` and `--residual` (it prints the
mismatch otherwise); a vocabulary hash different from the engine's is refused. Because the
fine-tune starts from the human net, the first epochs of (b) show how far self-play labels move
it; lower `--lr` (1e-4) or raise `--dropout` if the self-play val BCE rises while train falls
(the 1.4M self-play rows overfit larger nets from scratch, bead 589.20; that is the reason for
the pretrain). Both steps verify the export (max |`Evaluator.evaluate` - torch logit|, about
1e-6 on smoke runs).

Smoke runs (3,258 human rows then 3,850 gen8+gen9 self-play rows, h64, embed 8, residual,
3 epochs each, RTX 3080): human pretrain 12.9 s total, fine-tune with `--init` and
`--value-target search` 6.8 s, both exports verified. Leaf-eval speed (single thread, 102 gen9
random-battle states, `engine/examples/bench_net.rs`, the SPRT experiment was loading the
machine so treat as +-10%):

| evaluator | pure `Evaluator.logit` calls/s | MCTS iterations/s, 1 thread | vs stock evaluate |
| --- | --- | --- | --- |
| stock `evaluate` | 15.0M | 305-325k | 1.00 |
| gen7 net (v1, h16, residual) | 5.0M | 281-287k | 0.92 |
| v4, h64, embed 8 | 0.60-0.66M | 181-182k | 0.58 |
| v4, h128, embed 8 | 0.47-0.48M | 157-158k | 0.50 |

At equal wall time a v4 h64 net therefore searches about 0.64x as many iterations as the gen7
net, h128 about 0.55x. About 0.7 us of the 1.5 us (h64) is the v3 matchup block, which is
recomputed per leaf and does not depend on the hidden size.
