# Token transformer: model, trainer, value-overfitting mitigations (bead 589.68.3)

`src/showdown_bot/token_net.py` (model, export/load, `predict`), `src/showdown_bot/token_train.py` (trainer),
`tests/test_token_net.py`. Input layout, shard schema and target definitions: `docs/tokens.md` (TOKEN_VERSION 2:
the tokenizer leak of unrevealed opponent bench membership is fixed, re-export before training on older shards).

## Architecture

94 tokens, fixed layout (static shapes, `torch.compile` friendly): `[CLS, global, side x2, mon x12, pair x36,
history x16, own action x13, opp action x13]`. Pre-norm encoder (`F.scaled_dot_product_attention`, key mask),
default `d_model 256, 6 layers, 8 heads, MLP x4` = **6.08 M parameters** (`--layers 4`: 4.50 M).

* mon token = species + item + ability + sum(move embedding + move-position embedding) + `Linear(mon_cont)` +
  `Linear(mon_known)` (the known flags are what separates "unknown" id 1 from a revealed value) + slot embedding.
* side tokens = `Linear(side_cont)`; global token = `Linear(glob ++ match_agg)`; 36 pair tokens = `Linear(match_pair)` +
  own-slot + opp-slot embedding (all-zero pairs, i.e. hidden/fainted, are masked as keys).
* history tokens = side + kind + action id (move or species table by kind) + actor species + `Linear(hist_cont)`
  (pad rows masked). Action tokens = move emb + species emb + `Linear(act_cont)` + side emb (illegal slots masked as keys).
* a learned absolute embedding per token index gives the token type/position. Heads are computed in float32.

Heads: **policy** (one logit per own action token, softmax over `act_legal`), **value** (CLS, logit of P(own wins)),
**opponent action** (logit per opp action token for samples whose action was found in the scrubbed input, plus a
vocabulary head from CLS over move + species ids for every valid sample, tied to the embedding tables), **hidden sets**
(per opponent mon token: item / ability / Tera type classifiers and a move distribution, tied to the embeddings; only
for slots whose species is revealed and fields not yet revealed; revealed moves are masked out), **aux** (log turns left,
final alive fractions).

Inference: `net.export(path)` writes state dict + config + token-spec hash (+ `.json`); `TokenNet.load(path)` refuses a
spec mismatch (against `expect_spec` or the installed engine); `net.predict(batch)` takes the shard-field dict
(numpy) and returns numpy probabilities (`policy`, `value`, `opp_policy`, `opp_vocab`, `hid_*`, `aux`);
`token_net.batch_from_engine([tokens.split_buffers(...)])` builds a batch from live engine tokens (history rows default
to padding).

## Loss

`w_pol*CE(policy, pol_target) + w_val*BCE(value, target) + w_opp*(CE slot + 0.5*CE vocab) + w_hid*(item+ability+tera+move CE)
+ w_aux*aux`, every term masked by its validity field (defaults 1 / 1 / 0.3 / 0.3 / 0.05). Value target:
`--value-target search` (default) = `v_search` where valid else `z` (so human data = outcome, self-play = per-position search value),
`z`, or `mix`. Per-source scaling: `--src-loss NAME.TERM=V` (NAME = source name or kind), e.g. `--src-loss human.hid=0`.

## Value overfitting (randbats teams are unique per game, z is constant per game)

A value head with ids memorises team fingerprints. Built-in mitigations, all visible in the metrics:

* split strictly by `battle_hash` (done by the exporter; both seats share it), val/test never share a battle with train;
* `--id-dropout P`: each training sample has all categorical ids blanked with probability P; those samples train the value
  head only (policy/opp/hidden/aux masked for them);
* `--numbers-only`: ablation, ids blanked in training and evaluation;
* `--value-target search` (per-position targets) and `--max-per-battle K` (at most K random rows per battle per pass over a shard);
* `--dropout`, `--wd` (AdamW, no decay on norms/embeddings), early stopping on **val value BCE** (`--select value`, default;
  the best-by-loss checkpoint is also kept as `<out>.byloss.pt`, best-by-value as `<out>.byvalue.pt`);
* every evaluation reports the value BCE of an **HP-sum logistic baseline** (own minus opponent hp-fraction sum, fitted on a train sample)
  and of the **same net with ids blanked** (`val_bce_blind`); the final block says whether ids help, have no effect or hurt.

Smoke ablation (97k + 39k samples, ~1,200 steps, `--dropout 0.1`, held-out battles; value BCE vs z on self-play val / human test):

| run | self-play val | human test | blind (ids blanked) self-play | note |
| --- | ---: | ---: | ---: | --- |
| A: z target, no mitigation | 0.751 (best epoch 0.607, then 1.1) | 1.07 | 0.928 | memorises, ids hurt out of distribution |
| B: search target + id-dropout 0.5 + cap 6 | 0.520 | 0.641 | 0.524 | ids help slightly |
| C: search target, numbers only | **0.496** | 0.648 | 0.496 | |
| HP-sum logistic | 0.533 | n/a | | constant 0.693 |

Policy top-1 is hurt a little by id-dropout (B 0.442 vs A 0.458 self-play) but benefits from ids relative to C only slightly at this
data size. Full data will tell; the ablation flags make it a config choice.

## Trainer

Memory-mapped shard streaming (`Stream`: shuffled shard order each pass, 6 shards interleaved, rows shuffled in a shard,
resident memory = index arrays only), weighted mixture of source groups (`--weight selfplay=1 --weight human=0.5`; a name overrides its kind;
sources of a group share the group's weight in proportion to their size), single process, `torch.set_num_threads(2)`, one prefetch
*thread* (no worker processes). AdamW (betas 0.9/0.98), linear warmup + cosine, clip 1.0, bf16 autocast (`--amp fp16` uses GradScaler),
resumable (`<out>.last`, `--resume`, every epoch and `--ckpt-every` steps). An "epoch" is `--epoch-samples` draws from the mixture
(default: all train rows). Log: `log-<out stem>.txt` next to `--out`.

Per-epoch and final reports: policy top-1/top-3/CE vs uniform (>=2 legal actions), value BCE (vs ln2, base rate, HP-sum logistic, blanked ids,
and search-value-as-predictor on rows with `v_search`), opponent action slot top-1/NLL and vocab top-1/NLL, hidden item/ability/tera accuracy and
move recall@4. The final block ends with the bars: (1) self-play policy top-1 vs 0.472 (h32 prior, `docs/policy-prior.md`, gen5/6 val; the
exports here are gen7-9), (2) human policy top-1 on the 1800+ bucket (rating 4) of the test split vs 0.497 and all ratings vs 0.504
(`docs/human-policy.md`; those models had an elo input, this one has none), (3) value BCE, (4) opponent action / hidden sets.

## Full run

```
python -m showdown_bot.tokens export ... --out data/tokens/v2 --workers 2        # needs the sdb.21 wheel
python -m showdown_bot.token_train --data data/tokens/v2 --weight selfplay=1 --weight human=0.5 \
    --value-target search --id-dropout 0.3 --max-per-battle 12 --dropout 0.05 --wd 0.05 \
    --batch 256 --lr 5e-4 --warmup 1000 --epochs 6 --epoch-samples 2000000 --patience 3 \
    --out models/tok-d256.pt
```

Speed on the 3080 with the ladder using the CPU: about 4,400 samples/s at batch 256 (about 2,200 at batch 128), i.e. roughly 25 min per
6.5 M samples; RAM is the page cache of the memmaps plus about 1.5 GB of Python/torch; GPU memory is a few GB at batch 256.

## Latency (3080, fp16 autocast, inputs on the device, logits copied to the host; ladder busy on the CPU, so launch-bound and pessimistic)

| batch | 6 layers (6.08 M) | 4 layers (4.50 M) |
| ---: | ---: | ---: |
| 1 | 13.7 ms | 7.0 ms |
| 16 | 8.5 ms | 7.1 ms |
| 64 | 8.5 ms | 7.1 ms |
| 256 | 13.1 ms | 9.6 ms |

`python -m showdown_bot.token_net bench [--layers N]`. Batch 1 is dominated by kernel launches; CUDA graphs or `torch.compile` are the next step for a bot.

## Playing the net: `token-policy` bot and the transformer root prior (bead 589.68, step W2)

`src/showdown_bot/token_policy.py` (`TokenPolicy`: load, tokenise, batched forward), `players/tokenpolicy.py` (the bot),
`tests/test_token_policy.py`. **No engine or wheel change**: `monte_carlo_tree_search(..., root_prior_s1=)` (a
`{choice: probability}` dict, `engine/src/prior.rs::PriorConfig::root_s1`) already replaces a side's root prior in PUCT. It works
without a `.ppol`: the root then uses PUCT and every other node keeps decoupled UCB1.

View the net gets at every decision: the bot's own state from side one, the opponent scrubbed to what the set tracker has seen
(`tokens.reveal_rows(_opp_facts(...))`, the training view). The 16 history rows are padding (training saw real events; tracking
them live is not done, so the live policy may be a little weaker than the offline top-1). Legal actions are the engine's
`act_legal` (the same options as the search root); illegal slots carry zero mass, and the vector is renormalised after the
poke-env order mapping.

### Bot specs

| spec | meaning |
| --- | --- |
| `token-policy:model=<tok.pt>[:device=auto\|cpu\|cuda\|cuda:N][:temp=T][:pick=argmax\|sample]` | no search. Argmax of the policy head over legal actions; `temp=T` samples from `p^(1/T)` (`pick=sample` without `temp` samples at T=1) |
| `search-net:...:tok_prior=<tok.pt>[:tok_device=auto\|cpu\|cuda]` | our side's root prior = the net's policy (replaces the `.ppol` root prior; with `prior=` the `.ppol` still serves interior nodes and the opponent side, without it only the root is PUCT). `c_puct`, `prior_eps` (default 0.03 uniform mix) and `fpu` apply as for `.ppol` priors. Also accepted by `search` and `humansearch` |

Per decision the net runs **once** in the bot process (GPU if available), batched over the sampled worlds (the policy head is
averaged over them: our side is identical in every world, only the scrubbed opponent view differs), and the resulting dict is
shipped to every world's search worker. Opponent side: the `.ppol` mirror / `prior_opp=` setting is unchanged. The net's opponent
action head is keyed by action tokens whose unrevealed move slots have no name, so it cannot be turned into a root vector over
a sampled world's real options without per-world work; not done. If the net fails, the decision falls back to the unmodified
search (`bot.tok_prior_errors`).

### Latency per decision (3080, fp16 autocast, league/ladder busy on the CPU, 6.08 M model, median of 60)

| | tokenise | forward + host copy | total |
| --- | ---: | ---: | ---: |
| cuda, 1 world | 0.4 ms | 14.5 ms | 14.9 ms |
| cuda, 4 worlds (batched) | 1.0 ms | 15.0 ms | 15.7 ms |
| cpu (contended), 1 world | 0.4 ms | 50.8 ms | 51.2 ms |
| cpu (contended), 4 worlds | 1.0 ms | 141 ms | 142 ms |

`tok_prior` therefore adds about 16 ms per searched decision on the GPU (under 2% of a `time_ms=1000` search; it runs before the
search on the event-loop thread). Use `tok_device=cuda` for league runs. Games sharing a process share one net instance (cached per
path and device).

### Example SPRT commands

Best at the time of these runs (now the `best-gen7` spec preset; the current best is the `best` preset, see
`league.PRESETS`): `search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev`

```
# transformer prior replaces the .ppol at the root, everything else identical
python -m showdown_bot.league --bots "cand=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:tok_prior=models\tok-d256.pt:tok_device=cuda:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev,best=search-net:scale=1.5:c_puct=2:prior=models\prior-gen56-h32.ppol:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev" --sprt 0,21 --max-games 2000 --search-workers 32 --concurrency 4 --teams swap --out data/league/sprt-tokprior-vs-best --seed 1

# transformer root prior only (no .ppol; interior nodes plain UCB1): swap the cand spec for
cand=search-net:scale=1.5:c_puct=2:tok_prior=models\tok-d256.pt:tok_device=cuda:mode=argmax:time_ms=1000:worlds=4:evaluator=models\gen7-res-root-mlp16.pev

# the raw net as a no-search bot (a sanity opponent for any of the above)
tokpol=token-policy:model=models\tok-d256.pt:device=cuda
```

The flags follow `docs/league.md` ("SPRT duel"). `tests/test_token_policy.py::test_documented_specs_parse` runs these spellings
through `parse_spec`.
