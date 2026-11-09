# A tiny transformer for Pokemon battles (a learning project)

Goal: understand tokenization and transformers by building the smallest honest one on our
own self-play data.  Everything is a few hundred readable lines:

| file | what it is |
|---|---|
| `tokenizer.py` | battle state -> 14 tokens (ids + numbers) |
| `model.py` | attention, blocks, the full model, written from scratch |
| `train.py` | data loading, train/val split, baselines, training loop |
| `inspect_attention.py` | print what the SUMMARY token looks at |

Run it (from the repo root, needs `poke_engine` importable and a GPU is nice but not required):

```
python learn/train.py                 # ~6 min on an RTX 3080 (2.5 min of that is parsing)
python learn/inspect_attention.py
python learn/train.py --no_cats       # the ablation discussed in Results
```

## 1. What is a token?

A language model cannot read letters.  A *tokenizer* chops text into pieces and gives each
piece an integer id: `"unbelievable"` -> `["un", "believ", "able"]` -> `[348, 9021, 77]`.
The numbers mean nothing by themselves; they are just names.

Our "text" is a battle position, and the natural pieces are not sub-words but **Pokemon**.
One token = one whole Pokemon (its species, item, moves, HP, ...), plus one token for the
field.  A position is a *sequence of 14 tokens*:

```
 position:   0         1  2  3  4  5  6      7  8  9 10 11 12     13
 token:   [SUMMARY] [ own team, active first ] [ opp team, active first ] [FIELD]
                      ^^                          ^^
                      active mon                  active mon
```

A Pokemon token is described by two kinds of data:

* **categorical** (names): species, item, ability, status, 4 moves, tera type, 2 types.
  `Vocab` turns each name into an id (`"item=leftovers"` -> 412).  Names seen fewer than
  `min_count` times in training become `<unk>`; empty slots are `<pad>` (id 0).
* **numeric** (quantities): HP fraction, fainted, is-active, level, five stats, stat boosts,
  terastallized.  These are scaled to roughly 0..1.

## 2. Embeddings: lookup tables

A neural net needs vectors, not ids.  An *embedding* is literally a table with one row per
id: `emb[412]` is a learnable vector of 64 numbers.  Training nudges the rows so that
things that behave alike (two similar sweepers) end up with similar rows.  `<pad>` has a
row of zeros, so empty fields add nothing.

## 3. One vector per token: sum the ingredients

Each Pokemon token has 11 categorical ids but the transformer wants **one** 64-vector per
token.  So we:

1. look up all 11 ids and **add** the vectors (a bag of attributes: "Garchomp + Rocky Helmet
   + Earthquake + ...").
2. push the numeric features through `Linear(17, 64)` (a learned projection into the same
   64-dim space) and add that too.
3. add a **slot embedding** (which of the 14 positions) and a **side embedding**
   (summary/field, own, opponent).  Attention by itself has no idea about order or who owns
   what; without these, "our active Garchomp" and "their bench Garchomp" would look the same.

Result: a tensor of shape `(batch, 14, 64)`.

The SUMMARY token has no data of its own (all `<pad>`); only its slot embedding is learned.
It is a blank page the network can write a whole-position summary onto, like BERT's `[CLS]`.

## 4. Self-attention

Imagine our Pokemon token asking "is any of their team a threat to me?".  Attention does
exactly that in three steps, using three linear maps of each token's vector:

* **query** (Q): what am I looking for?
* **key** (K): what do I contain / advertise?
* **value** (V): what information do I hand over if you pick me?

For token *i* and token *j*: score = Q_i . K_j (big when the question matches the
advertisement).  Divide by sqrt(head size) so the numbers do not explode, then **softmax**
across *j* so the weights sum to 1.  Token *i*'s new vector = the weighted average of every
token's V.  Example: our Corviknight's query "who can hurt me?" matches the key of their Kyogre
strongly, so Kyogre's value gets mixed into Corviknight's new vector.

All 14 tokens do this simultaneously: the attention matrix is `(14, 14)`.

**Multi-head**: instead of one big attention we split the 64 dims into 4 heads of 16, each
with its own Q/K/V.  Different heads can learn different questions (one about threats, one
about hazards ...).  Their outputs are concatenated and mixed by one more Linear.

## 5. Block = attention + MLP, with residuals and LayerNorm

```
x = x + Attention(LayerNorm(x))      # tokens talk to each other
x = x + MLP(LayerNorm(x))            # each token thinks on its own
```

* **Residual** (`x + ...`): each layer only has to learn a small *edit*; gradients flow
  straight through, so deeper stacks train.
* **LayerNorm**: rescales each token vector to mean 0, variance 1 so values stay tame.
* **MLP**: Linear(64->256), GELU, Linear(256->64).

We stack 3 blocks.  After layer 1 each token knows a little about the others; after layer 3
it knows about relationships between relationships.

## 6. Heads: value and policy

* **Value head**: read the final SUMMARY vector, small MLP, one number, sigmoid -> P(win).
  Trained with binary cross-entropy against the real result.
* **Policy head**: one logit for each of 9 action slots = our active's 4 moves + 5 bench
  switches.  Move logit i = MLP(final active-mon vector, embedding of move i); switch logit
  j = MLP(final vector of bench mon j).  Illegal slots (not in the search's candidate list) are
  masked to -infinity so softmax gives them probability 0.  Tera and non-tera variants of a
  move are folded into the same slot.  The training target is the search's root visit share
  (a probability distribution); loss is cross-entropy against it.

## 7. Train/validation split by battle

We hold out 10% of **battles** and never train on them.  Splitting by *position* would be
cheating: positions from one battle are near-duplicates (same teams, same label) so the model
could "recognise the battle" instead of learning to play.  Both players' files of one battle
share a battle tag and go to the same side of the split.  The vocab is also built from the
training split only, as it would be at deployment.

## 8. Results

Data: 10,000 self-play files (4,000 gen9 + 6,000 gen8 random battles) = 325,874 positions
from 4,917 battles; 292k train / 34k val positions.  Model: d_model 64, 4 heads, 3 layers,
**248,963 parameters** (79,872 are the embedding table).  3 epochs, AdamW, batch 256.
Wall time ~215 s on an RTX 3080 after ~160 s parsing.  Value metrics use val positions from
turn >= 3.

| | value BCE (lower better) | value acc | policy CE | policy top-1 vs search |
|---|---|---|---|---|
| constant 0.5 / majority class | 0.6931 | 0.536 | | |
| HP-sum logistic regression | 0.6030 | 0.671 | | |
| uniform over legal actions | | | 1.857 | 0.168 |
| (floor: entropy of search targets) | | | 0.862 | |
| **transformer (full)** | **0.6369** | **0.659** | **1.678** | **0.331** |
| transformer, numbers only (`--no_cats`) | 0.5849 | 0.705 | 1.723 | 0.239 |

What this means:

* **Policy**: the transformer clearly learned something (top-1 agreement 33% vs 17% for
  uniform; CE 1.68 vs 1.86) but it is nowhere near the search (CE floor 0.86).  The search
  actually simulates damage; our model only sees stats and move *names*.
* **Value**: the full transformer is *worse than a one-feature HP logistic regression* on
  BCE and only ties on accuracy.  This is the most important lesson of the project, and it is
  **overfitting by memorisation**.  Each battle has unique random teams, and every position of
  a battle has the same label.  With species/items/moves as inputs the net can learn "this
  exact team set wins" - train accuracy climbs to ~80-99% while validation stays near chance
  (on our early runs with 3k files it reached 95% train / 45% val).  The ablation proves it:
  blank all the names and let the model see only numbers (HP, stats, boosts), and the value
  head *improves* (BCE 0.585, acc 0.705, beating the HP baseline) because it cannot memorise
  teams.  The names did help the policy (0.331 vs 0.239 top-1), where the label differs per
  position and move identity matters.
* What reduced the memorisation (but did not remove it): 10,000 files instead of 3,000
  (val BCE 0.69 -> 0.64), randomly blanking half the categorical fields during training
  (`--field_drop 0.5`), dropout 0.2, weight decay 0.1, and stopping after 3 epochs
  (validation BCE rises again by epoch 4-6).
* `inspect_attention.py` example (SUMMARY token, last layer): see `learn/out/attn_example.txt`
  after running.  Heads concentrate on a handful of single bench Pokemon (e.g. one head puts
  0.76 on the opponent's Flygon) rather than on the active mons or HP - consistent with the
  net using species identity as a team fingerprint rather than reading the actual board.  Attention
  is a hint, not a proof.

## 9. How the real bot's version differs

* **Hidden information.**  This toy reads the engine's *truth* state, so it sees the
  opponent's full sets (moves, items, abilities) that a real player could not know.  A real
  model must only see what has been revealed, masking the rest (or averaging over sampled
  possible worlds like our search does).  Because of this the toy cheats and its numbers are
  not comparable to a deployable model.
* **Data scale.**  ~5k battles is tiny; real systems use orders of magnitude more, and more
  diverse teams, which is the real cure for the memorisation seen above.
* **Reinforcement learning / self-play.**  The labels here are "who eventually won"
  (noisy) and a one-step imitation of a search.  The real loop is: use the network to guide
  search, play games, train on the improved search policy and outcomes, repeat.
* **Matchup tokens.**  Whether move X hurts Pokemon Y is a (move, target) interaction.  A
  stronger design adds tokens (or features) for each own-move vs each opponent mon (damage
  range, speed order, type effectiveness) so attention does not have to rediscover the type
  chart from 5k battles.
* **Tera / full action space.**  We fold tera variants into the base move and ignore items like
  forced switches, Z-moves, etc.
