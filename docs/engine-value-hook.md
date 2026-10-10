# Learned value hook in our poke-engine fork

Our fork of poke-engine lives in [`engine/`](../engine/README.md) (origin:
upstream `c2dbbda`, v0.0.48; ours is `0.0.48+sdb.1`). It adds an optional
learned leaf evaluator (linear or 2-layer MLP) to MCTS, and a fixed state
encoding used by both training and inference ([`engine/FEATURES.md`](../engine/FEATURES.md)).

- Build/install into the project `.venv`: `scripts/build_engine.ps1`
  (`--locked`, same Cargo features as Foul Play's Gen 9 engine).
- Python: `monte_carlo_tree_search(state, ..., evaluator=None)`;
  `poke_engine.Evaluator.load(path)`; `poke_engine.state_features(state)`.
- Tests: `cd engine && cargo test`,
  `tests/test_engine.py`.
- Benchmark: `python -m uv run python scripts/engine_bench.py`.

## Throughput (single thread, spike_engine.py state, Windows x64)

| evaluator | iter/s (1 s) | iter/s (2 s) | slowdown vs stock |
| --- | ---: | ---: | ---: |
| stock `evaluate()` | 256,612 | 234,778 | 1.00x |
| linear (738 features) | 204,689 | 191,614 | 1.23-1.25x |
| MLP hidden=64 | 163,454 | 161,011 | 1.46-1.57x |
| MLP hidden=256 | 100,097 | 99,275 | 2.36-2.56x |

Dummy random weights; one run per cell, so treat as +-5 %. Foul Play's venv still uses the unmodified upstream wheel.
