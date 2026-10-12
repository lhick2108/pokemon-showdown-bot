"""Label-preserving augmentations of the state feature vector (engine/FEATURES.md).

Layout: ``[own side 363][opp side 363][field 12]``; a side is 6 Pokemon slots of
53 values (slot 0 = active, slots 1..5 = bench in arbitrary team order) then 45
side extras. Everything here is a gather on torch tensors (no per-sample loops).

* ``permute_bench``: independently per sample and side, shuffle the bench slots.
* ``swap_sides``: side-two's view of a state is exactly the own/opp swap
  (checked on real states); the label becomes ``1 - y``, and the raw stock
  evaluation (residual column), which is antisymmetric, changes sign.
"""

from __future__ import annotations

import torch

SLOT = 53
N_SLOTS = 6
SIDE = SLOT * N_SLOTS + 45  # 363
FIELD = 12
N_FEATURES = 2 * SIDE + FIELD  # 738


def check_layout(names: list[str] | None = None) -> None:
    """Assert the constants above against ``poke_engine.state_feature_names()``."""
    if names is None:
        import poke_engine

        names = list(poke_engine.state_feature_names())
    assert len(names) == N_FEATURES, f"feature length {len(names)} != {N_FEATURES}"
    for s, side in enumerate(("own", "opp")):
        off = s * SIDE
        for k in range(N_SLOTS):
            first, last = names[off + SLOT * k], names[off + SLOT * k + SLOT - 1]
            assert first == f"{side}.p{k}.hp", (first, side, k)
            assert last == f"{side}.p{k}.speed", (last, side, k)
        extras = names[off + SLOT * N_SLOTS: off + SIDE]
        assert extras[0] == f"{side}.boost.attack" and extras[-1] == f"{side}.misc.alive_fraction", extras
        assert all(not n.startswith(f"{side}.p") for n in extras)
    assert names[2 * SIDE] == "field.weather.sun" and names[-1] == "field.trick_room"


def swap_index(n_in: int = N_FEATURES, device=None) -> torch.Tensor:
    """Column gather that swaps the own and opp side blocks (field and extra columns kept)."""
    idx = torch.arange(n_in, device=device)
    idx[:SIDE] = torch.arange(SIDE, 2 * SIDE, device=device)
    idx[SIDE:2 * SIDE] = torch.arange(SIDE, device=device)
    return idx


def swap_sides(x: torch.Tensor, flip: torch.Tensor, residual: bool = False) -> torch.Tensor:
    """Rows where ``flip`` (bool, (B,)) is true get their own/opp blocks swapped.

    With ``residual`` the last column is the raw stock evaluation and is negated on
    flipped rows.
    """
    swapped = x[:, swap_index(x.shape[1], x.device)]
    if residual:
        swapped[:, -1] = -swapped[:, -1]
    return torch.where(flip[:, None], swapped, x)


def bench_gather_index(perms: torch.Tensor, n_in: int = N_FEATURES) -> torch.Tensor:
    """Per-row column gather index from ``perms`` (B, 2, 5): a permutation of 0..4.

    Bench slot ``k+1`` of side ``s`` takes the block of slot ``perms[b, s, k] + 1``.
    """
    b, dev = perms.shape[0], perms.device
    idx = torch.arange(n_in, device=dev).repeat(b, 1)
    src = torch.cat([torch.zeros(b, 2, 1, dtype=torch.long, device=dev), perms + 1], dim=2)
    cols = (
        torch.arange(2, device=dev)[None, :, None, None] * SIDE
        + src[..., None] * SLOT
        + torch.arange(SLOT, device=dev)
    )  # (B, 2, 6, 53)
    for s in range(2):
        idx[:, s * SIDE: s * SIDE + SLOT * N_SLOTS] = cols[:, s].reshape(b, -1)
    return idx


def permute_bench(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Independently per row and side, randomly permute bench slots 1..5."""
    perms = torch.rand(x.shape[0], 2, 5, device=x.device, generator=generator).argsort(dim=2)
    return torch.gather(x, 1, bench_gather_index(perms, x.shape[1]))


class Augmenter:
    """Applies the configured augmentations to a training batch (x, y)."""

    def __init__(self, bench: bool, sides: bool, residual: bool = False, seed: int = 0, device="cpu"):
        check_layout()
        self.bench, self.sides, self.residual = bench, sides, residual
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.sides:
            flip = torch.rand(x.shape[0], device=x.device, generator=self.gen) < 0.5
            x = swap_sides(x, flip, self.residual)
            y = torch.where(flip, 1.0 - y, y)
        if self.bench:
            x = permute_bench(x, self.gen)
        return x, y
