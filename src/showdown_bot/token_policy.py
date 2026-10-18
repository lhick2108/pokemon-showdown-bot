"""Live inference with the token transformer (bead showdown-bot-589.68, step W2).

``TokenPolicy`` wraps an exported ``TokenNet`` (``models/tok-d256.pt``) for use inside a bot: it
tokenises engine state strings from our point of view (side one, opponent scrubbed with the
revealed-facts rows, exactly the training view), runs ONE batched forward over all states and
returns the policy head's distribution over our legal actions, keyed by the engine choice strings
(``policy_actions`` / ``root_get_all_options`` spelling: ``"tackle"``, ``"tackle-tera"``,
``"switch pikachu"``).

Two users (docs/token-net.md):

* the ``token-policy`` league bot (``players/tokenpolicy.py``): argmax or temperature sampling;
* ``tok_prior=<path.pt>`` on the search bots: the distribution replaces the root prior of our side
  (``root_prior_s1`` of ``monte_carlo_tree_search``), computed once per decision in the parent
  process (GPU if available) and shipped to every world's search worker.

The live history rows are padding (the network was trained with the last 16 events; the live bot does
not track them yet).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

N_ACT = 13
DEVICES = ("auto", "cpu", "cuda")

_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], "TokenPolicy"] = {}


def valid_device(text: str) -> bool:
    """``auto``, ``cpu``, ``cuda`` or ``cuda:N``."""
    t = text.strip().lower()
    return t in DEVICES or (t.startswith("cuda:") and t[5:].isdigit())


def resolve_device(device: str = "auto") -> str:
    import torch

    d = device.strip().lower()
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using the CPU")
        return "cpu"
    return d


def normalise(choices: Sequence[str], probs: Sequence[float]) -> dict[str, float]:
    """``{choice: p}`` over the entries with a name and positive probability, summing to 1."""
    out: dict[str, float] = {}
    for c, p in zip(choices, probs):
        if c and np.isfinite(p) and p > 0:
            out[c] = out.get(c, 0.0) + float(p)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {c: p / total for c, p in out.items()}


class TokenPolicy:
    """An exported ``TokenNet`` plus the tokeniser glue. One instance per (path, device)."""

    def __init__(self, net: Any, device: str, path: str = "") -> None:
        self.net = net
        self.device = device
        self.path = path
        self.calls = 0
        self.total_ms = 0.0
        self.last_ms = 0.0

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "TokenPolicy":
        """Cached per (resolved path, resolved device); refuses a token-spec mismatch."""
        from showdown_bot.token_net import TokenNet

        p = str(Path(path).resolve())
        if not Path(p).is_file():
            raise FileNotFoundError(f"token net not found: {p}")
        dev = resolve_device(device)
        with _LOCK:
            tp = _CACHE.get((p, dev))
            if tp is None:
                tp = _CACHE[(p, dev)] = cls(TokenNet.load(p, device=dev), dev, p)
        return tp

    # ------------------------------------------------------------------ tokenise
    @staticmethod
    def tokenise(states: Sequence[str], reveal: Any) -> tuple[dict[str, np.ndarray], list[list[str]]]:
        """Batch dict + the own-action choice names (13 per state, ``""`` for an empty slot)."""
        import poke_engine

        from showdown_bot import token_net, tokens

        lay = tokens.layout()
        bufs, names = [], []
        for s in states:
            cat, cont, ch = poke_engine.state_tokens_str(s, 1, reveal, True)
            bufs.append(tokens.split_buffers(cat, cont, lay))
            names.append([str(c) for c in ch[:N_ACT]])
        return token_net.batch_from_engine(bufs), names

    # ------------------------------------------------------------------ inference
    def predict_states(self, states: Sequence[str], reveal: Any) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
        """``(names [n][13], policy [n,13], legal [n,13])`` for ``states`` in ONE batched forward."""
        if not states:
            return [], np.zeros((0, N_ACT), np.float32), np.zeros((0, N_ACT), bool)
        t0 = time.perf_counter()
        batch, names = self.tokenise(states, reveal)
        out = self.net.predict(batch, device=self.device)
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        self.calls += 1
        self.total_ms += self.last_ms
        return names, out["policy"], batch["act_legal"][:, 0] > 0

    def distribution(self, states: Sequence[str], reveal: Any) -> dict[str, float]:
        """The policy over our legal choices, averaged over ``states`` (worlds), normalised.

        Our own side is identical in every world; the worlds only differ in what the scrubbed
        opponent view still reveals, so the mean is the natural single vector.
        """
        names, pol, legal = self.predict_states(states, reveal)
        if not names:
            return {}
        acc: dict[str, float] = {}
        for n, p, lg in zip(names, pol, legal):
            for c, x, ok in zip(n, p, lg):
                if c and ok:
                    acc[c] = acc.get(c, 0.0) + float(x) / len(names)
        return normalise(list(acc), list(acc.values()))

    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0


def pick(dist: dict[str, float], temp: float | None, rng: Any) -> str:
    """argmax (``temp`` None) or a draw from ``dist ** (1/temp)``."""
    if not dist:
        raise ValueError("empty distribution")
    if temp is None:
        return max(dist, key=dist.__getitem__)
    names = list(dist)
    w = np.array([dist[c] for c in names], dtype=np.float64) ** (1.0 / temp)
    w = w / w.sum()
    return names[int(rng.choices(range(len(names)), weights=w.tolist())[0])]
