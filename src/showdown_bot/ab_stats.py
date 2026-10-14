"""Statistics for ladder A/B logs (data/ladder/ab.jsonl): numpy only."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from showdown_bot.bench import wilson_interval

Z = 1.96
Z_POWER80 = 0.8416


def read_records(path: Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def arm_summary(records: Sequence[dict]) -> dict[str, dict]:
    """Per arm: games (decided), wins, ties, win rate + Wilson CI, mean opponent rating."""
    out: dict[str, dict] = {}
    for r in records:
        name = r.get("arm") or "?"
        d = out.setdefault(name, {"games": 0, "wins": 0, "ties": 0, "opp": []})
        if r["result"] == "tie":
            d["ties"] += 1
            continue
        d["games"] += 1
        d["wins"] += r["result"] == "win"
        if r.get("opponent_rating") is not None:
            d["opp"].append(r["opponent_rating"])
    for d in out.values():
        n, w = d["games"], d["wins"]
        d["rate"] = w / n if n else float("nan")
        d["ci"] = wilson_interval(w, n) if n else (float("nan"), float("nan"))
        d["mean_opp"] = float(np.mean(d.pop("opp"))) if d["opp"] else float("nan")
    return out


def diff_ci(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float, float]:
    """p2 - p1 with a Wald 95% interval (unpooled)."""
    p1, p2 = w1 / n1, w2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p2 - p1
    return d, d - Z * se, d + Z * se


def games_needed(p1: float, p2: float, *, z_beta: float = 0.0) -> float:
    """Games per arm so the 95% CI of the difference excludes 0 at the observed effect
    (z_beta=0, about 50% power) or with 80% power (z_beta=0.8416)."""
    d = abs(p2 - p1)
    if d == 0:
        return float("inf")
    var = p1 * (1 - p1) + p2 * (1 - p2)
    return (Z + z_beta) ** 2 * var / d**2


def logistic_irls(X: np.ndarray, y: np.ndarray, iters: int = 50, ridge: float = 1e-6):
    """Logistic regression by IRLS. Returns (beta, covariance)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-(X @ beta)))
    w = np.clip(p * (1 - p), 1e-9, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1]))
    return beta, cov


def adjusted_effects(records: Sequence[dict], arms: Sequence[str]) -> dict:
    """Logistic win ~ arm + (opp_rating - our_rating)/400, effects vs arms[0].

    Effect = P(win | arm k, equal ratings) - P(win | arms[0], equal ratings), in
    probability points, with a delta-method 95% CI. Rows lacking either rating are skipped.
    """
    rows = [r for r in records
            if r["result"] != "tie" and r.get("arm") in arms
            and r.get("opponent_rating") is not None and r.get("rating_before") is not None]
    res = {"n": len(rows), "effects": {}}
    if len(rows) < len(arms) + 2:
        return res
    X = np.zeros((len(rows), len(arms) + 1))
    X[:, 0] = 1.0
    y = np.zeros(len(rows))
    for i, r in enumerate(rows):
        k = arms.index(r["arm"])
        if k:
            X[i, k] = 1.0
        X[i, -1] = (r["opponent_rating"] - r["rating_before"]) / 400.0
        y[i] = r["result"] == "win"
    beta, cov = logistic_irls(X, y)
    res["rating_coef"] = float(beta[-1])
    sig = lambda x: 1 / (1 + math.exp(-x))
    p0 = sig(beta[0])
    for k in range(1, len(arms)):
        pk = sig(beta[0] + beta[k])
        # gradient of sig(b0 + bk) - sig(b0) wrt (b0, bk)
        g = np.zeros(len(beta))
        g[0] = pk * (1 - pk) - p0 * (1 - p0)
        g[k] = pk * (1 - pk)
        se = math.sqrt(float(g @ cov @ g))
        eff = pk - p0
        res["effects"][arms[k]] = {"effect": eff, "lo": eff - Z * se, "hi": eff + Z * se,
                                   "log_odds": float(beta[k]), "p_ref": p0, "p_arm": pk}
    return res


def format_report(records: Sequence[dict]) -> str:
    summ = arm_summary(records)
    if not summ:
        return "no records"
    names = sorted(summ)
    lines = [f"{'arm':<14}{'games':>6}{'wins':>6}{'ties':>5}{'win%':>8}  {'Wilson 95% CI':<16}{'mean opp':>9}"]
    for n in names:
        d = summ[n]
        lines.append(f"{n:<14}{d['games']:>6}{d['wins']:>6}{d['ties']:>5}{d['rate']:>8.1%}  "
                     f"[{d['ci'][0]:.1%}, {d['ci'][1]:.1%}]".ljust(16)
                     + f"{d['mean_opp']:>9.0f}")
    ref = names[0]
    for other in names[1:]:
        a, b = summ[ref], summ[other]
        if not (a["games"] and b["games"]):
            continue
        d, lo, hi = diff_ci(a["wins"], a["games"], b["wins"], b["games"])
        lines.append(f"\n{other} - {ref}: {d * 100:+.1f} pts  95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}]"
                     f"  ({'significant' if lo > 0 or hi < 0 else 'not significant'})")
        n50 = games_needed(a["rate"], b["rate"])
        n80 = games_needed(a["rate"], b["rate"], z_beta=Z_POWER80)
        if math.isfinite(n50):
            lines.append(f"  games per arm to resolve the observed difference at 95%: "
                         f"~{math.ceil(n50)} (about 50% power), ~{math.ceil(n80)} for 80% power"
                         f"  (now {a['games']} / {b['games']})")
        else:
            lines.append("  observed difference is 0: no sample size resolves it")
    adj = adjusted_effects(records, names)
    if adj["effects"]:
        lines.append(f"\nLogistic win ~ arm + (opp - our rating)/400 on {adj['n']} games "
                     f"(rating coefficient {adj['rating_coef']:+.2f} per 400 pts):")
        for n, e in adj["effects"].items():
            lines.append(f"  {n} vs {ref} at equal ratings: {e['effect'] * 100:+.1f} pts "
                         f"95% CI [{e['lo'] * 100:+.1f}, {e['hi'] * 100:+.1f}]  "
                         f"(P {e['p_ref']:.1%} -> {e['p_arm']:.1%}, log-odds {e['log_odds']:+.2f})")
    else:
        lines.append("\nLogistic adjustment skipped: too few games with both ratings recorded")
    return "\n".join(lines)
