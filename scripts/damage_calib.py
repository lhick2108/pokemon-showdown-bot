"""Calibrate damage-roll inference against ground truth (589.29).

    python scripts/damage_calib.py run --games 40 --workers 4 --out calib.jsonl
    python scripts/damage_calib.py analyze calib.jsonl [--abs-hp 2 --rel-margin 0.04 --abs-pct 1.6]

``run`` plays in-process self-play (like scripts/selfplay.py) with a hook on
``damage_infer`` that, for every clean single-hit observation the inferer
evaluates, also computes the engine's predicted MAX damage with the TRUE
opposing set (read from the peer player's battle) in the very same state, and
logs observed / predicted-max plus the context (move, abilities, items,
weather, terrain, screens, status, boosts, tera).  Non-crit damage must land in
[0.85, 1.0] of the max.

``analyze`` reports the ratio distribution, the share outside the inference's
rejection window, the outliers by feature, and the false-rejection rate for
given margins (with TRUE sets every rejection is a false one).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def to_id(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _true_mon(battle, peer):
    opp = battle.opponent_active_pokemon
    for key, value in battle.opponent_team.items():
        if value is opp and key in peer.team:
            return peer.team[key]
    for bmon in peer.team.values():
        if to_id(bmon.species) == to_id(opp.species):
            return bmon
    return None


def _mon_ctx(mon) -> dict:
    return {
        "species": str(mon.species),
        "ability": str(mon.ability or ""),
        "item": str(mon.item or ""),
        "status": getattr(mon.status, "name", None),
        "boosts": {k: v for k, v in mon.boosts.items() if v},
        "tera": bool(mon.is_terastallized),
        "types": [t.name.lower() for t in mon.types if t is not None],
        "level": mon.level,
    }


def make_hook(out_path: Path):
    from showdown_bot import recorder
    from showdown_bot.sets import SampledSet

    from showdown_bot import sets

    handle = out_path.open("a", encoding="utf-8")
    recent: dict[int, list] = collections.defaultdict(list)
    original = sets.OpponentModel.observe_events

    def observe_events(self, battle_tag, events, battle=None):
        log = recent[id(battle)]
        for e in events:
            if len(e) > 1 and e[1] == "request":
                continue
            log.append("|".join(map(str, e)))
        del log[:-60]
        return original(self, battle_tag, events, battle)

    sets.OpponentModel.observe_events = observe_events

    # The peer's battle object is a little AHEAD of ours (it has already applied this
    # message: items knocked off, abilities changed), so the true set is snapshotted at
    # each decision (turn start) via the recorder's truth-state builder.
    snaps: dict[tuple, list] = collections.defaultdict(list)
    original_truth = recorder.build_truth_state

    def build_truth_state(battle, other):
        if other is not None:
            for bmon in other.team.values():
                tera = (
                    (getattr(bmon, "_last_request", None) or {}).get("teraType")
                    or getattr(bmon.tera_type, "name", None)
                    or bmon.tera_type
                )
                snaps[(battle.battle_tag, other.player_role, to_id(bmon.species))].append(
                    (battle.turn, SampledSet(
                        species=bmon.species, level=int(bmon.level or 100), role="",
                        moves=tuple(bmon.moves.keys()), ability=bmon.ability or bmon.base_ability,
                        item=(bmon.item if bmon.item and bmon.item != "unknown_item" else ""),
                        tera_type=str(tera).lower() if tera else None))
                )
        return original_truth(battle, other)

    recorder.build_truth_state = build_truth_state

    def hook(battle, obs, opp_attacks, predict, illusion_p=0.0):
        peer = recorder.peer_battle(battle)
        if peer is None:
            return
        true = _true_mon(battle, peer)
        if true is None:
            return
        history = snaps.get((battle.battle_tag, peer.player_role, to_id(true.species)), [])
        usable = [h for h in history if h[0] <= battle.turn]
        if not usable:
            return
        sampled = usable[-1][1]
        if not sampled.ability or not sampled.moves:
            return  # turn-1 snapshot before the peer's request populated the set
        from showdown_bot import damage_infer

        if opp_attacks and to_id(sampled.ability) in damage_infer._UNRELIABLE_ABILITIES:
            return
        predicted = predict(to_id(true.species), sampled)
        if predicted is None:
            return
        high, maxhp = predicted
        ours, theirs = battle.active_pokemon, battle.opponent_active_pokemon
        pct_high = high / maxhp * 100
        if opp_attacks:
            observed = (obs.hp_before - obs.hp_after) * maxhp
        else:
            observed = (obs.hp_before - obs.hp_after) * 100.0
            high = pct_high
        ctx = {
            "battle": battle.battle_tag, "turn": battle.turn, "opp_attacks": opp_attacks,
            "move": obs.move_id, "hp_before": obs.hp_before, "hp_after": obs.hp_after,
            "observed": observed, "high": high, "maxhp": maxhp, "pct_high": pct_high,
            "ratio": observed / high if high else None,
            "weather": [w.name for w in battle.weather], "fields": [f.name for f in battle.fields],
            "our_sc": [s.name for s in battle.side_conditions],
            "opp_sc": [s.name for s in battle.opponent_side_conditions],
            "attacker": _mon_ctx(theirs if opp_attacks else ours),
            "defender": _mon_ctx(ours if opp_attacks else theirs),
            "illusion_p": illusion_p, "true_species": to_id(true.species),
            "true_ability": sampled.ability, "true_item": sampled.item,
            "true_moves": list(sampled.moves), "events": list(recent[id(battle)]),
        }
        handle.write(json.dumps(ctx) + "\n")
        handle.flush()

    return hook


def cmd_run(args) -> int:
    import selfplay  # scripts/selfplay.py

    from showdown_bot import damage_infer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    damage_infer.CALIB_HOOK = make_hook(out)
    argv = ["--games", str(args.games), "--pairs", str(args.pairs), "--workers", str(args.workers),
            "--out", str(Path(args.selfplay_out)), "--worlds", str(args.worlds)]
    return selfplay.main(argv)


def _features(r: dict) -> list[str]:
    c = [f"move:{r['move']}"]
    for who, m in (("att", r["attacker"]), ("def", r["defender"])):
        if m["ability"]:
            c.append(f"{who}-ability:{to_id(m['ability'])}")
        if m["item"] and m["item"] != "unknown_item":
            c.append(f"{who}-item:{to_id(m['item'])}")
        if m["status"]:
            c.append(f"{who}-status:{m['status']}")
        if m["tera"]:
            c.append(f"{who}-tera")
        c.extend(f"{who}-boost:{stat}" for stat in m["boosts"])
    c.extend(f"weather:{w}" for w in r["weather"])
    c.extend(f"field:{f}" for f in r["fields"])
    c.extend(f"sidecond:{s}" for s in r["our_sc"] + r["opp_sc"])
    return c


def consistent(r, args) -> bool:
    """The inferer's own acceptance test, with the margins under evaluation."""
    from showdown_bot import damage_infer

    damage_infer.REL_MARGIN, damage_infer.ABS_HP_MARGIN = args.rel_margin, args.abs_hp
    damage_infer.ABS_PCT_MARGIN = args.abs_pct
    hp = r["hp_before"] * (r["maxhp"] if r["opp_attacks"] else 100.0)
    return damage_infer.roll_consistent(r["high"], r["observed"], r["opp_attacks"], hp)


def cmd_analyze(args) -> int:
    rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r["ratio"] is not None]
    print(f"observations with truth: {len(rows)}")
    for label, sub in (("opp attacks us (exact HP)", [r for r in rows if r["opp_attacks"]]),
                       ("we attack opp (percent HP)", [r for r in rows if not r["opp_attacks"]])):
        if not sub:
            continue
        ratios = sorted(r["ratio"] for r in sub)
        n = len(ratios)

        def q(p):
            return ratios[min(n - 1, int(p * n))]

        print(f"\n== {label}: n={n}")
        print("ratio min/p1/p5/p50/p95/p99/max: "
              f"{ratios[0]:.3f} {q(.01):.3f} {q(.05):.3f} {q(.5):.3f} {q(.95):.3f} {q(.99):.3f} {ratios[-1]:.3f}")
        lo = [r for r in sub if r["ratio"] < 0.85]
        hi = [r for r in sub if r["ratio"] > 1.0]
        print(f"strictly outside [0.85,1.0]: {len(lo) + len(hi)} ({(len(lo) + len(hi)) / n:.2%}); "
              f"<0.85: {len(lo)}; >1.0: {len(hi)}")
        bad = [r for r in sub if not consistent(r, args)]
        print(f"FALSE REJECTIONS at rel={args.rel_margin} hp={args.abs_hp} pct={args.abs_pct}: "
              f"{len(bad)} ({len(bad) / n:.2%})")
        base = collections.Counter()
        for r in sub:
            base.update(set(_features(r)))
        for kind, picks in (("low", lo), ("high", hi), ("rejected", bad)):
            if not picks:
                continue
            cnt = collections.Counter()
            for r in picks:
                cnt.update(set(_features(r)))
            print(f"  {kind} outliers (n={len(picks)}): feature  outlier/total-with-feature")
            for feat, k in sorted(cnt.items(), key=lambda kv: (-kv[1] / base[kv[0]], -kv[1]))[: args.top]:
                print(f"    {feat:40s} {k:4d} / {base[feat]}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--games", type=int, default=40)
    r.add_argument("--pairs", type=int, default=4)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--worlds", type=int, default=2)
    r.add_argument("--out", default="damage_calib.jsonl")
    r.add_argument("--selfplay-out", default=str(REPO / "data" / "selfplay"))
    a = sub.add_parser("analyze")
    a.add_argument("file")
    from showdown_bot import damage_infer

    a.add_argument("--rel-margin", type=float, default=damage_infer.REL_MARGIN)
    a.add_argument("--abs-hp", type=float, default=damage_infer.ABS_HP_MARGIN)
    a.add_argument("--abs-pct", type=float, default=damage_infer.ABS_PCT_MARGIN)
    a.add_argument("--top", type=int, default=15)
    args = p.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
