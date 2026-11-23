"""List the bot's switch decisions with the search's own scores.

    python scripts/switch_audit.py RECORDS_DIR_OR_FILE [...]

For each voluntary or forced switch: chosen option's visit-weighted mean score (pooled over
worlds), the best-scoring option, the best non-switch option, and the chosen option's share
of the final policy. Flags (!) a switch whose mean score trails the best option by > 0.02.
"""

from __future__ import annotations

import sys
from pathlib import Path

from showdown_bot import recorder


def main() -> None:
    flagged = total = 0
    for arg in sys.argv[1:]:
        for path in recorder.iter_game_files(arg):
            game = recorder.read_game(path)
            print(Path(path).name[:45])
            for d in game["decisions"]:
                chosen = str(d.get("chosen") or "")
                if not d.get("worlds") or not chosen.startswith("switch"):
                    continue
                agg: dict[str, list[float]] = {}
                for w in d["worlds"]:
                    for m in w["moves"]:
                        v = agg.setdefault(m["choice"], [0, 0.0])
                        v[0] += m["visits"]
                        v[1] += m["score"]
                mean = {c: s / max(n, 1) for c, (n, s) in agg.items()}
                if chosen not in mean:
                    continue
                total += 1
                best = max(mean, key=mean.get)
                stays = [c for c in mean if not c.startswith("switch")]
                stay = max(stays, key=mean.get) if stays else None
                bad = mean[best] - mean[chosen] > 0.02
                flagged += bad
                stay_txt = f"{stay} {mean[stay]:.3f}" if stay else "-"
                print(f"  {'!' if bad else ' '} t{d['turn']:>2} chose {chosen:<26} {mean[chosen]:.3f} | "
                      f"best {best} {mean[best]:.3f} | best stay {stay_txt} | "
                      f"share {d['policy'].get(chosen, 0):.2f}")
    print(f"{flagged} of {total} switches trail the best option by > 0.02")


if __name__ == "__main__":
    main()
