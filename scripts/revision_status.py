"""Read-only progress dashboard for Cleber 21/out v2. No model/API calls.

Usage: uv run python scripts/revision_status.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCENARIOS = ("fakewhatsapp-full", "fakewhatsapp-10pct", "fakewhatsapp-5pct",
             "fakebr-10pct", "fakebr-5pct",
             "covid19br-full", "covid19br-10pct", "covid19br-5pct",
             "factckbr-full", "factckbr-10pct", "factckbr-5pct")
SEEDS = (42, 43, 44)
ARMS = ("all", "random-k", "score-k", "easy-k")
REPLAY_ARMS = ("real", "static", "ai")


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"WARN invalid JSON: {path}: {exc}")
        return None


def completed_replays(base, group, arms):
    found = defaultdict(list)
    for path in (base / group).glob("*/results.json"):
        data = load(path)
        if data and data.get("arm") in arms and data.get("test", {}).get("f1") is not None:
            try:
                seed = int(data["config"]["random_state"])
            except (KeyError, ValueError):
                print(f"WARN missing seed: {path}")
                continue
            found[(seed, data["arm"])].append(path)
    return found


def dashboard(base):
    for scenario in SCENARIOS:
        print(f"\n{scenario}")
        replay_group = f"{scenario}-weighted-replay-21oct-v2"
        replay = completed_replays(base, replay_group, REPLAY_ARMS)
        for seed in SEEDS:
            marks = " ".join(f"{arm}:{'✓' if (seed, arm) in replay else '·'}" for arm in REPLAY_ARMS)
            print(f"  B/replay s{seed}: {marks}")
        candidates = defaultdict(list)
        for path in (base / "revision-candidates").glob(f"{scenario}-s*/candidates.json"):
            data = load(path)
            if data and data.get("scenario") == scenario:
                candidates[data.get("seed")].append((path, data))
        for seed in SEEDS:
            entries = candidates.get(seed, [])
            if not entries:
                print(f"  C/controlled s{seed}: candidates:· arms:· · · ·")
                continue
            for path, data in entries:
                state = "complete" if data.get("complete") else f"partial {len(data.get('batches', []))}/{data.get('cycles', '?')}"
                print(f"  C/controlled s{seed}: candidates:{state} tokens:{data.get('tokens_used', 0)} ({path})")
                matches = defaultdict(list)
                # Discover results by exact source, not merely group/seed.
                for group in base.glob("revision-controlled*"):
                    for result in group.glob("*/results.json"):
                        row = load(result)
                        if row and Path(row.get("source", "")) == path:
                            matches[row.get("arm")].append(result)
                print("    " + " ".join(f"{arm}:{'✓' if matches[arm] else '·'}" for arm in ARMS))
                for arm, results in matches.items():
                    if len(results) > 1:
                        print(f"    WARN duplicate {arm} for {path}: {len(results)} runs; do not average blindly")
        for (seed, arm), paths in replay.items():
            if len(paths) > 1:
                print(f"  WARN duplicate B/replay s{seed} {arm}: {len(paths)} runs; inspect source")
    print("\nLegend: ✓ completed results.json; · pending. This does not validate methodology or human annotations.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    dashboard(args.outputs)


if __name__ == "__main__":
    main()
